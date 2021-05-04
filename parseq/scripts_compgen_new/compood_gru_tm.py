import json
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from typing import Dict, List

import sklearn
import wandb

import qelos as q
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader
from transformers import AutoTokenizer, BertModel

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree, tree_to_lisp
from parseq.rnn1 import Encoder
from parseq.scripts_compgen_new.transformer import TransformerConfig, TransformerStack
from parseq.scripts_compgen_new.transformerdecoder import TransformerStack as TransformerStackDecoder
from parseq.scripts_compgen_new.compood import run as run_tm, compute_auc_and_fprs, ORDERLESS
from parseq.scripts_compgen_new.compood_gru import run as run_gru
from parseq.vocab import Vocab


class HybridSeqDecoder(torch.nn.Module):
    def __init__(self, *decoders, mcdropout=-1, **kw):
        super(HybridSeqDecoder, self).__init__(**kw)
        self.maindecoder = decoders[0]
        self.alldecoders = decoders
        self.mcdropout = mcdropout

    def forward(self, x: torch.Tensor,
                     gold: torch.Tensor = None):  # --> implement how decoder operates end-to-end
        preds, prednll, maxmaxnll, entropy, total, avgconf, sumnll, stepsused = self.maindecoder.get_prediction(x)

        def tensor_to_trees(x, vocab: Vocab):
            xstrs = [vocab.tostr(x[i]).replace("@START@", "") for i in range(len(x))]
            xstrs = [re.sub("::\d+", "", xstr) for xstr in xstrs]
            trees = []
            for xstr in xstrs:
                # drop everything after @END@, if present
                xstr = xstr.split("@END@")
                xstr = xstr[0]
                # add an opening parentheses if not there
                xstr = xstr.strip()
                if len(xstr) == 0 or xstr[0] != "(":
                    xstr = "(" + xstr
                # balance closing parentheses
                parenthese_imbalance = xstr.count("(") - xstr.count(")")
                xstr = xstr + ")" * max(0, parenthese_imbalance)  # append missing closing parentheses
                xstr = "(" * -min(0, parenthese_imbalance) + xstr  # prepend missing opening parentheses
                try:
                    tree = taglisp_to_tree(xstr)
                    if isinstance(tree, tuple) and len(tree) == 2 and tree[0] is None:
                        tree = None
                except Exception as e:
                    tree = None
                trees.append(tree)
            return trees

        # compute loss and metrics
        gold_trees = tensor_to_trees(gold, vocab=self.maindecoder.vocab)
        pred_trees = tensor_to_trees(preds, vocab=self.maindecoder.vocab)
        treeaccs = [float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device), "stepsused": stepsused}

        runs = max(self.mcdropout, 1)
        probses = []
        preds = preds[:, 1:]
        if self.mcdropout > 0:
            self.train()
        else:
            self.eval()
        for decoder in self.alldecoders:
            for i in range(runs):
                d, logits = decoder.train_forward(x, preds)
                probses.append(torch.softmax(logits, -1))
        self.eval()
        probses = sum(probses) / len(probses)
        probses = probses[:, :-1]
        probs = probses
        mask = preds > 0
        nlls = torch.gather(probs, 2, preds[:, :, None])[:, :, 0]
        nlls = -torch.log(nlls)

        avgnll = (nlls * mask).sum(-1) / mask.float().sum(-1).clamp(1e-6)
        sumnll = (nlls * mask).sum(-1)
        maxnll, _ = (nlls + (1 - mask.float()) * -10e6).max(-1)
        entropy = (-torch.log(probs.clamp_min(1e-7)) * probs).sum(-1)
        entropy = (entropy * mask).sum(-1) / mask.float().sum(-1).clamp(1e-6)
        ret["decnll"] = avgnll
        ret["sumnll"] = sumnll
        ret["maxmaxnll"] = maxnll
        ret["entropy"] = entropy
        return ret, pred_trees


def run(lr=0.0001,
        enclrmul=0.01,
        smoothing=0.,
        gradnorm=3,
        tmbatsize=60,
        grubatsize=60,
        tmepochs=16,
        gruepochs=16,
        patience=10,
        validinter=3,
        validfrac=0.1,
        warmup=3,
        cosinelr=False,
        dataset="scan/length",
        mode="normal",          # "normal", "noinp"
        maxsize=50,
        seed=42,
        hdim=768,
        tmnumlayers=6,
        grunumlayers=2,
        numheads=12,
        tmdropout=0.1,
        grudropout=0.1,
        worddropout=0.,
        bertname="bert-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        mcdropout=-1,
        version="grutm_v3"
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    grusettings = {(k[3:] if k.startswith("gru") else k): v for k, v in settings.items() if not k.startswith("tm")}
    grudecoder, indtestds, oodtestds = run_gru(**grusettings)

    tmsettings = {(k[2:] if k.startswith("tm") else k): v for k, v in settings.items() if not k.startswith("gru")}
    tmdecoder, _, _ = run_tm(**tmsettings)

    # create a model that uses tmdecoder to generate output and uses both to measure OOD
    decoder = HybridSeqDecoder(tmdecoder, grudecoder, mcdropout=mcdropout)
    results = evaluate(decoder, indtestds, oodtestds, batsize=tmbatsize, device=device)
    print("Results of the hybrid OOD:")
    print(json.dumps(results, indent=3))

    wandb.init(project=f"compood_gru_tm_baseline_v3", config=settings, reinit=True)
    for k, v in results.items():
        for metric, ve in v.items():
            settings.update({f"{k}_{metric}": ve})

    wandb.config.update(settings)
    q.pp_dict(settings)


def cat_dicts(x:List[Dict]):
    out = {}
    for k, v in x[0].items():
        out[k] = []
    for xe in x:
        for k, v in xe.items():
            out[k].append(v)
    for k, v in out.items():
        out[k] = torch.cat(v, 0)
    return out


def evaluate(model, idds, oodds, batsize=10, device=torch.device("cpu")):
    """
    :param model:       Decoder model
    :param idds:     dataset with in-distribution examples
    :param oodds:      dataset with out-of-distribution examples
    :return:
    """
    iddl = DataLoader(idds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    ooddl = DataLoader(oodds, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    _, idouts = q.eval_loop(model, iddl, device=device)
    idouts = cat_dicts(idouts[0])
    _, oodouts = q.eval_loop(model, ooddl, device=device)
    oodouts = cat_dicts(oodouts[0])

    # decnll_res = compute_auc_and_fprs(idouts["decnll"], oodouts["decnll"], "decnll")
    # sumnll_res = compute_auc_and_fprs(idouts["sumnll"], oodouts["sumnll"], "sumnll")
    # maxnll_res = compute_auc_and_fprs(idouts["maxmaxnll"], oodouts["maxmaxnll"], "maxmaxnll")
    # entropy_res = compute_auc_and_fprs(idouts["entropy"], oodouts["entropy"], "entropy")
    decnll_res = compute_auc_and_fprs(oodouts["decnll"], idouts["decnll"], "decnll")
    sumnll_res = compute_auc_and_fprs(oodouts["sumnll"], idouts["sumnll"], "sumnll")
    maxnll_res = compute_auc_and_fprs(oodouts["maxmaxnll"], idouts["maxmaxnll"], "maxmaxnll")
    entropy_res = compute_auc_and_fprs(oodouts["entropy"], idouts["entropy"], "entropy")
    return {"decnll": decnll_res, "maxnll": maxnll_res, "entropy": entropy_res, "sumnll": sumnll_res}


def run_experiment(
        lr=-1.,
        enclrmul=-1.,
        smoothing=-1.,
        gradnorm=2,
        tmbatsize=-1,
        grubatsize=-1,
        tmepochs=-1,      # probably 11 is enough
        gruepochs=-1,
        patience=100,
        validinter=-1,
        warmup=3,
        cosinelr=False,
        dataset="default",
        datasets="both",
        mode="normal",
        maxsize=-1,
        seed=-1,
        hdim=-1,
        tmnumlayers=-1,
        grunumlayers=-1,
        numheads=-1,
        tmdropout=-1.,
        grudropout=-1.,
        worddropout=-1.,
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        trainonvalidonly=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        mcdropout=-1,
        ):

    settings = locals().copy()
    del settings["datasets"]

    ranges = {
        "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3",
                    "cfq/mcd1", "cfq/mcd2", "cfq/mcd3"],
        # "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"],
        # "dataset": ["cfq/mcd1", "cfq/mcd2", "cfq/mcd3"],
        # "dataset": ["scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd3"],
        "tmdropout": [0.1, 0.25, 0.5],
        "grudropout": [0.1, 0.25, 0.5],
        "worddropout": [0.],
        "seed": [42, 87646464, 456852],
        "tmepochs": [20, 25],
        "gruepochs": [40, 25],
        # "epochs": [25],
        "tmbatsize": [50],
        "grubatsize": [256, 128],
        # "batsize": [100],
        "hdim": [384],
        "numheads": [12],
        "tmnumlayers": [6],
        "grunumlayers": [2],
        # "numlayers": [2],
        "lr": [0.0005],
        "enclrmul": [1.],                  # use 1.
        "smoothing": [0.],
        "validinter": [1],
        "mcdropout": [0, 5],
    }

    if datasets == "both":
        pass
    elif datasets == "cfq":
        ranges["dataset"] = ["cfq/mcd1", "cfq/mcd2", "cfq/mcd3"]
    elif datasets == "scan":
        ranges["dataset"] = ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"]
    elif datasets == "mcd":
        ranges["dataset"] = ["cfq/mcd1", "cfq/mcd2", "cfq/mcd3", "scan/mcd1", "scan/mcd2", "scan/mcd3"]
    elif datasets == "nonmcd":
        ranges["dataset"] = ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left"]

    for k in ranges:
        if k in settings:
            if isinstance(settings[k], str) and settings[k] != "default":
                ranges[k] = [settings[k]]
            elif isinstance(settings[k], (int, float)) and settings[k] >= 0:
                ranges[k] = [settings[k]]
            else:
                pass
                # raise Exception(f"something wrong with setting '{k}'")
            del settings[k]

    def checkconfig(spec):
        if spec["dataset"].startswith("cfq"):
            if spec["gruepochs"] != 25:
                return False
        elif spec["dataset"].startswith("scan"):
            if spec["gruepochs"] != 40:
                return False
        if spec["dataset"].startswith("cfq"):
            if spec["grubatsize"] != 128:
                return False
        elif spec["dataset"].startswith("scan"):
            if spec["grubatsize"] != 256:
                return False
        if spec["dataset"].startswith("cfq"):
            if spec["tmepochs"] != 20:
                return False
        elif spec["dataset"].startswith("scan"):
            if spec["tmepochs"] != 25:
                return False
        return True

    print(__file__)
    p = __file__ + f".baseline.{dataset}"
    q.run_experiments_random(
        run, ranges, path_prefix=p, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)