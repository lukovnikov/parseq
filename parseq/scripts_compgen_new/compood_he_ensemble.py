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
from parseq.scripts_compgen_new.compood import run as run_tm, compute_auc_and_fprs, ORDERLESS, evaluate, \
    TransformerDecoderCell
from parseq.scripts_compgen_new.compood_gru import run as run_gru, GRUDecoderCell
from parseq.vocab import Vocab


class HybridSeqDecoder(torch.nn.Module):
    def __init__(self, *decoders, **kw):
        super(HybridSeqDecoder, self).__init__(**kw)
        self.decodercells = torch.nn.ModuleList()
        self.vocab = decoders[0].vocab
        self.max_size = decoders[0].max_size
        for decoder in decoders:
            for tagger in decoder.tagger:
                self.decodercells.append(tagger)

    def forward(self, x: torch.Tensor,
                     gold: torch.Tensor = None):  # --> implement how decoder operates end-to-end
        preds, prednll, maxmaxnll, entropy, total, avgconf, sumnll, stepsused, allprobs, allmask\
            = self.get_prediction(x)

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
        gold_trees = tensor_to_trees(gold, vocab=self.vocab)
        pred_trees = tensor_to_trees(preds, vocab=self.vocab)
        treeaccs = [float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device), "stepsused": stepsused}

        ret["decnll"] = prednll
        ret["sumnll"] = sumnll
        ret["maxmaxnll"] = maxmaxnll
        ret["entropy"] = entropy
        ret["avgconf"] = avgconf
        return ret, pred_trees

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_size
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@START@"]
        # yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        cells = self.decodercells

        # run encoder
        encs, encmasks = zip(*[cell.encode_source(x) for cell in cells])

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        caches = [None for _ in cells]
        conf_acc = None
        maxprob_acc = None
        maxmaxnll = None
        total = None
        entropy = None
        allprobs = []
        allmask = []
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run tagger
            # y = torch.cat([y, yend], 1)
            logitses, newcaches, probses = [], [], []
            for i, cell in enumerate(cells):
                if isinstance(cell, GRUDecoderCell):
                    prevout = y[:, -1]
                elif isinstance(cell, TransformerDecoderCell):
                    prevout = y
                logits, _cache = cell(tokens=prevout, enc=encs[i], encmask=encmasks[i], cache=caches[i])
                logitses.append(logits)

                _logits = logitses[i]
                if isinstance(cell, GRUDecoderCell):
                    _logits = _logits
                elif isinstance(cell, TransformerDecoderCell):
                    _logits = _logits[:, -1]
                probses.append(torch.softmax(_logits, -1))

                newcaches.append(_cache)
            caches = newcaches
            # logitses, caches = zip(*[tagger(tokens=y, enc=encs[i], encmask=encmasks[i], cache=caches[i]) for i, tagger in enumerate(cells)])
            # probses = [torch.softmax(logits[:, -1], -1) for logits in logitses]
            probs = sum(probses) / len(probses)     # average over all ensemble elements

            allprobs.append(probs)
            maxprobs, preds = probs.max(-1)
            _entropy = (-torch.log(probs.clamp_min(1e-7)) * probs).sum(-1)
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            allmask.append((~ended).long())
            total = total if total is not None else torch.zeros_like(maxprobs)
            total = total + torch.ones_like(maxprobs) * (~ended).float()
            conf_acc = conf_acc if conf_acc is not None else torch.ones_like(maxprobs)
            # conf_acc = conf_acc + maxprobs * (~ended).float()
            conf_acc = conf_acc * torch.where(ended, torch.ones_like(maxprobs), maxprobs)
            maxprob_acc = maxprob_acc if maxprob_acc is not None else torch.zeros_like(maxprobs)
            maxprob_acc = maxprob_acc + -torch.log(maxprobs) * (~ended).float()
            maxmaxnll = maxmaxnll if maxmaxnll is not None else torch.zeros_like(maxprobs)
            maxmaxnll = torch.max(maxmaxnll, torch.where(ended, torch.zeros_like(maxprobs), -torch.log(maxprobs)))
            entropy = entropy if entropy is not None else torch.zeros_like(_entropy)
            entropy = entropy + _entropy * (~ended).float()
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        allprobs = torch.stack(allprobs, 1)
        allmask = torch.stack(allmask, 1)

        return newy, maxprob_acc/total, maxmaxnll, entropy/total, total, conf_acc, maxprob_acc, steps_used.float(), allprobs, allmask



def run(tmlr=0.0001,
        grulr=0.0005,
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
        maxsize=50,
        seed=42,
        hdim=768,
        tmnumlayers=6,
        grunumlayers=2,
        numheads=12,
        tmdropout=0.1,
        grudropout=0.1,
        worddropout=0.,
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        ensemble=1,
        version="he_v1"
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)
    settings["tminnerensemble"] = True

    grusettings = {(k[3:] if k.startswith("gru") else k): v for k, v in settings.items() if not k.startswith("tm")}
    tmsettings = {(k[2:] if k.startswith("tm") else k): v for k, v in settings.items() if not k.startswith("gru")}

    grudecoder, indtestds, oodtestds = run_gru(**grusettings)
    tmdecoder, _, _ = run_tm(**tmsettings)

    # create a model that uses tmdecoder to generate output and uses both to measure OOD
    wandb.init(project=f"compood_he_ensemble", config=settings, reinit=True)
    decoder = HybridSeqDecoder(tmdecoder, grudecoder)
    results = evaluate(decoder, indtestds, oodtestds, batsize=tmbatsize, device=device)
    print("Results of the hybrid OOD:")
    print(json.dumps(results, indent=3))

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


def run_experiment(
        tmlr=-1.,
        grulr=-1.,
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
        ensemble=1,
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
        "tmlr": [0.0001],
        "grulr": [0.0005],
        "enclrmul": [1.],                  # use 1.
        "smoothing": [0.],
        "validinter": [1]
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
            if spec["gruepochs"] not in (25, 0, 1):
                return False
        elif spec["dataset"].startswith("scan"):
            if spec["gruepochs"] not in (40, 0, 1):
                return False
        if spec["dataset"].startswith("cfq"):
            if spec["grubatsize"] != 128:
                return False
        elif spec["dataset"].startswith("scan"):
            if spec["grubatsize"] != 256:
                return False
        if spec["dataset"].startswith("cfq"):
            if spec["tmepochs"] not in (20, 0, 1):
                return False
        elif spec["dataset"].startswith("scan"):
            if spec["tmepochs"] not in (25, 0, 1):
                return False
        return True

    print(__file__)
    p = __file__ + f".baseline.{dataset}"
    q.run_experiments_random(
        run, ranges, path_prefix=None, check_config=checkconfig, **settings)


# python compood_he_ensemble.py -dataset scan/mcd1 -gpu 0 -tmdropout 0.25 -grudropout 0.1 -gruepochs 1 -tmepochs 1


if __name__ == '__main__':
    q.argprun(run_experiment)