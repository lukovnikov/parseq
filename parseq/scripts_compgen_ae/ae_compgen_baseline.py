import json
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from typing import Dict

import wandb

import qelos as q
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from parseq.scripts_compgen_ae.data import get_datasets
from parseq.scripts_compgen_ae.models.dec import SeqDecoderBaseline
from parseq.scripts_compgen_ae.models.rnn import GRUDecoderCell
from parseq.scripts_compgen_ae.models.tm import TransformerDecoderCell
from torch.utils.data import DataLoader

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader
from transformers import AutoTokenizer, BertModel

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree, tree_to_lisp
from parseq.scripts_compgen_new.transformer import TransformerConfig, TransformerStack
from parseq.scripts_compgen_new.transformerdecoder import TransformerStack as TransformerStackDecoder
from parseq.vocab import Vocab


class Model(torch.nn.Module):
    def __init__(self, m, **kw):
        super(Model, self).__init__(**kw)
        self.model = m

    def forward(self, *args, **kw):
        return self.model(*args, **kw)


def run(lr=0.0001,
        enclrmul=0.01,
        smoothing=0.,
        gradnorm=3,
        batsize=60,
        epochs=16,
        patience=10,
        validinter=3,
        validfrac=0.1,
        warmup=3,
        cosinelr=False,
        dataset="scan/length",
        model="rnn",          # "rnn", "tm", "rnn-ae"
        augmode="random",
        maxsize=50,
        seed=42,
        hdim=768,
        numlayers=6,
        numheads=12,
        dropout=0.1,
        worddropout=0.,
        bertname="bert-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        smalltrainvalid=False,
        version="v1"
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    wandb.init(project=f"compgen_ae", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    tt = q.ticktock("script")
    tt.tick("data")
    trainvalidds = None
    traindl, validdl, testdl, auginpdl, augoutdl, tokenizer = get_datasets(dataset=dataset, augmode=augmode, tokenizer="vanilla",
                                                                           batsize=batsize, recompute=recomputedata, recomputeds=recomputedata,
                                                                           seed=seed, validfrac=validfrac)
    inpdic, fldic = tokenizer.inpvocab, tokenizer.outvocab
    tt.tock()

    tt.tick("model")
    if model == "rnn":
        cell = GRUDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, dropout=dropout, worddropout=worddropout)
        print(cell)
    elif model == "tm":
        cell = TransformerDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numheads=numheads,
                                      dropout=dropout, worddropout=worddropout,
                                      bertname=bertname, userelpos=userelpos, useabspos=not userelpos)
        print(f"one layer of decoder: \n {cell.decoder.block[0]}")
    decoder = SeqDecoderBaseline(cell, vocab=fldic, max_size=maxsize, smoothing=smoothing)
    decoder = Model(decoder)
    tt.tock()

    if testcode:
        tt.tick("testcode")
        batch = next(iter(traindl))
        # out = cell(batch[1])
        tt.tick("train")
        out = decoder(*batch)
        tt.tock()
        decoder.train(False)
        tt.tick("test")
        out = decoder(*batch)
        tt.tock()
        tt.tock("testcode")

    tloss = make_array_of_metrics("loss", "acc", "elemacc", reduction="mean")
    tmetrics = make_array_of_metrics("treeacc", "bleu", "nll", "acc", "elemacc", reduction="mean")
    vmetrics = make_array_of_metrics("treeacc", "bleu", "nll", "acc", "elemacc", reduction="mean")
    xmetrics = make_array_of_metrics("treeacc", "bleu", "nll", "acc", "elemacc", reduction="mean")

    # region parameters
    def get_parameters(m, _lr, _enclrmul):
        bertparams = []
        otherparams = []
        for k, v in m.named_parameters():
            if k.startswith("model.cell.encoder"):
                bertparams.append(v)
            else:
                otherparams.append(v)
        if len(bertparams) == 0:
            raise Exception("No encoder parameters found!")
        paramgroups = [{"params": bertparams, "lr": _lr * _enclrmul},
                       {"params": otherparams}]
        return paramgroups
    # endregion

    def get_optim(_m, _lr, _enclrmul, _wreg=0):
        paramgroups = get_parameters(_m, _lr=lr, _enclrmul=_enclrmul)
        optim = torch.optim.Adam(paramgroups, lr=lr, weight_decay=_wreg)
        return optim

    def clipgradnorm(_m=None, _norm=None):
        torch.nn.utils.clip_grad_norm_(_m.parameters(), _norm)

    if patience < 0 and "mcd" in dataset:
        patience = epochs
    eyt = q.EarlyStopper(vmetrics[0], patience=patience, min_epochs=10, more_is_better=True,
                         remember_f=lambda: deepcopy(decoder.model))

    def wandb_logger():
        d = {}
        for loss in tloss:
            d["train_"+loss.name] = loss.get_epoch_error()
        if evaltrain:
            for loss in tmetrics:
                d["train_"+loss.name] = loss.get_epoch_error()
        for loss in vmetrics:
            d["valid_"+loss.name] = loss.get_epoch_error()
        for loss in xmetrics:
            d["test_"+loss.name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim = get_optim(decoder, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        assert t_max > (warmup + 10)
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(low=0., high=1.0, steps=t_max-warmup) >> (0. * lr)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, on_before_optim_step=[lambda : clipgradnorm(_m=decoder, _norm=gradnorm)])

    if trainonvalidonly:
        traindl = validdl
        validdl = testdl

    trainepoch = partial(q.train_epoch, model=decoder,
                         dataloader=traindl,
                         optim=optim,
                         losses=tloss,
                         device=device,
                         _train_batch=trainbatch,
                         on_end=[lambda: lr_schedule.step()])

    trainevalepoch = partial(q.test_epoch,
                             model=decoder,
                             losses=tmetrics,
                             dataloader=traindl,
                             device=device)

    on_end_v = [lambda: eyt.on_epoch_end(), lambda: wandb_logger()]
    validepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=vmetrics,
                         dataloader=validdl,
                         device=device,
                         on_end=on_end_v)
    testepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=xmetrics,
                         dataloader=testdl,
                         device=device)

    tt.tick("training")
    if evaltrain:
        validfs = [trainevalepoch, validepoch]
    else:
        validfs = [validepoch]
    validfs = validfs + [testepoch]
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=validfs,
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    tt.tick("running test before reloading")
    testepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=xmetrics,
                         dataloader=testdl,
                         device=device)

    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock("ran test")

    if eyt.remembered is not None:
        tt.msg("reloading best")
        decoder.model = eyt.remembered

        tt.tick("rerunning validation")
        validres = validepoch()
        tt.tock(f"Validation results: {validres}")

    tt.tick("running train")
    trainres = trainevalepoch()
    print(f"Train tree acc: {trainres}")
    tt.tock()

    tt.tick("running test")
    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock()

    settings.update({"final_train_loss": tloss[0].get_epoch_error()})
    settings.update({"final_train_tree_acc": tmetrics[0].get_epoch_error()})
    settings.update({"final_valid_tree_acc": vmetrics[0].get_epoch_error()})
    settings.update({"final_test_tree_acc": xmetrics[0].get_epoch_error()})

    wandb.config.update(settings)
    q.pp_dict(settings)


def run_experiment(
        lr=-1.,
        enclrmul=-1.,
        smoothing=-1.,
        gradnorm=2,
        batsize=-1,
        epochs=-1,      # probably 11 is enough
        patience=-1,
        validinter=-1,
        warmup=3,
        cosinelr=False,
        dataset="default",
        model="rnn",
        augmode="random",
        maxsize=-1,
        seed=-1,
        hdim=-1,
        numlayers=-1,
        numheads=-1,
        dropout=-1.,
        worddropout=-1.,
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        trainonvalidonly=False,
        smalltrainvalid=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        ):

    settings = locals().copy()

    ranges = {
        "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left",
                    "scan/mcd1", "scan/mcd2", "scan/mcd3",
                    "cfq/random", "cfq/mcd1", "cfq/mcd2", "cfq/mcd3"],
        "dropout": [0.1, 0.25, 0.5],
        "worddropout": [0.],
        "seed": [42, 87646464, 456852],
        "epochs": [15],
        "batsize": [60],
        "hdim": [768],
        "numheads": [12],
        "numlayers": [6],
        "lr": [0.0001],
        "enclrmul": [0.01],                  # use 1.
        "smoothing": [0.],
        "validinter": [1],
    }

    if bertname.startswith("none") or bertname == "vanilla":
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["epochs"] = [58]
        ranges["hdim"] = [384]
        ranges["numheads"] = [6]
        ranges["batsize"] = [256]
        ranges["validinter"] = [3]

    if "rnn" in re.split("\s+-\|", model):
        ranges["numlayers"] = [2]
        ranges["hdim"] = [300]
        ranges["lr"] = [0.001]
        ranges["enclrmul"] = [1.]
        ranges["epochs"] = [50]
        ranges["validinter"] = [1]

    if dataset.startswith("cfq"):
        settings["maxsize"] = 160
    elif dataset.startswith("scan"):
        settings["maxsize"] = 50

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
        return True

    print(__file__)
    q.run_experiments_random(
        run, ranges, path_prefix=None, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)