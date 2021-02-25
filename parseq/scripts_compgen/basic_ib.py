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
from torch.utils.data import DataLoader

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader
from transformers import AutoTokenizer, BertModel

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees
from parseq.scripts_compgen.baseline import BasicRelPosEmb, TransformerEmbeddings, ORDERLESS, load_ds, \
    SeqDecoderBaseline, TransformerDecoderCell
from parseq.scripts_compgen.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab


class TransformerDecoderCellWithVIB(TransformerDecoderCell):
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, useabspos=True,
                 relposmode="basic", relposrng=10,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(TransformerDecoderCellWithVIB, self).__init__(dim,
                                                            vocab=vocab, numlayers=numlayers, numheads=numheads,
                                                            userelpos=userelpos, useabspos=useabspos, relposmode=relposmode, relposrng=relposrng,
                                                            dropout=dropout, maxpos=maxpos, bertname=bertname, **kw)

        # reparam network:
        self.vib_linA = None
        self.vib_linA = torch.nn.Sequential(torch.nn.Linear(dim, dim*2), torch.nn.ReLU())

        self.vib_lin_mu = torch.nn.Linear(dim*2, dim)
        self.vib_lin_logvar = torch.nn.Linear(dim*2, dim)

    def sample_z(self, x):
        if self.vib_linA is not None:
            x = self.vib_linA(x)
        mu, logvar = self.vib_lin_mu(x), self.vib_lin_logvar(x)
        if self.training:
            ret = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            ret = None
        return ret, (mu, logvar)        # (batsize, seqlen, dim)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
        padmask = (tokens != 0)
        try:
            embs = self.dec_emb(tokens)
        except Exception as e:
            raise e
        if self.absposemb is not None:
            posembs = self.absposemb(torch.arange(tokens.size(1), device=tokens.device))[None]
            embs = embs + posembs
        relpos = None
        if self.relposemb is not None:      # compute relative positions
            positions = torch.arange(tokens.size(1), device=tokens.device)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        if cache is not None:
            embs = embs[:, -1:, :]
            if relpos is not None:
                relpos = relpos[:, -1:, :, :]
        _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=True,
                     past_key_value_states=cache,
                     relpos=relpos)
        ret = _ret[0]
        c = ret
        cache = _ret[1]

        zs, (mu, logvar) = self.sample_z(c)        # (batsize, seqlen, dim)
        if self.training:
            logits = self.out(zs)  # (batsize, seqlen, vocabsize)
            return logits, cache, (mu, logvar)
        else:
            assert zs is None
            logits = self.out(mu)
            return logits, cache


class SeqDecoderBaselineWithVIB(SeqDecoderBaseline):
    def __init__(self, tagger:TransformerDecoderCellWithVIB, vocab=None, max_size:int=100, smoothing:float=0., priorweight=1., **kw):
        super(SeqDecoderBaselineWithVIB, self).__init__(tagger=tagger, vocab=vocab, max_size=max_size, smoothing=smoothing, **kw)
        self.priorweight = priorweight

    def compute_loss(self, logits, tgt, mulogvar):
        """
        :param logits:      (batsize, seqlen, vocsize)
        :param tgt:         (batsize, seqlen)
        :return:
        """
        mask = (tgt != 0).float()
        # batsize, seqlen = tgt.size()

        logprobs = self.logsm(logits)
        if self.smoothing > 0:
            loss = self.loss(logprobs, tgt)
        else:
            # print(tgt.max(), tgt.min(), logprobs.size(-1))
            # print()
            loss = self.loss(logprobs.permute(0, 2, 1), tgt)      # (batsize, seqlen)
        loss = loss * mask
        loss = loss.sum(-1)

        priorkl = torch.zeros(loss.size(0), device=loss.device)
        if self.priorweight > 0:
            mu, logvar = mulogvar  # (batsize, seqlen, dim)
            priorkls = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1)    # (batsize, seqlen)
            priorkls = priorkls * mask
            priorkl = priorkls.sum(-1) * self.priorweight
            loss = loss + priorkl

        best_pred = logits.max(-1)[1]   # (batsize, seqlen)
        best_gold = tgt
        same = best_pred == best_gold
        same = same | ~(mask.bool())
        acc = same.all(-1)  # (batsize,)
        return loss, priorkl, acc.float()

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits, cache, (mu, logvar) = self.tagger(tokens=newy, enc=enc, encmask=encmask, cache=None)
        # logits: (numsamples, batsize, seqlen, vocabsize)
        # cache: ...
        # mu, sigma: (numsamples, batsize, dim)
        # compute loss: different versions do different masking and different targets
        loss, priorkl, acc = self.compute_loss(logits, tgt, (mu, logvar))
        return {"loss": loss, "priorkl": priorkl, "acc": acc}, logits


def run(lr=0.0001,
        enclrmul=0.1,
        smoothing=0.1,
        gradnorm=3,
        batsize=60,
        epochs=16,
        patience=10,
        validinter=3,
        validfrac=0.1,
        warmup=3,
        cosinelr=False,
        dataset="scan/length",
        maxsize=50,
        seed=42,
        hdim=768,
        numlayers=6,
        numheads=12,
        dropout=0.1,
        bertname="bert-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        priorweight=1.,
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    wandb.init(project=f"compgen_baseline", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    tt = q.ticktock("script")
    tt.tick("data")
    trainds, validds, testds, fldic = load_ds(dataset=dataset, validfrac=validfrac, bertname="bert"+bertname[4:])

    tt.tick("dataloaders")
    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    validdl = DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    testdl = DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    # print(json.dumps(next(iter(trainds)), indent=3))
    # print(next(iter(traindl)))
    # print(next(iter(validdl)))
    tt.tock()
    tt.tock()

    tt.tick("model")
    cell = TransformerDecoderCellWithVIB(hdim, vocab=fldic, numlayers=numlayers, numheads=numheads, dropout=dropout,
                                  bertname=bertname, userelpos=userelpos, useabspos=not userelpos)
    decoder = SeqDecoderBaselineWithVIB(cell, vocab=fldic, max_size=maxsize, smoothing=smoothing, priorweight=priorweight)
    print(f"one layer of decoder: \n {cell.decoder.block[0]}")
    tt.tock()

    if testcode:
        tt.tick("testcode")
        batch = next(iter(traindl))
        # out = tagger(batch[1])
        tt.tick("train")
        out = decoder(*batch)
        tt.tock()
        decoder.train(False)
        tt.tick("test")
        out = decoder(*batch)
        tt.tock()
        tt.tock("testcode")

    tloss = make_array_of_metrics("loss", "priorkl", "acc", reduction="mean")
    tmetrics = make_array_of_metrics("treeacc", reduction="mean")
    vmetrics = make_array_of_metrics("treeacc", reduction="mean")
    xmetrics = make_array_of_metrics("treeacc", reduction="mean")

    # region parameters
    def get_parameters(m, _lr, _enclrmul):
        bertparams = []
        otherparams = []
        for k, v in m.named_parameters():
            if "encoder_model." in k:
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

    if patience < 0:
        patience = epochs
    eyt = q.EarlyStopper(vmetrics[0], patience=patience, min_epochs=30, more_is_better=True,
                         remember_f=lambda: deepcopy(cell))

    def wandb_logger():
        d = {}
        for name, loss in zip(["loss", "priorkl", "acc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim = get_optim(cell, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        assert t_max > (warmup + 10)
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(low=0., high=1.0, steps=t_max-warmup) >> (0. * lr)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, on_before_optim_step=[lambda : clipgradnorm(_m=cell, _norm=gradnorm)])

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

    tt.tick("training")
    if evaltrain:
        validfs = [trainevalepoch, validepoch]
    else:
        validfs = [validepoch]
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
        decoder.tagger = eyt.remembered
        tagger = eyt.remembered

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
        patience=100,
        validinter=-1,
        warmup=3,
        cosinelr=False,
        dataset="default",
        maxsize=50,
        seed=-1,
        hdim=-1,
        numlayers=-1,
        numheads=-1,
        dropout=-1.,
        bertname="none-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        priorweight=-1.,
        ):

    settings = locals().copy()

    _dataset = None
    if dataset.endswith("mcd"):
        _dataset = [dataset+str(i) for i in range(3)]

    ranges = {
        "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"],
        "dropout": [0.1, 0.25, 0.5],
        "seed": [42, 87646464, 456852],
        "epochs": [15],
        "batsize": [60],
        "hdim": [768],
        "numheads": [12],
        "numlayers": [6],
        "lr": [0.0001],
        "enclrmul": [0.1],                  # use 1.
        "smoothing": [0., 0.1],
        # "patience": [-1],
        # "warmup": [20],
        "validinter": [2],
        # "gradacc": [1],
        "priorweight": [0.01, 0.03, 0.1, 0.3, 1.],
    }

    if _dataset is not None:
        ranges["dataset"] = _dataset
        settings["dataset"] = "default"

    if bertname.startswith("none"):
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["epochs"] = [71]
        ranges["hdim"] = [384]
        ranges["numheads"] = [6]
        ranges["batsize"] = [256]
        ranges["validinter"] = [5]

        ranges["dropout"] = [0.1]
        ranges["smoothing"] = [0.]

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
    p = __file__ + f".baseline.{dataset}"
    q.run_experiments_random(
        run, ranges, path_prefix=p, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)