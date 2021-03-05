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
from parseq.scripts_compgen.baseline import BasicRelPosEmb, TransformerEmbeddings, TransformerDecoderCell, load_ds
from parseq.scripts_compgen.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab


class SetModel(torch.nn.Module):
    """ Predicts the set of output tokens.
    Implemented as a tagging model that labels every input position with one of the output tokens.
    The probability that a certain output token is in the set is computed
    by taking the max logit over sequence dimension and taking Sigmoid of that.
    """
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, useabspos=True,
                 relposmode="basic", relposrng=10,
                 dropout:float=0., sidedrop=0., maxpos=512, bertname="bert-base-uncased", mode="normal", priorweight=0., **kw):
        super(SetModel, self).__init__(**kw)
        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.userelpos = userelpos
        self.relposrng = relposrng
        self.useabspos = useabspos

        self.out = torch.nn.Linear(self.dim, self.vocabsize)
        self.bertname = bertname
        if self.bertname.startswith("none") or self.bertname == "vanilla":
            self.encrelposemb = None
            if self.userelpos is True:
                if relposmode == "basic":
                    self.encrelposemb = BasicRelPosEmb(self.dim, relposrng)
                # elif relposmode == "mod":
                #     self.relposemb = ModRelPosEmb(self.dim, relposrng, levels=4)
                else:
                    raise Exception(f"Unrecognized relposmode '{relposmode}'")
            bname = "bert" + self.bertname[4:]
            if self.bertname == "vanilla":
                inpvocabsize = inpvocab.number_of_ids()
            else:
                tokenizer = AutoTokenizer.from_pretrained(bname)
                inpvocabsize = tokenizer.vocab_size
            encconfig = TransformerConfig(vocab_size=inpvocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                          d_kv=int(self.dim / numheads), attention_dropout_rate=0.,
                                          num_layers=numlayers, num_heads=numheads,
                                          dropout_rate=dropout, sideways_dropout=sidedrop, vib_att=mode.replace(" ", "")=="vibatt")
            encemb = TransformerEmbeddings(encconfig.vocab_size, encconfig.d_model, dropout=dropout,
                                           max_position_embeddings=maxpos, useabspos=useabspos)
            self.encoder_model = TransformerStack(encconfig, encemb, rel_emb=self.encrelposemb)
        else:
            self.encoder_model = BertModel.from_pretrained(self.bertname,
                                                           hidden_dropout_prob=min(dropout, 0.2),
                                                           attention_probs_dropout_prob=min(dropout, 0.1))
        self.adapter = None
        if self.encoder_model.config.hidden_size != self.dim:
            self.adapter = torch.nn.Linear(self.encoder_model.config.hidden_size, self.dim, bias=False)

        self.reset_parameters()

        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.mode = mode
        self.priorweight = priorweight

        if self.mode == "vib":
            self.vib_lin_mu = torch.nn.Linear(dim, dim)
            self.vib_lin_logvar = torch.nn.Linear(dim, dim)

    def reset_parameters(self):
        pass

    def sample_z(self, x, xmask):
        mu, logvar = self.vib_lin_mu(x), self.vib_lin_logvar(x)
        if self.training:
            ret = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            ret = mu

        priorkl = torch.zeros(ret.size(0), device=ret.device)
        if self.priorweight > 0 and self.training:
            priorkls = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1)  # (batsize, seqlen)
            priorkls = priorkls * xmask.float()
            priorkl = priorkls.sum(-1)
        return ret, priorkl        # (batsize, seqlen, dim)

    def encode(self, x):
        encmask = (x != 0)
        relpos = None
        if self.encrelposemb is not None:      # compute relative positions
            positions = torch.arange(x.size(1), device=x.device)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        if relpos is not None:
            encoderouts = self.encoder_model(x, attention_mask=encmask, relpos=relpos)
        else:
            encoderouts = self.encoder_model(x, attention_mask=encmask)
        encs = encoderouts[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        if self.mode == "vibatt":
            priorkl = encoderouts[1]
            priorkl = priorkl * encmask
            priorkl = priorkl.sum(-1)
            return encs, encmask, priorkl
        else:
            return encs, encmask

    def compute_loss(self, logits, golds):
        bce = self.bce(logits, golds)       # (batsize, vocsize)
        bce = bce.sum(-1)

        preds = (torch.sigmoid(logits) > 0.5)        # (batsize, vocsize)
        acc = torch.all(preds == golds, 1)

        return bce, acc.float()

    def forward(self, x:torch.Tensor, y:torch.Tensor):      # integer tensors
        priorloss = None
        if self.mode == "vibatt":
            enc, encmask, priorloss = self.encode(x)
        else:
            enc, encmask = self.encode(x)

        if self.mode == "vib":
            enc, priorloss = self.sample_z(enc, encmask)

        seqlogits = self.out(enc)      # (batsize, seqlen, vocsize)
        seqlogits = seqlogits + -9999 * (1 - encmask.float())[:, :, None]    # --> sets masked positions to -infty
        logits, _ = seqlogits.max(1)    # (batsize, vocsize) -- max logits for every possible output token
        golds = torch.zeros_like(logits)
        golds[:, 0] = 1
        golds.scatter_(-1, y, 1)

        loss, acc = self.compute_loss(logits, golds)
        if priorloss is None:
            priorloss = torch.zeros_like(loss)

        priorloss = priorloss * self.priorweight
        loss = loss + priorloss
        return {"loss": loss, "priorkl": priorloss, "acc": acc}, logits


def run(lr=0.0001,
        enclrmul=0.1,
        smoothing=0.1,
        gradnorm=3,
        batsize=60,
        epochs=16,
        patience=10,
        validinter=1,
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
        sidedrop=0.0,
        bertname="bert-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        mode="normal",      # "normal", "vib", "aib"
        priorweight=1.,
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    wandb.init(project=f"compgen_set", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    tt = q.ticktock("script")
    tt.tick("data")
    trainds, validds, testds, fldic, inpdic = load_ds(dataset=dataset, validfrac=validfrac, bertname=bertname, recompute=recomputedata)
    if trainonvalid:
        trainds = trainds + validds
        validds = testds

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
    model = SetModel(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numheads=numheads, dropout=dropout, sidedrop=sidedrop,
                                  bertname=bertname, userelpos=userelpos, useabspos=not userelpos, mode=mode, priorweight=priorweight)
    tt.tock()

    if testcode:
        tt.tick("testcode")
        batch = next(iter(traindl))
        # out = tagger(batch[1])
        tt.tick("train")
        out = model(*batch)
        tt.tock()
        model.train(False)
        tt.tick("test")
        out = model(*batch)
        tt.tock()
        tt.tock("testcode")

    tloss = make_array_of_metrics("loss", "priorkl", "acc", reduction="mean")
    tmetrics = make_array_of_metrics("loss", "priorkl", "acc", reduction="mean")
    vmetrics = make_array_of_metrics("loss", "priorkl", "acc", reduction="mean")
    xmetrics = make_array_of_metrics("loss", "priorkl", "acc", reduction="mean")

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
                         remember_f=lambda: deepcopy(model))

    def wandb_logger():
        d = {}
        for name, loss in zip(["loss", "priorkl", "acc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["acc"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["acc"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim = get_optim(model, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        assert t_max > (warmup + 10)
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(low=0., high=1.0, steps=t_max-warmup) >> (0. * lr)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, on_before_optim_step=[lambda : clipgradnorm(_m=model, _norm=gradnorm)])

    print("using test data for validation")
    validdl = testdl

    if trainonvalidonly:
        traindl = validdl
        validdl = testdl

    trainepoch = partial(q.train_epoch, model=model,
                         dataloader=traindl,
                         optim=optim,
                         losses=tloss,
                         device=device,
                         _train_batch=trainbatch,
                         on_end=[lambda: lr_schedule.step()])

    trainevalepoch = partial(q.test_epoch,
                             model=model,
                             losses=tmetrics,
                             dataloader=traindl,
                             device=device)

    on_end_v = [lambda: eyt.on_epoch_end(), lambda: wandb_logger()]
    validepoch = partial(q.test_epoch,
                         model=model,
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
                         model=model,
                         losses=xmetrics,
                         dataloader=testdl,
                         device=device)

    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock("ran test")

    if eyt.remembered is not None:
        tt.msg("reloading best")
        model = eyt.remembered

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
    settings.update({"final_train_acc": tmetrics[2].get_epoch_error()})
    settings.update({"final_valid_acc": vmetrics[2].get_epoch_error()})
    settings.update({"final_test_acc": xmetrics[2].get_epoch_error()})

    wandb.config.update(settings)
    q.pp_dict(settings)


def run_experiment(
        lr=-1.,
        enclrmul=-1.,
        smoothing=-1.,
        gradnorm=2,
        batsize=-1,
        epochs=-1,      # probably 11 is enough
        warmup=3,
        cosinelr=False,
        dataset="default",
        maxsize=-1,
        seed=-1,
        hdim=-1,
        numlayers=-1,
        numheads=-1,
        dropout=-1.,
        sidedrop=0.,
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        trainonvalidonly=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        mode="normal",      # "normal", "vib"
        priorweight=-1.,
        ):

    settings = locals().copy()
    settings["patience"] = -1
    settings["validinter"] = 1

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
        # "validinter": [2],
        # "gradacc": [1],
        # "sidedrop": [0., 0.5, 0.9],
    }

    if bertname.startswith("none") or bertname == "vanilla":
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["epochs"] = [40]
        ranges["hdim"] = [384]
        ranges["numheads"] = [6]
        ranges["batsize"] = [512]
        # ranges["validinter"] = [3]

        ranges["dropout"] = [0.1]
        ranges["smoothing"] = [0.]

    if dataset.startswith("cfq"):
        settings["maxsize"] = 200
    elif dataset.startswith("scan"):
        settings["maxsize"] = 50

    if mode in ("vib", "vibatt"):
        ranges["priorweight"] = [0.01, 0.1, 1.]

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