import json
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from math import ceil
from time import sleep
from tqdm import tqdm
from transformers.models.t5.modeling_t5 import T5Stack
from typing import Dict, Callable

import wandb

import qelos as q
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import Adafactor, T5TokenizerFast, T5Model

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree
from parseq.scripts_cbqa.adapter_t5 import AdaptedT5WordEmbeddings
from parseq.scripts_cbqa.metaqa_dataset import MetaQADatasetLoader, KBDataset, QADataset
from parseq.scripts_resplit.t5 import load_t5_tokenizer, load_vanilla_t5, load_adaptered_t5, CosineWithRestart
from parseq.vocab import Vocab


# use only encoder part of T5 for QA


class ModelClassifier(torch.nn.Module):
    def __init__(self, dim):
        super(ModelClassifier, self).__init__()

        self.l1 = torch.nn.Linear(dim*2, dim)

        self.classifier_l1p1 = torch.nn.Linear(dim, dim)
        self.classifier_l1p2 = torch.nn.Linear(dim, dim)
        self.classifier_l1aux = torch.nn.Linear(dim, 1)
        self.classifier_l2 = torch.nn.Linear(dim*2, 1)
        self.classifier_nonlin = torch.nn.ReLU()

    def forward(self, xenc, cenc):
        ccat = torch.cat([xenc, cenc], -1)
        x = self.l1(ccat)
        x = self.classifier_nonlin(x)
        x = torch.cat([x, x], -1)
        x = self.classifier_l2(x)[:, 0]
        return x
        # _scores1p1 = self.classifier_l1p1(xenc)
        # _scores1p2 = self.classifier_l1p2(cenc)
        # _scores1 = _scores1p1 + _scores1p2
        # _scores2 = (xenc * cenc * self.classifier_l1aux.weight[None, :, 0])
        # _scores = self.classifier_nonlin(torch.cat([_scores1, _scores2], -1))
        # _scores = self.classifier_l2(_scores)[:, 0]
        # return _scores


class Model(torch.nn.Module):
    randomizedeval = True
    maxnumentsperexample = 2

    def __init__(self, encoder:T5Stack, dim, attdim=512, nheads=4, elemtensors=None, cachebatsize=100):
        super(Model, self).__init__()
        self.encoder = encoder
        self.xread = AttentionReadout(dim, attdim, nheads)
        self.entread = AttentionReadout(dim, attdim, nheads)
        self.classifier = ModelClassifier(dim)

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.elemtensors = elemtensors
        self.repr_cache = None
        self.cachebatsize = cachebatsize

    def copy_from(self, m:type("Model")):
        self.encoder = m.encoder
        self.xread = m.xread
        self.entread = m.entread
        self.classifier = m.classifier

    def get_tunable_params_and_set_requires_grad(self):
        ret = self.parameters()
        return ret

    def _compute_score(self, xenc, cenc):
        _scores = self.classifier(xenc, cenc)
        return _scores

    def _compute_scores_all(self, xenc, cenc):  # (batsize, dim), (outsize, dim)
        _cenc = cenc.repeat(xenc.size(0), 1)
        _xenc = xenc.repeat_interleave(cenc.size(0), 0)
        scores = self._compute_score(_xenc, _cenc)   # (batsize * outsize,)
        scores = scores.view(xenc.size(0), -1)
        return scores

    def train_forward(self, x, posids):
        # encode questions:
        xmask = x != 0
        xenc = self.encoder(x, attention_mask=xmask)[0]
        xenc = self.xread(xenc, attention_mask=xmask)

        # select entities to encode
        # copy positive ids into selection
        selection = set()
        for posids_ in posids:
            addition = set()
            # copy positive ids
            addition.update(posids_)
            # add some negative ids
            totalrange = set(range(len(self.elemtensors))) - addition - selection
            negsel = set(random.choices(list(totalrange), k=self.maxnumentsperexample-1))
            addition.update(negsel)
            if len(addition) > self.maxnumentsperexample:
                addition = set(random.choices(list(addition), k=self.maxnumentsperexample))
            # update global selection
            selection.update(addition)
        selection = sorted(list(selection))
        selectionmap = {v: k for k, v in enumerate(selection)}

        # compute supervision matrix (batsize, numcan)
        supmat = torch.zeros(len(x), len(selection), dtype=torch.long, device=x.device)
        for i, posids_ in enumerate(posids):
            for posid in posids_:
                if posid in selectionmap:
                    supmat[i, selectionmap[posid]] = 1
        supvec = supmat.view(-1)

        # get the entity strings to encode
        selection_tensors = [self.elemtensors[selection_i][None, :] for selection_i in selection]   # list of tensors
        selection_tensors = autocollate(selection_tensors)[0]
        selection_tensors = selection_tensors.to(x.device)

        # encode entity strings
        selmask = selection_tensors != 0
        selenc = self.encoder(selection_tensors, attention_mask=selmask)[0]
        selenc = self.entread(selenc, attention_mask=selmask)

        # compute scores
        xenc_repeated = xenc.repeat_interleave(len(selection), 0)       # (batsize * numcan, dim)
        selenc_repeated = selenc.repeat(len(x), 1)                      # (numcan * batsize, dim)
        scores = self._compute_score(xenc_repeated, selenc_repeated)    # (batsize * numcan,) scores

        # compute per-example losses and accuracies
        scoremat = scores.view(len(x), len(selection))
        loss = self.loss(scoremat, supmat.float()).sum(-1)
        accuracies = ((scoremat >= 0).long() == supmat).float().mean(-1)

        return {"loss": loss, "accuracy": accuracies, "negacc": torch.zeros_like(accuracies).fill_(-1)}, None

    def train_forward_(self, x, pos, neg):
        xmask = x != 0
        xenc = self.encoder(x, attention_mask=xmask)[0]
        xenc = self.xread(xenc, attention_mask=xmask)

        posmask = pos != 0
        posenc = self.encoder(pos, attention_mask=posmask)[0]
        posenc = self.entread(posenc, attention_mask=posmask)

        # posscores = (xenc * posenc).sum(-1)   # batch dot product
        posscores = self._compute_score(xenc, posenc)

        negmask = neg != 0
        negenc = self.encoder(neg, attention_mask=negmask)[0]
        negenc = self.entread(negenc, attention_mask=negmask)

        # negscores = (xenc * negenc).sum(-1)
        negscores = self._compute_score(xenc, negenc)

        loss = self.loss(posscores, torch.ones_like(posscores)) + self.loss(negscores, torch.zeros_like(negscores))
        accuracy = ((posscores >= 0).float() + (negscores < 0).float()) / 2
        negacc = (negscores < 0).float()
        return {"loss": loss, "accuracy": accuracy, "negacc": negacc}, None

    def test_forward(self, x, posids, randomized=None):
        _isdebugging = False
        xmask = x != 0
        xenc = self.encoder(x, attention_mask=xmask)[0]
        xenc = self.xread(xenc, attention_mask=xmask)

        device = x.device
        elem_reprs = self.get_elem_repr(device=device, _isdebugging=_isdebugging)

        randomized = randomized if randomized is not None else self.randomizedeval
        if randomized is not None and randomized is not False:  # instead of using all reprs, randomly select some negatives for evaluation purposes
            randomized = 20 if randomized is True else randomized
            selection = set()
            for posids_i in posids:
                selection.update(posids_i)
            totalrange = list(set(range(len(elem_reprs))) - selection)
            negsel = set(random.choices(totalrange, k=len(x) * randomized))
            selection.update(negsel)
            selection = sorted(list(selection))
            selectionmap = {v: k for k, v in enumerate(selection)}
            mappedposids = [set([selectionmap[posid] for posid in posids_i]) for posids_i in posids]
            _posids = posids
            posids = mappedposids
            if not _isdebugging:
                assert(max(selection) < len(elem_reprs))
            _elem_reprs = elem_reprs
            elem_reprs = torch.stack([elem_reprs[selection_i if selection_i < len(elem_reprs) else 0]
                          for selection_i in selection], 0)

        scores = []
        for elem_repr_batch in elem_reprs.split(self.cachebatsize, 0):
            # score_batch = torch.einsum("bd,md->bm", xenc, elem_repr_batch)
            score_batch = self._compute_scores_all(xenc, elem_repr_batch)
            scores.append(score_batch)
        scores = torch.cat(scores, 1)
        predicted = scores >= 0    # TODO: try different thresholds or prediction method?

        precisions, recalls, fscores = [], [], []
        for predicted_row, posids_i in zip(predicted.unbind(0), posids):
            predids_i = set(torch.nonzero(predicted_row)[:, 0].detach().cpu().numpy())
            precisions.append(len(predids_i & posids_i) / max(1, len(predids_i)))
            recalls.append(len(predids_i & posids_i) / max(1, len(posids_i)))

        fscores = [2 * recall_i * precision_i / max(1e-6, (recall_i + precision_i))
                   for precision_i, recall_i in zip(precisions, recalls)]
        return {
            "precision": torch.tensor(precisions),
            "recall": torch.tensor(recalls),
            "fscore": torch.tensor(fscores)
        }, None

    def get_elem_repr(self, device=torch.device("cpu"), _isdebugging=False):
        if self.repr_cache is None:
            # compute cache
            elemtensors = self.elemtensors
            if _isdebugging:
                elemtensors = elemtensors[:100]   # ONLY FOR DEBUGGING
            dl = DataLoader(Dataset([(t,) for t in elemtensors]), batch_size=self.cachebatsize, shuffle=False, collate_fn=autocollate)
            acc = []
            for batch in dl:
                x = batch[0]
                x = x.to(device)
                xmask = x != 0
                elemenc = self.encoder(x, attention_mask=xmask)[0]
                elemenc = self.entread(elemenc, attention_mask=xmask)
                acc.append(elemenc)
            reprs = torch.cat(acc, 0)
            self.repr_cache = reprs
        return self.repr_cache

    def clear_cache(self):
        self.repr_cache = None

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        else:
            return self.test_forward(*args, **kwargs)


class AttentionReadout(torch.nn.Module):
    def __init__(self, dim, attdim, heads):
        super(AttentionReadout, self).__init__()
        self.dim, self.attdim, self.nheads = dim, attdim, heads
        self.proj = torch.nn.Linear(dim, attdim)
        self.att = torch.nn.Linear(attdim//heads, heads)
        self.projback = torch.nn.Linear(attdim, dim)

    def forward(self, x, attention_mask=None):  # (batsize, seqlen, dim), (batsize, seqlen)
        xproj = self.proj(x).view(x.size(0), x.size(1), self.nheads, self.attdim//self.nheads)
        queries = self.att.weight
        xweights = torch.einsum("bshd,hd->bsh", xproj, queries)  # (batsize, seqlen, nheads)
        xweights = torch.where(attention_mask[:, :, None] == 1, xweights, torch.log(torch.zeros_like(xweights)))
        xalpha = torch.softmax(xweights, 1)        # (batsize, seqlen, nheads)
        xvals = torch.einsum("bshd,bsh->bhd", xproj, xalpha).view(x.size(0), -1)
        ret = self.projback(xvals)
        return ret


ORDERLESS = {"@WHERE", "@OR", "@AND", "@QUERY", "(@WHERE", "(@OR", "(@AND", "(@QUERY"}


def load_ds(dataset="metaqa/1", tokname="t5-small", recompute=False, subset=None):
    """
    :param dataset:
    :param validfrac:       how much of the original IID train set is used for IID validation set
    :param recompute:
    :param inptok_name:
    :return:
    """
    tt = q.ticktock("data")
    tt.tick(f"loading '{dataset}'")

    dataset, whichhops = dataset.split("/")

    tok = T5TokenizerFast.from_pretrained(tokname, additional_special_tokens=["[SEP1]", "[SEP2]", "[ANS]", "[ENT]", "[REL]"], extra_ids=0)

    tt.tick("loading data")
    kbds = MetaQADatasetLoader().load_kb(tok, recompute=recompute)
    qads = MetaQADatasetLoader().load_qa(whichhops, kbds[0].baseds, tok, recompute=recompute, subset=subset)
    print("length KBDS:", len(kbds))
    print("length QADS:", len(qads))
    print("length QADS train:", len(qads[0]))
    print("length QADS eval train:", len(qads[1]))
    print("length QADS valid:", len(qads[2]))
    print("length QADS test:", len(qads[3]))
    tt.tock("loaded data")

    kblens = []
    for tripletensor, posans, negans in tqdm(kbds[0]):
        kblens.append(tripletensor.size(-1))
    print(f"KB examples avg/max length is {np.mean(kblens):.1f}/{max(kblens)}")

    qalens = []
    for question, _ in tqdm(qads[0]):
        qalens.append(question.size(-1))
    for question, _ in tqdm(qads[2]):
        qalens.append(question.size(-1))
    for question, _ in tqdm(qads[3]):
        qalens.append(question.size(-1))

    print(f"QA examples avg/max length is {np.mean(qalens):.1f}/{max(qalens)}")
    return (tok,) + qads + kbds


def collate_fn(x, pad_value=0, numtokens=5000):
    lens = [len(xe[1]) for xe in x]
    a = list(zip(lens, x))
    a = sorted(a, key=lambda xe: xe[0], reverse=True)
    maxnum = int(numtokens/max(lens))
    b = a[:maxnum]
    b = [be[1] for be in b]
    ret = autocollate(b, pad_value=pad_value)
    return ret


def run(lr=0.0001,
        kbpretrain=False,
        useadafactor=False,
        gradnorm=3,
        gradacc=1,
        batsize=60,
        testbatsize=-1,
        epochs=16,
        validinter=3.,
        validfrac=0.1,
        warmup=0.1,
        cosinecycles=0,
        modelsize="small",
        usedefaultmodel=False,
        dataset="metaqa/1",
        maxsize=-1,
        seed=42,
        dropout=0.,
        testcode=False,
        debugcode=False,
        gpu=-1,
        recomputedata=False,
        version="v0",
        ):
    """
    :param lrmul:       multiplier for learning rate for secondary parameters
    :param modelsize:      what size of T5 model to use
    :param ftmode:      "ft" --> finetune entire model, pretrained weights are secondary params (lr=lr*lrmul)
                        otherwise --> see docs of load_t5() in t5.py
    :param originalinout:  if True, the original input and output layers of the original T5 decoder are used
    :param ptsize:
    :param testcode:
    :param evaltrain:
    :param trainonvalid:      training data = train + valid, validation data = test (DON'T USE THIS, ONLY FOR DEBUGGING!)
    :param trainonvalidonly:  training data = valid, validation data = test (DON'T USE THIS, ONLY FOR DEBUGGING!)
    :param recomputedata:     recompute data in data loading
    """

    settings = locals().copy()
    q.pp_dict(settings, indent=3)

    run = wandb.init(project=f"t5-cbqa-ftbase", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    if testbatsize == -1:
        testbatsize = batsize

    tt = q.ticktock("script")
    tt.tick("data")
    tok, trainds, evaltrainds, validds, testds, kbtrainds, kbvalidds = \
        load_ds(dataset=dataset, tokname=f"google/t5-v1_1-{modelsize}",
                recompute=recomputedata,
                subset=batsize*10 if debugcode else None)

    tt.tick("dataloaders")
    NUMWORKERS = 0

    kbtraindl = DataLoader(kbtrainds, batch_size=batsize, shuffle=True, collate_fn=autocollate, num_workers=NUMWORKERS)
    kbvaliddl = DataLoader(kbvalidds, batch_size=testbatsize, shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)

    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate, num_workers=NUMWORKERS)
    evaltraindl = DataLoader(evaltrainds, batch_size=testbatsize, shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)
    validdl = DataLoader(validds, batch_size=testbatsize, shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)
    testdl = DataLoader(testds, batch_size=testbatsize, shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)

    tt.tock()
    tt.tock()

    tt.tick("model")
    modelname = f"google/t5-v1_1-{modelsize}"
    t5model = T5Model.from_pretrained(modelname)
    emb = AdaptedT5WordEmbeddings(t5model.encoder.embed_tokens, tok)
    t5model.encoder.embed_tokens = emb
    encoder = t5model.encoder
    m = Model(encoder, encoder.config.d_model,
              elemtensors=kbtrainds.baseds.elems_pretokenized, cachebatsize=testbatsize*5)

    def _set_dropout(m, _p=0.):
        if isinstance(m, torch.nn.Dropout):
            m.p = _p
    m.apply(partial(_set_dropout, _p=dropout))
    tt.tock()

    if testcode:
        tt.tick("testcode")
        batch = next(iter(traindl))
        out = m(*batch)
        with torch.no_grad():
            m.train(False)
            batch = next(iter(validdl))
            out = m(*batch)
            m.train(True)
        tt.tock()

    # loss and metrics for fine-tuning on QA data
    tloss = make_array_of_metrics("loss", "accuracy", "negacc", reduction="mean")
    tmetrics = make_array_of_metrics("precision", "recall", "fscore", reduction="mean")
    vmetrics = make_array_of_metrics("precision", "recall", "fscore", reduction="mean")
    xmetrics = make_array_of_metrics("precision", "recall", "fscore", reduction="mean")

    kbtloss = make_array_of_metrics("loss", reduction="mean")
    kbtmetrics = make_array_of_metrics("precision", "recall", "fscore", reduction="mean")

    params = m.get_tunable_params_and_set_requires_grad()
    if useadafactor:
        tt.msg("Using AdaFactor.")
        optim = Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        tt.msg("Using Adam.")
        optim = torch.optim.Adam(params, lr=lr, weight_decay=0)

    if useadafactor:
        gradnorm = 1e6

    patience = epochs
    eyt = q.EarlyStopper(vmetrics[2], patience=patience, min_epochs=5, more_is_better=True,
                         remember_f=lambda: deepcopy(m))

    def wandb_logger_qaft():
        d = {}
        for name, loss in zip(["loss", "accuracy"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["precision", "recall", "fscore"], tmetrics):
            d["evaltrain_"+name] = loss.get_epoch_error()
        for name, loss in zip(["precision", "recall", "fscore"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        for name, loss in zip(["precision", "recall", "fscore"], xmetrics):
            d["test_"+name] = loss.get_epoch_error()
        wandb.log(d)

    def wandb_logger_kbft():
        d = {}
        for name, loss in zip(["loss"], kbtloss):
            d["kbtrain_"+name] = loss.get_epoch_error()
        for name, loss in zip(["precision", "recall", "fscore"], kbtmetrics):
            d["kbvalid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs * len(traindl) / gradacc
    warmupsteps = int(round(warmup * t_max))
    print(f"Total number of updates: {t_max} . Warmup: {warmupsteps}")
    if cosinecycles == 0:       # constant lr
        tt.msg("Using constant LR with warmup")
        lr_schedule = q.sched.Linear(0, 1, steps=warmupsteps) >> 1.
    else:
        tt.msg("Using cosine LR with restart and with warmup")
        lr_schedule = q.sched.Linear(0, 1, steps=warmupsteps) >> (CosineWithRestart(high=1., low=0.1, cycles=cosinecycles, steps=t_max-warmupsteps)) >> 0.1

    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    validepoch = partial(q.test_epoch,
                         model=m,
                         losses=vmetrics,
                         dataloader=validdl,
                         device=device,
                         on_end=[lambda: eyt.on_epoch_end(), lambda: m.clear_cache(), lambda: wandb_logger_qaft()])

    trainbatch = partial(q.train_batch,
                         gradient_accumulation_steps=gradacc,
                         on_before_optim_step=[
                             lambda: torch.nn.utils.clip_grad_norm_(params, gradnorm),
                             lambda: lr_schedule.step()
                         ]
                         )

    trainepoch = partial(q.train_epoch, model=m,
                         dataloader=traindl,
                         optim=optim,
                         losses=tloss,
                         device=device,
                         _train_batch=trainbatch,
                         on_end=[])

    trainevalepoch = partial(q.test_epoch,
                             model=m,
                             losses=tmetrics,
                             dataloader=evaltraindl,
                             device=device,
                             on_end=[lambda: m.clear_cache()])

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=[trainevalepoch, validepoch] if not debugcode else [],
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()]
                   )
    tt.tock("done training")

    settings.update({"train_loss_at_end": tloss[0].get_epoch_error()})

    testepoch = partial(q.test_epoch,
                         model=m,
                         losses=xmetrics,
                         dataloader=testdl,
                         device=device,
                         on_end=[lambda: m.clear_cache()])

    tt.tick("running test before reloading")
    testres = testepoch()

    settings.update({"test_fscore_at_end": xmetrics[2].get_epoch_error()})
    tt.tock(f"Test results: {testres}")

    if eyt.remembered is not None:
        tt.msg("reloading model with best validation accuracy")
        m.copy_from(eyt.remembered)

        tt.tick("running evaluation on subset of train")
        trainres = trainevalepoch()
        settings.update({"train_fscore_at_earlystop": tmetrics[2].get_epoch_error()})
        tt.tock(f"Train results: {trainres}")

        tt.tick("rerunning validation")
        validres = validepoch()
        settings.update({"valid_fscore_at_earlystop": vmetrics[2].get_epoch_error()})
        tt.tock(f"Validation results: {validres}")

        tt.tick("running test")
        testres = testepoch()
        settings.update({"test_fscore_at_earlystop": xmetrics[2].get_epoch_error()})
        tt.tock(f"Test results: {testres}")

    wandb.config.update(settings)
    q.pp_dict(settings)
    run.finish()
    # sleep(15)


def run_experiment(
        lr=-1.,
        kbpretrain=False,
        useadafactor=False,
        gradnorm=2,
        gradacc=1,
        batsize=-1,
        testbatsize=-1,
        epochs=-1,
        validinter=-1.,
        validfrac=0.1,
        warmup=0.1,
        cosinecycles=0,
        modelsize="small",
        dataset="default",
        maxsize=-1,
        seed=-1,
        dropout=-1.,
        testcode=False,
        debugcode=False,
        gpu=-1,
        recomputedata=False,
        ):

    settings = locals().copy()

    ranges = {
        "dataset": ["metaqa/1"],
        "dropout": [0.1],
        "seed": [42, 87646464, 456852],
        "epochs": [50],
        "batsize": [60],
        "lr": [0.0005],
        "modelsize": ["base"],
        "validinter": [2],
    }

    for k in ranges:
        if k in settings:
            if isinstance(settings[k], str) and settings[k] != "default":
                ranges[k] = settings[k].split(",")
            elif isinstance(settings[k], (int, float)) and settings[k] >= 0:
                ranges[k] = [settings[k]]
            else:
                pass
                # raise Exception(f"something wrong with setting '{k}'")
            del settings[k]

    def checkconfig(spec):
        return True

    q.run_experiments_random(
        run, ranges, path_prefix=None, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)