import itertools

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
from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration
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
from parseq.scripts_cbqa.adapter_t5 import AdaptedT5WordEmbeddings, AdaptedT5LMHead
from parseq.scripts_cbqa.metaqa_dataset import MetaQADatasetLoader, KBDataset, QADataset
from parseq.scripts_resplit.t5 import load_t5_tokenizer, load_vanilla_t5, load_adaptered_t5, CosineWithRestart
from parseq.vocab import Vocab


# uses decoder to generate answer string

class Model(torch.nn.Module):
    randomizedeval = True
    maxnumentsperexample = 1
    maxlen=100

    def __init__(self, model:T5Model, dim, maxsetsize=1000, tok=None):
        super(Model, self).__init__()
        self.model = model
        self.tok = tok

    def get_tunable_params_and_set_requires_grad(self):
        ret = self.parameters()
        return ret

    def train_forward(self, x, y):
        # encode questions:
        xmask = x != 0
        yinp = y[:, :-1]
        ylabel = y[:, 1:].clone().detach()
        ylabel[ylabel == 0] = -100

        outputs = self.model(input_ids=x, attention_mask=xmask,
                             decoder_input_ids=yinp, labels=ylabel)
        loss, logits = outputs.loss, outputs.logits
        _, preds = torch.max(logits, -1)
        same = preds == ylabel
        elemacc = same & (ylabel != -100)
        elemacc = elemacc.float().sum(1) / (ylabel != -100).float().sum(1)
        same |= (ylabel == -100)
        acc = torch.all(same, -1).float()
        loss = torch.ones(len(x), device=x.device) * loss
        return {"loss": loss, "accuracy": acc, "elemacc": elemacc}, None

    def test_forward(self, x, y):
        xmask = x != 0
        maxlen = min(self.maxlen, y.size(1) + 1)

        preds = self.model.generate(
            input_ids = x,
            decoder_input_ids=y[:, 0:1],
            attention_mask = xmask,
            max_length=maxlen,
            num_beams=1,
            do_sample=False,
            )

        if preds.size(-1) < y.size(-1):
            # acc = torch.zeros(len(x), device=x.device)
            app = torch.zeros(preds.size(0), y.size(1) - preds.size(1), device=preds.device, dtype=preds.dtype)
            preds = torch.cat([preds, app], 1)
            same = preds == y
        else:
            same = preds[..., :y.size(-1)] == y
        same |= y == 0
        acc = torch.all(same, -1).float()

        # predstring = [self.tok.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=True) for g in preds]
        # targetstring = [self.tok.decode(t, skip_special_tokens=False, clean_up_tokenization_spaces=True) for t in y.unbind(0)]
        return {"accuracy": acc}, None

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        else:
            return self.test_forward(*args, **kwargs)


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

    extratokens = ["[SEP1]", "[SEP2]", "[ANS]", "[ENT]", "[REL]", "[SEPITEM]", "[BOS]", "[ENDOFSET]"]
    extratokens = extratokens + [f"[ITEM-{i}]" for i in range(1000)]
    tok = T5TokenizerFast.from_pretrained(tokname, additional_special_tokens=extratokens, extra_ids=0)

    tt.tick("loading data")
    kbds = MetaQADatasetLoader().load_kb(tok, recompute=recompute, mode="seqset")
    qads = MetaQADatasetLoader().load_qa(whichhops, kbds[0].baseds, tok, recompute=recompute, subset=subset, mode="seqset")
    print("length KBDS:", len(kbds))
    print("length QADS:", len(qads))
    print("length QADS train:", len(qads[0]))
    print("length QADS eval train:", len(qads[1]))
    print("length QADS valid:", len(qads[2]))
    print("length QADS test:", len(qads[3]))
    tt.tock("loaded data")

    kblens = []
    kbanswerlens = []
    for tripletensors, answertensors in tqdm(kbds[0]):
        for tripletensor in tripletensors:
            kblens.append(tripletensor.size(-1))
        for answertensor in answertensors:
            kbanswerlens.append(answertensor.size(-1))
    print(f"KB triple avg/max length is {np.mean(kblens):.1f}/{max(kblens)}")
    print(f"KB answer avg/max length is {np.mean(kbanswerlens):.1f}/{max(kbanswerlens)}")

    qalens = []
    qaanswerlens = []
    for questions, answers in tqdm(qads[0]):
        for question in questions:
            qalens.append(question.size(-1))
        for answer in answers:
            qaanswerlens.append(answer.size(-1))
    for questions, answers in tqdm(qads[2]):
        for question in questions:
            qalens.append(question.size(-1))
        for answer in answers:
            qaanswerlens.append(answer.size(-1))
    for questions, answers in tqdm(qads[3]):
        for question in questions:
            qalens.append(question.size(-1))
        for answer in answers:
            qaanswerlens.append(answer.size(-1))

    print(f"QA questions avg/max length is {np.mean(qalens):.1f}/{max(qalens)}")
    print(f"QA answers avg/max length is {np.mean(qaanswerlens):.1f}/{max(qaanswerlens)}")
    return (tok,) + qads + kbds


def collate_fn(data):
    newdata = []
    for datapoint in data:
        inps, outs = datapoint
        for inp, out in zip(inps, outs):
            newdata.append((inp, out))
    ret = autocollate(newdata)
    return ret


def run(lr=0.0001,
        kbpretrain=False,
        useadafactor=False,
        gradnorm=3,
        gradacc=1,
        batsize=60,
        maxlen=30,
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

    run = wandb.init(project=f"t5-cbqa-ftbase-setdec", config=settings, reinit=True)
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
                subset=5000 if debugcode else None)

    tt.tick("dataloaders")
    NUMWORKERS = 0

    kbtraindl = DataLoader(kbtrainds, batch_size=batsize, shuffle=True, collate_fn=collate_fn, num_workers=NUMWORKERS)
    kbvaliddl = DataLoader(kbvalidds, batch_size=testbatsize, shuffle=False, collate_fn=collate_fn, num_workers=NUMWORKERS)

    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=collate_fn, num_workers=NUMWORKERS)
    evaltraindl = DataLoader(evaltrainds, batch_size=testbatsize, shuffle=False, collate_fn=collate_fn, num_workers=NUMWORKERS)
    validdl = DataLoader(validds, batch_size=testbatsize, shuffle=False, collate_fn=collate_fn, num_workers=NUMWORKERS)
    testdl = DataLoader(testds, batch_size=testbatsize, shuffle=False, collate_fn=collate_fn, num_workers=NUMWORKERS)

    tt.tock()
    tt.tock()

    tt.tick("model")
    modelname = f"google/t5-v1_1-{modelsize}"
    t5model = T5ForConditionalGeneration.from_pretrained(modelname)
    emb = AdaptedT5WordEmbeddings(t5model.encoder.embed_tokens, tok)
    t5model.encoder.embed_tokens = emb
    t5model.decoder.embed_tokens = emb
    t5model.shared = emb
    lmhead = AdaptedT5LMHead(t5model.lm_head, tok)
    t5model.lm_head = lmhead
    m = Model(t5model, t5model.config.d_model, tok=tok)
    m.maxlen = maxlen

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
    tloss = make_array_of_metrics("loss", "accuracy", "elemacc", reduction="mean")
    tmetrics = make_array_of_metrics("accuracy", reduction="mean")
    vmetrics = make_array_of_metrics("accuracy", reduction="mean")
    xmetrics = make_array_of_metrics("accuracy", reduction="mean")

    kbtloss = make_array_of_metrics("loss", "accuracy", reduction="mean")
    kbtmetrics = make_array_of_metrics("accuracy", reduction="mean")

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
    eyt = q.EarlyStopper(vmetrics[0], patience=patience, min_epochs=5, more_is_better=True,
                         remember_f=lambda: deepcopy(m))

    def wandb_logger_qaft():
        d = {}
        for name, loss in zip(["loss", "accuracy", "elemacc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["accuracy"], tmetrics):
            d["evaltrain_"+name] = loss.get_epoch_error()
        for name, loss in zip(["accuracy"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        for name, loss in zip(["accuracy"], xmetrics):
            d["test_"+name] = loss.get_epoch_error()
        wandb.log(d)

    def wandb_logger_kbft():
        d = {}
        for name, loss in zip(["loss", "accuracy"], kbtloss):
            d["kbtrain_"+name] = loss.get_epoch_error()
        for name, loss in zip(["accuracy"], kbtmetrics):
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
                         on_end=[lambda: eyt.on_epoch_end(), lambda: wandb_logger_qaft()])

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
                         on_start=[],
                         on_end=[])

    trainevalepoch = partial(q.test_epoch,
                             model=m,
                             losses=tmetrics,
                             dataloader=evaltraindl,
                             device=device,
                             on_end=[])

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=[trainevalepoch, validepoch] if not debugcode else [trainevalepoch],
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter,
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
        maxlen=30,
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
        "dropout": [0.],
        "seed": [42, 87646464, 456852],
        "epochs": [50],
        "batsize": [60],
        "lr": [0.0005],
        "modelsize": ["base"],
        "validinter": [5],
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