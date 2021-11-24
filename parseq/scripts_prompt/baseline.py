import json
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from typing import Dict, Callable

import wandb

import qelos as q
import torch
import numpy as np
from torch.utils.data import DataLoader

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree
from parseq.scripts_prompt.t5 import load_t5_tokenizer, load_t5
from parseq.vocab import Vocab


class SeqDecoderT5(torch.nn.Module):

    def __init__(self, model,
                 max_size:int=100,
                 batch_to_strs:Callable=None,
                 dropout=0.,
                 **kw):
        super(SeqDecoderT5, self).__init__(**kw)
        self.model = model
        self.max_size = max_size
        self.batch_to_strs = batch_to_strs
        self.dropout = dropout
        # TODO: use dropout!

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

    def tensor_to_trees(self, x):
        xstrs = self.batch_to_strs(x)
        xstrs = [xi.replace("@START@", "") for xi in xstrs]
        xstrs = [re.sub("::\d+", "", xstr) for xstr in xstrs]
        trees = []
        for xstr in xstrs:
            # drop everything after @END@, if present
            xstr = xstr.split("@END@")[0].strip()
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

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        preds = self.model.generate(x, max_length=self.max_size)

        # compute loss and metrics
        gold_trees = self.tensor_to_trees(gold)
        pred_trees = self.tensor_to_trees(preds)
        treeaccs = [float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device)}
        return ret, pred_trees

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        modelout = self.model(
            input_ids=x,
            attention_mask=(x!=0).long(),
            labels=y
        )
        loss, logits = modelout.loss, modelout.logits
        _, preds = logits.max(-1)
        same = (preds == y) & (y != 0)
        elemacc = same.float().sum(1) / (y != 0).sum(1)
        elemacc = elemacc.mean(0)
        same |= (y == 0)
        seqacc = torch.all(same, 1).float().mean(0)
        return {"loss": loss, "acc": seqacc}, logits

    def extract_training_example(self, x, y):
        ymask = (y != 0).float()
        ylens = ymask.sum(1).long()
        newy = y
        newy = torch.cat([torch.ones_like(newy[:, 0:1]) * self.vocab["@START@"], newy], 1)
        newy = torch.cat([newy, torch.zeros_like(newy[:, 0:1])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            newy[i, ylen+1] = self.vocab["@END@"]

        goldy = newy[:, 1:]
        # tgt = torch.zeros(goldy.size(0), goldy.size(1), self.vocab.number_of_ids(), device=goldy.device)
        # tgt = tgt.scatter(2, goldy[:, :, None], 1.)
        # tgtmask = (goldy != 0).float()

        newy = newy[:, :-1]
        return x, newy, goldy


class Tokenizer(object):
    def __init__(self, inptok=None, outtok=None, outvocab:Vocab=None, **kw):
        super(Tokenizer, self).__init__(**kw)
        self.inptok = inptok
        self.outtok = outtok
        self.outvocab = outvocab

    def tokenize(self, inps:str, outs):
        # input:
        inputs = self.inptok(inps, return_tensors='pt')
        inptensor = inputs.input_ids[0]

        if self.outtok is None:
            outtoks = self.get_out_toks(outs)
            outtensor = self.tensorize_output(outtoks, self.outvocab)
        else:
            outputs = self.outtok(outs.lower(), return_tensors='pt')
            outtensor = outputs.input_ids[0]
        ret = {"inps": inps, "outs":outs, "inptensor": inptensor, "outtensor": outtensor}
        ret = (ret["inptensor"], ret["outtensor"])
        return ret

    def get_toks(self, x):
        return x.strip().split(" ")

    def get_out_toks(self, x):
        return self.get_toks(x)

    def tensorize_output(self, x, vocab):
        ret = [vocab[xe] for xe in x]
        ret = torch.tensor(ret)
        return ret


ORDERLESS = {"@WHERE", "@OR", "@AND", "@QUERY", "(@WHERE", "(@OR", "(@AND", "(@QUERY"}


def load_ds(dataset="scan/random", validfrac=0.1, recompute=False, inptok_name=None, originalinout=False):
    tt = q.ticktock("data")
    tt.tick(f"loading '{dataset}'")

    inptok = load_t5_tokenizer(inptok_name)

    if dataset.startswith("cfq/") or dataset.startswith("scan/mcd"):
        key = f"{dataset}|inptok=t5-{inptok_name}|originalinout={originalinout}"
        print(f"validfrac is ineffective with dataset '{dataset}'")
    else:
        key = f"{dataset}|validfrac={validfrac}|inptok=t5-{inptok_name}|originalinout={originalinout}"

    shelfname = os.path.basename(__file__) + ".cache.shelve"
    if not recompute:
        tt.tick(f"loading from shelf (key '{key}')")
        with shelve.open(shelfname) as shelf:
            if key not in shelf:
                recompute = True
                tt.tock("couldn't load from shelf")
            else:
                shelved = shelf[key]
                trainex, validex, testex, fldic = shelved["trainex"], shelved["validex"], shelved["testex"], shelved["fldic"]
                _inptok_name = shelved["inptok_name"]
                trainds, validds, testds = Dataset(trainex), Dataset(validex), Dataset(testex)
                tt.tock("loaded from shelf")

    if recompute:
        tt.tick("loading data")
        splits = dataset.split("/")
        dataset, splits = splits[0], splits[1:]
        split = "/".join(splits)
        if dataset == "scan":
            ds = SCANDatasetLoader().load(split, validfrac=validfrac)
        elif dataset == "cfq":
            ds = CFQDatasetLoader().load(split + "/modent")
        else:
            raise Exception(f"Unknown dataset: '{dataset}'")
        tt.tock("loaded data")

        if not originalinout:
            tt.tick("creating tokenizer")
            tokenizer = Tokenizer(inptok=inptok)
            tt.tock("created tokenizer")

            print(len(ds))

            tt.tick("dictionaries")
            inplens, outlens = [0], []
            fldic = Vocab()
            for x in ds:
                inplens.append(len(tokenizer.inptok(x[0]).input_ids))
                outtoks = tokenizer.get_out_toks(x[1])
                outlens.append(len(outtoks))
                for tok in outtoks:
                    fldic.add_token(tok, seen=x[2] == "train")
            fldic.finalize(min_freq=0, top_k=np.infty)
            tokenizer.outvocab = fldic
            print(
                f"input avg/max length is {np.mean(inplens):.1f}/{max(inplens)}, output avg/max length is {np.mean(outlens):.1f}/{max(outlens)}")
            print(f"output vocabulary size: {len(fldic.D)} at output")
            tt.tock("built dictionaries")
        else:
            tt.tick("creating tokenizer")
            tokenizer = Tokenizer(inptok=inptok, outtok=inptok)
            tt.tock("created tokenizer")

            tt.msg("using input tokenizer for output")
            inplens, outlens = [0], []
            for x in ds:
                inplens.append(len(tokenizer.inptok(x[0]).input_ids))
                outlens.append(len(tokenizer.outtok(x[1]).input_ids))
            print(
                f"input avg/max length is {np.mean(inplens):.1f}/{max(inplens)}, output avg/max length is {np.mean(outlens):.1f}/{max(outlens)}")
            fldic = None

        tt.tick("tensorizing")
        trainds = ds.filter(lambda x: x[-1] == "train").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        validds = ds.filter(lambda x: x[-1] == "valid").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        testds = ds.filter(lambda x: x[-1] == "test").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        # ds = ds.map(lambda x: tokenizer.tokenize(x[0], x[1]) + (x[2],)).cache(True)
        tt.tock("tensorized")

        tt.tick("shelving")
        with shelve.open(shelfname) as shelf:
            shelved = {
                "trainex": trainds.examples,
                "validex": validds.examples,
                "testex": testds.examples,
                "fldic": fldic,
                "inptok_name": inptok_name,
            }
            shelf[key] = shelved
        tt.tock("shelved")

    tt.tock(f"loaded '{dataset}'")
    tt.msg(f"#train={len(trainds)}, #valid={len(validds)}, #test={len(testds)}")
    return trainds, validds, testds, fldic


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
        lrmul=0.1,
        gradnorm=3,
        batsize=60,
        epochs=16,
        patience=10,
        validinter=3,
        validfrac=0.1,
        warmup=3,
        cosinelr=False,
        modelsize="small",
        ftmode="ft",  # "ft" (finetune) or "inoutonly" or "sh(allow)/de(ep)+a(dd)/r(eplace)+st(atic)/dy(namic)"
        originalinout=False,
        ptsize=5,  # length of prompt (only effective if ftmode is not "ft" or "inoutonly"
        dataset="scan/length",
        maxsize=50,
        seed=42,
        dropout=0.1,
        testcode=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        version="v1",
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

    wandb.init(project=f"compgen_prompt", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    tt = q.ticktock("script")
    tt.tick("data")
    trainds, validds, testds, fldic = \
        load_ds(dataset=dataset,
                validfrac=validfrac,
                inptok_name=modelsize,
                originalinout=originalinout,
                recompute=recomputedata)
    if trainonvalid:
        tt.msg("TRAINING ON TRAIN+VALID, VALIDATING ON TEST !!!!!!!")
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
    out_vocab_size = None
    if fldic is not None:
        out_vocab_size = fldic.number_of_ids()
    else:
        assert originalinout
    pt_type = ftmode
    if ftmode == "ft":
        if originalinout:
            pt_type = None
        else:
            pt_type = "inoutonly"
    t5tok, t5, _ = load_t5(modelsize=modelsize, use_lm100k=True, pt_type=pt_type, pt_size=ptsize, out_vocab_size=out_vocab_size)
    if fldic is None:
        batchtostrs = lambda x: t5tok.decode(x)     # TODO: test
    else:
        batchtostrs = lambda x: [fldic.tostr(x[i]) for i in range(len(x))]
    decoder = SeqDecoderT5(t5, max_size=maxsize, batch_to_strs=batchtostrs, dropout=dropout)
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

    tloss = make_array_of_metrics("loss", "acc", reduction="mean")
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
        for name, loss in zip(["loss", "acc"], tloss):
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

    if trainonvalidonly:
        tt.msg("TRAINING ON VALID ONLY, VALIDATING ON TEST !!!!!!!")
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
        lrmul=-1.,
        gradnorm=2,
        batsize=-1,
        epochs=-1,
        patience=100,
        validinter=-1,
        warmup=3,
        cosinelr=False,
        modelsize="small",
        ftmode="ft",        # "ft" (finetune) or "inoutonly" or "sh(allow)/de(ep)+a(dd)/r(eplace)+st(atic)/dy(namic)"
        originalinout=False,
        ptsize=5,           # length of prompt (only effective if ftmode is not "ft" or "inoutonly"
        dataset="default",
        maxsize=-1,
        seed=-1,
        dropout=-1.,
        testcode=False,
        trainonvalidonly=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        ):

    settings = locals().copy()

    ranges = {
        "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"],
        "dropout": [0.1, 0.25, 0.5],
        "seed": [42, 87646464, 456852],
        "epochs": [15],
        "batsize": [60],
        "lr": [0.0001],
        "lrmul": [0.1],
        "modelsize": ["small", "base", "large"],
        # "patience": [-1],
        # "warmup": [20],
        "validinter": [2],
        # "gradacc": [1],
    }

    if dataset.startswith("cfq"):
        settings["maxsize"] = 160
    elif dataset.startswith("scan"):
        if originalinout:
            settings["maxsize"] = 300
        else:
            settings["maxsize"] = 55

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