import json
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from math import ceil
from time import sleep
from typing import Dict, Callable

import wandb

import qelos as q
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import Adafactor

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree
from parseq.scripts_resplit.t5 import load_t5_tokenizer, load_vanilla_t5, load_adaptered_t5, CosineWithRestart
from parseq.vocab import Vocab


class SeqDecoderT5(torch.nn.Module):

    def __init__(self, model,
                 max_size:int=100,
                 batch_to_strs:Callable=None,
                 eos_token_id=None,
                 **kw):
        super(SeqDecoderT5, self).__init__(**kw)
        self.model = model
        self.max_size = max_size
        self.batch_to_strs = batch_to_strs
        self.eos_token_id = eos_token_id

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
        preds = self.model.generate(x, max_length=self.max_size, eos_token_id=self.eos_token_id)

        # compute loss and metrics
        gold_trees = self.tensor_to_trees(gold)
        pred_trees = self.tensor_to_trees(preds)
        treeaccs = [float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device)}
        return ret, pred_trees

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # replace padded positions with -100 --> internally, T5 code ignores index -100 in CE loss
        labels = y - 100 * (y == 0).long()
        modelout = self.model(
            input_ids=x,
            attention_mask=(x!=0).long(),
            labels=labels
        )
        loss, logits = modelout.loss, modelout.logits
        _, preds = logits.max(-1)
        same = (preds == y) & (y != 0)
        elemacc = same.float().sum(1) / (y != 0).sum(1)
        elemacc = elemacc.mean(0)
        same |= (y == 0)
        seqacc = torch.all(same, 1).float().mean(0)
        return {"loss": loss, "acc": seqacc}, logits

    # def extract_training_example(self, x, y):
    #     ymask = (y != 0).float()
    #     ylens = ymask.sum(1).long()
    #     newy = y
    #     newy = torch.cat([torch.ones_like(newy[:, 0:1]) * self.vocab["@START@"], newy], 1)
    #     newy = torch.cat([newy, torch.zeros_like(newy[:, 0:1])], 1)       # append some zeros
    #     # append EOS
    #     for i, ylen in zip(range(len(ylens)), ylens):
    #         newy[i, ylen+1] = self.vocab["@END@"]
    #
    #     goldy = newy[:, 1:]
    #     # tgt = torch.zeros(goldy.size(0), goldy.size(1), self.vocab.number_of_ids(), device=goldy.device)
    #     # tgt = tgt.scatter(2, goldy[:, :, None], 1.)
    #     # tgtmask = (goldy != 0).float()
    #
    #     newy = newy[:, :-1]
    #     return x, newy, goldy


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
            outtoks = self.get_out_toks(outs) + [self.outvocab.endtoken]
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


def load_ds(dataset="scan/random", validfrac=0.1, recompute=False, inptok_name=None):
    """
    :param dataset:
    :param validfrac:       how much of the original IID train set is used for IID validation set
    :param recompute:
    :param inptok_name:
    :return:
    """
    tt = q.ticktock("data")
    tt.tick(f"loading '{dataset}'")

    inptok = load_t5_tokenizer(inptok_name)

    iidvalidisoodvalid = False
    if dataset.startswith("cfq/") or dataset.startswith("scan/mcd"):
        iidvalidisoodvalid = True

    key = f"{dataset}|validfrac={validfrac}|inptok=t5-{inptok_name}"

    shelfname = os.path.basename(__file__) + ".cache.shelve"
    if not recompute:
        tt.tick(f"loading from shelf (key '{key}')")
        with shelve.open(shelfname) as shelf:
            if key not in shelf:
                recompute = True
                tt.tock("couldn't load from shelf")
            else:
                shelved = shelf[key]
                trainex, iidvalidex, oodvalidex, ood2validex, testex, fldic = shelved["trainex"], shelved["iidvalidex"], shelved["oodvalidex"], shelved["ood2validex"], shelved["testex"], shelved["fldic"]
                _inptok_name = shelved["inptok_name"]
                trainds, iidvalidds, oodvalidds, ood2validds, testds = \
                    Dataset(trainex), Dataset(iidvalidex), Dataset(oodvalidex), Dataset(ood2validex), Dataset(testex)
                tt.tock("loaded from shelf")

    if recompute:
        tt.tick("loading data")
        splits = dataset.split("/")
        dataset, splits = splits[0], splits[1:]
        split = "/".join(splits)
        if dataset == "scan":
            ds = SCANDatasetLoader().load(split, validfrac=validfrac)
        elif dataset == "cfq":
            ds = CFQDatasetLoader().load(split + "/modent", validfrac=validfrac)
        else:
            raise Exception(f"Unknown dataset: '{dataset}'")
        tt.tock("loaded data")

        if True:
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
        iidvalidds = ds.filter(lambda x: x[-1] == "iidvalid").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        oodvalidds = ds.filter(lambda x: x[-1] == "oodvalid").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        ood2validds = ds.filter(lambda x: x[-1] == "ood2valid").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        testds = ds.filter(lambda x: x[-1] == "test").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        # ds = ds.map(lambda x: tokenizer.tokenize(x[0], x[1]) + (x[2],)).cache(True)
        tt.tock("tensorized")

        tt.tick("shelving")
        with shelve.open(shelfname) as shelf:
            shelved = {
                "trainex": trainds.examples,
                "iidvalidex": iidvalidds.examples,
                "oodvalidex": oodvalidds.examples,
                "ood2validex": ood2validds.examples,
                "testex": testds.examples,
                "fldic": fldic,
                "inptok_name": inptok_name,
            }
            shelf[key] = shelved
        tt.tock("shelved")

    subtrainex = trainds.examples + []
    random.shuffle(subtrainex)
    subtrainex = subtrainex[:len(subtrainex)//10]
    subtrainds = Dataset(subtrainex)

    tt.tock(f"loaded '{dataset}'")
    tt.msg(f"#train={len(trainds)}, #iidvalid={len(iidvalidds)}, #oodvalid={len(oodvalidds)}, #ood2valid={len(ood2validds)}, #test={len(testds)}")
    return trainds, subtrainds, iidvalidds, oodvalidds, ood2validds, testds, fldic


def collate_fn(x, pad_value=0, numtokens=5000):
    lens = [len(xe[1]) for xe in x]
    a = list(zip(lens, x))
    a = sorted(a, key=lambda xe: xe[0], reverse=True)
    maxnum = int(numtokens/max(lens))
    b = a[:maxnum]
    b = [be[1] for be in b]
    ret = autocollate(b, pad_value=pad_value)
    return ret


class CustomValidator:
    def __init__(self, *validepochfns, numbats=None, validinter=None, tt=None):
        super().__init__()
        self.tt = tt
        self.validepochfns = validepochfns      # functions that will be triggered
        self.numbats = numbats
        self.validinter = validinter
        self.batchmod = int(ceil(self.numbats * self.validinter))
        self.batchcount = 0
        self.epochcount = 0

    def run_fns(self):
        fnreses = []
        for fn in self.validepochfns:
            fnres = fn()
            fnreses.append(fnres)
        if self.tt is not None:
            self.tt.msg("validator: " + " - ".join([str(x) for x in fnreses]))

    def on_batch_end(self):
        if self.validinter < 1:
            self.batchcount += 1
            if self.batchcount % self.batchmod == 0 or self.batchcount % self.numbats == 0:
                self.run_fns()

            # reset count
            if self.batchcount % self.numbats == 0:
                self.batchcount = 0

    def on_epoch_end(self):
        self.batchcount = 0
        if self.validinter >= 1:
            if self.epochcount % self.validinter == 0:
                self.run_fns()
        self.epochcount += 1


def run(lr=0.0001,
        useadafactor=False,
        lrmul=0.1,
        gradnorm=3,
        batsize=60,
        testbatsize=-1,
        epochs=16,
        validinter=3.,
        validfrac=0.1,
        warmup=0.1,
        cosinecycles=0,
        modelsize="small",
        ftmode="ft",                # "ft" or "adapter"
        adapterdim=64,              # only active in adapter mode
        usedefaultmodel=False,
        dataset="scan/length",
        maxsize=-1,
        seed=42,
        dropout=0.,
        testcode=False,
        gpu=-1,
        trainonvalidonly=False,
        trainonood2only=False,
        recomputedata=False,
        version="v4",
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

    run = wandb.init(project=f"compgen_resplit", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    if testbatsize == -1:
        testbatsize = batsize

    tt = q.ticktock("script")
    tt.tick("data")
    trainds, subtrainds, iidvalidds, oodvalidds, ood2validds, testds, fldic = \
        load_ds(dataset=dataset,
                validfrac=validfrac,
                inptok_name=modelsize,
                recompute=recomputedata)

    if maxsize < 0:   # has not been manually overridden
        if dataset.startswith("cfq"):
            maxsize = 120
        elif dataset.startswith("scan"):
            maxsize = 55

    print(f"Maxsize: {maxsize}")

    tt.tick("dataloaders")
    NUMWORKERS = 0

    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate, num_workers=NUMWORKERS)
    subtraindl = DataLoader(subtrainds, batch_size=batsize, shuffle=True, collate_fn=autocollate, num_workers=NUMWORKERS)
    iidvaliddl = DataLoader(iidvalidds, batch_size=batsize if not trainonvalidonly else testbatsize,
                            shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)
    oodvaliddl = DataLoader(oodvalidds, batch_size=testbatsize, shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)
    ood2validdl = DataLoader(ood2validds, batch_size=batsize if trainonood2only else testbatsize,
                             shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)
    testdl = DataLoader(testds, batch_size=testbatsize, shuffle=False, collate_fn=autocollate, num_workers=NUMWORKERS)
    # print(json.dumps(next(iter(trainds)), indent=3))
    # print(next(iter(traindl)))
    # print(next(iter(validdl)))
    tt.tock()
    tt.tock()

    tt.tick("model")
    out_vocab_size = None
    assert fldic is not None
    out_vocab_size = fldic.number_of_ids()

    if ftmode == "ft":
        t5tok, t5, _ = load_vanilla_t5(modelsize=modelsize, use_lm100k=not usedefaultmodel, out_vocab_size=out_vocab_size)
    elif ftmode.startswith("ada"):
        t5tok, t5, _ = load_adaptered_t5(modelsize=modelsize, use_lm100k=not usedefaultmodel, adapterdim=adapterdim,
                                       out_vocab_size=out_vocab_size)

    t5.set_dropout(dropout)

    if fldic is None:
        batchtostrs = lambda x: t5tok.batch_decode(x)     # TODO: test
    else:
        batchtostrs = lambda x: [fldic.tostr(x[i]) for i in range(len(x))]
    decoder = SeqDecoderT5(t5, max_size=maxsize, batch_to_strs=batchtostrs, eos_token_id=fldic[fldic.endtoken])
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
    subtmetrics = make_array_of_metrics("treeacc", reduction="mean")
    iidvmetrics = make_array_of_metrics("treeacc", reduction="mean")
    oodvmetrics = make_array_of_metrics("treeacc", reduction="mean")
    ood2vmetrics = make_array_of_metrics("treeacc", reduction="mean")
    xmetrics = make_array_of_metrics("treeacc", reduction="mean")

    # region parameters
    def get_parameters(m:SeqDecoderT5, _lr, _lrmul, _ftmode):
        primary_params = []
        secondary_params = []
        if _ftmode == "ft":     # fine-tune all params using _lr*_lrmul and if _originalinout is True, then also finetune decoder inputs and outputs
            secondary_params += list(m.parameters())
            primary_params += list(m.model.lm_head.parameters()) + list(m.model.decoder.embed_tokens.parameters())
            # remove primary params from secondary
            i = 0
            while i < len(secondary_params):
                for primaryparam in primary_params:
                    if secondary_params[i] is primaryparam:
                        del secondary_params[i]
                        i -= 1
                        break
                i += 1
        else:
            primary_params = m.model.get_tunable_params_and_set_requires_grad()
        paramgroups = [{"params": primary_params, "lr": _lr}]
        if len(secondary_params) > 0:
            paramgroups.append({"params": secondary_params, "lr": _lr * _lrmul})
        return paramgroups
    # endregion

    def get_optim(_m, _lr, _lrmul, _ftmode, _wreg=0):
        paramgroups = get_parameters(_m, _lr=lr, _lrmul=_lrmul, _ftmode=_ftmode)
        if useadafactor:
            tt.msg("Using AdaFactor.")
            optim = Adafactor(paramgroups, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
        else:
            tt.msg("Using Adam.")
            optim = torch.optim.Adam(paramgroups, lr=lr, weight_decay=_wreg)
        return optim

    if useadafactor:
        gradnorm = 1e6

    def clipgradnorm(_m=None, _norm=None):
        torch.nn.utils.clip_grad_norm_(_m.parameters(), _norm)

    patience = epochs
    iideyt = q.EarlyStopper(iidvmetrics[0], patience=patience, min_epochs=5, more_is_better=True,
                         remember_f=lambda: deepcopy(decoder.model))
    oodeyt = q.EarlyStopper(oodvmetrics[0], patience=patience, min_epochs=5, more_is_better=True,
                         remember_f=lambda: deepcopy(decoder.model))
    ood2eyt = q.EarlyStopper(ood2vmetrics[0], patience=patience, min_epochs=5, more_is_better=True,
                            remember_f=lambda: deepcopy(decoder.model))

    def wandb_logger():
        d = {}
        for name, loss in zip(["loss", "acc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], iidvmetrics):
            d["iidvalid_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], oodvmetrics):
            d["oodvalid_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], ood2vmetrics):
            d["ood2valid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs * len(traindl)
    optim = get_optim(decoder, lr, lrmul, ftmode)
    warmupsteps = int(round(warmup * t_max))
    print(f"Total number of updates: {t_max} . Warmup: {warmupsteps}")

    if cosinecycles == 0:       # constant lr
        tt.msg("Using constant LR with warmup")
        lr_schedule = q.sched.Linear(0, 1, steps=warmupsteps) >> 1.
    else:
        tt.msg("Using cosine LR with restart and with warmup")
        lr_schedule = q.sched.Linear(0, 1, steps=warmupsteps) >> (CosineWithRestart(high=1., low=0.1, cycles=cosinecycles, steps=t_max-warmupsteps)) >> 0.1

    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    iidvalidepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=iidvmetrics,
                         dataloader=iidvaliddl,
                         device=device,
                         on_end=[lambda: iideyt.on_epoch_end()])

    oodvalidepoch = partial(q.test_epoch,
                            model=decoder,
                            losses=oodvmetrics,
                            dataloader=oodvaliddl,
                            device=device,
                            on_end=[lambda: oodeyt.on_epoch_end()])

    ood2validepoch = partial(q.test_epoch,
                            model=decoder,
                            losses=ood2vmetrics,
                            dataloader=ood2validdl,
                            device=device,
                            on_end=[lambda: ood2eyt.on_epoch_end()])

    validator = CustomValidator(iidvalidepoch, oodvalidepoch, ood2validepoch, lambda: wandb_logger(), numbats=len(traindl), validinter=validinter, tt=tt)

    trainbatch = partial(q.train_batch,
                         on_before_optim_step=[
                             lambda : clipgradnorm(_m=decoder, _norm=gradnorm),
                             lambda : lr_schedule.step()
                         ],
                         on_end=[lambda : validator.on_batch_end()]
                         )

    if trainonvalidonly:
        # assert False
        tt.msg("TRAINING ON IID VALID ONLY !!!!!!!")
        traindl = iidvaliddl

    if trainonood2only:
        tt.msg("TRAINING ON OOD2 VALID ONLY !!!!!!!!")
        traindl = ood2validdl

    trainepoch = partial(q.train_epoch, model=decoder,
                         dataloader=traindl,
                         optim=optim,
                         losses=tloss,
                         device=device,
                         _train_batch=trainbatch,
                         on_end=[lambda: validator.on_epoch_end()])

    trainevalepoch = partial(q.test_epoch,
                             model=decoder,
                             losses=subtmetrics,
                             dataloader=subtraindl,
                             device=device)

    tt.tick("training")
    validfs = [iidvalidepoch, oodvalidepoch, ood2validepoch]
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=None,
                   max_epochs=epochs,
                   check_stop=[lambda: iideyt.check_stop() and oodeyt.check_stop() and ood2eyt.check_stop()]
                   )
    tt.tock("done training")

    settings.update({"train_loss_at_end": tloss[0].get_epoch_error()})

    tt.tick("running test before reloading")
    testepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=xmetrics,
                         dataloader=testdl,
                         device=device)

    testres = testepoch()
    settings.update({"test_tree_acc_at_end": xmetrics[0].get_epoch_error()})
    tt.tock(f"Test tree acc: {testres}")

    if iideyt.remembered is not None:
        tt.msg("reloading model with best IID validation accuracy")
        decoder.model = iideyt.remembered
        model = iideyt.remembered

        tt.tick("running evaluation on subset of train")
        trainres = trainevalepoch()
        settings.update({"train_tree_acc_at_iidearly": tmetrics[0].get_epoch_error()})
        tt.tock(f"Train tree acc: {trainres}")

        tt.tick("rerunning validation")
        iidvalidres = iidvalidepoch()
        settings.update({"iidvalid_tree_acc_at_iidearly": iidvmetrics[0].get_epoch_error()})
        tt.tock(f"IID validation results: {iidvalidres}")

        tt.tick("running test")
        testres = testepoch()
        settings.update({"test_tree_acc_at_iidearly": xmetrics[0].get_epoch_error()})
        tt.tock(f"Test tree acc: {testres}")

    if oodeyt.remembered is not None:
        tt.msg("reloading model with best OOD validation accuracy")
        decoder.model = oodeyt.remembered
        model = oodeyt.remembered

        tt.tick("running evaluation on subset of train")
        trainres = trainevalepoch()
        settings.update({"train_tree_acc_at_oodearly": tmetrics[0].get_epoch_error()})
        tt.tock(f"Train tree acc: {trainres}")

        tt.tick("rerunning validation")
        oodvalidres = oodvalidepoch()
        settings.update({"oodvalid_tree_acc_at_oodearly": oodvmetrics[0].get_epoch_error()})
        tt.tock(f"OOD validation results: {oodvalidres}")

        tt.tick("running test")
        testres = testepoch()
        settings.update({"test_tree_acc_at_oodearly": xmetrics[0].get_epoch_error()})
        tt.tock(f"Test tree acc: {testres}")

    if ood2eyt.remembered is not None:
        tt.msg("reloading model with best OOD2 validation accuracy")
        decoder.model = ood2eyt.remembered
        model = ood2eyt.remembered

        tt.tick("running evaluation on subset of train")
        trainres = trainevalepoch()
        settings.update({"train_tree_acc_at_ood2early": tmetrics[0].get_epoch_error()})
        tt.tock(f"Train tree acc: {trainres}")

        tt.tick("rerunning validation")
        oodvalidres = ood2validepoch()
        settings.update({"ood2valid_tree_acc_at_ood2early": ood2vmetrics[0].get_epoch_error()})
        tt.tock(f"OOD2 validation results: {oodvalidres}")

        tt.tick("running test")
        testres = testepoch()
        settings.update({"test_tree_acc_at_ood2early": xmetrics[0].get_epoch_error()})
        tt.tock(f"Test tree acc: {testres}")

    wandb.config.update(settings)
    q.pp_dict(settings)
    run.finish()
    # sleep(15)


def run_experiment(
        lr=-1.,
        lrmul=-1.,
        useadafactor=False,
        gradnorm=2,
        batsize=-1,
        testbatsize=-1,
        epochs=-1,
        validinter=-1.,
        validfrac=0.1,
        warmup=0.1,
        cosinecycles=0,
        modelsize="small",
        ftmode="ft",        # "ft" (finetune) or "adapter"
        adapterdim=-1,
        usedefaultmodel=False,
        dataset="default",
        maxsize=-1,
        seed=-1,
        dropout=-1.,
        testcode=False,
        trainonvalidonly=False,
        trainonood2only=False,
        gpu=-1,
        recomputedata=False,
        ):

    settings = locals().copy()

    ranges = {
        "dataset": ["cfq/mcd1new", "cfq/mcd2new", "cfq/mcd3new"],
        "dropout": [0.1],
        "seed": [42, 87646464, 456852],
        "epochs": [50],
        "batsize": [60],
        "lr": [0.0005],
        "lrmul": [1.],
        "modelsize": ["small", "base", "large"],
        # "patience": [-1],
        # "warmup": [20],
        "validinter": [2],
        # "gradacc": [1],
        "adapterdim": [64],
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

    if dataset.endswith("/mcd"):
        ranges["dataset"] = [dataset + f"{i}" for i in range(1, 4)]

    def checkconfig(spec):
        return True

    q.run_experiments_random(
        run, ranges, path_prefix=None, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)