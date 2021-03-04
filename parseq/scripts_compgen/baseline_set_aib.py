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


class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, useabspos=True,
                 relposmode="basic", relposrng=10,
                 dropout:float=0., maxpos=512, weightmode="vanilla", **kw):
        super(TransformerEncoder, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.userelpos = userelpos
        self.relposrng = relposrng
        self.useabspos = useabspos

        self.weightmode = weightmode
        if self.weightmode.startswith("none") or self.weightmode == "vanilla":
            self.encrelposemb = None
            if self.userelpos is True:
                if relposmode == "basic":
                    self.encrelposemb = BasicRelPosEmb(self.dim, relposrng)
                # elif relposmode == "mod":
                #     self.relposemb = ModRelPosEmb(self.dim, relposrng, levels=4)
                else:
                    raise Exception(f"Unrecognized relposmode '{relposmode}'")
            bname = "bert" + self.weightmode[4:]
            if self.weightmode == "vanilla":
                inpvocabsize = self.vocabsize
            else:
                tokenizer = AutoTokenizer.from_pretrained(bname)
                inpvocabsize = tokenizer.vocab_size
            config = TransformerConfig(vocab_size=inpvocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                          d_kv=int(self.dim / numheads),
                                          num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)
            encemb = TransformerEmbeddings(config.vocab_size, config.d_model, dropout=dropout,
                                           max_position_embeddings=maxpos, useabspos=useabspos)
            self.encoder_model = TransformerStack(config, encemb, rel_emb=self.encrelposemb)
        else:
            self.encoder_model = BertModel.from_pretrained(self.weightmode,
                                                           hidden_dropout_prob=min(dropout, 0.2),
                                                           attention_probs_dropout_prob=min(dropout, 0.1))
        self.adapter = None
        if self.encoder_model.config.hidden_size != self.dim:
            self.adapter = torch.nn.Linear(self.encoder_model.config.hidden_size, self.dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x=None, xemb=None, xmask=None):
        return self.encode(x=x, xemb=xemb, xmask=xmask)

    def encode(self, x:torch.Tensor=None, xemb:torch.Tensor=None, xmask:torch.Tensor=None):
        """
        :param x:       (batsize, seqlen) integer ids
        :param xemb:    (batsize, seqlen, dim) floats -- embeddings for x ==> normal embeddings are not used
        :return:
        """
        assert (x is None or xemb is None)
        assert not (x is None and xemb is None)
        encmask = (x != 0) if xmask is None and x is not None else xmask
        relpos = None
        if self.encrelposemb is not None:      # compute relative positions
            positions = torch.arange(x.size(1), device=x.device)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        if relpos is not None:
            encs = self.encoder_model(x, inputs_embeds=xemb, attention_mask=encmask, relpos=relpos)[0]
        else:
            encs = self.encoder_model(x, inputs_embeds=xemb, attention_mask=encmask)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask


class SetDecoder(torch.nn.Module):
    """ Predicts the set of output tokens.
    Implemented as a tagging model that labels every input position with one of the output tokens.
    The probability that a certain output token is in the set is computed
    by taking the max logit over sequence dimension and taking Sigmoid of that.
    """
    def __init__(self, dim, vocab:Vocab=None, encoder:TransformerEncoder=None, **kw):
        super(SetDecoder, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = self.vocab.number_of_ids()
        self.dim = dim

        self.encoder_model = encoder

        self.out = torch.nn.Linear(self.dim, self.vocabsize)
        self.reset_parameters()

        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def reset_parameters(self):
        pass

    def compute_loss(self, logits, golds):
        bce = self.bce(logits, golds)       # (batsize, vocsize)
        bce = bce.sum(-1)

        preds = (torch.sigmoid(logits) > 0.5)        # (batsize, vocsize)
        acc = torch.all(preds == golds, 1)

        return bce, acc.float()

    def forward(self, x:torch.Tensor, y:torch.Tensor, retenc=False):      # integer tensors
        enc, encmask = self.encoder_model(x)

        seqlogits = self.out(enc)      # (batsize, seqlen, vocsize)
        seqlogits = seqlogits + -9999 * (1 - encmask.float())[:, :, None]    # --> sets masked positions to -infty
        logits, _ = seqlogits.max(1)    # (batsize, vocsize) -- max logits for every possible output token
        golds = torch.zeros_like(logits)
        golds[:, 0] = 1
        golds.scatter_(-1, y, 1)

        loss, acc = self.compute_loss(logits, golds)

        if retenc is False:
            return {"loss": loss, "acc": acc}, logits
        elif retenc is True:
            return {"loss": loss, "acc": acc}, logits, enc, encmask


class AdvTagger(torch.nn.Module):
    def __init__(self, advenc:TransformerEncoder=None, maskfrac=0.2, vocab:Vocab=None, **kw):
        super(AdvTagger, self).__init__(**kw)
        self.advenc = advenc
        self.maskfrac = maskfrac
        self.dim = self.advenc.dim
        self.mask_emb = torch.nn.Embedding(1, self.dim)
        self.vocab = vocab
        self.advout = torch.nn.Linear(self.dim, self.vocab.number_of_ids())

    def forward(self, enc, encmask):
        # compute inputs to advenc
        mask = torch.rand_like(encmask, dtype=torch.float)
        mask = mask > self.maskfrac  # 1 if the position in x is masked and should be predicted, zero otherwise
        advinp = self.mask_emb.weight[0][None, None, :] * ((encmask & ~mask)[:, :, None].float())
        advinp = advinp + (mask & encmask)[:, :, None].float() * enc

        # run advenc
        advenc, advencmask = self.advenc(xemb=advinp, xmask=encmask)  # (batsize, seqlen, dim)

        # compute adversary's output probabilities and loss
        advout = self.advout(advenc)  # (batsize, seqlen, vocsize)
        return advout, mask


class AdvTrainmodelMain(torch.nn.Module):
    def __init__(self, this, **kw):
        super(AdvTrainmodelMain, self).__init__(**kw)
        self.this = this

    def forward(self, *args, **kw):
        return self.this(*args, **kw, mode="main")


class AdvTrainmodelAdv(torch.nn.Module):
    def __init__(self, this, **kw):
        super(AdvTrainmodelAdv, self).__init__(**kw)
        self.this = this

    def forward(self, *args, **kw):
        return self.this(*args, **kw, mode="adv")


class AdvModel(torch.nn.Module):
    def __init__(self, setdecoder:SetDecoder=None, adv:AdvTagger=None, numsamples:int=1, advcontrib=1., **kw):
        super(AdvModel, self).__init__(**kw)
        self.core = setdecoder
        self.adv = adv
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.numsamples = numsamples
        self.advcontrib = advcontrib

    @property
    def main_trainmodel(self):
        return AdvTrainmodelMain(self)

    @property
    def adv_trainmodel(self):
        return AdvTrainmodelAdv(self)

    def compute_adversary(self, x:torch.Tensor, enc:torch.Tensor, encmask:torch.Tensor):
        advout, mask = self.adv(enc, encmask)

        advces = self.ce(advout.permute(0, 2, 1), x)  # (batsize, seqlen)
        advces = advces * (~mask & encmask)
        advce = advces.sum(-1)  # (batsize,)

        # compute entropy of output distribution
        adventropies = torch.softmax(advout, -1) * torch.log_softmax(advout, -1).clamp_min(-10e6)
        bestentropy = -np.log(1/advout.size(-1))
        adventropies = -adventropies.sum(-1)
        adventropy = (bestentropy - adventropies)
        adventropy = adventropy * (~mask & encmask)
        adventropy = adventropy.sum(-1)
        return advce, adventropy, advout

    def forward_pred(self, x:torch.Tensor, y:torch.Tensor):
        # run through encoder and get predictions from decoder
        retdic, logits = self.core(x, y)
        return retdic, logits

    def forward_main(self, x:torch.Tensor, y:torch.Tensor):
        # run through encoder and get predictions from decoder
        retdic, logits, enc, encmask = self.core(x, y, retenc=True)

        adve_acc = 0
        for _ in range(self.numsamples):
            _, adve, advout = self.compute_adversary(x, enc, encmask)
            adve_acc = adve_acc + adve
        adve = adve_acc / self.numsamples
        adve = adve * self.advcontrib

        retdic["mainloss"] = retdic["loss"]
        retdic["advloss"] = adve
        retdic["loss"] = retdic["loss"] + adve
        return retdic, logits

    def forward_adv(self, x:torch.Tensor, y:torch.Tensor):
        # run through encoder and get predictions from decoder
        with torch.no_grad():
            retdic, logits, enc, encmask = self.core(x, y, retenc=True)
        enc = enc.detach()

        advce, _, advout = self.compute_adversary(x, enc, encmask)

        retdic["mainloss"] = retdic["loss"]
        retdic["advloss"] = advce
        retdic["loss"] = advce
        return retdic, logits

    def forward(self, x, y, mode="pred"):
        if mode == "main":
            ret = self.forward_main(x, y)
        elif mode == "adv":
            ret = self.forward_adv(x, y)
        elif mode == "pred":
            ret = self.forward_pred(x, y)
        return ret


def adv_train_epoch(main_model=None, adv_model=None,
                    main_dataloader=None, adv_dataloader=None,
                    main_optim=None, adv_optim=None,
                    main_losses=None, adv_losses=None,
                    adviters=1,
                    device=torch.device("cpu"), tt=q.ticktock(" -"),
                    current_epoch=0, max_epochs=0,
                    _main_train_batch=q.train_batch, _adv_train_batch=q.train_batch,
                    on_start=tuple(), on_end=tuple(),
                    print_every_batch=False):
    """
    Performs an epoch of training on given model, with data from given dataloader, using given optimizer,
    with loss computed based on given losses.
    :param model:
    :param dataloader:
    :param optim:
    :param losses:  list of loss wrappers
    :param device:  device to put batches on
    :param tt:
    :param current_epoch:
    :param max_epochs:
    :param _train_batch:    train batch function, default is train_batch
    :param on_start:
    :param on_end:
    :return:
    """
    for loss in main_losses + adv_losses:
        loss.push_epoch_to_history(epoch=current_epoch-1)
        loss.reset_agg()
        loss.to(device)

    main_model.to(device)
    adv_model.to(device)

    [e() for e in on_start]

    q.epoch_reset(main_model)
    q.epoch_reset(adv_model)
    adv_optim.zero_grad()
    main_optim.zero_grad()


    adv_dl_iter = iter(adv_dataloader)
    k = 0
    for i, main_batch in enumerate(main_dataloader):

        # do 'adviters' of adversarial updates
        j = adviters
        while j > 0:
            try:
                adv_batch = next(adv_dl_iter)
                adv_optim.zero_grad()
                ttmsg = _adv_train_batch(batch=adv_batch, model=adv_model, optim=adv_optim, losses=adv_losses, device=device,
                                         batch_number=k, max_batches=len(adv_dataloader), current_epoch=current_epoch,
                                         max_epochs=max_epochs)
                ttmsg = "adv: " + ttmsg
                if print_every_batch:
                    tt.msg(ttmsg)
                else:
                    tt.live(ttmsg)
                j -= 1
                k += 1
            except StopIteration as e:
                adv_dl_iter = iter(adv_dataloader)
                k = 0

        # do main update
        main_optim.zero_grad()
        ttmsg = _main_train_batch(batch=main_batch, model=main_model, optim=main_optim, losses=main_losses, device=device,
                                 batch_number=i, max_batches=len(main_dataloader), current_epoch=current_epoch,
                                 max_epochs=max_epochs)
        ttmsg = "main: " + ttmsg
        if print_every_batch:
            tt.msg(ttmsg)
        else:
            tt.live(ttmsg)
        j -= 1

    tt.stoplive()
    [e() for e in on_end]
    ttmsg = "main: " + q.pp_epoch_losses(*main_losses) + " -- adv: " + q.pp_epoch_losses(*adv_losses)
    return ttmsg


def run(lr=0.0001,
        enclrmul=0.1,
        smoothing=0.1,
        gradnorm=3,
        batsize=60,
        epochs=16,
        patience=-1,
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
        bertname="bert-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        adviters=3,             # adversary updates per main update
        advreset=10,            # reset adversary after this number of epochs
        advcontrib=1.,
        advmaskfrac=0.2,
        advnumsamples=3,
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    wandb.init(project=f"compgen_set_aib", config=settings, reinit=True)
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
    traindl_main = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    traindl_adv = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    validdl = DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    testdl = DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    if trainonvalidonly:
        traindl_main = DataLoader(validds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
        traindl_adv = DataLoader(validds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
        validdl = testdl
    # print(json.dumps(next(iter(trainds)), indent=3))
    # print(next(iter(traindl)))
    # print(next(iter(validdl)))
    tt.tock()
    tt.tock()

    tt.tick("model")
    encoder = TransformerEncoder(hdim, vocab=inpdic, numlayers=numlayers, numheads=numheads,
            dropout=dropout, weightmode=bertname, userelpos=userelpos, useabspos=not userelpos)
    advencoder = TransformerEncoder(hdim, vocab=inpdic, numlayers=numlayers, numheads=numheads,
            dropout=dropout, weightmode="vanilla", userelpos=userelpos, useabspos=not userelpos)
    setdecoder = SetDecoder(hdim, vocab=fldic, encoder=encoder)
    adv = AdvTagger(advencoder, maskfrac=advmaskfrac, vocab=inpdic)
    model = AdvModel(setdecoder, adv, numsamples=advnumsamples, advcontrib=advcontrib)
    tt.tock()

    if testcode:
        tt.tick("testcode")
        batch = next(iter(traindl_main))
        # out = tagger(batch[1])
        tt.tick("train")
        out = model(*batch)
        tt.tock()
        model.train(False)
        tt.tick("test")
        out = model(*batch)
        tt.tock()
        tt.tock("testcode")

    tloss_main = make_array_of_metrics("loss", "mainloss", "advloss", "acc", reduction="mean")
    tloss_adv = make_array_of_metrics("loss", reduction="mean")
    tmetrics = make_array_of_metrics("loss", "acc", reduction="mean")
    vmetrics = make_array_of_metrics("loss", "acc", reduction="mean")
    xmetrics = make_array_of_metrics("loss", "acc", reduction="mean")

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
        for name, loss in zip(["loss", "mainloss", "advloss", "acc"], tloss_main):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["advloss"], tloss_adv):
            d["train_adv_"+name] = loss.get_epoch_error()
        for name, loss in zip(["acc"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["acc"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim_main = get_optim(model.core, lr, enclrmul)
    optim_adv = get_optim(model.adv, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        assert t_max > (warmup + 10)
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(low=0., high=1.0, steps=t_max-warmup) >> (0. * lr)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule_main = q.sched.LRSchedule(optim_main, lr_schedule)
    lr_schedule_adv = q.sched.LRSchedule(optim_adv, lr_schedule)

    trainbatch_main = partial(q.train_batch, on_before_optim_step=[lambda : clipgradnorm(_m=model.core, _norm=gradnorm)])
    trainbatch_adv = partial(q.train_batch, on_before_optim_step=[lambda: clipgradnorm(_m=model.adv, _norm=gradnorm)])

    print("using test data for validation")
    validdl = testdl

    trainepoch = partial(adv_train_epoch,
                         main_model=model.main_trainmodel, adv_model=model.adv_trainmodel,
                         main_dataloader=traindl_main, adv_dataloader=traindl_adv,
                         main_optim=optim_main, adv_optim=optim_adv,
                         main_losses=tloss_main, adv_losses=tloss_adv,
                         adviters=adviters,
                         device=device,
                         print_every_batch=True,
                         _main_train_batch=trainbatch_main, _adv_train_batch=trainbatch_adv,
                         on_end=[lambda: lr_schedule_main.step(), lambda: lr_schedule_adv.step()])

    # eval epochs
    trainevalepoch = partial(q.test_epoch,
                             model=model,
                             losses=tmetrics,
                             dataloader=traindl_main,
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
        assert False
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
    settings.update({"final_train_acc": tmetrics[1].get_epoch_error()})
    settings.update({"final_valid_acc": vmetrics[1].get_epoch_error()})
    settings.update({"final_test_acc": xmetrics[1].get_epoch_error()})

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
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        trainonvalidonly=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        adviters=10,
        advreset=10,
        advcontrib=1.,
        advmaskfrac=0.2,
        advnumsamples=3,
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