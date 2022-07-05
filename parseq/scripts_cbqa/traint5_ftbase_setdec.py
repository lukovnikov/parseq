import fire
from pathlib import Path

import itertools

import json
import random
import re
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

import wandb

import qelos as q
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import Adafactor, T5TokenizerFast, T5Model

from parseq.datasets import autocollate

from parseq.eval import make_array_of_metrics
from parseq.scripts_cbqa.adapter_t5 import AdaptedT5WordEmbeddings, AdaptedT5LMHead
from parseq.scripts_cbqa.metaqa_dataset import MetaQADatasetLoader
from parseq.scripts_resplit.t5 import CosineWithRestart

# uses decoder to generate answer string

def collate_fn(data):
    newdata = []
    for datapoint in data:
        inps, outs = datapoint
        for inp, out in zip(inps, outs):
            newdata.append((inp, out))
    ret = autocollate(newdata)
    return ret


def decode(x, tok):
    return tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=True).replace("<pad>", "").replace("</s>", "")


class Model(torch.nn.Module):
    randomizedeval = True
    maxnumentsperexample = 1
    maxlen=100

    def __init__(self, model:T5Model, dim, maxsetsize=1000, tok=None):
        super(Model, self).__init__()
        self.model = model
        self.tok = tok

    def copy_from(self, m):
        self.model = m.model

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

        inpstring = [decode(xe, self.tok) for xe in x.unbind(0)]
        predstring = [decode(g, self.tok) for g in preds.unbind(0)]
        targetstring = [decode(t, self.tok) for t in y.unbind(0)]

        stracc = [float(a == b) for a, b in zip(predstring, targetstring)]
        stracc = torch.tensor(stracc, device=x.device)

        ret = list(zip(inpstring, predstring, targetstring))
        return {"accuracy": stracc}, ret

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        else:
            return self.test_forward(*args, **kwargs)


class Main():
    DATAMODE = "seqset"
    TESTMETRIC = "accuracy"
    VARIANT = "setdec"
    MAXLEN = 30

    def __init__(self):
        super(Main, self).__init__()

    def get_model(self):
        return Model

    def load_ds(self, dataset="metaqa/1", tokname="t5-small", recompute=False, subset=None):
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

        extra_tokens = ["[SEP1]", "[SEP2]", "[ANS]", "[ENT]", "[REL]", "[SEPITEM]", "[BOS]", "[ENDOFSET]", "[LASTITEM]"] # + [f"extra_id_{i}" for i in range(0)]
        extra_tokens = extra_tokens + [f"[ITEM-{i}]" for i in range(1000)] + [f"[TOTAL-{i}]" for i in range(1000)]
        tok = T5TokenizerFast.from_pretrained(tokname, additional_special_tokens=extra_tokens, extra_ids=0)

        tt.tick("loading data")
        kbds = MetaQADatasetLoader().load_kb(tok, recompute=recompute, subset=subset, mode=self.DATAMODE)
        qads = MetaQADatasetLoader().load_qa(whichhops, kbds[0].baseds, tok, recompute=recompute, subset=subset, mode=self.DATAMODE)
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

    def run(self,
            lr=0.0001,
            kbpretrainepochs=0,
            kbpretrainvalidinter=-1,
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
            seed=42,
            dropout=0.,
            testcode=False,
            debugcode=False,
            gpu=-1,
            recomputedata=False,
            version="v0",
            nosave=False,
            loadfrom="",
            usesavedconfig=False,
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
        del settings["self"]

        if usesavedconfig and len(loadfrom) > 0:
            print("loading into locals() is not supported by Python so calling run() again")
            with open(f"runs/{self.VARIANT}/{loadfrom}/run.config", "r") as f:
                loadedconfig = json.load(f)
                settings.update(loadedconfig)
                self.run(**settings)
                return

        q.pp_dict(settings, indent=3)

        wandbrun = wandb.init(project=f"t5-cbqa-ftbase-{self.VARIANT}", config=settings, reinit=True)
        configsavepath = Path(f"runs/{self.VARIANT}/{wandbrun.name}/run.config")
        modelsavepath = Path(f"runs/{self.VARIANT}/{wandbrun.name}/model.ckpt")
        if not nosave:
            configsavepath.parent.mkdir(parents=True, exist_ok=True)
            with configsavepath.open("w") as f:
                json.dump(settings, f, indent=4)

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

        if testbatsize == -1:
            testbatsize = batsize
        if kbpretrainvalidinter == -1:
            kbpretrainvalidinter = validinter

        tt = q.ticktock("script")
        tt.tick("data")
        tok, trainds, evaltrainds, validds, testds, kbtrainds, kbevaltrainds = \
            self.load_ds(dataset=dataset, tokname=f"google/t5-v1_1-{modelsize}",
                    recompute=recomputedata,
                    subset=5000 if debugcode else None)

        tt.tick("dataloaders")
        NUMWORKERS = 0

        kbtraindl = DataLoader(kbtrainds, batch_size=batsize, shuffle=True, collate_fn=collate_fn, num_workers=NUMWORKERS)
        kbevaltraindl = DataLoader(kbevaltrainds, batch_size=testbatsize, shuffle=False, collate_fn=collate_fn, num_workers=NUMWORKERS)

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
        m = self.get_model()(t5model, t5model.config.d_model, tok=tok)

        if len(loadfrom) > 0:
            tt.tick(f"Loading model from runs/{self.VARIANT}/{loadfrom}/model.ckpt")
            m.load_state_dict(torch.load(f"runs/{self.VARIANT}/{loadfrom}/model.ckpt"))
            tt.tock()

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
        tmetrics = make_array_of_metrics(self.TESTMETRIC, reduction="mean")
        vmetrics = make_array_of_metrics(self.TESTMETRIC, reduction="mean")
        xmetrics = make_array_of_metrics(self.TESTMETRIC, reduction="mean")

        kbtloss = make_array_of_metrics("loss", "accuracy", "elemacc", reduction="mean")
        kbtmetrics = make_array_of_metrics(self.TESTMETRIC, reduction="mean")

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
            for name, loss in zip([self.TESTMETRIC], tmetrics):
                d["evaltrain_"+name] = loss.get_epoch_error()
            for name, loss in zip([self.TESTMETRIC], vmetrics):
                d["valid_"+name] = loss.get_epoch_error()
            for name, loss in zip([self.TESTMETRIC], xmetrics):
                d["test_"+name] = loss.get_epoch_error()
            wandb.log(d)

        def wandb_logger_kbft():
            d = {}
            for name, loss in zip(["loss", "accuracy", "elemacc"], kbtloss):
                d["kbtrain_"+name] = loss.get_epoch_error()
            for name, loss in zip([self.TESTMETRIC], kbtmetrics):
                d["kbevaltrain_"+name] = loss.get_epoch_error()
            wandb.log(d)

        if not nosave:
            tt.msg(f"Saving model at {modelsavepath} at every epoch")
        def modelsaver():
            if not nosave:
                modelsavepath.parent.mkdir(parents=True, exist_ok=True)
                torch.save(m.state_dict(), modelsavepath)

        # do training on KB data
        if kbpretrainepochs > 0:
            print("Pretraining on KB data")

            t_max = epochs * len(traindl) / gradacc
            warmupsteps = int(round(warmup * t_max))
            print(f"Total number of updates: {t_max} . Warmup: {warmupsteps}")
            if cosinecycles == 0:  # constant lr
                tt.msg("Using constant LR with warmup")
                lr_schedule = q.sched.Linear(0, 1, steps=warmupsteps) >> 1.
            else:
                tt.msg("Using cosine LR with restart and with warmup")
                lr_schedule = q.sched.Linear(0, 1, steps=warmupsteps) >> (
                    CosineWithRestart(high=1., low=0.1, cycles=cosinecycles, steps=t_max - warmupsteps)) >> 0.1

            lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

            kbvalidepoch = partial(q.test_epoch,
                                   model=m,
                                   losses=kbtmetrics,
                                   dataloader=kbevaltraindl,
                                   device=device,
                                   on_end=[lambda: wandb_logger_kbft()])

            kbtrainbatch = partial(q.train_batch,
                                   gradient_accumulation_steps=gradacc,
                                   on_before_optim_step=[
                                       lambda: torch.nn.utils.clip_grad_norm_(params, gradnorm),
                                       lambda: lr_schedule.step()
                                   ]
                                   )

            kbtrainepoch = partial(q.train_epoch, model=m,
                                   dataloader=kbtraindl,
                                   optim=optim,
                                   losses=kbtloss,
                                   device=device,
                                   _train_batch=kbtrainbatch,
                                   on_start=[],
                                   on_end=[modelsaver])

            tt.tick("training")
            q.run_training(run_train_epoch=kbtrainepoch,
                           run_valid_epoch=[kbvalidepoch] if not debugcode else [kbvalidepoch],
                           max_epochs=kbpretrainepochs,
                           validinter=kbpretrainvalidinter,
                           )
            tt.tock("done training")

        # do training on QA data
        print("Training on QA data")
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
                             on_end=[modelsaver])

        trainevalepoch = partial(q.test_epoch,
                                 model=m,
                                 losses=tmetrics,
                                 dataloader=evaltraindl,
                                 device=device,
                                 on_end=[])

        tt.tick("training")
        q.run_training(run_train_epoch=trainepoch,
                       run_valid_epoch=[trainevalepoch, validepoch] if not debugcode else [trainevalepoch, validepoch],
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
                            on_end=[])

        tt.tick("running test before reloading")
        testres = testepoch()

        settings.update({"test_acc_at_end": xmetrics[0].get_epoch_error()})
        tt.tock(f"Test results: {testres}")

        if eyt.remembered is not None:
            tt.msg("reloading model with best validation metric")
            m = eyt.remembered

        tt.tick("running evaluation on subset of train")
        trainres, _ = self.evaluate(m, evaltrainds, tok=tok, batsize=testbatsize, device=device, maxnumans=100, savename=f"runs/{self.VARIANT}/{wandbrun.name}/evaltrain.outs.json")
        settings.update({"train_fscore_at_earlystop": trainres["fscore"]})
        tt.tock(f"Train results: {trainres}")

        tt.tick("running evaluation on validation set")
        validres, _ = self.evaluate(m, validds, tok=tok, batsize=testbatsize, device=device, maxnumans=100, savename=f"runs/{self.VARIANT}/{wandbrun.name}/valid.outs.json")
        settings.update({"valid_fscore_at_earlystop": validres["fscore"]})
        tt.tock(f"Validation results: {validres}")

        tt.tick("running test")
        testres, _ = self.evaluate(m, testds, tok=tok, batsize=testbatsize, device=device, maxnumans=100, savename=f"runs/{self.VARIANT}/{wandbrun.name}/test.outs.json")
        settings.update({"test_fscore_at_earlystop": testres["fscore"]})
        tt.tock(f"Test results: {testres}")

        wandb.config.update(settings)
        q.pp_dict(settings)
        wandbrun.finish()
        # sleep(15)

    def evaluate(self, m:Model, ds, tok=None, batsize=10, device=torch.device("cpu"), maxnumans=100, savename=None):
        expected = self.get_expected(ds, tok=tok)
        predictions = self.get_predictions(m, ds, tok=tok, batsize=batsize, device=device, maxnumans=maxnumans)

        metrics = self.compute_metrics(predictions, expected)

        fscores, precisions, recalls = list(zip(*list(metrics.values())))
        ret = {"fscore": np.mean(fscores),
               "precision": np.mean(precisions),
               "recall":  np.mean(recalls),
               }

        if savename is not None:
            p = Path(savename)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w") as f:
                json.dump({"expected": {k: tuple(v) for k, v in expected.items()},
                           "predictions": {k: tuple(v) for k, v in predictions.items()},
                           "metrics": metrics},
                          f)

        return ret, predictions

    def compute_metrics(self, preds, exps):
        missing = 0
        ret = {}
        for k, exp in exps.items():
            pred = preds[k] if k in preds else set()
            if k not in preds:
                missing += 1
            tp = len(pred & exp)
            recall = tp / max(1e-6, len(exp))
            precision = tp / max(1e-6, len(pred))
            fscore = 2 * recall * precision / max(1e-6, (recall + precision))
            ret[k] = (fscore, precision, recall)
        # assert missing == 0
        return ret

    def get_expected(self, ds, tok=None):
        rootds = ds.rootds
        ret = {}
        for example in rootds.examples:
            inptensor, answers, _ = example
            inpstr = decode(inptensor, tok)
            if inpstr not in ret:
                ret[inpstr] = set()
            ret[inpstr].update(answers)
        return ret

    def get_predictions(self, m:Model, ds, tok=None, batsize=10, device=torch.device("cpu"), maxnumans=100):
        tto = q.ticktock("pred")
        tto.tick("predicting")
        tt = q.ticktock("-")
        dssize = len(ds)
        vocab = {k: v for k, v in tok.vocab.items()}

        m.eval()
        m.to(device)

        ret = {}
        with torch.no_grad():
            batch = []
            i = 0
            done = False
            while not done:
                while len(batch) < batsize:
                    inpstr = ds[i][0][0]
                    inpstr = decode(inpstr, tok)
                    batch.append((inpstr, ds[i][0][0], ds[i][1][0]))
                    batch[-1][-1][0] = vocab[f"[ITEM-0]"]
                    i += 1
                    if i == len(ds):
                        done = True
                        break

                # tt.msg(f"{i}/{len(ds)}")
                tt.live(f"{i}/{len(ds)}")
                packedbatch = autocollate(batch)
                packedbatch = q.recmap(packedbatch, lambda x: x.to(device) if hasattr(x, "to") else x)

                _, outs = m(*packedbatch[1:])
                newbatch = []
                for j, out in enumerate(outs):
                    inpstr = batch[j][0]
                    _inpstr, predstr, _ = out
                    if inpstr not in ret:
                        ret[inpstr] = set()
                    match = re.match(r"\[ITEM-(\d+)\]\[TOTAL-(\d+)\](.+)", predstr)
                    ansid, numans, ans = 0, 0, None
                    if match:
                        ansid = int(match.group(1))
                        numans = int(match.group(2))
                        ans = match.group(3)
                        ret[inpstr].add(re.sub(r"\[[^\]]+\]", "", ans).strip())
                        if ansid + 1 >= min(numans, maxnumans):     # we can remove this example from the batch
                            pass
                        else:
                            newbatch.append(batch[j])
                            newbatch[-1][-1][0] = vocab[f"[ITEM-{ansid+1}]"]

                batch = newbatch

        return ret


def run_experiment(
        lr=-1.,
        kbpretrainepochs=0,
        kbpretrainvalidinter=-1,
        useadafactor=False,
        gradnorm=2,
        gradacc=1,
        batsize=-1,
        maxlen=-1,
        testbatsize=-1,
        epochs=-1,
        validinter=-1.,
        validfrac=0.1,
        warmup=0.1,
        cosinecycles=0,
        modelsize="small",
        dataset="default",
        seed=-1,
        dropout=-1.,
        testcode=False,
        debugcode=False,
        gpu=-1,
        recomputedata=False,
        nosave=False,
        loadfrom="",
        usesavedconfig=False,
        ):

    settings = locals().copy()

    main = Main()
    maxlen = main.MAXLEN if maxlen == -1 else maxlen
    settings["maxlen"] = maxlen

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
        main.run, ranges, path_prefix=None, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)

    # python traint5_ftbase_setdec.py -gpu 0 -batsize 150 -testbatsize 300 -epochs 16 -kbpretrainvalidinter 5 -validinter 1 -modelsize base -kbpretrainepochs 121 -seed 42