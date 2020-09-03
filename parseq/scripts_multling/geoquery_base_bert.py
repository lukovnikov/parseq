# encoding: utf-8
"""
A script for running the following experiments:
* dataset: Overnight
* model: BART encoder + vanilla Transformer decoder for LF
* training:  normal (CE on teacher forced target)
"""
import json
import random
import string
from copy import deepcopy
from functools import partial
from typing import Callable, Set

import fire
# import wandb

import qelos as q   # branch v3
import numpy as np
import torch
from torch.utils.data import DataLoader

from parseq.datasets import OvernightDatasetLoader, pad_and_default_collate, autocollate
from parseq.decoding import merge_metric_dicts
from parseq.eval import SeqAccuracies, TreeAccuracy, make_array_of_metrics, CELoss
from parseq.grammar import tree_to_lisp_tokens, lisp_to_tree
from parseq.scripts_multling.geoquery_data import load_multilingual_geoquery
from parseq.vocab import SequenceEncoder, Vocab
from transformers import AutoTokenizer, AutoModel, BartConfig, BartModel, BartForConditionalGeneration


UNKID = 3

DATA_RESTORE_REVERSE = False


class BartGenerator(BartForConditionalGeneration):
    def __init__(self, config:BartConfig, encoderconfig=None):
        super(BartGenerator, self).__init__(config)
        self.encoderconfig = encoderconfig
        self.outlin = torch.nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
        **unused
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = self.outlin(outputs[0])
        outputs = (lm_logits,) + outputs[1:] + outputs[0:1]  # Add hidden states and attention if they are here
        return outputs


class BartGeneratorTrain(torch.nn.Module):
    def __init__(self, model:BartGenerator, smoothing=0., tensor2tree:Callable=None, orderless:Set[str]=set(), **kw):
        super(BartGeneratorTrain, self).__init__(**kw)
        self.model = model

        # CE loss
        self.ce = CELoss(ignore_index=model.config.pad_token_id, smoothing=smoothing)

        # accuracies
        self.accs = SeqAccuracies()
        self.accs.padid = model.config.pad_token_id
        self.accs.unkid = UNKID

        self.treeacc = TreeAccuracy(tensor2tree=tensor2tree,
                                    orderless=orderless)

        self.metrics = [self.ce, self.accs, self.treeacc]

    def forward(self, input_ids, _, output_ids, *args, **kwargs):
        ret = self.model(input_ids,
                         attention_mask=input_ids!=self.model.encoderconfig.pad_token_id,
                         decoder_input_ids=output_ids)
        probs = ret[0]
        _, predactions = probs.max(-1)
        outputs = [metric(probs, predactions, output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


class BartGeneratorTest(BartGeneratorTrain):
    def __init__(self, model:BartGenerator, maxlen:int=5, numbeam:int=1,
                 tensor2tree:Callable=None, orderless:Set[str]=set(), **kw):
        super(BartGeneratorTest, self).__init__(model, **kw)
        self.maxlen, self.numbeam = maxlen, numbeam
        # accuracies
        self.accs = SeqAccuracies()
        self.accs.padid = model.config.pad_token_id
        self.accs.unkid = UNKID

        self.treeacc = TreeAccuracy(tensor2tree=tensor2tree,
                                    orderless=orderless)

        self.metrics = [self.accs, self.treeacc]

    def forward(self, _, input_ids, output_ids, *args, **kwargs):
        ret = self.model.generate(input_ids,
                                  decoder_input_ids=output_ids[:, 0:1],
                                  attention_mask=input_ids!=self.model.encoderconfig.pad_token_id,
                                  max_length=self.maxlen,
                                  num_beams=self.numbeam)
        outputs = [metric(None, ret[:, 1:], output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


def create_model(encoder_name="xlm-roberta-base",
                 dec_vocabsize=None, dec_layers=6, dec_dim=640, dec_heads=8, dropout=0., dropoutdec=0.,
                 maxlen=20, smoothing=0., numbeam=1, tensor2tree=None):
    # if encoder_name != "bert-base-uncased":
    #     raise NotImplemented(f"encoder '{encoder_name}' not supported yet.")
    pretrained = AutoModel.from_pretrained(encoder_name)
    encoder = pretrained

    class BertEncoderWrapper(torch.nn.Module):
        def __init__(self, model, dropout=0., **kw):
            super(BertEncoderWrapper, self).__init__(**kw)
            self.model = model
            self.proj = torch.nn.Linear(pretrained.config.hidden_size, dec_dim, bias=False)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, input_ids, attention_mask=None):
            ret, _ = self.model(input_ids, attention_mask=attention_mask)
            if pretrained.config.hidden_size != dec_dim:
                ret = self.proj(ret)
            # ret = self.dropout(ret)
            ret = (ret, None, None)
            return ret

    encoder = BertEncoderWrapper(encoder, dropout=dropout)

    decoder_config = BartConfig(d_model=dec_dim,
                                pad_token_id=0,
                                bos_token_id=1,
                                vocab_size=dec_vocabsize,
                                decoder_attention_heads=dec_heads//2,
                                decoder_layers=dec_layers,
                                dropout=dropoutdec,
                                attention_dropout=min(0.1, dropout/2),
                                decoder_ffn_dim=dec_dim*4,
                                encoder_attention_heads=dec_heads,
                                encoder_layers=dec_layers,
                                encoder_ffn_dim=dec_dim*4,
                                )
    model = BartGenerator(decoder_config, encoder.model.config)
    model.model.encoder = encoder

    orderless = {"op:and", "SW:concat"}

    trainmodel = BartGeneratorTrain(model, smoothing=smoothing, tensor2tree=tensor2tree, orderless=orderless)
    testmodel = BartGeneratorTest(model, maxlen=maxlen, numbeam=numbeam, tensor2tree=tensor2tree, orderless=orderless)
    return trainmodel, testmodel


def _tensor2tree(x, D:Vocab=None):
    # x: 1D int tensor
    x = list(x.detach().cpu().numpy())
    x = [D(xe) for xe in x]
    x = [xe for xe in x if xe != D.padtoken]

    # find first @END@ and cut off
    parentheses_balance = 0
    for i in range(len(x)):
        if x[i] ==D.endtoken:
            x = x[:i]
            break
        elif x[i] == "(" or x[i][-1] == "(":
            parentheses_balance += 1
        elif x[i] == ")":
            parentheses_balance -= 1
        else:
            pass

    # balance parentheses
    while parentheses_balance > 0:
        x.append(")")
        parentheses_balance -= 1
    i = len(x) - 1
    while parentheses_balance < 0 and i > 0:
        if x[i] == ")":
            x.pop(i)
            parentheses_balance += 1
        else:
            break
        i -= 1

    # convert to nltk.Tree
    try:
        tree, parsestate = lisp_to_tree(" ".join(x), None)
    except Exception as e:
        tree = None
    return tree


def collate_fn(x, pad_value_nl=0, pad_value_fl=0):
    y = list(zip(*x))
    assert(len(y) == 3)
    y[0] = torch.stack(q.pad_tensors(y[0], 0, pad_value_nl), 0)
    y[1] = torch.stack(q.pad_tensors(y[1], 0, pad_value_nl), 0)
    y[2] = torch.stack(q.pad_tensors(y[2], 0, pad_value_fl), 0)
    return tuple(y)


def run(sourcelang="en",
        targetlang="en",
        lr=0.001,
        enclrmul=0.1,
        numbeam=1,
        cosinelr=False,
        warmup=0.,
        batsize=20,
        epochs=100,
        dropout=0.1,
        dropoutdec=0.1,
        wreg=1e-9,
        gradnorm=3,
        smoothing=0.,
        patience=5,
        gpu=-1,
        seed=123456789,
        encoder="xlm-roberta-base",
        numlayers=6,
        hdim=600,
        numheads=8,
        maxlen=30,
        localtest=False,
        printtest=False,
        trainonvalid=False,
        ):
    settings = locals().copy()
    print(json.dumps(settings, indent=4))
    # wandb.init(project=f"overnight_pretrain_bert-{domain}",
    #            reinit=True, config=settings)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tt = q.ticktock("script")
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt.tick("loading data")

    nltok_name = encoder
    tds, vds, xds, nltok, flenc = load_multilingual_geoquery(sourcelang, targetlang, nltok_name=nltok_name, trainonvalid=trainonvalid)
    tt.msg(f"{len(tds)/(len(tds) + len(vds) + len(xds)):.2f}/{len(vds)/(len(tds) + len(vds) + len(xds)):.2f}/{len(xds)/(len(tds) + len(vds) + len(xds)):.2f} ({len(tds)}/{len(vds)}/{len(xds)}) train/valid/test")
    tdl = DataLoader(tds, batch_size=batsize, shuffle=True, collate_fn=partial(collate_fn, pad_value_nl=nltok.pad_token_id))
    vdl = DataLoader(vds, batch_size=batsize, shuffle=False, collate_fn=partial(collate_fn, pad_value_nl=nltok.pad_token_id))
    xdl = DataLoader(xds, batch_size=batsize, shuffle=False, collate_fn=partial(collate_fn, pad_value_nl=nltok.pad_token_id))
    tt.tock("data loaded")

    tt.tick("creating model")
    trainm, testm = create_model(encoder_name=encoder,
                                 dec_vocabsize=flenc.vocab.number_of_ids(),
                                 dec_layers=numlayers,
                                 dec_dim=hdim,
                                 dec_heads=numheads,
                                 dropout=dropout,
                                 dropoutdec=dropoutdec,
                                 smoothing=smoothing,
                                 maxlen=maxlen,
                                 numbeam=numbeam,
                                 tensor2tree=partial(_tensor2tree, D=flenc.vocab)
                                 )
    tt.tock("model created")

    # run a batch of data through the model
    if localtest:
        batch = next(iter(tdl))
        out = trainm(*batch)
        print(out)
        out = testm(*batch)
        print(out)

    metrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    vmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    xmetrics = make_array_of_metrics("seq_acc", "tree_acc")

    trainable_params = list(trainm.named_parameters())
    exclude_params = set()
    # exclude_params.add("model.model.inp_emb.emb.weight")  # don't train input embeddings if doing glove
    if len(exclude_params) > 0:
        trainable_params = [(k, v) for k, v in trainable_params if k not in exclude_params]

    tt.msg("different param groups")
    encparams = [v for k, v in trainable_params if k.startswith("model.model.encoder.model")]
    otherparams = [v for k, v in trainable_params if not k.startswith("model.model.encoder.model")]
    if len(encparams) == 0:
        raise Exception("No encoder parameters found!")
    paramgroups = [{"params": encparams, "lr": lr * enclrmul},
                   {"params": otherparams}]

    optim = torch.optim.Adam(paramgroups, lr=lr, weight_decay=wreg)

    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(trainm.parameters(), gradnorm)


    eyt = q.EarlyStopper(vmetrics[-1], patience=patience, min_epochs=10, more_is_better=True, remember_f=lambda: deepcopy(trainm.model))
    # def wandb_logger():
    #     d = {}
    #     for name, loss in zip(["loss", "elem_acc", "seq_acc", "tree_acc"], metrics):
    #         d["_train_"+name] = loss.get_epoch_error()
    #     for name, loss in zip(["seq_acc", "tree_acc"], vmetrics):
    #         d["_valid_"+name] = loss.get_epoch_error()
    #     wandb.log(d)
    t_max = epochs
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=t_max-warmup) >> 0.
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=trainm, dataloader=tdl, optim=optim, losses=metrics,
                         _train_batch=trainbatch, device=device, on_end=[lambda: lr_schedule.step()])
    validepoch = partial(q.test_epoch, model=testm, dataloader=vdl, losses=vmetrics, device=device, on_end=[lambda: eyt.on_epoch_end()])#, on_end=[lambda: wandb_logger()])

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs, check_stop=[lambda: eyt.check_stop()])
    tt.tock("done training")

    if eyt.remembered is not None:
        trainm.model = eyt.remembered
        testm.model = eyt.remembered
    tt.msg("reloaded best")

    tt.tick("testing")
    validresults = q.test_epoch(model=testm, dataloader=vdl, losses=vmetrics, device=device)
    testresults = q.test_epoch(model=testm, dataloader=xdl, losses=xmetrics, device=device)
    print(validresults)
    print(testresults)
    tt.tock("tested")

    if printtest:
        predm = testm.model
        predm.to(device)
        c, t = 0, 0
        for testbatch in iter(xdl):
            input_ids = testbatch[0]
            output_ids = testbatch[1]
            input_ids = input_ids.to(device)
            ret = predm.generate(input_ids, attention_mask=input_ids != predm.config.pad_token_id,
                                      max_length=maxlen)
            inp_strs = [nltok.decode(input_idse, skip_special_tokens=True, clean_up_tokenization_spaces=False) for input_idse in input_ids]
            out_strs = [flenc.vocab.tostr(rete.to(torch.device("cpu"))) for rete in ret]
            gold_strs = [flenc.vocab.tostr(output_idse.to(torch.device("cpu"))) for output_idse in output_ids]

            for x, y, g in zip(inp_strs, out_strs, gold_strs):
                print(" ")
                print(f"'{x}'\n--> {y}\n <=> {g}")
                if y == g:
                    c += 1
                else:
                    print("NOT SAME")
                t += 1
        print(f"seq acc: {c/t}")
        # testout = q.eval_loop(model=testm, dataloader=xdl, device=device)
        # print(testout)

    print("done")
    # settings.update({"train_seqacc": losses[]})

    for metricarray, datasplit in zip([metrics, vmetrics, xmetrics], ["train", "valid", "test"]):
        for metric in metricarray:
            settings[f"{datasplit}_{metric.name}"] = metric.get_epoch_error()

    # wandb.config.update(settings)
    # print(settings)
    return settings


def run_experiments(lang="en", gpu=-1):
    ranges = {
        "lr": [0.0001, 0.00001], #[0.001, 0.0001, 0.00001],
        "enclrmul": [1., 0.1], #[1., 0.1, 0.01],
        "warmup": [2],
        "epochs": [100], #[50, 100],
        "numheads": [8, 12, 16],
        "numlayers": [3, 6, 9],
        "dropout": [.1, .2],
        "hdim": [384, 768, 960], #[192, 384, 768, 960],
        "seed": [12345678], #, 98387670, 23655798, 66453829],     # TODO: add more later
        "cosinelr": [True],
    }
    ranges = {
        "lr": [0.0001],
        "enclrmul": [0.1],
        "warmup": [0],
        "epochs": [50, 100, 75],
        "numheads": [12],
        "numlayers": [3],
        "dropout": [.1],
        "hdim": [768],
        "cosinelr": [True],
        "seed": [12345678],     # TODO: add more later
    }
    p = __file__ + f".{lang}"
    def check_config(x):
        effectiveenclr = x["enclrmul"] * x["lr"]
        if effectiveenclr < 0.00001:
            return False
        dimperhead = x["hdim"] / x["numheads"]
        if dimperhead < 20 or dimperhead > 100:
            return False
        return True

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config,
                      lang=lang, gpu=gpu)


def run_experiments_seed(sourcelang="en", targetlang="en", lr=-1., batsize=-1, patience=-1, enclrmul=-1., hdim=-1, dropout=-1., dropoutdec=-1., numlayers=-1, numheads=-1, gpu=-1, epochs=-1,
                         smoothing=0.2, numbeam=1, trainonvalid=False, cosinelr=False):
    ranges = {
        "lr": [0.0001],
        "batsize": [20],
        "patience": [5],
        "enclrmul": [0.1],
        "warmup": [0],
        "epochs": [50],
        "numheads": [12],
        "numlayers": [3],
        "dropout": [.1],
        "dropoutdec": [.1],
        "hdim": [768],
        "cosinelr": [cosinelr],
        "seed": [12345678, 65748390, 98387670, 23655798, 66453829],     # TODO: add more later
    }
    if lr >= 0:
        ranges["lr"] = [lr]
    if batsize >= 0:
        ranges["batsize"] = [batsize]
    if patience >= 0:
        ranges["patience"] = [patience]
    if hdim >= 0:
        ranges["hdim"] = [hdim]
    if dropout >= 0:
        ranges["dropout"] = [dropout]
    if dropoutdec >= 0:
        ranges["dropoutdec"] = [dropoutdec]
    if numlayers >= 0:
        ranges["numlayers"] = [numlayers]
    if numheads >= 0:
        ranges["numheads"] = [numheads]
    if enclrmul >= 0:
        ranges["enclrmul"] = [enclrmul]
    if epochs >= 0:
        ranges["epochs"] = [epochs]
    p = __file__ + f".{sourcelang}-{targetlang}"
    def check_config(x):
        effectiveenclr = x["enclrmul"] * x["lr"]
        # if effectiveenclr < 0.000005:
        #     return False
        dimperhead = x["hdim"] / x["numheads"]
        if dimperhead < 20 or dimperhead > 100:
            return False
        return True

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config,
                      sourcelang=sourcelang, targetlang=targetlang, gpu=gpu, smoothing=smoothing, numbeam=numbeam,
                      trainonvalid=trainonvalid)


if __name__ == '__main__':
    # ret = fire.Fire(run)
    # print(ret)
    # q.argprun(run_experiments)
    fire.Fire(run_experiments_seed)