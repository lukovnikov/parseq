# encoding: utf-8
"""
A script for running the following zero-shot domain transfer experiments:
* dataset: Overnight
* model: BART encoder + vanilla Transformer decoder for LF
    * lexical token representations are computed based on lexicon
* training: normal (CE on teacher forced target)
"""


import random
import string
from copy import deepcopy
from functools import partial
from typing import Callable, Set

import qelos as q   # branch v3
import numpy as np
import torch
from nltk import Tree
from torch.utils.data import DataLoader

from parseq.datasets import OvernightDatasetLoader, pad_and_default_collate, autocollate, Dataset
from parseq.decoding import merge_metric_dicts
from parseq.eval import SeqAccuracies, TreeAccuracy, make_array_of_metrics, CELoss
from parseq.grammar import tree_to_lisp_tokens, lisp_to_tree
from parseq.vocab import SequenceEncoder, Vocab
from transformers import AutoTokenizer, AutoModel, BartConfig, BartModel, BartForConditionalGeneration


UNKID = 3


def get_labels_from_tree(x:Tree):
    ret = {x.label()}
    for child in x:
        ret |= get_labels_from_tree(child)
    return ret


overnightlexicon = {
    "": ""
}


def get_shared_tokens(domains="restaurants,housing,blocks,calendar,calendarplus,publications,recipes"):
    ret = {'agg:arg:sum', 'SW:CNT-arg:<=', 'cond:arg:>=', 'agg:arg:avg', 'SW:CNT-arg:max', 'cond:arg:<', 'cond:arg:<=', 'number', 'SW:ensureNumericProperty', 'cond:arg:>', 'cond:has', 'SW:concat', 'SW:getProperty', 'cond:arg:=', 'date', 'SW:CNT-arg:>', 'op:and', 'arg:~type', 'SW:CNT-arg:=', 'cond:arg:!=', 'SW:CNT-arg:>=', 'SW:ensureNumericEntity', 'arg:min', 'SW:superlative', 'SW:CNT-arg:<', 'arg:max', 'SW:CNT-arg:min'}
    # return ret
    domains = domains.split(",")
    tokendomaincounts = {}
    alltokens = set()
    for domain in domains:
        domaintokens = set()
        ds = OvernightDatasetLoader(simplify_mode="light").load(domain)
        for example in ds.examples:
            example_tokens = get_labels_from_tree(example[1])
            if "SW:reverse" in example_tokens:
                print(example)
            domaintokens |= set(example_tokens)
        for domaintoken in domaintokens:
            if domaintoken not in tokendomaincounts:
                tokendomaincounts[domaintoken] = 0
            tokendomaincounts[domaintoken] += 1
        alltokens |= domaintokens
    sharedtokens = set([k for k, v in tokendomaincounts.items() if v > 1])
    print("tokens needing lexicon")
    for token in alltokens:
        if token not in ret:
            print(token)
    return ret


def get_maximum_spanning_examples(examples, mincoverage=1, loadedex=None):
    """
    Sort given examples by the degree they span their vocabulary.
    First examples maximally increase how much least seen tokens are seen.
    :param examples:
    :param mincoverage: the minimum number of times every token must be covered.
     If the token occurs less than 'mincoverage' number of times in given 'examples',
      all examples with that token are included but the 'mincoverage' criterion is not satisfied!
    :return:
    """
    tokencounts = {}
    uniquetokensperexample = []
    examplespertoken = {}        # reverse index from token to example number
    for i, example in enumerate(examples):
        exampletokens = set(get_labels_from_tree(example[1]))
        uniquetokensperexample.append(exampletokens)
        for token in exampletokens:
            if token not in tokencounts:
                tokencounts[token] = 0
            tokencounts[token] += 1
            if token not in examplespertoken:
                examplespertoken[token] = set()
            examplespertoken[token].add(i)

    scorespertoken = {k: len(examples) / len(examplespertoken[k]) for k in examplespertoken.keys()}

    selectiontokencounts = {k: 0 for k, v in tokencounts.items()}

    if loadedex is not None:
        for i, example in enumerate(loadedex):
            exampletokens = set(get_labels_from_tree(example[1]))
            for token in exampletokens:
                if token in selectiontokencounts:
                    selectiontokencounts[token] += 1

    def get_example_score(i):
        minfreq = min(selectiontokencounts.values())
        ret = 0
        for token in uniquetokensperexample[i]:
            ret += 1/8 ** (selectiontokencounts[token] - minfreq)
        return ret

    exampleids = set(range(len(examples)))
    outorder = []

    i = 0

    while len(exampleids) > 0:
        sortedexampleids = sorted(exampleids, key=get_example_score, reverse=True)
        outorder.append(sortedexampleids[0])
        exampleids -= {sortedexampleids[0]}
        # update selection token counts
        for token in uniquetokensperexample[sortedexampleids[0]]:
            selectiontokencounts[token] += 1
        minfreq = np.infty
        for k, v in selectiontokencounts.items():
            if tokencounts[k] < mincoverage and selectiontokencounts[k] >= tokencounts[k]:
                pass
            else:
                minfreq = min(minfreq, selectiontokencounts[k])
        i += 1
        if minfreq >= mincoverage:
            break

    out = [examples[i] for i in outorder]
    return out


def load_ds(traindomains="restaurants",
            testdomain="housing",
            min_freq=1,
            mincoverage=1,
            top_k=np.infty,
            nl_mode="bart-large"):
    traindomains = traindomains.split(",")
    allex = []
    for traindomain in traindomains:
        ds = OvernightDatasetLoader(simplify_mode="light").load(domain=traindomain)
        allex += ds[(None, None, lambda x: x in ("train", "valid"))].map(lambda x: (x[0], x[1], x[2], traindomain)).examples       # don't use test examples

    testds = OvernightDatasetLoader(simplify_mode="light").load(domain=testdomain)
    sortedexamples = get_maximum_spanning_examples(testds[(None, None, "train")].examples, mincoverage=mincoverage, loadedex=allex)

    allex += testds[(None, None, "ftvalid")].map(lambda x: (x[0], x[1], x[2], testdomain)).examples
    allex += testds[(None, None, "test")].map(lambda x: (x[0], x[1], x[2], testdomain)).examples
    allex += [(ex[0], ex[1], "fttrain", testdomain) for ex in sortedexamples]

    ds = Dataset(allex)

    seqenc_vocab = Vocab(padid=1, startid=0, endid=2, unkid=UNKID)
    seqenc = SequenceEncoder(vocab=seqenc_vocab, tokenizer=tree_to_lisp_tokens,
                             add_start_token=True, add_end_token=True)
    for example in ds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=example[2] in ("train", "finetune"))
    seqenc.finalize_vocab(min_freq=min_freq, top_k=top_k)

    nl_tokenizer = AutoTokenizer.from_pretrained(nl_mode)
    def tokenize(x):
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0],
               seqenc.convert(x[1], return_what="tensor"),
               x[2],
               x[0], x[1], x[3])
        return ret
    tds, ftds, vds, fvds, xds = ds[(None, None, "train", None)].map(tokenize), \
                          ds[(None, None, "fttrain", None)].map(tokenize), \
                          ds[(None, None, "valid", None)].map(tokenize), \
                          ds[(None, None, "ftvalid", None)].map(tokenize), \
                          ds[(None, None, "test", None)].map(tokenize)
    return tds, ftds, vds, fvds, xds, nl_tokenizer, seqenc


class BartGenerator(BartForConditionalGeneration):
    def __init__(self, config:BartConfig):
        super(BartGenerator, self).__init__(config)
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
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
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

    def forward(self, input_ids, output_ids, *args, **kwargs):
        ret = self.model(input_ids, attention_mask=input_ids!=self.model.config.pad_token_id, decoder_input_ids=output_ids)
        probs = ret[0]
        _, predactions = probs.max(-1)
        outputs = [metric(probs, predactions, output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


class BartGeneratorTest(BartGeneratorTrain):
    def __init__(self, model:BartGenerator, maxlen:int=5, numbeam:int=None,
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

    def forward(self, input_ids, output_ids, *args, **kwargs):
        ret = self.model.generate(input_ids, attention_mask=input_ids!=self.model.config.pad_token_id, max_length=self.maxlen, num_beams=self.numbeam)
        outputs = [metric(None, ret[:, 1:], output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


def create_model(encoder_name="bart-large",
                 dec_vocabsize=None, dec_layers=6, dec_dim=640, dec_heads=8, dropout=0.,
                 maxlen=20, smoothing=0., tensor2tree=None):
    if encoder_name != "bart-large":
        raise NotImplemented(f"encoder '{encoder_name}' not supported yet.")
    pretrained = AutoModel.from_pretrained(encoder_name)
    encoder = pretrained.encoder

    if pretrained.config.d_model != dec_dim:
        class BartEncoderWrapper(torch.nn.Module):
            def __init__(self, model, **kw):
                super(BartEncoderWrapper, self).__init__(**kw)
                self.model = model
                self.proj = torch.nn.Linear(pretrained.config.d_model, dec_dim, bias=False)

            def forward(self, input_ids, attention_mask=None):
                ret = self.model(input_ids, attention_mask=attention_mask)
                ret = (self.proj(ret[0]), ret[1], ret[2])
                return ret

        encoder = BartEncoderWrapper(encoder)

    decoder_config = BartConfig(d_model=dec_dim,
                                pad_token_id=1,
                                vocab_size=dec_vocabsize,
                                decoder_attention_heads=dec_heads//2,
                                decoder_layers=dec_layers,
                                dropout=dropout,
                                decoder_ffn_dim=dec_dim*4,
                                encoder_attention_heads=dec_heads,
                                encoder_layers=dec_layers,
                                encoder_ffn_dim=dec_dim*4,
                                )
    model = BartGenerator(decoder_config)
    model.model.encoder = encoder

    orderless = {"op:and", "SW:concat"}

    trainmodel = BartGeneratorTrain(model, smoothing=smoothing, tensor2tree=tensor2tree, orderless=orderless)
    testmodel = BartGeneratorTest(model, maxlen=maxlen, numbeam=None, tensor2tree=tensor2tree, orderless=orderless)
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
        i -= 1

    # convert to nltk.Tree
    try:
        tree, parsestate = lisp_to_tree(" ".join(x), None)
    except Exception as e:
        tree = None
    return tree


def run(traindomains="recipes,blocks,calendar,housing,publications,calendarplus",
        testdomain="restaurants",
        mincoverage=2,
        lr=0.001,
        enclrmul=0.1,
        ftlr=0.0001,
        cosinelr=False,
        warmup=0.,
        batsize=30,
        epochs=100,
        dropout=0.1,
        wreg=1e-9,
        gradnorm=3,
        smoothing=0.,
        patience=5,
        gpu=-1,
        seed=123456789,
        encoder="bart-large",
        numlayers=6,
        hdim=600,
        numheads=8,
        maxlen=50,
        localtest=False,
        printtest=False,
        ):
    settings = locals().copy()
    print(locals())
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tt = q.ticktock("script")
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt.tick("loading data")
    tds, ftds, vds, fvds, xds, nltok, flenc = load_ds(traindomains=traindomains, testdomain=testdomain, nl_mode=encoder, mincoverage=mincoverage)
    tdl = DataLoader(tds, batch_size=batsize, shuffle=True, collate_fn=partial(autocollate, pad_value=1))
    ftdl = DataLoader(ftds, batch_size=batsize, shuffle=True, collate_fn=partial(autocollate, pad_value=1))
    vdl = DataLoader(vds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=1))
    fvdl = DataLoader(fvds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=1))
    xdl = DataLoader(xds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=1))
    tt.tock("data loaded")

    tt.tick("creating model")
    trainm, testm = create_model(encoder_name=encoder,
                                 dec_vocabsize=flenc.vocab.number_of_ids(),
                                 dec_layers=numlayers,
                                 dec_dim=hdim,
                                 dec_heads=numheads,
                                 dropout=dropout,
                                 smoothing=smoothing,
                                 maxlen=maxlen,
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

    # region pretrain on all domains
    metrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    vmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    xmetrics = make_array_of_metrics("seq_acc", "tree_acc")

    trainable_params = list(trainm.named_parameters())
    exclude_params = set()
    # exclude_params.add("model.model.inp_emb.emb.weight")  # don't train input embeddings if doing glove
    if len(exclude_params) > 0:
        trainable_params = [(k, v) for k, v in trainable_params if k not in exclude_params]

    tt.msg("different param groups")
    encparams = [v for k, v in trainable_params if k.startswith("model.model.encoder")]
    otherparams = [v for k, v in trainable_params if not k.startswith("model.model.encoder")]
    if len(encparams) == 0:
        raise Exception("No encoder parameters found!")
    paramgroups = [{"params": encparams, "lr": lr * enclrmul},
                   {"params": otherparams}]

    optim = torch.optim.Adam(paramgroups, lr=lr, weight_decay=wreg)

    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(trainm.parameters(), gradnorm)

    eyt = q.EarlyStopper(vmetrics[1], patience=patience, min_epochs=10, more_is_better=True, remember_f=lambda: deepcopy(trainm.model))

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
    validepoch = partial(q.test_epoch, model=testm, dataloader=vdl, losses=vmetrics, device=device, on_end=[lambda: eyt.on_epoch_end()])

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs, check_stop=[lambda: eyt.check_stop()])
    tt.tock("done training")

    if eyt.get_remembered() is not None:
        tt.msg("reloaded")
        trainm.model = eyt.get_remembered()
        testm.model = eyt.get_remembered()

    # endregion

    # region finetune
    ftmetrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    ftvmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    ftxmetrics = make_array_of_metrics("seq_acc", "tree_acc")

    ftoptim = torch.optim.Adam(trainm.parameters(), lr=ftlr, weight_decay=wreg)

    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(trainm.parameters(), gradnorm)

    eyt = q.EarlyStopper(ftvmetrics[1], patience=patience, min_epochs=10, more_is_better=True,
                         remember_f=lambda: deepcopy(trainm.model))

    t_max = epochs
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=t_max - warmup) >> 0.
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(ftoptim, lr_schedule)

    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=trainm, dataloader=ftdl, optim=ftoptim, losses=ftmetrics,
                         _train_batch=trainbatch, device=device, on_end=[lambda: lr_schedule.step()])
    validepoch = partial(q.test_epoch, model=testm, dataloader=fvdl, losses=ftvmetrics, device=device,
                         on_end=[lambda: eyt.on_epoch_end()])

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()])
    tt.tock("done training")

    if eyt.get_remembered() is not None:
        tt.msg("reloaded")
        trainm.model = eyt.get_remembered()
        testm.model = eyt.get_remembered()

    # endregion

    tt.tick("testing")
    validresults = q.test_epoch(model=testm, dataloader=fvdl, losses=vmetrics, device=device)
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

    for metricarray, datasplit in zip([ftmetrics, ftvmetrics, ftxmetrics], ["train", "valid", "test"]):
        for metric in metricarray:
            settings[f"{datasplit}_{metric.name}"] = metric.get_epoch_error()

    # print(settings)
    return settings


def run_experiments(domain="restaurants", gpu=-1, patience=10, cosinelr=False,):
    ranges = {
        "lr": [0.0001, 0.00001], #[0.001, 0.0001, 0.00001],
        "ftlr": [0.0001],
        "enclrmul": [1., 0.1], #[1., 0.1, 0.01],
        "warmup": [2],
        "epochs": [100], #[50, 100],
        "numheads": [8, 12, 16],
        "numlayers": [3, 6, 9],
        "dropout": [.1],
        "hdim": [768, 960], #[192, 384, 768, 960],
        "seed": [12345678, 65748390], #, 98387670, 23655798, 66453829],     # TODO: add more later
    }
    p = __file__ + f".{domain}"
    def check_config(x):
        effectiveenclr = x["enclrmul"] * x["lr"]
        if effectiveenclr < 0.00001:
            return False
        dimperhead = x["hdim"] / x["numheads"]
        if dimperhead < 20 or dimperhead > 100:
            return False
        return True

    alldomains = {"recipes", "restaurants", "blocks", "calendar", "housing", "publications"}
    traindomains = ",".join(alldomains - {domain})

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config,
                      traindomains=traindomains, testdomain=domain,
                      gpu=gpu, patience=patience, cosinelr=cosinelr)


def run_experiments_seed(domain="restaurants", gpu=-1, patience=10, cosinelr=False,):
    ranges = {
        "lr": [0.0001],
        "enclrmul": [0.1],
        "warmup": [0],
        "epochs": [100],
        "numheads": [16],
        "numlayers": [3],
        "dropout": [.1],
        "hdim": [960],
        "seed": [12345678, 65748390, 98387670, 23655798, 66453829],     # TODO: add more later
    }
    p = __file__ + f".{domain}"
    def check_config(x):
        effectiveenclr = x["enclrmul"] * x["lr"]
        if effectiveenclr < 0.000005:
            return False
        dimperhead = x["hdim"] / x["numheads"]
        if dimperhead < 20 or dimperhead > 100:
            return False
        return True

    alldomains = {"recipes", "restaurants", "blocks", "calendar", "housing", "publications"}
    traindomains = ",".join(alldomains - {domain})

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config,
                      traindomains=traindomains, testdomain=domain,
                      gpu=gpu, patience=patience, cosinelr=cosinelr)



if __name__ == '__main__':
    # ret = q.argprun(run)
    # print(ret)
    q.argprun(run_experiments)
    # q.argprun(run_experiments_seed)