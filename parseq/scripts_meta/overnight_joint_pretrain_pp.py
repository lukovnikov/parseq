# encoding: utf-8
"""
A script for running the following zero-shot domain transfer experiments:
* dataset: Overnight
* model: BART encoder + vanilla Transformer decoder for LF
    * lexical token representations are computed based on lexicon
* training: normal (CE on teacher forced target)
"""
import faulthandler
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
from nltk import Tree
from torch.utils.data import DataLoader

from parseq.datasets import OvernightDatasetLoader, pad_and_default_collate, autocollate, Dataset
from parseq.decoding import merge_metric_dicts
from parseq.eval import SeqAccuracies, TreeAccuracy, make_array_of_metrics, CELoss
from parseq.grammar import tree_to_lisp_tokens, lisp_to_tree
from parseq.vocab import SequenceEncoder, Vocab
from transformers import AutoTokenizer, AutoModel, BartConfig, BartModel, BartForConditionalGeneration
from transformers.modeling_bart import SinusoidalPositionalEmbedding

UNKID = 3

DATA_RESTORE_REVERSE = False


def get_labels_from_tree(x:Tree):
    ret = {x.label()}
    for child in x:
        ret |= get_labels_from_tree(child)
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
            exampletokens = set(example[1])
            # exampletokens = set(get_labels_from_tree(example[1]))
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
    print(f"{len(out)}/{len(examples)} examples loaded from domain")
    return out


def get_lf_abstract_transform(examples):
    """
    Receives examples from different domains in the format (_, out_tokens, split, domain).
    Returns a function that transforms a sequence of domain-specific output tokens
        into a sequence of domain-independent tokens, abstracting domain-specific tokens/subtrees.
    :param examples:
    :return:
    """
    # get shared vocabulary
    domainspertoken = {}
    domains = set()
    for i, example in enumerate(examples):
        if "train" in example[2]:
            exampletokens = set(example[1])
            for token in exampletokens:
                if token not in domainspertoken:
                    domainspertoken[token] = set()
                domainspertoken[token].add(example[3])
            domains.add(example[3])

    sharedtokens = set([k for k, v in domainspertoken.items() if len(v) == len(domains)])
    replacement = "@ABS@"

    def example_transform(x):
        abslf = [xe if xe in sharedtokens else replacement for xe in x]
        return abslf

    return example_transform


def load_ds(traindomains=("restaurants",),
            testdomain="housing",
            min_freq=1,
            mincoverage=1,
            top_k=np.infty,
            nl_mode="bert-base-uncased",
            fullsimplify=False,
            onlyabstract=False,
            pretrainsetting="all+lex",    # "all", "lex" or "all+lex"
            finetunesetting="lex",        # "lex", "all", "min"
            ):
    """
    :param traindomains:
    :param testdomain:
    :param min_freq:
    :param mincoverage:
    :param top_k:
    :param nl_mode:
    :param fullsimplify:
    :param add_domain_start:
    :param onlyabstract:
    :param pretrainsetting:     "all": use all examples from every domain
                                "lex": use only lexical examples
                                "all+lex": use both
    :param finetunesetting:     "lex": use lexical examples
                                "all": use all training examples
                                "min": use minimal lexicon-covering set of examples
                            ! Test is always over the same original test set.
                            ! Validation is over a fraction of training data
    :return:
    """
    general_tokens = {
        "(", ")", "arg:~type", "arg:type", "op:and", "SW:concat", "cond:has",
        "arg:<=", "arg:<", "arg:>=", "arg:>", "arg:!=", "arg:=", "SW:superlative",
        "SW:CNT-arg:min", "SW:CNT-arg:<", "SW:CNT-arg:<=", "SW:CNT-arg:>=", "SW:CNT-arg:>",
        "SW:CNT-arg:max", "SW:CNT-arg:=", "arg:max",
    }

    def tokenize_and_add_start(t):
        tokens = tree_to_lisp_tokens(t)
        starttok = "@START@"
        tokens = [starttok] + tokens
        return tokens

    sourceex = []
    for traindomain in traindomains:
        ds = OvernightDatasetLoader(simplify_mode="light" if not fullsimplify else "full", simplify_blocks=True,
                                    restore_reverse=DATA_RESTORE_REVERSE, validfrac=.10)\
            .load(domain=traindomain)
        sourceex += ds[(None, None, lambda x: x in ("train", "valid", "lexicon"))].map(lambda x: (x[0], x[1], x[2], traindomain)).examples       # don't use test examples

    testds = OvernightDatasetLoader(simplify_mode="light" if not fullsimplify else "full", simplify_blocks=True, restore_reverse=DATA_RESTORE_REVERSE)\
        .load(domain=testdomain)

    targetex = testds.map(lambda x: x + (testdomain,)).examples

    pretrainex = []
    if "all" in pretrainsetting.split("+"):
        pretrainex += [(a, tokenize_and_add_start(b), "pretrain", d) for a, b, c, d in sourceex if c == "train"]
    if "lex" in pretrainsetting.split("+"):
        pretrainex += [(a, tokenize_and_add_start(b), "pretrain", d) for a, b, c, d in sourceex if c == "lexicon"]

    pretrainvalidex = [(a, tokenize_and_add_start(b), "pretrainvalid", d) for a, b, c, d in sourceex if c == "valid"]

    if finetunesetting == "all":
        finetunetrainex = [(a, tokenize_and_add_start(b), "fttrain", d) for a, b, c, d in targetex if c == "train"]
    elif finetunesetting == "lex":
        finetunetrainex = [(a, tokenize_and_add_start(b), "fttrain", d) for a, b, c, d in targetex if c == "lexicon"]
    elif finetunesetting == "min":
        finetunetrainex = get_maximum_spanning_examples([(a, b, c, d) for a, b, c, d in targetex if c == "train"],
                                      mincoverage=mincoverage,
                                      loadedex=[e for e in pretrainex if e[2] == "pretrain"])
        finetunetrainex = [(a, tokenize_and_add_start(b), "fttrain", d) for a, b, c, d in finetunetrainex]
    finetunevalidex = [(a, tokenize_and_add_start(b), "ftvalid", d) for a, b, c, d in targetex if c == "valid"]
    finetunetestex = [(a, tokenize_and_add_start(b), "fttest", d) for a, b, c, d in targetex if c == "test"]
    print(f"Using mode \"{finetunesetting}\" for finetuning data: "
          f"\n\t{len(finetunetrainex)} training examples")


    allex = pretrainex + pretrainvalidex + finetunetrainex + finetunevalidex + finetunetestex
    ds = Dataset(allex)

    if onlyabstract:
        et = get_lf_abstract_transform(ds[lambda x: x[3] != testdomain].examples)
        ds = ds.map(lambda x: (x[0], et(x[1]), x[2], x[3]))

    seqenc_vocab = Vocab(padid=0, startid=1, endid=2, unkid=UNKID)
    seqenc = SequenceEncoder(vocab=seqenc_vocab, tokenizer=lambda x: x,
                             add_start_token=False, add_end_token=True)
    for example in ds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=example[2] in ("pretrain", "fttrain"))
    seqenc.finalize_vocab(min_freq=min_freq, top_k=top_k)

    generaltokenmask = torch.zeros(seqenc_vocab.number_of_ids(), dtype=torch.long)
    for token, tokenid in seqenc_vocab.D.items():
        if token in general_tokens:
            generaltokenmask[tokenid] = 1

    nl_tokenizer = AutoTokenizer.from_pretrained(nl_mode)
    def tokenize(x):
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0],
               seqenc.convert(x[1], return_what="tensor"),
               x[2],
               x[0], x[1], x[3])
        return ret
    tds, ftds, vds, fvds, xds = ds[(None, None, "pretrain", None)].map(tokenize), \
                          ds[(None, None, "fttrain", None)].map(tokenize), \
                          ds[(None, None, "pretrainvalid", None)].map(tokenize), \
                          ds[(None, None, "ftvalid", None)].map(tokenize), \
                          ds[(None, None, "fttest", None)].map(tokenize)
    return tds, ftds, vds, fvds, xds, nl_tokenizer, seqenc, generaltokenmask


class BartGenerator(BartForConditionalGeneration):
    def __init__(self, config:BartConfig, emb=None, outlin=None, **kw):
        super(BartGenerator, self).__init__(config, **kw)
        if emb is not None:
            self.model.shared = emb
            self.model.decoder.embed_tokens = emb
        if outlin is not None:
            self.outlin = outlin
        else:
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
        ret = self.model.generate(input_ids,
                                  decoder_input_ids=output_ids[:, 0:1],
                                  attention_mask=input_ids!=self.model.config.pad_token_id,
                                  max_length=self.maxlen,
                                  num_beams=self.numbeam)
        outputs = [metric(None, ret[:, 1:], output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


def create_model(encoder_name="bert-base-uncased", resetmode="none",
                 dec_vocabsize=None, dec_layers=6, dec_dim=640, dec_heads=8, dropout=0., decoderdropout=0.,
                 maxlen=20, smoothing=0., numbeam=1, tensor2tree=None, generaltokenmask=None):
    if encoder_name != "bert-base-uncased":
        raise NotImplementedError(f"encoder '{encoder_name}' not supported yet.")
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
            ret = self.dropout(ret)
            ret = (ret, None, None)
            return ret

    encoder = BertEncoderWrapper(encoder, dropout=dropout)

    decoder_config = BartConfig(d_model=dec_dim,
                                pad_token_id=0,
                                bos_token_id=1,
                                vocab_size=dec_vocabsize,
                                decoder_attention_heads=dec_heads//2,
                                decoder_layers=dec_layers,
                                dropout=decoderdropout,
                                attention_dropout=min(0.1, dropout/2),
                                decoder_ffn_dim=dec_dim*4,
                                encoder_attention_heads=dec_heads,
                                encoder_layers=dec_layers,
                                encoder_ffn_dim=dec_dim*4,
                                )
    decoder_config.relative_position = False

    if resetmode == "none":
        emb, outlin = None, None
    else:
        emb = SpecialEmbedding(decoder_config.vocab_size,
                               decoder_config.d_model,
                               decoder_config.pad_token_id,
                               metarare_targets=generaltokenmask)
        outlin = SpecialOutlin(decoder_config.d_model,
                               decoder_config.vocab_size,
                               metarare_targets=generaltokenmask)

    model = BartGenerator(decoder_config, emb, outlin)
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
        i -= 1

    # convert to nltk.Tree
    try:
        tree, parsestate = lisp_to_tree(" ".join(x), None)
    except Exception as e:
        tree = None
    return tree


class SpecialEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 metarare_targets=None, init_std=0.02):
        super(SpecialEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.register_buffer("metarare_targets", metarare_targets)
        # self.metarare = self.weight[self.metarare_source, :]
        # self.base_emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.extra_emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.init_std
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # metarare_targets are 1 for domain-general tokens
        base_emb = super(SpecialEmbedding, self).forward(input)
        extra_emb = self.extra_emb(input)
        switch = (1-self.metarare_targets[input]).float()
        emb = switch[:, :, None] * extra_emb + (1 - switch[:, :, None]) * base_emb
        return emb


class SpecialOutlin(torch.nn.Linear):
    def __init__(self, dim, vocsize, metarare_targets=None, bias=True, init_std=0.02):
        super(SpecialOutlin, self).__init__(dim, vocsize, bias=bias)
        self.register_buffer("metarare_targets", metarare_targets)
        # self.metarare = self.weight[self.metarare_source, :]
        # self.base_emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.extra_lin = torch.nn.Linear(dim, vocsize, bias=bias)
        self.init_std = init_std
        self.apply(self._init_weights)
        # self.extra_lin.weight.data.fill_(0)
        # self.extra_lin.bias.data.fill_(0)

    def _init_weights(self, module):
        std = self.init_std
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_logits = super(SpecialOutlin, self).forward(input)
        extra_logits = self.extra_lin(input)
        switch = (1 - self.metarare_targets[None, None, :]).float()

        logits = switch * extra_logits + (1 - switch) * base_logits
        return logits


class Reinitializer(object):
    def __init__(self, model, resetafter=-1, resetevery=-1, numresets=0, resetgeneral=False, resetothers=tuple(), **kw):
        super(Reinitializer, self).__init__(**kw)
        self.model = model
        self.resetafter, self.resetevery, self.numresets = resetafter, resetevery, numresets
        self.count = 0
        self.counted_resets = 0
        self.resetgeneral = resetgeneral
        self.resetothers = resetothers

    def __call__(self):
        doreset = False
        if self.numresets > 0 and self.count >= self.resetafter:
            if self.count == self.resetafter:
                doreset = True
                self.counted_resets += 1
            elif self.counted_resets < self.numresets and (self.count - self.resetafter) % self.resetevery == 0:
                doreset = True
                self.counted_resets += 1
        if doreset:
            print("reinitializing domain-specific part of model")
            reset_special_inner(self.model, resetgeneral=self.resetgeneral)
            for resetother in self.resetothers:
                resetother.reset()
        self.count += 1 # advance counter


def reset_special_inner(m, resetgeneral=False):
    if resetgeneral:
        if isinstance(m, SpecialEmbedding):
            m.apply(m._init_weights)
        elif isinstance(m, SpecialOutlin):
            m.apply(m._init_weights)
        else:
            pass
        for child in m.children():
            reset_special_inner(child, resetgeneral=resetgeneral)
    else:
        if isinstance(m, SpecialEmbedding):
            m.extra_emb.apply(m._init_weights)
        elif isinstance(m, SpecialOutlin):
            m.extra_lin.apply(m._init_weights)
        else:
            pass
        for child in m.children():
            reset_special_inner(child, resetgeneral=resetgeneral)


def run(traindomains="ALL",
        domain="recipes",
        mincoverage=2,
        lr=0.001,
        enclrmul=0.1,
        numbeam=1,
        ftlr=0.0001,
        cosinelr=False,
        warmup=0.,
        batsize=30,
        pretrainbatsize=100,
        epochs=100,
        resetmode="none",
        pretrainepochs=100,
        minpretrainepochs=10,
        dropout=0.1,
        decoderdropout=0.5,
        wreg=1e-9,
        gradnorm=3,
        smoothing=0.,
        patience=5,
        gpu=-1,
        seed=123456789,
        encoder="bert-base-uncased",
        numlayers=6,
        hdim=600,
        numheads=8,
        maxlen=30,
        localtest=False,
        printtest=False,
        fullsimplify=True,
        nopretrain=False,
        onlyabstract=False,
        pretrainsetting="all",  # "all", "all+lex", "lex"
        finetunesetting="min",      # "lex", "all", "min"
        ):
    settings = locals().copy()
    print(json.dumps(settings, indent=4))

    numresets, resetafter, resetevery = 0, 0, 0
    if resetmode == "none":
        pass
    elif resetmode == "once":
        resetafter = 15
        resetevery = 5
        numresets = 1
    elif resetmode == "more":
        resetafter = 15
        resetevery = 5
        numresets = 3
    elif resetmode == "forever":
        resetafter = 15
        resetevery = 5
        numresets = 1000

    print(f'Resetting: "{resetmode}": {numresets} times, first after {resetafter} epochs, then every {resetevery} epochs')

    # wandb.init(project=f"overnight_joint_pretrain_fewshot_{pretrainsetting}-{finetunesetting}-{domain}",
    #            reinit=True, config=settings)
    if traindomains == "ALL":
        alldomains = {"recipes", "restaurants", "blocks", "calendar", "housing", "publications"}
        traindomains = alldomains - {domain, }
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tt = q.ticktock("script")
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt.tick("loading data")
    tds, ftds, vds, fvds, xds, nltok, flenc, generaltokenmask = \
        load_ds(traindomains=traindomains, testdomain=domain, nl_mode=encoder, mincoverage=mincoverage,
                fullsimplify=fullsimplify, onlyabstract=onlyabstract,
                pretrainsetting=pretrainsetting, finetunesetting=finetunesetting)
    tt.msg(f"{len(tds)/(len(tds) + len(vds)):.2f}/{len(vds)/(len(tds) + len(vds)):.2f} ({len(tds)}/{len(vds)}) train/valid")
    tt.msg(f"{len(ftds)/(len(ftds) + len(fvds) + len(xds)):.2f}/{len(fvds)/(len(ftds) + len(fvds) + len(xds)):.2f}/{len(xds)/(len(ftds) + len(fvds) + len(xds)):.2f} ({len(ftds)}/{len(fvds)}/{len(xds)}) fttrain/ftvalid/test")
    tdl = DataLoader(tds, batch_size=pretrainbatsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0))
    ftdl = DataLoader(ftds, batch_size=batsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0))
    vdl = DataLoader(vds, batch_size=pretrainbatsize, shuffle=False, collate_fn=partial(autocollate, pad_value=0))
    fvdl = DataLoader(fvds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=0))
    xdl = DataLoader(xds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=0))
    tt.tock("data loaded")

    tt.tick("creating model")
    trainm, testm = create_model(encoder_name=encoder,
                                 dec_vocabsize=flenc.vocab.number_of_ids(),
                                 dec_layers=numlayers,
                                 dec_dim=hdim,
                                 dec_heads=numheads,
                                 dropout=dropout,
                                 decoderdropout=decoderdropout,
                                 smoothing=smoothing,
                                 maxlen=maxlen,
                                 numbeam=numbeam,
                                 tensor2tree=partial(_tensor2tree, D=flenc.vocab),
                                 generaltokenmask=generaltokenmask,
                                 resetmode=resetmode
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

    if resetmode != "none":
        minpretrainepochs = resetafter + (numresets - 1) * resetevery
    eyt = q.EarlyStopper(vmetrics[1], patience=patience, min_epochs=minpretrainepochs,
                         more_is_better=True, remember_f=lambda: deepcopy(trainm.model))

    reinit = Reinitializer(trainm.model, resetafter=resetafter, resetevery=resetevery, numresets=numresets, resetothers=[eyt])

    # def wandb_logger():
    #     d = {}
    #     for name, loss in zip(["loss", "elem_acc", "seq_acc", "tree_acc"], metrics):
    #         d["train_"+name] = loss.get_epoch_error()
    #     for name, loss in zip(["seq_acc", "tree_acc"], vmetrics):
    #         d["valid_"+name] = loss.get_epoch_error()
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
    validepoch = partial(q.test_epoch, model=testm, dataloader=vdl, losses=vmetrics, device=device,
                         on_end=[lambda: eyt.on_epoch_end(), lambda: reinit()])#, lambda: wandb_logger()])

    if not nopretrain:
        tt.tick("pretraining")
        q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=pretrainepochs,
                       check_stop=[lambda: eyt.check_stop()])
        tt.tock("done pretraining")

    if eyt.get_remembered() is not None:
        tt.msg("reloaded")
        trainm.model = eyt.get_remembered()
        testm.model = eyt.get_remembered()

    # endregion

    # region finetune
    ftmetrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    ftvmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    ftxmetrics = make_array_of_metrics("seq_acc", "tree_acc")

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
    paramgroups = [{"params": encparams, "lr": ftlr * enclrmul},
                   {"params": otherparams}]

    ftoptim = torch.optim.Adam(paramgroups, lr=ftlr, weight_decay=wreg)

    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(trainm.parameters(), gradnorm)

    # def wandb_logger_ft():
    #     d = {}
    #     for name, loss in zip(["loss", "elem_acc", "seq_acc", "tree_acc"], ftmetrics):
    #         d["ft_train_" + name] = loss.get_epoch_error()
    #     for name, loss in zip(["seq_acc", "tree_acc"], ftvmetrics):
    #         d["ft_valid_" + name] = loss.get_epoch_error()
    #     wandb.log(d)

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
                         on_end=[])#, lambda: wandb_logger_ft()])

    tt.tick("finetuning")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done finetuning")

    # endregion

    tt.tick("testing")
    validresults = q.test_epoch(model=testm, dataloader=fvdl, losses=ftvmetrics, device=device)
    testresults = q.test_epoch(model=testm, dataloader=xdl, losses=ftxmetrics, device=device)
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

    # wandb.config.update(settings)
    # print(settings)
    return settings


def run_experiments(domain="restaurants", gpu=-1, patience=10, cosinelr=False, mincoverage=2, fullsimplify=True, uselexicon=False):
    ranges = {
        "lr": [0.0001, 0.00001], #[0.001, 0.0001, 0.00001],
        "ftlr": [0.00003],
        "enclrmul": [1., 0.1], #[1., 0.1, 0.01],
        "warmup": [2],
        "epochs": [100], #[50, 100],
        "pretrainepochs": [100],
        "numheads": [8, 12, 16],
        "numlayers": [3, 6, 9],
        "dropout": [.1],
        "hdim": [768, 960], #[192, 384, 768, 960],
        "seed": [12345678], #, 98387670, 23655798, 66453829],      # TODO: add more later
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

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config,
                      domain=domain, fullsimplify=fullsimplify, uselexicon=uselexicon,
                      gpu=gpu, patience=patience, cosinelr=cosinelr, mincoverage=mincoverage)


def run_experiments_seed(domain="default", gpu=-1, patience=10, cosinelr=False, fullsimplify=True, batsize=50,
                         smoothing=0.2, dropout=.1, decoderdropout=0.5, numlayers=3, numheads=12, hdim=768, pretrainbatsize=100,
                         resetmode="none", mincoverage=2,
                         nopretrain=False, numbeam=1, onlyabstract=False, pretrainsetting="all", finetunesetting="min",
                         epochs=67, pretrainepochs=60, minpretrainepochs=10):
    ranges = {
        "domain": ["recipes", "restaurants", "blocks", "calendar", "housing", "publications"],
        "lr": [0.0001],
        "ftlr": [0.0001],
        "enclrmul": [0.1],
        "warmup": [2],
        "epochs": [epochs],
        "pretrainepochs": [pretrainepochs],
        "numheads": [numheads],
        "numlayers": [numlayers],
        "dropout": [dropout],
        "decoderdropout": [decoderdropout],
        "smoothing": [smoothing],
        "hdim": [hdim],
        "numbeam": [numbeam],
        "batsize": [batsize],
        "seed": [12345678, 65748390, 98387670, 23655798, 66453829],     # TODO: add more later
    }
    p = __file__ + f".{domain}"
    if domain != "default":
        ranges["domain"] = [domain]
    def check_config(x):
        effectiveenclr = x["enclrmul"] * x["lr"]
        if effectiveenclr < 0.000005:
            return False
        dimperhead = x["hdim"] / x["numheads"]
        if dimperhead < 20 or dimperhead > 100:
            return False
        return True

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config, mincoverage=mincoverage,
                      domain=domain, fullsimplify=fullsimplify, resetmode=resetmode,
                      gpu=gpu, patience=patience, cosinelr=cosinelr, pretrainbatsize=pretrainbatsize,
                      pretrainsetting=pretrainsetting, finetunesetting=finetunesetting,
                      nopretrain=nopretrain, onlyabstract=onlyabstract, minpretrainepochs=minpretrainepochs)



if __name__ == '__main__':
    faulthandler.enable()
    # ret = q.argprun(run)
    # print(ret)
    # q.argprun(run_experiments)
    fire.Fire(run_experiments_seed)