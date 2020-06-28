# encoding: utf-8
"""
A script for running the following zero-shot domain transfer experiments:
* dataset: Overnight
* model: BART encoder + vanilla Transformer decoder for LF
    * lexical token representations are computed based on lexicon
* training: normal (CE on teacher forced target)
"""
import faulthandler
import itertools
import json
import math
import os
import random
import re
import shelve
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

from parseq.datasets import OvernightDatasetLoader, pad_and_default_collate, autocollate, Dataset, BatchDataset
from parseq.decoding import merge_metric_dicts
from parseq.eval import SeqAccuracies, TreeAccuracy, make_array_of_metrics, CELoss
from parseq.grammar import tree_to_lisp_tokens, lisp_to_tree
from parseq.vocab import SequenceEncoder, Vocab
from transformers import AutoTokenizer, AutoModel, BartConfig, BartModel, BartForConditionalGeneration, BertLayer, \
    BertModel
from transformers.activations import ACT2FN
from transformers.modeling_bart import SinusoidalPositionalEmbedding, DecoderLayer, SelfAttention, LayerNorm

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
        exampletokens = set(example[1])
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
    print(f"{len(out)}/{len(examples)} examples loaded from domain")
    return out


def get_lf_abstract_transform(examples, general_tokens=None):
    """
    Receives examples from different domains in the format (_, out_tokens, split, domain).
    Returns a function that transforms a sequence of domain-specific output tokens
        into a sequence of domain-independent tokens, abstracting domain-specific tokens/subtrees.
    :param examples:
    :return:
    """
    if general_tokens is not None:
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
    else:
        sharedtokens = set()
        sharedtokens.update(general_tokens)

    sharedtokens.add("@ABS@")
    sharedtokens.add("@ABSSTART@")
    sharedtokens.add("@END@")
    sharedtokens.add("@START@")
    sharedtokens.add("@META@")
    sharedtokens.add("@UNK@")
    sharedtokens.add("@PAD@")
    sharedtokens.add("@METARARE@")
    replacement = "@ABS@"

    def example_transform(x):
        abslf = [xe if xe in sharedtokens else replacement for xe in x]
        abslf = ["@ABSSTART@"] + abslf[1:]
        return abslf

    return example_transform


def tokenize_and_add_start(t, _domain, meta=False, add_domain_start=False, general_tokens=None):
    tokens = tree_to_lisp_tokens(t)
    if general_tokens is not None:
        newtokens = []
        for token in tokens:
            if re.match("@[^@]+@", token) or token in general_tokens:
                newtokens.append(token)
            else:
                newtokens.append(f"{_domain}|{token}")
        tokens = newtokens
    starttok = f"@START/{_domain}@" if add_domain_start else "@START@"
    tokens = [starttok] + tokens
    # if not meta:
    #     starttok = f"@START/{_domain}@" if add_domain_start else "@START@"
    #     tokens = [starttok] + tokens
    # else:
    #     starttok = f"@META/{_domain}@" if add_domain_start else "@META@"
    #     tokens = [starttok] + tokens
    return tokens


def load_ds(traindomains=("restaurants",),
            testdomain="housing",
            min_freq=1,
            mincoverage=1,
            top_k=np.infty,
            batsize=10,
            ftbatsize=-1,
            nl_mode="bert-base-uncased",
            fullsimplify=False,
            add_domain_start=True,
            supportsetting="lex",   # "lex" or "min" or "train"
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
    ftbatsize = batsize if ftbatsize < 0 else ftbatsize
    general_tokens = {
        "(", ")", "arg:~type", "arg:type", "op:and", "SW:concat", "cond:has",
        "arg:<=", "arg:<", "arg:>=", "arg:>", "arg:!=", "arg:=", "SW:superlative",
        "SW:CNT-arg:min", "SW:CNT-arg:<", "SW:CNT-arg:<=", "SW:CNT-arg:>=", "SW:CNT-arg:>",
        "SW:CNT-arg:max", "SW:CNT-arg:=", "arg:max",
    }

    domains = {}
    alltrainex = []
    for domain in list(traindomains) + [testdomain]:
        ds = OvernightDatasetLoader(simplify_mode="light" if not fullsimplify else "full", simplify_blocks=True,
                                    restore_reverse=DATA_RESTORE_REVERSE, validfrac=.10)\
            .load(domain=domain)
        domainexamples = [(a, b, c) for a, b, c in ds.examples]
        if supportsetting == "lex":
            domainexamples = [(a, b, "finetune" if c == "lexicon" else c)
                              for a, b, c in domainexamples]
        else:
            domainexamples = [(a, b, c) for a, b, c in domainexamples if c != "lexicon"]
        if domain != testdomain:
            alltrainex += [(a, b, c, domain) for a, b, c in domainexamples if c == "train"]
        domains[domain] = domainexamples

    for domain in domains:
        domains[domain] = [(a, tokenize_and_add_start(b, domain, meta=c=="finetune", add_domain_start=add_domain_start,
                                                      general_tokens=general_tokens), c)
                           for a, b, c in domains[domain]]

    if supportsetting == "min" or supportsetting == "train":
        for domain, domainexamples in domains.items():
            mindomainexamples = get_maximum_spanning_examples([(a, b, c) for a, b, c in domainexamples if c == "train"],
                                          mincoverage=mincoverage, #loadedex=None)
                                          loadedex=[a for a in alltrainex if a[3] != domain])
            domains[domain] = domains[domain] + [(a, b, "finetune") for a, b, c in mindomainexamples]

    allex = []
    for domain in domains:
        allex += [(a, b, c, domain) for a, b, c in domains[domain]]
    ds = Dataset(allex)

    et = get_lf_abstract_transform(ds[lambda x: x[3] != testdomain].examples, general_tokens=general_tokens)
    ds = ds.map(lambda x: (x[0], x[1], et(x[1]), x[2], x[3]))

    seqenc_vocab = Vocab(padid=0, startid=1, endid=2, unkid=UNKID)
    seqenc_vocab.add_token("@ABS@", seen=np.infty)
    seqenc_vocab.add_token("@ABSSTART@", seen=np.infty)
    seqenc_vocab.add_token("@METARARE@", seen=np.infty)
    seqenc_vocab.add_token("@META@", seen=np.infty)
    seqenc = SequenceEncoder(vocab=seqenc_vocab, tokenizer=lambda x: x,
                             add_start_token=False, add_end_token=True)
    for example in ds.examples:
        seqenc.inc_build_vocab(example[1], seen=example[3] in ("train", "finetune") if example[4] != testdomain else example[3] == "finetune")
        seqenc.inc_build_vocab(example[2], seen=example[3] in ("train", "finetune") if example[4] != testdomain else example[3] == "finetune")
    seqenc.finalize_vocab(min_freq=min_freq, top_k=top_k)

    generaltokenmask = torch.zeros(seqenc_vocab.number_of_ids(), dtype=torch.long)
    for token, tokenid in seqenc_vocab.D.items():
        if token in general_tokens:
            generaltokenmask[tokenid] = 1

    tokenmasks = {"_general": generaltokenmask}
    metararemask = torch.zeros_like(generaltokenmask)

    for domain in domains:
        tokenmasks[domain] = torch.zeros_like(generaltokenmask)
        for token, tokenid in seqenc_vocab.D.items():
            if token.startswith(domain + "|"):
                tokenmasks[domain][tokenid] = 1
        metararemask += tokenmasks[domain]

    tokenmasks["_metarare"] = metararemask.clamp_max(1)

    abstractmask = torch.zeros_like(generaltokenmask)
    for token in ["@ABS@", "@ABSSTART@"]:
        abstractmask[seqenc_vocab[token]] = 1
    tokenmasks["_abstract"] = abstractmask

    specialmask = torch.zeros_like(generaltokenmask)
    for token, tokenid in seqenc_vocab.D.items():
        if re.match("@[^@]+@", token) and not token in ["@ABS@", "@ABSSTART@"]:
            specialmask[tokenid] = 1
    tokenmasks["_special"] = specialmask

    nl_tokenizer = AutoTokenizer.from_pretrained(nl_mode)
    def tokenize(x):
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0],
               seqenc.convert(x[1], return_what="tensor"),
               seqenc.convert(x[2], return_what="tensor"),
               x[3],
               x[0], x[1], x[2], x[3])
        return ret

    sourceret = {}
    targetret = {}
    for domain in domains:
        finetunekey = "finetune"
        if supportsetting == "train" and domain != testdomain:
            finetunekey = "train"
        finetuneds = ds[lambda x: x[3] == finetunekey and x[4] == domain].map(tokenize)
        trainds = ds[lambda x: x[3] == "train" and x[4] == domain].map(tokenize)
        validds = ds[lambda x: x[3] == "valid" and x[4] == domain].map(tokenize)
        testds = ds[lambda x: x[3] == "test" and x[4] == domain].map(tokenize)
        if domain == testdomain:
            ret = targetret
        else:
            ret = sourceret
        ret[domain] = {
            "finetune":DataLoader(finetuneds, batch_size=ftbatsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0)),
            "train": DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0)),
            "valid": DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=0)),
            "test": DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=0))
        }

    # populate the "all" domain
    allsourceret = {
        "finetune": DataLoader(ds[lambda x: x[3] == finetunekey and x[4] in traindomains].map(tokenize),
                               batch_size=ftbatsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0)),
        "train": DataLoader(ds[lambda x: x[3] == "train" and x[4] in traindomains].map(tokenize),
                            batch_size=batsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0)),
        "valid": DataLoader(ds[lambda x: x[3] == "valid" and x[4] in traindomains].map(tokenize),
                            batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=0)),
        "test": DataLoader(ds[lambda x: x[3] == "test" and x[4] in traindomains].map(tokenize),
                           batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=0)),
    }
    return sourceret, targetret, allsourceret, nl_tokenizer, seqenc, tokenmasks


def apply_withpath(m:torch.nn.Module, fn:Callable, mpath=None):
    """ Apply function 'fn' recursively on 'm' and its submodules, where 'fn' gets 'm' and 'mpath' as argument """
    fn(m, mpath)
    for name, child in m.named_children():
        apply_withpath(child, fn, f"{mpath}.{name}" if mpath is not None else f"{name}")



class TransformerLayerAdapter(torch.nn.Module):
    def __init__(self, dim, hdim, **kw):
        super(TransformerLayerAdapter, self).__init__(**kw)
        self.fc1 = torch.nn.Linear(dim, hdim)
        self.fc2 = torch.nn.Linear(hdim, dim)
        self.layernorm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        innerresidual = x
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = x + innerresidual
        x = self.layernorm(x)
        return x


class GatedTransformerLayerAdapter(TransformerLayerAdapter):
    def __init__(self, dim, hdim, biasoffset=-3.3, **kw):
        super(GatedTransformerLayerAdapter, self).__init__(dim, hdim, **kw)
        self.fc3 = torch.nn.Linear(hdim, dim)
        self.biasoffset = biasoffset

    def forward(self, x):
        innerresidual = x
        h = self.fc1(x)
        h = torch.relu(h)
        x = self.fc2(h)
        z = self.fc3(h)
        z = torch.sigmoid(z + self.biasoffset)
        x = x * z + innerresidual * (1 - z)
        x = self.layernorm(x)
        return x


class AdaptedBartDecoderLayer(torch.nn.Module):
    def __init__(self, decoderlayer:DecoderLayer=None, compression=2, ):
        super().__init__()
        self.core = decoderlayer
        self.adapter = GatedTransformerLayerAdapter(self.core.embed_dim, self.core.embed_dim//compression)

    def forward(
            self,
            x,
            encoder_hidden_states,
            encoder_attn_mask=None,
            layer_state=None,
            causal_mask=None,
            decoder_padding_mask=None,
    ):
        x, self_attn_weights, layer_state = self.core(x, encoder_hidden_states,
            encoder_attn_mask=encoder_attn_mask, layer_state=layer_state, causal_mask=causal_mask,
            decoder_padding_mask=decoder_padding_mask)
        x = self.adapter(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class AdaptedBertEncoderLayer(torch.nn.Module):
    def __init__(self, core:BertLayer, compression=2):
        super(AdaptedBertEncoderLayer, self).__init__()
        self.core = core
        self.dim = core.output.dense.out_features
        self.hdim = self.dim // compression
        self.adapter = GatedTransformerLayerAdapter(self.dim, self.hdim)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        outputs = self.core(hidden_states, attention_mask=attention_mask, head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask)
        adapted = self.adapter(outputs[0])
        outputs = (adapted,) + outputs[1:]
        return outputs


class BartGenerator(BartForConditionalGeneration):
    def __init__(self, config:BartConfig, emb=None, outlin=None, tokenmasks=None):
        super(BartGenerator, self).__init__(config)
        if emb is not None:
            self.model.shared = emb
            self.model.decoder.embed_tokens = emb
        if outlin is not None:
            self.outlin = outlin
        else:
            self.outlin = torch.nn.Linear(config.d_model, config.vocab_size)
            self.outlin.apply(self._init_weights)

        for tmkey in tokenmasks:
            if tmkey in ["_general", "_metarare", "_special"]:
                pass
            else:
                tm = (tokenmasks[tmkey] + tokenmasks["_general"] + tokenmasks["_special"]).clamp_max(1)
                self.register_buffer(f"tokenmask_{tmkey}", tm)

    def _init_weights(self, module):
        std = self.config.init_std
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

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
            tokenmask=None,
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
        if tokenmask is not None: # apply token mask
            tokenmask = getattr(self, f"tokenmask_{tokenmask}")
            lm_logits += torch.log(tokenmask[None, None, :].float())
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
        return outputs


class BartGeneratorTrain(torch.nn.Module):
    def __init__(self, model:BartGenerator, smoothing=0., tensor2tree:Callable=None, orderless:Set[str]=set(),
                 maxlen:int=100, numbeam:int=1, **kw):
        super(BartGeneratorTrain, self).__init__(**kw)
        self.model = model

        # CE loss
        self.ce = CELoss(ignore_index=model.config.pad_token_id, smoothing=smoothing)

        # accuracies
        self.accs = SeqAccuracies()
        self.accs.padid = model.config.pad_token_id
        self.accs.unkid = UNKID

        self.tensor2tree = tensor2tree
        self.orderless = orderless
        self.maxlen, self.numbeam = maxlen, numbeam
        self.treeacc = TreeAccuracy(tensor2tree=tensor2tree,
                                    orderless=orderless)

        self.metrics = [self.ce, self.accs, self.treeacc]
        self.current_tokenmask = None

    def forward(self, input_ids, output_ids, *args, **kwargs):
        ret = self.model(input_ids, attention_mask=input_ids!=self.model.config.pad_token_id,
                         decoder_input_ids=output_ids[:, :-1], tokenmask=self.current_tokenmask)
        probs = ret[0]
        _, predactions = probs.max(-1)
        outputs = [metric(probs, predactions, output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret

    def get_test_model(self, maxlen:int=None, numbeam:int=None):
        maxlen = self.maxlen if maxlen is None else maxlen
        numbeam = self.numbeam if numbeam is None else numbeam
        ret = BartGeneratorTest(self.model, maxlen=maxlen, numbeam=numbeam,
                                tensor2tree=self.tensor2tree, orderless=self.orderless)
        return ret


class AbstractBartGeneratorTrain(torch.nn.Module):
    def __init__(self, model:BartGeneratorTrain, **kw):
        super(AbstractBartGeneratorTrain, self).__init__(**kw)
        self.model = model

    def forward(self, input_ids, _, output_ids, *args, **kwargs):
        ret = self.model(input_ids, output_ids, *args, tokenmask="_abstract", **kwargs)
        return ret


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
        self.current_tokenmask = None

    def forward(self, input_ids, output_ids, *args, **kwargs):
        ret = self.model.generate(input_ids,
                                  decoder_input_ids=output_ids[:, 0:1],
                                  attention_mask=input_ids!=self.model.config.pad_token_id,
                                  max_length=self.maxlen,
                                  num_beams=self.numbeam,
                                  tokenmask=self.current_tokenmask)
        outputs = [metric(None, ret[:, 1:], output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


class SpecialEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 metarare_targets=None, init_std=0.02, nospecialshared=False):
        super(SpecialEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.register_buffer("metarare_targets", metarare_targets)
        # self.metarare = self.weight[self.metarare_source, :]
        # self.base_emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.extra_emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.metarare_emb = torch.nn.Embedding(1, embedding_dim)
        self.init_std = init_std
        self.apply(self._init_weights)
        # self.extra_emb.weight.data.fill_(0)
        self.nospecialshared = nospecialshared

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
        # metarare_targets are 1 for domain-specific tokens
        base_emb = super(SpecialEmbedding, self).forward(input)
        metarare_emb = self.metarare_emb(torch.zeros_like(input))
        extra_emb = self.extra_emb(input)
        switch = self.metarare_targets[input].float()
        if self.nospecialshared:
            emb = switch[:, :, None] * extra_emb + (1 - switch[:, :, None]) * base_emb
        else:
            emb = switch[:, :, None] * (extra_emb + metarare_emb) \
                  + (1 - switch[:, :, None]) * base_emb
        return emb


class SpecialOutlin(torch.nn.Linear):
    def __init__(self, dim, vocsize, metarare_targets=None, bias=True, init_std=0.02, nospecialshared=False):
        super(SpecialOutlin, self).__init__(dim, vocsize, bias=bias)
        self.register_buffer("metarare_targets", metarare_targets)
        # self.metarare = self.weight[self.metarare_source, :]
        # self.base_emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.extra_lin = torch.nn.Linear(dim, vocsize, bias=bias)
        self.metarare_lin = torch.nn.Linear(dim, 1, bias=bias)
        self.init_std = init_std
        self.apply(self._init_weights)
        # self.extra_lin.weight.data.fill_(0)
        # self.extra_lin.bias.data.fill_(0)
        self.nospecialshared = nospecialshared

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
        metarare_logits = self.metarare_lin(input)
        switch = self.metarare_targets[None, None, :].float()

        if self.nospecialshared:
            logits = switch * extra_logits + (1 - switch) * base_logits
        else:
            logits = switch * (extra_logits + metarare_logits) + (1 - switch) * base_logits
        return logits


def create_model(encoder_name="bert-base-uncased",
                 dec_vocabsize=None, dec_layers=6, dec_dim=640, dec_heads=8, dropout=0.,
                 maxlen=20, smoothing=0., numbeam=1, tensor2tree=None,
                 tokenmasks=None,
                 metarare="no", useadapters=False, nospecialshared=False):
    if encoder_name != "bert-base-uncased":
        raise NotImplementedError(f"encoder '{encoder_name}' not supported yet.")
    pretrained = BertModel.from_pretrained(encoder_name)
    # replace layers with adapted layers
    if useadapters:
        for i, layer in enumerate(pretrained.encoder.layer):
            pretrained.encoder.layer[i] = AdaptedBertEncoderLayer(layer, compression=4)
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
                                eos_token_id=2,
                                vocab_size=dec_vocabsize,
                                decoder_attention_heads=dec_heads//2,
                                decoder_layers=dec_layers,
                                dropout=dropout,
                                attention_dropout=min(0.1, dropout/2),
                                decoder_ffn_dim=dec_dim*4,
                                encoder_attention_heads=dec_heads,
                                encoder_layers=dec_layers,
                                encoder_ffn_dim=dec_dim*4,
                                )

    # create special embeddings and output layer
    if metarare == "no":
        emb, outlin = None, None
    else:
        if "emb" in metarare.split("+") or metarare == "yes":
            print("using metarare emb")
            # emb = torch.nn.Embedding(decoder_config.vocab_size, decoder_config.d_model, decoder_config.pad_token_id)
            emb = SpecialEmbedding(decoder_config.vocab_size,
                                   decoder_config.d_model,
                                   decoder_config.pad_token_id,
                                   metarare_targets=tokenmasks["_metarare"],
                                   nospecialshared=nospecialshared)
        else:
            emb = None
        if "outlin" in metarare.split("+") or metarare == "yes":
            print("using metarare outlin")
            # outlin = torch.nn.Linear(decoder_config.d_model, decoder_config.vocab_size)
            outlin = SpecialOutlin(decoder_config.d_model,
                                   decoder_config.vocab_size,
                                   metarare_targets=tokenmasks["_metarare"],
                                   nospecialshared=nospecialshared)
        else:
            outlin = None
        #
        # def _init_weights(module):
        #     std = 0.02
        #     if isinstance(module, torch.nn.Linear):
        #         module.weight.data.normal_(mean=0.0, std=std)
        #         if module.bias is not None:
        #             module.bias.data.zero_()
        #     elif isinstance(module, SinusoidalPositionalEmbedding):
        #         pass
        #     elif isinstance(module, torch.nn.Embedding):
        #         module.weight.data.normal_(mean=0.0, std=std)
        #         if module.padding_idx is not None:
        #             module.weight.data[module.padding_idx].zero_()
        # emb.apply(_init_weights)
        # outlin.apply(_init_weights)
    #     print("using special embs and linouts")
    # else:
    #     emb = torch.nn.Embedding(decoder_config.vocab_size, decoder_config.d_model, decoder_config.pad_token_id)
    #     outlin = torch.nn.Linear(decoder_config.d_model, decoder_config.vocab_size)
    #     emb = None
    #     outlin = None

    model = BartGenerator(decoder_config, emb, outlin, tokenmasks=tokenmasks)
    model.model.encoder = encoder
    if useadapters:
        for i, layer in enumerate(model.model.decoder.layers):
            model.model.decoder.layers[i] = AdaptedBartDecoderLayer(layer)

    orderless = {"op:and", "SW:concat"}

    trainmodel = BartGeneratorTrain(model, smoothing=smoothing, tensor2tree=tensor2tree, orderless=orderless,
                                    maxlen=maxlen, numbeam=numbeam)
    abstrainmodel = AbstractBartGeneratorTrain(trainmodel)
    # testmodel = BartGeneratorTest(model, maxlen=maxlen, numbeam=numbeam, tensor2tree=tensor2tree, orderless=orderless)
    return trainmodel, abstrainmodel


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


def move_grad(source=None, target=None):
    source_params = {k: v for k, v in source.named_parameters()}
    for k, v in target.named_parameters():
        assert(v.size() == source_params[k].size())
        if source_params[k].grad is not None:
            if v.grad is None:
                v.grad = source_params[k].grad
            else:
                v.grad += source_params[k].grad
    source.zero_grad()


def reset_special_grads_inner(_m:torch.nn.Module, mode="none"):
        # for paramname, param in _m.named_parameters():
        #     if paramname not in ["model.model.decoder.embed_tokens.extra_emb.weight",
        #                          "model.outlin.extra_lin.weight", "model.outlin.extra_lin.weight"]:
        #         param.grad = None
    if mode == "metarare":    # train everything
        pass
    elif mode == "extrasplit":
        for paramname, param in _m.named_parameters():
            isadapterparam = False
            isextraparam = False
            isspecialparam = False
            parent = None
            m = _m
            namesplits = paramname.split(".")
            for namepiece in namesplits:
                m = getattr(m, namepiece)
                if isinstance(m, TransformerLayerAdapter):
                    isadapterparam = True
                    break
                if isinstance(parent, (SpecialEmbedding, SpecialOutlin)):
                    isspecialparam = True
                    if namepiece == "extra_emb" or namepiece == "extra_lin":
                        isextraparam = True
                        break
                parent = m

            dotrain = False
            if isadapterparam:
                dotrain = dotrain or True
            if isextraparam:
                dotrain = dotrain or True
            if not dotrain:
                param.grad = None
    elif mode == "split" or mode == "metararetokensonly" \
            or "inner:onlyemb" in mode.split("+"):   # train only embeddings and output layer
        for paramname, param in _m.named_parameters():
            dotrain = False
            for e in ["model.model.decoder.embed_tokens", "model.outlin"]:
                if paramname.startswith(e):
                    dotrain = dotrain or True
            if not dotrain:
                param.grad = None
    elif mode == "adapter" or mode == "adaptersplit" or mode == "adapterinner":     # finetune only adapters and embeddings and output layers
        for paramname, param in _m.named_parameters():
            isadapterparam = False
            m = _m
            namesplits = paramname.split(".")
            for namepiece in namesplits:
                m = getattr(m, namepiece)
                if isinstance(m, TransformerLayerAdapter):
                    isadapterparam = True
                    break

            dotrain = False
            if isadapterparam:
                dotrain = dotrain or True
            else:
                for e in ["model.model.decoder.embed_tokens", "model.outlin"]:
                    if paramname.startswith(e):
                        dotrain = dotrain or True
            if not dotrain:
                param.grad = None
    elif mode == "MAS" or mode == "MASinner": # finetune only extra vectors, and adapters in the decoder
        for paramname, param in _m.named_parameters():
            isadapterparam = False
            isspecial = False
            isbertparam = False
            m = _m
            namesplits = paramname.split(".")
            for namepiece in namesplits:
                m = getattr(m, namepiece)
                if isinstance(m, TransformerLayerAdapter):
                    isadapterparam = True
                    break
                elif isinstance(m, BertModel):
                    isbertparam = True
                elif isinstance(m, (SpecialEmbedding, SpecialOutlin)):
                    isspecial = True
                    break
            isspecial = isspecial and "extra_emb" in namesplits or "extra_lin" in namesplits
            isdecoderadapterparam = isadapterparam and not isbertparam
            isoriginalbertparam = (not isadapterparam) and isbertparam
            if not isdecoderadapterparam and not isspecial:
                param.grad = None

    elif "inner:all" in mode.split("+"):
        pass

    # else:
    #     if isinstance(_m.model.model.decoder.embed_tokens, SpecialEmbedding):
    #         _m.model.model.decoder.embed_tokens.weight.grad = None
    #         _m.model.model.decoder.metarare_emb.weight.grad = None
    #     if isinstance(_m.model.outlin, SpecialOutlin):
    #         _m.model.outlin.weight.grad = None
    #         _m.model.outlin.metarare_lin.weight.grad = None
    #         if _m.model.outlin.bias is not None:
    #             _m.model.outlin.bias.grad = None
    #             _m.model.outlin.metarare_lin.bias.grad = None

def reset_special_inner(m):
    if isinstance(m, SpecialEmbedding):
        m.extra_emb.apply(m._init_weights)
    elif isinstance(m, SpecialOutlin):
        m.extra_lin.apply(m._init_weights)
    else:
        pass
    for child in m.children():
        reset_special_inner(child)

def reset_special_grads_outer(_m, mode="none"):
    # if mode == "metararetokensonly":
    #     if isinstance(_m.model.model.decoder.embed_tokens, SpecialEmbedding):
    #         _m.model.model.decoder.embed_tokens.extra_emb.weight.grad = None
    #     if isinstance(_m.model.outlin, SpecialOutlin):
    #         _m.model.outlin.extra_lin.weight.grad = None
    #         _m.model.outlin.extra_lin.bias.grad = None
    if mode == "metarare" or mode == "metararetokensonly" or mode=="extrasplit":    # train everything except Special layers's, extra vectors
        if isinstance(_m.model.model.decoder.embed_tokens, SpecialEmbedding):
            _m.model.model.decoder.embed_tokens.extra_emb.weight.grad = None
        elif isinstance(_m.model.outlin, SpecialOutlin):
            _m.model.outlin.extra_lin.weight.grad = None
            _m.model.outlin.extra_lin.bias.grad = None
    elif mode == "split" or "outer:noemb" in mode.split("+"):   # don't train any embeddings/output layer
        for paramname, param in _m.named_parameters():
            for e in ["model.model.decoder.embed_tokens", "model.outlin"]:
                if paramname.startswith(e):
                    param.grad = None
    elif mode == "adapter":         # don't train original bert weights
        for paramname, param in _m.named_parameters():
            isadapterparam = False
            m = _m
            namesplits = paramname.split(".")
            for namepiece in namesplits:
                m = getattr(m, namepiece)
                if isinstance(m, TransformerLayerAdapter):
                    isadapterparam = True
                    break

            dotrain = False
            if paramname.startswith("model.model.encoder"):
                if isadapterparam:
                    dotrain = dotrain or True
            else:
                dotrain = dotrain or True
            if not dotrain:
                param.grad = None
    elif mode == "adaptersplit":     # finetune only adapters and embeddings and output layers
        for paramname, param in _m.named_parameters():
            isadapterparam = False
            m = _m
            namesplits = paramname.split(".")
            for namepiece in namesplits:
                m = getattr(m, namepiece)
                if isinstance(m, TransformerLayerAdapter):
                    isadapterparam = True
                    break

            donttrain = False
            if paramname.startswith("model.model.encoder"):
                donttrain = donttrain or True    # don't train anything in encoder
            else:
                if isadapterparam:
                    donttrain = donttrain or True
                else:
                    for e in ["model.model.decoder.embed_tokens", "model.outlin"]:
                        if paramname.startswith(e):
                            donttrain = donttrain or True
            if donttrain:
                param.grad = None
    elif mode == "MAS": # finetune only general token vectors, decoder, and adapters in encoder
        for paramname, param in _m.named_parameters():
            isadapterparam = False
            isspecial = False
            isbertparam = False
            m = _m
            namesplits = paramname.split(".")
            for namepiece in namesplits:
                m = getattr(m, namepiece)
                if isinstance(m, TransformerLayerAdapter):
                    isadapterparam = True
                    break
                elif isinstance(m, BertModel):
                    isbertparam = True
                elif isinstance(m, (SpecialEmbedding, SpecialOutlin)):
                    isspecial = True
                    break
            isspecial = isspecial and "extra_emb" in namesplits or "extra_lin" in namesplits
            isdecoderadapterparam = isadapterparam and not isbertparam
            isoriginalbertparam = (not isadapterparam) and isbertparam
            if isdecoderadapterparam or isspecial or isoriginalbertparam:
                param.grad = None
    elif "outer:all" in mode.split("+") \
            or mode == "adapter" \
            or mode == "adapterinner" \
            or mode == "MASinner":
        pass


def infiter(a):
    while True:
        for ae in a:
            yield ae


def infiter2(a):
    while True:
        yield next(iter(a))


def cat_batches(*x, pad_value=0):
    y = list(zip(*x))
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.Tensor):
            y[i] = q.pad_tensors(yi, 1, pad_value)
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.Tensor):
            y[i] = torch.cat(yi, 0)
        elif isinstance(yi, tuple):
            _yi = yi[0]
            for yij in yi[1:]:
                _yi = _yi + yij
            y[i] = _yi
    return y


def meta_train_epoch(model=None,
                     absmodel=None,
                     data=None,
                         allsourcedata=None,
                         injecttraindata=False,
                     optim=None,
                     get_ft_model=None,
                     get_ft_optim=None,
                         ftbatsize=None,
                     losses=None,
                     abslosses=None,
                     ftlosses=None,
                     device=torch.device("cpu"),
                     tt=q.ticktock(" -"),
                current_epoch=0,
                     max_epochs=0,
                     finetunesteps=1,
                     outersteps=1,
                     gradmode="none",   # "none", "metarare", ...
                     on_start=tuple(),
                     on_end=tuple(),
                print_every_batch=False,
                     clipgradnorm=None,
                     outergradnorm=3,
                     innergradnorm=3,
                     gradacc=1,
                     abstract_contrib=0.,):
    """
    Performs an epoch of training on given model, with data from given dataloader, using given optimizer,
    with loss computed based on given losses.
    :param model:
    :param data: dictionary from domains to dicts of dataloaders
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
    for loss in losses:
        loss.push_epoch_to_history(epoch=current_epoch-1)
        loss.reset_agg()
        loss.loss.to(device)

    model.to(device)
    absmodel.to(device)

    [e() for e in on_start]

    q.epoch_reset(model)
    optim.zero_grad()
    numbatsperdomain = {k: len(data[k]["train"]) for k in data}
    totalnumtrainbats = sum(numbatsperdomain.values())
    probbatsperdomain = {k: numbatsperdomain[k] / totalnumtrainbats for k in numbatsperdomain}

    # iter-ize training dataloaders in data
    for k, v in data.items():
        v["_train"] = iter(v["train"])

    outerstep_i = 0
    outersteps_counter = outersteps
    while True:
        outerbatch = None
        do_inner = False
        exhausted_domains = set()
        while outerbatch is None:
            assert(outersteps_counter <= outersteps)
            if outersteps_counter == outersteps:   # switch domain only every 'outersteps' steps
                ks, vs = zip(*probbatsperdomain.items())
                chosendomain = np.random.choice(ks, p=vs)
                outersteps_counter = 0
                do_inner = True
            try:
                outerbatch = next(data[chosendomain]["_train"])
                outersteps_counter += 1
                # outerbatch["tokenmask"] = chosendomain
            except StopIteration as e:
                # print(f"stopping iteration - outerstep_i: {outerstep_i}")
                exhausted_domains.add(chosendomain)
                outerbatch = None
                if len(exhausted_domains) == len(data):
                    break
                if outersteps_counter != outersteps:
                    outersteps_counter = outersteps


        if outerbatch is None:
            break

        # perform K number of inner steps
        inneriter = infiter2(data[chosendomain]["finetune"])
        # extra_inneriter = infiter2(allsourcedata["train"])

        # oldemb = ftmodel.model.model.decoder.embed_tokens.weight + 0
        # oldlin = ftmodel.model.outlin.weight + 0

        for loss in ftlosses:
            loss.push_epoch_to_history(epoch=str(current_epoch - 1)+"."+chosendomain)
            loss.reset_agg()
            loss.loss.to(device)

        ftmodel = get_ft_model(model)
        if do_inner and finetunesteps > 0:
            ftoptim = get_ft_optim(ftmodel)
            innerbatch = next(inneriter)
            # create a dataloader from this batch
            innerbatchdl = DataLoader(BatchDataset(innerbatch),
                batch_size=ftbatsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0))
            innerbatchiter = infiter2(innerbatchdl)
            for innerstep_i in range(finetunesteps):
                _innerbatch = next(innerbatchiter)
                if injecttraindata:
                    assert(False)
                    # extra_innerbatch = next(extra_inneriter)
                    # innerbatch = cat_batches(innerbatch, extra_innerbatch)
                # innerbatch["tokenmask"] = chosendomain
                ftmodel.current_tokenmask = chosendomain
                ttmsg = q.train_batch(batch=_innerbatch, model=ftmodel, optim=ftoptim, losses=ftlosses, device=device,
                                      batch_number=innerstep_i, max_batches=finetunesteps, current_epoch=current_epoch,
                                      max_epochs=max_epochs,
                                      on_before_optim_step=[
                                          lambda: clipgradnorm(_m=ftmodel, _norm=innergradnorm),
                                          partial(reset_special_grads_inner, _m=ftmodel, mode=gradmode)])
                if print_every_batch:
                    tt.msg(ttmsg)
                else:
                    tt.live(ttmsg)
        # after K inner updates
        # perform outer update on main model weights
        # do outer update:
        #   1. obtain gradient on inner-updated model using outerbatch,
        #   2. apply gradient on main model weights
        ftmodel.current_tokenmask = chosendomain
        ttmsg = q.train_batch(batch=outerbatch, model=ftmodel, optim=None, losses=losses, device=device,
                             batch_number=outerstep_i, max_batches=totalnumtrainbats, current_epoch=current_epoch,
                             max_epochs=max_epochs, gradient_accumulation_steps=gradacc)
                            # , on_before_optim_step=[
                            #     partial(clipgradnorm, _m=model),
                            #     partial(copy_grad, source=ftmodel, target=model)])
        move_grad(ftmodel, model)

        # do abstract prediction
        if abstract_contrib > 0.:
            abs_ttmsg = q.train_batch(batch=outerbatch, model=absmodel, optim=None, losses=abslosses, device=device,
                                      batch_number=outerstep_i, max_batches=totalnumtrainbats, current_epoch=current_epoch,
                                      max_epochs=max_epochs, gradient_accumulation_steps=gradacc,
                                      loss_scale=abstract_contrib)
        else:
            abs_ttmsg = "N/A"

        reset_special_grads_outer(model, mode=gradmode)

        clipgradnorm(_m=model, _norm=outergradnorm)

        # do optim step
        _do_optim_step = ((outerstep_i+1) % gradacc) == 0
        _do_optim_step = _do_optim_step or (outerstep_i+1) == totalnumtrainbats  # force optim step at the end of epoch
        if _do_optim_step:
            optim.step()
            optim.zero_grad()

        if print_every_batch:
            tt.msg(ttmsg + " -- " + abs_ttmsg)
        else:
            tt.live(ttmsg + " -- " + abs_ttmsg)

        outerstep_i += 1
    tt.stoplive()
    [e() for e in on_end]
    ttmsg = q.pp_epoch_losses(*losses) + " -- " + q.pp_epoch_losses(*abslosses)
    return ttmsg


def meta_test_epoch(model=None,
                    data=None,
                         allsourcedata=None,
                         injecttraindata=False,
                     get_ft_model=None,
                     get_ft_optim=None,
                    ftbatsize=None,
                    gradmode="none",
                    losses=None,
                    ftlosses=None,
                    finetunesteps=1,
                    bestfinetunestepsvar=None,
                    bestfinetunestepswhichmetric=None,
                    bestfinetunelowerisbetter=False,
                    evalinterval=-1,
                    mode="valid", # "valid" or "test"
                    device=torch.device("cpu"),
                    clipgradnorm=None,
            current_epoch=0, max_epochs=0, print_every_batch=False,
            on_start=tuple(), on_start_batch=tuple(), on_end_batch=tuple(), on_end=tuple(),
                    on_outer_start=tuple(), on_outer_end=tuple()):
    if evalinterval < 0:
        evalinterval = 1
    """
    Performs a test epoch. If run=True, runs, otherwise returns partially filled function.
    :param model:
    :param dataloader:
    :param losses:
    :param device:
    :param current_epoch:
    :param max_epochs:
    :param on_start:
    :param on_start_batch:
    :param on_end_batch:
    :param on_end:
    :return:
    """
    tt = q.ticktock(" -")
    model.to(device)
    q.epoch_reset(model)
    [e() for e in on_outer_start]

    lossesperdomain = {}
    stepperevals = []

    for domain in data:
        stepperevals.append([])
        lossesperdomain[domain] = []
        # doing one domain
        domaindata = data[domain]
        # perform fine-tuning (with early stopping if valid is given
        ftmodel = get_ft_model(model)
        ftoptim = get_ft_optim(ftmodel)
        ftmodel.train()
        inneriter = infiter2(domaindata["finetune"])
        extra_inneriter = infiter2(allsourcedata["train"])

        for loss in ftlosses:
            loss.push_epoch_to_history(epoch=str(current_epoch - 1)+"."+domain)
            loss.reset_agg()
            loss.loss.to(device)

        innerbatch = next(inneriter)
        # create a dataloader from this batch
        innerbatchdl = DataLoader(BatchDataset(innerbatch),
            batch_size=ftbatsize, shuffle=True, collate_fn=partial(autocollate, pad_value=0))
        innerbatchiter = infiter2(innerbatchdl)
        for innerstep_i in range(finetunesteps):
            _innerbatch = next(innerbatchiter)
            if injecttraindata:
                assert(False)
                # extra_innerbatch = next(extra_inneriter)
                # innerbatch = cat_batches(innerbatch, extra_innerbatch)
            ftmodel.current_tokenmask = domain
            # innerbatch["tokenmask"] = domain
            ttmsg = q.train_batch(batch=_innerbatch, model=ftmodel, optim=ftoptim, losses=ftlosses, device=device,
                                  batch_number=innerstep_i, max_batches=finetunesteps, current_epoch=current_epoch, max_epochs=max_epochs,
                                  on_before_optim_step=[partial(clipgradnorm, _m=ftmodel),
                                                        partial(reset_special_grads_inner, _m=ftmodel, mode=gradmode)])
            if print_every_batch:
                tt.msg(ttmsg)
            else:
                tt.live(ttmsg)

            test_ftmodel = ftmodel.get_test_model()

            if ((innerstep_i+1) % evalinterval == 0): #"(innerstep_i+1 == finetunesteps):
                _losses = deepcopy(losses)
                dataname = "valid" if mode == "valid" else "test"
                q.test_epoch(test_ftmodel, dataloader=domaindata[dataname], losses=_losses, device=device,
                             current_epoch=current_epoch, max_epochs=max_epochs, print_every_batch=print_every_batch,
                             on_start=on_start, on_end=on_end, on_start_batch=on_start_batch, on_end_batch=on_end_batch)
                lossesperdomain[domain].append(_losses)
                stepperevals[-1].append(innerstep_i)

    # find best number of steps
    metricsmatrix = np.zeros((len(lossesperdomain), math.ceil(finetunesteps / evalinterval), len(losses)))
    for i, domain in enumerate(sorted(lossesperdomain.keys())):
        for j, steplosses in enumerate(lossesperdomain[domain]):
            for k, lossval in enumerate(steplosses):
                metricsmatrix[i, j, k] = lossval.get_epoch_error()
    metricsmatrix = metricsmatrix.mean(0)   # (numevals, numlosses)
    if mode == "valid":
        critvals = metricsmatrix[:, bestfinetunestepswhichmetric]   # (numevals)
        critvals = critvals * (1 if bestfinetunelowerisbetter is False else -1)
        k = np.argmax(critvals)
        evalstep = stepperevals[0][k]
        bestfinetunestepsvar.v = evalstep
    else:
        print("metricsmatrix:")
        print(metricsmatrix)
        evalstep = q.v(bestfinetunestepsvar)
        k = math.floor(q.v(bestfinetunestepsvar) / evalinterval)

    for loss, _loss in zip(losses, metricsmatrix[k, :]):
        loss.epoch_agg_values.append(_loss)
        loss.epoch_agg_sizes.append(1)

    tt.stoplive()
    [e() for e in on_outer_end]
    ttmsg = q.pp_epoch_losses(*losses) + f" [@{evalstep+1}]"
    return ttmsg


class Reinitializer(object):
    def __init__(self, model, interval, **kw):
        super(Reinitializer, self).__init__(**kw)
        self.model = model
        self.interval = interval
        self.count = 1

    def __call__(self):
        if self.interval >= 1:
            if self.count % self.interval == 0:
                print("reinitializing domain-specific part of model")
                reset_special_inner(self.model)
                self.count = 1  # reset counter
            else:
                self.count += 1 # advance counter


def run(traindomains="ALL",
        domain="restaurants",
        mincoverage=2,
        lr=0.0001,
        enclrmul=0.1,
        numbeam=1,
        ftlr=0.0001,
        cosinelr=False,
        warmup=0.,
        batsize=30,
        ftbatsize=-1,
        supportsize=-1,
        epochs=100,
        finetunesteps=5,
        outersteps=1,
        maxfinetunesteps=4,
        evalinterval=2,
        testevalinterval=1,
        dropout=0.1,
        wreg=1e-9,
        gradnorm=3,
        ftgradnorm=-1,
        gradacc=1,
        smoothing=0.,
        patience=20,
        gpu=-1,
        seed=123456789,
        encoder="bert-base-uncased",
        numlayers=6,
        hdim=600,
        numheads=8,
        maxlen=30,
        fullsimplify=True,
        domainstart=False,
        supportsetting="lex",   # "lex" or "min"
        metarare="no",
        abscontrib=1.,
        gradmode="none",    # "none", "metarare", "metarareonly", "split"
        injecttraindata=False,
        useadapters=False,
        resetspecialinner=False,
        reinitspecialinner=False,
        reinitinterval=0,
        nospecialshared=False,
        validinter=1,
        startmtafter=0,
        ):
    settings = locals().copy()
    print(json.dumps(settings, indent=4))
    ftgradnorm = gradnorm if ftgradnorm < 0 else ftgradnorm
    ftbatsize = batsize if ftbatsize < 0 else ftbatsize
    supportsize = batsize if supportsize < 0 else supportsize
    # wandb.init(project=f"overnight_joint_pretrain_fewshot_{pretrainsetting}-{finetunesetting}-{domain}",
    #            reinit=True, config=settings)
    if traindomains == "ALL":
        alldomains = {"recipes", "restaurants", "blocks", "calendar", "housing", "publications"}
        traindomains = alldomains - {domain, }
    else:
        traindomains = set(traindomains.split("+"))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tt = q.ticktock("script")
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    if useadapters:
        assert(gradmode=="none", gradmode == "adapter" or gradmode == "adaptersplit")

    if gradmode == "split" and injecttraindata == False:
        tt.msg("probably makes sense to inject training data (-injecttraindata) when gradmode is \"split\"")
    if injecttraindata:
        tt.msg("injecting some training examples from all observed domains during finetuning!")

    tt.tick("loading data")
    sourcedss, targetdss, allsourceds, nltok, flenc, tokenmasks = \
        load_ds(traindomains=traindomains, testdomain=domain, nl_mode=encoder, mincoverage=mincoverage,
                fullsimplify=fullsimplify, add_domain_start=domainstart, batsize=batsize, ftbatsize=supportsize,
                supportsetting=supportsetting)
    tt.tock("data loaded")

    tt.tick("creating model")
    trainm, abstrainm = create_model(encoder_name=encoder,
                                 dec_vocabsize=flenc.vocab.number_of_ids(),
                                 dec_layers=numlayers,
                                 dec_dim=hdim,
                                 dec_heads=numheads,
                                 dropout=dropout,
                                 smoothing=smoothing,
                                 maxlen=maxlen,
                                 numbeam=numbeam,
                                 tensor2tree=partial(_tensor2tree, D=flenc.vocab),
                                 tokenmasks=tokenmasks,
                                 metarare=metarare,
                                 useadapters=useadapters,
                                     nospecialshared=nospecialshared,
                                 )
    tt.tock("model created")

    # region pretrain on all domains
    metrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    absmetrics = make_array_of_metrics("loss", "tree_acc")
    ftmetrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    vmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    vftmetrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    xmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    xftmetrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")

    # region parameters
    def get_parameters(m, _lr, _enclrmul):
        trainable_params = list(m.named_parameters())

        # tt.msg("different param groups")
        encparams = [v for k, v in trainable_params if k.startswith("model.model.encoder")]
        otherparams = [v for k, v in trainable_params if not k.startswith("model.model.encoder")]
        if len(encparams) == 0:
            raise Exception("No encoder parameters found!")
        paramgroups = [{"params": encparams, "lr": _lr * _enclrmul},
                       {"params": otherparams}]
        return paramgroups
    # endregion

    def get_optim(_m, _lr, _enclrmul, _wreg=0):
        paramgroups = get_parameters(_m, _lr=_lr, _enclrmul=_enclrmul)
        optim = torch.optim.Adam(paramgroups, lr=_lr, weight_decay=_wreg)
        return optim

    def clipgradnorm(_m=None, _norm=None):
        torch.nn.utils.clip_grad_norm_(_m.parameters(), _norm)

    eyt = q.EarlyStopper(vmetrics[1], patience=patience, min_epochs=patience, more_is_better=True, remember_f=lambda: deepcopy(trainm.model))
    # def wandb_logger():
    #     d = {}
    #     for name, loss in zip(["loss", "elem_acc", "seq_acc", "tree_acc"], metrics):
    #         d["train_"+name] = loss.get_epoch_error()
    #     for name, loss in zip(["seq_acc", "tree_acc"], vmetrics):
    #         d["valid_"+name] = loss.get_epoch_error()
    #     wandb.log(d)
    t_max = epochs
    optim = get_optim(trainm, lr, enclrmul, wreg)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=t_max-warmup) >> 0.
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    def get_ft_model(x, _resetspecialinner=False):
        _x = deepcopy(x)
        if _resetspecialinner:
            reset_special_inner(_x)
        return _x

    if startmtafter > 0:
        shelfpath = "metapretrain.shelf"
        tdomains = ",".join([str(x) for x in sorted(traindomains)])
        pretrainmodelsettings = f"traindomains:{tdomains}" \
                                f"+testdomain:{domain}" \
                                f"+dropout:{dropout}" \
                                f"+numlayers:{numlayers}" \
                                f"+hdim:{hdim}" \
                                f"+numheads:{numheads}" \
                                f"+seed:{seed}" \
                                f"+metarare:{metarare}" \
                                f"+lr:{lr}" \
                                f"+useadapters:{useadapters}" \
                                f"+batsize:{batsize}" \
                                f"+gradnorm:{gradnorm}" \
                                f"+gradacc:{gradacc}"
        with shelve.open(shelfpath) as shelf:
            if True:
            # if pretrainmodelsettings not in shelf:
                pretrainepoch = partial(meta_train_epoch,
                                     model=trainm,
                                     absmodel=abstrainm,
                                     data=sourcedss,
                                     allsourcedata=allsourceds,
                                     injecttraindata=injecttraindata,
                                     optim=optim,
                                     get_ft_model=partial(get_ft_model, _resetspecialinner=False),
                                     get_ft_optim=partial(get_optim,
                                                          _lr=ftlr,
                                                          _enclrmul=enclrmul,
                                                          _wreg=wreg),
                                    ftbatsize=batsize,
                                     gradmode="none",
                                     losses=metrics,
                                     abslosses=absmetrics,
                                     ftlosses=ftmetrics,
                                     finetunesteps=0,
                                     outersteps=1,
                                     clipgradnorm=clipgradnorm,
                                     outergradnorm=gradnorm,
                                     innergradnorm=ftgradnorm,
                                     device=device,
                                     on_end=[],
                                     gradacc=gradacc,
                                     abstract_contrib=0.,)

                tt.tick("pre-pretraining")
                q.run_training(run_train_epoch=pretrainepoch,
                               max_epochs=startmtafter)
                tt.tock("done pre-pretraining")
                # tt.tick("saving in shelf")
                # shelf[pretrainmodelsettings] = trainm.state_dict()
                # tt.tock("saved in shelf")

        # with shelve.open(shelfpath) as shelf:
        #     tt.tick("loading from shelf")
        #     trainmdict = shelf[pretrainmodelsettings]
        #     trainm.load_state_dict(trainmdict)
        #     # assert(torch.all(trainm.model.outlin.weight == abstrainm.model.outlin.weight))
        #     tt.tock("loaded from shelf")
        #     # optim = get_optim(trainm, lr, enclrmul, wreg)

        if reinitspecialinner:
            tt.msg("resetting special inner")
            reset_special_inner(trainm)

    reinitializer = Reinitializer(trainm, reinitinterval)

    trainepoch = partial(meta_train_epoch,
                         model=trainm,
                         absmodel=abstrainm,
                         data=sourcedss,
                         allsourcedata=allsourceds,
                         injecttraindata=injecttraindata,
                         optim=optim,
                         get_ft_model=partial(get_ft_model, _resetspecialinner=resetspecialinner),
                         get_ft_optim=partial(get_optim,
                                              _lr=ftlr,
                                              _enclrmul=enclrmul,
                                              _wreg=wreg),
                         ftbatsize=ftbatsize,
                         gradmode=gradmode,
                         losses=metrics,
                         abslosses=absmetrics,
                         ftlosses=ftmetrics,
                         finetunesteps=finetunesteps,
                         outersteps=outersteps,
                         clipgradnorm=clipgradnorm,
                         outergradnorm=gradnorm,
                         innergradnorm=ftgradnorm,
                         device=device,
                         on_start=[lambda: reinitializer()],
                         on_end=[lambda: lr_schedule.step()],
                         gradacc=gradacc,
                         abstract_contrib=abscontrib,)

    bestfinetunesteps = q.hyperparam(0)
    validepoch = partial(meta_test_epoch,
                        model=trainm,
                        data=targetdss,
                         allsourcedata=allsourceds,
                         injecttraindata=injecttraindata,
                        get_ft_model=partial(get_ft_model, _resetspecialinner=resetspecialinner),
                        get_ft_optim=partial(get_optim,
                                             _lr=ftlr,
                                             _enclrmul=enclrmul,
                                             _wreg=wreg),
                         ftbatsize=ftbatsize,
                        gradmode=gradmode,
                        bestfinetunestepsvar=bestfinetunesteps,
                        bestfinetunestepswhichmetric=1,
                        losses=vmetrics,
                        ftlosses=vftmetrics,
                        finetunesteps=maxfinetunesteps,
                        mode="valid",
                        evalinterval=evalinterval,
                        clipgradnorm=partial(clipgradnorm, _norm=ftgradnorm),
                        device=device,
                        print_every_batch=False,
                        on_outer_end=[lambda: eyt.on_epoch_end()])

    # print(testepoch())

    tt.tick("pretraining")
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=validepoch,
                   validinter=validinter,
                   max_epochs=epochs-startmtafter,
                   check_stop=[lambda: eyt.check_stop()])
    tt.tock("done pretraining")

    if eyt.get_remembered() is not None and validinter == 1:
        tt.msg(f"best finetune steps: {q.v(bestfinetunesteps)+1}")
        trainm.model = eyt.get_remembered()
        tt.msg("reloaded")

    testepoch = partial(meta_test_epoch,
                        model=trainm,
                        data=targetdss,
                         allsourcedata=allsourceds,
                         injecttraindata=injecttraindata,
                        get_ft_model=partial(get_ft_model, _resetspecialinner=resetspecialinner),
                        get_ft_optim=partial(get_optim,
                                             _lr=ftlr,
                                             _enclrmul=enclrmul,
                                             _wreg=wreg),
                         ftbatsize=ftbatsize,
                        gradmode=gradmode,
                        bestfinetunestepsvar=maxfinetunesteps-1,
                        losses=xmetrics,
                        ftlosses=xftmetrics,
                        finetunesteps=maxfinetunesteps,
                        mode="test",
                        evalinterval=testevalinterval,
                        clipgradnorm=partial(clipgradnorm, _norm=ftgradnorm),
                        device=device,
                        print_every_batch=False,
                        on_outer_end=[lambda: eyt.on_epoch_end()])

    tt.tick(f"testing with @{q.v(bestfinetunesteps)+1} (out of {q.v(maxfinetunesteps)}) steps")
    testmsg = testepoch(finetunesteps=maxfinetunesteps)
    tt.msg(testmsg)
    tt.tock("tested")
    # endregion
    #
    # # region finetune
    # ftmetrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    # ftvmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    # ftxmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    #
    # trainable_params = list(trainm.named_parameters())
    # exclude_params = set()
    # # exclude_params.add("model.model.inp_emb.emb.weight")  # don't train input embeddings if doing glove
    # if len(exclude_params) > 0:
    #     trainable_params = [(k, v) for k, v in trainable_params if k not in exclude_params]
    #
    # tt.msg("different param groups")
    # encparams = [v for k, v in trainable_params if k.startswith("model.model.encoder")]
    # otherparams = [v for k, v in trainable_params if not k.startswith("model.model.encoder")]
    # if len(encparams) == 0:
    #     raise Exception("No encoder parameters found!")
    # paramgroups = [{"params": encparams, "lr": ftlr * enclrmul},
    #                {"params": otherparams}]
    #
    # ftoptim = torch.optim.Adam(paramgroups, lr=ftlr, weight_decay=wreg)
    #
    # clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(trainm.parameters(), gradnorm)
    #
    # eyt = q.EarlyStopper(ftvmetrics[1], patience=1000, min_epochs=10, more_is_better=True,
    #                      remember_f=lambda: deepcopy(trainm.model))
    #
    # # def wandb_logger_ft():
    # #     d = {}
    # #     for name, loss in zip(["loss", "elem_acc", "seq_acc", "tree_acc"], ftmetrics):
    # #         d["ft_train_" + name] = loss.get_epoch_error()
    # #     for name, loss in zip(["seq_acc", "tree_acc"], ftvmetrics):
    # #         d["ft_valid_" + name] = loss.get_epoch_error()
    # #     wandb.log(d)
    #
    # t_max = epochs
    # print(f"Total number of updates: {t_max} .")
    # if cosinelr:
    #     lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=t_max - warmup) >> 0.
    # else:
    #     lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    # lr_schedule = q.sched.LRSchedule(ftoptim, lr_schedule)
    #
    # trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    # trainepoch = partial(q.train_epoch, model=trainm, dataloader=ftdl, optim=ftoptim, losses=ftmetrics,
    #                      _train_batch=trainbatch, device=device, on_end=[lambda: lr_schedule.step()])
    # validepoch = partial(q.test_epoch, model=testm, dataloader=fvdl, losses=ftvmetrics, device=device,
    #                      on_end=[lambda: eyt.on_epoch_end()])#, lambda: wandb_logger_ft()])
    #
    # tt.tick("training")
    # q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs,
    #                check_stop=[lambda: eyt.check_stop()])
    # tt.tock("done training")
    #
    # if eyt.get_remembered() is not None:
    #     tt.msg("reloaded")
    #     trainm.model = eyt.get_remembered()
    #     testm.model = eyt.get_remembered()
    #
    # # endregion
    #
    # tt.tick("testing")
    # validresults = q.test_epoch(model=testm, dataloader=fvdl, losses=ftvmetrics, device=device)
    # testresults = q.test_epoch(model=testm, dataloader=xdl, losses=ftxmetrics, device=device)
    # print(validresults)
    # print(testresults)
    # tt.tock("tested")
    # # settings.update({"train_seqacc": losses[]})
    #
    # for metricarray, datasplit in zip([ftmetrics, ftvmetrics, ftxmetrics], ["train", "valid", "test"]):
    #     for metric in metricarray:
    #         settings[f"{datasplit}_{metric.name}"] = metric.get_epoch_error()
    #
    # # wandb.config.update(settings)
    settings.update({"test_tree_accuracy": xmetrics[-1].get_epoch_error()})
    print(settings)
    return settings

# python overnight_new_meta_pretrain.py -gpu 0 -numbeam 5 -supportsetting min -metarare emb+linout -gradmode metarare -resetspecialinner -startmtafter 15 -abscontrib 0. -numlayers 3 -seed 87646464 -dropout .2 -finetunesteps 5
def run_experiments(domain="undefined", gpu=-1, lr=0.0001, ftlr=0.0001, enclrmul=0.1, patience=-1, cosinelr=False, fullsimplify=True, batsize=50, ftbatsize=-1, supportsize=-1,
                         smoothing=0., dropout=0.3, numlayers=3, numheads=12, hdim=768, domainstart=False, gradacc=1, gradnorm=3, ftgradnorm=-1,
                         numbeam=1, supportsetting="train", abscontrib=-1., metarare="undefined", finetunesteps=5, outersteps=1, gradmode="undefined",
                         maxfinetunesteps=100, evalinterval=20, testevalinterval=5, epochs=60, injecttraindata=False, useadapters=False,
                        seed=-1, mincoverage=2, resetspecialinner=False, reinitspecialinner=False, reinitinterval=-1, nospecialshared=False, validinter=1,
                    startmtafter=0):
    ranges = {
        "domain": ["recipes", "blocks", "calendar", "housing", "publications"],
        "lr": [lr],
        "ftlr": [ftlr],
        "enclrmul": [enclrmul],
        "warmup": [0],
        "epochs": [epochs],
        "numheads": [numheads],
        "numlayers": [2, 3, 4, 5],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "smoothing": [smoothing],
        "supportsetting": ["min", "train"],
        "hdim": [hdim],
        "finetunesteps": [3, 5, 10, 15],
        "numbeam": [numbeam],
        "batsize": [batsize],
        "ftbatsize": [ftbatsize],
        "abscontrib": [0., 0.1],
        "seed": [87646464, 98765456, 23655798],
        "gradmode": ["none", "metarare"],
        "metarare": ["no", "yes"],
        "reinitinterval": [0, 1, 3, 5]
    }
    p = __file__ + f".{domain}"
    if domain != "undefined":
        ranges["domain"] = [domain]
    if gradmode != "undefined":
        ranges["gradmode"] = [gradmode]
    if metarare != "undefined":
        ranges["metarare"] = [metarare]
    if metarare != "supportsetting":
        ranges["supportsetting"] = [supportsetting]
    if seed >= 0:
        ranges["seed"] = [seed]
    if abscontrib >= 0:
        ranges["abscontrib"] = [abscontrib]
    if dropout >= 0:
        ranges["dropout"] = [dropout]
    if finetunesteps >= 0:
        ranges["finetunesteps"] = [finetunesteps]
    if numlayers >= 0:
        ranges["numlayers"] = [numlayers]
    if reinitinterval >= 0:
        ranges["reinitinterval"] = [reinitinterval]

    def check_config(x):
        # effectiveenclr = x["enclrmul"] * x["lr"]
        # if effectiveenclr < 0.000005:
        #     return False
        dimperhead = x["hdim"] / x["numheads"]
        if dimperhead < 20 or dimperhead > 100:
            return False
        if x["metarare"] == "no" and x["gradmode"] == "metarare":
            return False
        return True

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config,
                      fullsimplify=fullsimplify,
                      gpu=gpu, patience=patience, cosinelr=cosinelr,
                      domainstart=domainstart,
                      supportsetting=supportsetting,
                      abscontrib=abscontrib,
                      outersteps=outersteps,
                      mincoverage=mincoverage,
                      gradacc=gradacc, gradnorm=gradnorm,
                      maxfinetunesteps=maxfinetunesteps,
                      evalinterval=evalinterval,
                      testevalinterval=testevalinterval,
                      injecttraindata=injecttraindata,
                      useadapters=useadapters,
                      resetspecialinner=resetspecialinner,
                      reinitspecialinner=reinitspecialinner,
                      nospecialshared=nospecialshared,
                      ftgradnorm=ftgradnorm,
                      validinter=validinter,
                      startmtafter=startmtafter,
                      supportsize=supportsize)


def run_experiments_seed(domain="restaurants", gpu=-1, lr=0.0001, ftlr=0.0001, patience=10, cosinelr=False, fullsimplify=True, batsize=50,
                         smoothing=0., dropout=.1, numlayers=3, numheads=12, hdim=768, domainstart=False, gradacc=1,
                         numbeam=1, supportsetting="lex", abscontrib=.1, nometarare=False, finetunesteps=1, gradmode="none",
                         maxfinetunesteps=30, evalinterval=5, epochs=100, injecttraindata=False):
    ranges = {
        "lr": [lr],
        "ftlr": [ftlr],
        "enclrmul": [0.1],
        "warmup": [0],
        "epochs": [epochs],
        "numheads": [numheads],
        "numlayers": [numlayers],
        "dropout": [dropout],
        "smoothing": [smoothing],
        "hdim": [hdim],
        "numbeam": [numbeam],
        "batsize": [batsize],
        "seed": [12345678, 65748390, 98387670, 23655798, 66453829],     # TODO: add more later
    }
    p = __file__ + f".{domain}"
    def check_config(x):
        # effectiveenclr = x["enclrmul"] * x["lr"]
        # if effectiveenclr < 0.000005:
        #     return False
        dimperhead = x["hdim"] / x["numheads"]
        if dimperhead < 20 or dimperhead > 100:
            return False
        return True

    q.run_experiments(run, ranges, path_prefix=p, check_config=check_config,
                      domain=domain, fullsimplify=fullsimplify,
                      gpu=gpu, patience=patience, cosinelr=cosinelr,
                      domainstart=domainstart,
                      supportsetting=supportsetting,
                      abscontrib=abscontrib,
                      finetunesteps=finetunesteps,
                      gradmode=gradmode,
                      gradacc=gradacc,
                      nometarare=nometarare,
                      maxfinetunesteps=maxfinetunesteps,
                      evalinterval=evalinterval,
                      injecttraindata=injecttraindata)



if __name__ == '__main__':
    faulthandler.enable()
    # ret = q.argprun(run)
    # print(ret)
    q.argprun(run_experiments)
    # fire.Fire(run_experiments)