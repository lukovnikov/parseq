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
import random
import re
import string
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Callable, Set, Dict, Union, List

import fire
# import wandb

import qelos as q   # branch v3
import numpy as np
import torch
from nltk import Tree
from torch.utils.data import DataLoader

from parseq.datasets import OvernightDatasetLoader, pad_and_default_collate, autocollate, Dataset
from parseq.decoding import merge_metric_dicts, SeqDecoder, StopDecoding
from parseq.eval import SeqAccuracies, TreeAccuracy, make_array_of_metrics, CELoss
from parseq.grammar import tree_to_lisp_tokens, lisp_to_tree
from parseq.states import State, TrainableDecodableState
from parseq.transitions import TransitionModel, LSTMCellTransition
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


# region data
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
            exampletokens = set(example[1])
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
    if general_tokens is None:
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


def tokenize_and_add_start(t, _domain, general_tokens=None):
    tokens = tree_to_lisp_tokens(t)
    if general_tokens is not None:
        newtokens = []
        for token in tokens:
            if re.match("@[^@]+@", token) or token in general_tokens:
                newtokens.append(token)
            else:
                # if token in ("-1", "0", "1", "2", "3"):
                #     print("Numeric token!")
                newtokens.append(f"{_domain}|{token}")
        tokens = newtokens
    starttok = "@START@"
    tokens = [starttok] + tokens
    return tokens


def load_ds(traindomains=("restaurants",),
            testdomain="housing",
            min_freq=1,
            mincoverage=1,
            top_k=np.infty,
            batsize=10,
            ftbatsize=-1,
            nl_mode="bert-base-uncased",
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
    fullsimplify=True

    if supportsetting == "lex":
        if mincoverage > 1:
            print(f"Changing mincoverage to 1 because supportsetting=='{lex}', mincoverage was {mincoverage}.")
            mincoverage = 1

    general_tokens = {
        "(", ")", "arg:~type", "arg:type", "op:and", "SW:concat", "cond:has",
        "arg:<=", "arg:<", "arg:>=", "arg:>", "arg:!=", "arg:=", "SW:superlative",
        "SW:CNT-arg:min", "SW:CNT-arg:<", "SW:CNT-arg:<=", "SW:CNT-arg:>=", "SW:CNT-arg:>",
        "SW:CNT-arg:max", "SW:CNT-arg:=", "arg:max", "arg:min", ".size",
        "agg:arg:sum", "agg:arg:avg"}
        # "-1", "0", "1", "2", "3", "5", "10", "15", "30", "40", "300", "2000", "1000", "1500", "800", "2015", "2004"}

    domains = {}
    for domain in list(traindomains) + [testdomain]:
        ds = OvernightDatasetLoader(simplify_mode="light" if not fullsimplify else "full", simplify_blocks=True,
                                    restore_reverse=DATA_RESTORE_REVERSE, validfrac=.10)\
            .load(domain=domain)
        domainexamples = [(a, b, c) for a, b, c in ds.examples]
        if supportsetting == "lex":
            domainexamples = [(a, b, "support" if c == "lexicon" else c)
                              for a, b, c in domainexamples]
        else:
            domainexamples = [(a, b, c) for a, b, c in domainexamples if c != "lexicon"]
        domains[domain] = domainexamples

    alltrainex = []
    for domain in domains:
        domains[domain] = [(a, tokenize_and_add_start(b, domain, general_tokens=general_tokens), c)
                           for a, b, c in domains[domain]]
        alltrainex += [(a, b, c, domain) for a, b, c in domains[domain]]

    if True or supportsetting == "min" or supportsetting == "train":
        for domain, domainexamples in domains.items():
            print(domain)
            loadedex = [a for a in alltrainex if a[3] == domain and a[2] == "support"]
            loadedex += [a for a in alltrainex if (a[3] != domain and a[2] == "train")]
            mindomainexamples = get_maximum_spanning_examples([(a, b, c) for a, b, c in domainexamples if c == "train"],
                                          mincoverage=mincoverage, #loadedex=None)
                                          loadedex=loadedex)
            domains[domain] = domains[domain] + [(a, b, "support") for a, b, c in mindomainexamples]

    allex = []
    for domain in domains:
        allex += [(a, b, c, domain) for a, b, c in domains[domain]]
    ds = Dataset(allex)

    et = get_lf_abstract_transform(ds[lambda x: x[3] != testdomain].examples, general_tokens=general_tokens)
    ds = ds.map(lambda x: (x[0], x[1], et(x[1]), x[2], x[3]))

    seqenc_vocab = Vocab(padid=0, unkid=1, startid=2, endid=3)
    seqenc_vocab.add_token("@ABS@", seen=np.infty)
    seqenc_vocab.add_token("@ABSSTART@", seen=np.infty)
    seqenc_vocab.add_token("@METARARE@", seen=np.infty)
    seqenc_vocab.add_token("@META@", seen=np.infty)
    seqenc = SequenceEncoder(vocab=seqenc_vocab, tokenizer=lambda x: x,
                             add_start_token=False, add_end_token=True)
    for example in ds.examples:
        seqenc.inc_build_vocab(example[1], seen=example[3] in ("train", "support") if example[4] != testdomain else example[3] == "support")
        seqenc.inc_build_vocab(example[2], seen=example[3] in ("train", "support") if example[4] != testdomain else example[3] == "support")
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
               x[3], x[4],
               x[0], x[1], x[2], x[3])
        return ret
    allex = [tokenize(ex) for ex in ds.examples]
    return allex, nl_tokenizer, seqenc, tokenmasks


def pack_loaded_ds(allex, traindomains, testdomain):
    trainex = []
    validex = []
    testex = []
    supportex = {}
    for ex in allex:
        if ex[3] == "support":
            exdomain = ex[4]
            if exdomain not in supportex:
                supportex[exdomain] = []
            supportex[exdomain].append(ex)
        else:
            if ex[4] == testdomain:
                if ex[3] == "test":
                    testex.append(ex)
            elif ex[4] in traindomains:
                if ex[3] == "train":
                    trainex.append(ex)
                elif ex[3] == "valid":
                    validex.append(ex)
                elif ex[3] == "test":
                    pass
    trainds = Dataset(trainex)
    validds = Dataset(validex)
    testds = Dataset(testex)

    def supportretriever(x, domain_mems=None):
        domainex = domain_mems[x[4]]
        mem = [(x[0], x[1]) for x in domainex]
        mem = autocollate(mem)
        ret = (x[0], x[1],) + tuple(mem)
        return ret

    trainds = trainds.map(partial(supportretriever, domain_mems=supportex)).cache()
    validds = validds.map(partial(supportretriever, domain_mems=supportex)).cache()
    testds = testds.map(partial(supportretriever, domain_mems=supportex)).cache()
    trainds[0]
    trainds[0]
    return trainds, validds, testds


def load_data(traindomains=("restaurants",),
              testdomain="housing",
              supportsetting="lex", # "lex" or "min"
              batsize=5,
              numworkers=0,
              ):
    allex, nltok, flenc, tokenmasks = \
        load_ds(traindomains=traindomains,
                testdomain=testdomain,
                supportsetting=supportsetting,
                )
    trainds, validds, testds = pack_loaded_ds(allex, traindomains, testdomain)

    def collatefn(x, pad_value=0):
        y = list(zip(*x))
        for i, yi in enumerate(y):
            if isinstance(yi[0], torch.LongTensor):
                if yi[0].dim() == 1:
                    y[i] = q.pad_tensors(yi, 0, pad_value)
                elif yi[0].dim() == 2:
                    y[i] = q.pad_tensors(yi, (0, 1), pad_value)
        for i, yi in enumerate(y):
            if isinstance(yi[0], torch.Tensor):
                y[i] = torch.stack(yi, 0)
        supmask_source = y[2][:, :, 0] == y[2][0, 0, 0]
        supmask_target = y[3][:, :, 0] == y[3][0, 0, 0]
        assert (torch.allclose(supmask_source.float(), supmask_target.float()))
        y[2][:, :, 0] = y[2][0, 0, 0]
        y[3][:, :, 0] = y[3][0, 0, 0]
        y.append(supmask_target)
        return y

    traindl = DataLoader(trainds, batsize, shuffle=True, num_workers=numworkers, collate_fn=collatefn)
    validdl = DataLoader(validds, batsize, shuffle=True, num_workers=numworkers, collate_fn=collatefn)
    testdl = DataLoader(testds, batsize, shuffle=True, num_workers=numworkers, collate_fn=collatefn)

    return traindl, validdl, testdl, nltok, flenc, tokenmasks
# endregion

def apply_withpath(m:torch.nn.Module, fn:Callable, mpath=None):
    """ Apply function 'fn' recursively on 'm' and its submodules, where 'fn' gets 'm' and 'mpath' as argument """
    fn(m, mpath)
    for name, child in m.named_children():
        apply_withpath(child, fn, f"{mpath}.{name}" if mpath is not None else f"{name}")


class GatedFF(torch.nn.Module):
    def __init__(self, indim, odim, dropout=0., activation=None, zdim=None, **kw):
        super(GatedFF, self).__init__(**kw)
        self.dim = indim
        self.odim = odim
        self.zdim = self.dim * 4 if zdim is None else zdim

        self.activation = torch.nn.CELU() if activation is None else activation

        self.linA = torch.nn.Linear(self.dim, self.zdim)
        self.linB = torch.nn.Linear(self.zdim, self.odim)
        self.linMix = torch.nn.Linear(self.zdim, self.odim)
        # self.linMix.bias.data.fill_(3.)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *inps):
        h = inps[-1]
        _h = torch.cat(inps, -1)
        _cand = self.linA(self.dropout(_h))
        _cand = self.activation(_cand)
        cand = self.linB(self.dropout(_cand))
        mix = torch.sigmoid(self.linMix(_cand))
        ret = h * mix + cand * (1 - mix)
        return ret


class SGRUCell(torch.nn.Module):
    def __init__(self, dim, bias=True, dropout=0., **kw):
        super(SGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 5, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        inp = self.dropout(inp)
        gates = self.gateW(inp)
        gates = list(gates.chunk(5, 1))
        rx = torch.sigmoid(gates[0])
        rh = torch.sigmoid(gates[1])
        z_gates = gates[2:5]
        # z_gates[2] = z_gates[2] - self.gate_bias
        z = torch.softmax(torch.stack(z_gates, 2), -1)
        inp = torch.cat([x * rx, h * rh], 1)
        inp = self.dropout(inp)
        u = self.gateU(inp)
        u = torch.tanh(u)
        h_new = torch.stack([x, h, u], 2) * z
        h_new = h_new.sum(-1)
        return h_new


class SuperBasicDecoderState(TrainableDecodableState):
    """ Most basic decoder state for use with parseq SeqDecoder.
        Needs a End-Of-Sequence token id specified! """
    EOS_ID = 3
    def __init__(self, *args, eos_id=EOS_ID, **kw):
        super(SuperBasicDecoderState, self).__init__(*args, **kw)
        self._is_terminated = None
        self.eos_id = eos_id

    def is_terminated(self):
        if self._is_terminated is None:
            return [False]
        return self._is_terminated

    def start_decoding(self):
        pass

    def step(self, actions:torch.Tensor=None):
        action_ids = list(actions.cpu().numpy())
        if self._is_terminated is None:
            self._is_terminated = [x == self.eos_id for x in action_ids]
        self._is_terminated = [x == self.eos_id or y == True for x, y in zip(action_ids, self._is_terminated)]
        if not hasattr(self, "followed_actions"):
            self.followed_actions = torch.zeros_like(actions[:, None])[:, :0]
        self.followed_actions = torch.cat([self.followed_actions, actions[:, None]], 1)

    def get_gold(self, i:int=None):
        if i is None:
            return self.gold
        else:
            if i >= self.gold.size(1):
                raise StopDecoding()
            return self.gold[:, i]


class DecoderInputLayer(torch.nn.Module):
    def __init__(self, vocsize, dim, encdim=None, unkid=1, unktoks:Set[int]=set(), **kw):
        """
        :param vocsize:     Number of unique words in vocabulary
        :param dim:         Embedding dimension
        :param encdim:      dimension of encoding passed as prev_inp_summ in forward()
        :param unkid:       the id to map unknown tokens to
        :param unktoks:     set of integer ids of unknown tokens. These tokens get mapped to unkid.
        :param kw:
        """
        super(DecoderInputLayer, self).__init__(**kw)
        self.vocsize = vocsize
        self.dim = dim
        self.encdim = encdim if encdim is not None else dim
        self.unktoks = unktoks
        self.unkid = unkid
        mapper = torch.arange(vocsize)
        for unktok in unktoks:
            mapper[unktok] = unkid
        self.register_buffer("mapper", mapper)

        self.emb = torch.nn.Embedding(vocsize, dim)

        self.outdim = self.dim + self.encdim

    def forward(self, ids, prev_inp_summ, prev_mem_summ=None):
        _ids = ids
        ids = self.mapper[ids]

        emb = self.emb(ids)
        out = torch.cat([emb, prev_inp_summ], 1)
        return out


class DecoderOutputLayer(torch.nn.Module):
    """
    Decoder Output Layer that combines generation and copying from a memory.
    Unktokens are not able to be produced by generation!

    """
    def __init__(self, dim, vocsize, unktoks:Set[int]=set(), dropout=0., **kw):
        """
        :param dim:     dimension of input vectors
        :param vocsize: number of unique words in vocabulary
        :param unktoks: set of integer ids of tokens that are unknown. Their generation probabilities will be set to zero.
        :param kw:
        """
        super(DecoderOutputLayer, self).__init__(**kw)
        self.dim = dim
        self.vocsize = vocsize
        self.merger = SGRUCell(dim)
        self.merger2 = SGRUCell(dim)
        self.mixer = torch.nn.Linear(dim, 2)
        self.outlin = torch.nn.Linear(dim, vocsize)
        self.dropout = dropout
        unktok_mask = torch.ones(vocsize)
        for unktok in unktoks:
            unktok_mask[unktok] = 0
        self.register_buffer("unktok_mask", unktok_mask)

    def forward(self, enc, summ, memids=None, memencs=None, memmask=None):
        """
        :param enc:     (batsize, dim)
        :param summ:    (batsize, indim)
        :param memids:  (batsize, memsize, memseqlen) - integer ids of sequences in given memory
        :param memencs: (batsize, memsize, memseqlen, memdim) - encoding of memory
        :param memmask: (batsize, memsize, memseqlen) - bool mask whether memory token is padding
        :return:
        """
        if memmask is None:
            memmask = torch.ones_like(memids).float()
        # compute
        h = self.merger(summ, enc)

        # compute probs from memory
        probs_mem = torch.zeros(h.size(0), self.vocsize, device=h.device)
            # (batsize, vocsize)

        # compute attentions to memory
        mem_weights = torch.einsum("bd,bmzd->bmz", h, memencs)
        mem_weights = mem_weights + torch.log(memmask.float())
        # mem_weights_size = mem_weights.size()
        mem_weights = mem_weights.view(mem_weights.size(0), -1)
        mem_alphas = torch.softmax(mem_weights, -1) # (batsize, memsize*memseqlen)
        mem_summ = mem_alphas[:, :, None] * memencs.contiguous().view(memencs.size(0), memencs.size(1) * memencs.size(2), -1)
        mem_summ = mem_summ.sum(1)
        mem_toks = memids.contiguous().view(memids.size(0), -1)  # (batsize, memsize*memseqlen)
        probs_mem = probs_mem.scatter_add(1, mem_toks, mem_alphas)

        h = self.merger2(h, mem_summ)

        logits_gen = self.outlin(h)
        mask_gen = self.unktok_mask[None, :].repeat(logits_gen.size(0), 1)
        logits_gen = logits_gen + torch.log(mask_gen)
        probs_gen = torch.softmax(logits_gen, -1)
        # # prevent naningrad
        # _probs_gen = torch.stack([torch.zeros_like(_probs_gen),
        #                         _probs_gen], 0)
        # probs_gen = _probs_gen.gather(0, mask_gen.long()[None])[0]

        mix = self.mixer(h)
        mix = torch.softmax(mix, -1)
        probs = mix[:, 0:1] * probs_gen + mix[:, 1:2] * probs_mem

        # prevent naningrad
        _probs = torch.stack([torch.zeros_like(probs),
                              probs], 0)
        probs = _probs.gather(0, (probs != 0).long()[None])[0]

        # probs = probs.clamp_min(1e-4)
        return probs, mem_summ


class InnerLSTMDecoderCell(TransitionModel):
    """
        LSTM cell based decoder cell.
        This decoder cell will be used to "encode" the output parts of the memory.
    """
    def __init__(self, inplayer:DecoderInputLayer=None,
                 dim=None, encdim=None, numlayers=1, dropout=0.,
                 eos_id=3, **kw):
        """
        :param inplayer:    DecoderInputLayer used for embedding.
        :param dim:         encoding output dimension
        :param encdim:      not used # TODO
        :param numlayers:
        :param dropout:
        :param kw:
        """
        super(InnerLSTMDecoderCell, self).__init__(**kw)
        self.inplayer = inplayer
        self.dim = dim
        self.encdim = encdim if encdim is not None else dim
        dims = [inplayer.outdim] + [dim]*numlayers
        lstms = [torch.nn.LSTMCell(dims[i], dims[i+1]) for i in range(len(dims) - 1)]
        self.lstm_transition = LSTMCellTransition(*lstms, dropout=dropout)
        self.dropout = dropout

        self.merger = SGRUCell(self.dim, dropout=self.dropout)

        self.inp_att_qlin = None #torch.nn.Linear(self.dim, self.dim)
        self.inp_att_klin = None #torch.nn.Linear(self.encdim, self.dim)

        self.eos_id = eos_id

    def get_init_state(self, batsize, device=torch.device("cpu")):
        return SuperBasicDecoderState(lstmstate=self.lstm_transition.get_init_state(batsize, device=device), eos_id=self.eos_id)

    def inp_att(self, q, k, v, kmask=None):
        """
        :param q:   (batsize, dim)
        :param k:   (batsize, seqlen, dim)
        :param kmask:   (batsize, seqlen)
        :return:
        """
        if self.inp_att_qlin is not None:
            q = self.inp_att_qlin(q)
        if self.inp_att_klin is not None:
            k = self.inp_att_klin(k)
        if kmask is None:
            kmask = torch.ones_like(k[:, :, 0])
        weights = torch.einsum("bd,bsd->bs", q, k)
        weights = weights + torch.log(kmask.float())
        alphas = torch.softmax(weights, -1)
        summ = alphas[:, :, None] * v
        summ = summ.sum(1)
        return alphas, summ, weights

    def forward(self, x:State):
        inp = self.inplayer(x.prev_actions, x.prev_summ)

        enc, new_lstmstate = self.lstm_transition(inp, x.lstmstate)
        x.lstmstate = new_lstmstate

        alphas, summ, _ = self.inp_att(enc, x.ctx, x.ctx, x.ctxmask)
        x.prev_summ = summ

        enc = self.merger(enc, summ)
        #
        # if "enc" not in x:
        #     x.enc = enc[:, None, :][:, :0, :]
        # x.enc = torch.cat([x.enc, enc[:, None, :]], 1)
        return enc, x


class InnerDecoder(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, starttokens, y, ctx, ctxmask=None):
        """
        Computes the "decoding" encoding of output sequence 'y' given the context 'ctx'.

        :param starttokens  (batsize*memsize,) int ids of starting tokens
        :param y:           (batsize*memsize, seqlen)  output sequence int ids
        :param ctx:         (batsize*memsize, inseqlen, indim) encodings of the input side of memory
        :param ctxmask:     (batsize*memsize, inseqlen) mask for input side
        :return:            (batsize*memsize, seqlen, dim) - encoding of output sequence y
        """
        pass


class StateInnerDecoder(InnerDecoder):
    """ State-based decoder to use to encode memory output side.
        Wraps a SeqDecoder and creates a state from args. """
    def __init__(self, cell:TransitionModel, maxtime=100, **kw):
        super(StateInnerDecoder, self).__init__(**kw)
        self.decoder = SeqDecoder(cell, maxtime=maxtime)
        self.cell = cell

    @classmethod
    def create_state(cls, cell, starttokens, y, ctx, ctxmask=None):
        state = cell.get_init_state(ctx.size(0), device=ctx.device)
        state.prev_actions = starttokens
        state.prev_summ = torch.zeros_like(ctx[:, 0])
        state.ctx = ctx
        state.gold = y
        state.ctxmask = ctxmask
        return state

    def forward(self, starttokens, y, ctx, ctxmask=None):
        # create state
        state = self.create_state(self.cell, starttokens, y, ctx, ctxmask=ctxmask)

        # decode from state
        _, newstate, enc, _, _ = self.decoder(state, tf_ratio=1., return_all=True)
        return enc


class LSTMDecoderCellWithMemory(TransitionModel):
    """
    Decoder cell for use with StateDecoderWithMemory.
    """
    def __init__(self, inplayer:DecoderInputLayer=None,
                 dim=None, encdim=None, numlayers=1, dropout=0.,
                 outlayer:DecoderOutputLayer=None,
                 eos_id=3, **kw):
        """
        :param inplayer:    DecoderInputLayer to embed input tokens
        :param dim:         dimensionality of embedding
        :param encdim:      not used ! # TODO
        :param numlayers:   number of LSTM cell layers
        :param dropout:
        :param outlayer:    DecoderOutputLayer
        :param kw:
        """
        super(LSTMDecoderCellWithMemory, self).__init__(**kw)
        self.inplayer = inplayer
        self.dim = dim
        self.encdim = encdim if encdim is not None else dim
        dims = [inplayer.outdim] + [dim]*numlayers
        lstms = [torch.nn.LSTMCell(dims[i], dims[i+1]) for i in range(len(dims) - 1)]
        self.lstm_transition = LSTMCellTransition(*lstms, dropout=dropout)
        self.outlayer = outlayer
        self.dropout = dropout

        self.inp_att_qlin = None #torch.nn.Linear(self.dim, self.dim)
        self.inp_att_klin = None #torch.nn.Linear(self.encdim, self.dim)

        self.eos_id = eos_id

    def get_init_state(self, batsize, device=torch.device("cpu")):
        return SuperBasicDecoderState(lstmstate=self.lstm_transition.get_init_state(batsize, device=device), eos_id=self.eos_id)

    def inp_att(self, q, k, v, kmask=None):
        """
        :param q:   (batsize, dim)
        :param k:   (batsize, seqlen, dim)
        :param kmask:   (batsize, seqlen)
        :return:
        """
        if self.inp_att_qlin is not None:
            q = self.inp_att_qlin(q)
        if self.inp_att_klin is not None:
            k = self.inp_att_klin(k)
        if kmask is None:
            kmask = torch.ones_like(k[:, :, 0])
        weights = torch.einsum("bd,bsd->bs", q, k)
        weights = weights + torch.log(kmask.float())
        alphas = torch.softmax(weights, -1)
        summ = alphas[:, :, None] * v
        summ = summ.sum(1)
        return alphas, summ, weights

    def forward(self, x:State):
        inp = self.inplayer(x.prev_actions, x.prev_inp_summ, x.prev_mem_summ)

        enc, new_lstmstate = self.lstm_transition(inp, x.lstmstate)
        x.lstmstate = new_lstmstate

        alphas, summ, _ = self.inp_att(enc, x.x_enc, x.x_enc, x.xmask)
        x.prev_inp_summ = summ

        out, mem_summ = self.outlayer(enc, summ, memids=x.memids,
                            memencs=x.memencs, memmask=x.memmask)
        x.prev_mem_summ = mem_summ
        return out, x


class DecoderWithMemory(ABC, torch.nn.Module):
    """ Defines forward interface for all memory based decoders. """
    @abstractmethod
    def forward(self, x_enc, starttokens, y=None, memids=None, memencs=None, xmask=None, memmask=None, istraining:bool=True)->Dict:
        pass


class StateDecoderWithMemory(DecoderWithMemory):
    """ State-based decoder with memory.
        Wraps a SeqDecoder and creates a state from args.
        Used for outer decoder.
        Decoder contains a memory.
    """
    def __init__(self, cell:TransitionModel, eval=tuple(), maxtime=100, **kw):
        super(StateDecoderWithMemory, self).__init__(**kw)
        self.decoder = SeqDecoder(cell, eval=eval, maxtime=maxtime)
        self.cell = cell

    @classmethod
    def create_state(cls, cell, x_enc, starttokens, y, memencs, memids, xmask, memmask):
        state = cell.get_init_state(x_enc.size(0), device=x_enc.device)
        state.prev_actions = starttokens
        state.prev_inp_summ = torch.zeros_like(x_enc[:, 0])
        state.prev_mem_summ = torch.zeros_like(memencs[:, 0, 0])
        state.x_enc = x_enc
        state.gold = y
        state.memids = memids
        state.memencs = memencs
        state.xmask = xmask
        state.memmask = memmask
        return state

    def forward(self, x_enc, starttokens, y=None, memids=None, memencs=None, xmask=None, memmask=None, istraining:bool=True):
        # create state
        state = self.create_state(self.cell, x_enc, starttokens, y, memencs, memids, xmask, memmask)

        # decode from state
        _, _, out, predactions, golds = self.decoder(state, tf_ratio=1. if istraining else 0., return_all=True)
        return out, predactions


class MetaSeqMemNN(torch.nn.Module):
    """
    Top-level module for memory-based meta learning for seq2seq
    """
    def __init__(self, encoder, memory_encoder, decoder:DecoderWithMemory, memory_decoder:InnerDecoder,
                 dim = None,
                 dropout=0., **kw):
        """

        :param memory_encoder:     module that encodes memory inputs
        :param memory_decoder:     module that encodes memory outputs as a decoder
        :param encoder:             module that encodes inputs
        :param decoder:             a decoder cell
        :param encdim:
        :param dropout:
        :param kw:
        """
        super(MetaSeqMemNN, self).__init__(**kw)
        self.dim = dim
        self.memory_encoder, self.memory_decoder = memory_encoder, memory_decoder
        self.encoder, self.decoder = encoder, decoder

        self.input_enc_lin = None #torch.nn.Linear(self.dim, self.dim)
        self.supinput_enc_lin = None #torch.nn.Linear(self.dim, self.dim)
        self.merge_x = SGRUCell(self.dim, dropout=dropout)

    def forward(self, x, y, xsup, ysup, supmask, istraining=None):
        """
        :param x:       2D (batsize, seqlen) int tensor for input
        :param xsup:    3D (batsize, memsize, seqlen) int tensor for input side of support set for every example
        :param y:       2D (batsize, seqlen) int tensor for output (can be (batsize, 1) during test)
        :param ysup:    3D (batsize, memsize, seqlen) int tensor for output side of support set for every example
        :param supmask: 2D (batsize, memsize) bool tensor specifying which memory entries are non-entries
                        (! non-entries must consist of some token(s) !)
        :return:
        """
        istraining = self.training if istraining is None else istraining
        # encode the input and output sides of the support set
        xsup_size = xsup.size()
        ysup_size = ysup.size()
        xsup = xsup.view(-1, xsup.size(-1))      # flatten batsize and memsize
        ysup = ysup.view(-1, ysup.size(-1))
        xsup_mask = xsup != 0
        ysup_mask = ysup != 0
        xsup_enc = self.memory_encoder(xsup, mask=xsup_mask)   # (batsize*memsize, seqlen, dim)
        ysup_enc = self.memory_decoder(ysup[:, 0], ysup[:, 1:],
                                       xsup_enc,  # (batsize*memsize, seqlen, dim)
                                       ctxmask=xsup_mask)
        xsup_enc = xsup_enc.view(xsup_size + (xsup_enc.size(-1),))     # (batsize, memsize, seqlen, dim)
        ysup_enc = ysup_enc.view(ysup_size[:2] + ysup_enc.size()[-2:])
        xsup_mask = xsup_mask.view(xsup_size)   # (batsize, memsize, seqlen)
        ysup_mask = ysup_mask.view(ysup_size)
        ysup_mask = ysup_mask[:, :, 1:]
        ysup = ysup.view(ysup_size)
        ysup = ysup[:, :, 1:]

        # region compute input encoding
        # compute encoding
        x_mask = x != 0
        x_enc_base = self.encoder(x, mask=x_mask)   # (batsize, seqlen, dim)

        # compute attention
        x_att, xsup_summ, _ = self.align_inputs(x_enc_base, xsup_enc,
                        mask=x_mask, supmask=xsup_mask.float() * supmask[:, :, None].float())
            # (batsize, seqlen, memsize, memseqlen) and (batsize, seqlen, dim)

        # compute encodings
        _x_enc_base = x_enc_base.contiguous().view(-1, x_enc_base.size(-1))
        _xsup_summ = xsup_summ.contiguous().view(-1, xsup_summ.size(-1))
        x_enc = self.merge_x(_xsup_summ, _x_enc_base)
        x_enc = x_enc.view(x_enc_base.size())
        # endregion

        # run decoder
        # if not istraining:
        #     assert(y.size(1) == 1)

        out, predactions = self.decoder(x_enc, y[:, 0], y[:, 1:], memids=ysup, memencs=ysup_enc,
                              xmask=x_mask, memmask=ysup_mask.float() * supmask[:, :, None].float(),
                              istraining=istraining)
        return out, predactions

    def align_inputs(self, x_enc_base, xsup_enc, mask=None, supmask=None):
        """
        :param x_enc_base:  3D float tensor (batsize, seqlen, dim)
        :param xsup_enc:    4D float tensor (batsize, memsize, seqlen, dim)
        :param mask:        2D mask for x_enc_base (batsize, seqlen)
        :param supmask:     3D mask for xsup_enc (batsize, memsize, seqlen)
        :return:
        """
        _x_enc_base = x_enc_base
        _xsup_enc = xsup_enc
        if self.input_enc_lin is not None:
            x_enc_base = self.input_enc_lin(x_enc_base)
        if self.supinput_enc_lin is not None:
            xsup_enc = self.supinput_enc_lin(xsup_enc)

        if mask is None:
            mask = torch.ones_like(x_enc_base[:, :, 0])
        if supmask is None:
            supmask = torch.ones_like(xsup_enc[:, :, :, 0])

        att_weights = torch.einsum("bsd,bmzd->bsmz", x_enc_base, xsup_enc)    # (batsize, xseqlen, memsize, xmemseqlen)
        # set attention to -infty for masked support tokens
        att_weights = att_weights + torch.log(supmask.unsqueeze(1).float())
        att_weights = att_weights / np.sqrt(xsup_enc.size(-1))
        att_weights_size = att_weights.size()
        att_weights = att_weights.view(att_weights.size(0), att_weights.size(1), -1)    # (batsize, xseqlen, memsize*xmemseqlen)
        att = torch.softmax(att_weights, -1)    # (batsize, seqlen, memsize*xmemseqlen)
        att = att.view(att_weights_size)  # (batsize, seqlen, memsize, memseqlen)
        att_weights = att_weights.view(att_weights_size)

        # TODO: more memory-efficient?
        summ = torch.einsum("bsmz,bmzd->bsd", att, _xsup_enc)
        return att, summ, att_weights


def create_lstm_model(encoder, vocsize, dim, numlayers=2, dropout=0., unktokens:Set[int]=None, eos_id=3, maxlen=100):
    unktokens = set() if unktokens is None else unktokens

    inplayer = DecoderInputLayer(vocsize, dim, unktoks=unktokens)
    outlayer = DecoderOutputLayer(dim, vocsize, unktoks=unktokens, dropout=0.)
    cell = LSTMDecoderCellWithMemory(inplayer, dim, outlayer=outlayer, numlayers=numlayers, eos_id=eos_id, dropout=dropout)
    dec = StateDecoderWithMemory(cell, maxtime=maxlen)
    memcell = InnerLSTMDecoderCell(inplayer, dim, numlayers=numlayers, eos_id=eos_id, dropout=dropout)
    memdec = StateInnerDecoder(memcell)
    m = MetaSeqMemNN(encoder, encoder, dec, memdec, dim, dropout)
    return m


class TrainModel(torch.nn.Module):
    def __init__(self, model:MetaSeqMemNN, tensor2tree:Callable=None, orderless:Set[str]=set(),
                 maxlen:int=100, smoothing:float=0., padid:int=0, **kw):
        super(TrainModel, self).__init__(**kw)
        self.model = model

        # CE loss
        self.ce = CELoss(ignore_index=padid,
                         smoothing=smoothing,
                         mode="probs")

        # accuracies
        self.accs = SeqAccuracies()
        self.accs.padid = padid
        self.accs.unkid = UNKID

        self.tensor2tree = tensor2tree
        self.orderless = orderless
        self.maxlen = maxlen
        self.treeacc = TreeAccuracy(tensor2tree=tensor2tree,
                                    orderless=orderless)

        self.metrics = [self.ce, self.accs, self.treeacc]

    def forward(self, x, y, xsup, ysup, supmask, istraining=True):
        probs, predactions = self.model(x, y, xsup, ysup, supmask, istraining)
        # _, predactions = probs.max(-1)
        outputs = [metric(probs, predactions, y[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, (probs, predactions)


class TestModel(torch.nn.Module):
    def __init__(self, model: MetaSeqMemNN, tensor2tree: Callable = None, orderless: Set[str] = set(),
                 maxlen: int = 100, smoothing: float = 0., padid: int = 0, **kw):
        super(TestModel, self).__init__(**kw)
        self.model = model

        # accuracies
        self.accs = SeqAccuracies()
        self.accs.padid = padid
        self.accs.unkid = UNKID

        self.tensor2tree = tensor2tree
        self.orderless = orderless
        self.maxlen = maxlen
        self.treeacc = TreeAccuracy(tensor2tree=tensor2tree,
                                    orderless=orderless)

        self.metrics = [self.accs, self.treeacc]

    def forward(self, x, y, xsup, ysup, supmask, istraining=False):
        probs, predactions = self.model(x, y, xsup, ysup, supmask, istraining)
        # _, predactions = probs.max(-1)
        outputs = [metric(probs, predactions, y[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, (probs, predactions)



def create_model(encoder_name="bert-base-uncased",
                 dec_vocabsize=None,
                 dec_layers=2,
                 dec_dim=200,
                 dropout=0.,
                 maxlen=30,
                 smoothing=0.,
                 tensor2tree=None,
                 tokenmasks=None,
                 ):
    if encoder_name != "bert-base-uncased":
        raise NotImplementedError(f"encoder '{encoder_name}' not supported yet.")
    pretrained = BertModel.from_pretrained(encoder_name)
    encoder = pretrained

    class BertEncoderWrapper(torch.nn.Module):
        def __init__(self, model, dropout=0., **kw):
            super(BertEncoderWrapper, self).__init__(**kw)
            self.inner_bert_model = model
            self.proj = torch.nn.Linear(pretrained.config.hidden_size, dec_dim, bias=False) \
                if dec_dim != pretrained.config.hidden_size \
                else None
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, input_ids, mask=None):
            ret, _ = self.inner_bert_model(input_ids, attention_mask=mask)
            if pretrained.config.hidden_size != dec_dim:
                ret = self.proj(ret)
                ret = self.dropout(ret)
            return ret

    encoder = BertEncoderWrapper(encoder, dropout=0.)
    unktokens = set(tokenmasks["_metarare"].nonzero()[:, 0].cpu().numpy())
    m = create_lstm_model(encoder, dec_vocabsize, dim=dec_dim, numlayers=dec_layers,
                          dropout=dropout, unktokens=unktokens, eos_id=3, maxlen=maxlen)

    orderless = {"op:and", "SW:concat"}

    trainmodel = TrainModel(m,
                            smoothing=smoothing,
                            tensor2tree=tensor2tree,
                            orderless=orderless,
                            maxlen=maxlen)

    testmodel = TestModel(m,
                            smoothing=smoothing,
                            tensor2tree=tensor2tree,
                            orderless=orderless,
                            maxlen=maxlen)

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


def infiter(a):
    while True:
        for ae in a:
            yield ae


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


def run(traindomains="ALL",
        domain="restaurants",
        supportsetting="lex",   # "lex" or "min"
        mincoverage=2,
        lr=0.0001,
        enclrmul=0.1,
        numbeam=1,
        cosinelr=False,
        warmup=0.,
        batsize=30,
        epochs=100,
        evalinterval=2,
        dropout=0.1,
        wreg=1e-9,
        gradnorm=3,
        gradacc=1,
        smoothing=0.,
        patience=20,
        gpu=-1,
        seed=123456789,
        encoder="bert-base-uncased",
        numlayers=2,
        hdim=200,
        numheads=8,
        maxlen=30,
        fullsimplify=True,
        abscontrib=1.,
        ):
    settings = locals().copy()
    print(json.dumps(settings, indent=4))
    # wandb.init(project=f"overnight_joint_pretrain_fewshot_{pretrainsetting}-{finetunesetting}-{domain}",
    #            reinit=True, config=settings)
    if traindomains == "ALL":
        alldomains = {"recipes", "restaurants", "calendar", "housing", "publications"}  # blocks
        traindomains = alldomains - {domain, }
    else:
        traindomains = set(traindomains.split("+"))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tt = q.ticktock("script")
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt.tick("loading data")
    traindl, validdl, testdl, nltok, flenc, tokenmasks = \
        load_data(traindomains=traindomains,
                  testdomain=domain,
                  supportsetting=supportsetting,
                  batsize=batsize,)
    tt.tock("data loaded")

    tt.tick("creating model")
    trainm, testm = create_model(encoder_name=encoder,
                                 dec_vocabsize=flenc.vocab.number_of_ids(),
                                 dec_layers=numlayers,
                                 dec_dim=hdim,
                                 dropout=dropout,
                                 smoothing=smoothing,
                                 maxlen=maxlen,
                                 tensor2tree=partial(_tensor2tree, D=flenc.vocab),
                                 tokenmasks=tokenmasks,
                                 )
    # print(trainm)
    tt.tock("model created")

    # region pretrain on all domains
    metrics = make_array_of_metrics("loss", "elem_acc", "seq_acc", "tree_acc")
    vmetrics = make_array_of_metrics("seq_acc", "tree_acc")
    xmetrics = make_array_of_metrics("seq_acc", "tree_acc")

    # region parameters
    def get_parameters(m, _lr, _enclrmul):
        bertparams = []
        otherparams = []
        for k, v in m.named_parameters():
            if "inner_bert_model" in k:
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
        paramgroups = get_parameters(_m, _lr=lr, _enclrmul=enclrmul)
        optim = torch.optim.Adam(paramgroups, lr=lr, weight_decay=wreg)
        return optim

    def clipgradnorm(_m=None, _norm=None):
        torch.nn.utils.clip_grad_norm_(_m.parameters(), _norm)

    eyt = q.EarlyStopper(vmetrics[1], patience=patience, min_epochs=30, more_is_better=True, remember_f=lambda: deepcopy(trainm.model))
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

    trainbatch = partial(q.train_batch, gradient_accumulation_steps=gradacc,
                                        on_before_optim_step=[lambda : clipgradnorm(_m=trainm, _norm=gradnorm)])

    trainepoch = partial(q.train_epoch, model=trainm,
                                        dataloader=traindl,
                                        optim=optim,
                                        losses=metrics,
                                        device=device,
                                        _train_batch=trainbatch,
                                        on_end=[lambda: lr_schedule.step()])

    validepoch = partial(q.test_epoch, model=testm,
                                       losses=xmetrics,
                                       dataloader=testdl,
                                       device=device,
                                       on_end=[lambda: eyt.on_epoch_end()])

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=validepoch,
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()])
    tt.tock("done training")

    testepoch =  partial(q.test_epoch, model=testm,
                                      losses=xmetrics,
                                      dataloader=testdl,
                                      device=device)

    testmsg = testepoch()
    tt.msg(testmsg)
    tt.tock("tested")


def run_experiments(domain="restaurants", gpu=-1, lr=0.0001, ftlr=0.0001, enclrmul=0.1, patience=10, cosinelr=False, fullsimplify=True, batsize=50,
                         smoothing=0., dropout=.1, numlayers=3, numheads=12, hdim=768, domainstart=False, gradacc=1, gradnorm=3,
                         numbeam=1, supportsetting="lex", abscontrib=.1, metarare="undefined", finetunesteps=1, gradmode="undefined",
                         maxfinetunesteps=30, evalinterval=5, epochs=25, injecttraindata=False, useadapters=False):
    ranges = {
        "lr": [lr],
        "ftlr": [ftlr],
        "enclrmul": [enclrmul],
        "warmup": [0],
        "epochs": [epochs],
        "numheads": [numheads],
        "numlayers": [numlayers],
        "dropout": [dropout],
        "smoothing": [smoothing],
        "hdim": [hdim],
        "numbeam": [numbeam],
        "batsize": [batsize],
        "seed": [87646464],
        "gradmode": ["none", "split", "inner:all+outer:noemb", "metarare"],
        "metarare": ["no", "emb", "outlin", "emb+outlin"]
    }
    p = __file__ + f".{domain}"
    if gradmode != "undefined":
        ranges["gradmode"] = [gradmode]
    if metarare != "undefined":
        ranges["metarare"] = [metarare]

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
                      domain=domain, fullsimplify=fullsimplify,
                      gpu=gpu, patience=patience, cosinelr=cosinelr,
                      domainstart=domainstart,
                      supportsetting=supportsetting,
                      abscontrib=abscontrib,
                      finetunesteps=finetunesteps,
                      gradacc=gradacc, gradnorm=gradnorm,
                      maxfinetunesteps=maxfinetunesteps,
                      evalinterval=evalinterval,
                      injecttraindata=injecttraindata,
                      useadapters=useadapters)


def run_experiments_seed(domain="restaurants", gpu=-1, lr=0.0001, ftlr=0.0001, patience=10, cosinelr=False, fullsimplify=True, batsize=50,
                         smoothing=0., dropout=.1, numlayers=3, numheads=12, hdim=768, domainstart=False, gradacc=3,
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
    ret = q.argprun(run)
    # print(ret)
    # q.argprun(run_experiments)
    # fire.Fire(run_experiments)