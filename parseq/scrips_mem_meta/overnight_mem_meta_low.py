# encoding: utf-8
"""
A script for running the following zero-shot domain transfer experiments:
* dataset: Overnight
* model: SEG-NMT
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
from IPython import embed

import qelos as q   # branch v3
import numpy as np
import torch
from nltk import Tree
from torch.utils.data import DataLoader

from parseq.datasets import OvernightDatasetLoader, pad_and_default_collate, autocollate, Dataset
from parseq.decoding import merge_metric_dicts, SeqDecoder, StopDecoding
from parseq.eval import SeqAccuracies, TreeAccuracy, make_array_of_metrics, CELoss
from parseq.grammar import tree_to_lisp_tokens, lisp_to_tree
from parseq.nn import SGRUCell
from parseq.states import State, TrainableDecodableState
from parseq.transitions import TransitionModel, LSTMCellTransition
from parseq.vocab import SequenceEncoder, Vocab
from transformers import AutoTokenizer, AutoModel, BartConfig, BartModel, BartForConditionalGeneration, BertLayer, \
    BertModel
from transformers.activations import ACT2FN
from transformers.modeling_bart import SinusoidalPositionalEmbedding, DecoderLayer, SelfAttention, LayerNorm

UNKID = 1

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
            nl_mode="basic",
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

    if nl_mode == "basic":
        nl_tokenizer = SequenceEncoder(tokenizer=lambda x: x.split(),
                                 add_start_token=True, add_end_token=True)
        for example in ds.examples:
            nl_tokenizer.inc_build_vocab(example[0], seen=example[3] in ("train", "support") if example[4] != testdomain else example[3] == "support")
        nl_tokenizer.finalize_vocab(min_freq=min_freq, top_k=top_k)
        nl_tok_f = lambda x: nl_tokenizer.convert(x, return_what="tensor")
    else:
        nl_tokenizer = AutoTokenizer.from_pretrained(nl_mode)
        nl_tok_f = lambda x: nl_tokenizer(x, return_tensors="pt")[0]
    def tokenize(x):
        ret = (nl_tok_f(x[0]),
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
    def __init__(self, dim, embdim, vocsize, unktoks:Set[int]=set(), dropout=0., **kw):
        """
        :param dim:     dimension of input vectors
        :param vocsize: number of unique words in vocabulary
        :param unktoks: set of integer ids of tokens that are unknown. Their generation probabilities will be set to zero.
        :param kw:
        """
        super(DecoderOutputLayer, self).__init__(**kw)
        self.dim = dim
        self.embdim = embdim
        self.vocsize = vocsize
        self.attvectormasker = torch.nn.Linear(dim * 2 + embdim, dim * 2 + embdim)
        self.mixer = torch.nn.Linear(dim*2, 2)
        self.outlin = torch.nn.Linear(dim*2, vocsize)
        # self.bilinear_lin = torch.nn.Linear(dim*2, dim*2, bias=False)
        self.dropout = dropout
        unktok_mask = torch.ones(vocsize)
        for unktok in unktoks:
            unktok_mask[unktok] = 0
        self.register_buffer("unktok_mask", unktok_mask)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.attvectormasker.bias, +3)

    def forward(self,
                enc,
                encsumm,
                embsumm,
                memencs=None,
                memencsumm=None,
                memembsumm=None,
                memmask=None,
                memids=None):
        """
        :param enc:     (batsize, dim)
        :param encsumm:    (batsize, indim)
        :param memids:  (batsize, memsize, memseqlen) - integer ids of sequences in given memory
        :param memkeys: (batsize, memsize, memseqlen, memdim) - input summaries
        :param memencs: (batsize, memsize, memseqlen, memdim) - output states of memory decoder
        :param memmask: (batsize, memsize, memseqlen) - bool mask whether memory token is padding
        :return:
        """
        if memmask is None:
            memmask = torch.ones_like(memids).float()
        # compute
        # TODO: use only summary
        query = torch.cat([embsumm, torch.zeros_like(encsumm), enc], -1)
        # attvector_mask = torch.sigmoid(self.attvectormasker(query))
        # query = query * attvector_mask
        keys = torch.cat([memembsumm, torch.zeros_like(memencsumm), memencs], -1)

        # compute probs from memory
        probs_mem = torch.zeros(query.size(0), self.vocsize, device=query.device)
            # (batsize, vocsize)

        # compute attentions to memory
        # query = self.bilinear_lin(query)
        mem_weights = torch.einsum("bd,bmzd->bmz", query, keys)
        mem_weights = mem_weights + torch.log(memmask.float())
        # mem_weights_size = mem_weights.size()
        mem_weights = mem_weights.view(mem_weights.size(0), -1)
        mem_alphas = torch.softmax(mem_weights, -1) # (batsize, memsize*memseqlen)
        mem_enc_summ = mem_alphas[:, :, None] * memencs.contiguous().view(memencs.size(0), memencs.size(1) * memencs.size(2), -1)
        mem_enc_summ = mem_enc_summ.sum(1)
        # mem_key_summ = mem_alphas[:, :, None] * memkeys.contiguous().view(memkeys.size(0), memkeys.size(1) * memkeys.size(2), -1)
        # mem_key_summ = mem_key_summ.sum(1)
        mem_toks = memids.contiguous().view(memids.size(0), -1)  # (batsize, memsize*memseqlen)
        probs_mem = probs_mem.scatter_add(1, mem_toks, mem_alphas)

        h = torch.cat([encsumm, enc], -1)
        logits_gen = self.outlin(h)
        mask_gen = self.unktok_mask[None, :].float()
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
        return probs, mem_enc_summ


class InnerLSTMDecoderCell(TransitionModel):
    """
        LSTM cell based decoder cell.
        This decoder cell will be used to "encode" the output parts of the memory.
    """
    def __init__(self, inplayer:DecoderInputLayer=None,
                 transition:LSTMCellTransition=None,
                 dropout=0.,
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
        self.lstm_transition = transition
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
        inp = self.inplayer(x.prev_actions, x.prev_summ)

        enc, new_lstmstate = self.lstm_transition(inp, x.lstmstate)
        # x.lstmstate = new_lstmstate

        alphas, summ, _ = self.inp_att(enc, x.ctx, x.ctx, x.ctxmask)
        x.prev_summ = summ
        return enc, alphas, x


class InnerDecoder(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, starttokens, y, ctx, ctxmask=None, prev_summ=None):
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
    def create_state(cls, cell, starttokens, y, ctx, ctxmask=None, prev_summ=None):
        state = cell.get_init_state(ctx.size(0), device=ctx.device)
        state.prev_actions = starttokens
        state.prev_summ = prev_summ
        state.ctx = ctx
        state.gold = y
        state.ctxmask = ctxmask
        return state

    def forward(self, starttokens, y, ctx, ctxmask=None, prev_summ=None):
        # create state
        state = self.create_state(self.cell, starttokens, y, ctx, ctxmask=ctxmask, prev_summ=prev_summ)
        state.start_decoding()

        alphas = []
        encs = []

        i = 0

        all_terminated = state.all_terminated()
        while not all_terminated:
            try:
                enc, att, state = self.cell(state)
                # feed next
                goldactions = state.get_gold(i)
                state.step(goldactions)
                all_terminated = state.all_terminated()
                alphas.append(att)
                encs.append(enc)
                i += 1
            except StopDecoding as e:
                all_terminated = True

        alphas = torch.stack(alphas, 1)
        encs = torch.stack(encs, 1)

        return encs, alphas


class LSTMDecoderCellWithMemory(TransitionModel):
    """
    Decoder cell for use with StateDecoderWithMemory.
    """
    def __init__(self, inplayer:DecoderInputLayer=None,
                 transition:LSTMCellTransition=None,
                 dropout=0.,
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
        # self.dim = dim
        # self.encdim = encdim if encdim is not None else dim
        # dims = [inplayer.outdim] + [dim]*numlayers
        self.lstm_transition = transition
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
        inp = self.inplayer(x.prev_actions, x.prev_summ)

        enc, new_lstmstate = self.lstm_transition(inp, x.lstmstate)
        # x.lstmstate = new_lstmstate

        alphas, encsumm, _ = self.inp_att(enc, x.x_enc, x.x_enc, x.xmask)
        embsumm = torch.einsum("bz,bzd->bd", alphas, x.x_emb)
        x.prev_summ = encsumm

        out, mem_summ = self.outlayer(
            enc,
            encsumm,
            embsumm,
            memencs=x.memencs,
            memencsumm=x.memencsumm,
            memembsumm=x.memembsumm,
            memids=x.memids,
            memmask=x.memmask)
        x.prev_mem_summ = mem_summ
        return out, x


class DecoderWithMemory(ABC, torch.nn.Module):
    """ Defines forward interface for all memory based decoders. """
    @abstractmethod
    def forward(self,
                x_enc,              # TODO: comments
                starttokens,
                y=None,
                x_emb=None,
                memids=None,
                memencs=None,
                memencsumm=None,
                memembsumm=None,
                xmask=None,
                memmask=None,
                prev_summ=None,
                istraining: bool = True)->Dict:
        pass


class StateDecoderWithMemory(DecoderWithMemory):
    """ State-based decoder with memory.
        Wraps a SeqDecoder and creates a state from args.
        Used for outer decoder.
        Decoder contains a memory.
    """
    def __init__(self, cell:TransitionModel, maxtime=100, **kw):
        super(StateDecoderWithMemory, self).__init__(**kw)
        self.decoder = SeqDecoder(cell, maxtime=maxtime)
        self.cell = cell

    @classmethod
    def create_state(cls, cell,
                x_enc,
                starttokens,
                y=None,
                x_emb=None,
                memids=None,
                memencs=None,
                memencsumm=None,
                memembsumm=None,
                xmask=None,
                memmask=None,
                prev_summ=None):
        state = cell.get_init_state(x_enc.size(0), device=x_enc.device)
        state.prev_actions = starttokens
        state.prev_summ = prev_summ if prev_summ is not None else torch.zeros_like(x_enc[:, 0])
        state.prev_mem_summ = torch.zeros_like(memencs[:, 0, 0])
        state.x_enc = x_enc
        state.x_emb = x_emb
        state.gold = y
        state.memids = memids
        state.memencs = memencs
        state.memencsumm = memencsumm
        state.memembsumm = memembsumm
        state.xmask = xmask
        state.memmask = memmask
        return state

    def forward(self,
                x_enc,
                starttokens,
                y=None,
                x_emb=None,
                memids=None,
                memencs=None,
                memencsumm=None,
                memembsumm=None,
                xmask=None,
                memmask=None,
                prev_summ=None,
                istraining:bool=True):
        # create state
        state = self.create_state(self.cell,
                                  x_enc,
                                  starttokens,
                                  y=y,
                                  x_emb=x_emb,
                                  memids=memids,
                                  memencs=memencs,
                                  memencsumm=memencsumm,
                                  memembsumm=memembsumm,
                                  xmask=xmask,
                                  memmask=memmask,
                                  prev_summ=prev_summ)

        # decode from state
        _, _, out, predactions, golds = self.decoder(state, tf_ratio=1. if istraining else 0., return_all=True)
        return out, predactions


class MetaSeqMemNN(torch.nn.Module):
    """
    Top-level module for memory-based meta learning for seq2seq
    """
    def __init__(self, encoder, memory_encoder,
                 decoder:DecoderWithMemory, memory_decoder:InnerDecoder,
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

    def forward(self, x, y, xmem, ymem, memmask, istraining=None):
        """
        :param x:       2D (batsize, seqlen) int tensor for input
        :param xmem:    3D (batsize, memsize, seqlen) int tensor for input side of support set for every example
        :param y:       2D (batsize, seqlen) int tensor for output (can be (batsize, 1) during test)
        :param ymem:    3D (batsize, memsize, seqlen) int tensor for output side of support set for every example
        :param memmask: 2D (batsize, memsize) bool tensor specifying which memory entries are non-entries
                        (! non-entries must consist of some token(s) !)
        :return:
        """
        istraining = self.training if istraining is None else istraining
        # encode the input and output sides of the support set
        xmem_size = xmem.size()
        ymem_size = ymem.size()
        xmem = xmem.view(-1, xmem.size(-1))      # flatten batsize and memsize
        ymem = ymem.view(-1, ymem.size(-1))
        xmem_mask = xmem != 0
        ymem_mask = ymem != 0
        xmem_enc_z, xmem_enc, xmem_emb = self.memory_encoder(xmem, mask=xmem_mask)   # (batsize*memsize, seqlen, dim)

        ymem_enc, ymem_alphas = self.memory_decoder(
            ymem[:, 0],
            ymem[:, 1:],
            xmem_enc,  # (batsize*memsize, seqlen, dim)
            ctxmask=xmem_mask,
            prev_summ=xmem_enc_z)
        """ ymem_enc:    (batsize, outlen, dim) sequence of decoder states
            ymem_alphas: (batsize, outlen, inplen) sequence of attentions computed from each decoder state """
        # xmem_enc = xmem_enc.view(xmem_size + (xmem_enc.size(-1),))     # (batsize, memsize, seqlen, dim)
        # ymem_alphas = ymem_alphas.view(ymem_size[:2] + ymem_alphas.size()[-2:])
        ymem_enc = ymem_enc.view(ymem_size[:2] + ymem_enc.size()[-2:])
        # xmem_mask = xmem_mask.view(xmem_size)   # (batsize, memsize, seqlen)

        # compute memory inputs
        ymem_enc_summ = torch.einsum("bsz,bzd->bsd", ymem_alphas, xmem_enc)
        ymem_emb_summ = torch.einsum("bsz,bzd->bsd", ymem_alphas, xmem_emb)

        ymem_enc_summ = ymem_enc_summ.view(ymem_size[:2] + ymem_enc_summ.size()[-2:])
        ymem_emb_summ = ymem_emb_summ.view(ymem_size[:2] + ymem_emb_summ.size()[-2:])

        ymem_mask = ymem_mask.view(ymem_size)
        ymem_mask = ymem_mask[:, :, 1:]
        ymem = ymem.view(ymem_size)
        ymem = ymem[:, :, 1:]

        # compute encoding
        x_mask = x != 0
        x_enc_z, x_enc, x_emb = self.encoder(x, mask=x_mask)   # (batsize, seqlen, dim)

        # run decoder
        # if not istraining:
        #     assert(y.size(1) == 1)

        out, predactions = self.decoder(x_enc,
                                        y[:, 0],
                                        y[:, 1:],
                                        x_emb=x_emb,
                                        memids=ymem,
                                        memencs=ymem_enc,
                                        memencsumm=ymem_enc_summ,
                                        memembsumm=ymem_emb_summ,
                                        xmask=x_mask,
                                        memmask=ymem_mask.float() * memmask[:, :, None].float(),
                                        prev_summ=x_enc_z,
                                        istraining=istraining)
        return out, predactions


def create_lstm_model(encoder, vocsize, dim=-1, embdim=-1, numlayers=2, dropout=0., unktokens:Set[int]=None, eos_id=3, maxlen=100):
    unktokens = set() if unktokens is None else unktokens

    inplayer = DecoderInputLayer(vocsize, dim, unktoks=unktokens)
    outlayer = DecoderOutputLayer(dim, embdim, vocsize, unktoks=unktokens, dropout=0.)

    dims = [inplayer.outdim] + [dim] * numlayers
    lstms = [torch.nn.LSTMCell(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
    lstm_transition = LSTMCellTransition(*lstms, dropout=dropout)

    cell = LSTMDecoderCellWithMemory(inplayer, lstm_transition, outlayer=outlayer, eos_id=eos_id, dropout=dropout)
    dec = StateDecoderWithMemory(cell, maxtime=maxlen)
    memcell = InnerLSTMDecoderCell(inplayer, lstm_transition, eos_id=eos_id, dropout=dropout)
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
        self.model.eval()
        probs, predactions = self.model(x, y, xsup, ysup, supmask, istraining)
        # _, predactions = probs.max(-1)
        outputs = [metric(probs, predactions, y[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, (probs, predactions)


def cosine_sim(a, b):
    d = np.dot(a, b)
    n = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
    return d/n


def create_nl_emb(D:Vocab, min_freq:int=0):
    class CustomEmb(torch.nn.Module):
        def __init__(self,
                     m:torch.nn.Embedding,
                     m2:torch.nn.Embedding,
                     mix, mapper=None, **kw):
            super(CustomEmb, self).__init__(**kw)
            self.main_m = m
            self.aux_m = m2
            self.register_buffer("mix", mix)
            self.register_buffer("mapper", mapper)
            self.embedding_dim = self.main_m.embedding_dim
            self.num_embeddings = self.main_m.num_embeddings

        def forward(self, x:torch.Tensor):
            if self.mapper is not None:
                x = self.mapper[x]
            main_emb = self.main_m(x)
            aux_emb = self.aux_m(x)
            mix = self.mix[x].unsqueeze(-1)
            emb = mix * aux_emb + (1 - mix) * main_emb
            return emb

    glove_vectors, glove_D = q.VectorLoader.load_glove_data("glove.300d")

    W = torch.zeros(D.number_of_ids(), glove_vectors.shape[1])
    switch = torch.zeros(D.number_of_ids())
    mapper = torch.arange(D.number_of_ids())
    for k, v in D.D.items():
        if k in glove_D:
            W[v, :] = torch.tensor(glove_vectors[glove_D[k]])
            switch[v] = 1
        if k not in glove_D and D.counts[k] < min_freq:
            mapper[k] = D[D.unktoken]
    glove_m = torch.nn.Embedding.from_pretrained(W, padding_idx=D[D.padtoken])

    vanilla_m = torch.nn.Embedding(D.number_of_ids(), glove_m.embedding_dim, padding_idx=D[D.padtoken])

    m = CustomEmb(vanilla_m, glove_m, switch, mapper=mapper)
    return m


def create_model(inpD=None,
                 outD=None,
                 num_layers=2,
                 dim=200,
                 dropout=0.,
                 maxlen=30,
                 smoothing=0.,
                 tensor2tree=None,
                 tokenmasks=None,
                 ):

    # TODO: create glove embeddings using dictionary

    class LSTMEncoder(torch.nn.Module):
        def __init__(self, emb, dim, numlayers=2, dropout=0., **kw):
            super(LSTMEncoder, self).__init__(**kw)
            self.emb = emb
            if self.emb.embedding_dim != dim:
                self.adapter = torch.nn.Linear(self.emb.embedding_dim, dim)
            else:
                self.adapter = None
            self.lstm = torch.nn.LSTM(dim, dim//2, numlayers, dropout=dropout, bidirectional=True)
            self.adapt_z = torch.nn.Linear(dim, dim)

        def forward(self, x, mask=None):
            emb = self.emb(x)
            if self.adapter is not None:
                _emb = self.adapter(emb)
            else:
                _emb = emb
            packed_emb, unsorter = q.seq_pack(_emb, mask)
            enc_packed, states = self.lstm(packed_emb)
            enc, recon_mask = q.seq_unpack(enc_packed, unsorter)
            z = states[0].index_select(1, unsorter)[-2:]\
                .transpose(0, 1).contiguous()\
                .view(enc.size(0), enc.size(2))
            z = torch.tanh(self.adapt_z(z))
            return z, enc, emb

    emb = create_nl_emb(inpD)
    encoder = LSTMEncoder(emb, dim, numlayers=num_layers, dropout=dropout)

    unktokens = set(tokenmasks["_metarare"].nonzero()[:, 0].cpu().numpy())
    m = create_lstm_model(encoder, outD.number_of_ids(),
                          dim=dim, embdim=emb.embedding_dim,
                          numlayers=num_layers,
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
        else:
            break
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
        lr=0.001,
        enclrmul=0.1,
        numbeam=1,
        cosinelr=False,
        warmup=0.,
        batsize=10,
        epochs=100,
        evalinterval=2,
        dropout=0.2,
        wreg=1e-9,
        gradnorm=3,
        gradacc=1,
        smoothing=0.,
        patience=20,
        gpu=-1,
        seed=123456789,
        encoder="bert-base-uncased",
        numlayers=2,
        hdim=100,
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
    traindl, validdl, testdl, nlenc, flenc, tokenmasks = \
        load_data(traindomains=traindomains,
                  testdomain=domain,
                  supportsetting=supportsetting,
                  batsize=batsize,)
    tt.tock("data loaded")

    tt.tick("creating model")
    trainm, testm = create_model(inpD=nlenc.vocab,
                                 outD=flenc.vocab,
                                 num_layers=numlayers,
                                 dim=hdim,
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
            if ".aux_m." in k:
                continue
            if "inner_bert_model" in k:
                bertparams.append(v)
            else:
                otherparams.append(v)
        # if len(bertparams) == 0:
        #     raise Exception("No encoder parameters found!")
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

    tt.tick("making predictions")
    testm.eval()
    inps, golds, predictions = [], [], []
    for batch in testdl:
        batch = q.recmap(batch, lambda x: x.to(device) if hasattr(x, "to") else x)
        outs = testm(*batch)
        for inpse, goldse, outse in zip(list(batch[0]), list(batch[1]), list(outs[1][1])):
            inpstr = nlenc.vocab.tostr(inpse)
            goldstr = flenc.vocab.tostr(goldse)
            predstr = flenc.vocab.tostr(outse)
            inps.append(inpstr)
            golds.append(goldstr)
            predictions.append(predstr)
    tt.tock("made predictions")
    embed()


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