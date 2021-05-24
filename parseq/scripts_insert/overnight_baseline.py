import math
import re

from prompt_toolkit.formatted_text import PygmentsTokens

import json
import random
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Dict, List

import torch
import wandb
from nltk import Tree
import numpy as np
from torch.utils.data import DataLoader

import qelos as q

from parseq.datasets import OvernightDatasetLoader, autocollate, Dataset
from parseq.eval import make_array_of_metrics
from parseq.grammar import are_equal_trees, lisp_to_tree, tree_to_lisp_tokens
from parseq.scripts_insert.overnight_treeinsert import extract_info
from parseq.scripts_insert.util import reorder_tree, flatten_tree
from parseq.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


ORDERLESS = {"op:and", "SW:concat", "filter", "call-SW:concat"}


def tree_to_seq(x:Tree):
    xstr = tree_to_lisp_tokens(x, _bracketize_leafs=False)
    # xstr = ["@BOS@"] + xstr + ["@EOS@"]
    return xstr


def make_numbered_tokens(x:List[str]):
    counts = {}
    y = []
    for xe in x:
        if xe not in counts:
            counts[xe] = 0
        counts[xe] += 1
        y.append(f"{xe}::{counts[xe]}")
    return y


def load_ds(domain="restaurants", nl_mode="bert-base-uncased",
            trainonvalid=False, noreorder=False, numbered=False):
    """
    Creates a dataset of examples which have
    * NL question and tensor
    * original FL tree
    * reduced FL tree with slots (this is randomly generated)
    * tensor corresponding to reduced FL tree with slots
    * mask specifying which elements in reduced FL tree are terminated
    * 2D gold that specifies whether a token/action is in gold for every position (compatibility with MML!)
    """
    # orderless = {"op:and", "SW:concat"}     # only use in eval!!
    orderless = ORDERLESS

    if domain.startswith("syn"):
        if domain.startswith("syn-rep-"):
            words = list(set(["cat", "dog", "person", "camera", "tv", "woman", "man", "sum", "ting", "wong", "blackbird",
                     "plane", "computer", "pc", "bert", "captain", "slow", "went", "home", "car", "bike", "train", 
                     "fox", "virus", "vaccine", "pharma", "company", "eu", "uk", "us", "israel", "iran", "syria", 
                     "russia", "denmark", "capital", "wallstreetbets", "reddit", "option", "short", "squeeze"]))
            
            NUMTRAIN = 400
            NUMVALID = 100
            NUMTEST = 100
            L = 30
            assert L < len(set(words))
            rep = int(domain[len("syn-rep-"):])
            examples = []
            for NUMEX, splitname in zip([NUMTRAIN, NUMVALID, NUMTEST], ["train", "valid", "test"]):
                for i in range(NUMEX):
                    words_i = words[:]
                    random.shuffle(words_i)
                    example = []
                    k = 0
                    for j in range(L):
                        example.append(words_i.pop(0))
                        k += 1
                        if k == rep:
                            example.append("and")
                            k = 0
                        if len(example) > L:
                            break
                    examples.append((" ".join(example), example, splitname))
            ds = Dataset(examples)
        else:
            raise Exception(f"Unknown domain '{domain}'")
    elif domain.startswith("top-eval"):
        examples = []
        with open("../../datasets/top/eval.tsv") as f:
            lines = f.readlines()
            for line in lines:
                splits = line.strip().split("\t")
                nl = splits[1]
                if domain == "top-eval-nl":
                    fl = nl.replace("'", "|").replace("\"", "|").replace("\s+", " ")
                    fl = f"[ SENTENCE {fl} ]"
                else:
                    fl = splits[2].replace("'", "|").replace("\"", "|").replace("[", " [ ").replace("]", " ] ").replace("\s+", " ")
                fl = lisp_to_tree(fl, brackets="[]")
                examples.append((nl, tree_to_seq(fl)))
        splits = ["train"]*500 + ["valid"]*50 + ["test"]*50
        examples = examples[:len(splits)]
        examples = [(x[0], x[1], y) for x, y in zip(examples, splits)]
        ds = Dataset(examples)
    elif domain == "geo":
        ds = GeoDatasetLoader().load(trainonvalid=trainonvalid)
    else:
        ds = OvernightDatasetLoader(simplify_mode="none").load(domain=domain, trainonvalid=trainonvalid)
        # ds contains 3-tuples of (input, output tree, split name)

        if not noreorder:
            ds = ds.map(lambda x: (x[0], reorder_tree(x[1], orderless=orderless), x[2]))
        ds = ds.map(lambda x: (x[0], tree_to_seq(x[1]), x[2]))

    if numbered:
        ds = ds.map(lambda x: (x[0], make_numbered_tokens(x[1]), x[2]))

    vocab = Vocab(padid=0, startid=2, endid=3, unkid=1)
    vocab.add_token("@BOS@", seen=np.infty)
    vocab.add_token("@EOS@", seen=np.infty)
    vocab.add_token("@STOP@", seen=np.infty)

    nl_tokenizer = BertTokenizer.from_pretrained(nl_mode)

    tds, vds, xds = ds[lambda x: x[2] == "train"], \
                    ds[lambda x: x[2] == "valid"], \
                    ds[lambda x: x[2] == "test"]

    seqenc = SequenceEncoder(vocab=vocab, tokenizer=lambda x: x,
                             add_start_token=False, add_end_token=False)
    for example in tds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=True)
    for example in vds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=False)
    for example in xds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=False)
    seqenc.finalize_vocab(min_freq=0)

    def mapper(x):
        seq = seqenc.convert(x[1], return_what="tensor")
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0], seq)
        return ret

    tds_seq = tds.map(mapper)
    vds_seq = vds.map(mapper)
    xds_seq = xds.map(mapper)
    return tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc, orderless


class SeqInsertionTagger(torch.nn.Module):
    """ A tree insertion tagging model takes a sequence representing a tree
        and produces distributions over tree modification actions for every (non-terminated) token.
    """
    @abstractmethod
    def forward(self, tokens:torch.Tensor, **kw):
        """
        :param tokens:      (batsize, seqlen)       # all are open!
        :return:
        """
        pass


class TokenEmb(torch.nn.Module):
    def __init__(self, vocab, dim, factorized=False, pooler="sum", **kw):
        super(TokenEmb, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.factorized = factorized
        self.pooler = pooler
        if self.factorized:
            self.token_vocab = {}
            next_token_id = 0
            self.number_vocab = {}
            next_number_id = 0
            self.register_buffer("id_to_token", torch.zeros(self.vocabsize, dtype=torch.long))
            self.register_buffer("id_to_number", torch.zeros(self.vocabsize, dtype=torch.long))
            for k, v in self.vocab.D.items():
                if "::" in k:
                    token, number = k.split("::")
                else:
                    token, number = k, "0"
                if token not in self.token_vocab:
                    self.token_vocab[token] = next_token_id
                    next_token_id += 1
                self.id_to_token[v] = self.token_vocab[token]
                if number not in self.number_vocab:
                    self.number_vocab[number] = next_number_id
                    next_number_id += 1
                self.id_to_number[v] = self.number_vocab[number]
            self.token_emb = torch.nn.Embedding(max(self.token_vocab.values()) + 1, dim)
            self.number_emb = torch.nn.Embedding(max(self.number_vocab.values()) + 1, dim)
        else:
            self.emb = torch.nn.Embedding(self.vocabsize, dim)

    def forward(self, x):
        if self.factorized:
            token = self.id_to_token[x]
            tokenemb = self.token_emb(token)
            number = self.id_to_number[x]
            numberemb = self.number_emb(number)
            if self.pooler == "sum":
                emb = tokenemb + numberemb
            elif self.pooler == "max":
                emb = torch.max(tokenemb, numberemb)
            elif self.pooler == "mul":
                emb = tokenemb * numberemb
            else:
                raise Exception(f"unknown pooler: '{self.pooler}'")
        else:
            emb = self.emb(x)
        return emb


class TokenOut(torch.nn.Module):
    def __init__(self, dim, vocab, factorized=False, pooler="sum", **kw):
        super(TokenOut, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.factorized = factorized
        self.pooler = pooler
        if self.factorized:
            self.token_vocab = {}
            next_token_id = 0
            self.number_vocab = {}
            next_number_id = 0
            self.register_buffer("token_to_id", torch.zeros(self.vocabsize, dtype=torch.long))
            self.register_buffer("number_to_id", torch.zeros(self.vocabsize, dtype=torch.long))
            self.register_buffer("id_to_token", torch.zeros(self.vocabsize, dtype=torch.long))
            self.register_buffer("id_to_number", torch.zeros(self.vocabsize, dtype=torch.long))
            for k, v in self.vocab.D.items():
                if "::" in k:
                    token, number = k.split("::")
                else:
                    token, number = k, "0"
                if token not in self.token_vocab:
                    self.token_vocab[token] = next_token_id
                    next_token_id += 1
                self.token_to_id[self.token_vocab[token]] = v
                self.id_to_token[v] = self.token_vocab[token]
                if number not in self.number_vocab:
                    self.number_vocab[number] = next_number_id
                    next_number_id += 1
                self.number_to_id[self.number_vocab[number]] = v
                self.id_to_number[v] = self.number_vocab[number]
            self.token_lin = torch.nn.Linear(dim, max(self.token_vocab.values()) + 1)
            self.number_lin = torch.nn.Linear(dim, max(self.number_vocab.values()) + 1)
            self.token_to_id = self.token_to_id[:next_token_id]
            self.number_to_id = self.number_to_id[:next_number_id]
        else:
            self.lin = torch.nn.Linear(dim, self.vocabsize)

    def forward(self, x):
        if self.factorized:
            if self.pooler == "sum":
                # compute scores for tokens and numbers separately
                tokenscores = self.token_lin(x)     # (batsize, seqlen, numtok)
                numberscores = self.number_lin(x)   # (batsize, seqlen, numnum)
                # distribute them and sum up
                tokenscores = torch.index_select(tokenscores, -1, self.id_to_token)
                numberscores = torch.index_select(numberscores, -1, self.id_to_number)
                scores = tokenscores + numberscores
            else:
                raise Exception(f'Pooler "{self.pooler}" is not supported yet.')
        else:
            scores = self.lin(x)
        return scores


class GRUDecoderCell(torch.nn.Module):
    def __init__(self, dim, indim=None, vocab:Vocab=None, numlayers:int=2, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased",
                 vocab_factorized=False, mode="baseline", **kw):
        super(GRUDecoderCell, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.indim = indim if indim is not None else dim

        self.emb = TokenEmb(vocab, self.indim, factorized=vocab_factorized, pooler="sum")

        self.decoder = torch.nn.ModuleList([torch.nn.GRUCell((self.indim + self.dim if i == 0 else self.dim), self.dim)
                                            for i in range(numlayers)])
        self.dropout = torch.nn.Dropout(dropout)

        self.preout = torch.nn.Linear(self.dim*2, self.dim)
        self.out = torch.nn.Linear(self.dim, self.vocabsize)
        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname,
                                                    hidden_dropout_prob=min(dropout, 0.2),
                                                    attention_probs_dropout_prob=min(dropout, 0.1))
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bertname)
        # def set_dropout(m:torch.nn.Module):
        #     if isinstance(m, torch.nn.Dropout):
        #         m.p = dropout
        # self.bert_model.apply(set_dropout)

        self.adapter = None
        if self.bert_model.config.hidden_size != self.dim:
            self.adapter = torch.nn.Linear(self.bert_model.config.hidden_size, self.dim, bias=False)

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
        encs = self.bert_model(x, attention_mask=encmask)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def test_forward(self, tokens: torch.Tensor = None, enc=None, encmask=None, cache=None):
        return self(tokens=tokens, enc=enc, encmask=encmask, cache=cache)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
        """
        :param tokens:      (batsize, ) int ids
        :param enc:         (batsize, seqlen, dim)
        :param encmask:     (batsize, seqlen)
        :param cache:       list of rnn states for every rnn layer
        :return:
        """
        embs = self.emb(tokens)
        if cache is None:
            rnn_states = [torch.zeros(tokens.size(0), self.dim, device=tokens.device) for _ in range(len(self.decoder))]
            prev_summ = torch.zeros(tokens.size(0), self.dim, device=tokens.device)
        else:
            rnn_states, prev_summ = cache

        # pass through rnns
        rnn_outs = []
        lower_rep = torch.cat([embs, prev_summ], 1)
        for rnn_layer, rnn_state in zip(self.decoder, rnn_states):
            lower_rep = self.dropout(lower_rep)
            new_rnn_out = rnn_layer(lower_rep, rnn_state)
            rnn_outs.append(new_rnn_out)
            lower_rep = new_rnn_out

        # use attention
        # (batsize, seqlen, dim) and (batsize, dim)
        scores = torch.einsum("bsd,bd->bs", enc, lower_rep)
        scores = scores + (encmask.float() - 1) * 99999
        weights = torch.softmax(scores, 1)

        summaries = torch.einsum("bs,bsd->bd", weights, enc)
        c = torch.cat([lower_rep, summaries], 1)
        c = self.preout(c)
        c = torch.nn.functional.leaky_relu(c, 0.1)

        logits = self.out(c)
        return logits, (rnn_outs, summaries)


class TMDecoderCell(torch.nn.Module):
    def __init__(self, dim, indim=None, vocab:Vocab=None, numlayers:int=2, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased",
                 vocab_factorized=False, **kw):
        super(TMDecoderCell, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.indim = indim if indim is not None else dim

        self.emb = TokenEmb(vocab, self.indim, factorized=vocab_factorized, pooler="sum")

        decoderconfig = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim*4,
                                          num_layers=numlayers, num_heads=numheads, dropout_rate=dropout,
                                          use_relative_position=False, use_causal_mask=True, n_positions=maxpos)
        decoderconfig.is_decoder = True
        self.decoder = TransformerStack(decoderconfig, embed_tokens=None)
        self.dropout = torch.nn.Dropout(dropout)

        self.preout = torch.nn.Linear(self.dim, self.dim)
        self.out = torch.nn.Linear(self.dim, self.vocabsize)
        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname,
                                                    hidden_dropout_prob=min(dropout, 0.2),
                                                    attention_probs_dropout_prob=min(dropout, 0.1))
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bertname)
        # def set_dropout(m:torch.nn.Module):
        #     if isinstance(m, torch.nn.Dropout):
        #         m.p = dropout
        # self.bert_model.apply(set_dropout)

        self.adapter = None
        if self.bert_model.config.hidden_size != self.dim:
            self.adapter = torch.nn.Linear(self.bert_model.config.hidden_size, self.dim, bias=False)

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
        encs = self.bert_model(x, attention_mask=encmask)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def test_forward(self, tokens: torch.Tensor = None, enc=None, encmask=None, cache=None):
        logits, cache = self(tokens=tokens[:, None], enc=enc, encmask=encmask, cache=cache)
        return logits[:, 0], cache

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
        """
        :param tokens:      (batsize, ) int ids
        :param enc:         (batsize, seqlen, dim)
        :param encmask:     (batsize, seqlen)
        :param cache:       list of rnn states for every rnn layer
        :return:
        """
        embs = self.emb(tokens)
        padmask = tokens != 0

        _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                            encoder_hidden_states=enc, encoder_attention_mask=encmask,
                            use_cache=True, past_key_value_states=cache)

        c = _ret[0]
        cache = _ret[1]

        c = self.preout(c)
        c = torch.nn.functional.leaky_relu(c, 0.1)

        logits = self.out(c)
        return logits, cache


class SeqDecoder(torch.nn.Module):
    def __init__(self, cell:GRUDecoderCell,
                 vocab=None,
                 max_steps:int=100,
                 usejoint=False,
                 mode="baseline",
                 **kw):
        super(SeqDecoder, self).__init__(**kw)
        self.cell = cell
        self.vocab = vocab
        self.max_steps = max_steps
        self.kldiv = torch.nn.KLDivLoss(reduction="none")
        self.logsm = torch.nn.LogSoftmax(-1)
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

        self.usejoint = usejoint

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

    def extract_training_example(self, x: torch.Tensor, y: torch.Tensor):
        # y: (batsize, seqlen) ids, padded with zeros
        ymask = (y != 0).float()
        ytotallens = ymask.sum(1).long()

        newy = torch.zeros(y.size(0), y.size(1) + 2, device=y.device).long()
        newy[:, 0] = self.vocab["@BOS@"]        # make them start with BOS
        newy[:, 1:-1] = y                       # copy sequence
        for i in range(len(ytotallens)):
            newy[i, ytotallens[i]+1] = self.vocab["@EOS@"]
        # torch.scatter(newy, 1, ytotallens[:, None], self.vocab["@EOS@"])     # end with EOS

        y_inp = newy[:, :-1]
        tgt = newy[:, 1:]
        tgtmask = tgt != 0

        return x, y_inp, tgt, tgtmask

    def compute_loss(self, logits, tgt, mask=None):
        """
        :param logits:      (batsize, seqlen, vocsize)
        :param tgt:         (batsize, seqlen)
        :param mask:        (batsize, seqlen)
        :return:
        """
        kl = self.ce(logits.transpose(2, 1), tgt)      # (batsize, seqlen)
        if mask is not None:
            kl = kl * mask
        kl = kl.sum(-1)     # (batsize,)


        best_pred = logits.max(-1)[1]   # (batsize, seqlen)
        best_gold = tgt
        same = best_pred == best_gold
        if mask is not None:
            same = same | ~(mask.bool())
        acc = same.all(-1)  # (batsize,)

        return kl, acc.float()

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.cell.encode_source(x)
        # run through tagger: the same for all versions
        cache = None
        logits, cache = self.cell(newy, enc=enc, encmask=encmask, cache=cache)
        # compute loss: different versions do different masking and different targets
        loss, acc = self.compute_loss(logits, tgt, mask=tgtmask)
        return {"loss": loss, "acc": acc}, logits

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        y[:] = self.vocab["@BOS@"]

        # run encoder
        enc, encmask = self.cell.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y).bool()
        cache = None
        # paststates = []
        # logitses = []
        outputses = []
        while step < self.max_steps and not torch.all(ended): #(newy is None or not (y.size() == newy.size() and torch.all(y == newy))):
            y = newy if newy is not None else y
            # run tagger
            logits, cache = self.cell.test_forward(tokens=y, enc=enc, encmask=encmask, cache=cache)
            # states, summaries = cache
            # paststates.append(states)
            # logitses.append(logits[:, None])
            bestpred = logits.max(1)[1]
            newy = bestpred
            outputses.append(bestpred[:, None])

            _ended = bestpred == self.vocab["@EOS@"]

            step += 1
            ended = ended | _ended
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        outputses = torch.cat(outputses, 1)
        return outputses, steps_used.float()

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        preds, stepsused = self.get_prediction(x)

        def tensor_to_trees(x, vocab:Vocab):
            xstrs = [vocab.tostr(x[i]).replace("@BOS@", "").replace("@NT@", "") for i in range(len(x))]
            xstrs = [re.sub("::\d+", "", xstr) for xstr in xstrs]
            trees = []
            for xstr in xstrs:
                # drop everything after @END@, if present
                xstr = xstr.split("@EOS@")
                xstr = xstr[0]
                # add an opening parentheses if not there
                xstr = xstr.strip()
                if len(xstr) == 0 or xstr[0] != "(":
                    xstr = "(" + xstr
                # balance closing parentheses
                parenthese_imbalance = xstr.count("(") - xstr.count(")")
                xstr = xstr + ")" * max(0, parenthese_imbalance)        # append missing closing parentheses
                xstr = "(" * -min(0, parenthese_imbalance) + xstr       # prepend missing opening parentheses
                try:
                    tree = lisp_to_tree(xstr)
                    if isinstance(tree, tuple) and len(tree) == 2 and tree[0] is None:
                        tree = None
                except Exception as e:
                    tree = None
                trees.append(tree)
            return trees

        # compute loss and metrics
        gold_trees = tensor_to_trees(gold, vocab=self.vocab)
        pred_trees = tensor_to_trees(preds, vocab=self.vocab)
        treeaccs = [float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device), "stepsused": stepsused}
        return ret, pred_trees


class TreeDecoderCell(GRUDecoderCell):
    def __init__(self, dim, indim=None, vocab:Vocab=None, numlayers:int=2, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased",
                 vocab_factorized=False, mode="baseline", **kw):
        super(TreeDecoderCell, self).__init__(dim, indim=indim, vocab=vocab, numlayers=numlayers, dropout=dropout, bertname=bertname,
                                              vocab_factorized=vocab_factorized, mode=mode, **kw)
        dims = self.get_dims(self.indim, self.dim, numlayers)
        self.decoder = torch.nn.ModuleList(
            [torch.nn.GRUCell(dims[i], dims[i+1]) for i in range(numlayers)])
        self.subtree_start = self.vocab["("]
        self.subtree_end = self.vocab[")"]

    def get_dims(self, indim, dim, numlayers):
        dims = [indim + dim]
        for _ in range(numlayers):
            dims.append(dim)
        return dims

    def get_all_states(self, tokens_tm1, tokens_tm2=None, paststates=None, levels=None):
        # mix states from paststates according to levels and current token
        device = tokens_tm1.device
        parent_states = [[torch.zeros(self.dim, device=device) for __ in range(tokens_tm1.size(0))] for _ in range(len(self.decoder))]
        prev_states = [[torch.zeros(self.dim, device=device) for __ in range(tokens_tm1.size(0))] for _ in range(len(self.decoder))]
        reduce_states = [[torch.zeros(self.dim, device=device) for __ in range(tokens_tm1.size(0))] for _ in range(len(self.decoder))]

        movement = [0 for _ in range(tokens_tm1.size(0))]

        newlevels = [[] for _ in range(len(levels))]

        for i, levels_i in enumerate(levels):
            # determine current depth
            prevdepth = levels_i[-1] if len(levels_i) > 0 else 0
            if tokens_tm1[i] == self.subtree_end:
                curdepth = prevdepth - 1
                movement[i] = -1
            # elif tokens_tm2 is not None and tokens_tm2[i] == self.subtree_start:
            #     curdepth = prevdepth + 1
            elif tokens_tm1[i] == self.subtree_start:
                curdepth = prevdepth + 1
                movement[i] = +1
            else:
                curdepth = prevdepth
            curdepth = max(0, curdepth)

            # go backwards in time and find which state was parent, prev and child using levels
            prev_state_found = False
            reduce_state_found = False
            parent_state_found = False
            for j in range(1, len(levels_i) + 1):
                depth = levels_i[-j]
                if depth < curdepth:  # timestep -j-1 is parent
                    # assert (depth == curdepth - 1)
                    for k in range(len(parent_states)):
                        parent_states[k][i] = paststates[-j][k][i]
                    # copy last seen sibling state
                    parent_state_found = True
                elif depth > curdepth:  # timestep -j-1 is child of previous sibling
                    if not reduce_state_found:
                        for k in range(len(reduce_states)):
                            reduce_states[k][i] = paststates[-j][k][i]
                        reduce_state_found = True
                elif depth == curdepth:  # prevstate
                    if not prev_state_found:
                        for k in range(len(prev_states)):
                            prev_states[k][i] = paststates[-j][k][i]
                        prev_state_found = True

                if parent_state_found or (curdepth == 0 and prev_state_found):
                    # stop looking if parent state found or current depth is 0 and previous state is found
                    break

            newlevels[i] = levels[i] + [curdepth]
        parent_states = [torch.stack([ps_i for ps_i in ps], 0) for ps in parent_states]
        prev_states = [torch.stack([ps_i for ps_i in ps], 0) for ps in prev_states]
        reduce_states = [torch.stack([ps_i for ps_i in ps], 0) for ps in reduce_states]
        movement = torch.tensor(movement, dtype=torch.long, device=device)
        return parent_states, prev_states, reduce_states, newlevels, movement

    def merge_states(self, parent_states, prev_states, reduce_states, movement):
        """ every of the input states is a list of number of layers of (batsize, dim) """
        ret_states = [parent_state + prev_state + reduce_state
                      for parent_state, prev_state, reduce_state
                      in zip(parent_states, prev_states, reduce_states)]
        return ret_states, None

    def get_rnn_inputs(self, embs, prev_summ, parent_states, reduce_states, movement):
        ret = torch.cat([embs, prev_summ], 1)
        return ret

    def test_forward(self, tokens:torch.Tensor=None, tokens_tm2=None, enc=None, encmask=None,
                paststates=None, prev_summ=None, levels=None):
        return self(tokens=tokens, tokens_tm2=tokens_tm2, enc=enc, encmask=encmask,
                    paststates=paststates, prev_summ=prev_summ, levels=levels)

    def forward(self, tokens:torch.Tensor=None, tokens_tm2=None, enc=None, encmask=None,
                paststates=None, prev_summ=None, levels=None):
        """
        :param tokens:      (batsize, ) int ids
        :param enc:         (batsize, seqlen, dim)
        :param encmask:     (batsize, seqlen)
        :param paststates:  list per time step of list per layer of previous states
        :param prev_summ: (batsize, dim) previous attention summary
        :param levels:      (batsize, prevseqlen) specifies depth of previous states
        :return:
        """
        embs = self.emb(tokens)
        if prev_summ is None:
            prev_summ = torch.zeros(tokens.size(0), self.dim, device=tokens.device)

        parent_states, prev_states, reduce_states, newlevels, movement = self.get_all_states(tokens, tokens_tm2, paststates, levels)

        rnn_states = self.merge_states(parent_states, prev_states, reduce_states, movement)
        inps = self.get_rnn_inputs(embs, prev_summ, parent_states, reduce_states, movement)
        lower_rep = inps
        # pass through rnns
        paststates.append([])
        for rnn_layer, rnn_state in zip(self.decoder, rnn_states):
            lower_rep = self.dropout(lower_rep)
            new_rnn_out = rnn_layer(lower_rep, rnn_state)
            paststates[-1].append(new_rnn_out)
            lower_rep = new_rnn_out

        lower_rep = self.dropout(lower_rep)

        # use attention
        # (batsize, seqlen, dim) and (batsize, dim)
        scores = torch.einsum("bsd,bd->bs", enc, lower_rep)
        scores = scores + (encmask.float() - 1) * 99999
        weights = torch.softmax(scores, 1)

        summaries = torch.einsum("bs,bsd->bd", weights, enc)
        c = torch.cat([lower_rep, summaries], 1)
        c = self.preout(c)
        c = torch.nn.functional.leaky_relu(c, 0.1)

        logits = self.out(c)
        return logits, (paststates, summaries, newlevels)


class SeqTreeDecoderCell(TreeDecoderCell):
    """ Tree decoder cell that actually is a normal sequence decoder"""
    def merge_states(self, parent_states, prev_states, reduce_states, movement):
        # if movement equals -1 then take reduce state because we just popped one
        # if movement equals +1 then take parent state because we just pushed one
        movement = movement[:, None]
        states = [(movement == -1) * reduce_state + (movement == +1) * parent_state + (movement == 0) * prev_state
                  for reduce_state, parent_state, prev_state
                  in zip(reduce_states, parent_states, prev_states)]
        return states


class SimpleTreeDecoderCell(TreeDecoderCell):    # TODO: verify this
    """ Simple tree decoder cell.
        Completely discards the reduce states (from previous siblings's children),
        and concatenates parent state to input.
    """
    def get_dims(self, indim, dim, numlayers):
        dims = [indim + dim + dim]
        for _ in range(numlayers):
            dims.append(dim)
        return dims

    def merge_states(self, parent_states, prev_states, reduce_states, movement):
        movement = movement[:, None]
        states = [(movement != +1) * prev_state + (movement == +1) * parent_state
                  for prev_state, parent_state in zip(prev_states, parent_states)]
        return states

    def get_rnn_inputs(self, embs, prev_summ, parent_states, reduce_states, movement):
        ret = torch.cat([embs, prev_summ, parent_states[-1]], 1)
        return ret


class ReduceSummTreeDecoderCell(TreeDecoderCell):    # TODO: verify this
    """ Stack summary tree decoder cell.
        Takes into account reduce states as inputs at timesteps where movement is downwards,
        and concatenates parent state to input.
    """

    def get_dims(self, indim, dim, numlayers):
        dims = [indim + dim]
        for _ in range(numlayers):
            dims.append(dim)
        return dims

    def merge_states(self, parent_states, prev_states, reduce_states, movement):
        movement = movement[:, None]
        states = [(movement != +1) * prev_state + (movement == +1) * parent_state
                  for prev_state, parent_state in zip(prev_states, parent_states)]
        return states

    def get_rnn_inputs(self, embs, prev_summ, parent_states, reduce_states, movement):
        movement = movement.float()[:, None]
        extra_embs = (movement == -1) * reduce_states[-1]
        embs = embs + extra_embs
        ret = torch.cat([embs, prev_summ], 1)
        return ret


class TreeDecoder(SeqDecoder):
    def __init__(self, cell:TreeDecoderCell,
                 vocab=None,
                 max_steps:int=100,
                 usejoint=False,
                 mode="baseline",
                 **kw):
        super(TreeDecoder, self).__init__(cell, vocab=vocab, max_steps=max_steps, usejoint=usejoint, mode=mode, **kw)

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.cell.encode_source(x)
        # run through tagger: the same for all versions
        paststates = []
        prev_summ = None
        levels = [[] for _ in range(newy.size(0))]
        logitses = []
        for i in range(newy.size(1)):       # teacher forced
            newy_tm1 = newy[:, i]
            newy_tm2 = newy[:, i-1] if i > 0 else None
            logits, (state, prev_summ, levels) = \
                self.cell(newy_tm1, tokens_tm2=newy_tm2, enc=enc, encmask=encmask,
                                    paststates=paststates,
                                    prev_summ=prev_summ,
                                    levels=levels)
            logitses.append(logits[:, None])
        # compute loss: different versions do different masking and different targets
        logitses = torch.cat(logitses, 1)
        loss, acc = self.compute_loss(logitses, tgt, mask=tgtmask)
        return {"loss": loss, "acc": acc}, logits

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        y[:] = self.vocab["@BOS@"]

        # run encoder
        enc, encmask = self.cell.encode_source(x)

        step = 0
        ended = torch.zeros_like(y).bool()
        logitses = []
        outputses = []
        paststates = []
        prev_summ = None
        levels = [[] for _ in range(x.size(0))]
        y_tm2 = None
        while step < self.max_steps and not torch.all(ended): #(newy is None or not (y.size() == newy.size() and torch.all(y == newy))):
            # run tagger
            logits, (paststates, prev_summ, levels) = \
                self.cell(tokens=y, tokens_tm2=y_tm2, enc=enc, encmask=encmask,
                          paststates=paststates,
                          prev_summ=prev_summ,
                          levels=levels)
            logitses.append(logits[:, None])
            bestpred = logits.max(1)[1]
            y_tm2 = y
            y = bestpred
            outputses.append(bestpred[:, None])

            _ended = bestpred == self.vocab["@EOS@"]

            step += 1
            ended = ended | _ended
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        outputses = torch.cat(outputses, 1)
        return outputses, steps_used.float()


def compute_oracle_path_selects(y:torch.Tensor):
    """
    :param y:   integer tensor
    :return:    list of lists of select indices of which positions have been selected at this decoding step
    """
    ylen = (y > 0).long().sum().cpu().item()
    yindexes = range(ylen)
    selects = [[]]      # in the first step, nothing was selected

    segments = [yindexes[:]]

    while len(segments) > 0:         # while not completely decoded
        all_segments_empty = True
        # select the middle for each segment
        selects.append([])
        selects[-1] = selects[-2][:]        # copy all previous selects
        next_segments = []
        for segment in segments:
            # take middle, split in two
            if len(segment) % 2 == 1:
                middle_pos = int((len(segment) - 1)/2)
            else:
                middle_pos = (len(segment) - 1)/2
                middle_pos = random.choice([int(math.floor(middle_pos)), int(math.ceil(middle_pos))])
            selects[-1].append(segment[middle_pos])
            left, right = segment[:middle_pos], segment[middle_pos+1:]
            if len(left) > 0:
                next_segments.append(left)
            if len(right) > 0:
                next_segments.append(right)
        segments = next_segments
    return selects


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def retsup_cmp(a, b):
    """
    :param a, b:
    :return:
    """
    ret = b[1] - a[1]
    if ret == 0:
        al = a[0]
        bl = b[0]
        if al == bl:
            ret = 0
        elif al < bl:
            ret = -1
        else:
            ret = 1
    return ret


def run(domain="restaurants",
        mode="baseline",         # "baseline", "ltr", "uniform", "binary"
        probthreshold=0.,       # 0. --> parallel, >1. --> serial, 0.< . <= 1. --> semi-parallel
        lr=0.0001,
        enclrmul=0.1,
        batsize=50,
        epochs=1000,
        hdim=366,
        numlayers=2,
        numheads=6,
        dropout=0.1,
        noreorder=False,
        trainonvalid=False,
        seed=87646464,
        gpu=-1,
        patience=-1,
        gradacc=1,
        cosinelr=False,
        warmup=0,
        gradnorm=3,
        validinter=10,
        maxsteps=100,
        testcode=False,
        evaltrain=False,
        cooldown=0,
        usejoint=False,
        ):
    settings = locals().copy()
    settings["version"] = "vcr"
    q.pp_dict(settings)

    wandb.init(project=f"baseline_overnight", config=settings, reinit=True)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading")
    tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds(domain, trainonvalid=trainonvalid, noreorder=noreorder, numbered=False)
    tt.tock("loaded")

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model

    if mode == "gru":
        tagger = GRUDecoderCell(hdim, vocab=flenc.vocab, numlayers=numlayers, dropout=dropout, mode=mode)
        decoder = SeqDecoder(tagger, flenc.vocab, max_steps=maxsteps, mode=mode)
    elif mode == "tm":
        tagger = TMDecoderCell(hdim, vocab=flenc.vocab, numlayers=numlayers, dropout=dropout, mode=mode, numheads=numheads)
        decoder = SeqDecoder(tagger, flenc.vocab, max_steps=maxsteps, mode=mode)
    elif "tree" in mode:
        if mode == "simpletree":
            tagger = SimpleTreeDecoderCell(hdim, vocab=flenc.vocab, numlayers=numlayers, dropout=dropout, mode=mode)
        elif mode == "seqtree":
            tagger = SeqTreeDecoderCell(hdim, vocab=flenc.vocab, numlayers=numlayers, dropout=dropout, mode=mode)
        elif mode == "reducesummtree":
            tagger = ReduceSummTreeDecoderCell(hdim, vocab=flenc.vocab, numlayers=numlayers, dropout=dropout, mode=mode)
        decoder = TreeDecoder(tagger, flenc.vocab, max_steps=maxsteps, mode=mode)

    print(tagger.decoder)

    # test run
    if testcode:
        batch = next(iter(tdl_seq))
        out = decoder(*batch)
        decoder.train(False)
        batch = next(iter(tdl_seq))
        out = decoder(*batch)

    tloss = make_array_of_metrics("loss", "acc", reduction="mean")
    tmetrics = make_array_of_metrics("treeacc", "stepsused", reduction="mean")
    vmetrics = make_array_of_metrics("treeacc", "stepsused", reduction="mean")
    xmetrics = make_array_of_metrics("treeacc", "stepsused", reduction="mean")


    # region parameters
    def get_parameters(m, _lr, _enclrmul):
        bertparams = []
        otherparams = []
        for k, v in m.named_parameters():
            if "bert_model." in k:
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
    eyt = q.EarlyStopper(vmetrics[0], patience=int(round(patience/validinter)), min_epochs=int(round(30/validinter)), more_is_better=True, remember_f=lambda: deepcopy(tagger))

    def wandb_logger():
        d = {}
        for name, loss in zip(["CE"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc", "stepsused"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc", "stepsused"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim = get_optim(tagger, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        assert t_max > (warmup + cooldown + 10)
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(low=0., high=1.0, steps=t_max-warmup) >> (0. * lr)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, gradient_accumulation_steps=gradacc,
                                        on_before_optim_step=[lambda : clipgradnorm(_m=tagger, _norm=gradnorm)])

    trainepoch = partial(q.train_epoch, model=decoder,
                                        dataloader=tdl_seq,
                                        optim=optim,
                                        losses=tloss,
                                        device=device,
                                        _train_batch=trainbatch,
                                        on_end=[lambda: lr_schedule.step()])

    trainevalepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=tmetrics,
                         dataloader=tdl_seq,
                         device=device)

    on_end_v = [lambda: eyt.on_epoch_end(), lambda: wandb_logger()]

    validepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=vmetrics,
                         dataloader=vdl_seq,
                         device=device,
                         on_end=on_end_v)

    tt.tick("training")
    if evaltrain:
        valid_epoch_fs = [trainevalepoch, validepoch]
    else:
        valid_epoch_fs = [validepoch]
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=valid_epoch_fs,
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    if eyt.remembered is not None and not trainonvalid:
        tt.msg("reloading best")
        decoder.cell = eyt.remembered
        cell = eyt.remembered

        tt.tick("rerunning validation")
        validres = validepoch()
        tt.tock(f"Validation results: {validres}")

    tt.tick("running train")
    trainres = trainevalepoch()
    print(f"Train tree acc: {trainres}")
    tt.tock()

    tt.tick("running test")
    testepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=xmetrics,
                         dataloader=xdl_seq,
                         device=device)
    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock()

    settings.update({"final_train_CE": tloss[0].get_epoch_error()})
    settings.update({"final_train_tree_acc": tmetrics[0].get_epoch_error()})
    settings.update({"final_valid_tree_acc": vmetrics[0].get_epoch_error()})
    settings.update({"final_test_tree_acc": xmetrics[0].get_epoch_error()})
    settings.update({"final_train_steps_used": tmetrics[1].get_epoch_error()})
    settings.update({"final_valid_steps_used": vmetrics[1].get_epoch_error()})
    settings.update({"final_test_steps_used": xmetrics[1].get_epoch_error()})

    wandb.config.update(settings)
    q.pp_dict(settings)


def run_experiment(domain="default",    #
                   domains="default",     # first or second
                   mode="tm",         # "baseline", "ltr", "uniform", "binary"
        lr=-1.,
        enclrmul=-1.,
        batsize=-1,
        epochs=-1,
        hdim=-1,
        numlayers=-1,
        numheads=-1,
        dropout=-1.,
        noreorder=False,
        trainonvalid=False,
        seed=-1,
        gpu=-1,
        patience=-1,
        gradacc=-1,
        cosinelr=False,
        warmup=-1,
        gradnorm=3.,
        validinter=-1,
        maxsteps=90,
        evaltrain=False,
        usejoint=False,
                   testcode=False,
        ):

    settings = locals().copy()
    del settings["domains"]

    ranges = {
        "domain": ["socialnetwork", "blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"],
        "dropout": [0.0, 0.1, 0.25, 0.4],
        "epochs": [121],
        "batsize": [50],
        "hdim": [768, ], # 366],
        "numlayers": [2],
        "lr": [0.001, ], #0.0001, 0.000025],
        "enclrmul": [0.1],                  # use 1.
        "seed": [87646464, 42, 456852],
        "patience": [-1],
        "warmup": [20],
        "validinter": [1],
        "gradacc": [1],
        "numheads": [12]
    }
    if domains == "first":
        ranges["domain"] = ["calendar", "recipes", "publications", "restaurants"]
    elif domains == "second":
        ranges["domain"] = ["blocks", "housing", "basketball", "socialnetwork"]
    elif domains == "all" or domains == "default":
        ranges["domain"] = ["socialnetwork", "blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"]

    # if mode == "baseline":        # baseline
    ranges["validinter"] = [1]
    if mode == "tm":
        ranges["numlayers"] = [6]

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

    print(__file__)
    p = __file__ + f".baseline.{domain}"
    q.run_experiments_random(
        run, ranges, path_prefix=None, check_config=None, **settings)



if __name__ == '__main__':
    q.argprun(run_experiment)
    # DONE: fix orderless for no simplification setting used here
    # DONE: make baseline decoder use cached decoder states
    # DONE: in unsimplified Overnight, the filters are nested but are interchangeable! --> use simplify filters ?!
    # python overnight_seqinsert.py -gpu 0 -domain ? -lr 0.0001 -enclrmul 1. -hdim 768 -dropout 0.3 -numlayers 6 -numheads 12

    # -batsize 30 -gpu 0 -lr 0.0005 -evaltrain -epochs 301 -dropout 0.2 -domain publications -numlayers 2 -hdim 768 -enclrmul 0.1 -patience 15 -mode tree