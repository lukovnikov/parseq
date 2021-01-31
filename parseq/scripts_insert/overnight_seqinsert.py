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
from parseq.grammar import tree_to_lisp_tokens, are_equal_trees, lisp_to_tree
from parseq.scripts_insert.overnight_treeinsert import extract_info
from parseq.scripts_insert.util import reorder_tree, flatten_tree
from parseq.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


ORDERLESS = {"op:and", "SW:concat", "filter", "call-SW:concat"}


def tree_to_seq(x:Tree):
    xstr = tree_to_lisp_tokens(x)
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
    else:
        ds = OvernightDatasetLoader(simplify_mode="none").load(domain=domain, trainonvalid=trainonvalid)
        # ds contains 3-tuples of (input, output tree, split name)

        if not noreorder:
            ds = ds.map(lambda x: (x[0], reorder_tree(x[1], orderless=orderless), x[2]))
        ds = ds.map(lambda x: (x[0], tree_to_seq(x[1]), x[2]))
        ds = ds.map(lambda x: (x[0], x[1][1:-1], x[2]))

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



class TransformerTagger(SeqInsertionTagger):
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", baseline=False, vocab_factorized=False, **kw):
        super(TransformerTagger, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.baseline = baseline
        config = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                   num_layers=numlayers, num_heads=numheads, dropout_rate=dropout,
                                   use_relative_position=False)

        self.emb = TokenEmb(vocab, config.d_model, factorized=vocab_factorized, pooler="sum")
        self.posemb = torch.nn.Embedding(maxpos, config.d_model)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = baseline
        self.decoder = TransformerStack(decoder_config)

        if baseline:
            self.out = torch.nn.Linear(self.dim, self.vocabsize)
        else:
            # self.out = torch.nn.Linear(self.dim * 2, self.vocabsize)
            self.out = TokenOut(self.dim * 2, self.vocab, factorized=vocab_factorized, pooler="sum")
        # self.out = MOS(self.dim, self.vocabsize, K=mosk)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname,
                                                    hidden_dropout_prob=min(dropout, 0.2),
                                                    attention_probs_dropout_prob=min(dropout, 0.1))
        # def set_dropout(m:torch.nn.Module):
        #     if isinstance(m, torch.nn.Dropout):
        #         m.p = dropout
        # self.bert_model.apply(set_dropout)

        self.adapter = None
        if self.bert_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.bert_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
        encs = self.bert_model(x)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
        padmask = (tokens != 0)
        # if not self.baseline:
        #     padmask = padmask[:, 1:]
        embs = self.emb(tokens)
        posembs = self.posemb(torch.arange(tokens.size(1), device=tokens.device))[None]
        embs = embs + posembs
        use_cache = False
        if self.baseline:
            use_cache = True
        if cache is not None:
            embs = embs[:, -1:, :]
        _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=use_cache,
                            past_key_value_states=cache)
        ret = _ret[0]
        cache = None
        if self.baseline:
            c = ret
            cache = _ret[1]
        else:
            c = torch.cat([ret[:, 1:], ret[:, :-1]], -1)
        logits = self.out(c)
        # logits = logits + torch.log(self.vocab_mask[None, None, :])
        if self.baseline:
            return logits, cache
        else:
            return logits
        # probs = self.out(ret[0], self.vocab_mask[None, None, :])
        # return probs


class SeqInsertionDecoder(torch.nn.Module):
    # default_termination_mode = "slot"
    # default_decode_mode = "parallel"

    def __init__(self, tagger:SeqInsertionTagger,
                 vocab=None,
                 prob_threshold=0.,
                 max_steps:int=20,
                 max_size:int=100,
                 end_offset=0.,
                 oraclemix=0.,
                 **kw):
        super(SeqInsertionDecoder, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_steps = max_steps
        self.max_size = max_size
        self.kldiv = torch.nn.KLDivLoss(reduction="none")
        self.logsm = torch.nn.LogSoftmax(-1)
        self.prob_threshold = prob_threshold
        self.end_offset = end_offset

        self.oraclemix = oraclemix

        # self.termination_mode = self.default_termination_mode if termination_mode == "default" else termination_mode
        # self.decode_mode = self.default_decode_mode if decode_mode == "default" else decode_mode

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

    @abstractmethod
    def extract_training_example(self, x, y):
        pass

    def compute_loss(self, logits, tgt, mask=None):
        """
        :param logits:      (batsize, seqlen, vocsize)
        :param tgt:         (batsize, seqlen, vocsize)
        :param mask:        (batsize, seqlen)
        :return:
        """
        logprobs = self.logsm(logits)
        kl = self.kldiv(logprobs, tgt)      # (batsize, seqlen, vocsize)
        kl = kl.sum(-1)                     # (batsize, seqlen)
        if mask is not None:
            kl = kl * mask
        kl = kl.sum(-1)


        best_pred = logits.max(-1)[1]   # (batsize, seqlen)
        best_gold = tgt.max(-1)[1]
        same = best_pred == best_gold
        if mask is not None:
            same = same | ~(mask.bool())
        acc = same.all(-1)  # (batsize,)

        # get probability of best predictions
        tgt_probs = torch.gather(tgt, -1, best_pred.unsqueeze(-1))  # (batsize, seqlen, 1)
        recall = (tgt_probs > 0).squeeze(-1)
        if mask is not None:
            recall = recall | ~(mask.bool())
        recall = recall.all(-1)
        return kl, acc.float(), recall.float()
        # return kl

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits = self.tagger(tokens=newy, enc=enc, encmask=encmask)
        # compute loss: different versions do different masking and different targets
        loss, acc, recall = self.compute_loss(logits, tgt[:, :-1], mask=tgtmask[:, :-1])
        return {"loss": loss, "acc": acc, "recall": recall}, logits

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.zeros(x.size(0), 2, device=x.device, dtype=torch.long)
        y[:, 0] = self.vocab["@BOS@"]
        y[:, 1] = self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.tagger.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        while step < self.max_steps and not torch.all(ended): #(newy is None or not (y.size() == newy.size() and torch.all(y == newy))):
            y = newy if newy is not None else y
            # run tagger
            logits = self.tagger(tokens=y, enc=enc, encmask=encmask)
            # logprobs = torch.log_softmax(logits, -1)
            # logprobs[:, :, self.vocab["@END@"]] = logprobs[:, :, self.vocab["@END@"]] - self.end_offset
            # probs = torch.exp(logprobs)
            probs = torch.softmax(logits, -1)
            probs[:, :, self.vocab["@END@"]] = probs[:, :, self.vocab["@END@"]] - self.end_offset
            predprobs, preds = probs.max(-1)
            predprobs, preds = predprobs.cpu().detach().numpy(), preds.cpu().detach().numpy()
            _y = y.cpu().detach().numpy()
            newy = torch.zeros(y.size(0), min(self.max_size, y.size(1) * 2), device=y.device, dtype=torch.long)
            _ended = torch.zeros_like(y[:, 0]).bool()
            # update sequences
            for i in range(len(y)):
                k = 0
                p_i = preds[i]
                pp_mask = p_i != self.vocab["@END@"]
                pp_i = predprobs[i] * pp_mask
                pp_mask = _y[i] == self.vocab["@EOS@"]
                pp_mask = np.cumsum(pp_mask, -1)
                pp_i = pp_i * (pp_mask[:-1] == 0)
                # pp_i = (pp_i > 0) * np.random.rand(*pp_i.shape)
                prob_thresh = min(self.prob_threshold, max(pp_i))
                terminated = True
                for j in range(len(y[i])):          # loop, advance j = j+1
                    if k >= newy.size(1):
                        break
                    newy[i, k] = y[i, j]            # copy from previous target sequence
                    k += 1                          # advance newy pointer to next position
                    y_ij = _y[i, j]
                    if y_ij == self.vocab["@EOS@"]: # if token was EOS, terminate generation
                        break  # stop
                    if j >= len(p_i):               # if we reached beyond the length of predictions, terminate
                        break
                    p_ij = p_i[j]                   # token predicted between j-th and j+1-st position in previous sequence
                    pp_ij = pp_i[j]                 # probability assigned to that token
                    if pp_ij >= prob_thresh:        # skip if probability was lower than threshold
                        if p_ij == self.vocab["@END@"]:         # if predicted token is @END@, do nothing
                            pass  # don't insert anything
                        else:  # insert what was predicted
                            if k >= newy.size(1):
                                break
                            newy[i, k] = preds[i, j]
                            k += 1                  # advance newy pointer to next position
                            terminated = False      # sequence changed so don't terminate
                _ended[i] = terminated

            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)  # prevent terminated examples from changing

            maxlen = (newy != 0).long().sum(-1).max()
            newy = newy[:, :maxlen]
            step += 1
            ended = ended | _ended
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
            lens = (newy != 0).long().sum(-1)
            maxlenreached = (lens == self.max_size)
            if torch.all(maxlenreached):
                break
        return newy, steps_used.float()

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        preds, stepsused = self.get_prediction(x)

        def tensor_to_trees(x, vocab:Vocab):
            xstrs = [vocab.tostr(x[i]).replace("@BOS@", "").replace("@EOS@", "") for i in range(len(x))]
            xstrs = [re.sub("::\d+", "", xstr) for xstr in xstrs]
            trees = []
            for xstr in xstrs:
                # drop everything after @END@, if present
                xstr = xstr.split("@END@")
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


class SeqInsertionDecoderUniform(SeqInsertionDecoder):
    # decode modes: "parallel", "serial" or "semiparallel":
    #    - parallel: execute actions at all slots simultaneously
    #           --> prob threshold = 0.
    #    - serial: execute action with highest probability across all slots, unless the action is an @END@
    #           --> prob threshold > 1.
    #    - semiparallel: execute actions for all slots where the highest probability is above a certain threshold,
    #                    unless the action is an @END@.
    #                    if there are no slots with highest probability higher than threshold, fall back to serial mode for this decoding step
    #           --> prob threshold between 0. and 1.
    # --> all modes naturally terminate once all slots predict an @END@
    # default_termination_mode = "slot"
    # default_decode_mode = "parallel"

    def get_slot_value_probs(self, slotvalues):     # uniform
        # uniformly distribute probability over unique tokens
        # then distribute uniformly over every position a token occurs in
        # example: A B B C D --> [0.25, 0.125, 0.125, 0.25, 0.25]
        # this way, when tgt is accumulated in training code, the distribution over tokens will be uniform
        prob = 1./len(set(slotvalues))
        token_freqs = {}
        for slotvalue in slotvalues:
            if slotvalue not in token_freqs:
                token_freqs[slotvalue] = 0
            token_freqs[slotvalue] += 1
        probs = [prob/token_freqs[slotvalue] for slotvalue in slotvalues]
        return probs

    def extract_training_example(self, x: torch.Tensor, y: torch.Tensor):
        # y: (batsize, seqlen) ids, padded with zeros
        ymask = (y != 0).float()
        ytotallens = ymask.sum(1)
        ylens = torch.rand(ytotallens.size(), device=ytotallens.device)
        ylens = (ylens * ytotallens).round().long()
        _ylens = ylens.cpu().numpy()
        # ylens contains the sampled lengths

        # for LTR: take 'ylens' leftmost tokens
        # for Uniform/Binary: randomly select 'ylens' tokens
        newy = torch.zeros(y.size(0), y.size(1) + 2, device=y.device).long()
        newy[:, 0] = self.vocab["@BOS@"]
        tgt = torch.zeros(y.size(0), y.size(1) + 2, self.vocab.number_of_ids(), device=y.device)
        # 'tgt' contains target distributions
        for i in range(newy.size(0)):
            if random.random() < self.oraclemix:
                # assert self.numbered is True
                selects = compute_oracle_path_selects(y[i])
                select = random.choice(selects)
                select = sorted(select)
            else:
                perm = torch.randperm(ytotallens[i].long().cpu().item())
                perm = perm[:ylens[i].long().cpu().item()]
                select, _ = perm.sort(-1)
                select = list(select.cpu().numpy())
            k = 1  # k is where in the new sampled sequence we're at

            slotvalues_acc = []
            slotvalues = []
            for j in range(int(ytotallens[i].cpu().item())):
                y_ij = y[i, j].cpu().item()
                if k <= len(select) and j == select[k - 1]:  # if j-th token in y should be k-th in newy
                    newy[i, k] = y[i, j]
                    slotvalues_acc.append(slotvalues)
                    slotvalues = []
                    k += 1
                else:  # otherwise, add
                    slotvalues.append(y_ij)
                    # tgt[i, k - 1, y_ij] = 1
            slotvalues_acc.append(slotvalues)
            newy[i, k] = self.vocab["@EOS@"]

            for j, slotvalues in enumerate(slotvalues_acc):
                if len(slotvalues) == 0:
                    tgt[i, j, self.vocab["@END@"]] = 1
                else:
                    for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
                        tgt[i, j, slotvalue] += float(valueprob)

        # normalize
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-6)
        tgtmask = (tgt.sum(-1) != 0).float()
        # make uniform for masked positions
        newymask = (newy != 0).float()
        uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
        tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
        # cut unnecessary padded elements from the right of newy
        newlen = newymask.sum(-1).max()
        newy = newy[:, :int(newlen)]
        tgt = tgt[:, :int(newlen)]
        tgtmask = tgtmask[:, :int(newlen)]

        return x, newy, tgt, tgtmask


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


class SeqInsertionDecoderBinary(SeqInsertionDecoderUniform):
    """ Differs from Uniform only in computing and using non-uniform weights for gold output distributions """
    def __init__(self, tagger:SeqInsertionTagger,
                 vocab=None,
                 prob_threshold=0.,
                 max_steps:int=20,
                 max_size:int=100,
                 oraclemix=0.,
                 tau=1.,
                 **kw):
        super(SeqInsertionDecoderBinary, self).__init__(tagger, vocab=vocab,
                                                        max_steps=max_steps,
                                                        max_size=max_size,
                                                        prob_threshold=prob_threshold,
                                                        oraclemix=oraclemix,
                                                        **kw)
        self.tau = tau

    def get_slot_value_probs(self, slotvalues):
        # assign higher probability to tokens closer to centre
        # when multiple tokens of the same type are present: keep score of the closest one to center
        # set distance for all other ones of the same token to infinity
        center = len(slotvalues) / 2 - 0.5
        distances = [abs(x - center) for x in range(len(slotvalues))]
        mindist_per_token = {}
        for slotvalue, distance in zip(slotvalues, distances):
            if slotvalue not in mindist_per_token:
                mindist_per_token[slotvalue] = +9999
            mindist_per_token[slotvalue] = min(mindist_per_token[slotvalue], distance)

        token_ranks = {}
        prev_rank = -1
        prev_dist = -1
        sorted_tokendists = sorted(mindist_per_token.items(), key=cmp_to_key(retsup_cmp))[::-1]
        for token, dist in sorted_tokendists:
            if prev_rank > 0 and prev_dist == dist:
                rank = prev_rank
            else:
                rank = prev_rank + 1
            token_ranks[token] = rank
            prev_rank = rank
            prev_dist = dist

        mindistances = [d for d in distances]
        for i, (slotvalue, distance) in enumerate(zip(slotvalues, distances)):
            if distance == mindist_per_token[slotvalue]:
                mindistances[i] = token_ranks[slotvalue]
            else:
                mindistances[i] = +99999

        probs = torch.softmax(-torch.tensor(mindistances)/self.tau, -1)
        probs = probs.numpy()
        return probs


# def get_slotvalues_maxspan(selected, yi):
#     if len(selected) == 0:
#         ret = [list(yi.cpu().detach().numpy())]
#         return ret
#     middle_j = int(len(selected)/2)
#     yilen = (yi > 0).sum()
#     yi = yi[:yilen]
#     middle_k = int(len(yi)/2)
#     left_selected, right_selected = selected[:middle_j], selected[middle_j+1:]
#     # search for the midmost element in selected in yi
#     yi_left, yi_right = None, None
#     foundit = 0
#
#     # compute earliest possible position for the middle element of selected
#     _left_selected = list(left_selected.cpu().detach().numpy())
#     _right_selected = list(right_selected.cpu().detach().numpy())
#     _yi = list(yi.cpu().detach().numpy())
#
#     i = 0
#     for e in _left_selected:
#         while e != _yi[i]:
#             i += 1
#         i += 1
#     earliest_pos = i
#
#     _yi = _yi[::-1]
#     i = 0
#     for e in _right_selected[::-1]:
#         while e != _yi[i]:
#             i += 1
#         i += 1
#     latest_pos = len(_yi) - i
#
#     # compute latest possible position for the middle
#     for l in range(math.ceil(len(yi)/2)+1):
#         foundit = 0
#         if len(yi) == 0 or len(selected) == 0:
#             print("something wrong")
#         if middle_k - l >= earliest_pos and middle_k - l < latest_pos and yi[middle_k - l] == selected[middle_j]:
#             foundit = -1
#         elif middle_k + l >= earliest_pos and middle_k + l < latest_pos and yi[middle_k + l] == selected[middle_j]:
#             foundit = +1
#
#         if foundit != 0:
#             splitpoint = middle_k + l*foundit
#             yi_left, yi_right = yi[:splitpoint], yi[splitpoint + 1:]
#             if len(left_selected) == 0:
#                 left_slotvalueses = [list(yi_left.cpu().detach().numpy())]
#             else:
#                 left_slotvalueses = get_slotvalues_maxspan(left_selected, yi_left)
#
#             if len(right_selected) == 0:
#                 right_slotvalueses = [list(yi_right.cpu().detach().numpy())]
#             else:
#                 right_slotvalueses = get_slotvalues_maxspan(right_selected, yi_right)
#
#             if left_slotvalueses is None or right_slotvalueses is None:
#                 continue
#
#             ret = left_slotvalueses + right_slotvalueses
#             return ret
#     if foundit == 0:
#         return None


# class SeqInsertionDecoderMaxspanBinary(SeqInsertionDecoderBinary):
#     def extract_training_example(self, x: torch.Tensor, y: torch.Tensor):
#         # y: (batsize, seqlen) ids, padded with zeros
#         ymask = (y != 0).float()
#         ytotallens = ymask.sum(1)
#         ylens = torch.rand(ytotallens.size(), device=ytotallens.device)
#         ylens = (ylens * ytotallens).round().long()
#         _ylens = ylens.cpu().numpy()
#         # ylens contains the sampled lengths
#
#         # for LTR: take 'ylens' leftmost tokens
#         # for Uniform/Binary: randomly select 'ylens' tokens
#         newy = torch.zeros(y.size(0), y.size(1) + 2, device=y.device).long()
#         newy[:, 0] = self.vocab["@BOS@"]
#         tgt = torch.zeros(y.size(0), y.size(1) + 2, self.vocab.number_of_ids(), device=y.device)
#         # 'tgt' contains target distributions
#         for i in range(newy.size(0)):
#             perm = torch.randperm(ytotallens[i].long().cpu().item())
#             perm = perm[:ylens[i].long().cpu().item()]
#             select, _ = perm.sort(-1)
#             select = list(select.cpu().numpy())
#             selected = y[i][select]
#
#             slotvalues_acc = get_slotvalues_maxspan(selected, y[i])
#
#             # k = 1  # k is where in the new sampled sequence we're at
#             #
#             # slotvalues_acc = []
#             # slotvalues = []
#             # for j in range(int(ytotallens[i].cpu().item())):
#             #     y_ij = y[i, j].cpu().item()
#             #     if k <= len(select) and j == select[k - 1]:  # if j-th token in y should be k-th in newy
#             #         newy[i, k] = y[i, j]
#             #         slotvalues_acc.append(slotvalues)
#             #         slotvalues = []
#             #         k += 1
#             #     else:  # otherwise, add
#             #         slotvalues.append(y_ij)
#             #         # tgt[i, k - 1, y_ij] = 1
#             # slotvalues_acc.append(slotvalues)
#             k = 1
#             j = len(slotvalues_acc[0])
#             for slotvalues in slotvalues_acc[1:]:
#                 newy[i, k] = y[i, j]
#                 k += 1
#                 j += len(slotvalues) + 1
#             newy[i, k] = self.vocab["@EOS@"]
#
#             for j, slotvalues in enumerate(slotvalues_acc):
#                 if len(slotvalues) == 0:
#                     tgt[i, j, self.vocab["@END@"]] = 1
#                 else:
#                     for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
#                         tgt[i, j, slotvalue] += float(valueprob)
#
#         # normalize
#         tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-6)
#         tgtmask = (tgt.sum(-1) != 0).float()
#         # make uniform for masked positions
#         newymask = (newy != 0).float()
#         uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
#         tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
#         # cut unnecessary padded elements from the right of newy
#         newlen = newymask.sum(-1).max()
#         newy = newy[:, :int(newlen)]
#         tgt = tgt[:, :int(newlen)]
#         tgtmask = tgtmask[:, :int(newlen)]
#
#         return x, newy, tgt, tgtmask


# class SeqInsertionDecoderAny(SeqInsertionDecoderUniform):
#     def get_slot_value_probs(self, slotvalues):     # uniform
#         probs = [1. for _ in slotvalues]
#         return probs
#
#     def compute_loss(self, logits, tgt, mask=None):
#         """
#         :param logits:
#         :param tgt:         will have a non-zero for every correct token
#         :param mask:        will be zero for positions that don't need insertion
#         :return:
#         """
#         probs = torch.softmax(logits, -1)
#         nonzero_tgt = (tgt > 0).float()
#         m = probs * nonzero_tgt
#         m = m.sum(-1)       # (batsize, seqlen)
#         loss = - torch.log(m.clamp_min(1e-6))
#         if mask is not None:
#             loss = loss * mask.float()
#         loss = loss.sum(-1)
#         return loss


# class SeqInsertionDecoderPredictiveBinary(SeqInsertionDecoderBinary):
#     ### Follows gold policy !!
#
#     def train_forward(self, x:torch.Tensor, gold:torch.Tensor):
#         enc, encmask = self.tagger.encode_source(x)
#         goldlens = (gold != 0).sum(-1)
#
#         y = torch.zeros(x.size(0), 2, device=x.device, dtype=torch.long)
#         y[:, 0] = self.vocab["@BOS@"]
#         y[:, 1] = self.vocab["@EOS@"]
#         ylens = (y != 0).sum(-1)
#
#         gold = torch.cat([y[:, 0:1], gold, torch.zeros_like(y[:, 1:2])], 1)
#         gold = gold.scatter(1, goldlens[:, None]+1, self.vocab["@EOS@"])
#         goldlens = (gold != 0).sum(-1)
#
#         yalign = torch.zeros_like(y)
#         yalign[:, 0] = 0
#         yalign[:, 1] = goldlens - 1
#
#         logitsacc = []
#         lossacc = torch.zeros(y.size(0), device=y.device)
#
#         newy = None
#         newyalign = None
#         ended = torch.zeros_like(y[:, 0]).bool()
#         while not torch.all(ended): #torch.any(ylens < goldlens):
#             # make newy the previous y
#             y = newy if newy is not None else y
#             yalign = newyalign if newyalign is not None else yalign
#
#             # region TRAIN
#             # compute target distribution and mask
#             tgt = torch.zeros(y.size(0), y.size(1), self.vocab.number_of_ids(),
#                               device=y.device)
#             _y = y.cpu().detach().numpy()
#             _yalign = yalign.cpu().detach().numpy()
#             _gold = gold.cpu().detach().numpy()
#
#             slotvalues_acc = []
#             for i in range(len(_y)):
#                 all_slotvalues = []
#                 for j in range(len(_y[i])):
#                     slotvalues = []
#                     if _y[i, j] == self.vocab["@EOS@"]:
#                         break
#                     # y_ij = _y[i, j]
#                     k = _yalign[i, j]   # this is where in the gold we're at
#                     k = k + 1
#                     # if j + 1 >= len(_yalign[i]):
#                     #     print("too large")
#                     #     pass
#                     while k < _yalign[i, j + 1]:
#                         slotvalues.append(_gold[i, k])
#                         k += 1
#                     if len(slotvalues) == 0:
#                         tgt[i, j, self.vocab["@END@"]] = 1
#                     else:
#                         for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
#                             tgt[i, j, slotvalue] += float(valueprob)
#                     all_slotvalues.append((slotvalues, self.get_slot_value_probs(slotvalues)))
#                 slotvalues_acc.append(all_slotvalues)
#             tgtmask = (tgt.sum(-1) != 0).float()
#             uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
#             tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
#             # run tagger on y
#             logits = self.tagger(tokens=y, enc=enc, encmask=encmask)
#             # do loss and backward
#             loss = self.compute_loss(logits, tgt[:, :-1], tgtmask[:, :-1])
#             loss.mean().backward(retain_graph=True)
#
#             lossacc = lossacc + loss.detach()
#             logitsacc.append(logits.detach().clone())
#             # endregion
#
#             # region STEP
#             # get argmax predicted token to insert at every slot
#             _logits = logits.cpu().detach().numpy()
#
#             newy = torch.zeros(y.size(0), y.size(1) + 1, device=y.device).long()
#             newyalign = torch.zeros(yalign.size(0), yalign.size(1) + 1, device=yalign.device).long()
#             _ended = torch.zeros_like(y[:, 0]).bool()
#             for i in range(len(y)):
#                 k = 0
#                 # randomly choose which slot to develop
#                 chosen_js = []
#                 for j in range(len(slotvalues_acc[i])):     # for every slot
#                     if _y[i, j] == self.vocab["@EOS@"]:
#                         break
#                     slotvalues = slotvalues_acc[i][j][0]
#                     if len(slotvalues) != 0:        # consider only positions where a real token must be predicted so not @END@
#                         chosen_js.append(j)
#
#                 terminated = True
#                 if len(chosen_js) == 0:     # if all slots terminated
#                     newyalign[i, :yalign.size(1)] = yalign[i]
#                     newy[i, :y.size(1)] = y[i]
#                 else:
#                     chosen_j = random.choice(chosen_js)
#
#                     for j in range(len(y[i])):
#                         if k >= newy.size(1):
#                             break
#                         newy[i, k] = y[i, j]        # copy
#                         newyalign[i, k] = yalign[i, j]
#                         k += 1
#                         y_ij = _y[i, j]
#                         if y_ij == self.vocab["@EOS@"]:  # if token was EOS, terminate generation
#                             break  # stop
#                         # if j >= len(p_i):  # if we reached beyond the length of predictions, terminate
#                         #     break
#
#                         if j == chosen_j:       # if we're at the chosen insertion slot:
#                             # get the most probable correct token
#                             logits_ij = logits[i, j]  # (vocabsize,)
#                             newprobs_ij = torch.zeros_like(logits_ij)
#                             slv = list(set(slotvalues_acc[i][j][0]))
#                             if len(slv) == 0:       # this slot is completed
#                                 newprobs_ij[self.vocab["@END@"]] = 1
#                             else:
#                                 for slv_item, slv_prob in zip(*slotvalues_acc[i][j]):
#                                     newprobs_ij[slv_item] += slv_prob
#                                 # newlogits_ij[slv] = logits_ij[slv]
#                             # pp_ij, p_ij = newlogits_ij.max(-1)
#                             p_ij = torch.multinomial(newprobs_ij, 1)[0]
#                             p_ij = p_ij.cpu().detach().item()
#                             if p_ij == self.vocab["@END@"]:  # if predicted token is @END@, do nothing
#                                 pass  # don't insert anything
#                             else:  # insert what was predicted
#                                 if k >= newy.size(1):
#                                     break
#                                 # insert token
#                                 newy[i, k] = p_ij
#                                 # align inserted token to gold
#                                 slotvalues = list(zip(*(slotvalues_acc[i][j] + (range(len(slotvalues_acc[i][j][0])),))))
#                                 slotvalues = sorted(slotvalues, key=lambda x: x[1], reverse=True)
#                                 aligned_pos = None
#                                 for slv, slp, sll in slotvalues:
#                                     if slv == p_ij:
#                                         aligned_pos = sll + newyalign[i, j] + 1
#                                         break
#                                 newyalign[i, k] = aligned_pos
#                                 k += 1  # advance newy pointer to next position
#                                 terminated = False  # sequence changed so don't terminate
#                 _ended[i] = terminated
#
#             y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
#             newy = torch.where(ended[:, None], y__, newy)  # prevent terminated examples from changing
#
#             maxlen = (newy != 0).long().sum(-1).max()
#             newy = newy[:, :maxlen]
#             # step += 1
#             ended = ended | _ended
#             # steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
#             lens = (newy != 0).long().sum(-1)
#             maxlenreached = (lens == self.max_size)
#             if torch.all(maxlenreached):
#                 break
#
#             # select a random slot by using mask
#             # find the most central token that is the same as predicted token for every slot and advance state
#
#             # endregion
#
#             logits = None
#             loss = None
#         lossacc.requires_grad = True
#         return {"loss": lossacc}, logitsacc


# class OldSeqInsertionDecoderPredictiveBinary(SeqInsertionDecoderBinary):
#     def train_forward(self, x:torch.Tensor, gold:torch.Tensor):
#         enc, encmask = self.tagger.encode_source(x)
#         goldlens = (gold != 0).sum(-1)
#
#         y = torch.zeros(x.size(0), 2, device=x.device, dtype=torch.long)
#         y[:, 0] = self.vocab["@BOS@"]
#         y[:, 1] = self.vocab["@EOS@"]
#         ylens = (y != 0).sum(-1)
#
#         gold = torch.cat([y[:, 0:1], gold, torch.zeros_like(y[:, 1:2])], 1)
#         gold = gold.scatter(1, goldlens[:, None]+1, self.vocab["@EOS@"])
#         goldlens = (gold != 0).sum(-1)
#
#         yalign = torch.zeros_like(y)
#         yalign[:, 0] = 0
#         yalign[:, 1] = goldlens - 1
#
#         logitsacc = []
#         lossacc = torch.zeros(y.size(0), device=y.device)
#
#         newy = None
#         newyalign = None
#         ended = torch.zeros_like(y[:, 0]).bool()
#         while not torch.all(ended): #torch.any(ylens < goldlens):
#             # make newy the previous y
#             y = newy if newy is not None else y
#             yalign = newyalign if newyalign is not None else yalign
#
#             # region TRAIN
#             # compute target distribution and mask
#             tgt = torch.zeros(y.size(0), y.size(1), self.vocab.number_of_ids(),
#                               device=y.device)
#             _y = y.cpu().detach().numpy()
#             _yalign = yalign.cpu().detach().numpy()
#             _gold = gold.cpu().detach().numpy()
#
#             slotvalues_acc = []
#             for i in range(len(_y)):
#                 all_slotvalues = []
#                 for j in range(len(_y[i])):
#                     slotvalues = []
#                     if _y[i, j] == self.vocab["@EOS@"]:
#                         break
#                     # y_ij = _y[i, j]
#                     k = _yalign[i, j]   # this is where in the gold we're at
#                     k = k + 1
#                     # if j + 1 >= len(_yalign[i]):
#                     #     print("too large")
#                     #     pass
#                     while k < _yalign[i, j + 1]:
#                         slotvalues.append(_gold[i, k])
#                         k += 1
#                     if len(slotvalues) == 0:
#                         tgt[i, j, self.vocab["@END@"]] = 1
#                     else:
#                         for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
#                             tgt[i, j, slotvalue] += float(valueprob)
#                     all_slotvalues.append((slotvalues, self.get_slot_value_probs(slotvalues)))
#                 slotvalues_acc.append(all_slotvalues)
#             tgtmask = (tgt.sum(-1) != 0).float()
#             uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
#             tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
#             # run tagger on y
#             logits = self.tagger(tokens=y, enc=enc, encmask=encmask)
#             # do loss and backward
#             loss = self.compute_loss(logits, tgt[:, :-1], tgtmask[:, :-1])
#             loss.mean().backward(retain_graph=True)
#
#             lossacc = lossacc + loss.detach()
#             logitsacc.append(logits.detach().clone())
#             # endregion
#
#             # region STEP
#             # get argmax predicted token to insert at every slot
#             # TODO: must predict the most probable correct tokens
#             # predprobs, preds = logits.max(-1)
#             # predprobs, preds = predprobs.cpu().detach().numpy(), preds.cpu().detach().numpy()
#             _logits = logits.cpu().detach().numpy()
#
#             newy = torch.zeros(y.size(0), y.size(1) + 1, device=y.device).long()
#             newyalign = torch.zeros(yalign.size(0), yalign.size(1) + 1, device=yalign.device).long()
#             _ended = torch.zeros_like(y[:, 0]).bool()
#             for i in range(len(y)):
#                 k = 0
#                 # randomly choose which slot to develop
#                 chosen_js = []
#                 for j in range(len(slotvalues_acc[i])):     # for every slot
#                     if _y[i, j] == self.vocab["@EOS@"]:
#                         break
#                     slotvalues = slotvalues_acc[i][j][0]
#                     if len(slotvalues) != 0:        # consider only positions where a real token must be predicted so not @END@
#                         chosen_js.append(j)
#
#                 terminated = True
#                 if len(chosen_js) == 0:     # if all slots terminated
#                     newyalign[i, :yalign.size(1)] = yalign[i]
#                     newy[i, :y.size(1)] = y[i]
#                 else:
#                     chosen_j = random.choice(chosen_js)
#
#                     for j in range(len(y[i])):
#                         if k >= newy.size(1):
#                             break
#                         newy[i, k] = y[i, j]        # copy
#                         newyalign[i, k] = yalign[i, j]
#                         k += 1
#                         y_ij = _y[i, j]
#                         if y_ij == self.vocab["@EOS@"]:  # if token was EOS, terminate generation
#                             break  # stop
#                         # if j >= len(p_i):  # if we reached beyond the length of predictions, terminate
#                         #     break
#
#                         if j == chosen_j:       # if we're at the chosen insertion slot:
#                             # get the most probable correct token
#                             logits_ij = logits[i, j]  # (vocabsize,)
#                             newlogits_ij = torch.zeros_like(logits_ij) - np.infty
#                             slv = list(set(slotvalues_acc[i][j][0]))
#                             if len(slv) == 0:       # this slot is completed
#                                 newlogits_ij[self.vocab["@END@"]] = 1
#                             else:
#                                 newlogits_ij[slv] = logits_ij[slv]
#                             pp_ij, p_ij = newlogits_ij.max(-1)
#                             p_ij = p_ij.cpu().detach().item()
#                             if p_ij == self.vocab["@END@"]:  # if predicted token is @END@, do nothing
#                                 pass  # don't insert anything
#                             else:  # insert what was predicted
#                                 if k >= newy.size(1):
#                                     break
#                                 # insert token
#                                 newy[i, k] = p_ij
#                                 # align inserted token to gold
#                                 slotvalues = list(zip(*(slotvalues_acc[i][j] + (range(len(slotvalues_acc[i][j][0])),))))
#                                 slotvalues = sorted(slotvalues, key=lambda x: x[1], reverse=True)
#                                 aligned_pos = None
#                                 for slv, slp, sll in slotvalues:
#                                     if slv == p_ij:
#                                         aligned_pos = sll + newyalign[i, j] + 1
#                                         break
#                                 newyalign[i, k] = aligned_pos
#                                 k += 1  # advance newy pointer to next position
#                                 terminated = False  # sequence changed so don't terminate
#                 _ended[i] = terminated
#
#             y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
#             newy = torch.where(ended[:, None], y__, newy)  # prevent terminated examples from changing
#
#             maxlen = (newy != 0).long().sum(-1).max()
#             newy = newy[:, :maxlen]
#             # step += 1
#             ended = ended | _ended
#             # steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
#             lens = (newy != 0).long().sum(-1)
#             maxlenreached = (lens == self.max_size)
#             if torch.all(maxlenreached):
#                 break
#
#             # select a random slot by using mask
#             # find the most central token that is the same as predicted token for every slot and advance state
#
#             # endregion
#
#             logits = None
#             loss = None
#         lossacc.requires_grad = True
#         return {"loss": lossacc}, logitsacc

#
# class SeqInsertionDecoderBinaryPredictive(SeqInsertionDecoderPredictive, SeqInsertionDecoderBinary): pass
# class SeqInsertionDecoderUniformPredictive(SeqInsertionDecoderPredictive, SeqInsertionDecoderUniform): pass


class SeqDecoderBaseline(SeqInsertionDecoder):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits, cache = self.tagger(tokens=newy, enc=enc, encmask=encmask, cache=None)
        # compute loss: different versions do different masking and different targets
        loss = self.compute_loss(logits, tgt, mask=tgtmask)
        return {"loss": loss}, logits

    def extract_training_example(self, x, y):
        ymask = (y != 0).float()
        ylens = ymask.sum(1).long()
        newy = y
        newy = torch.cat([torch.ones_like(newy[:, 0:1]) * self.vocab["@BOS@"], newy], 1)
        newy = torch.cat([newy, torch.zeros_like(newy[:, 0:1])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            newy[i, ylen+1] = self.vocab["@END@"]

        goldy = newy[:, 1:]
        tgt = torch.zeros(goldy.size(0), goldy.size(1), self.vocab.number_of_ids(), device=goldy.device)
        tgt = tgt.scatter(2, goldy[:, :, None], 1.)
        tgtmask = (goldy != 0).float()

        newy = newy[:, :-1]
        return x, newy, tgt, tgtmask

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@BOS@"]
        # yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.tagger.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        cache = None
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run tagger
            # y = torch.cat([y, yend], 1)
            logits, cache = self.tagger(tokens=y, enc=enc, encmask=encmask, cache=cache)
            _, preds = logits.max(-1)
            preds = preds[:, -1]
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        return newy, steps_used.float()


class SeqInsertionDecoderLTR(SeqInsertionDecoder):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    # def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
    #     # extract a training example from y:
    #     x, newy, tgt, tgtmask = self.extract_training_example(x, y)
    #     enc, encmask = self.tagger.encode_source(x)
    #     # run through tagger: the same for all versions
    #     logits = self.tagger(tokens=newy, enc=enc, encmask=encmask)
    #     # compute loss: different versions do different masking and different targets
    #     loss = self.compute_loss(logits, tgt, mask=tgtmask)
    #     return {"loss": loss}, logits

    def extract_training_example(self, x, y):
        # y: (batsize, seqlen) ids, padded with zeros
        ymask = (y != 0).float()
        ytotallens = ymask.sum(1)
        ylens = torch.rand(ytotallens.size(), device=ytotallens.device)
        ylens = (ylens * ytotallens).round().long()
        _ylens = ylens.cpu().numpy()
        # ylens contains the sampled lengths

        # mask randomly chosen tails
        z = torch.arange(y.size(1), device=y.device)
        _y = torch.where(z[None, :] < ylens[:, None], y, torch.zeros_like(y))
        _y = torch.cat([_y, torch.zeros_like(_y[:, 0:1])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            _y[i, ylen] = self.vocab["@EOS@"]
        # prepend BOS
        newy = torch.cat([torch.ones_like(y[:, 0:1]) * self.vocab["@BOS@"], _y], 1)

        _y = torch.cat([y, torch.zeros_like(y[:, 0:1])], 1)
        golds = _y.gather(1, ylens[:, None]).squeeze(1)       # (batsize,)
        golds = torch.where(golds != 0, golds, torch.ones_like(golds) * self.vocab["@END@"])        # when full sequence has been fed, and mask is what remains, make sure that we have @EOS@ instead
        tgt = torch.zeros(newy.size(0), newy.size(1), self.vocab.number_of_ids(), device=newy.device)

        for i, tgt_pos, tgt_val in zip(range(len(ylens)), ylens, golds):
            tgt[i, tgt_pos, tgt_val] = 1

        # normalize
        tgt = tgt / tgt.sum(-1).clamp_min(1e-6)[:, :, None]
        tgtmask = (tgt.sum(-1) != 0).float()
        # make uniform for masked positions
        newymask = (newy != 0).float()
        uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
        tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
        # cut unnecessary padded elements from the right of newy
        newlen = newymask.sum(-1).max()
        newy = newy[:, :int(newlen)]
        tgt = tgt[:, :int(newlen)]
        tgtmask = tgtmask[:, :int(newlen)]

        return x, newy, tgt, tgtmask

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@BOS@"]
        yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.tagger.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run tagger
            y_ = torch.cat([y, yend], 1)
            logits = self.tagger(tokens=y_, enc=enc, encmask=encmask)
            _, preds = logits[:, -1].max(-1)
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        return newy, steps_used.float()


def run(domain="restaurants",
        mode="baseline",         # "baseline", "ltr", "uniform", "binary"
        probthreshold=0.,       # 0. --> parallel, >1. --> serial, 0.< . <= 1. --> semi-parallel
        lr=0.0001,
        enclrmul=0.1,
        batsize=50,
        epochs=1000,
        hdim=366,
        numlayers=6,
        numheads=6,
        dropout=0.1,
        noreorder=False,
        trainonvalid=False,
        seed=87646464,
        gpu=-1,
        patience=-1,
        gradacc=1,
        cosinelr=False,
        warmup=20,
        gradnorm=3,
        validinter=10,
        maxsteps=20,
        maxsize=75,
        testcode=False,
        numbered=False,
        betternumbered=False,
        oraclemix=0.,
        goldtemp=1.,
        evaltrain=False,
        cooldown=50
        ):
    if betternumbered:
        numbered = True
    settings = locals().copy()
    settings["version"] = "v2.1"
    q.pp_dict(settings)

    wandb.init(project=f"seqinsert_overnight_v2", config=settings, reinit=True)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading")
    tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds(domain, trainonvalid=trainonvalid, noreorder=noreorder, numbered=numbered)
    tt.tock("loaded")

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model
    tagger = TransformerTagger(hdim, flenc.vocab, numlayers, numheads, dropout, baseline=mode=="baseline", vocab_factorized=betternumbered)

    if mode == "baseline":
        decoder = SeqDecoderBaseline(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize)
    elif mode == "ltr":
        decoder = SeqInsertionDecoderLTR(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize)
    elif mode == "uniform":
        decoder = SeqInsertionDecoderUniform(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold, oraclemix=oraclemix)
    elif mode == "binary":
        decoder = SeqInsertionDecoderBinary(tagger, flenc.vocab,
                                            max_steps=maxsteps,
                                            max_size=maxsize,
                                            prob_threshold=probthreshold,
                                            oraclemix=oraclemix,
                                            tau=goldtemp)
    # elif mode == "maxspanbinary":
    #     decoder = SeqInsertionDecoderMaxspanBinary(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)
    # elif mode == "predictivebinary":
    #     decoder = SeqInsertionDecoderPredictiveBinary(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)
    # elif mode == "any":
    #     decoder = SeqInsertionDecoderAny(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)

    # test run
    if testcode:
        batch = next(iter(tdl_seq))
        # out = tagger(batch[1])
        # out = decoder(*batch)
        decoder.train(False)
        out = decoder(*batch)

    tloss = make_array_of_metrics("loss", "acc", "recall", reduction="mean")
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
    eyt = q.EarlyStopper(vmetrics[0], patience=patience, min_epochs=30, more_is_better=True, remember_f=lambda: deepcopy(tagger))

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
        decoder.tagger = eyt.remembered
        tagger = eyt.remembered

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

    if mode != "baseline":
        calibrate_end = False
        if calibrate_end:
            # calibrate END offset
            tt.tick("running termination calibration")
            end_offsets = [0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            decoder.prob_threshold = 1.

            end_offset_values = []

            best_offset = 0.
            best_offset_value = 0.
            for end_offset in end_offsets:
                tt.tick("rerunning validation")
                decoder.end_offset = end_offset
                validres = validepoch()
                tt.tock(f"Validation results: {validres}")
                end_offset_values.append(vmetrics[0].get_epoch_error())
                if vmetrics[0].get_epoch_error() > best_offset_value:
                    best_offset = end_offset
                    best_offset_value = vmetrics[0].get_epoch_error()
                tt.tock("done")
            print(f"offset results: {dict(zip(end_offsets, end_offset_values))}")
            print(f"best offset: {best_offset}")

            decoder.end_offset = best_offset

        # run different prob_thresholds:
        # thresholds = [0., 0.3, 0.5, 0.6, 0.75, 0.85, 0.9, 0.95,  1.]
        thresholds = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1.]
        for threshold in thresholds:
            tt.tick("running test for threshold " + str(threshold))
            decoder.prob_threshold = threshold
            testres = testepoch()
            print(f"Test tree acc for threshold {threshold}: testres: {testres}")
            settings.update({f"_thr{threshold}_acc": xmetrics[0].get_epoch_error()})
            settings.update({f"_thr{threshold}_len": xmetrics[1].get_epoch_error()})
            tt.tock("done")


    wandb.config.update(settings)
    q.pp_dict(settings)

# TODO: EOS balancing ?!

# TODO: model that follows predictive distribution during training and uses AnyToken loss

def run_experiment(domain="default",    #
                   mode="baseline",         # "baseline", "ltr", "uniform", "binary"
                   probthreshold=-1.,
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
        maxsize=90,
        numbered=False,
        betternumbered=False,
        oraclemix=0.,
        goldtemp=-1.,
        evaltrain=False,
                   testcode=False,
        ):

    settings = locals().copy()

    ranges = {
        "domain": ["socialnetwork", "blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "epochs": [121],
        "batsize": [50],
        "hdim": [366, 768],
        "numheads": [6, 12],
        "numlayers": [6, 8, 12],
        "lr": [0.0001, 0.000025],
        "enclrmul": [1., 0.1],                  # use 1.
        "seed": [87646464],
        "patience": [-1],
        "warmup": [20],
        "validinter": [10],
        "gradacc": [1],
    }

    if mode == "baseline":        # baseline
        ranges["validinter"] = [5]
    elif mode.startswith("predictive"):
        ranges["validinter"] = [1]
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["dropout"] = [0.0, 0.1, 0.3]     # use 0.
        ranges["hdim"] = [768]
        ranges["numlayers"] = [6]
        ranges["numheads"] = [12]
        ranges["numbered"] = [False]
    else:
        if settings["domain"] != "default":
            domains = settings["domain"].split(",")
            ranges["domain"] = domains
            settings["domain"] = "default"
        else:
            # ranges["domain"] = ["blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"]
            # ranges["domain"] = ["calendar", "restaurants", "publications", "recipes"]
            ranges["domain"] = ["blocks", "housing", "socialnetwork", "basketball"]
        # ranges["domain"] = ["restaurants", "recipes"]
        ranges["batsize"] = [30]
        ranges["dropout"] = [0.2, 0.1, 0.0]     # use 0.
        # ranges["lr"] = [0.0001]                 # use 0.000025
        ranges["validinter"] = [10]
        ranges["epochs"] = [161]
        ranges["hdim"] = [768]
        ranges["numlayers"] = [6]
        ranges["numheads"] = [12]
        ranges["probthreshold"] = [0.]
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["goldtemp"] = [1.0, 0.1]

    if mode == "ltr":
        ranges["lr"] = [0.0001, 0.000025]
        ranges["warmup"] = [50]
        ranges["epochs"] = [501]
        ranges["validinter"] = [25]
        ranges["gradacc"] = [10]
        ranges["hdim"] = [768]
        ranges["numlayers"] = [6]
        ranges["numheads"] = [12]

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
        if spec["hdim"] == 366 and spec["numheads"] != 6:
            return False
        if spec["hdim"] == 768 and spec["numheads"] != 12:
            return False
        return True

    print(__file__)
    p = __file__ + f".baseline.{domain}"
    q.run_experiments_random(
        run, ranges, path_prefix=p, check_config=checkconfig, **settings)



if __name__ == '__main__':
    q.argprun(run_experiment)
    # DONE: fix orderless for no simplification setting used here
    # DONE: make baseline decoder use cached decoder states
    # DONE: in unsimplified Overnight, the filters are nested but are interchangeable! --> use simplify filters ?!
    # python overnight_seqinsert.py -gpu 0 -domain ? -lr 0.0001 -enclrmul 1. -hdim 768 -dropout 0.3 -numlayers 6 -numheads 12