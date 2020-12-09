from prompt_toolkit.formatted_text import PygmentsTokens

import json
import random
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Dict

import torch
import wandb
from nltk import Tree
import numpy as np
from torch.utils.data import DataLoader

import qelos as q

from parseq.datasets import OvernightDatasetLoader, autocollate
from parseq.eval import make_array_of_metrics
from parseq.grammar import tree_to_lisp_tokens, are_equal_trees, lisp_to_tree
from parseq.scripts_insert.overnight_treeinsert import extract_info
from parseq.scripts_insert.util import reorder_tree, flatten_tree
from parseq.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


ORDERLESS = {"op:and", "SW:concat"}


def tree_to_seq(x:Tree):
    xstr = tree_to_lisp_tokens(x)
    # xstr = ["@BOS@"] + xstr + ["@EOS@"]
    return xstr


def load_ds(domain="restaurants", nl_mode="bert-base-uncased",
            trainonvalid=False, noreorder=False):
    """
    Creates a dataset of examples which have
    * NL question and tensor
    * original FL tree
    * reduced FL tree with slots (this is randomly generated)
    * tensor corresponding to reduced FL tree with slots
    * mask specifying which elements in reduced FL tree are terminated
    * 2D gold that specifies whether a token/action is in gold for every position (compatibility with MML!)
    """
    orderless = {"op:and", "SW:concat"}     # only use in eval!!

    ds = OvernightDatasetLoader(simplify_mode="none").load(domain=domain, trainonvalid=trainonvalid)
    # ds contains 3-tuples of (input, output tree, split name)

    if not noreorder:
        ds = ds.map(lambda x: (x[0], reorder_tree(x[1], orderless=orderless), x[2]))
    ds = ds.map(lambda x: (x[0], tree_to_seq(x[1]), x[2]))

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


class TransformerTagger(SeqInsertionTagger):
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", baseline=False, **kw):
        super(TransformerTagger, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.baseline = baseline
        config = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                   num_layers=numlayers, num_heads=numheads, dropout_rate=dropout,
                                   use_relative_position=False)

        self.emb = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.posemb = torch.nn.Embedding(maxpos, config.d_model)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = baseline
        self.decoder = TransformerStack(decoder_config)

        if baseline:
            self.out = torch.nn.Linear(self.dim, self.vocabsize)
        else:
            self.out = torch.nn.Linear(self.dim * 2, self.vocabsize)
        # self.out = MOS(self.dim, self.vocabsize, K=mosk)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname)
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

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None):
        padmask = (tokens != 0)
        # if not self.baseline:
        #     padmask = padmask[:, 1:]
        embs = self.emb(tokens)
        posembs = self.posemb(torch.arange(tokens.size(1), device=tokens.device))[None]
        embs = embs + posembs
        ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=False)
        ret = ret[0]
        if self.baseline:
            c = ret
        else:
            c = torch.cat([ret[:, 1:], ret[:, :-1]], -1)
        logits = self.out(c)
        # logits = logits + torch.log(self.vocab_mask[None, None, :])
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
                 **kw):
        super(SeqInsertionDecoder, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_steps = max_steps
        self.max_size = max_size
        self.kldiv = torch.nn.KLDivLoss(reduction="none")
        self.logsm = torch.nn.LogSoftmax(-1)
        self.prob_threshold = prob_threshold

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
        return kl

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits = self.tagger(tokens=newy, enc=enc, encmask=encmask)
        # compute loss: different versions do different masking and different targets
        loss = self.compute_loss(logits, tgt[:, :-1], mask=tgtmask[:, :-1])
        return {"loss": loss}, logits

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
            probs = torch.softmax(logits, -1)
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
                prob_thresh = min(self.prob_threshold, max(pp_i))
                terminated = True
                for j in range(len(y[i])):
                    if k >= newy.size(1):
                        break
                    newy[i, k] = y[i, j]
                    k += 1
                    y_ij = _y[i, j]
                    if y_ij == self.vocab["@EOS@"]:
                        break  # stop
                    p_ij = p_i[j]
                    pp_ij = pp_i[j]
                    if pp_ij >= prob_thresh:
                        if p_ij == self.vocab["@END@"]:
                            pass  # don't insert anything
                        else:  # insert what was predicted
                            if k >= newy.size(1):
                                break
                            newy[i, k] = preds[i, j]
                            k += 1
                            terminated = False
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
        probs = [1./len(slotvalues) for _ in slotvalues]
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
            perm = torch.randperm(ytotallens[i].long().cpu().item())
            perm = perm[:ylens[i].long().cpu().item()]
            select, _ = perm.sort(-1)
            select = list(select.cpu().numpy())
            k = 1  # k is where in the new sampled sequence we're at

            slotvalues = []
            for j in range(int(ytotallens[i].cpu().item())):
                y_ij = y[i, j].cpu().item()
                if k <= len(select) and j == select[k - 1]:  # if j-th token in y should be k-th in newy
                    newy[i, k] = y[i, j]
                    if len(slotvalues) == 0:
                        tgt[i, k - 1, self.vocab["@END@"]] = 1
                    else:
                        for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
                            tgt[i, k - 1, slotvalue] = float(valueprob)
                    slotvalues = []
                    k += 1
                else:  # otherwise, add
                    slotvalues.append(y_ij)
                    # tgt[i, k - 1, y_ij] = 1

            if len(slotvalues) == 0:
                tgt[i, k - 1, self.vocab["@END@"]] = 1
            else:
                for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
                    tgt[i, k - 1, slotvalue] = float(valueprob)

            newy[i, k] = self.vocab["@EOS@"]

        # normalize
        tgt = tgt / tgt.sum(-1).clamp_min(1e-6)[:, :, None]
        tgtmask = (tgt.sum(-1) != 0).float()
        # make uniform for masked positions
        newymask = (newy != 0).float()
        uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
        tgt = torch.where(newymask[:, :, None].bool(), tgt, uniform_tgt)
        # cut unnecessary padded elements from the right of newy
        newlen = newymask.sum(-1).max()
        newy = newy[:, :int(newlen)]
        tgt = tgt[:, :int(newlen)]
        tgtmask = tgtmask[:, :int(newlen)]

        return x, newy, tgt, tgtmask


class SeqInsertionDecoderBinary(SeqInsertionDecoderUniform):
    """ Differs from Uniform only in computing and using non-uniform weights for gold output distributions """
    def __init__(self, tagger:SeqInsertionTagger,
                 vocab=None,
                 prob_threshold=0.,
                 max_steps:int=20,
                 max_size:int=100,
                 tau=1.,
                 **kw):
        super(SeqInsertionDecoderBinary, self).__init__(tagger, vocab=vocab,
                                                        max_steps=max_steps,
                                                        max_size=max_size,
                                                        prob_threshold=prob_threshold,
                                                        **kw)
        self.tau = tau

    def get_slot_value_probs(self, slotvalues):
        center = len(slotvalues) / 2 - 0.5
        distances = [abs(x - center) for x in range(len(slotvalues))]
        distances = torch.tensor(distances)
        probs = torch.softmax(-distances/self.tau, -1)
        probs = probs.numpy()
        return probs


class SeqDecoderBaseline(SeqInsertionDecoder):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits = self.tagger(tokens=newy, enc=enc, encmask=encmask)
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
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run tagger
            # y = torch.cat([y, yend], 1)
            logits = self.tagger(tokens=y, enc=enc, encmask=encmask)
            _, preds = logits.max(-1)
            preds = preds[:, -1]
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        preds = newy
        return preds, steps_used.float()


class SeqInsertionDecoderLTR(SeqInsertionDecoder):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits = self.tagger(tokens=newy, enc=enc, encmask=encmask)
        # compute loss: different versions do different masking and different targets
        loss = self.compute_loss(logits, tgt, mask=tgtmask)
        return {"loss": loss}, logits

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
        ):

    settings = locals().copy()
    q.pp_dict(settings)
    wandb.init(project=f"seqinsert_overnight", config=settings, reinit=True)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading")
    tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds(domain, trainonvalid=trainonvalid, noreorder=noreorder)
    tt.tock("loaded")

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model
    tagger = TransformerTagger(hdim, flenc.vocab, numlayers, numheads, dropout, baseline=mode=="baseline")

    if mode == "baseline":
        decoder = SeqDecoderBaseline(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize)
    elif mode == "ltr":
        decoder = SeqInsertionDecoderLTR(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize)
    elif mode == "uniform":
        decoder = SeqInsertionDecoderUniform(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)
    elif mode == "binary":
        decoder = SeqInsertionDecoderBinary(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)

    # test run
    if testcode:
        batch = next(iter(tdl_seq))
        # out = tagger(batch[1])
        # out = decoder(*batch)
        decoder.train(False)
        out = decoder(*batch)

    tloss = make_array_of_metrics("loss", reduction="mean")
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
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=t_max-warmup) >> 0.
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
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=[validepoch],
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
        print(f"Validation results: {validres}")

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

# TODO: EOS balancing ?!


def run_experiment(domain="default",    #
                   mode="baseline",         # "baseline", "ltr", "uniform", "binary"
                   probthreshold=0.,
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
        gradacc=1,
        cosinelr=False,
        warmup=-1,
        gradnorm=3.,
        validinter=-1,
        maxsteps=75,        # TODO: stop insertion decoding when max size is reached
        maxsize=75,
                   testcode=False,
        ):

    settings = locals().copy()

    ranges = {
        "domain": ["calendar", "blocks", "housing", "recipes"], #["socialnetwork", "blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "epochs": [121],
        "batsize": [50],
        "hdim": [366, 768],
        "numheads": [6, 12],
        "numlayers": [6, 8, 12],
        "lr": [0.0001, 0.000025],
        "enclrmul": [1., 0.1],
        "seed": [87646464],
        "patience": [-1],
        "warmup": [20],
        "validinter": [15],

    }

    if True:        # baseline
        ranges["validinter"] = [5]

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

    # python overnight_seqinsert.py -gpu 0 -domain ? -lr 0.0001 -enclrmul 1. -hdim 768 -dropout 0.3 -numlayers 6 -numheads 12