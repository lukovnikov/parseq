import json
import math
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from typing import Dict, List, Union

import sklearn
import wandb

import qelos as q
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader
from transformers import AutoTokenizer, BertModel

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree, tree_to_lisp
from parseq.scripts_compgen_new.transformer import TransformerConfig, TransformerStack
from parseq.scripts_compgen_new.transformerdecoder import TransformerStack as TransformerStackDecoder
from parseq.vocab import Vocab

from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd


class Inspector(object):
    def __init__(self, **kw):
        super(Inspector, self).__init__(**kw)
        self.epochs = []
        self.is_on = False
        self.new_epoch()

    def turn_on(self):
        self.is_on = True

    def turn_off(self):
        self.is_on = False

    def add_batch(self, data):
        self.epochs[-1].append(data)

    def new_epoch(self):
        self.epochs.append([])



def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


class TransformerEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_size, dropout=0., pad_token_id=0, max_position_embeddings=512,
                 layer_norm_eps=1e-12, useabspos=True, usesinpos=False):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        if usesinpos:
            self.position_embeddings = PositionalEncoding(hidden_size, max_position_embeddings)
        else:
            self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

        self.useabspos = useabspos

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        ret = inputs_embeds

        if self.useabspos:
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
            position_embeddings = self.position_embeddings(position_ids)
            ret = ret + position_embeddings

        ret = self.LayerNorm(ret)
        ret = self.dropout(ret)
        return ret


class BasicRelPosEmb(torch.nn.Module):
    ## Note: Even if this is shared across layers, keep the execution separate between layers because attention weights are different
    def __init__(self, dim, rng=10, dropout=0., **kw):
        super(BasicRelPosEmb, self).__init__(**kw)
        self.dropout = torch.nn.Dropout(dropout)

        self.D = ["@PAD@"] + [str(i-rng) for i in range(rng)] + [str(i) for i in range(rng+1)]
        self.D = dict(zip(self.D, range(len(self.D))))
        self.emb = torch.nn.Embedding(len(self.D), dim, padding_idx=0)
        self.embv = torch.nn.Embedding(len(self.D), dim, padding_idx=0)

    def get_vectors(self, query, relpos, keyorvalue="key"):
        """
        :param q:       (batsize, numheads, qlen, dimperhead)
        :param relpos:  (batsize, qlen, klen, n)
        :return:        (batsize, numheads, qlen, klen, dimperhead)
        """
        ret = None
        for n in range(relpos.size(-1)):
            indexes = torch.arange(0, self.emb.num_embeddings, device=query.device).long()
            if keyorvalue.startswith("k"):
                embs = self.emb(indexes)# (numindexes, dim)
            elif keyorvalue.startswith("v"):
                embs = self.embv(indexes)
            embs = embs.view(embs.size(0), query.size(1), query.size(-1))        # (numindexes, numheads, dimperhead)
            vectors = relpos[:, :, :, n][embs]      # (batsize, qlen, klen, numheads, dimperhead)
            vectors = vectors.permute(0, 3, 1, 2, 4)
            if ret is None:
                ret = torch.zeros_like(vectors)
            ret = ret + vectors
        return ret

    def compute_scores(self, query, relpos):
        """
        :param q:       (batsize, numheads, qlen, dimperhead)
        :param relpos:  (batsize, qlen, klen, n)
        :return:
        """
        retscores = None
        for n in range(relpos.size(-1)):
            indexes = torch.arange(0, self.emb.num_embeddings, device=query.device).long()
            embs = self.emb(indexes)# (numindexes, dim)
            embs = self.dropout(embs)
            embs = embs.view(embs.size(0), query.size(1), query.size(-1))        # (numindexes, numheads, dimperhead)
            relpos_ = relpos[:, :, :, n]
            scores = torch.einsum("bhqd,nhd->bhqn", query, embs)  # (batsize, numheads, qlen, numindexes)
            relpos_ = relpos_[:, None, :, :].repeat(scores.size(0), scores.size(1), 1, 1)  # (batsize, numheads, qlen, klen)
            # print(scores.size(), relpos_.size())
            scores_ = torch.gather(scores, 3, relpos_)  # (batsize, numheads, qlen, klen)
            if retscores is None:
                retscores = torch.zeros_like(scores_)
            retscores = retscores + scores_
        return retscores        # (batsize, numheads, qlen, klen)

    def compute_context(self, weights, relpos):
        """
        :param weights: (batsize, numheads, qlen, klen)
        :param relpos:  (batsize, qlen, klen, 1)
        :return:    # weighted sum over klen (batsize, numheads, qlen, dimperhead)
        """
        ret = None
        batsize = weights.size(0)
        numheads = weights.size(1)
        qlen = weights.size(2)
        device = weights.device

        # Naive implementation builds matrices of (batsize, numheads, qlen, klen, dimperhead)
        # whereas normal transformer only (batsize, numheads, qlen, klen) and (batsize, numheads, klen, dimperhead)
        for n in range(relpos.size(-1)):
            relpos_ = relpos[:, :, :, n]

            # map relpos_ to compact integer space of unique relpos_ entries
            try:
                relpos_unique = relpos_.unique()
            except Exception as e:
                raise e
            mapper = torch.zeros(relpos_unique.max() + 1, device=device, dtype=torch.long)  # mapper is relpos_unique but the other way around
            mapper[relpos_unique] = torch.arange(0, relpos_unique.size(0), device=device).long()
            relpos_mapped = mapper[relpos_]     # (batsize, qlen, klen) but ids are from 0 to number of unique relposes

            # sum up the attention weights which refer to the same relpos id
            # scatter: src is weights, index is relpos_mapped[:, None, :, :]
            # scatter: gathered[batch, head, qpos, relpos_mapped[batch, head, qpos, kpos]]
            #               += weights[batch, head, qpos, kpos]
            gathered = torch.zeros(batsize, numheads, qlen, relpos_unique.size(0), device=device)
            gathered = torch.scatter_add(gathered, -1, relpos_mapped[:, None, :, :].repeat(batsize, numheads, 1, 1), weights)
            # --> (batsize, numheads, qlen, numunique): summed attention weights

            # get embeddings and update ret
            embs = self.embv(relpos_unique).view(relpos_unique.size(0), numheads, -1)        # (numunique, numheads, dimperhead)
            embs = self.dropout(embs)
            relposemb = torch.einsum("bhqn,nhd->bhqd", gathered, embs)
            if ret is None:
                ret = torch.zeros_like(relposemb)
            ret  = ret + relposemb
        return ret


class WordDropout(torch.nn.Module):
    def __init__(self, p=0., maskid=1, keepids=None, **kw):
        super(WordDropout, self).__init__(**kw)
        self.dropout = torch.nn.Dropout(p)
        self.maskid = maskid
        self.keepids = [] if keepids is None else keepids

    def forward(self, x):
        if self.training and self.dropout.p > 0:
            worddropoutmask = self.dropout(torch.ones_like(x).float()) > 0
            for keepid in self.keepids:
                worddropoutmask = worddropoutmask | (x == keepid)
            x = torch.where(worddropoutmask, x, torch.ones_like(x) * self.maskid)
        return x


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, maxlen=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(maxlen, d_model)
        position = torch.arange(0, maxlen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        posenc = self.pe[x]
        return posenc


class TransformerDecoderCell(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, usesinpos=False, useabspos=True,
                 relposmode="basic", relposrng=10, mode="normal",
                 dropout:float=0., worddropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(TransformerDecoderCell, self).__init__(**kw)
        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.userelpos = userelpos
        self.relposrng = relposrng
        self.useabspos = useabspos
        self.usesinpos = usesinpos
        self.mode = mode

        decconfig = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                      d_kv=int(self.dim/numheads),
                                      num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)

        self.dec_emb = torch.nn.Embedding(self.vocabsize+3, decconfig.d_model)

        self.relposemb = None
        if self.userelpos is True:
            if relposmode == "basic":
                print("using basic relative position encoding")
                # self.relposemb = [BasicRelPosEmb(self.dim, relposrng, dropout=dropout) for _ in range(decconfig.num_layers)]
                self.relposemb = BasicRelPosEmb(self.dim, relposrng, dropout=dropout)
            else:
                raise Exception(f"Unrecognized relposmode '{relposmode}'")

        self.absposemb = None
        if self.relposemb is None or self.useabspos is True:
            if self.usesinpos:
                print("using sinusoid absolute position encoding")
                self.absposemb = PositionalEncoding(decconfig.d_model, maxlen=maxpos)
            else:
                print("using learned absolute position encoding")
                self.absposemb = torch.nn.Embedding(maxpos, decconfig.d_model)

        decoder_config = deepcopy(decconfig)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = True
        self.decoder = TransformerStackDecoder(decoder_config, rel_emb=self.relposemb)

        self.out = torch.nn.Linear(self.dim, self.vocabsize+3)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.encrelposemb = None
        if self.bertname.startswith("none") or self.bertname == "vanilla":
            if self.userelpos is True:
                if relposmode == "basic":
                    # self.encrelposemb = [BasicRelPosEmb(self.dim, relposrng, dropout=dropout) for _ in range(numlayers)]
                    self.encrelposemb = BasicRelPosEmb(self.dim, relposrng, dropout=dropout)
                else:
                    raise Exception(f"Unrecognized relposmode '{relposmode}'")
            bname = "bert" + self.bertname[4:]
            if self.bertname == "vanilla":
                inpvocabsize = inpvocab.number_of_ids()
                self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken], [self.inpvocab[self.inpvocab.padtoken]])
            else:
                tokenizer = AutoTokenizer.from_pretrained(bname)
                inpvocabsize = tokenizer.vocab_size
                self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken],
                                                  [self.inpvocab["[CLS]"], self.inpvocab["[SEP]"], self.inpvocab[self.inpvocab.padtoken]])
            encconfig = TransformerConfig(vocab_size=inpvocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                          d_kv=int(self.dim/numheads),
                                          num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)
            encemb = TransformerEmbeddings(encconfig.vocab_size, encconfig.d_model, dropout=dropout, max_position_embeddings=maxpos, useabspos=useabspos, usesinpos=usesinpos)
            self.encoder_model = TransformerStack(encconfig, encemb, rel_emb=self.encrelposemb)
        else:
            self.encoder_model = BertModel.from_pretrained(self.bertname,
                                                           hidden_dropout_prob=min(dropout, 0.2),
                                                           attention_probs_dropout_prob=min(dropout, 0.1))
            tokenizer = AutoTokenizer.from_pretrained(self.bertname)
            inpvocabsize = tokenizer.vocab_size
            self.inpvocab = Vocab()
            for tok, id in tokenizer.vocab.items():
                self.inpvocab.D[tok] = id
            self.inpvocab.masktoken = "[MASK]"
            self.inpvocab.unktoken = "[UNK]"
            self.inpvocab.padtoken = "[PAD]"
            self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken], [self.inpvocab["[CLS]"], self.inpvocab["[SEP]"], self.inpvocab[self.inpvocab.padtoken]])

        self.adapter = None
        if self.encoder_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.encoder_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.worddropout = WordDropout(worddropout, self.vocab[self.vocab.masktoken], [self.vocab[self.vocab.padtoken]])

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
        if self.mode == "zeroinp":
            encs = torch.zeros(x.size(0), x.size(1), self.dim, device=x.device)
            return encs, encmask
        x = self.inpworddropout(x)
        relpos = None
        if self.encrelposemb is not None:      # compute relative positions
            positions = torch.arange(x.size(1), device=x.device)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        if relpos is not None:
            encs = self.encoder_model(x, attention_mask=encmask, relpos=relpos)[0]
        else:
            encs = self.encoder_model(x, attention_mask=encmask)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
        padmask = (tokens != 0)
        embs = self.dec_emb(tokens)
        if self.absposemb is not None:
            posembs = self.absposemb(torch.arange(tokens.size(1), device=tokens.device))[None]
            embs = embs + posembs
        relpos = None
        if self.relposemb is not None:      # compute relative positions
            positions = torch.arange(tokens.size(1), device=tokens.device)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        if cache is not None:
            embs = embs[:, -1:, :]
            if relpos is not None:
                relpos = relpos[:, -1:, :, :]
        _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=True,
                     past_key_value_states=cache,
                     relpos=relpos)
        ret = _ret[0]
        c = ret
        cache = _ret[1]
        logits = self.out(c)
        return logits, cache


class SeqDecoderBaseline(torch.nn.Module):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def __init__(self, tagger:Union[TransformerDecoderCell, torch.nn.ModuleList],
                 vocab=None,
                 max_size:int=100,
                 smoothing:float=0.,
                 mode="normal",
                 mcdropout=-1,
                 innerensemble=False,
                 **kw):
        super(SeqDecoderBaseline, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_size = max_size
        self.smoothing = smoothing
        self.mode = mode
        if self.smoothing > 0:
            self.loss = q.SmoothedCELoss(reduction="none", ignore_index=0, smoothing=smoothing, mode="logprobs")
        else:
            self.loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)

        self.logsm = torch.nn.LogSoftmax(-1)

        self.mcdropout = mcdropout
        self.innerensemble = innerensemble
        self.ensemble = len(self.tagger) if isinstance(self.tagger, torch.nn.ModuleList) else -1

        assert not (self.ensemble > 0 and self.mcdropout > 0), "can't have both mc dropout and regular ensemble"

        self.inspect = False

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

    def compute_loss(self, logits, tgt):

        """
        :param logits:      (batsize, seqlen, vocsize)
        :param tgt:         (batsize, seqlen)
        :return:
        """
        mask = (tgt != 0).float()

        logprobs = self.logsm(logits)
        if self.smoothing > 0:
            loss = self.loss(logprobs, tgt)
        else:
            loss = self.loss(logprobs.permute(0, 2, 1), tgt)      # (batsize, seqlen)
        loss = loss * mask
        loss = loss.sum(-1)

        best_pred = logits.max(-1)[1]   # (batsize, seqlen)
        best_gold = tgt
        same = best_pred == best_gold

        elemacc = same.float().sum(-1) / mask.float().sum(-1)

        same = same | ~(mask.bool())
        acc = same.all(-1)  # (batsize,)
        return loss, acc.float(), elemacc

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        preds, prednll, maxmaxnll, entropy, total, avgconf, sumnll, stepsused, allprobs, allmask\
            = self.get_prediction(x)

        def tensor_to_trees(x, vocab:Vocab):
            xstrs = [vocab.tostr(x[i]).replace("@START@", "") for i in range(len(x))]
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
                    tree = taglisp_to_tree(xstr)
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

        if (self.mcdropout > 0 or self.ensemble > 0) and not self.innerensemble:
            probses = []
            preds = preds[:, 1:]
            if self.mcdropout > 0:
                self.train()
                for i in range(self.mcdropout):
                    d, logits = self.train_forward(x, preds)
                    probses.append(torch.softmax(logits[0], -1))
                self.eval()
                probses = sum(probses) / len(probses)
                probses = probses[:, :-1]
            elif self.ensemble > 0:
                d, logitses = self.train_forward(x, preds)
                probses = [torch.softmax(logits, -1) for logits in logitses]
                probses = sum(probses) / len(probses)
                probses = probses[:, :-1]
            probs = probses
            mask = preds > 0
            confs = torch.gather(probs, 2, preds[:, :, None])[:, :, 0]
            nlls = -torch.log(confs)

            avgconf = (confs + (1-mask.float())).prod(-1)
            avgnll = (nlls * mask).sum(-1) / mask.float().sum(-1).clamp(1e-6)
            sumnll = (nlls * mask).sum(-1)
            maxnll, _ = (nlls + (1 - mask.float()) * -10e6).max(-1)
            entropy = (-torch.log(probs.clamp_min(1e-7)) * probs).sum(-1)
            entropy = (entropy * mask).sum(-1) / mask.float().sum(-1).clamp(1e-6)
            ret["decnll"] = avgnll
            ret["sumnll"] = sumnll
            ret["maxmaxnll"] = maxnll
            ret["entropy"] = entropy
            ret["avgconf"] = avgconf
        else:
            ret["decnll"] = prednll
            ret["sumnll"] = sumnll
            ret["maxmaxnll"] = maxmaxnll
            ret["entropy"] = entropy
            ret["avgconf"] = avgconf

            if self.inspect:
                ret["inspect_x"] = x
                ret["inspect_gold"] = gold
                ret["inspect_pred"] = preds[:, 1:]
                ret["inspect_probs"] = allprobs
                ret["inspect_mask"] = allmask
        return ret, pred_trees

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        taggers = [self.tagger] if self.ensemble <= 0 else self.tagger
        outdict = {"loss": 0, "acc": 0, "elemacc": 0}
        outlogits = []
        # extract a training example from y:
        x, newy, tgt = self.extract_training_example(x, y)
        for tagger in taggers:
            enc, encmask = tagger.encode_source(x)
            # run through tagger: the same for all versions
            logits, cache = tagger(tokens=newy, enc=enc, encmask=encmask, cache=None)
            # compute loss: different versions do different masking and different targets
            loss, acc, elemacc = self.compute_loss(logits, tgt)
            outdict["loss"] += loss
            outdict["acc"] += acc
            outdict["elemacc"] += elemacc
            outlogits.append(logits)
        outdict["acc"] /= len(taggers)
        outdict["elemacc"] /= len(taggers)
        return outdict, outlogits

    def extract_training_example(self, x, y):
        ymask = (y != 0).float()
        ylens = ymask.sum(1).long()
        newy = y
        newy = torch.cat([torch.ones_like(newy[:, 0:1]) * self.vocab["@START@"], newy], 1)
        newy = torch.cat([newy, torch.zeros_like(newy[:, 0:1])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            newy[i, ylen+1] = self.vocab["@END@"]

        goldy = newy[:, 1:]
        # tgt = torch.zeros(goldy.size(0), goldy.size(1), self.vocab.number_of_ids(), device=goldy.device)
        # tgt = tgt.scatter(2, goldy[:, :, None], 1.)
        # tgtmask = (goldy != 0).float()

        newy = newy[:, :-1]
        return x, newy, goldy

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_size
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@START@"]
        # yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        if self.ensemble > 0:
            if self.innerensemble:
                taggers = self.tagger   # if inner ensemble, average at every time step
            else:
                taggers = [self.tagger[0]]  # if not inner, use only one for prediction
        elif self.mcdropout > 0:
            if self.innerensemble:
                taggers = [self.tagger for _ in range(self.mcdropout)]
                wastraining = self.tagger.training
                self.tagger.train()
            else:
                taggers = [self.tagger]
        else:
            taggers = [self.tagger]

        # run encoder
        encs, encmasks = zip(*[tagger.encode_source(x) for tagger in taggers])

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        caches = [None for _ in taggers]
        conf_acc = None
        maxprob_acc = None
        maxmaxnll = None
        total = None
        entropy = None
        allprobs = []
        allmask = []
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run tagger
            # y = torch.cat([y, yend], 1)
            logitses, caches = zip(*[tagger(tokens=y, enc=encs[i], encmask=encmasks[i], cache=caches[i]) for i, tagger in enumerate(taggers)])
            probses = [torch.softmax(logits[:, -1], -1) for logits in logitses]
            probs = sum(probses) / len(probses)     # average over all ensemble elements

            allprobs.append(probs)
            maxprobs, preds = probs.max(-1)
            _entropy = (-torch.log(probs.clamp_min(1e-7)) * probs).sum(-1)
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            allmask.append((~ended).long())
            total = total if total is not None else torch.zeros_like(maxprobs)
            total = total + torch.ones_like(maxprobs) * (~ended).float()
            conf_acc = conf_acc if conf_acc is not None else torch.ones_like(maxprobs)
            # conf_acc = conf_acc + maxprobs * (~ended).float()
            conf_acc = conf_acc * torch.where(ended, torch.ones_like(maxprobs), maxprobs)
            maxprob_acc = maxprob_acc if maxprob_acc is not None else torch.zeros_like(maxprobs)
            maxprob_acc = maxprob_acc + -torch.log(maxprobs) * (~ended).float()
            maxmaxnll = maxmaxnll if maxmaxnll is not None else torch.zeros_like(maxprobs)
            maxmaxnll = torch.max(maxmaxnll, torch.where(ended, torch.zeros_like(maxprobs), -torch.log(maxprobs)))
            entropy = entropy if entropy is not None else torch.zeros_like(_entropy)
            entropy = entropy + _entropy * (~ended).float()
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        allprobs = torch.stack(allprobs, 1)
        allmask = torch.stack(allmask, 1)

        if self.mcdropout > 0 and self.innerensemble and not wastraining:
            self.tagger.eval()

        return newy, maxprob_acc/total, maxmaxnll, entropy/total, total, conf_acc, maxprob_acc, steps_used.float(), allprobs, allmask


# def autocollate(x:Dict, pad_value=0):
#     y = {}
#     for k in x[0]:
#         yk = []
#         for xe in x:
#             yk.append(xe[k])
#         y[k] = yk
#         if isinstance(yk[0], torch.LongTensor) and yk[0].dim() == 1:
#             y[k] = q.pad_tensors(yk, 0, pad_value)
#     for k, yk in y.items():
#         if isinstance(yk[0], torch.Tensor):
#             yk = [yij[None] for yij in yk]
#             y[k] = torch.cat(yk, 0)
#     return y


class Tokenizer(object):
    def __init__(self, bertname="bert-base-uncased", inpvocab:Vocab=None, outvocab:Vocab=None, **kw):
        super(Tokenizer, self).__init__(**kw)
        self.inpvocab = inpvocab
        self.tokenizer = None if bertname == "vanilla" else AutoTokenizer.from_pretrained(bertname)
        self.outvocab = outvocab

    def tokenize(self, inps, outs):
        if self.tokenizer is not None:
            inptoks = self.tokenizer.tokenize(inps)
        else:
            inptoks = ["@START@"] + self.get_toks(inps) + ["@END@"]
        outtoks = self.get_out_toks(outs)
        if self.tokenizer is not None:
            inptensor = self.tokenizer.encode(inps, return_tensors="pt")[0]
        else:
            inptensor = self.tensorize_output(inptoks, self.inpvocab)
        ret = {"inps": inps, "outs":outs, "inptoks": inptoks, "outtoks": outtoks,
               "inptensor": inptensor, "outtensor": self.tensorize_output(outtoks, self.outvocab)}
        ret = (ret["inptensor"], ret["outtensor"])
        return ret

    def get_toks(self, x):
        return x.strip().split(" ")

    def get_out_toks(self, x):
        return self.get_toks(x)

    def tensorize_output(self, x, vocab):
        ret = [vocab[xe] for xe in x]
        ret = torch.tensor(ret)
        return ret


ORDERLESS = {"@WHERE", "@OR", "@AND", "@QUERY", "(@WHERE", "(@OR", "(@AND", "(@QUERY"}


def load_ds(dataset="scan/random", validfrac=0.1, recompute=False,
            bertname="bert-base-uncased", jumpdiffpos=False):
    tt = q.ticktock("data")
    tt.tick(f"loading '{dataset}'")
    if bertname.startswith("none"):
        bertname = "bert" + bertname[4:]
    if dataset.startswith("cfq/") or dataset.startswith("scan/mcd"):
        key = f"{dataset}|bertname={bertname}"
        print(f"validfrac is ineffective with dataset '{dataset}'")
    else:
        key = f"{dataset}|validfrac={validfrac}|bertname={bertname}"

    shelfname = os.path.basename(__file__) + ".cache.shelve"
    if not recompute:
        tt.tick(f"loading from shelf (key '{key}')")
        with shelve.open(shelfname) as shelf:
            if key not in shelf:
                recompute = True
                tt.tock("couldn't load from shelf")
            else:
                shelved = shelf[key]
                trainex, validex, testex, fldic = shelved["trainex"], shelved["validex"], shelved["testex"], shelved["fldic"]
                inpdic = shelved["inpdic"] if "inpdic" in shelved else None
                trainds, validds, testds = Dataset(trainex), Dataset(validex), Dataset(testex)
                tt.tock("loaded from shelf")

    if recompute:
        tt.tick("loading data")
        splits = dataset.split("/")
        dataset, splits = splits[0], splits[1:]
        split = "/".join(splits)
        if dataset == "scan":
            ds = SCANDatasetLoader().load(split, validfrac=validfrac)
        elif dataset == "cfq":
            ds = CFQDatasetLoader().load(split + "/modent")
        else:
            raise Exception(f"Unknown dataset: '{dataset}'")
        tt.tock("loaded data")

        tt.tick("creating tokenizer")
        tokenizer = Tokenizer(bertname=bertname)
        tt.tock("created tokenizer")

        print(len(ds))

        tt.tick("dictionaries")
        inpdic = Vocab()
        inplens, outlens = [0], []
        fldic = Vocab()
        for x in ds:
            outtoks = tokenizer.get_out_toks(x[1])
            outlens.append(len(outtoks))
            for tok in outtoks:
                fldic.add_token(tok, seen=x[2] == "train")
            inptoks = tokenizer.get_toks(x[0])
            for tok in inptoks:
                inpdic.add_token(tok, seen=x[2] == "train")
        inpdic.finalize(min_freq=0, top_k=np.infty)
        fldic.finalize(min_freq=0, top_k=np.infty)
        print(
            f"input avg/max length is {np.mean(inplens):.1f}/{max(inplens)}, output avg/max length is {np.mean(outlens):.1f}/{max(outlens)}")
        print(f"output vocabulary size: {len(fldic.D)} at output, {len(inpdic.D)} at input")
        tt.tock()

        tt.tick("tensorizing")
        tokenizer.inpvocab = inpdic
        tokenizer.outvocab = fldic
        trainds = ds.filter(lambda x: x[-1] == "train").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        validds = ds.filter(lambda x: x[-1] == "valid").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        testds = ds.filter(lambda x: x[-1] == "test").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        # ds = ds.map(lambda x: tokenizer.tokenize(x[0], x[1]) + (x[2],)).cache(True)
        tt.tock("tensorized")

        tt.tick("shelving")
        with shelve.open(shelfname) as shelf:
            shelved = {
                "trainex": trainds.examples,
                "validex": validds.examples,
                "testex": testds.examples,
                "fldic": fldic,
                "inpdic": inpdic,
            }
            shelf[key] = shelved
        tt.tock("shelved")

    if dataset == "scan/add_jump" and jumpdiffpos:
        # change primitive jump examples to examples where jump occurs on different positions
        maxlen = 0
        newexamples = []
        jumpinp = torch.tensor([inpdic["@START@"], inpdic["jump"], inpdic["@END@"]])
        jumpout = torch.tensor([fldic["(@R@"], fldic["I_JUMP"], fldic[")"]])
        for inp, out in trainds.examples:
            maxlen = max(maxlen, len(inp), len(out))
        for inp, out in trainds.examples:
            if inp.shape == jumpinp.shape and out.shape == jumpout.shape \
                    and torch.all(inp == jumpinp) and torch.all(out == jumpout):
                # this is a primitive jump example
                numbermasks = random.sample(range(0, maxlen-3), 1)[0]
                newinp = [inpdic["@START@"]] + [inpdic["@PAD@"]] * numbermasks + [inpdic["jump"]] + [inpdic["@END@"]]
                newout = [fldic["(@R@"]] + [fldic["@PAD@"]] * numbermasks + [fldic["I_JUMP"]] + [fldic[")"]]
                newexamples.append((torch.tensor(newinp), torch.tensor(newout)))
            else:
                newexamples.append((inp, out))
        trainds._examples = newexamples

    tt.tock(f"loaded '{dataset}'")
    tt.msg(f"#train={len(trainds)}, #valid={len(validds)}, #test={len(testds)}")

    tt.msg("Overlap of validation with train:")
    overlaps = compute_overlaps(trainds, validds)
    print(json.dumps(overlaps, indent=4))

    tt.msg("Overlap of test with train:")
    overlaps = compute_overlaps(trainds, testds)
    print(json.dumps(overlaps, indent=4))

    return trainds, validds, testds, fldic, inpdic


def compute_overlaps(train, test):
    inp_overlap = []
    out_overlap = []
    both_overlap = []
    traininps, trainouts, trainboths = set(), set(), set()
    for i in tqdm(range(len(train))):
        ex = train[i]
        inpstr = list(ex[0].cpu().numpy())
        inpstr = " ".join([str(exe) for exe in inpstr])
        outstr = list(ex[1].cpu().numpy())
        outstr = " ".join([str(exe) for exe in outstr])
        traininps.add(inpstr)
        trainouts.add(outstr)
        trainboths.add(inpstr+"|"+outstr)

    for i in tqdm(range(len(test))):
        ex = test[i]
        inpstr = list(ex[0].cpu().numpy())
        inpstr = " ".join([str(exe) for exe in inpstr])
        outstr = list(ex[1].cpu().numpy())
        outstr = " ".join([str(exe) for exe in outstr])
        if inpstr in traininps:
            inp_overlap.append(inpstr)
        if outstr in trainouts:
            out_overlap.append(outstr)
        if inpstr + "|" + outstr in trainboths:
            both_overlap.append(inpstr + "|" + outstr)

    ret = {"inps": len(inp_overlap)/len(test),
           "outs": len(out_overlap)/len(test),
           "both": len(both_overlap)/len(test),}
    return ret


def collate_fn(x, pad_value=0, numtokens=5000):
    lens = [len(xe[1]) for xe in x]
    a = list(zip(lens, x))
    a = sorted(a, key=lambda xe: xe[0], reverse=True)
    maxnum = int(numtokens/max(lens))
    b = a[:maxnum]
    b = [be[1] for be in b]
    ret = autocollate(b, pad_value=pad_value)
    return ret


def run(lr=0.0001,
        enclrmul=0.01,
        smoothing=0.,
        gradnorm=3,
        batsize=60,
        epochs=16,
        patience=10,
        validinter=3,
        validfrac=0.1,
        warmup=3,
        cosinelr=False,
        dataset="scan/length",
        mode="normal",          # "normal", "noinp"
        maxsize=50,
        seed=42,
        hdim=768,
        numlayers=6,
        numheads=12,
        dropout=0.1,
        worddropout=0.,
        bertname="bert-base-uncased",
        testcode=False,
        usesinpos=False,
        userelpos=False,
        jumpdiffpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        mcdropout=-1,
        ensemble=-1,
        innerensemble=False,
        version="v5"
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    projectname = f"compood_baseline_v3"
    wandb.init(project=projectname, config=settings, reinit=True)
    runname = wandb.run.name

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    if maxsize < 0:
        if dataset.startswith("cfq"):
            maxsize = 155
        elif dataset.startswith("scan"):
            maxsize = 50
        print(f"maxsize: {maxsize}")

    tt = q.ticktock("script")
    tt.msg(f"Run name: {runname}")
    tt.tick("data")
    trainds, validds, testds, fldic, inpdic = load_ds(dataset=dataset, validfrac=validfrac, bertname=bertname, recompute=recomputedata, jumpdiffpos=jumpdiffpos)


    if "mcd" in dataset.split("/")[1]:
        print(f"Setting patience to -1 because MCD (was {patience})")
        print(f"This means there is no early stopping.")
        patience = -1

    # if smalltrainvalid:
    if True: #"mcd" in dataset.split("/")[1]:
        realtrainds = []
        indtestds = []
        splits = [True for _ in range(int(round(len(trainds) * 0.1)))]
        splits = splits + [False for _ in range(len(trainds)-len(splits))]
        random.shuffle(splits)
        for i in range(len(trainds)):
            if splits[i] is True:
                indtestds.append(trainds[i])
            else:
                realtrainds.append(trainds[i])
        trainds = Dataset(realtrainds)
        indtestds = Dataset(indtestds)
        tt.msg("split off 10% of training data for in-distribution test set")
    # else:
    #     indtestds = Dataset([x for x in validds.examples])
    #     tt.msg("using validation set as in-distribution test set")
    tt.msg(f"in-distribution: {len(indtestds)}, OOD: {len(testds)}")
    if trainonvalid:
        trainds = trainds + validds
        validds = testds

    tt.tick("dataloaders")
    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    validdl = DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    testdl = DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    indtestdl = DataLoader(indtestds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    # print(json.dumps(next(iter(trainds)), indent=3))
    # print(next(iter(traindl)))
    # print(next(iter(validdl)))
    tt.tock()
    tt.tock()

    tt.tick("model")
    if ensemble <= 0:
        cell = TransformerDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numheads=numheads,
                                      dropout=dropout, worddropout=worddropout, mode=mode,
                                      bertname=bertname, userelpos=userelpos, usesinpos=usesinpos, useabspos=not userelpos)
        print(f"one layer of decoder: \n {cell.decoder.block[0]}")
    else:
        cell = torch.nn.ModuleList([TransformerDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numheads=numheads,
                                      dropout=dropout, worddropout=worddropout, mode=mode,
                                      bertname=bertname, userelpos=userelpos, usesinpos=usesinpos, useabspos=not userelpos)
                for _ in range(ensemble)])
        print(f"one layer of decoder: \n {cell[0].decoder.block[0]}")
    decoder = SeqDecoderBaseline(cell, vocab=fldic, max_size=maxsize, smoothing=smoothing, mode=mode, mcdropout=mcdropout, innerensemble=innerensemble)
    tt.tock()

    if testcode:
        tt.tick("testcode")
        batch = next(iter(traindl))
        # out = tagger(batch[1])
        tt.tick("train")
        out = decoder(*batch)
        tt.tock()
        decoder.train(False)
        tt.tick("test")
        out = decoder(*batch)
        tt.tock()
        tt.tock("testcode")

    tloss = make_array_of_metrics("loss", "elemacc", "acc", reduction="mean")
    metricnames = ["treeacc", "decnll", "maxmaxnll", "entropy"]
    tmetrics = make_array_of_metrics(*metricnames, reduction="mean")
    vmetrics = make_array_of_metrics(*metricnames, reduction="mean")
    indxmetrics = make_array_of_metrics(*metricnames, reduction="mean")
    oodxmetrics = make_array_of_metrics(*metricnames, reduction="mean")

    # region parameters
    def get_parameters(m, _lr, _enclrmul):
        bertparams = []
        otherparams = []
        for k, v in m.named_parameters():
            if "encoder_model." in k:
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

    eyt = q.EarlyStopper(vmetrics[0], patience=patience, min_epochs=30, more_is_better=True,
                         remember_f=lambda: deepcopy(cell))

    def wandb_logger():
        d = {}
        for name, loss in zip(["loss", "acc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        if evaltrain:
            for name, loss in zip(metricnames, tmetrics):
                d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(metricnames, vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        for name, loss in zip(metricnames, indxmetrics):
            d["indtest_"+name] = loss.get_epoch_error()
        for name, loss in zip(metricnames, oodxmetrics):
            d["oodtest_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim = get_optim(cell, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        assert t_max > (warmup + 10)
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(low=0., high=1.0, steps=t_max-warmup) >> (0. * lr)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, on_before_optim_step=[lambda : clipgradnorm(_m=cell, _norm=gradnorm)])

    if trainonvalidonly:
        traindl = validdl
        validdl = testdl

    trainepoch = partial(q.train_epoch, model=decoder,
                         dataloader=traindl,
                         optim=optim,
                         losses=tloss,
                         device=device,
                         _train_batch=trainbatch,
                         on_end=[lambda: lr_schedule.step()])

    trainevalepoch = partial(q.test_epoch,
                             model=decoder,
                             losses=tmetrics,
                             dataloader=traindl,
                             device=device)

    on_end_v = [lambda: eyt.on_epoch_end(), lambda: wandb_logger()]
    validepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=vmetrics,
                         dataloader=validdl,
                         device=device,
                         on_end=on_end_v)
    indtestepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=indxmetrics,
                         dataloader=indtestdl,
                         device=device)
    oodtestepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=oodxmetrics,
                         dataloader=testdl,
                         device=device)

    tt.tick("training")
    if evaltrain:
        validfs = [trainevalepoch, validepoch]
    else:
        validfs = [validepoch]
    # validfs = validfs + [indtestepoch, oodtestepoch]

    # results = evaluate(decoder, indtestds, testds, batsize=batsize, device=device)
    # print(json.dumps(results, indent=4))

    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=validfs,
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    tt.tick("running test before reloading")
    testres = oodtestepoch()
    print(f"Test tree acc: {testres}")
    tt.tock("ran test")

    if eyt.remembered is not None and patience >= 0:
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

    tt.tick("running ID test")
    testres = indtestepoch()
    print(f"ID test tree acc: {testres}")
    tt.tock()

    tt.tick("running OOD test")
    testres = oodtestepoch()
    print(f"OOD test tree acc: {testres}")
    tt.tock()

    results = evaluate(decoder, indtestds, testds, batsize=batsize, device=device,
                       savep=f"{projectname}.{version}.{runname}.outputs.json",
                       inpdic=inpdic, fldic=fldic)
    print(json.dumps(results, indent=4))

    settings.update({"final_train_loss": tloss[0].get_epoch_error()})
    settings.update({"final_train_tree_acc": tmetrics[0].get_epoch_error()})
    settings.update({"final_valid_tree_acc": vmetrics[0].get_epoch_error()})
    settings.update({"final_indtest_tree_acc": indxmetrics[0].get_epoch_error()})
    settings.update({"final_oodtest_tree_acc": oodxmetrics[0].get_epoch_error()})
    for k, v in results.items():
        for metric, ve in v.items():
            settings.update({f"{k}_{metric}": ve})

    wandb.config.update(settings)
    q.pp_dict(settings)

    return decoder, indtestds, testds


def cat_dicts(x:List[Dict]):
    out = {}
    for k, v in x[0].items():
        out[k] = []
    for xe in x:
        for k, v in xe.items():
            out[k].append(v)
    for k, v in out.items():
        if not k.startswith("inspect_"):
            out[k] = torch.cat(v, 0)
    return out


def save_outputs(idouts, oodouts, inpdic, fldic):
    """
    :param idouts:       contains output dictionary for in-distribution data
    :param oodouts:      contains output dictionary for out-of-distribution data
    :return:    {"id": {...}, "ood": {...}} of transformed outputs
    """
    ret = {"id": save_outputs_one(idouts, inpdic, fldic),
           "ood": save_outputs_one(oodouts, inpdic, fldic)}
    return ret


def save_outputs_one(outs, inpdic, fldic):
    """
    :param outs:    dictionary containing "inspect_x", "inspect_gold", "inspect_pred", "inspect_probs", "inspect_mask"
    :return:        list of dicts with keys     "input":List[str],
                                                "gold":List[str],
                                                "pred":List[str],
                                                "entropies": List[float]
                                                "bestprobs": List[float] ]
    """
    ret = []
    for batch_id in range(len(outs["inspect_x"])):
        for i in range(len(outs["inspect_x"][batch_id])):
            inp_tokens = inpdic.tostr(outs["inspect_x"][batch_id][i]).split()
            gold_tokens = fldic.tostr(outs["inspect_gold"][batch_id][i]).split()
            pred_tokens = fldic.tostr(outs["inspect_pred"][batch_id][i]).split()
            mask = outs["inspect_mask"][batch_id][i]
            numtokens = mask.sum().item()
            # assert numtokens == len(pred_tokens)
            probs = outs["inspect_probs"][batch_id][i]
            entropies = (-torch.log(probs.clamp_min(1e-7)) * probs).sum(-1).detach().cpu().numpy()[:numtokens]
            bestprobs = probs.max(-1)[0].detach().cpu().numpy()[:numtokens]
            ret.append({
                "input": inp_tokens,
                "gold": gold_tokens,
                "pred": pred_tokens,
                "entropies": [float(n) for n in entropies],
                "bestprobs": [float(n) for n in bestprobs],
            })
    return ret


def evaluate(model, idds, oodds, batsize=10, device=torch.device("cpu"),
             savep=None, inpdic=None, fldic=None):
    """
    :param model:       Decoder model
    :param idds:     dataset with in-distribution examples
    :param oodds:      dataset with out-of-distribution examples
    :return:
    """
    iddl = DataLoader(idds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    ooddl = DataLoader(oodds, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    model.inspect = True

    _, _idouts = q.eval_loop(model, iddl, device=device)
    idouts = cat_dicts(_idouts[0])
    _, _oodouts = q.eval_loop(model, ooddl, device=device)
    oodouts = cat_dicts(_oodouts[0])

    # decnll_res = compute_auc_and_fprs(idouts["decnll"], oodouts["decnll"], "decnll")
    # sumnll_res = compute_auc_and_fprs(idouts["sumnll"], oodouts["sumnll"], "sumnll")
    # maxnll_res = compute_auc_and_fprs(idouts["maxmaxnll"], oodouts["maxmaxnll"], "maxmaxnll")
    # entropy_res = compute_auc_and_fprs(idouts["entropy"], oodouts["entropy"], "entropy")
    decnll_res = compute_auc_and_fprs(-oodouts["decnll"], -idouts["decnll"], "decnll")
    sumnll_res = compute_auc_and_fprs(-oodouts["sumnll"], -idouts["sumnll"], "sumnll")
    maxnll_res = compute_auc_and_fprs(-oodouts["maxmaxnll"], -idouts["maxmaxnll"], "maxmaxnll")
    entropy_res = compute_auc_and_fprs(-oodouts["entropy"], -idouts["entropy"], "entropy")

    # compute histogram of confidence vs accuracy --> 10 confidence bins, compute accuracy for each
    df = np.stack([oodouts["avgconf"].detach().cpu().numpy(), oodouts["treeacc"].detach().cpu().numpy()], 1)
    Nbins = 25
    rdf = np.zeros((Nbins, 2))
    np.nan_to_num(df, False)
    for i in range(df.shape[0]):
        bin = min(int(math.floor(df[i, 0]*Nbins)), Nbins-1)
        rdf[bin, 0] += 1
        rdf[bin, 1] += df[i, 1]
    rdf[:, 1] = rdf[:, 1] / rdf[:, 0]
    np.nan_to_num(rdf, False)
    print("Calibration table:")
    print(rdf)

    wandb.log({"calibtable": wandb.Table(data=rdf, columns=["count", "treeacc"])})

    if savep is not None and inpdic is not None and fldic is not None:
        print("Saving outputs")
        analysis = save_outputs(idouts, oodouts, inpdic, fldic)
        with open(savep, "w") as f:
            json.dump(analysis, f, indent=3)
            print(f"Saved outputs in {savep}")

    return {"decnll": decnll_res, "maxnll": maxnll_res, "entropy": entropy_res, "sumnll": sumnll_res}


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def compute_auc_and_fprs(posscores, negscores, kind=""):
    try:
        posscores = -posscores.detach().cpu().numpy()
        negscores = -negscores.detach().cpu().numpy()
        poslabels = np.ones_like(posscores)
        neglabels = np.zeros_like(negscores)

        labels = np.concatenate([poslabels, neglabels], 0)
        scores = np.concatenate([posscores, negscores], 0)
        df = np.stack([labels, scores], 1)
        df = pd.DataFrame(df, columns=["Labels", "Scores"])

        wandb.log({f"hist_{kind}": px.histogram(df, x="Scores", color="Labels", color_discrete_sequence=px.colors.qualitative.Plotly, nbins=50)})

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)

        roc_df = pd.DataFrame(np.stack([fpr, tpr], 1), columns=["FPR", "TPR"])
        wandb.log({f"aucroc_{kind}": px.line(roc_df, x="FPR", y="TPR", color_discrete_sequence=px.colors.qualitative.Plotly)})

        aucroc = sklearn.metrics.auc(fpr, tpr)

        prec, rec, pr_thresholds = sklearn.metrics.precision_recall_curve(labels, scores)
        aucpr = sklearn.metrics.auc(rec, prec)

        ret = {"aucroc": aucroc, "aucpr": aucpr}
        # compute FPR-K's
        fpr, tpr = list(fpr), list(tpr)
        ks = [80, 90, 95, 99]
        for k in ks:
            ret[f"fpr{k}"] = fpr[-1]
        for k in ks:
            for i in range(len(tpr)):
                if tpr[i] >= k/100:
                    ret[f"fpr{k}"] = fpr[i]
                    break

        # posscores = list(posscores)
        # negscores = list(negscores)
        #
        # posscores = sorted(posscores)
        # for k in ks:
        #     ret[f"pred{k}"] = 1.
        #     th = posscores[int(round((1-k/100)*len(posscores)))]
        #     for i in range(len(thresholds)):
        #         if thresholds[i] <= th:
        #             ret[f"pred{k}"] = fpr[i]
        #             break
        # for k in ks:
        #     poss

    except Exception as e:
        ret = {"aucroc": 0., "aucpr": 0.}
        ks = [80, 90, 95, 99]
        for k in ks:
            ret[f"fpr{k}"] = 1.
    return ret


def run_experiment(
        lr=-1.,
        enclrmul=-1.,
        smoothing=-1.,
        gradnorm=2,
        batsize=-1,
        epochs=-1,      # probably 11 is enough
        patience=100,
        validinter=-1,
        warmup=3,
        cosinelr=False,
        dataset="default",
        datasets="both",
        mode="normal",
        maxsize=-1,
        seed=-1,
        hdim=-1,
        numlayers=-1,
        numheads=-1,
        dropout=-1.,
        worddropout=-1.,
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        usesinpos=False,
        jumpdiffpos=False,
        trainonvalidonly=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        mcdropout=-1,
        ensemble=-1,
        innerensemble=False,
        ):

    settings = locals().copy()
    del settings["datasets"]

    ranges = {
        # "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"],
        # "dataset": ["cfq/mcd1", "cfq/mcd2", "cfq/mcd3"],
        "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2",
                    "scan/mcd3", "cfq/mcd1", "cfq/mcd2", "cfq/mcd3"],
        # "dataset": ["scan/mcd1", "scan/mcd2", "scan/mcd3"],
        "dropout": [0.1, 0.25, 0.5],
        "worddropout": [0.],
        "seed": [42, 87646464, 456852],
        "epochs": [20, 25],
        "batsize": [50],
        "hdim": [768],
        "numheads": [12],
        "numlayers": [6],
        "lr": [0.0001],
        "enclrmul": [0.01],                  # use 1.
        "smoothing": [0.],
        "validinter": [1],
        "mcdropout": [0, 5],
    }

    if datasets == "both":
        pass
    elif datasets == "cfq":
        ranges["dataset"] = ["cfq/mcd1", "cfq/mcd2", "cfq/mcd3"]
    elif datasets == "scan":
        ranges["dataset"] = ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"]
    elif datasets == "mcd":
        ranges["dataset"] = ["cfq/mcd1", "cfq/mcd2", "cfq/mcd3", "scan/mcd1", "scan/mcd2", "scan/mcd3"]
    elif datasets == "nonmcd":
        ranges["dataset"] = ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left"]

    if bertname.startswith("none") or bertname == "vanilla":
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["hdim"] = [384]
        ranges["numheads"] = [6]

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
        if spec["dataset"].startswith("cfq"):
            if spec["epochs"] not in (20, 1, 0, 30, 15, 10):
                return False
        elif spec["dataset"].startswith("scan"):
            if spec["epochs"] not in (25, 1, 0, 30, 15, 10):
                return False
        return True

    print(__file__)
    p = __file__ + f".baseline.{dataset}"
    q.run_experiments_random(
        run, ranges, path_prefix=None, check_config=checkconfig, **settings)


# python compood.py -epochs 25 -datasets scan -gpu 0 -dropout 0.25 -ensemble 5 -mcdropout 0 -innerensemble -batsize 60


if __name__ == '__main__':
    q.argprun(run_experiment)