import json
import math
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from typing import Dict, List

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
from parseq.rnn1 import Encoder
from parseq.scripts_compgen_new.compood import evaluate, TransformerEmbeddings
from parseq.scripts_compgen_new.transformer import TransformerConfig, TransformerStack
from parseq.scripts_compgen_new.transformerdecoder import TransformerStack as TransformerStackDecoder
from parseq.vocab import Vocab

from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd


class Embeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_size, dropout=0., pad_token_id=0,
                 layer_norm_eps=1e-12):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        ret = inputs_embeds

        ret = self.LayerNorm(ret)
        ret = self.dropout(ret)
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


class DecoderCell(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=2, numtmlayers=6,
                 mode="normal",
                 dropout:float=0., worddropout:float=0., numheads=6,
                 noencoder=False, **kw):
        super(DecoderCell, self).__init__(**kw)
        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.mode = mode
        self.noencoder = noencoder
        self.numlayers = numlayers
        self.numtmlayers = numtmlayers

        self.dec_emb = torch.nn.Embedding(self.vocabsize+3, self.dim)
        dims = [self.dim + self.dim] + [self.dim for _ in range(numlayers)]
        self.dec_stack = torch.nn.ModuleList([torch.nn.GRUCell(dims[i], dims[i+1]) for i in range(numlayers)])
        self.dropout = torch.nn.Dropout(dropout)
        self.attn_linQ = None
        self.attn_linK = None
        self.attn_linV = None
        # self.attn_linQ = torch.nn.Linear(self.dim, self.dim)
        # self.attn_linK = torch.nn.Linear(self.dim, self.dim)
        # self.attn_linV = torch.nn.Linear(self.dim, self.dim)

        self.preout = torch.nn.Linear(self.dim + self.dim, self.dim)
        self.preoutnonlin = torch.nn.CELU()
        if self.mode == "cont":
            pass
        else:
            self.out = torch.nn.Linear(self.dim, self.vocabsize+3)

        inpvocabsize = inpvocab.number_of_ids()
        if not self.noencoder:
            encconfig = TransformerConfig(vocab_size=inpvocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                          d_kv=int(self.dim/numheads),
                                          num_layers=self.numtmlayers, num_heads=numheads, dropout_rate=dropout)
            encemb = TransformerEmbeddings(encconfig.vocab_size, encconfig.d_model, dropout=dropout, max_position_embeddings=1000, useabspos=True)
            self.encoder_model = TransformerStack(encconfig, encemb)
            # self.encoder_model = Encoder(inpvocabsize+5, self.dim, int(self.dim/2), num_layers=numlayers, dropout=dropout)

        self.adapter = None
        self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken],
                                          [self.inpvocab[self.inpvocab.padtoken]])
        self.worddropout = WordDropout(worddropout, self.vocab[self.vocab.masktoken], [self.vocab[self.vocab.padtoken]])

        self.lenlin = torch.nn.Linear(self.dim * 2, self.dim)
        self.lennonlin = torch.nn.CELU()
        self.lenbias = torch.nn.Linear(self.dim, 1)
        self.lenscale = torch.nn.Linear(self.dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.lenbias.weight)
        torch.nn.init.zeros_(self.lenscale.weight)
        torch.nn.init.zeros_(self.lenbias.bias)
        torch.nn.init.zeros_(self.lenscale.bias)

    def encode_source(self, x, xmask=None):
        if self.noencoder:
            assert xmask is not None
            encs = x
        else:
            xmask = (x != 0) if xmask is None else xmask
            x = self.inpworddropout(x)
            encs = self.encoder_model(x, attention_mask=xmask)[0]
        return encs, xmask

    def predict_length(self, encs, encmask):        # (batsize, seqlen, dim), (batsize, seqlen)
        maxpool, _ = torch.max(encs + (1 - encmask[:, :, None].float()) * -1e5, 1)
        meanpool = torch.sum(encs * encmask[:, :, None].float(), 1) / torch.sum(encmask[:, :, None].float(), 1).clamp_min(1e-6)
        pool = torch.cat([maxpool, meanpool], -1)
        h = self.lenlin(pool)
        h = self.lennonlin(h)
        lenbias = self.lenbias(h)[:, 0]
        lenscale = self.lenscale(h)[:, 0]
        lenscale = (torch.celu(lenscale)+1).clamp_min(0)
        original_lengths = encmask.sum(-1)
        lens = original_lengths * lenscale + lenbias
        return lens

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None, padmask=None):
        padmask = (tokens != 0) if padmask is None else padmask

        if self.mode == "cont":
            embs = tokens
        else:
            embs = self.dec_emb(tokens)
        if cache is None:
            cache = {"states": [{"h_tm1": torch.zeros(enc.size(0), self.dim, device=enc.device)} for _ in self.dec_stack],
                     "prevatt": torch.zeros_like(enc[:, 0])}

        prev_att = cache["prevatt"]
        inps = torch.cat([embs, prev_att], -1)
        for l, layer in enumerate(self.dec_stack):
            prev_state = cache["states"][l]["h_tm1"]
            inps = self.dropout(inps)
            h_t = layer(inps, prev_state)
            cache["states"][l]["h_tm1"] = h_t
            inps = h_t

        if self.attn_linQ is not None:
            h_t = self.attn_linQ(h_t)
        if self.attn_linK is not None:
            encK = self.attn_linK(enc)
        else:
            encK = enc
        if self.attn_linV is not None:
            encV = self.attn_linV(enc)
        else:
            encV = enc

        # attention
        weights = torch.einsum("bd,bsd->bs", h_t, encK)
        weights = weights.masked_fill(encmask == 0, float('-inf'))
        alphas = torch.softmax(weights, -1)
        summary = torch.einsum("bs,bsd->bd", alphas, encV)
        cache["prevatt"] = summary

        out = torch.cat([h_t, summary], -1)
        out = self.preout(out)
        out = self.preoutnonlin(out)

        if self.mode == "cont":
            logits = out
        else:
            logits = self.out(out)

        return logits, cache


class S2S(torch.nn.Module):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def __init__(self, tagger:DecoderCell,
                 vocab=None,
                 max_size:int=100,
                 smoothing:float=0.,
                 **kw):
        super(S2S, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_size = max_size
        self.smoothing = smoothing
        if self.smoothing > 0:
            self.loss = q.SmoothedCELoss(reduction="none", ignore_index=0, smoothing=smoothing, mode="logprobs")
        else:
            self.loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)

        self.logsm = torch.nn.LogSoftmax(-1)

    def forward(self, x, y, xmask=None):
        if self.training:
            return self.train_forward(x, y, xmask=xmask)
        else:
            return self.test_forward(x, y, xmask=xmask)

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

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None, xmask=None):   # --> implement how decoder operates end-to-end
        preds, stepsused = self.get_prediction(x, xmask=xmask)

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
        return ret, pred_trees

    def train_forward(self, x:torch.Tensor, y:torch.Tensor, xmask=None):  # --> implement one step training of tagger
        # extract a training example from y:
        newy, tgt = self.extract_training_example(y)
        enc, encmask = self.tagger.encode_source(x, xmask=xmask)
        # run through tagger: the same for all versions
        logits = self.get_prediction_train(newy, enc, encmask)
        # compute loss: different versions do different masking and different targets
        loss, acc, elemacc = self.compute_loss(logits, tgt)
        return {"loss": loss, "ce": loss, "acc": acc, "elemacc": elemacc}, logits

    def get_prediction_train(self, tokens: torch.Tensor, enc: torch.Tensor, encmask=None):
        cache = None
        logitses = []
        for i in range(tokens.size(1)):
            logits, cache = self.tagger(tokens=tokens[:, i], enc=enc, encmask=encmask, cache=cache)
            logitses.append(logits)
        logitses = torch.stack(logitses, 1)
        return logitses

    def extract_training_example(self, y):
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
        return newy, goldy

    def get_prediction(self, x:torch.Tensor, xmask=None):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_size
        # initialize empty ys:
        y = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.vocab["@START@"]
        # yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.tagger.encode_source(x, xmask=xmask)

        step = 0
        # newy = torch.zeros(x.size(0), 0, dtype=torch.long, device=x.device)
        newy = y[:, None]       # will be removed
        ended = torch.zeros_like(y).bool()
        cache = None
        while step < self.max_size and not torch.all(ended):
            logits, cache = self.tagger(tokens=y, enc=enc, encmask=encmask, cache=cache)
            probs = torch.softmax(logits, -1)
            maxprobs, preds = probs.max(-1)
            _entropy = (-torch.log(probs.clamp_min(1e-7)) * probs).sum(-1)
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))

            preds = torch.where(ended, torch.zeros_like(preds), preds)
            newy = torch.cat([newy, preds[:, None]], 1)

            y = preds
        return newy, steps_used.float()


class S2Z(torch.nn.Module):
    def __init__(self, tagger: DecoderCell,
                 vocab=None,
                 max_size: int = 100,
                 **kw):
        super(S2Z, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_size = max_size

    def forward(self, x):
        batsize = x.size(0)
        device = x.device
        # encode
        encs, encmask = self.tagger.encode_source(x)
        # predict lengths
        # lens = self.tagger.predict_length(encs, encmask)
        lens = (encmask > 0).sum(1).float()
        # lens = torch.ones(batsize, device=device) * 2
        # return encs, encmask, lens

        # run continuous autoregressive decoding
        maxlen = int(lens.round().long().max().detach().cpu().numpy())
        maxlen = min(maxlen, self.max_size)
        z = [torch.zeros(batsize, self.tagger.dim, device=device)]
        zmask = [torch.ones(batsize, device=device, dtype=torch.bool)]
        cache = None
        for i in range(maxlen+1):
            nextz, cache = self.tagger(tokens=z[-1], enc=encs, encmask=encmask, cache=cache)
            nextzmask = i < lens
            nextz = torch.where(nextzmask[:, None], nextz, torch.zeros_like(nextz))
            z.append(nextz)
            zmask.append(nextzmask)
        z = torch.stack(z, 1)
        zmask = torch.stack(zmask, 1)
        return z, zmask, lens


class S2Z2S(torch.nn.Module):
    def __init__(self,
                 tagger_s2z:DecoderCell,
                 tagger_z2s:DecoderCell,
                 vocab=None,
                 max_size:int=100,
                 smoothing:float=0.,
                 **kw):
        super(S2Z2S, self).__init__(**kw)
        self.s2z = S2Z(tagger_s2z, vocab=vocab, max_size=max_size)
        self.z2s = S2S(tagger_z2s, vocab=vocab, max_size=max_size, smoothing=smoothing)

        self.lenloss = torch.nn.MSELoss(reduction="none")

    def compute_loss(self, zlens, y):
        ylens = (y != 0).sum(1).float()
        lenmse = self.lenloss(zlens, ylens)
        lendist = (ylens.round().long() - zlens.round().long()).abs()
        return lenmse, lendist

    def forward(self, x, y):
        z, zmask, zlens = self.s2z(x)
        ret, pred = self.z2s(z, y, xmask=zmask)

        # lenmse, lendist = self.compute_loss(zlens, y)
        # ret["lenmse"] = lenmse
        # ret["lendist"] = lendist

        # if "loss" in ret:       # add len mse to loss
        #     ret["loss"] += lenmse

        return ret, pred


class Butterfly(torch.nn.Module):
    pass



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


def load_ds(dataset="scan/random", validfrac=0.1, recompute=False, bertname="bert-base-uncased"):
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
        numlayers=2,
        numtmlayers=6,
        numheads=6,
        dropout=0.1,
        worddropout=0.,
        bertname="bert-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        version="v1"
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    # torch.backends.cudnn.enabled = False

    wandb.init(project=f"compgen_butterfly", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    if maxsize < 0:
        if dataset.startswith("cfq"):
            maxsize = 155
        elif dataset.startswith("scan"):
            maxsize = 55
        print(f"maxsize: {maxsize}")

    tt = q.ticktock("script")
    tt.tick("data")
    trainds, validds, testds, fldic, inpdic = load_ds(dataset=dataset, validfrac=validfrac, bertname=bertname,
                                                      recompute=recomputedata)

    if "mcd" in dataset.split("/")[1]:
        print(f"Setting patience to -1 because MCD (was {patience})")
        patience = -1

    tt.msg(f"TRAIN DATA: {len(trainds)}")
    tt.msg(f"DEV DATA: {len(validds)}")
    tt.msg(f"TEST DATA: {len(testds)}")
    if trainonvalid:
        assert False
        trainds = trainds + validds
        validds = testds

    tt.tick("dataloaders")
    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    validdl = DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    testdl = DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    tt.tock()
    tt.tock()

    tt.tick("model")
    # cell = GRUDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, dropout=dropout, worddropout=worddropout)
    # decoder = S2S(cell, vocab=fldic, max_size=maxsize, smoothing=smoothing)
    cell1 = DecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numtmlayers=numtmlayers,
                        dropout=dropout, worddropout=worddropout, mode="cont", numheads=numheads)
    cell2 = DecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numtmlayers=numtmlayers,
                        dropout=dropout, worddropout=worddropout, mode="normal", noencoder=True, numheads=numheads)
    decoder = S2Z2S(cell1, cell2, vocab=fldic, max_size=maxsize, smoothing=smoothing)
    # print(f"one layer of decoder: \n {cell.decoder.block[0]}")
    print(decoder)
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
    metricnames = ["treeacc"]
    tmetrics = make_array_of_metrics(*metricnames, reduction="mean")
    vmetrics = make_array_of_metrics(*metricnames, reduction="mean")
    xmetrics = make_array_of_metrics(*metricnames, reduction="mean")

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
                         remember_f=lambda: deepcopy(decoder.state_dict()))

    def wandb_logger():
        d = {}
        for name, loss in zip(["loss", "acc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        if evaltrain:
            for name, loss in zip(metricnames, tmetrics):
                d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(metricnames, vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        for name, loss in zip(metricnames, xmetrics):
            d["test_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim = get_optim(decoder, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        assert t_max > (warmup + 10)
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(low=0., high=1.0, steps=t_max-warmup) >> (0. * lr)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, on_before_optim_step=[lambda : clipgradnorm(_m=decoder, _norm=gradnorm)])

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
    testepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=xmetrics,
                         dataloader=testdl,
                         device=device)

    tt.tick("training")
    if evaltrain:
        validfs = [trainevalepoch, validepoch]
    else:
        validfs = [validepoch]
    validfs = validfs + [testepoch]

    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=validfs,
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    tt.tick("running test before reloading")
    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock("ran test")

    if eyt.remembered is not None and patience >= 0:
        tt.msg("reloading best")
        decoder.load_state_dict(eyt.remembered)

        tt.tick("rerunning validation")
        validres = validepoch()
        tt.tock(f"Validation results: {validres}")

    tt.tick("running train")
    trainres = trainevalepoch()
    print(f"Train tree acc: {trainres}")
    tt.tock()

    tt.tick("running test")
    testres = testepoch()
    print(f"test tree acc: {testres}")
    tt.tock()

    settings.update({"final_train_loss": tloss[0].get_epoch_error()})
    settings.update({"final_train_tree_acc": tmetrics[0].get_epoch_error()})
    settings.update({"final_valid_tree_acc": vmetrics[0].get_epoch_error()})
    settings.update({"final_test_tree_acc": xmetrics[0].get_epoch_error()})

    wandb.config.update(settings)
    q.pp_dict(settings)

    return decoder, testds


def cat_dicts(x:List[Dict]):
    out = {}
    for k, v in x[0].items():
        out[k] = []
    for xe in x:
        for k, v in xe.items():
            out[k].append(v)
    for k, v in out.items():
        out[k] = torch.cat(v, 0)
    return out


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
        numtmlayers=-1,
        dropout=-1.,
        worddropout=-1.,
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        trainonvalidonly=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        ):

    settings = locals().copy()
    del settings["datasets"]

    ranges = {
        "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3",
                    "cfq/mcd1", "cfq/mcd2", "cfq/mcd3"],
        # "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"],
        # "dataset": ["cfq/mcd1", "cfq/mcd2", "cfq/mcd3"],
        # "dataset": ["scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd3"],
        "dropout": [0.1, 0.25, 0.5],
        "worddropout": [0.],
        "seed": [42, 87646464, 456852],
        "epochs": [40, 25],
        # "epochs": [25],
        "batsize": [256, 128],
        # "batsize": [100],
        "hdim": [384],
        "numlayers": [2],
        "numtmlayers": [6],
        "numheads": [12],
        # "numlayers": [2],
        "lr": [0.0005],
        "enclrmul": [1.],                  # use 1.
        "smoothing": [0.],
        "validinter": [1],
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
        # if spec["dataset"].startswith("cfq"):
        #     if spec["epochs"] != 25 or spec["batsize"] != 128:
        #         return False
        # elif spec["dataset"].startswith("scan"):
        #     if spec["epochs"] != 40 or spec["batsize"] != 256:
        #         return False
        return True

    print(__file__)
    p = __file__ + f".baseline.{dataset}"
    q.run_experiments_random(
        run, ranges, path_prefix=None, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)