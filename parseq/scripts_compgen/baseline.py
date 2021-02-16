import json
import re
from copy import deepcopy
from typing import Dict

import datasets
import qelos as q
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertModel

from parseq.grammar import lisp_to_tree, are_equal_trees
from parseq.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab


class TransformerTagger(torch.nn.Module):
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

        self.emb = torch.nn.Embedding(self.vocabsize, config.d_model)
        self.posemb = torch.nn.Embedding(maxpos, config.d_model)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = baseline
        self.decoder = TransformerStack(decoder_config)

        if baseline:
            self.out = torch.nn.Linear(self.dim, self.vocabsize)
        else:
            self.out = torch.nn.Linear(self.dim * 2, self.vocabsize)

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


ORDERLESS = {}


class SeqDecoderBaseline(torch.nn.Module):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def __init__(self, tagger:TransformerTagger,
                 vocab=None,
                 max_steps:int=20,
                 max_size:int=100,
                 **kw):
        super(SeqDecoderBaseline, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_steps = max_steps
        self.max_size = max_size
        self.kldiv = torch.nn.KLDivLoss(reduction="none")
        self.logsm = torch.nn.LogSoftmax(-1)

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

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


def autocollate(x:Dict, pad_value=0):
    y = {}
    for k in x[0]:
        yk = []
        for xe in x:
            yk.append(xe[k])
        y[k] = yk
        if isinstance(yk[0], torch.LongTensor) and yk[0].dim() == 1:
            y[k] = q.pad_tensors(yk, 0, pad_value)
    for k, yk in y.items():
        if isinstance(yk[0], torch.Tensor):
            yk = [yij[None] for yij in yk]
            y[k] = torch.cat(yk, 0)
    return y


class Tokenizer(object):
    def __init__(self, bertname="bert-base-uncased", outvocab:Vocab=None, **kw):
        super(Tokenizer, self).__init__(**kw)
        self.tokenizer = AutoTokenizer.from_pretrained(bertname)
        self.outvocab = outvocab

    def tokenize(self, inps, outs):
        inptoks = self.tokenizer.tokenize(inps)
        outtoks = self.get_out_toks(outs)
        outret = self.tokenizer.encode(inps, return_tensors="pt")[0]
        ret = {"inps": inps, "outs":outs, "inptoks": inptoks, "outtoks": outtoks, "inptensor": outret, "outtensor": self.tensorize_output(outtoks)}
        return ret

    def get_out_toks(self, x):
        return x.split(" ")

    def tensorize_output(self, x):
        ret = [self.outvocab[xe] for xe in x]
        ret = torch.tensor(ret)
        return ret


def load_ds(dataset="scan", split="simple"):
    tt = q.ticktock("data")
    tt.tick("loading")
    scan_ds = datasets.load_dataset("scan", split)
    print(scan_ds)
    tt.tock("loaded")

    tokenizer = Tokenizer()

    print(len(scan_ds))
    print(scan_ds)
    print(len(scan_ds["train"]))
    print(scan_ds["train"][0])
    print(scan_ds)

    tt.tick("validation set")
    ret = scan_ds["train"].train_test_split(0.1)
    # print(scan_ds)
    print(ret)
    scan_ds["valid"] = ret["test"]
    scan_ds["train"] = ret["train"]
    # scan_ds.train_test_split(0.1)
    print(scan_ds)
    tt.tock()

    tt.tick("dictionaries")
    fldic = Vocab()
    for x in scan_ds["train"]:
        for tok in tokenizer.get_out_toks(x["actions"]):
            fldic.add_token(tok, seen=True)
    for x in scan_ds["valid"]:
        for tok in tokenizer.get_out_toks(x["actions"]):
            fldic.add_token(tok, seen=False)
    for x in scan_ds["test"]:
        for tok in tokenizer.get_out_toks(x["actions"]):
            fldic.add_token(tok, seen=False)

    fldic.finalize(min_freq=0, top_k=np.infty)
    print(f"output vocabulary size: {len(fldic.D)}")
    tt.tock()

    tt.tick("tensorizing")
    tokenizer.outvocab = fldic
    scan_ds = scan_ds.map(lambda x: tokenizer.tokenize(x["commands"], x["actions"]))
    tt.tock("tensorized")

    return scan_ds, fldic


def run(lr=0.001, split="simple", batsize=10):

    tt = q.ticktock("script")
    tt.tick("data")
    ds, fldic = load_ds(dataset="scan", split=split)

    tt.tick("dataloaders")
    ds.set_format(type='torch', columns=['inptensor', 'outtensor'])
    trainds = DataLoader(ds["train"], batch_size=batsize, shuffle=True, collate_fn=autocollate)
    validds = DataLoader(ds["valid"], batch_size=batsize, shuffle=False, collate_fn=autocollate)
    testds = DataLoader(ds["test"], batch_size=batsize, shuffle=False, collate_fn=autocollate)
    # print(json.dumps(next(iter(trainds)), indent=3))
    print(next(iter(trainds)))
    tt.tock()
    tt.tock()

    # print(len(scan_ds[0]))
    # print(len(scan_ds[1]))
    # print(dir(scan_ds))


if __name__ == '__main__':
    q.argprun(run)