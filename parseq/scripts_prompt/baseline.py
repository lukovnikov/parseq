import json
import os
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from typing import Dict

import wandb

import qelos as q
import torch
import numpy as np
from torch.utils.data import DataLoader

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader
from transformers import AutoTokenizer, T5Model

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree
from parseq.scripts_compgen.transformer import TransformerConfig, TransformerStack
from parseq.scripts_compgen.transformerdecoder import TransformerStack as TransformerStackDecoder
from parseq.vocab import Vocab


class TransformerEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_size, dropout=0., pad_token_id=0, max_position_embeddings=512,
                 layer_norm_eps=1e-12, useabspos=True):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
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
    def __init__(self, dim, rng=10, **kw):
        super(BasicRelPosEmb, self).__init__(**kw)

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
            relposemb = torch.einsum("bhqn,nhd->bhqd", gathered, embs)
            if ret is None:
                ret = torch.zeros_like(relposemb)
            ret  = ret + relposemb
        return ret


class TransformerDecoderCell(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, useabspos=True,
                 relposmode="basic", relposrng=10,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(TransformerDecoderCell, self).__init__(**kw)
        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.userelpos = userelpos
        self.relposrng = relposrng
        self.useabspos = useabspos
        decconfig = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                          d_kv=int(self.dim/numheads),
                                   num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)

        self.dec_emb = torch.nn.Embedding(self.vocabsize, decconfig.d_model)
        self.slot_emb = torch.nn.Embedding(1, decconfig.d_model)

        self.relposemb = None
        if self.userelpos is True:
            if relposmode == "basic":
                self.relposemb = BasicRelPosEmb(self.dim, relposrng)
            # elif relposmode == "mod":
            #     self.relposemb = ModRelPosEmb(self.dim, relposrng, levels=4)
            else:
                raise Exception(f"Unrecognized relposmode '{relposmode}'")

        self.absposemb = None
        if self.relposemb is None or self.useabspos is True:
            self.absposemb = torch.nn.Embedding(maxpos, decconfig.d_model)

        decoder_config = deepcopy(decconfig)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = True
        self.decoder = TransformerStackDecoder(decoder_config, rel_emb=self.relposemb)

        self.out = torch.nn.Linear(self.dim, self.vocabsize)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        if self.bertname.startswith("none") or self.bertname == "vanilla":
            self.encrelposemb = None
            if self.userelpos is True:
                if relposmode == "basic":
                    self.encrelposemb = BasicRelPosEmb(self.dim, relposrng)
                # elif relposmode == "mod":
                #     self.relposemb = ModRelPosEmb(self.dim, relposrng, levels=4)
                else:
                    raise Exception(f"Unrecognized relposmode '{relposmode}'")
            bname = "bert" + self.bertname[4:]
            if self.bertname == "vanilla":
                inpvocabsize = inpvocab.number_of_ids()
            else:
                tokenizer = AutoTokenizer.from_pretrained(bname)
                inpvocabsize = tokenizer.vocab_size
            encconfig = TransformerConfig(vocab_size=inpvocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                          d_kv=int(self.dim/numheads),
                                       num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)
            encemb = TransformerEmbeddings(encconfig.vocab_size, encconfig.d_model, dropout=dropout, max_position_embeddings=maxpos, useabspos=useabspos)
            self.encoder_model = TransformerStack(encconfig, encemb, rel_emb=self.encrelposemb)
        else:
            self.encoder_model = BertModel.from_pretrained(self.bertname,
                                                     hidden_dropout_prob=min(dropout, 0.2),
                                                     attention_probs_dropout_prob=min(dropout, 0.1))

        self.adapter = None
        if self.encoder_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.encoder_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
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
        embs = self.dec_emb(tokens)     # (bs, seqlen, dim)
        slots = self.slot_emb.weight[0][None, None].repeat(embs.size(0), embs.size(1), 1)     # (1, seqlen, dim)
        if self.absposemb is not None:
            posembs = self.absposemb(torch.arange(tokens.size(1)+1, device=tokens.device))[None]
            embs = embs + posembs[:, :-1, :]
            slots = slots + posembs[:, 1:, :]
        relpos = None
        if self.relposemb is not None:      # compute relative positions
            positions = torch.arange(tokens.size(1)+1, device=tokens.device)
            positions = torch.cat([positions[:, None], positions[:, None]], 1).view(-1)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos[1:-1, 1:-1]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        embs = torch.cat([embs[:, :, None, :], slots[:, :, None, :]], 2).view(embs.size(0), -1, embs.size(2))
        padmask = torch.cat([padmask[:, :, None], padmask[:, :, None]], 2).view(padmask.size(0), -1)

        if cache is not None:
            embs = embs[:, -2:, :]
            if relpos is not None:
                relpos = relpos[:, -2:, :, :]

        _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=True,
                     past_key_value_states=cache,
                     relpos=relpos)
        ret = _ret[0]
        c = ret.view(ret.size(0), int(ret.size(1)/2), 2, ret.size(2))[:, :, 1, :]
        cache = _ret[1]
        logits = self.out(c)
        return logits, cache

    # def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
    #     padmask = (tokens != 0)
    #     embs = self.dec_emb(tokens)
    #     if self.absposemb is not None:
    #         posembs = self.absposemb(torch.arange(tokens.size(1), device=tokens.device))[None]
    #         embs = embs + posembs
    #     relpos = None
    #     if self.relposemb is not None:      # compute relative positions
    #         positions = torch.arange(tokens.size(1), device=tokens.device)
    #         relpos = positions[None, :] - positions[:, None]
    #         relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
    #         relpos = relpos[None, :, :, None]
    #     if cache is not None:
    #         embs = embs[:, -1:, :]
    #         if relpos is not None:
    #             relpos = relpos[:, -1:, :, :]
    #     _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
    #                  encoder_hidden_states=enc,
    #                  encoder_attention_mask=encmask, use_cache=True,
    #                  past_key_value_states=cache,
    #                  relpos=relpos)
    #     ret = _ret[0]
    #     c = ret
    #     cache = _ret[1]
    #     logits = self.out(c)
    #     return logits, cache


class SeqDecoderBaseline(torch.nn.Module):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def __init__(self, tagger:TransformerDecoderCell,
                 vocab=None,
                 max_size:int=100,
                 smoothing:float=0.,
                 **kw):
        super(SeqDecoderBaseline, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_size = max_size
        self.smoothing = smoothing
        if self.smoothing > 0:
            self.loss = q.SmoothedCELoss(reduction="none", ignore_index=0, smoothing=smoothing, mode="logprobs")
        else:
            self.loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)

        self.logsm = torch.nn.LogSoftmax(-1)

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
        same = same | ~(mask.bool())
        acc = same.all(-1)  # (batsize,)
        return loss, acc.float()

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        preds, stepsused = self.get_prediction(x)

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

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits, cache = self.tagger(tokens=newy, enc=enc, encmask=encmask, cache=None)
        # compute loss: different versions do different masking and different targets
        loss, acc = self.compute_loss(logits, tgt)
        return {"loss": loss, "acc": acc}, logits

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
    return trainds, validds, testds, fldic, inpdic

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
        enclrmul=0.1,
        smoothing=0.1,
        gradnorm=3,
        batsize=60,
        epochs=16,
        patience=10,
        validinter=3,
        validfrac=0.1,
        warmup=3,
        cosinelr=False,
        dataset="scan/length",
        maxsize=50,
        seed=42,
        hdim=768,
        numlayers=6,
        numheads=12,
        dropout=0.1,
        bertname="bert-base-uncased",
        testcode=False,
        userelpos=False,
        gpu=-1,
        evaltrain=False,
        trainonvalid=False,
        trainonvalidonly=False,
        recomputedata=False,
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    wandb.init(project=f"compgen_baseline", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    tt = q.ticktock("script")
    tt.tick("data")
    trainds, validds, testds, fldic, inpdic = load_ds(dataset=dataset, validfrac=validfrac, bertname=bertname, recompute=recomputedata)
    if trainonvalid:
        trainds = trainds + validds
        validds = testds

    tt.tick("dataloaders")
    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    validdl = DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    testdl = DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    # print(json.dumps(next(iter(trainds)), indent=3))
    # print(next(iter(traindl)))
    # print(next(iter(validdl)))
    tt.tock()
    tt.tock()

    tt.tick("model")
    cell = TransformerDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numheads=numheads, dropout=dropout,
                                  bertname=bertname, userelpos=userelpos, useabspos=not userelpos)
    decoder = SeqDecoderBaseline(cell, vocab=fldic, max_size=maxsize, smoothing=smoothing)
    print(f"one layer of decoder: \n {cell.decoder.block[0]}")
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

    tloss = make_array_of_metrics("loss", "acc", reduction="mean")
    tmetrics = make_array_of_metrics("treeacc", reduction="mean")
    vmetrics = make_array_of_metrics("treeacc", reduction="mean")
    xmetrics = make_array_of_metrics("treeacc", reduction="mean")

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

    if patience < 0:
        patience = epochs
    eyt = q.EarlyStopper(vmetrics[0], patience=patience, min_epochs=30, more_is_better=True,
                         remember_f=lambda: deepcopy(cell))

    def wandb_logger():
        d = {}
        for name, loss in zip(["loss", "acc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
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

    tt.tick("training")
    if evaltrain:
        validfs = [trainevalepoch, validepoch]
    else:
        validfs = [validepoch]
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=validfs,
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    tt.tick("running test before reloading")
    testepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=xmetrics,
                         dataloader=testdl,
                         device=device)

    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock("ran test")

    if eyt.remembered is not None:
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
    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock()

    settings.update({"final_train_loss": tloss[0].get_epoch_error()})
    settings.update({"final_train_tree_acc": tmetrics[0].get_epoch_error()})
    settings.update({"final_valid_tree_acc": vmetrics[0].get_epoch_error()})
    settings.update({"final_test_tree_acc": xmetrics[0].get_epoch_error()})

    wandb.config.update(settings)
    q.pp_dict(settings)


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
        maxsize=-1,
        seed=-1,
        hdim=-1,
        numlayers=-1,
        numheads=-1,
        dropout=-1.,
        bertname="vanilla",
        testcode=False,
        userelpos=False,
        trainonvalidonly=False,
        evaltrain=False,
        gpu=-1,
        recomputedata=False,
        ):

    settings = locals().copy()

    ranges = {
        "dataset": ["scan/random", "scan/length", "scan/add_jump", "scan/add_turn_left", "scan/mcd1", "scan/mcd2", "scan/mcd3"],
        "dropout": [0.1, 0.25, 0.5],
        "seed": [42, 87646464, 456852],
        "epochs": [15],
        "batsize": [60],
        "hdim": [768],
        "numheads": [12],
        "numlayers": [6],
        "lr": [0.0001],
        "enclrmul": [0.1],                  # use 1.
        "smoothing": [0., 0.1],
        # "patience": [-1],
        # "warmup": [20],
        "validinter": [2],
        # "gradacc": [1],
    }

    if bertname.startswith("none") or bertname == "vanilla":
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["epochs"] = [58]
        ranges["hdim"] = [384]
        ranges["numheads"] = [6]
        ranges["batsize"] = [256]
        ranges["validinter"] = [3]

        ranges["dropout"] = [0.1]
        ranges["smoothing"] = [0.]

    if dataset.startswith("cfq"):
        settings["maxsize"] = 160
    elif dataset.startswith("scan"):
        settings["maxsize"] = 50

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
        return True

    print(__file__)
    p = __file__ + f".baseline.{dataset}"
    q.run_experiments_random(
        run, ranges, path_prefix=p, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)