import json
import os
import random
import re
import shelve
import itertools
from copy import deepcopy
from functools import partial
from typing import Dict

import wandb

import qelos as q
import torch
import numpy as np
from nltk import Tree
from torch.utils.data import DataLoader

from parseq.datasets import SCANDatasetLoader, autocollate, Dataset, CFQDatasetLoader
from transformers import AutoTokenizer, BertModel

from parseq.eval import make_array_of_metrics
from parseq.grammar import lisp_to_tree, are_equal_trees, taglisp_to_tree, tree_to_taglisp, tree_to_lisp
from parseq.scripts_compgen.baseline import TransformerDecoderCell, BasicRelPosEmb, TransformerEmbeddings
from parseq.scripts_compgen.transformer import TransformerConfig, TransformerStack
from parseq.scripts_compgen.transformerdecoder import TransformerStack as TransformerStackDecoder
from parseq.vocab import Vocab


class QTransformerDecoderCell(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, useabspos=True,
                 relposmode="basic", relposrng=10,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(QTransformerDecoderCell, self).__init__(**kw)
        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.userelpos = userelpos
        self.relposrng = relposrng
        self.useabspos = useabspos
        decconfig = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                      d_kv=int(self.dim / numheads),
                                      num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)

        self.dec_emb = torch.nn.Embedding(self.vocabsize, decconfig.d_model)

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
        self.decoder = TransformerStack(decoder_config, rel_emb=self.relposemb)

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
                                          d_kv=int(self.dim / numheads),
                                          num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)
            encemb = TransformerEmbeddings(encconfig.vocab_size, encconfig.d_model, dropout=dropout,
                                           max_position_embeddings=maxpos, useabspos=useabspos)
            self.encoder_model = TransformerStack(encconfig, encemb, rel_emb=self.encrelposemb)
        else:
            self.encoder_model = BertModel.from_pretrained(self.bertname,
                                                           hidden_dropout_prob=min(dropout, 0.2),
                                                           attention_probs_dropout_prob=min(dropout, 0.1))

        self.adapter = None
        if self.encoder_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.encoder_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.out_mu = torch.nn.Linear(self.dim, int(self.dim/2))
        self.out_logvar = torch.nn.Linear(self.dim, int(self.dim/2))
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
        mu = self.out_mu(c)
        logvar = self.out_logvar(c)

        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) # (batsize, seqlen, dim/2)
        priorkls = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)      # (bs, seqlen)
        priorkls = priorkls * padmask
        priorkl = priorkls.sum(-1)

        return z, priorkl, cache


class PTransformerDecoderCell(TransformerDecoderCell):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, useabspos=True,
                 relposmode="basic", relposrng=10,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(PTransformerDecoderCell, self).__init__(dim, vocab=vocab, inpvocab=inpvocab, numlayers=numlayers,
                 numheads=numheads, userelpos=userelpos, useabspos=useabspos, relposmode=relposmode, relposrng=relposrng,
                 dropout=dropout, maxpos=maxpos, bertname=bertname, **kw)

        self.dec_emb = torch.nn.Embedding(self.vocabsize, int(self.dim/2))
        self.mixproj = None #torch.nn.Linear(self.dim, self.dim)
        self.reset_parameters()

    def forward(self, tokens:torch.Tensor=None, z:torch.Tensor=None, enc=None, encmask=None, cache=None):
        padmask = (tokens != 0)
        _padmask = padmask
        embs = self.dec_emb(tokens)     # (bs, seqlen, dim)
        embs = torch.cat([embs, z], -1)
        embs = self.mixproj(embs) if self.mixproj is not None else embs

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


class SeqDecoderOrderVAE(torch.nn.Module):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def __init__(self,
                 ptagger:PTransformerDecoderCell,
                 qtagger:QTransformerDecoderCell,
                 vocab=None,
                 max_size:int=100,
                 smoothing:float=0.,
                 priorweight=1.,
                 tree_compare=None,
                 **kw):
        super(SeqDecoderOrderVAE, self).__init__(**kw)
        self.ptagger = ptagger
        self.qtagger = qtagger
        self.vocab = vocab
        self.max_size = max_size
        self.smoothing = smoothing
        self.priorweight = priorweight
        if self.smoothing > 0:
            self.loss = q.SmoothedCELoss(reduction="none", ignore_index=0, smoothing=smoothing, mode="logprobs")
        else:
            self.loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)

        self.logsm = torch.nn.LogSoftmax(-1)
        self.tree_compare = tree_compare if tree_compare is not None else partial(are_equal_trees, orderless=ORDERLESS, unktoken="@UNK@")

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
        treeaccs = [float(self.tree_compare(gold_tree, pred_tree))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device), "stepsused": stepsused}
        return ret, pred_trees

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, qy, py, tgt = self.extract_training_example(x, y)

        # run through q
        enc, encmask = self.qtagger.encode_source(x)
        # run through tagger: the same for all versions
        z, priorkl, _ = self.qtagger(tokens=qy, enc=enc, encmask=encmask, cache=None)
        # z: (bs, seqlen, dim/2), priorkl: (bs,)

        # run through p
        enc, encmask = self.ptagger.encode_source(x)
        # run through tagger: the same for all versions
        z = z[:, 1:, :]
        logits, _ = self.ptagger(tokens=py, z=z, enc=enc, encmask=encmask, cache=None)

        # compute loss: different versions do different masking and different targets
        loss, acc = self.compute_loss(logits, tgt)
        loss = self.priorweight * priorkl + loss
        return {"loss": loss, "priorkl": priorkl, "acc": acc}, logits

    def extract_training_example(self, x, y):
        ymask = (y != 0).float()
        ylens = ymask.sum(1).long()
        newy = y
        newy = torch.cat([torch.ones_like(newy[:, 0:1]) * self.vocab["@START@"], newy], 1)
        newy = torch.cat([newy, torch.zeros_like(newy[:, 0:2])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            newy[i, ylen+1] = self.vocab["@END@"]

        return x, newy[:, :-1], newy[:, :-2], newy[:, 1:-1]

    def sample_from_prior(self, bs, seqlen, device=torch.device("cpu")):
        z = torch.randn(bs, seqlen, int(self.ptagger.dim / 2), device=device)
        return z

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_size
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@START@"]
        # yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.ptagger.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        cache = None
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            z = self.sample_from_prior(y.size(0), y.size(1), device=x.device)
            logits, cache = self.ptagger(tokens=y, z=z, enc=enc, encmask=encmask, cache=cache)
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


class Reorderer(object):
    entitypattern = "m\d{1,2}"
    variablepattern = "\?x\d{1,2}"

    def __init__(self, seed=None, inpD:Vocab=None, outD:Vocab=None, orderless=ORDERLESS, reassign_entities_and_variables=True, **kw):
        super(Reorderer, self).__init__(**kw)
        self.seed = seed
        self.rnd = random.Random(seed)
        self.inpD = inpD
        self.outD = outD
        self.reassign_ents_vars = reassign_entities_and_variables
        self.orderless = orderless

        self.validentities = set()
        self.validvariables = set()
        for k, v in self.outD.D.items():
            if re.match(f"^{self.entitypattern}$", k.lower()):
                self.validentities.add(k.lower())
            if re.match(f"^{self.variablepattern}$", k.lower()):
                self.validvariables.add(k.lower())

    def __call__(self, x):
        _x = x
        x, y = x
        recons = taglisp_to_tree(y)
        y = self._reorder_rec(recons)
        y = tree_to_taglisp(y)

        if self.reassign_ents_vars:
            ve = list(self.validentities)
            self.rnd.shuffle(ve)
            entmap = list(zip(sorted(ve), ve))
            vv = list(self.validvariables)
            self.rnd.shuffle(vv)
            varmap = list(zip(sorted(vv), vv))
            mapper = dict(entmap + varmap)

            xmapped = [mapper[xe.lower()].upper() if xe.lower() in mapper else xe for xe in x.split()]
            ymapped = [mapper[ye] if ye in mapper else ye for ye in y.split()]

            x = " ".join(xmapped)
            y = " ".join(ymapped)
        return (x, y)

    def _reorder_rec(self, x:Tree):
        if x.label() in self.orderless:
            children = None
            if len(x) > 0:
                children = [self._reorder_rec(child) for child in x[:]]
                self.rnd.shuffle(children)
            ret = Tree(x.label(), children=children)
        else:
            ret = Tree(x.label(), children = [self._reorder_rec(child) for child in x[:]])
        return ret

    def normalize_entities_and_variables(self, x:Tree, generic=False):
        allents = []
        allvars = []
        queue = [x]
        while len(queue) > 0:
            head = queue.pop(0)
            if re.match(f"^{self.entitypattern}$", head.label()):
                allents.append(head.label())
            elif re.match(f"^{self.variablepattern}$", head.label()):
                allvars.append(head.label())
            queue = head[:] + queue
        mapper = {}
        i = 0
        for k in allents:
            if k not in mapper:
                mapper[k] = sorted(list(self.validentities))[i] if not generic else "@ENT@"
                i += 1
        i = 0
        for k in allvars:
            if k not in mapper:
                mapper[k] = sorted(list(self.validvariables))[i] if not generic else "@VAR@"
                i += 1

        def remap(_x:Tree, _mapper):
            return Tree(_mapper[_x.label()] if _x.label() in _mapper else _x.label(),
                        children=[remap(_xe, _mapper) for _xe in _x])
        ret = remap(x, mapper)
        return ret

    def normalize_tree(self, x:Tree):
        # sort tree children alphabetically
        anonstrs, children = [], []
        rets = [self.normalize_tree(child) for child in x]
        if x.label() in self.orderless:
            rets = sorted(rets, key=lambda x: x[0])
        if len(rets) > 0:
            anonstrs, children = zip(*rets)
            anonstrs, children = list(anonstrs), list(children)
        if re.match(self.entitypattern, x.label()):
            anonret = "@ENT@"
        elif re.match(self.variablepattern, x.label()):
            anonret = "@VAR@"
        else:
            anonret = x.label()
        if len(anonstrs) > 0:
            anonret = f"({anonret} {' '.join(anonstrs)})"
        ret = Tree(x.label(), children=children)
        return anonret, ret

    def tree_to_lisptokens(self, x:Tree):
        xstr = tree_to_lisp(x)
        xstr = xstr.replace("(", " ( ").replace(")", " ) ")
        xstr = re.sub("\s+", " ", xstr)
        xstr = xstr.split(" ")
        return xstr

    def sorted_tree_tandem(self, x:Tree, y:Tree):
        """ Sort both x and y trees by tree x """
        # sort tree children alphabetically
        xchildren, ychildren = [], []
        rets = [self.sorted_tree_tandem(xchild, ychild) for xchild, ychild in zip(x[:], y[:])]
        if x.label() in self.orderless:
            rets = sorted(rets, key=lambda x: x[0])
        if len(rets) > 0:
            xchildren, ychildren = zip(*rets)
            xchildren, ychildren = list(xchildren), list(ychildren)
        xret = Tree(x.label(), children=xchildren)
        yret = Tree(y.label(), children=ychildren)
        return xret, yret

    def hashed_ents_vars(self, x:Tree, current=None):
        xstr = self.tree_to_lisptokens(x)

        done = False
        _xstr = xstr
        current = None
        while not done:
            done = True
            new_xstr = []
            for _xstre, xstre in zip(_xstr, xstr):
                if (re.match(self.entitypattern, xstre) or re.match(self.variablepattern, xstre)) and (_xstre == xstre or _xstre == "ANON$"):
                    if current is None or xstre == current:
                        current = xstre
                        _xstre = "$"
                    else:
                        _xstre = "ANON$"
                        done = False
                else:
                    _xstre = _xstre
                new_xstr.append(_xstre)
            newtree = lisp_to_tree(" ".join(new_xstr))
            newtree, newx = self.sorted_tree_tandem(newtree, x)
            xstr = self.tree_to_lisptokens(newx)
            _xstr = self.tree_to_lisptokens(newtree)
            hsh = int(abs(hash(str(newtree))) % 10e4)
            _xstr = [f"${hsh}" if e == "$" else e for e in _xstr]
            current = None
        ret = lisp_to_tree(" ".join(_xstr))
        return ret

    def reassignments(self, x:Tree):
        # get all entities and variables
        allents = []
        allvars = []
        queue = [x]
        while len(queue) > 0:
            head = queue.pop(0)
            if re.match(f"^{self.entitypattern}$", head.label()):
                allents.append(head.label())
            elif re.match(f"^{self.variablepattern}$", head.label()):
                allvars.append(head.label())
            queue = head[:] + queue
        allents = sorted(list(set(allents)))
        allvars = sorted(list(set(allvars)))

        for entperm in itertools.permutations(allents):
            for varperm in itertools.permutations(allvars):
                mapper = {}
                i = 0
                for k in allents:
                    mapper[k] = entperm[i]
                    i += 1
                i = 0
                for k in allvars:
                    mapper[k] = varperm[i]
                    i += 1

                def remap(_x: Tree, _mapper):
                    return Tree(_mapper[_x.label()] if _x.label() in _mapper else _x.label(),
                                children=[remap(_xe, _mapper) for _xe in _x])

                ret = remap(x, mapper)
                yield ret

    def are_equal_trees(self, x:Tree, y:Tree, use_terminator=False):
        _x = self.normalize_entities_and_variables(x, generic=True)
        _y = self.normalize_entities_and_variables(y, generic=True)
        # print(_x)
        # print(_y)
        ret = are_equal_trees(_x, _y, orderless=self.orderless, unktoken=self.outD[self.outD.unktoken], use_terminator=use_terminator)
        if ret is False:
            return False
        _x = x
        _y = y
        _x = self.normalize_tree(x)[1]
        _y = self.normalize_tree(y)[1]
        _x = self.normalize_entities_and_variables(_x, generic=False)
        _y = self.normalize_entities_and_variables(_y, generic=False)
        # print(_x)
        # print(_y)
        ret = are_equal_trees(_x, _y, orderless=self.orderless, unktoken=self.outD[self.outD.unktoken], use_terminator=use_terminator)
        if ret is False:
            # return False
            # print(_x)
            # print(_y)
            # iterate over possible reassignments of one tree
            for reassigned_y in self.reassignments(_y):
                # reassigned_y = self.normalize_entities_and_variables(reassigned_y, generic=False)
                if are_equal_trees(reassigned_y, _x, orderless=self.orderless, unktoken=self.outD[self.outD.unktoken], use_terminator=use_terminator):
                    return True
            return False
        return ret


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
                ds, fldic = shelved["ds"], shelved["fldic"]
                ds = Dataset(examples=ds)
                inpdic = shelved["inpdic"] if "inpdic" in shelved else None
                tt.tock("loaded from shelf")

    if recompute:
        tt.tick("loading data")
        splits = dataset.split("/")
        dataset, splits = splits[0], splits[1:]
        split = "/".join(splits)
        if dataset == "scan":
            print("this dataset is unsuitable for order vae")
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

        tt.tick("shelving")
        with shelve.open(shelfname) as shelf:
            shelved = {
                "ds": ds.examples,
                "fldic": fldic,
                "inpdic": inpdic,
            }
            shelf[key] = shelved
        tt.tock("shelved")

        print(
            f"input avg/max length is {np.mean(inplens):.1f}/{max(inplens)}, output avg/max length is {np.mean(outlens):.1f}/{max(outlens)}")
        print(f"output vocabulary size: {len(fldic.D)} at output, {len(inpdic.D)} at input")
        tt.tock()

    tt.tick("creating splits")
    tt.tick("creating tokenizer")
    tokenizer = Tokenizer(bertname=bertname)
    tt.tock("created tokenizer")
    tokenizer.inpvocab = inpdic
    tokenizer.outvocab = fldic

    reorderer = Reorderer(inpD=inpdic, outD=fldic)

    trainds = ds.filter(lambda x: x[-1] == "train")\
        .map(lambda x: x[:-1])\
        .map(reorderer)\
        .map(lambda x: tokenizer.tokenize(x[0], x[1]))
    validds = ds.filter(lambda x: x[-1] == "valid")\
        .map(lambda x: x[:-1])\
        .map(lambda x: tokenizer.tokenize(x[0], x[1]))
    testds = ds.filter(lambda x: x[-1] == "test")\
        .map(lambda x: x[:-1])\
        .map(lambda x: tokenizer.tokenize(x[0], x[1]))
    # ds = ds.map(lambda x: tokenizer.tokenize(x[0], x[1]) + (x[2],)).cache(True)
    tt.tock()

    tt.tock(f"loaded '{dataset}'")
    tt.msg(f"#train={len(trainds)}, #valid={len(validds)}, #test={len(testds)}")
    return trainds, validds, testds, fldic, inpdic, reorderer


def try_data(dataset="cfq/mcd1", validfrac=0.1, bertname="vanilla", recompute=False, gpu=-1):
    trainds, validds, testds, fldic, inpdic = load_ds(dataset=dataset, validfrac=validfrac, bertname=bertname, recompute=recompute)
    reorderer = Reorderer(inpD=inpdic, outD=fldic)
    ex1 = trainds[0]
    ex2 = trainds[0]
    print(inpdic.tostr(ex1[0]))
    print(inpdic.tostr(ex2[0]))
    print(fldic.tostr(ex1[1]))
    print(fldic.tostr(ex2[1]))
    a = 0
    c = 0
    tt = q.ticktock("trydata")
    tt.tick("trying data")
    for j in range(min(len(testds), 100000)):
        print(j, c)
        for _ in range(1):
            ex1 = trainds[j]
            ex2 = trainds[j]
            equal = reorderer.are_equal_trees(taglisp_to_tree(fldic.tostr(ex1[1])),
                                        taglisp_to_tree(fldic.tostr(ex2[1])))
            print(equal)
            if equal is False:
                c += 1
    print(c)
    tt.tock()


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
        dataset="cfq/mcd1",
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
        priorweight=1.,
        ):

    settings = locals().copy()
    q.pp_dict(settings, indent=3)
    # wandb.init()

    wandb.init(project=f"compgen_ordervae", config=settings, reinit=True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device("cuda", gpu)

    tt = q.ticktock("script")
    tt.tick("data")
    trainds, validds, testds, fldic, inpdic, reorderer = load_ds(dataset=dataset, validfrac=validfrac, bertname=bertname, recompute=recomputedata)
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
    qtagger = QTransformerDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numheads=numheads, dropout=dropout,
                                  bertname=bertname, userelpos=userelpos, useabspos=not userelpos)
    ptagger = PTransformerDecoderCell(hdim, vocab=fldic, inpvocab=inpdic, numlayers=numlayers, numheads=numheads,
                                      dropout=dropout,
                                      bertname=bertname, userelpos=userelpos, useabspos=not userelpos)
    decoder = SeqDecoderOrderVAE(qtagger=qtagger, ptagger=ptagger, vocab=fldic, max_size=maxsize, smoothing=smoothing,
                                 priorweight=priorweight, tree_compare=lambda x, y: reorderer.are_equal_trees(x, y))
    print(f"one layer of decoder: \n {ptagger.decoder.block[0]}")
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

    tloss = make_array_of_metrics("loss", "priorkl", "acc", reduction="mean")
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
                         remember_f=lambda: deepcopy(decoder))

    def wandb_logger():
        d = {}
        for name, loss in zip(["loss", "priorkl", "acc"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
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
        priorweight=-1.,
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
        "numlayers": [5],
        "lr": [0.0001],
        "enclrmul": [0.1],                  # use 1.
        "smoothing": [0., 0.1],
        # "patience": [-1],
        # "warmup": [20],
        "validinter": [2],
        # "gradacc": [1],
        "priorweight": [1., 0.1, 0.01, 0.001]
    }

    if bertname.startswith("none") or bertname == "vanilla":
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["epochs"] = [40]
        ranges["hdim"] = [384]
        ranges["numheads"] = [6]
        ranges["batsize"] = [64]
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
    # q.argprun(try_data)
    q.argprun(run_experiment)