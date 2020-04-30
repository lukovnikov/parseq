import math
import os
import random
import re
import sys
from abc import ABC
from functools import partial
from typing import *

import dgl
import dill as dill
import torch
import numpy as np
import ujson

import qelos as q
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from nltk import PorterStemmer, Tree

from torch.utils.data import DataLoader

# from funcparse.decoding import TransitionModel, TFActionSeqDecoder, LSTMCellTransition, BeamActionSeqDecoder, \
#     GreedyActionSeqDecoder, TFTokenSeqDecoder
# from funcparse.grammar import FuncGrammar, passtr_to_pas
# from funcparse.states import FuncTreeState, FuncTreeStateBatch, BasicState, BasicStateBatch
# from funcparse.vocab import VocabBuilder, SentenceEncoder, FuncQueryEncoder
# from funcparse.nn import TokenEmb, PtrGenOutput, SumPtrGenOutput, BasicGenOutput
from parseq.decoding import SeqDecoder, BeamDecoder, BeamTransition, merge_metric_dicts
from parseq.eval import CELoss, SeqAccuracies, make_array_of_metrics, DerivedAccuracy, TreeAccuracy, Metric, Loss, BCELoss
from parseq.grammar import prolog_to_pas, lisp_to_pas, pas_to_prolog, pas_to_tree, tree_size, tree_to_prolog, \
    tree_to_lisp, lisp_to_tree, are_equal_trees
from parseq.nn import TokenEmb, BasicGenOutput, PtrGenOutput, PtrGenOutput2, load_pretrained_embeddings, GRUEncoder, \
    LSTMEncoder
from parseq.states import DecodableState, BasicDecoderState, State, TreeDecoderState, ListState, TrainableState
from parseq.transitions import TransitionModel, LSTMCellTransition, LSTMTransition, GRUTransition
from parseq.util import DatasetSplitProxy
from parseq.vocab import SequenceEncoder, Vocab


def stem_id_words(pas, idparents, stem=False, strtok=None):
    if stem is True:
        assert(not isinstance(pas, tuple))
    if not isinstance(pas, tuple):
        if stem is True:
            assert(isinstance(pas, str))
            if re.match(r"'([^']+)'", pas):
                pas = re.match(r"'([^']+)'", pas).group(1)
                pas = strtok(pas)
                return [("str", pas)]
            else:
                return [pas]
        else:
            return [pas]
    else:
        tostem = pas[0] in idparents
        children = [stem_id_words(k, idparents, stem=tostem, strtok=strtok)
                    for k in pas[1]]
        children = [a for b in children for a in b]
        return [(pas[0], children)]


def stem_id_words_tree(tree:Tree, idparents, stem=False, strtok=None):
    if stem is True:
        assert(len(tree) == 0)  # should be leaf
    if len(tree) == 0:
        if stem is True:
            if re.match(r"'([^']+)'", tree.label()):
                pas = re.match(r"'([^']+)'", tree.label()).group(1)
                pas = strtok(pas)
                return [Tree("str", pas)]
            else:
                return [tree]
        else:
            return [tree]
    else:
        tostem = tree.label() in idparents
        children = [stem_id_words_tree(k, idparents, stem=tostem, strtok=strtok)
                    for k in tree]
        children = [a for b in children for a in b]
        return [Tree(tree.label(), children)]


def pas2toks(pas):
    if not isinstance(pas, tuple):
        return [pas]
    else:
        children = [pas2toks(k) for k in pas[1]]
        ret = [pas[0]] if pas[0] != "@NAMELESS@" else []
        ret.insert(0, "(")
        for child in children:
            ret += child
            # ret.append(",")
        # ret.pop(-1)
        ret.append(")")
        return ret


def tree2toks(tree:Tree):
    if len(tree) == 0:
        return [tree.label()]
    else:
        children = [tree2toks(x) for x in tree]
        ret = [tree.label()] if tree.label() != "@NAMELESS@" else []
        ret.insert(0, "(")
        for child in children:
            ret += child
        ret.append(")")
        return ret


def basic_query_tokenizer(x:str, strtok=None):
    pas = lisp_to_pas(x)
    # idpreds = set("_cityid _countryid _stateid _riverid _placeid".split(" "))
    # idpreds = set("cityid stateid countryid riverid placeid".split(" "))
    idpreds = set()
    pas = stem_id_words(pas, idpreds, strtok=strtok)[0]
    ret = pas2toks(pas)
    return ret


def tree_query_tokenizer(x:Tree, strtok=None):
    idpreds = set()
    _x = stem_id_words_tree(x, idpreds, strtok=strtok)[0]
    ret = tree2toks(_x)
    return ret


def try_basic_query_tokenizer():
    stemmer = PorterStemmer()
    x = "answer(cityid('new york', _))"
    y = basic_query_tokenizer(x, strtok=lambda x: [stemmer.stem(xe) for xe in x.split()])
    # print(y)


class RankState(TrainableState):
    def __init__(self,
                 inp_tensor:torch.Tensor=None,
                 gold_tensor:torch.Tensor=None,
                 candtensors:torch.Tensor=None,
                 candgold:torch.Tensor=None,
                 alignments:torch.Tensor=None,
                 alignment_entropies:torch.Tensor=None,
                 sentence_vocab:Vocab=None,
                 query_vocab:Vocab=None,
                 **kw):
        if inp_tensor is None:
            super(RankState, self).__init__(**kw)
        else:
            super(RankState, self).__init__(inp_tensor=inp_tensor,
                                            gold_tensor=gold_tensor,
                                            candtensors=candtensors,
                                            candgold=candgold,
                                            alignments=alignments,
                                            alignment_entropies=alignment_entropies,
                                            **kw)
            self.sentence_vocab = sentence_vocab
            self.query_vocab = query_vocab

    def make_copy(self, ret=None, detach=None, deep=True):
        ret = super(RankState, self).make_copy(ret=ret, detach=detach, deep=deep)
        ret.sentence_vocab = self.sentence_vocab
        ret.query_vocab = self.query_vocab
        return ret

    @classmethod
    def merge(cls, states:List['RankState'], ret=None):
        assert(all([state.sentence_vocab == states[0].sentence_vocab and state.query_vocab == states[0].query_vocab for state in states]))
        ret = super(RankState, cls).merge(states, ret=ret)
        ret.sentence_vocab = states[0].sentence_vocab
        ret.query_vocab = states[0].query_vocab
        return ret

    def __getitem__(self, item):
        ret = super(RankState, self).__getitem__(item)
        ret.sentence_vocab = self.sentence_vocab
        ret.query_vocab = self.query_vocab
        return ret

    def __setitem__(self, key, value:'RankState'):
        assert(value.sentence_vocab == self.sentence_vocab and value.query_vocab == self.query_vocab)
        ret = super(RankState, self).__setitem__(key, value)
        return ret

    def get_gold(self, i:int=None) -> torch.Tensor:
        if i is None:
            return self.candgold
        else:
            assert(False)   # can't have time slicing


class GeoDatasetRank(object):
    def __init__(self,
                 p="geoquery_gen/run4/",
                 min_freq:int=2,
                 splits=None, **kw):
        super(GeoDatasetRank, self).__init__(**kw)
        self._initialize(p)
        self.splits_proportions = splits

    def _initialize(self, p):
        self.data = {}
        with open(os.path.join(p, "trainpreds.json")) as f:
            trainpreds = ujson.load(f)
        with open(os.path.join(p, "testpreds.json")) as f:
            testpreds = ujson.load(f)
        splits = ["train"]*len(trainpreds) + ["test"] * len(testpreds)
        preds = trainpreds + testpreds

        self.sentence_encoder = SequenceEncoder(tokenizer=lambda x: x.split())
        self.query_encoder = SequenceEncoder(tokenizer=lambda x: x.split())

        # build vocabularies
        for i, (example, split) in enumerate(zip(preds, splits)):
            self.sentence_encoder.inc_build_vocab(" ".join(example["sentence"]), seen=split=="train")
            self.query_encoder.inc_build_vocab(" ".join(example["gold"]), seen=split=="train")
            for can in example["candidates"]:
                self.query_encoder.inc_build_vocab(" ".join(can["tokens"]), seen=False)
        # for word, wordid in self.sentence_encoder.vocab.D.items():
        #     self.query_encoder.vocab.add_token(word, seen=False)
        self.sentence_encoder.finalize_vocab()
        self.query_encoder.finalize_vocab()

        self.build_data(preds, splits)

    def build_data(self, examples:Iterable[dict], splits:Iterable[str]):
        maxlen_in, maxlen_out = 0, 0
        for example, split in zip(examples, splits):
            inp, out = " ".join(example["sentence"]), " ".join(example["gold"])
            inp_tensor, inp_tokens = self.sentence_encoder.convert(inp, return_what="tensor,tokens")
            gold_tree = lisp_to_tree(" ".join(example["gold"][:-1]))
            if not isinstance(gold_tree, Tree):
                assert(gold_tree is not None)
            gold_tensor, gold_tokens = self.query_encoder.convert(out, return_what="tensor,tokens")

            candidate_tensors, candidate_tokens, candidate_align_tensors = [], [], []
            candidate_align_entropies = []
            candidate_trees = []
            candidate_same = []
            for cand in example["candidates"]:
                cand_tree, _ = lisp_to_tree(" ".join(cand["tokens"][:-1]), None)
                if cand_tree is None:
                    cand_tree = Tree("@UNK@", [])
                assert(cand_tree is not None)
                cand_tensor, cand_tokens = self.query_encoder.convert(" ".join(cand["tokens"]), return_what="tensor,tokens")
                candidate_tensors.append(cand_tensor)
                candidate_tokens.append(cand_tokens)
                candidate_align_tensors.append(torch.tensor(cand["alignments"]))
                candidate_align_entropies.append(torch.tensor(cand["align_entropies"]))
                candidate_trees.append(cand_tree)
                candidate_same.append(are_equal_trees(cand_tree, gold_tree, orderless={"and", "or"}, unktoken="@NOUNKTOKENHERE@"))

            candidate_tensor = torch.stack(q.pad_tensors(candidate_tensors, 0), 0)
            candidate_align_tensor = torch.stack(q.pad_tensors(candidate_align_tensors, 0), 0)
            candidate_align_entropy = torch.stack(q.pad_tensors(candidate_align_entropies, 0), 0)
            candidate_same = torch.tensor(candidate_same)

            state = RankState(inp_tensor[None, :],
                              gold_tensor[None, :],
                              candidate_tensor[None, :, :],
                              candidate_same[None, :],
                              candidate_align_tensor[None, :],
                              candidate_align_entropy[None, :],
                              self.sentence_encoder.vocab,
                              self.query_encoder.vocab,
                              )
            if split not in self.data:
                self.data[split] = []
            self.data[split].append(state)
            maxlen_in = max(maxlen_in, len(inp_tokens))
            maxlen_out = max(maxlen_out, candidate_tensor.size(-1), gold_tensor.size(-1))
        self.maxlen_input = maxlen_in
        self.maxlen_output = maxlen_out

    def get_split(self, split:str):
        return DatasetSplitProxy(self.data[split])

    @staticmethod
    def collate_fn(data:Iterable):
        goldmaxlen = 0
        inpmaxlen = 0
        data = [state.make_copy(detach=True, deep=True) for state in data]
        for state in data:
            goldmaxlen = max(goldmaxlen, state.gold_tensor.size(1))
            inpmaxlen = max(inpmaxlen, state.inp_tensor.size(1))
            goldmaxlen = max(goldmaxlen, state.candtensors.size(-1))
        inp_tensors = q.pad_tensors([state.inp_tensor for state in data], 1, 0)
        gold_tensors = q.pad_tensors([state.gold_tensor for state in data], 1, 0)
        candtensors = q.pad_tensors([state.candtensors for state in data], 2, 0)
        alignments = q.pad_tensors([state.alignments for state in data], 2, 0)
        alignment_entropies = q.pad_tensors([state.alignment_entropies for state in data], 2, 0)

        for i, state in enumerate(data):
            state.inp_tensor = inp_tensors[i]
            state.gold_tensor = gold_tensors[i]
            state.candtensors = candtensors[i]
            state.alignments = alignments[i]
            state.alignment_entropies = alignment_entropies[i]
        ret = data[0].merge(data)
        return ret

    def dataloader(self, split:str=None, batsize:int=5, shuffle=None):
        if split is None:   # return all splits
            ret = {}
            for split in self.data.keys():
                ret[split] = self.dataloader(batsize=batsize, split=split, shuffle=shuffle)
            return ret
        else:
            assert(split in self.data.keys())
            shuffle = shuffle if shuffle is not None else split in ("train", "train+valid")
            dl = DataLoader(self.get_split(split), batch_size=batsize, shuffle=shuffle, collate_fn=type(self).collate_fn)
            return dl


def try_dataset():
    tt = q.ticktock("dataset")
    tt.tick("building dataset")
    ds = GeoDatasetRank()
    train_dl = ds.dataloader("train", batsize=20)
    test_dl = ds.dataloader("test", batsize=20)
    examples = set()
    examples_list = []
    duplicates = []
    testexamples = set()
    testexamples_list = []
    testduplicates = []
    for b in train_dl:
        for i in range(len(b)):
            example = b.inp_strings[i] + " --> " + str(b.gold_trees[i])
            if example in examples:
                duplicates.append(example)
            examples.add(example)
            examples_list.append(example)
            # print(example)
    for b in test_dl:
        for i in range(len(b)):
            example = b.inp_strings[i] + " --> " + str(b.gold_trees[i])
            if example in examples:
                testduplicates.append(example)
            testexamples.add(example)
            testexamples_list.append(example)

    print(f"duplicates within train: {len(duplicates)} from {len(examples_list)} total")
    print(f"duplicates from test to train: {len(testduplicates)} from {len(testexamples_list)} total:")
    for x in testduplicates:
        print(x)
    tt.tock("dataset built")


class Ranker(torch.nn.Module):
    def __init__(self, model,
                 eval:List[Union[Metric, Loss]]=tuple(),
                 evalseq:List[Union[Metric, Loss]]=tuple(),
                 **kw):
        super(Ranker, self).__init__(**kw)
        self.model = model
        self._metrics = eval
        self._seq_metrics = evalseq

    def forward(self, x:RankState):
        inp_tensors = x.inp_tensor
        cand_tensors = x.candtensors
        alignments = x.alignments
        align_entropies = x.alignment_entropies

        candscores = self.model(inp_tensors, cand_tensors, alignments, align_entropies)
        _, candpred = candscores.max(-1)
        # _, candpred = x.candgold.max(-1)
        candgold = x.candgold.to(torch.float)

        candtensors = x.candtensors
        pred_tensor = candtensors.gather(1, candpred[:, None, None].repeat(1, 1, candtensors.size(2)))[:, 0]
        gold_tensor = x.gold_tensor

        rank_metrics = [metric(candscores, candpred, candgold, x) for metric in self._metrics]
        seq_metrics = [metric(None, pred_tensor, gold_tensor, x) for metric in self._seq_metrics]
        metrics = merge_metric_dicts(*(rank_metrics + seq_metrics))
        return metrics, x


class RankModel(torch.nn.Module):
    def __init__(self, embdim, hdim, numlayers:int=1, dropout=0.,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 **kw):
        super(RankModel, self).__init__(**kw)

        inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
        # _, covered_word_ids = load_pretrained_embeddings(inpemb.emb, sentence_encoder.vocab.D,
        #                                                  p="../../data/glove/glove300uncased")  # load glove embeddings where possible into the inner embedding class
        # inpemb._do_rare(inpemb.rare_token_ids - covered_word_ids)
        self.inp_emb = inpemb

        encoder = LSTMEncoder(embdim, hdim // 2, num_layers=numlayers, dropout=dropout, bidirectional=True)
        self.inp_enc = encoder

        decoder_emb = torch.nn.Embedding(query_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        self.out_emb = decoder_emb

        encoder = LSTMEncoder(embdim, hdim // 2, num_layers=numlayers, dropout=dropout, bidirectional=True)
        self.out_enc = encoder

        self.lin_map = torch.nn.Sequential(torch.nn.Linear(hdim, hdim), torch.nn.Tanh())

    def forward(self, inptensor, candtensors, alignments, align_entropies):
        inpemb = self.inp_emb(inptensor)
        _, inpenc = self.inp_enc(inpemb, mask=inptensor!=0)
        inpenc = inpenc[-1][0]  # top output state of bilstm
        inpenc = self.lin_map(inpenc)

        outtensor = candtensors.view(-1, candtensors.size(-1))
        outemb = self.out_emb(outtensor)
        _, outenc = self.out_enc(outemb, mask=outtensor!=0)
        outenc = outenc[-1][0]
        outenc = outenc.view(candtensors.size(0), candtensors.size(1), outenc.size(-1))

        scores = inpenc[:, None, :] * outenc
        scores = scores.sum(-1)
        return scores


class ParikhRankModel(torch.nn.Module):
    def __init__(self, embdim, encdim=None, fdim=None, gdim=None, dropout=0., sidessame=True,
                 sentence_encoder=None, query_encoder=None, **kw):
        super(ParikhRankModel, self).__init__(**kw)
        self.encdim = encdim if encdim is not None else embdim
        self.fdim = fdim if fdim is not None else embdim
        self.gdim = gdim if gdim is not None else embdim
        self.dropout = torch.nn.Dropout(dropout)

        inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
        # _, covered_word_ids = load_pretrained_embeddings(inpemb.emb, sentence_encoder.vocab.D,
        #                                                  p="../../data/glove/glove300uncased")  # load glove embeddings where possible into the inner embedding class
        # inpemb._do_rare(inpemb.rare_token_ids - covered_word_ids)
        self.inp_emb = inpemb
        self.out_emb = torch.nn.Embedding(query_encoder.vocab.number_of_ids(), embdim, padding_idx=0)

        self.FlinA = torch.nn.Sequential(torch.nn.Linear(self.encdim, self.fdim), torch.nn.ReLU())
        self.FlinB = self.FlinA if sidessame else torch.nn.Sequential(torch.nn.Linear(self.encdim, self.fdim), torch.nn.ReLU())
        self.Glin1 = torch.nn.Sequential(torch.nn.Linear(self.encdim *2, self.gdim), torch.nn.ReLU())
        self.Glin2 = self.Glin1 if sidessame else torch.nn.Sequential(torch.nn.Linear(self.encdim *2, self.gdim), torch.nn.ReLU())
        self.Hlin = torch.nn.Sequential(torch.nn.Linear(self.gdim*2, 1))

    def forward(self, inptensor, candtensors, alignments, align_entropies):
        a = self.inp_emb(inptensor)
        a_stripe = a
        a_mapped = self.FlinA(self.dropout(a_stripe))

        b = self.out_emb(candtensors)
        b_stripe = b
        b_mapped = self.FlinB(self.dropout(b_stripe))

        e_ij = torch.einsum("bsd,bczd->bcsz", a_mapped, b_mapped)
        _alpha_j = torch.softmax(e_ij, 2)   # bc(s!)z
        _beta_i = torch.softmax(e_ij, 3)    # bcs(z!)
        alpha_j = _alpha_j[:, :, :, :, None] * a_stripe[:, None, :, None, :]
        beta_i = _beta_i[:, :, :, :, None] * b_stripe[:, :, None, :, :]
        alpha_j = alpha_j.sum(2)    # bczd
        beta_i = beta_i.sum(3)      # bcsd

        v_1i = self.Glin1(self.dropout(torch.cat([a_stripe[:, None, :, :].repeat(1, beta_i.size(1), 1, 1), beta_i], -1)))
        v_2j = self.Glin2(self.dropout(torch.cat([b_stripe, alpha_j], -1)))

        v_1 = v_1i.sum(2)
        v_2 = v_2j.sum(2)       # bcd

        y = self.Hlin(torch.cat([v_1, v_2], 2))[:, :, 0]    # bc
        return y




class TreeGRUCell(torch.nn.Module):
    def __init__(self, nodedim, reldim, hdim, dropout=0., **kw):
        super(TreeGRUCell, self).__init__(**kw)
        self.node_gru = torch.nn.GRUCell(nodedim, hdim)
        self.rel_map = torch.nn.GRUCell(reldim, hdim)
        self.dropout = torch.nn.Dropout(dropout)

    def message_func(self, edges):
        msg = self.rel_map(edges.data["x"], edges.src["h"])
        return {"msg": msg}

    def reduce_func(self, nodes):
        red = nodes.mailbox["msg"].sum(1)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = self.node_gru(nodes.data["x"], nodes.data["red"])
        return {"h": h}


class TreeGRUEncoder(torch.nn.Module):
    def __init__(self, embdim, hdim, numrels, num_layers=1, dropout=0., bidirectional=False, **kw):
        super(TreeGRUEncoder, self).__init__(**kw)
        self.hdim = hdim
        self.rel_emb = torch.nn.Embedding(numrels, embdim, padding_idx=0)
        self.rev_rel_emb = torch.nn.Embedding(numrels, embdim, padding_idx=0)

        dims = [embdim]
        for _ in range(1, num_layers):
            dims.append(hdim*2 if bidirectional else hdim)
        # multi layer that works with bidir and without re-doing everything into a graph
        self.bu_cells = torch.nn.ModuleList([TreeGRUCell(_embdim, embdim, hdim, dropout=dropout) for _embdim in dims])
        self.td_cells = torch.nn.ModuleList([TreeGRUCell(_embdim, embdim, hdim, dropout=dropout) for _embdim in dims]) if bidirectional else None

    def forward(self, embs, parents, rels, mask=None):
        # build dgl graph

        # add nodes
        g = dgl.DGLGraph()
        g.add_nodes(embs.size(0)*embs.size(1))

        if self.td_cells is not None:
            g2 = dgl.DGLGraph()
            g2.add_nodes(embs.size(0)*embs.size(1))

        # add edges
        maxtime = embs.size(1)
        for i in range(len(embs)):
            for t in range(embs.size(1)):
                if parents[i, t].item() != -1:
                    g.add_edge(i * maxtime + t, i * maxtime + parents[i, t].item(),
                               {"relid": rels[i, t][None],
                                "x": self.rel_emb(rels[i, t])[None]})
                    if self.td_cells is not None:
                        g2.add_edge(i * maxtime + parents[i, t].item(), i * maxtime + t,
                                    {"relid": rels[i, t][None],
                                     "x": self.rev_rel_emb(rels[i, t])[None]})

        states = embs

        for i in range(len(self.bu_cells)):
            bu_cell = self.bu_cells[i]
            td_cell = self.td_cells[i] if self.td_cells is not None else None
            g.ndata["x"] = states.view(-1, states.size(-1))
            g.ndata["red"] = torch.zeros(g.ndata["x"].size(0), self.hdim, device=states.device)

            if td_cell is not None:
                g2.ndata["x"] = states.view(-1, states.size(-1))
                g2.ndata["red"] = torch.zeros(g.ndata["x"].size(0), self.hdim, device=states.device)

            g_traversal_order = dgl.topological_nodes_generator(g)
            g.prop_nodes(g_traversal_order,
                         message_func=bu_cell.message_func,
                         reduce_func=bu_cell.reduce_func,
                         apply_node_func=bu_cell.apply_node_func)

            if td_cell is not None:
                g2_traversal_order = dgl.topological_nodes_generator(g2)
                g2.prop_nodes(g2_traversal_order,
                             message_func=td_cell.message_func,
                             reduce_func=td_cell.reduce_func,
                             apply_node_func=td_cell.apply_node_func)

            bu_states = g.ndata["h"].view(states.size(0), states.size(1), -1)
            states = bu_states
            if td_cell is not None:
                td_states = g2.ndata["h"].view(states.size(0), states.size(1), -1)
                states = torch.cat([bu_states, td_states], 2)
        return states


def try_tree_gru_encoder():
    m = TreeGRUEncoder(10, 20, 10, 2, .1, True)
    embs = torch.nn.Parameter(torch.randn(2, 5, 10))
    parents = torch.tensor([
        [-1, -1, 1, 2, -1],
        [-1, -1, 1, 1, -1]
    ])
    rels = torch.ones_like(parents)
    y = m(embs, parents, rels)
    print(y.size())
    y.sum().backward()
    print(embs.grad.norm())



class TreeRankModel(torch.nn.Module):
    open_token = "("
    close_token = ")"
    end_token = "@END@"
    def __init__(self, embdim, hdim, numlayers:int=1, dropout=0.,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 **kw):
        super(TreeRankModel, self).__init__(**kw)

        self.open_id = query_encoder.vocab[self.open_token]
        self.close_id = query_encoder.vocab[self.close_token]
        self.end_id = query_encoder.vocab[self.end_token]
        self.query_vocab = query_encoder.vocab

        inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
        # _, covered_word_ids = load_pretrained_embeddings(inpemb.emb, sentence_encoder.vocab.D,
        #                                                  p="../../data/glove/glove300uncased")  # load glove embeddings where possible into the inner embedding class
        # inpemb._do_rare(inpemb.rare_token_ids - covered_word_ids)
        self.inp_emb = inpemb
        self.inp_enc = LSTMEncoder(embdim, hdim // 2, num_layers=numlayers, dropout=dropout, bidirectional=True)

        self.out_emb = torch.nn.Embedding(query_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        self.out_enc = TreeGRUEncoder(embdim, hdim, 10, num_layers=numlayers, dropout=dropout, bidirectional=False)

        self.lin_map = torch.nn.Sequential(torch.nn.Linear(hdim, hdim), torch.nn.Tanh())

    def get_parents_and_rels(self, x):     # x: tensor of ids of out vocabulary
        rels = []
        parents = []
        roots = [0 for _ in range(len(x))]
        for i in range(len(x)):
            xe = list(x[i].detach().cpu().numpy())
            relse = []
            parentse = []
            stacke = [0]
            pstacke = [-1]
            for j in range(len(xe)):
                if xe[j] == 0 or xe[j] == self.end_id:
                    relse.append(0)
                    parentse.append(-1)
                elif self.open_id == xe[j]:
                    relse.append(0)
                    stacke.append(0)
                    parentse.append(-1)
                    pstacke.append(None)
                elif self.close_id == xe[j]:
                    relse.append(stacke[-2])
                    parentse.append(-1)
                    stacke[-2] += 1
                    stacke.pop(-1)
                    pstacke.pop(-1)
                else:
                    relse.append(stacke[-1])
                    if pstacke[-1] is None:
                        pstacke[-1] = j
                        parentse.append(pstacke[-2])
                        if roots[i] == 0:
                            roots[i] = j
                    else:
                        parentse.append(pstacke[-1])
                    stacke[-1] += 1
            rels.append(relse)
            parents.append(parentse)
        rels = torch.tensor(rels).to(x.device)
        parents = torch.tensor(parents).to(x.device)
        roots = torch.tensor(roots).to(x.device)
        return parents, rels, roots

    def forward(self, inptensor, candtensors, alignments, align_entropies):
        inpemb = self.inp_emb(inptensor)
        _, inpenc = self.inp_enc(inpemb, mask=inptensor!=0)
        inpenc = inpenc[-1][0]  # top output state of bilstm
        inpenc = self.lin_map(inpenc)

        outtensor = candtensors.view(-1, candtensors.size(-1))
        outemb = self.out_emb(outtensor)

        parents, rels, roots = self.get_parents_and_rels(outtensor) # relation of this element wrt its parent

        outenc = self.out_enc(outemb, parents, rels, mask=outtensor!=0)
        finalenc = outenc.gather(1, roots[:, None, None].repeat(1, 1, outenc.size(-1)))[:, 0]
        finalenc = finalenc.view(candtensors.size(0), candtensors.size(1), outenc.size(-1))

        scores = inpenc[:, None, :] * finalenc
        scores = scores.sum(-1)
        return scores


class BasicGenModel(TransitionModel):
    def __init__(self, embdim, hdim, numlayers:int=1, dropout=0.,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 feedatt=False, store_attn=True, **kw):
        super(BasicGenModel, self).__init__(**kw)

        inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
        # _, covered_word_ids = load_pretrained_embeddings(inpemb.emb, sentence_encoder.vocab.D,
        #                                                  p="../../data/glove/glove300uncased")  # load glove embeddings where possible into the inner embedding class
        # inpemb._do_rare(inpemb.rare_token_ids - covered_word_ids)
        self.inp_emb = inpemb

        encoder_dim = hdim
        encoder = LSTMEncoder(embdim, hdim // 2, num_layers=numlayers, dropout=dropout, bidirectional=True)
        # encoder = q.LSTMEncoder(embdim, *([encoder_dim // 2] * numlayers), bidir=True, dropout_in=dropout)
        self.inp_enc = encoder

        decoder_emb = torch.nn.Embedding(query_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        self.out_emb = decoder_emb

        dec_rnn_in_dim = embdim + (encoder_dim if feedatt else 0)
        decoder_rnn = LSTMTransition(dec_rnn_in_dim, hdim, numlayers, dropout=dropout)
        self.out_rnn = decoder_rnn

        decoder_out = BasicGenOutput(hdim + encoder_dim, vocab=query_encoder.vocab)
        # decoder_out.build_copy_maps(inp_vocab=sentence_encoder.vocab)
        self.out_lin = decoder_out

        self.att = q.Attention(q.SimpleFwdAttComp(hdim, encoder_dim, hdim), dropout=min(0.1, dropout))

        self.enc_to_dec = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(encoder_dim, hdim),
            torch.nn.Tanh()
        ) for _ in range(numlayers)])

        self.feedatt = feedatt
        self.nocopy = True

        self.store_attn = store_attn

        self.reset_parameters()

    def reset_parameters(self):
        def _param_reset(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    torch.nn.init.uniform_(m.bias, -0.1, 0.1)
            elif type(m) in (torch.nn.LSTM, torch.nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name or "bias" in name:
                        torch.nn.init.uniform(param, -0.1, 0.1)
            elif type(m) == torch.nn.Embedding:
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                torch.nn.init.constant_(m.weight[0], 0)
        # self.apply(_param_reset)

    def forward(self, x:State):
        if not "mstate" in x:
            x.mstate = State()
        mstate = x.mstate
        init_states = []
        if not "ctx" in mstate:
            # encode input
            inptensor = x.inp_tensor
            mask = inptensor != 0
            inpembs = self.inp_emb(inptensor)
            # inpembs = self.dropout(inpembs)
            inpenc, final_encs = self.inp_enc(inpembs, mask)
            for i, final_enc in enumerate(final_encs):    # iter over layers
                _fenc = self.enc_to_dec[i](final_enc[0])
                init_states.append(_fenc)
            mstate.ctx = inpenc
            mstate.ctx_mask = mask

        ctx = mstate.ctx
        ctx_mask = mstate.ctx_mask

        emb = self.out_emb(x.prev_actions)

        if not "rnnstate" in mstate:
            init_rnn_state = self.out_rnn.get_init_state(emb.size(0), emb.device)
            # uncomment next line to initialize decoder state with last state of encoder
            # init_rnn_state[f"{len(init_rnn_state)-1}"]["c"] = final_enc
            if len(init_states) == init_rnn_state.h.size(1):
                init_rnn_state.h = torch.stack(init_states, 1).contiguous()
            mstate.rnnstate = init_rnn_state

        if "prev_summ" not in mstate:
            # mstate.prev_summ = torch.zeros_like(ctx[:, 0])
            mstate.prev_summ = final_encs[-1][0]
        _emb = emb
        if self.feedatt == True:
            _emb = torch.cat([_emb, mstate.prev_summ], 1)
        enc, new_rnnstate = self.out_rnn(_emb, mstate.rnnstate)
        mstate.rnnstate = new_rnnstate

        alphas, summ, scores = self.att(enc, ctx, ctx_mask)
        mstate.prev_summ = summ
        enc = torch.cat([enc, summ], -1)

        if self.training:
            out_mask = None
        else:
            out_mask = x.get_out_mask(device=enc.device)

        if self.nocopy is True:
            outs = self.out_lin(enc, out_mask)
        else:
            outs = self.out_lin(enc, x.inp_tensor, scores, out_mask=out_mask)
        outs = (outs,) if not q.issequence(outs) else outs
        # _, preds = outs.max(-1)

        if self.store_attn:
            if "stored_attentions" not in x:
                x.stored_attentions = torch.zeros(alphas.size(0), 0, alphas.size(1), device=alphas.device)
            x.stored_attentions = torch.cat([x.stored_attentions, alphas.detach()[:, None, :]], 1)

        return outs[0], x


def do_rare_stats(ds, sentence_rare_tokens=None, query_rare_tokens=None):
    # how many examples contain rare words, in input and output, in both train and test
    def get_rare_portions(examples:List[State]):
        total = 0
        rare_in_question = 0
        rare_in_query = 0
        rare_in_both = 0
        rare_in_either = 0
        for example in examples:
            total += 1
            question_tokens = example.inp_tokens[0]
            query_tokens = example.gold_tokens[0]
            both = True
            either = False
            _sentence_rare_tokens = example.sentence_vocab.rare_tokens if sentence_rare_tokens is None else sentence_rare_tokens
            if len(set(question_tokens) & _sentence_rare_tokens) > 0:
                rare_in_question += 1
                either = True
            else:
                both = False
            _query_rare_tokens = example.query_vocab.rare_tokens if query_rare_tokens is None else query_rare_tokens
            if len(set(query_tokens) & _query_rare_tokens) > 0:
                either = True
                rare_in_query += 1
            else:
                both = False
            if both:
                rare_in_both += 1
            if either:
                rare_in_either += 1
        return rare_in_question / total, rare_in_query/total, rare_in_both/total, rare_in_either/total
    print("RARE STATS:::")
    print("training data:")
    ris, riq, rib, rie = get_rare_portions(ds.data["train"])
    print(f"\t In question: {ris} \n\t In query: {riq} \n\t In both: {rib} \n\t In either: {rie}")
    print("test data:")
    ris, riq, rib, rie = get_rare_portions(ds.data["test"])
    print(f"\t In question: {ris} \n\t In query: {riq} \n\t In both: {rib} \n\t In either: {rie}")
    return


def tensor2tree(x, D:Vocab=None):
    # x: 1D int tensor
    x = list(x.detach().cpu().numpy())
    x = [D(xe) for xe in x]
    x = [xe for xe in x if xe != D.padtoken]

    # find first @END@ and cut off
    parentheses_balance = 0
    for i in range(len(x)):
        if x[i] == D.endtoken:
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


def split_tokenizer(x):
    x = x.replace("?", " ?"). \
        replace(".", " ."). \
        replace(",", " ,"). \
        replace("'", " '")
    x = re.sub("\s+", " ", x)
    return x.lower().split()


def run(lr=0.001,
        batsize=20,
        epochs=70,
        embdim=128,
        encdim=400,
        numlayers=1,
        beamsize=5,
        dropout=.5,
        wreg=1e-10,
        cuda=False,
        gpu=0,
        minfreq=2,
        gradnorm=3.,
        smoothing=0.1,
        cosine_restarts=1.,
        seed=123456,
        ):
    localargs = locals().copy()
    print(locals())
    torch.manual_seed(seed)
    np.random.seed(seed)
    tt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    ds = GeoDatasetRank()
    print(f"max lens: {ds.maxlen_input} (input) and {ds.maxlen_output} (output)")
    tt.tock("data loaded")

    # do_rare_stats(ds)

    # model = TreeRankModel(embdim=embdim, hdim=encdim, dropout=dropout, numlayers=numlayers,
    #                          sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder)
    #
    model = ParikhRankModel(embdim=encdim, dropout=dropout,
                            sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder)

    # sentence_rare_tokens = set([ds.sentence_encoder.vocab(i) for i in model.inp_emb.rare_token_ids])
    # do_rare_stats(ds, sentence_rare_tokens=sentence_rare_tokens)
    ranker = Ranker(model, eval=[BCELoss(mode="logits", smoothing=smoothing)],
                           evalseq=[SeqAccuracies(),
                                    TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab), orderless={"and", "or"})])

    losses = make_array_of_metrics("loss", "seq_acc", "tree_acc")
    vlosses = make_array_of_metrics("seq_acc", "tree_acc")

    # 4. define optim
    # optim = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wreg)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wreg)

    # lr schedule
    if cosine_restarts >= 0:
        # t_max = epochs * len(train_dl)
        t_max = epochs
        print(f"Total number of updates: {t_max}")
        lr_schedule = q.WarmupCosineWithHardRestartsSchedule(optim, 0, t_max, cycles=cosine_restarts)
        reduce_lr = [lambda: lr_schedule.step()]
    else:
        reduce_lr = []

    # 6. define training function
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), gradnorm)
    # clipgradnorm = lambda: None
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=ranker, dataloader=ds.dataloader("train", batsize), optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=reduce_lr)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=ranker, dataloader=ds.dataloader("test", batsize), losses=vlosses, device=device)

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")

    # testing
    tt.tick("testing")
    testresults = q.test_epoch(model=ranker, dataloader=ds.dataloader("test", batsize), losses=vlosses, device=device)
    print("validation test results: ", testresults)
    tt.tock("tested")
    tt.tick("testing")
    testresults = q.test_epoch(model=ranker, dataloader=ds.dataloader("test", batsize), losses=vlosses, device=device)
    print("test results: ", testresults)
    tt.tock("tested")

    # save model?
    tosave = input("Save this model? 'y(es)'=Yes, <int>=overwrite previous, otherwise=No) \n>")
    # if True:
    #     overwrite = None
    if tosave.lower() == "y" or tosave.lower() == "yes" or re.match("\d+", tosave.lower()):
        overwrite = int(tosave) if re.match("\d+", tosave) else None
        p = q.save_run(model, localargs, filepath=__file__, overwrite=overwrite)
        q.save_dataset(ds, p)
        _model, _localargs = q.load_run(p)
        _ds = q.load_dataset(p)

        _freedecoder = BeamDecoder(_model, maxtime=100, beamsize=beamsize, copy_deep=True,
                                  eval=[SeqAccuracies()],
                                  eval_beam=[TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
                                                          orderless={"op:and", "SW:concat"})])

        # testing
        tt.tick("testing reloaded")
        _testresults = q.test_epoch(model=_freedecoder, dataloader=_ds.dataloader("test", batsize),
                                    losses=beamlosses, device=device)
        print(_testresults)
        tt.tock("tested")

        # save predictions
        _, testpreds = q.eval_loop(_freedecoder, ds.dataloader("test", batsize=batsize, shuffle=False), device=device)
        testout = get_outputs_for_save(testpreds)
        _, trainpreds = q.eval_loop(_freedecoder, ds.dataloader("train", batsize=batsize, shuffle=False), device=device)
        trainout = get_outputs_for_save(trainpreds)

        with open(os.path.join(p, "trainpreds.json"), "w") as f:
            ujson.dump(trainout, f)

        with open(os.path.join(p, "testpreds.json"), "w") as f:
            ujson.dump(testout, f)


def get_outputs_for_save(states):   # list of beam states
    ret = []
    for beamstate in states:
        for i in range(len(beamstate)):
            example = get_output_for_example(beamstate[i])
            ret.append(example)
    return ret


def get_output_for_example(x):
    sentence_vocab = x.bstates.get(0).sentence_vocab
    query_vocab = x.bstates.get(0).query_vocab

    sentence = sentence_vocab.tostr(x.bstates.get(0).inp_tensor[0], return_tokens=True)
    gold = query_vocab.tostr(x.bstates.get(0).gold_tensor[0], return_tokens=True)

    cands = []

    for bstate in x.bstates._list:
        cand = query_vocab.tostr(bstate.followed_actions[0], return_tokens=True)
        alignments = []
        align_entropies = []

        for i in range(len(cand)):
            att = bstate.stored_attentions[0, i]
            entropy = - (att.clamp_min(1e-6).log() * att).sum()
            _, amax = att.max(-1)
            alignments.append(amax.cpu().item())
            align_entropies.append(entropy.cpu().item())


        cands.append({
            "tokens": cand,
            "alignments": alignments,
            "align_entropies": align_entropies
        })

    ret = {
        "sentence": sentence,
        "gold": gold,
        "candidates": cands
    }
    return ret


if __name__ == '__main__':
    # try_basic_query_tokenizer()
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)
    # q.argprun(run_rerank)
    # try_tree_gru_encoder()