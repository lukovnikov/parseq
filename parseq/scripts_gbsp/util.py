import json
import os
import time
from copy import deepcopy, copy
from functools import partial
from timeit import timeit
from typing import List

import numpy as np
from nltk import Tree
import torch
from parseq.datasets import Dataset, autocollate
from parseq.grammar import lisp_to_tree, prolog_to_tree, tree_size
from parseq.vocab import Vocab
from torch.utils.data import DataLoader


def lisp_str_to_tree(s:str):
    ret = prolog_to_tree(s)
    ret = postprocess(ret)
    return ret


def postprocess(x):
    if isinstance(x, str):
        # make a string node
        xs = x[1:-1].split(" ")
        xs = [Tree("_" + xe + "_", []) for xe in xs]
        ret = Tree("@STR@", xs)
        return ret
    else:
        x[:] = [postprocess(child) for child in x[:]]
        return x


class GeoSplitsLoader(object):
    def __init__(self, p="../../datasets/spanbased_datasets/geo/funql", **kw):
        super(GeoSplitsLoader, self).__init__(**kw)
        self.p = p

    def load(self, split=None):     # None or "orig" will load original split, other options: "len", "template"
        if split is None or split == "orig" or split == "":
            split = ""
        else:
            split = "_" + split
        d = {"train": [], "dev": [], "test": []}
        examples = []
        for a in d:
            p = a + split + ".json"
            with open(os.path.join(self.p, p)) as f:
                for line in f.readlines():
                    jl = json.loads(line)
                    x = jl["question"]
                    y = lisp_str_to_tree(jl["program"])
                    d[a].append((x, y))
                    examples.append((x, y, a))
        return Dataset(examples)


def collect_toks_from_tree(x:Tree, orderless=None):
    ret = {x.label()}
    retedge = set()
    if orderless is None:
        orderless = set()
    for i, child in enumerate(x):
        _ret, _retedge = collect_toks_from_tree(child, orderless=orderless)
        ret |= _ret
        retedge |= _retedge
        if x.label() in orderless:
            retedge.add(":child")
        else:
            retedge.add(f":child-{i+1}")
    return ret, retedge


class Tree2Struct(object):
    def __init__(self, outvocab, edgevocab, orderless=None, **kw):
        super(Tree2Struct, self).__init__(**kw)
        self.outvocab, self.edgevocab = outvocab, edgevocab
        self.orderless = set() if orderless is None else orderless

    def maptree(self, y:Tree):
        ret = Tree(self.outvocab[y.label()], [self.maptree(child) for child in y])
        return ret

    def __call__(self, y):
        labels, _ = collect_toks_from_tree(y)
        labels = [self.outvocab[label] for label in labels]
        tree = self.maptree(y)
        return (labels, tree)


def load_geo(split="orig", inptokenizer="vanilla", batsize=16):
    orderless = {"intersection"}
    dl = GeoSplitsLoader()
    ds = dl.load(split)

    inpvocab = Vocab()
    outvocab = Vocab()
    outvocab.add_token("@NONE@", seen=1e6)
    outedgevocab = Vocab()

    for example in ds.examples:
        x, y, tvx = example
        xtoks = x.split(" ")
        for xtok in xtoks:
            inpvocab.add_token(xtok, seen=tvx=="train")
        ytoks, edgetoks = collect_toks_from_tree(y, orderless=orderless)
        for ytok in ytoks:
            outvocab.add_token(ytok, seen=tvx=="train")
        for edgetok in edgetoks:
            outedgevocab.add_token(edgetok, seen=tvx=="train")

    inpvocab.finalize()
    outvocab.finalize()
    outedgevocab.finalize()

    t2t = Tree2Struct(outvocab, outedgevocab, orderless=orderless)
    ds = ds.map(lambda x: (x[0],) + t2t(x[1]) + x[2:])
    if inptokenizer == "vanilla":
        ds.map(lambda x: (torch.tensor([inpvocab[xe] for xe in x[0].split(" ")]).long(),)
                         + x[1:])
    else:
        ds.map(lambda x: (inptokenizer(x[0]),) + x[1:])
        inpvocab = inptokenizer

    tds = ds.filter(lambda x: x[-1] == "train").cache(True)
    vds = ds.filter(lambda x: x[-1] == "dev").cache(True)
    xds = ds.filter(lambda x: x[-1] == "test").cache(True)

    tdl = DataLoader(tds, batsize, shuffle=True, collate_fn=autocollate)
    vdl = DataLoader(vds, batsize, shuffle=False, collate_fn=autocollate)
    xdl = DataLoader(xds, batsize, shuffle=False, collate_fn=autocollate)

    return (tdl, vdl, xdl), (inpvocab, outvocab, outedgevocab)


def supervise_edges(tree:Tree, scores:torch.Tensor, labels:torch.LongTensor):
    """
    Computes supervision for edge structure decisions.
    Computes best possible implementation of a given tree based on scores computed for potential edges.
    :param tree: tree to be constructed. Must use integer ids on nodes
    :param scores: scores for potential arcs 1st dimension=source, 2nd=destination
    :param labels: what label every node in the scores matrix has
    :return:
    """
    # current implementation is exhaustive search
    assert labels.dim() == 1
    assert labels.dtype == torch.long
    assert scores.dim() == 2
    assert scores.dtype == torch.float

    # build dictionary mapping labels to positions in the matrix
    label2pos = {}
    labels = list(labels.cpu().numpy())
    for i, k in enumerate(labels):
        if k not in label2pos:
            label2pos[k] = set()
        label2pos[k].add(i)

    print(label2pos)

    bestscore, bestimpl, _label2pos, bound = _supervise_edges_rec(None, tree, scores, label2pos)

    out = torch.zeros_like(scores)
    for (a, b) in bestimpl:
        out[a, b] = 1
    return out, bestscore


def _supervise_edges_rec(frompos, tree, scores, label2pos, fromscore=0, bound=np.infty):
    # search one level
    bestscore = np.infty
    bestimpl = None
    bestlabelpos = None
    treelabel = tree.label()
    positions = label2pos[treelabel]
    if frompos is not None:     # then sort positions by increasing cost to do greedy search
        sortscores = [scores[frompos, pos].item() for pos in positions]
        sortedpositions = sorted(zip(sortscores, positions), key=lambda x: x[0])
        positions = [p[1] for p in sortedpositions]
    broken = True
    for pos in positions:     # explore every possible assignment of this tree's label
        _label2pos = copy(label2pos)
        _label2pos[treelabel] = _label2pos[treelabel] - {pos,}
        if len(_label2pos[treelabel]) == 0:
            del _label2pos[treelabel]
        score = fromscore
        impl = set()
        if frompos is not None:  # not root
            score += scores[frompos, pos].item()
            impl = {(frompos, pos)}
        if score > bound:
            continue
        else:
            broken = False
        for child in tree:                  # explore every possible implementation of the children
            score, __bestimpl, _label2pos, bound = _supervise_edges_rec(pos, child, scores, _label2pos, fromscore=score, bound=bound)
            if _label2pos is not None and len(_label2pos) == 0:        # tree complete
                bound = score if score < bound else bound
            if __bestimpl is not None:
                impl |= __bestimpl
        if score < bestscore:
            bestscore = score
            bestimpl = impl
            bestlabelpos = _label2pos

    return bestscore, bestimpl, bestlabelpos, bound


def treemaplabel(tree:Tree, fn=lambda x: x):
    tree.set_label(fn(tree.label()))
    for child in tree:
        treemaplabel(child, fn)


def tst_supervise_edges_line(N=5):
    tree = Tree.fromstring("(1 (2 (3 (4 (5)))))")
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(5, 5)
    labels = torch.tensor([1,2,3,4,5])

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print(torch.diag(weights, 1).sum().item())


def tst_supervise_edges_multichild(N=5):
    tree = Tree.fromstring("(1 (2 (3 (4) (5))))")
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(5, 5)
    labels = torch.tensor([1,2,3,4,5])

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def tst_supervise_edges_simple_ambig(N=5):
    tree = Tree.fromstring("(1 (2 (3 (4) (4))))")
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(5, 5)
    labels = torch.tensor([1,2,3,4,4])

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def tst_supervise_edges_abbaa(N=5):
    tree = Tree.fromstring("(1 (2 (2 (1 (1)))))")
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(5, 5)
    labels = torch.tensor([1, 1, 1, 2, 2])

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def tst_supervise_edges_interleave_20():
    tree = Tree.fromstring("(1 (1 (2 (2 (1 (1 (2 (2 (1 (1 (2 (2 (1 (1 (2 (2 (1 (1 (2 (2))))))))))))))))))))")
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(20, 20)
    labels = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def tst_supervise_edges_interleave_10():
    tree = Tree.fromstring("(1 (1 (2 (2 (1 (1 (2 (2 (1 (1))))))))))")
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(10, 10)
    labels = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def tst_supervise_edges_interleave_12():
    tree = Tree.fromstring("(1 (1 (2 (2 (1 (1 (2 (2 (1 (1 (2 (2 ))))))))))))")
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(12, 12)
    labels = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def tst_supervise_edges_ones(N=8):
    s = "(1 " * N + ")" * N
    tree = Tree.fromstring(s)
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(N, N)
    labels = torch.ones(N).long()

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def tst_supervise_edges_interleave(num=2, reps=6):
    s = " ".join([f"({i} ({i} " for i in range(1, num+1)])
    print(s)
    s = s * (reps//2) + " ".join([")" * (reps * num)])
    print(s)

    tree = Tree.fromstring(s)
    treemaplabel(tree, lambda x: int(x))
    print(tree)
    weights = torch.rand(num*reps, num*reps)
    labels = [[i for i in range(1, num+1)] for _ in range(reps)]
    labels = [i for ii in labels for i in ii]
    labels = torch.tensor(labels).long()
    print(labels)

    out, score = supervise_edges(tree, weights, labels)
    print(out)
    print(weights)
    print(score)
    print((weights*out).sum().item())


def relabel_tree(labels, tree):
    labelD = {x: i for (i, x) in enumerate(labels)}
    return _relabel_tree(tree, labelD)


def _relabel_tree(tree, labelD):
    children = [_relabel_tree(child, labelD) for child in tree]
    tree = Tree(labelD[tree.label()], children)
    return tree


def build_label_multiplicities(x:Tree):
    labels = {x.label(): 1}
    for child in x:
        _labels = build_label_multiplicities(child)
        for k, v in _labels.items():
            if k not in labels:
                labels[k] = v
            else:
                labels[k] += v
    return labels


def tst_geo_supervise_edges():
    # load geo length based split

    dls, vocabs = load_geo("len")
    print(len(dls))


    maxambis = []
    start = time.time()
    for dl in dls:
        for batch in dl:
            # print(batch)
            examples = list(zip(*batch))
            # print(examples[0])
            for example in examples:
                labels, tree = example[1], example[2]
                label_multiplicities = build_label_multiplicities(tree)
                maxambi = 1
                for k, v in label_multiplicities.items():
                    maxambi = max(maxambi, v)
                if maxambi > 1:
                    print(f"ambiguous! {maxambi}")
                    print(tree)
                    maxambis.append(maxambi)
                treesize = tree_size(tree)
                weights = torch.ones(treesize, treesize)
                _labels = []
                _labels = [label for label in labels for _ in range(label_multiplicities[label])]
                _labels = torch.tensor(_labels)
                out, score = supervise_edges(tree, weights, _labels)
    end = time.time()
    print(f"Done in {end-start} seconds.")
    print(f"Number of ambi: {len(maxambis)}, max ambi: {max(maxambis)}")
    print(vocabs[1].D)






if __name__ == '__main__':
    # tst_supervise_edges_line()
    # tst_supervise_edges_multichild()
    # tst_supervise_edges_simple_ambig()

    # tst_supervise_edges_abbaa()

    tst_geo_supervise_edges()

    # N = 10
    # # res = timeit(partial(tst_supervise_edges_ones, 8), number=N)
    # res = timeit(partial(tst_supervise_edges_interleave, 1, 8), number=N)
    # print("results")
    # print(res/N)