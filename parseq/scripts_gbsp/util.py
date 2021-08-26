import json
import os
import time
from copy import deepcopy, copy
from functools import partial
from multiprocessing import Pool
from timeit import timeit
from typing import List

import numpy as np
import psutil
import ray
from nltk import Tree
import torch
from scipy.optimize import linear_sum_assignment

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


def supervise_tokens_batch(targets:List[List[int]], scores:torch.Tensor, inpmask:torch.Tensor, none_id=1):
    asses = []
    for targetse, scorese, inpmaske in zip(targets, scores.unbind(0), inpmask.unbind(0)):
        # append as many NONEs to targetse to match non-masked input elements
        assert inpmaske.sum().item() >= len(targetse)
        targetse += [none_id] * (inpmaske.sum().item() - len(targetse))
        ass = supervise_tokens(targetse, scorese)
        asses.append((ass,))
    ret = autocollate(asses)
    return ret


def supervise_tokens(targets:List[int], scores:torch.Tensor):
    assert scores.dim() == 2        # (seqlen, numclass)
    realsize = len(targets)
    targets = torch.tensor(targets).to(scores.device)
    _scores = scores[:realsize]     # (numtargets, numclass)
    _scores = _scores[:, targets]   # (numtargets, numtargets)
    _scores = _scores.detach().cpu().numpy()
    rows, cols = linear_sum_assignment(-_scores)
    assert np.allclose(rows, np.arange(0, rows.shape[0]))
    outs = targets[torch.tensor(cols)]
    return outs


def supervise_edges_batch(trees:List[Tree], scores:torch.Tensor, labels:torch.LongTensor, pool=None):
    """
    Parallel computation of batch supervision
    :param trees:
    :param scores:
    :param labels:
    :return:
    """
    # # parallel implementation with ray
    # args = list(zip(trees, scores.detach().cpu().unbind(0), labels.detach().cpu().unbind(0)))
    # futures = [supervise_edges_ray.remote(*argses) for argses in args]
    # ret = ray.get(futures)
    # outs, masks, scores = zip(*ret)
    # outs = torch.stack(outs, 0)
    # masks = torch.stack(masks, 0)
    # return outs, masks, scores

    # normal implementations
    # with Pool() as pool:
    # pool = Pool(4)
    args = list(zip(trees, scores.detach().cpu().unbind(0), labels.detach().cpu().unbind(0)))
    if pool is not None:
        ret = pool.starmap(supervise_edges, args)
    else:
        ret = [supervise_edges(*argses) for argses in args]
    outs, masks, scores = zip(*ret)
    outs = torch.stack(outs, 0)
    masks = torch.stack(masks, 0)
    return outs, masks, scores


@ray.remote
def supervise_edges_ray(tree:Tree, scores:torch.Tensor, labels:torch.LongTensor):
    return supervise_edges(tree, scores, labels)


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
    origsize = scores.size(0)
    treesize = tree_size(tree)
    _scores = scores[:treesize, :treesize]
    _labels = labels[:treesize]

    # build dictionary mapping labels to positions in the matrix
    label2pos = {}
    _labels = list(_labels.cpu().numpy())
    for i, k in enumerate(_labels):
        if k not in label2pos:
            label2pos[k] = set()
        label2pos[k].add(i)

    print(label2pos)

    bestscore, bestimpl, _label2pos, bound = _supervise_edges_rec(None, tree, _scores, label2pos)

    out = torch.zeros_like(scores)
    for (a, b) in bestimpl:
        out[a, b] = 1
    mask = torch.zeros_like(scores).bool()
    mask[:treesize, :treesize] = 1
    return out, mask, bestscore


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

    # numcpu = psutil.cpu_count(logical=False)
    # ray.init(num_cpus=numcpu)

    # pool = Pool(8)
    pool = None
    maxambis = []
    start = time.time()
    for dl in dls:
        for batch in dl:
            # print(batch)
            examples = list(zip(*batch))
            batsize = len(examples)
            # print(examples[0])
            treearg = []
            maxsize = 0
            labelsarg = []
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
                # weights = torch.ones(treesize, treesize)
                _labels = []
                _labels = [label for label in labels for _ in range(label_multiplicities[label])]
                _labels = torch.tensor(_labels)
                treearg.append(tree)
                labelsarg.append((_labels,))
                maxsize = max(maxsize, treesize)
            weightsarg = torch.ones(batsize, maxsize, maxsize)
            labelsarg = autocollate(labelsarg)[0]
            out, masks, score = supervise_edges_batch(treearg, weightsarg, labelsarg, pool=pool)
    end = time.time()
    print(f"Done in {end-start:.3f} seconds.")
    print(f"Number of ambi: {len(maxambis)}, max ambi: {max(maxambis)}")
    print(vocabs[1].D)


def tst_supervise_tokens():
    tokens = [1, 1, 1, 2, 2, 3]
    scores = torch.rand(10, 5)
    ass = supervise_tokens(tokens, scores)
    print(ass)


def tst_supervise_tokens_batch():
    tokens = [[2, 2, 2, 3, 3, 4, 4], [2, 3, 4], [2, 3, 3, 4, 4]]
    scores = torch.rand(3, 10, 5)
    inpmask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    asses = supervise_tokens_batch(tokens, scores, inpmask)
    print(asses)


def tst_geo_supervise_tokens():
    # load geo length based split

    dls, vocabs = load_geo("len")
    print(len(dls))

    # numcpu = psutil.cpu_count(logical=False)
    # ray.init(num_cpus=numcpu)

    # pool = Pool(8)
    pool = None
    maxambis = []
    start = time.time()
    for dl in dls:
        for batch in dl:
            # print(batch)
            examples = list(zip(*batch))
            batsize = len(examples)
            inpmask = []
            labelsarg = []
            alllabels = set()
            for example in examples:
                question, labels = example[0], example[1]
                alllabels |= set(labels)
                labelsarg.append(sorted(labels[:]))
                inpmask.append((torch.ones(len(question.split(" ")) * 2, dtype=torch.long),))
            inpmask = autocollate(inpmask)[0]
            assert 1 not in alllabels
            weights = torch.rand(batsize, inpmask.size(1), max(alllabels) + 1)
            asses = supervise_tokens_batch(labelsarg, weights, inpmask, vocabs[1].D["@NONE@"])
            print(asses)
    end = time.time()
    print(f"Done in {end-start:.3f} seconds.")
    print(vocabs[1].D)



if __name__ == '__main__':
    # tst_supervise_edges_line()
    # tst_supervise_edges_multichild()
    # tst_supervise_edges_simple_ambig()

    # tst_supervise_edges_abbaa()

    # tst_geo_supervise_edges()
    # tst_supervise_tokens_batch()
    tst_geo_supervise_tokens()
    # N = 10
    # # res = timeit(partial(tst_supervise_edges_ones, 8), number=N)
    # res = timeit(partial(tst_supervise_edges_interleave, 1, 8), number=N)
    # print("results")
    # print(res/N)