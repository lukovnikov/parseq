from copy import deepcopy, copy
from functools import partial
from timeit import timeit
from typing import List

import numpy as np
from nltk import Tree
import torch


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



if __name__ == '__main__':
    # tst_supervise_edges_line()
    # tst_supervise_edges_multichild()
    # tst_supervise_edges_simple_ambig()
    # tst_supervise_edges_abbaa()
    N = 10
    # res = timeit(partial(tst_supervise_edges_ones, 8), number=N)
    res = timeit(partial(tst_supervise_edges_interleave, 1, 8), number=N)
    print("results")
    print(res/N)