import itertools
import json
import math
from copy import copy
from typing import List

import torch
import numpy as np
import re

from nltk import Tree
from tqdm import tqdm

from parseq.datasets import CFQDatasetLoader
from parseq.grammar import taglisp_to_tree


def find_subgraphs_sparql(x:Tree, minsize=1, maxsize=4):
    assert x.label() == "@R@"
    assert x[0].label() == "@QUERY"
    selectclause = x[0][0]
    whereclause = x[0][1]
    # print(whereclause)
    clauses = whereclause[:] + [selectclause]
    clauses[:] = sorted(clauses, key=lambda x: str(x))
    # print(whereclause)
    combos = []
    for l in range(minsize, min(maxsize+1, len(clauses))):
        for combo in itertools.combinations(clauses, l):
            combos.append(str(combo))
    # for i in range(len(whereclause)):
    #     for j in range(len(whereclause)):
    #         if j > i:
    #             wherecombos.append((whereclause[i], whereclause[j]))
    return combos


class FrequencyDistribution():
    def __init__(self):
        super().__init__()
        self._counts = {}
        self.total = 0

    def __getitem__(self, item):
        if item not in self._counts:
            return 0
        else:
            return self._counts[item]

    def __setitem__(self, key, value):
        oldcount = 0
        if key in self._counts:
            oldcount = self._counts[key]
        self._counts[key] = value
        self.total = self.total - oldcount + value

    def __call__(self, key):
        return self.__getitem__(key) / self.total

    def tocounts(self):
        return copy(self._counts)

    def tofreqs(self):
        return {k: v/self.total for k, v in self._counts.items()}

    def keys(self):
        return self._counts.keys()

    def items(self):
        return self._counts.items()

    def elements(self):
        return list(self._counts.keys())

    def __add__(self, other):
        ret = FrequencyDistribution()
        for k in set(self._counts.keys()) | set(other._counts.keys()):
            ret[k] = self[k] + other[k]
        return ret

    def __sub__(self, other):
        ret = FrequencyDistribution()
        for k in set(self._counts.keys()) | set(other._counts.keys()):
            ret[k] = self[k] - other[k]
        return ret

    def __len__(self):
        return len(self._counts)

    def __contains__(self, item):
        return item in self._counts

    @staticmethod
    def compute_chernoff_coeff(self, other, alpha=0.5, weights=None):
        acc = 0
        dist1, dist2 = self, other
        if isinstance(dist1, FrequencyDistribution):
            dist1 = dist1.tofreqs()
        if isinstance(dist2, FrequencyDistribution):
            dist2 = dist2.tofreqs()
        for k in dist1:
            v1 = dist1[k]
            if k not in dist2:
                v2 = 0
            else:
                v2 = dist2[k]
            contrib = math.pow(v1, alpha) * math.pow(v2, 1 - alpha)
            if weights is not None:
                if k in weights:
                    contrib = contrib * weights[k]
            acc += contrib
        return acc

    def entropy(self):
        entr = 0
        for k in self._counts:
            entr = entr - self(k) * math.log(self(k))
        return entr


def compute_entropy_minus(fd:FrequencyDistribution, xs, entropy=None):
    """
    :param fd:       Original frequencies
    :param entropy:  The original entropy of 'fd'. If not specified, entropy is computed from scratch.
    :param xs:       A multiset (e.g. list, tuple) of elements in fd to subtract from the entropy
    :return:
    """
    pass


class DivergenceComputer():

    orderless = {"@QUERY", "@AND", "@OR", "@WHERE"}
    variablesize = {"@AND", "@OR", "@WHERE"}

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def _extract_atom(self, x:Tree):
        if x.label() in self.orderless:
            if x.label() in self.variablesize:
                childstr = "ARG*"
            else:
                childstr = " ".join(["ARG" for _ in range(len(x))])
        else:
            childstr = " ".join([f"ARG{i+1}" for i in range(len(x))])
        ret = f"({x.label()} {childstr})"
        return ret

    def extract_atoms(self, x:Tree):
        ret = []
        for child in x:
            ret = ret + self.extract_atoms(child)
        ret.append(self._extract_atom(x))
        return ret

    def extract_atom_dist(self, x:Tree):
        atoms = self.extract_atoms(x)
        fd = FrequencyDistribution()
        for atom in atoms:
            fd[atom] += 1
        return fd

    def extract_compounds(self, x:Tree):
        """ This method extracts simple compounds that consist of two elements: parent and child """
        compounds = find_subgraphs_sparql(x, minsize=2, maxsize=3)
        return compounds
        # if len(x) == 0:     # leaf
        #     retcomps = []
        #     retatom = self._extract_atom(x)
        #     retcomps += [retatom, "<>"]
        #     return retcomps, retatom
        # else:
        #     compounds = []
        #     xstr = self._extract_atom(x)
        #     childgroups = []
        #     for i, child in enumerate(x):
        #         childcomps, childatom = self.extract_compounds(child)
        #         compounds = compounds + childcomps
        #         if x.label() in self.orderless:
        #             connectstr = "ARG"
        #         else:
        #             connectstr = f"ARG-{i}"
        #         for childcomp in childcomps:
        #             childgroups.append((childcomp, connectstr))
        #     # TODO
        #     raise NotImplemented()
        #         # compounds.append(f"{xstr} - {connectstr} -> {childatom}")
        #     return compounds, xstr

    def extract_compound_dist(self, x:Tree):
        comps = self.extract_compounds(x)
        fd = FrequencyDistribution()
        for comp in comps:
            fd[comp] += 1
        return fd

    @staticmethod
    def compute_chernoff_coeff(dist1:FrequencyDistribution, dist2:FrequencyDistribution, alpha=0.5, weights=None):
        return FrequencyDistribution.compute_chernoff_coeff(dist1, dist2, alpha=alpha, weights=weights)

    def compute_atom_distribution(self, l:List):
        fd = FrequencyDistribution()
        for example in tqdm(l):
            t = example[1]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            fd += self.extract_atom_dist(t)
        return fd

    def compute_atom_distributions(self, ds):
        atomses = dict()
        for example in tqdm(ds):
            t = example[1]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            if example[2] not in atomses:
                atomses[example[2]] = FrequencyDistribution()
            atomses[example[2]] += self.extract_atom_dist(t)
        return atomses

    def compute_compound_distribution(self, l:List):
        fd = FrequencyDistribution()
        for example in tqdm(l):
            t = example[1]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            for comp in self.extract_compounds(t):
                fd[comp] += 1
        return fd

    def compute_compound_distributions(self, ds):
        compoundses = dict()
        for example in tqdm(ds):
            t = example[1]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            if example[2] not in compoundses:
                compoundses[example[2]] = FrequencyDistribution()
            for comp in self.extract_compounds(t):
                compoundses[example[2]][comp] += 1
        return compoundses

    def _compute_atom_divergences(self, dists):
        divergences = dict()
        for subsetname in dists:
            for subsetname2 in dists:
                if self.verbose:
                    print(f"computing divergence between {subsetname} and {subsetname2}")
                divergences[subsetname + "-" + subsetname2] = 1 - self.compute_chernoff_coeff(dists[subsetname], dists[subsetname2], 0.5)
        return divergences

    def compute_atom_divergences(self, ds):
        dists = self.compute_atom_distributions(ds)
        return self._compute_atom_divergences(dists)

    def _compute_compound_divergences(self, dists):
        divergences = dict()
        for subsetname in dists:
            for subsetname2 in dists:
                if self.verbose:
                    print(f"computing divergence between {subsetname} and {subsetname2}")
                divergences[subsetname + "-" + subsetname2] = 1 - self.compute_chernoff_coeff(dists[subsetname], dists[subsetname2], 0.1)
        return divergences

    def compute_compound_divergences(self, ds):
        dists = self.compute_compound_distributions(ds)
        return self._compute_compound_divergences(dists)


def try_cfq():
    ds = CFQDatasetLoader().load("mcd1/modent")
    print(ds[0])
    dc = DivergenceComputer()
    atomdists = dc.compute_atom_distributions(ds)
    print(atomdists)
    print(json.dumps(dc._compute_atom_divergences(atomdists)))


if __name__ == '__main__':
    try_cfq()