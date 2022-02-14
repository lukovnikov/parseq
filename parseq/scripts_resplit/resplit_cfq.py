import random
from numbers import Number

import itertools
import json
import math
from copy import copy
from typing import List, Tuple

import torch
import numpy as np
import re

from nltk import Tree
from tqdm import tqdm

from parseq.datasets import CFQDatasetLoader
from parseq.grammar import taglisp_to_tree, tree_size


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
        keys = set(dist1.keys()) & set(dist2.keys())
        for k in keys:
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

    def _check_numeric(self):
        # check if all keys are numbers:
        allnumbers = True
        for k in self._counts:
            if not isinstance(k, Number):
                allnumbers = False
                break
        if not allnumbers:
            raise Exception("Can not compute average for non-numerical keys.")

    def average(self):
        self._check_numeric()
        acc = 0
        for k in self._counts:
            acc += k * self(k)
        return acc


class DivergenceComputer():

    IDINDEX = 0
    NLINDEX = 1
    LFINDEX = 2
    SPLITINDEX = 3

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
        compounds = find_subgraphs_sparql(x, minsize=1, maxsize=3)
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

    def compute_size_distribution(self, l:List):
        fd = FrequencyDistribution()
        for example in tqdm(l):
            t = example[self.LFINDEX]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            s = tree_size(t)
            fd[s] += 1
        return fd

    def compute_atom_distribution(self, l:List):
        fd = FrequencyDistribution()
        for example in tqdm(l):
            t = example[self.LFINDEX]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            fd += self.extract_atom_dist(t)
        return fd

    def compute_atom_distributions(self, ds):
        atomses = dict()
        for example in tqdm(ds):
            t = example[self.LFINDEX]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            if example[self.SPLITINDEX] not in atomses:
                atomses[example[self.SPLITINDEX]] = FrequencyDistribution()
            atomses[example[self.SPLITINDEX]] += self.extract_atom_dist(t)
        return atomses

    def compute_compound_distribution(self, l:List):
        fd = FrequencyDistribution()
        for example in tqdm(l):
            t = example[self.LFINDEX]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            for comp in self.extract_compounds(t):
                fd[comp] += 1
        return fd

    def compute_compound_distributions(self, ds):
        compoundses = dict()
        for example in tqdm(ds):
            t = example[self.LFINDEX]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            if example[self.SPLITINDEX] not in compoundses:
                compoundses[example[self.SPLITINDEX]] = FrequencyDistribution()
            for comp in self.extract_compounds(t):
                compoundses[example[self.SPLITINDEX]][comp] += 1
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


def get_dist_similarity(x, dist):
    if isinstance(dist, FrequencyDistribution):
        dist = dist.tofreqs()
    score = 0
    for xe in x:
        score += dist[xe] if xe in dist else 0
    score = score / max(1e-6, len(x))
    return score


def filter_mcd(source: List[Tuple[str, Tree]], otheratoms, othercomps, N=10000,
               dc=None, coeffa=1, coeffb=1, coeffatom=1000):
    """
    :param source:      a list of examples from which to pick, in format (input, output)
    :param otheratoms:  atom distributions of other splits
    :param othercomps:  compound distributions of other splits
    :param N:           how many examples to retain in selection from source
    :return:   MCD selection of examples from source wrt all other distributions

    ideally, the training set shouldn't contain any compound from new selection
    and the new selection shouldn't contain any compound from test
    """
    assert len(source) > N
    assert len(otheratoms) == len(othercomps)

    trainatoms, testatoms = otheratoms
    traincomps, testcomps = othercomps
    traincomps = traincomps.tofreqs()
    testcomps = testcomps.tofreqs()

    print("creating selection")
    exstats = []
    for x in tqdm(source):
        # xatoms = dc.extract_atoms(x[dc.LFINDEX])
        xcomps = dc.extract_compounds(x[dc.LFINDEX])
        exstats.append((get_dist_similarity(xcomps, traincomps),
                        get_dist_similarity(xcomps, testcomps),
                        len(xcomps)))

    # select top-N examples with least overlap with train and test
    scores = [(i, coeffa * a + coeffb * b) for (i, (a, b, e)) in enumerate(exstats)]
    sortedscores = sorted(scores, key=lambda x: x[1])
    retids = [i for i, s in sortedscores[:N]]
    ret = []
    for i in retids:
        a = source[i]
        a = tuple(a) + ("ood2valid",)
        ret.append(a)
    return ret


def print_overlaps(dists):
    collections = dict()
    for k, v in dists.items():
        collections[k] = set(v.keys())
    ks = list(dists.keys())
    for i in range(len(ks)):
        fromname = ks[i]
        fromatoms = collections[ks[i]]
        for j in range(i, len(ks)):
            toname = ks[j]
            toatoms = collections[ks[j]]
            print(f"Overlap {fromname},{toname}: {len(fromatoms & toatoms)} / ({len(fromatoms)}, {len(toatoms)})")


def filter_traindata(traindata, testdata, newdevdata, N=10000, dc=None):
    newcompdist = dc.compute_compound_distribution(newdevdata).tofreqs()
    testdist = dc.compute_compound_distribution(testdata).tofreqs()

    scores = []
    for i, x in enumerate(tqdm(traindata)):
        xcomps = dc.extract_compounds(x[dc.LFINDEX])
        scores.append((i, get_dist_similarity(xcomps, newcompdist) - get_dist_similarity(xcomps, testdist)))

    sortedscores = sorted(scores, key=lambda x: x[dc.LFINDEX], reverse=True)
    retids = [i for i, s in sortedscores[N:]]
    ret = []
    for retid, score in retids:
        a = traindata[retid]
        a = a[:-1] + ("ood2valid",)
        ret.append(a)
    # ret = [(traindata[i][0], traindata[i][1], "newtrain") for i in retids]
    return ret


def resplit_cfq(split="mcd1", outp="../../datasets/cfq/", debug=False, N=10000):
    ds = CFQDatasetLoader().load(f"{split}/modent", validfrac=0, loadunused=True, keepids=True)

    if debug:
        ds = ds.examples
        random.shuffle(ds)
        ds = ds[:10000]
        N = 1000
    dc = DivergenceComputer()

    print("Atom distributions")
    atom_dists = dc.compute_atom_distributions(ds)
    print(json.dumps(dc._compute_atom_divergences(atom_dists), indent=3))

    print("Compound distributions")
    comp_dists = dc.compute_compound_distributions(ds)
    print(json.dumps(dc._compute_compound_divergences(comp_dists), indent=3))

    # extract unused examples
    print("Extracting unused examples")
    unused = [(ex[0], ex[1], taglisp_to_tree(ex[2])) for ex in tqdm(ds) if ex[-1] == "unused"]
    len(unused)

    # filtering mcd
    print("selecting examples for second ood set")
    newsplit = filter_mcd(unused, [atom_dists["train"], atom_dists["test"]], [comp_dists["train"], comp_dists["test"]],
                          N=N, dc=dc)

    # print("debug", newsplit[0])

    print("Computing distributions of new split")
    newsplit_atomdist = dc.compute_atom_distribution(newsplit)
    newsplit_compdist = dc.compute_compound_distribution(newsplit)
    _atom_dists = {k: v for k, v in atom_dists.items()}
    _atom_dists["newsplit"] = newsplit_atomdist
    _comp_dists = {k: v for k, v in comp_dists.items()}
    _comp_dists["newsplit"] = newsplit_compdist
    print(json.dumps(dc._compute_atom_divergences(_atom_dists), indent=3))
    print(json.dumps(dc._compute_compound_divergences(_comp_dists), indent=3))

    # for k in set(_atom_dists["train"].elements()) | set(_atom_dists["test"].elements()) | set(
    #         _atom_dists["newsplit"].elements()):
    #     print(f"{k}: {_atom_dists['train'](k):.5f} - {_atom_dists['test'](k):.5f} - {_atom_dists['newsplit'](k):.5f}")

    print_overlaps(_atom_dists)
    print_overlaps(_comp_dists)

    # extract ids for new json and check them
    ret = extract_example_ids(ds, newsplit, newtrain=None)
    with open(f"{outp}{split}new.json", "w") as f:
        json.dump(ret, f)
    return ret


    # BELOW CODE CHANGES TRAINING DATA
    # traindata = [(ex[0], taglisp_to_tree(ex[1])) for ex in tqdm(ds) if ex[2] == "train"]
    # testdata = [(ex[0], taglisp_to_tree(ex[1])) for ex in tqdm(ds) if ex[2] == "test"]
    # newdevdata = [(ex[0], ex[1]) for ex in tqdm(newsplit)]
    # newtrain = filter_traindata(traindata, testdata, newdevdata)
    #
    # # %%
    #
    # newtrain_atomdist = dc.compute_atom_distribution(newtrain)
    # newtrain_compdist = dc.compute_compound_distribution(newtrain)
    # __atom_dists = {k: v for k, v in _atom_dists.items()}
    # __atom_dists["newtrain"] = newtrain_atomdist
    # __comp_dists = {k: v for k, v in _comp_dists.items()}
    # __comp_dists["newtrain"] = newtrain_compdist
    # print(json.dumps(dc._compute_atom_divergences(__atom_dists), indent=3))
    # print(json.dumps(dc._compute_compound_divergences(__comp_dists), indent=3))
    #
    # # %%
    #
    # print_overlaps(__atom_dists)
    # print_overlaps(__comp_dists)


def extract_example_ids(ds, newood=None, newtrain=None):
    ret = {}
    for (i, nl, lf, split) in ds:
        if split not in ret:
            ret[split] = []
        ret[split].append(i)

    if newood is not None:
        ret["ood2valid"] = []
        for (i, nl, lf, split) in newood:
            ret["ood2valid"].append(i)

    if newtrain is not None:
        ret["train"] = []
        for (i, nl, lf, split) in newtrain:
            ret["train"].append(i)

    return ret


def try_cfq():
    ds = CFQDatasetLoader().load("mcd1/modent")
    print(ds[0])
    dc = DivergenceComputer()
    atomdists = dc.compute_atom_distributions(ds)
    print(atomdists)
    print(json.dumps(dc._compute_atom_divergences(atomdists)))


if __name__ == '__main__':
    resplit_cfq("mcd2", debug=False)