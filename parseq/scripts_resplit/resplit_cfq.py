import itertools
import math

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
    whereclause[:] = sorted(whereclause, key=lambda x: str(x))
    # print(whereclause)
    wherecombos = []
    for l in range(minsize, min(maxsize+1, len(whereclause))):
        for combo in itertools.combinations(whereclause, l):
            wherecombos.append(str(combo))
    # for i in range(len(whereclause)):
    #     for j in range(len(whereclause)):
    #         if j > i:
    #             wherecombos.append((whereclause[i], whereclause[j]))
    return wherecombos


class DivergenceComputer():

    orderless = {"@QUERY", "@AND", "@OR", "@WHERE"}
    variablesize = {"@AND", "@OR", "@WHERE"}

    def __init__(self, verbose=False):
        super(DivergenceComputer, self).__init__()
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

    def extract_coocs(self, x:Tree):
        """ This method extracts co-occurrences across the entire tree """
        atoms = self.extract_atoms(x)
        coocs = []
        for i, atom in enumerate(atoms):
            for j, atom2 in enumerate(atoms):
                if i != j:
                    coocs.append(f"{atom},{atom2}")
        return coocs

    @staticmethod
    def compute_chernoff_coeff(dist1, dist2, alpha=0.5, weights=None):
        acc = 0
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

    def compute_atom_distributions(self, ds):
        atomses = dict()
        c = 100000000000
        for example in tqdm(ds):
            # print(example)
            t = example[1]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            atoms = self.extract_atoms(t)
            # print(atoms)
            if example[2] not in atomses:
                atomses[example[2]] = dict()
            for atom in atoms:
                if atom not in atomses[example[2]]:
                    atomses[example[2]][atom] = 0
                atomses[example[2]][atom] += 1
            if c <= 0:
                break
            c -= 1
        for k, atoms in atomses.items():
            total = sum(atoms.values())
            for atoms_k in atoms:
                atoms[atoms_k] = atoms[atoms_k] / total
        return atomses

    def compute_compound_distributions(self, ds):
        compoundses = dict()
        c = 100000000000
        for example in tqdm(ds):
            # print(example)
            t = example[1]
            if not isinstance(t, Tree):
                t = taglisp_to_tree(t)
            compounds = self.extract_compounds(t)
            # print(atoms)
            if example[2] not in compoundses:
                compoundses[example[2]] = dict()
            for compound in compounds:
                if compound not in compoundses[example[2]]:
                    compoundses[example[2]][compound] = 0
                compoundses[example[2]][compound] += 1
            if c <= 0:
                break
            c -= 1
        for k, compounds in compoundses.items():
            total = sum(compounds.values())
            for compounds_k in compounds:
                compounds[compounds_k] = compounds[compounds_k] / total
        return compoundses

    def compute_cooc_distributions(self, ds):
        compoundses = dict()
        weights = dict()
        c = 100000000000
        totalex = 0
        for example in tqdm(ds):
            # print(example)
            compounds = self.extract_coocs(taglisp_to_tree(example[1]))
            # print(atoms)
            if example[2] not in compoundses:
                compoundses[example[2]] = dict()
            for compound in compounds:
                if compound not in compoundses[example[2]]:
                    compoundses[example[2]][compound] = 0
                compoundses[example[2]][compound] += 1
            for compound in set(compounds):
                if compound not in weights:
                    weights[compound] = 0
                weights[compound] += 1
            if c <= 0:
                break
            c -= 1
            totalex += 1
        for k in weights:
            weights[k] = 1/weights[k]     # inverse example frequency
        totalweight = sum(weights.values())
        for k in weights:
            weights[k] = weights[k] * totalex / totalweight
        for k, compounds in compoundses.items():
            total = sum(compounds.values())
            for compounds_k in compounds:
                compounds[compounds_k] = compounds[compounds_k] / total
        return compoundses, weights

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

    def _compute_cooc_divergences(self, dists, fs):
        """ dists contains distributions per subset, fs contains in what proportion of examples a co-occurrence occurs"""
        divergences = dict()
        for subsetname in dists:
            for subsetname2 in dists:
                if self.verbose:
                    print(f"computing divergence between {subsetname} and {subsetname2}")
                divergences[subsetname + "-" + subsetname2] = 1 - self.compute_chernoff_coeff(dists[subsetname], dists[subsetname2], 0.1, weights=fs)
        return divergences

    def compute_cooc_divergences(self, ds):
        dists, dfs = self.compute_cooc_distributions(ds)
        return self._compute_cooc_divergences(dists, dfs)


def try_cfq():
    ds = CFQDatasetLoader().load("random/modent")
    print(ds[0])


if __name__ == '__main__':
    try_cfq()