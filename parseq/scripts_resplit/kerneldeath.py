import json
from typing import List, Tuple
from nltk import Tree
from tqdm import tqdm
from parseq.datasets import CFQDatasetLoader
from parseq.grammar import taglisp_to_tree
from parseq.scripts_resplit.resplit_cfq import DivergenceComputer, FrequencyDistribution

ds = CFQDatasetLoader().load("mcd1/modent", validfrac=0, loadunused=True)
dc = DivergenceComputer()

atom_dists = dc.compute_atom_distributions(ds)
print(json.dumps(dc._compute_atom_divergences(atom_dists), indent=3))

comp_dists = dc.compute_compound_distributions(ds)
print(json.dumps(dc._compute_compound_divergences(comp_dists), indent=3))