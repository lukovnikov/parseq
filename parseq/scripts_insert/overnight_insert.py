from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple

import numpy as np
from nltk import Tree

import qelos as q
from parseq.datasets import OvernightDatasetLoader
from parseq.grammar import tree_to_lisp_tokens, tree_to_prolog
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer


class TreeAction(ABC):
    """ Abstract class representing a tree action, when executed on a tree, returns another tree"""
    @abstractmethod
    def apply(self, tree:Tree, *args, **kwargs)->Tree:
        pass

    def __call__(self, tree:Tree, *args, **kwargs):
        return self.apply(tree, *args, **kwargs)


class ReversibleTreeAction(TreeAction):
    """ A reversible tree action """
    @abstractmethod
    def unapply(self, tree:Tree, *args, **kwargs)->Tree:
        pass


def add_parentheses(x:Tree):
    ret = deepcopy(x)
    queue = [ret]
    while len(queue) > 0:
        first = queue.pop(0)
        if len(first) > 0:
            queue += first[:]
            first.insert(0, Tree("(", []))
            first.append(Tree(")", []))
    return ret


def uncomplete_tree(x:Tuple[str, Tree, str]):
    """ Input is tuple (nl, fl, split)
        Output is a randomly uncompleted tree, every node annotated whether it's terminated and what actions are good at that node
    """
    # region 1. initialize annotations
    fl:Tree = deepcopy(x[1])
    queue = [fl]
    while len(queue) > 0:
        first = queue.pop(0)
        first.is_terminated = True
        first.gold_actions = set(["@PAD@"])
        queue += first[:]
    # endregion

    # region 2. 

    return x


def load_ds(domain="restaurants", min_freq=0, top_k=np.infty, nl_mode="bert-base", trainonvalid=False):
    """
    Creates a dataset of examples which have
    * NL question and tensor
    * original FL tree
    * reduced FL tree with slots (this is randomly generated)
    * tensor corresponding to reduced FL tree with slots
    * mask specifying which elements in reduced FL tree are terminated
    * 2D gold that specifies whether a token/action is in gold for every position (compatibility with MML!)
    """
    ds = OvernightDatasetLoader(simplify_mode="light").load(domain=domain, trainonvalid=trainonvalid)
    ds = ds.map(lambda x: (x[0], add_parentheses(x[1]), x[2]))
    ds = ds.map(uncomplete_tree)

    seqenc_vocab = Vocab(padid=0, startid=2, endid=3, unkid=1)
    seqenc_vocab.add_token("@TERMINATE@", seen=np.infty)
    seqenc_vocab.add_token("@REMOVE@", seen=np.infty)
    seqenc_vocab.add_token("@SLOT@", seen=np.infty)

    seqenc = SequenceEncoder(vocab=seqenc_vocab, tokenizer=tree_to_lisp_tokens,
                             add_start_token=True, add_end_token=True)
    for example in ds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=example[2] == "train")
    seqenc.finalize_vocab(min_freq=min_freq, top_k=top_k)

    nl_tokenizer = BertTokenizer.from_pretrained(nl_mode)
    def tokenize(x):
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0],
               seqenc.convert(x[1], return_what="tensor"),
               x[2],
               x[0], x[1])
        return ret
    tds, vds, xds = ds[(None, None, "train")].map(tokenize), \
                    ds[(None, None, "valid")].map(tokenize), \
                    ds[(None, None, "test")].map(tokenize)
    return tds, vds, xds, nl_tokenizer, seqenc


if __name__ == '__main__':
    q.argprun(load_ds)