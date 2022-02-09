import csv
import json
import os
import random
import re
import shutil
import sys
import tarfile
import timeit
import urllib
import urllib.request
from abc import abstractmethod
from copy import copy
from typing import List, Tuple, Callable, Union

import nltk
import torch
import ujson
from nltk import Tree, Nonterminal
import numpy as np
import qelos as q
from nltk.parse.generate import generate
from scipy.special import softmax
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from parseq.grammar import lisp_to_pas, pas_to_tree, tree_size, tree_to_lisp, tree_to_lisp_tokens, lisp_to_tree, \
    tree_to_taglisp, taglisp_to_tree, prolog_to_tree
from parseq.vocab import SequenceEncoder, Vocab

import multiprocessing as mp


class Dataset(object):
    def __init__(self, examples=tuple(), **kw):
        """
        :param examples:    a collection of examples. Can be any iterable of a tuple, list, dict, ...
                            Default: empty tuple
        :param kw:
        """
        super(Dataset, self).__init__(**kw)
        self._examples = list(examples)

    @property
    def examples(self):
        ret = [self[i] for i in range(len(self))]
        return ret

    @property
    def rootds(self):
        if hasattr(self, "baseds"):
            return self.baseds.rootds
        else:
            return self

    def __len__(self):
        return len(self._examples)

    def _example_fits_filter(self, ex, f):
        if isinstance(f, Callable) and f(ex):
            return True
        elif isinstance(f, tuple):
            assert (isinstance(ex, (tuple, list)))
            assert (len(f) == len(ex))
            keep = True
            for fe, exe in zip(f, ex):
                if fe is not None:
                    if isinstance(fe, Callable):
                        keep = keep and fe(exe)
                    else:
                        keep = keep and (fe == exe)
            if keep:
                return True
        elif isinstance(f, dict):
            assert (isinstance(ex, dict))
            keep = True
            for k in f:
                assert (k in ex)
                if isinstance(f[k], Callable):
                    keep = keep and f[k](ex[k])
                else:
                    keep = keep and (f[k] == ex[k])
            if keep:
                True
        return False

    def filter(self, f):
        ret = []
        for i, ex in enumerate(self._examples):
            if self._example_fits_filter(ex, f):
                ret.append(ex)
        ret = Dataset(ret)
        return ret

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            ret = self.filter(item)
            return ret
        elif isinstance(item, str):     # interpreted as split, then last column is assumed to be the split
            ret = self.filter(lambda x: x[-1] == item).map(lambda x: x[:-1])
            return ret
        else:
            ret = self._examples[item]
            return ret

    def map(self, f):
        """
        Create a MappedDataset that will apply given function f on an example on __getitem__
        """
        ret = MappedDataset(self, f)
        return ret

    def __add__(self, other):   # concatenate this dataset with another
        raise NotImplemented()


class CachedDataset(object):
    def __init__(self, use_cache=False, **kw):
        super(CachedDataset, self).__init__(**kw)
        self.use_cache = use_cache
        self._examples_cache = {}
        self.baseds = None

    def cache(self, compute_now=False):
        """ Enable cache at this level. """
        self.enable_cache()
        if compute_now:
            [self[i] for i in range(len(self))]
        return self

    def clear_cache(self):
        self._examples_cache = {}
        return self

    def disable_cache(self):
        self.use_cache = False
        return self

    def enable_cache(self):
        self.use_cache = True
        return self

    def deep_disable_cache(self):
        self.disable_cache()
        if isinstance(self.baseds, CachedDataset):
            self.baseds.deep_disable_cache()
        return self

    def deep_enable_cache(self):
        self.enable_cache()
        if isinstance(self.baseds, CachedDataset):
            self.baseds.deep_enable_cache()
        return self


class MappedDataset(Dataset, CachedDataset):
    def __init__(self, baseds:Dataset, f, use_cache=False, **kw):
        self._kw = copy(kw)
        super(MappedDataset, self).__init__(use_cache=use_cache, **kw)
        self.baseds = baseds
        self.f = f

    def __len__(self):
        return len(self.baseds)

    @property
    def examples(self):
        ret = [self[i] for i in range(len(self))]
        return ret

    def filter(self, f):
        newbase = self.baseds.filter(lambda ex: self._example_fits_filter(self.f(ex), f))
        ret = newbase.map(self.f)
        return ret

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            ret = self.filter(item)
            return ret
        elif isinstance(item, str):     # interpreted as split, then last column is assumed to be the split
            ret = self.filter(lambda x: x[-1] == item).map(lambda x: x[:-1])
            return ret
        else:
            if self.use_cache and item in self._examples_cache:
                ret = self._examples_cache[item]
                return ret
            else:
                example = self.baseds[item]
                ret = self.f(example)
                if self.use_cache:
                    self._examples_cache[item] = ret
                return ret


class IterableDataset(Dataset, torch.utils.data.IterableDataset):
    def __init__(self, **kw):
        super(IterableDataset, self).__init__(**kw)

    def __len__(self):
        return sys.maxsize

    @property
    def examples(self):
        return None

    def __iter__(self):
        return self

    def filter(self, f):
        return FilteredIterableDataset(self, f)

    def map(self, f):
        return IterableMappedDataset(self, f)


class FilteredIterableDataset(IterableDataset):
    def __init__(self, baseds:IterableDataset, f):
        self.baseds = baseds
        self.f = f

    def __next__(self):
        ret = None
        while ret is None:
            ret = next(self.baseds)
            if not self.f(ret):
                ret = None
        return ret


class IterableMappedDataset(IterableDataset):
    def __init__(self, baseds:IterableDataset, f, **kw):
        self._kw = copy(kw)
        super(IterableMappedDataset, self).__init__(**kw)
        self.baseds = baseds
        self.f = f

    def filter(self, f):
        newbase = self.baseds.filter(lambda ex: self._example_fits_filter(self.f(ex), f))
        ret = newbase.map(self.f)
        return ret

    def __next__(self):
        example = next(self.baseds)
        ret = self.f(example)
        return ret


class GeneratedDataset(Dataset, CachedDataset):
    def __init__(self, N=int(9e9), seed=12345678, use_cache=False, maxlen=int(9e9), **kw):
        super(GeneratedDataset, self).__init__(use_cache=use_cache, **kw)
        self.seed = seed
        self.N = N
        self.maxlen = maxlen

    def advance_seed(self):
        rng = np.random.RandomState(self.seed)
        newseed = rng.randint(10000000, 99999999)
        self.seed = newseed

    def __len__(self):
        return self.N

    def filter(self, f):
        if self.N >= 1e6:
            print(f"WARNING: number of examples is huge ({self.N}), a stored Dataset is being created. Please reconsider.")
        ret = []
        for i in tqdm(range(self.N), disable=False):
            ex = self[i]
            if self._example_fits_filter(ex, f):
                ret.append(ex)
        ret = Dataset(ret)
        return ret

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            return self.filter(item)
        else:
            if self.use_cache and item in self._examples_cache:
                ret = self._examples_cache[item]
                return ret
            rng = np.random.RandomState(self.seed + item)
            ret = self.generate(rng=rng)
            if self.use_cache:
                self._examples_cache[item] = ret
            return ret

    @property
    def examples(self):
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def generate(self, rng=None): pass

    def map(self, f):
        return GeneratedMappedDataset(self, f)


class GeneratedMappedDataset(MappedDataset):
    @property
    def examples(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            return self.filter(item)
        else:
            if self.use_cache and item in self._examples_cache:
                ret = self._examples_cache[item]
                return ret
            else:
                example = self.baseds[item]
                ret = self.f(example)
                if self.use_cache:
                    self._examples_cache[item] = ret
                return ret

    def map(self, f):
        return GeneratedMappedDataset(self, f)


class Pipeline(object):
    def __init__(self, **kw):
        super(Pipeline, self).__init__(**kw)
        self.transforms = []

    def add(self, f):
        self.transforms.append(f)
        return self

    def __call__(self, x):
        ret = x
        for f in self.transforms:
            ret = f(ret)
        return ret


class BatchDataset(Dataset):
    """ Dataset created by providing it a batch from a dataloader. """
    def __init__(self, batchdata, **kw):
        examples = zip(*batchdata)
        super(BatchDataset, self).__init__(examples, **kw)


class PCFGDataset(GeneratedDataset):
    def __init__(self, pcfg, N=int(1e6), seed=12345678, temperature=1., **kw):
        super(PCFGDataset, self).__init__(N=N, seed=seed, **kw)
        self._pcfg = pcfg
        self.temperature = temperature

    def generate(self, start=None, rng=None):
        rng = np.random.RandomState(self.seed) if rng is None else rng
        start = self._pcfg.start() if start is None else start
        productions = self._pcfg.productions(start)
        productions, probs = zip(*[(prod, prod.prob()) for prod in productions])
        if self.temperature != 1.:
            probs = softmax(np.log(probs) * self.temperature)
        chosen_prod = rng.choice(productions, p=probs)
        ret = []
        for rhse in chosen_prod.rhs():
            if isinstance(rhse, Nonterminal):
                rete = self.generate(rhse, rng)
                ret.append(rete)
            else:
                ret.append(rhse)
        if isinstance(ret[0], str):
            ret = Tree(ret[0], ret[1:])
        else:
            assert(len(ret) == 1)
            assert(isinstance(ret[0], Tree))
            ret = ret[0]

        # check size
        ret_size = tree_length(ret, count_brackets=True)
        if ret_size < self.maxlen:
            return ret
        else:
            # generate another one
            rng = np.random.RandomState(rng.randint(10000000, 99999999))
            return self.generate(start=start, rng=rng)


def tree_length(x:Union[Tree, str], count_brackets=False):
    if isinstance(x, str):
        return 1
    elif isinstance(x, Tree):
        if len(x) == 0:
            return 1
        else:
            return 1 + sum([tree_length(xe, count_brackets=count_brackets) for xe in x]) + (2 if count_brackets else 0)
    else:
        raise Exception()


def tree_depth(x:Union[Tree, str]):
    if isinstance(x, str):
        return 1
    elif isinstance(x, Tree):
        if len(x) == 0:
            return 1
        else:
            return 1 + max([tree_depth(subtree) for subtree in x])


class PCFGBuilder(object):
    def __init__(self, orderless=tuple(), **kw):
        super(PCFGBuilder, self).__init__(**kw)
        self.orderless = set(orderless)

    def build(self, examples=tuple()):
        """
        :param examples:    tuple or list of nltk Trees
        :return: 
        """
        allproductions = []
        for example in examples:
            q = example
            t = self.grammarify(q)
            t = Tree("S", [t])
            productions = t.productions()
            allproductions += productions
        pcfg = nltk.induce_pcfg(Nonterminal("S"), allproductions)
        return pcfg

    def grammarify(self, x, parents=tuple()):
        if len(x) > 0:
            children = [self.grammarify(xe, parents=parents+(x,)) for xe in x]
            children = [
                Tree(f"NT-{x.label()}-ARG{i}", [xe])
                        if x.label() not in self.orderless else
                        Tree(f"NT-{x.label()}-ARG", [xe])
                for i, xe in enumerate(children)]
            children = ["(", x.label()] + children + [")"]
            t = Tree(f"NT-{x.label()}", children)
            # t = Tree(f"S", [t])
        else:
            t = x.label()
        return t


def build_vocab_from_pcfg(pcfg, min_freq=0, top_k=np.infty)->Vocab:
    vocab = Vocab()
    vocab.add_token("(")
    vocab.add_token(")")
    for rule in pcfg.productions():
        vocab.add_token(str(rule.lhs()))
        for rhse in rule.rhs():
            vocab.add_token(str(rhse))
    vocab.finalize(min_freq=min_freq, top_k=top_k)
    return vocab


def autocollate(x, pad_value=0):
    y = list(zip(*x))
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.Tensor) and yi[0].dtype in (torch.int64, torch.int32, torch.int16):# and yi[0].dim() == 1:
            y[i] = q.pad_tensors(yi, list(range(yi[0].dim())), pad_value)
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.Tensor):
            yi = [yij[None] for yij in yi]
            y[i] = torch.cat(yi, 0)
    return y


def pad_and_default_collate(x, pad_value=0):
    y = list(zip(*x))
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.Tensor) and yi[0].dtype in (torch.int64, torch.int32, torch.int16) and yi[0].dim() == 1:
            y[i] = q.pad_tensors(yi, 0, pad_value)
    x = list(zip(*y))
    ret = default_collate(x)
    return ret


# region NOISE FUNCTIONS
class TokenMasker(object):
    mask_symbol = "@MASK@"
    def __init__(self, p=.2, seed=None, **kw):
        super(TokenMasker, self).__init__(**kw)
        self.p = p
        self.rng = np.random.RandomState(seed)

    def __call__(self, tokens:List[str])->List[str]:
        ret = []
        i = 0
        while i < len(tokens):
            if self.p > self.rng.random_sample():
                ret.append(self.mask_symbol)
            else:
                ret.append(tokens[i])
            i += 1
        return ret


class TokenDeleter(object):
    def __call__(self, tokens:List[str])->List[str]:
        ret = []
        i = 0
        while i < len(tokens):
            if self.p > self.rng.random_sample():
                pass
            else:
                ret.append(tokens[i])
            i += 1
        return ret


class SpanMasker(TokenMasker):
    def __init__(self, p=.1, lamda=2.2, seed=None, **kw):
        """
        :param p: probability of replacing a span with mask symbol
        :param lamda: parameter for Poisson distribution on the length of span (0 ==> length 1 always)
        :param kw:
        """
        super(SpanMasker, self).__init__(p=p, seed=seed, **kw)
        self.lamda = lamda

    def __call__(self, tokens:List[str])->List[str]:
        ret = []
        i = 0
        while i < len(tokens):
            if self.p > self.rng.random_sample():
                l = self.rng.poisson(self.lamda)
                ret.append(self.mask_symbol)
                i += l
            else:
                ret.append(tokens[i])
                i += 1
        return ret


class SubtreeMasker(TokenMasker):
    def __call__(self, x:Tree):
        if self.p > self.rng.random_sample():
            return Tree(self.mask_symbol, [])
        else:
            return Tree(x.label(), [self(xe) for xe in x])

# endregion


# region SPECIFIC DATASETS

class MultilingualGeoqueryDatasetLoader(object):
    def __init__(self,
                 p="../datasets/geo880_multiling/geoquery",
                 validfrac=0.2,
                 **kw):
        super(MultilingualGeoqueryDatasetLoader, self).__init__(**kw)
        self.p = p
        self.validfrac = validfrac

    def load(self, lang:str="en"):
        data = []
        with open(os.path.join(self.p, f"geo-{lang}.json")) as f:
            data = json.load(f)
        print(f"{len(data)} examples loaded for language {lang}")
        return Dataset(data)


def try_multilingual_geoquery_dataset_loader():
    dsl = MultilingualGeoqueryDatasetLoader()
    dsl.load("fa")
    print("done")


class GeoDatasetLoader(object):
    def __init__(self,
                 p="../datasets/geo880dong",
                 validfrac=0.1, **kw):
        super(GeoDatasetLoader, self).__init__(**kw)
        self._p = p
        self.validfrac = validfrac


class OvernightDatasetLoader(object):
    def __init__(self,
                 p="../datasets/overnightData/",
                 mp="../datasets/overnight/",
                 pcache="../datasets/overnightCache/",
                 usecache=False,
                 validfrac=.2,
                 simplify_mode="full",      # or "none"
                 simplify_blocks=False,
                 restore_reverse=False,
                 load_lexicon=False,
                 **kw):
        super(OvernightDatasetLoader, self).__init__(**kw)
        self._simplify_filters = True        # if True, filter expressions are converted to orderless and-expressions
        self._pcache = pcache if usecache else None
        self._usecache = usecache
        self.validfrac = validfrac
        self._p = p
        self._mp = mp
        self.simplify_mode = simplify_mode      # "full" or "light"
        self.simplify_blocks = simplify_blocks
        self._restore_reverse = restore_reverse
        self.load_lexicon = load_lexicon

    @property
    def full_simplify(self):
        return self.simplify_mode == "full"

    def load(self, domain:str="restaurants", trainonvalid=False):

        examples, lexicon = self._initialize(self._p, self._mp, domain)

        # _examples = examples
        # examples = [example for example in _examples if example[2] != "train"]
        if lexicon is not None:
            examples += [(nl, lf, "lexicon") for nl, lf in lexicon]
        '''
        examples is a list of tuples with each tuple being nl (utterance), lf (logical form) 
        and (text/train/valid/lexicon)
        '''
        if trainonvalid:
            _examples = examples
            examples = []
            for example in _examples:
                if example[2] == "train":
                    examples.append(example)
                elif example[2] == "valid":
                    examples.append((example[0], example[1], "train"))
                elif example[2] == "test":
                    examples.append((example[0], example[1], "valid"))
                    examples.append(example)
        return Dataset(examples)

    def _ztree_to_lf(self, ztree):
        afterstring = set()
        def simplify_tree(t:Tree):
            if t.label() == "call":     # remove call, make first arg of it the parent of the other args
                assert(len(t[0]) == 0)
                # if not t[0].label().startswith("SW."):
                #     print(t)
                # assert(t[0].label().startswith("SW."))
                t.set_label(t[0].label())
                del t[0]
            elif t.label() == "string": # remove, annotate
                afterstring.update(set([tc.label() for tc in t]))
                assert(len(t) == 1)
                assert(len(t[0]) == 0)
                t.set_label(f"arg:{t[0].label()}")
                del t[0]
            if t.label().startswith("edu.stanford.nlp.sempre.overnight.SimpleWorld."):
                t.set_label("SW:" + t.label()[len("edu.stanford.nlp.sempre.overnight.SimpleWorld."):])
            if t.label() == "SW:getProperty":
                assert(len(t) == 2)
                if self.full_simplify:
                    ret = simplify_tree(t[1])
                    ret.append(simplify_tree(t[0]))
                else:
                    children = [simplify_tree(te) for te in t]
                    ret = t
                    ret[:] = children
                return ret
            elif t.label() == "SW:singleton":
                assert(len(t) == 1)
                assert(len(t[0]) == 0)
                if not self.full_simplify:
                    t[0].set_label(f"singleton:{t[0].label()}")
                return simplify_tree(t[0])
            elif t.label() == "SW:ensureNumericProperty":
                assert(len(t) == 1)
                # assert(len(t[0]) == 1)
                # t[0][0].set_label(f"numeric:{t[0][0].label()}")
                if self.full_simplify:
                    ret = simplify_tree(t[0])
                else:
                    ret = t
                    ret[:] = [simplify_tree(te) for te in ret]
                return ret
            elif t.label() == "SW:ensureNumericEntity":
                assert(len(t) == 1)
                if self.full_simplify:
                    ret = simplify_tree(t[0])
                else:
                    ret = t
                    ret[:] = [simplify_tree(te) for te in ret]
                return ret
            elif t.label() == "SW:aggregate":
                assert(len(t) == 2)
                ret = simplify_tree(t[0])
                assert(ret.label() in ["arg:avg", "arg:sum"])
                assert(len(ret) == 0)
                ret.set_label(f"agg:{ret.label()}")
                ret.append(simplify_tree(t[1]))
                return ret
            else:
                t[:] = [simplify_tree(tc) for tc in t]
                return t

        def simplify_further(t):
            """ simplifies filters and count expressions """
            # replace filters with ands
            if t.label() == "SW:filter" and self._simplify_filters is True:
                if len(t) not in (2, 4):
                    raise Exception(f"filter expression should have 2 or 4 children, got {len(t)}")
                children = [simplify_further(tc) for tc in t]
                startset = children[0]
                if len(children) == 2:
                    condition = Tree("cond:has", [children[1]])
                elif len(children) == 4:
                    condition = Tree(f"cond:{children[2].label()}", [children[1], children[3]])
                conditions = [condition]
                if startset.label() == "op:and":
                    conditions = startset[:] + conditions
                else:
                    conditions = [startset] + conditions
                # check for same conditions:
                i = 0
                while i < len(conditions) - 1:
                    j = i + 1
                    while j < len(conditions):
                        if conditions[i] == conditions[j]:
                            # print(f"SAME!: {conditions[i]}, {conditions[j]}")
                            del conditions[j]
                            j -= 1
                        j += 1
                    i += 1

                ret = Tree(f"op:and", conditions)
                return ret
            # replace countSuperlatives with specific ones
            elif t.label() == "SW:countSuperlative":
                assert(t[1].label() in ["arg:max", "arg:min"])
                t.set_label(f"SW:CNT-{t[1].label()}")
                del t[1]
                t[:] = [simplify_further(tc) for tc in t]
            elif t.label() == "SW:countComparative":
                assert(t[2].label() in ["arg:<", "arg:<=", "arg:>", "arg:>=", "arg:=", "arg:!="])
                t.set_label(f"SW:CNT-{t[2].label()}")
                del t[2]
                t[:] = [simplify_further(tc) for tc in t]
            else:
                t[:] = [simplify_further(tc) for tc in t]
            return t

        def simplify_furthermore(t):
            """ replace reverse rels"""
            if t.label() == "arg:!type":
                t.set_label("arg:~type")
                return t
            elif t.label() == "SW:reverse":
                assert(len(t) == 1)
                assert(t[0].label().startswith("arg:"))
                assert(len(t[0]) == 0)
                t.set_label(f"arg:~{t[0].label()[4:]}")
                del t[0]
                return t
            elif t.label().startswith("cond:arg:"):
                assert(len(t) == 2)
                head = t[0]
                head = simplify_furthermore(head)
                if self.full_simplify:
                    assert(head.label().startswith("arg:"))
                    assert(len(head) == 0)
                    headlabel = f"arg:~{head.label()[4:]}"
                    headlabel = headlabel.replace("~~", "")
                    head.set_label(headlabel)
                    body = simplify_furthermore(t[1])
                    if t.label()[len("cond:arg:"):] != "=":
                        body = Tree(t.label()[5:], [body])
                    head.append(body)
                    return head
                else:
                    t[:] = [simplify_furthermore(tc) for tc in t]
                    return t
            else:
                t[:] = [simplify_furthermore(tc) for tc in t]
                return t

        def simplify_final(t):
            if t.label() == "SW:listValue":
                return t[0]
            else:
                return t
            # assert(t.label() == "SW:listValue")
            # assert(len(t) == 1)
            # return t[0]

        def remap_reverse_labels_blocks(t:Tree):
            mapd = {
                "arg:~left": "arg:right",
                "arg:~right": "arg:left",
                "arg:~below": "arg:above",
                "arg:~above": "arg:below"
            }
            if t.label() in mapd:
                t.set_label(mapd[t.label()])
            [remap_reverse_labels_blocks(te) for te in t]
            return t

        def restore_reverse(t:Tree)->Tree:
            tchildren = [restore_reverse(te) for te in t]
            t[:] = tchildren
            if t.label().startswith("arg:~") and not t.label() == "arg:~type":
                newt = Tree("SW:reverse", [t])
                t.set_label(f"arg:{t.label()[len('arg:~'):]}")
                return newt
            else:
                return t

        def simplify_sw(t:Tree)->Tree:
            if t.label().startswith("edu.stanford.nlp.sempre.overnight.SimpleWorld."):
                t.set_label("SW:" + t.label()[len("edu.stanford.nlp.sempre.overnight.SimpleWorld."):])
            t[:] = [simplify_sw(te) for te in t]
            return t

        def simplify_call(t:Tree) -> Tree:
            if t.label() == "call":     # remove call, make first arg of it the parent of the other args
                assert(len(t[0]) == 0)
                # if not t[0].label().startswith("SW."):
                #     print(t)
                # assert(t[0].label().startswith("SW."))
                t.set_label(f"call-{t[0].label()}")
                del t[0]

            t[:] = [simplify_call(tc) for tc in t]
            return t

        def simplify_filters(t:Tree)->Tree:
            if t.label() == "call-SW:filter" and self._simplify_filters is True:
                if len(t) not in (2, 4):
                    raise Exception(f"filter expression should have 2 or 4 children, got {len(t)}")
                children = [simplify_filters(tc) for tc in t]
                startset = children[0]
                if len(children) == 2:
                    condition = Tree("condition", [children[1]])
                elif len(children) == 4:
                    condition = Tree(f"condition", [children[1], children[2], children[3]])
                conditions = [condition]
                if startset.label() == "filter":
                    conditions = startset[:] + conditions
                else:
                    conditions = [startset] + conditions
                # check for same conditions:
                i = 0
                while i < len(conditions) - 1:
                    j = i + 1
                    while j < len(conditions):
                        if conditions[i] == conditions[j]:
                            # print(f"SAME!: {conditions[i]}, {conditions[j]}")
                            del conditions[j]
                            j -= 1
                        j += 1
                    i += 1

                ret = Tree(f"filter", conditions)
                return ret
            else:
                t[:] = [simplify_filters(tc) for tc in t]
            return t

        lf = ztree
        if self.simplify_mode == "none":
            lf = simplify_sw(lf)
            lf = simplify_call(lf)
            lf = simplify_filters(lf)
            return lf
        if self.simplify_mode != "none":
            lf = simplify_tree(lf)
            lf = simplify_further(lf)
            lf = simplify_furthermore(lf)
            lf = simplify_final(lf)
        if self.simplify_blocks:
            lf = remap_reverse_labels_blocks(lf)
        if self._restore_reverse:
            lf = restore_reverse(lf)
        return lf

    def grammarlines_to_lexicon(self, lines:List[str]):
        ret = []
        ltp = None
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                continue
            z, ltp = lisp_to_tree(line.strip(), ltp)
            if z is not None:
                if z.label() == "rule":
                    if z[0].label() not in {"$TypeNP", "$RelNP", "$Rel0NP", "$EntityNP1", "$EntityNP2", "$VP/NP", "$VP", "$Value"}:
                        assert(False)       # unexpected lexical entry type
                    assert(z[2].label() == "ConstantFn")
                    assert(len(z[2]) == 1)
                    # print(z)
                    lf = self._ztree_to_lf(z[2][0])
                    nl = z[1]
                    nl = [nl.label()] + [nle.label() for nle in nl]
                    ret.append((" ".join(nl), lf))
                ltp = None
        return ret

    def lines_to_examples(self, lines:List[str]):
        maxsize_before = 0
        avgsize_before = []
        avgdepth_before = []
        maxsize_before_brackets = 0
        avgsize_before_brackets = []
        maxsize_after = 0
        avgsize_after = []
        avgdepth_after = []
        maxsize_after_brackets = 0
        avgsize_after_brackets = []

        ret = []
        ltp = None
        j = 0
        for i, line in enumerate(lines):
            z, ltp = lisp_to_pas(line, ltp)
            if z is not None:
                # print(f"Example {j}:")
                question = z[1][0][1][0][1:-1]
                ztree = pas_to_tree(z[1][2][1][0])

                maxsize_before = max(maxsize_before, tree_length(ztree, count_brackets=False))
                avgsize_before.append(tree_length(ztree, count_brackets=False))
                avgdepth_before.append(tree_depth(ztree))
                maxsize_before_brackets = max(maxsize_before_brackets, tree_length(ztree, count_brackets=True))
                avgsize_before_brackets.append(tree_length(ztree, count_brackets=True))

                lf = self._ztree_to_lf(ztree)

                ret.append((question, lf))
                # print(ret[-1][0])
                # print(ret[-1][1])
                ltp = None
                maxsize_after = max(maxsize_after, tree_length(lf, count_brackets=False))
                avgsize_after.append(tree_length(lf, count_brackets=False))
                avgdepth_after.append(tree_depth(lf))
                maxsize_after_brackets = max(maxsize_after_brackets, tree_length(lf, count_brackets=True))
                avgsize_after_brackets.append(tree_length(lf, count_brackets=True))

                # print(pas_to_tree(z[1][2][1][0]))
                # print()
                j += 1

        avgsize_before = sum(avgsize_before) / len(avgsize_before)
        avgsize_after = sum(avgsize_after) / len(avgsize_after)
        avgdepth_before = sum(avgdepth_before) / len(avgdepth_before)
        avgdepth_after = sum(avgdepth_after) / len(avgdepth_after)
        avgsize_before_brackets = sum(avgsize_before_brackets) / len(avgsize_before_brackets)
        avgsize_after_brackets = sum(avgsize_after_brackets) / len(avgsize_after_brackets)


        print(f"Simplification results ({j} examples):")
        print(f"\t Max, Avg size, Avg depth before: {maxsize_before}, {avgsize_before:.2f} {avgdepth_before:.2f} (with brackets: {maxsize_before_brackets}, {avgsize_before_brackets:.2f})")
        print(f"\t Max, Avg size, Avg depth after: {maxsize_after}, {avgsize_after:.2f} {avgdepth_after:.2f} (with brackets: {maxsize_after_brackets}, {avgsize_after_brackets:.2f})")

        return ret

    def _load_cached(self, domain):
        train_cached = ujson.load(open(os.path.join(os.path.dirname(__file__), self._pcache, f"{domain}.train.json"), "r"))
        trainexamples = [(x, Tree.fromstring(y)) for x, y in train_cached]
        test_cached = ujson.load(open(os.path.join(os.path.dirname(__file__), self._pcache, f"{domain}.test.json"), "r"))
        testexamples = [(x, Tree.fromstring(y)) for x, y in test_cached]
        print("loaded from cache")
        return trainexamples, testexamples

    def _cache(self, domain:str, trainexamples:List[Tuple[str, Tree]], testexamples:List[Tuple[str, Tree]]):
        train_cached, test_cached = None, None
        if os.path.exists(os.path.join(os.path.dirname(__file__), self._pcache, f"{domain}.train.json")):
            try:
                train_cached = ujson.load(open(os.path.join(os.path.dirname(__file__), self._pcache, f"{domain}.train.json"), "r"))
                test_cached = ujson.load(open(os.path.join(os.path.dirname(__file__), self._pcache, f"{domain}.test.json"), "r"))
            except (IOError, ValueError) as e:
                pass
        trainexamples = [(x, str(y)) for x, y in trainexamples]
        testexamples = [(x, str(y)) for x, y in testexamples]

        if train_cached != trainexamples:
            with open(os.path.join(os.path.dirname(__file__), self._pcache, f"{domain}.train.json"), "w") as f:
                ujson.dump(trainexamples, f, indent=4, sort_keys=True)
        if test_cached != testexamples:
            with open(os.path.join(os.path.dirname(__file__), self._pcache, f"{domain}.test.json"), "w") as f:
                ujson.dump(testexamples, f, indent=4, sort_keys=True)
        print("saved in cache")

    def _initialize(self, p, mp, domain):
        self.data = {}

        trainexamples, testexamples = None, None
        if self._usecache:
            try:
                trainexamples, testexamples = self._load_cached(domain)
            except (IOError, ValueError) as e:
                pass

        if trainexamples is None:
            '''
                Train/Test/Grammar-lines: read the dataset and split it line by line.
                lexicon: all the tokens (entities and relations) which are used for creating semantic parse.
                Train/Test-examples: question and the corresponding target semantic parse (TargetFormula) represented as a nltk Tree.
                (http://www.nltk.org/_modules/nltk/tree.html)
            '''

            trainlines = [x.strip() for x in
                         open(os.path.join(os.path.dirname(__file__), p, f"{domain}.paraphrases.train.examples"), "r").readlines()]
            testlines = [x.strip() for x in
                        open(os.path.join(os.path.dirname(__file__), p, f"{domain}.paraphrases.test.examples"), "r").readlines()]

            grammarlines = [x.strip() for x in open(os.path.join(os.path.dirname(__file__), mp, f"{domain}.grammar"), "r").readlines()]

            trainexamples = self.lines_to_examples(trainlines)
            testexamples = self.lines_to_examples(testlines)
            if self.load_lexicon:
                lexicon = self.grammarlines_to_lexicon(grammarlines)
            else:
                lexicon = None

            if self._usecache:
                self._cache(domain, trainexamples, testexamples)

        questions, queries = tuple(zip(*(trainexamples + testexamples)))
        trainlen = int(round((1-self.validfrac) * len(trainexamples)))
        validlen = len(trainexamples) - trainlen
        splits = ["train"] * trainlen + ["valid"] * validlen
        rng = np.random.RandomState(12345678) # random number generator
        rng.shuffle(splits)
        assert(len(splits) == len(trainexamples))
        splits = splits + ["test"] * len(testexamples)

        # Splits represent train/test/valid indices.
        examples = list(zip(questions, queries, splits))
        return examples, lexicon


class TOPDatasetLoader(object):
    def __init__(self,
                 p="../datasets/top/",
                 include_unsupported=False, **kw):
        super(TOPDatasetLoader, self).__init__(**kw)
        self._p = p
        self._include_unsupported = include_unsupported

    def lines_to_examples(self, x:List[List[str]]):
        def convert_leaf_str_to_tree(_tree):
            if isinstance(_tree, str):
                return Tree(_tree, [])
            else:
                return Tree(_tree.label(), [convert_leaf_str_to_tree(_tree_e) for _tree_e in _tree])
        ret = []
        for example in x:
            if example[0].replace(" ", "") != example[1].replace(" ", ""):
                assert(example[0] == example[1])
            example = (example[1], example[2])
            tree = Tree.fromstring(example[1], brackets='[]')
            if not self._include_unsupported and tree.label().startswith("IN:UNSUPPORTED"):
                continue
            tree = convert_leaf_str_to_tree(tree)
            # print(tree)
            example = (example[0], tree)
            ret.append(example)
        return ret

    def load(self):
        p = self._p

        with open(os.path.join(os.path.dirname(__file__), p, f"train.tsv"), "r") as f:
            trainlines = [row for row in csv.reader(f, delimiter="\t")]
        with open(os.path.join(os.path.dirname(__file__), p, f"eval.tsv"), "r") as f:
            validlines = [row for row in csv.reader(f, delimiter="\t")]
        with open(os.path.join(os.path.dirname(__file__), p, f"test.tsv"), "r") as f:
            testlines = [row for row in csv.reader(f, delimiter="\t")]

        trainexamples = self.lines_to_examples(trainlines)
        validexamples = self.lines_to_examples(validlines)
        testexamples = self.lines_to_examples(testlines)

        questions, parses = tuple(zip(*(trainexamples + validexamples + testexamples)))
        splits = ["train"] * len(trainexamples) + ["valid"] * len(validexamples) + ["test"] * len(testexamples)

        examples = list(zip(questions, parses, splits))
        return Dataset(examples)


class OvernightPCFGBuilder(PCFGBuilder):
    def __init__(self, **kw):
        super(OvernightPCFGBuilder, self).__init__(("op:and",), **kw)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class CFQDatasetLoader(object):
    fullp = "https://storage.googleapis.com/cfq_dataset/cfq1.1.tar.gz"
    split_names = "mcd1,mcd2,mcd3,random"
    available_splits = tuple("mcd1,mcd2,mcd3,random".split(","))
    rename = {
        "mcd1": "mcd1",
        "mcd2": "mcd2",
        "mcd3": "mcd3",
        "random": "random_split",
        "question_pattern": "question_pattern_split",
        "question_complexity": "question_complexity_split",
        "query_pattern": "query_pattern_split",
        "query_complexity": "query_complexity_split"
    }
    remap = {
        "trainIdxs": "train",
        "devIdxs": "valid",
        "testIdxs": "test"
    }

    def __init__(self, path="../datasets/cfq/", **kw):
        super(CFQDatasetLoader, self).__init__(**kw)
        self.p = os.path.join(os.path.dirname(__file__), path)
        self.tt = q.ticktock("CFQDatasetLoader")
        self.tt.tick("make data")
        if os.path.exists(os.path.join(self.p, "_data.jsonl")):
            pass
        else:
            if os.path.exists(os.path.join(self.p, "_data.json")):
                pass
            else:
                self.download()
            self.clean_data()
        self.tt.tock()

    def clean_data(self):       # data can also be downloaded here: https://www.dropbox.com/s/7kjl25k0oea9ziy/cfq_data.jsonl?dl=0
        self.tt.tick("clean data")
        d = json.load(open(os.path.join(self.p, "_data.json")))
        # y = []
        with open(os.path.join(self.p, "_data.jsonl"), "w") as outf:
            for ds in tqdm(d):
                if "ruleTree" in ds:
                    del ds["ruleTree"]
                if "ruleIds" in ds:
                    del ds["ruleIds"]
                outf.write(json.dumps(ds) + "\n")
            # outf.flush()
        # json.dump(y, open(os.path.join(self.p, "_data.json")))
        os.remove(os.path.join(self.p, "_data.json"))
        self.tt.tock("cleaned data")

    def download(self):
        self.tt.tick("prepare")
        self.tt.tick("download")
        download_url(self.fullp, os.path.join(self.p, "cfq.tar.gz"))
        self.tt.tock()
        self.tt.tick("unpack")
        tar = tarfile.open(os.path.join(self.p, "cfq.tar.gz"), "r:gz")
        tar.extractall(self.p)
        tar.close()
        self.tt.tock()

        self.tt.tick("process")
        shutil.move(os.path.join(self.p, "cfq", "dataset.json"),
                    os.path.join(self.p, "_data.json"))
        files = os.listdir(os.path.join(self.p, "cfq", "splits"))
        for f in files:
            if f.endswith(".json"):
                shutil.move(os.path.join(self.p, "cfq", "splits", f), self.p)
        self.tt.tock()

        self.tt.tick("clean up")
        os.remove(os.path.join(self.p, "cfq.tar.gz"))
        shutil.rmtree(os.path.join(self.p, "cfq"))
        self.tt.tock()

    def load(self, split="random/modent", validfrac=0.1, seed=42, verbose=True, lispify=True, loadunused=False, keepids=False):
        """
                :param split: which split to use (see self.available_splits)
                must be of the form "split/version" where "split" part can be "mcdX" or "random" or other
                and where the "version" part is "full" (original question and query with mids)
                    or "noeid" (placeholders for entities)
                :return:
                """
        parts = split.split("/")
        split = parts[0]
        version = "full" if len(parts) == 1 else parts[1]

        assert split in self.available_splits, f"Split '{split}' not among available splits: {', '.join(self.available_splits)}"
        if verbose:
            print(f"loading split '{split}'")
        split = self.rename[split]
        examples = []
        splitidxs = json.load(open(os.path.join(self.p, f"{split}.json")))
        splitidxs = {self.remap[k]: v for k, v in splitidxs.items()}

        trainidxs = splitidxs["train"]

        if validfrac > 0:
            if verbose:
                print(f"splitting off a random {validfrac*100:.0f}% of 'train' for 'iidvalid' using seed {seed}")
            rnd = random.Random(seed)
            rnd.shuffle(trainidxs)
            numvalid = round(validfrac * len(trainidxs))
            valididxs = trainidxs[:numvalid]
            trainidxs = trainidxs[numvalid:]
            splitidxs["train"] = trainidxs
            splitidxs["iidvalid"] = valididxs

        if "valid" in splitidxs:        # rename provided validation set "oodvalid"
            splitidxs["oodvalid"] = splitidxs["valid"]
            del splitidxs["valid"]
        else:                           # copy "iidvalid" to "oodvalid"
            splitidxs["oodvalid"] = splitidxs["iidvalid"]

        all_lines = open(os.path.join(self.p, "_data.jsonl")).readlines()
        all_lines = [x.strip() for x in all_lines]

        if loadunused:
            usedids = set()
            for splitname, splitids in splitidxs.items():
                usedids |= set(splitids)
            splitidxs["unused"] = []
            for i in range(len(all_lines)):
                if i not in usedids:
                    splitidxs["unused"].append(i)

        for subsetname in splitidxs:
            if verbose:
                print(f"doing '{subsetname}'")
            for i in tqdm(splitidxs[subsetname], disable=not verbose):
                example = self.process_line(all_lines[i], lispify=lispify, version=version)
                if keepids:
                    examples.append((i,) + example + (subsetname,))
                else:
                    examples.append(example + (subsetname,))
        ret = Dataset(examples)
        return ret

    def process_line(self, line, lispify=False, version="full"):
        x = json.loads(line)
        if version == "full":
            ret = (x["question"], x["sparql"])
        elif version in ("modent", "noeid", "entplace", "anonent"):
            ret = (x["questionPatternModEntities"], x["sparqlPatternModEntities"])
        else:
            raise Exception(f"Unrecognized data version: '{version}'")

        if lispify:
            # print(ret[0])
            # print(ret[1])
            tree = sparql_to_tree(ret[1])
            treestr = tree_to_taglisp(tree)
            treestr = treestr.replace("(", " (").replace(")", " ) ")
            tree_recons = taglisp_to_tree(treestr)
            assert(tree == tree_recons)
            treestr = re.sub(r"\s+", " ", treestr)
            treestr = treestr.strip()
            ret = (ret[0], treestr)
            # print(ret[1])
        return ret


def sparql_to_tree(x):
    xs = re.split("([)\n\s+|\(\)])", x)
    # xs = [xse for xse in xs if xse not in {'', ' ', '\n'}]
    _xs = []
    prevspace = False
    for xe in xs:
        if xe in {'', ' ', '\n'}:
            xe = ''
            if prevspace:
                continue
            prevspace = True
        else:
            prevspace = False
        _xs.append(xe)
    xs = _xs

    buffer = []
    inselect = False
    inwhere = False
    inblock = False
    intriplepattern = False

    ORDERLESS = {"@QUERY", "@AND", "@OR", "@WHERE"}

    tree = Tree("@R@", [Tree("@QUERY", [])])
    curnode = tree[0]
    tree[0].parent = tree

    def resolve_buffer(c, b):
        _c = c
        node = Tree("@COND", [])
        c.append(node)
        node.parent = c
        c = node
        appendto = None
        for be in b:
            if be == "":
                pass
            elif be == "|":
                if c[-1].label() != "@OR":      # append next to the or
                    prev = c.pop(-1)
                    node = Tree("@OR", [prev])
                    prev.parent = node
                    c.append(node)
                    node.parent = c
                appendto = c[-1]
            elif be == "(":
                c = c[-1]
            elif be == ")":
                c = c.parent
            else:
                be = Tree(be, [])
                appendto = c if appendto is None else appendto
                appendto.append(be)
                be.parent = appendto
                appendto = None

        if c[0].label() == "filter":
            assert len(c) == 1
            assert c.parent[-1] is c
            c.parent[-1] = c[0]
        return c

    buffer = []
    for xe in xs:
        xe = xe.lower()
        if xe == "select":
            inselect = True
            node = Tree("@SELECT", [])
            curnode.append(node)
            node.parent = curnode
            curnode = node
        elif xe == "where":
            node = resolve_buffer(curnode, buffer)
            curnode[:] = curnode[0][:]
            buffer = []
            curnode = curnode.parent
            inselect = False
            inwhere = True
            node = Tree("@WHERE", [])
            curnode.append(node)
            node.parent = curnode
            curnode = node
        elif xe == "{":  # begin of block
            assert inwhere      # can only be inside "where" clause
            buffer = []
            inblock = True
        elif xe == "}":  # end of block
            node = resolve_buffer(curnode, buffer)
            buffer = []
            inblock = False
            curnode = curnode.parent
            assert curnode.label() == "@QUERY"
        elif xe == ".":     # end of a condition in "where" clause
            node = resolve_buffer(curnode, buffer)
            buffer = []
        else:
            buffer.append(xe)

    return tree


class SCANDatasetLoader(object):
    fullp = "https://raw.githubusercontent.com/brendenlake/SCAN/master/tasks.txt"
    split_names = "random,length,add_jump,add_turn_left"
    available_splits = tuple("random,length,add_jump,add_turn_left,mcd1,mcd2,mcd3".split(","))
    remap = {"train": "train", "valid": "valid", "test": "test", "trainIdxs": "train", "devIdxs": "valid", "testIdxs": "test"}

    def __init__(self, path="../datasets/scan/", **kw):
        super(SCANDatasetLoader, self).__init__(**kw)
        self.p = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(os.path.join(self.p, "all.txt")):
            pass
        else:
            print("file missing, downloading")
            download_url(self.fullp, os.path.join(self.p, "all.txt"))

    def load(self, split="random", validfrac=0.1, seed=42, verbose=True, lispify=True):
        """
        :param split: which split to use (see self.available_splits)
        :return:
        """
        assert split in self.available_splits, f"Split '{split}' not among available splits: {', '.join(self.available_splits)}"
        if verbose:
            print(f"loading split '{split}'")
        examples = []
        splitidxs = json.load(open(os.path.join(self.p, f"{split}.json")))
        splitidxs = {self.remap[k]: v for k, v in splitidxs.items()}

        trainidxs = splitidxs["train"]

        if validfrac > 0:
            if verbose:
                print(f"splitting off a random {validfrac*100:.0f}% of 'train' for 'iidvalid' using seed {seed}")
            rnd = random.Random(seed)
            rnd.shuffle(trainidxs)
            numvalid = round(validfrac * len(trainidxs))
            valididxs = trainidxs[:numvalid]
            trainidxs = trainidxs[numvalid:]
            splitidxs["train"] = trainidxs
            splitidxs["iidvalid"] = valididxs

        if "valid" in splitidxs:        # rename provided validation set "oodvalid"
            splitidxs["oodvalid"] = splitidxs["valid"]
            del splitidxs["valid"]
        else:                           # copy "iidvalid" to "oodvalid"
            splitidxs["oodvalid"] = splitidxs["test"]

        all_lines = open(os.path.join(self.p, "all.txt")).readlines()
        all_lines = [x.strip() for x in all_lines]
        for subsetname in splitidxs:
            if verbose:
                print(f"doing '{subsetname}'")
            for i in tqdm(splitidxs[subsetname], disable=not verbose):
                example = self.process_line(all_lines[i], lispify=lispify)
                examples.append(example + (subsetname,))
        ret = Dataset(examples)
        return ret

    def process_line(self, x:str, lispify=False):
        # print(x)
        assert x[:4] == "IN: "
        x = x[4:]
        splits = x.split(" OUT: ")
        assert len(splits) == 2
        inp, out = splits
        inp, out = inp.strip(), out.strip()
        if lispify:
            out = f"(@R@ {out} )"
        return (inp, out)

    def build_index_sets(self):
        splitnames = self.split_names.split(",")
        # load lines from all.txt
        try:
            all_lines = open(os.path.join(self.p, "all.txt")).readlines()
            all_lines = [x.strip() for x in all_lines]
            all_lines_D = dict(zip(all_lines, range(len(all_lines))))
            print(f"total: {len(all_lines_D)}")
            print(f"total unique: {len(set(all_lines_D.keys()))}")
            for splitname in splitnames:
                split_indexes = {"train": [], "test": []}
                print(f"doing {splitname}")
                trainlines = open(os.path.join(self.p, f"{splitname}_train.txt")).readlines()
                trainlines = [x.strip() for x in trainlines]
                testlines = open(os.path.join(self.p, f"{splitname}_test.txt")).readlines()
                testlines = [x.strip() for x in testlines]
                for trainline in trainlines:
                    assert trainline in all_lines_D, "line not in all lines"
                    split_indexes["train"].append(all_lines_D[trainline])
                for testline in testlines:
                    assert testline in all_lines_D, "line not in all lines"
                    split_indexes["test"].append(all_lines_D[testline])
                for k in split_indexes:
                    split_indexes[k] = sorted(split_indexes[k])
                print(f"num_train: {len(split_indexes['train'])}/{len(set(split_indexes['train']))}")
                print(f"num_test: {len(split_indexes['test'])}/{len(set(split_indexes['test']))}")
                print(f"num_total: {len(split_indexes['test']) + len(split_indexes['train'])}")
                json.dump(split_indexes, open(os.path.join(self.p, f"{splitname}.json"), "w"))
        except Exception as e:
            raise e


class COGSDatasetLoader(object):
    mainp = "https://raw.githubusercontent.com/najoungkim/COGS/main/data/"
    mapp = {"train": "train.tsv", "iidvalid": "dev.tsv", "iidtest": "test.tsv", "oodtest": "gen.tsv"}

    def __init__(self, path="../datasets/cogs/", **kw):
        """
        :param path:            where to find/store all dataset files
        """
        super(COGSDatasetLoader, self).__init__(**kw)
        self.p = path
        for k in self.mapp:
            if not os.path.exists(os.path.join(self.p, self.mapp[k])):
                print(f"File for '{k}' split missing, downloading from {self.mainp + self.mapp[k]}")
                download_url(self.mainp+self.mapp[k], os.path.join(self.p, self.mapp[k]))

    def load(self, oodvalidsize=3000, seed=42, verbose=True, lispify=True):
        examples = []

        # load all examples
        if verbose:
            print("loading all examples")
        for k in self.mapp:
            for line in open(os.path.join(self.p, self.mapp[k])).readlines():
                splits = line.strip().split("\t")
                assert(len(splits) == 3)
                examples.append((splits[0], splits[1], splits[2], k))

        # split off an OOD valid set from the OOD test set
        if verbose:
            print(f"splitting off {oodvalidsize} examples from oodtest for oodvalid")
        oodtestidxs = []
        for i, example in enumerate(examples):
            if example[3] == "oodtest":
                oodtestidxs.append(i)

        assert oodvalidsize > 0 and oodvalidsize < len(oodtestidxs)

        rs = random.Random(seed)
        rs.shuffle(oodtestidxs)
        oodvalididxs = set(oodtestidxs[:oodvalidsize])
        # oodtestidxs = oodtestidxs[oodvalidsize:]

        i = 0
        while i < len(examples):
            if i in oodvalididxs:
                examples[i] = (examples[i][0], examples[i][1], examples[i][2], "oodvalid")
            i += 1

        counts = {"train": 0, "iidvalid": 0, "oodvalid": 0, "iidtest": 0, "oodtest": 0}
        for example in examples:
            counts[example[3]] += 1

        if verbose:
            for k, v in counts.items():
                print(f"Size of {k} set:\t{v}")

        fldic = Vocab()

        # run string-level fixes on logical forms
        traininplens, trainoutlens, testinplens, testoutlens = [], [], [], []
        for i in tqdm(range(len(examples))):
            lf = examples[i][1]
            fixedlf = self.fix_lf_string(lf)
            tree = self.lfstr_to_tree(fixedlf)
            fl = self.tree_to_shortlisp(tree)
            for token in fl.split(" "):
                fldic.add_token(token, seen=examples[i][3] == "train")
            examples[i] = (examples[i][0], fl, examples[i][2], examples[i][3])
            if examples[i][3] == "train":
                traininplens.append(len(examples[i][0].split(" ")))
                trainoutlens.append(len(examples[i][1].split(" ")))
            elif examples[i][3].endswith("test"):
                testinplens.append(len(examples[i][0].split(" ")))
                testoutlens.append(len(examples[i][1].split(" ")))
        print("Max lens during training:")
        print(max(traininplens), max(trainoutlens))
        print("Max lens during testing:")
        print(max(testinplens), max(testoutlens))
        fldic.finalize()
        return examples

    @staticmethod
    def lfstr_to_tree(x:str):
        splits = [xe.strip() for xe in x.split("AND")]
        conditions = [prolog_to_tree(xe) for xe in splits]
        # reorder so that unaries are first
        unaries = []
        others = []
        for cond in conditions:
            if len(cond) == 1:
                unaries.append(cond)
            else:
                others.append(cond)
        conditions = unaries + others
        ret = Tree("@AND@", children=conditions)
        return ret

    @classmethod
    def tree_to_shortlisp(cls, x:Tree, variable_children=("@AND@",)):
        # e.g. V-AND 1ary-person X1 2ary-father X1 X2 )
        if isinstance(x, str):
            return x
        elif len(x) == 0:
            return x.label()
        else:
            assert isinstance(x, Tree)
            tail = " ".join([cls.tree_to_shortlisp(xe) for xe in x])
            if x.label() in variable_children:
                ret = f"D-{x.label()} {tail} )"
            else:
                ret = f"{len(x)}ary-{x.label()} " + tail
            return ret

    @classmethod
    def shortlisp_to_tree(cls, x:str):
        xs = x.split(" ")
        ret, tail = cls._shortlisp_to_tree(xs)
        return ret

    @classmethod
    def _shortlisp_to_tree(cls, x:List[str], arityparent=-1):
        head, *tail = x
        if re.match(r"\d+ary-.+", head) or re.match("D-.+", head):
            m = re.match(r"(\d+)ary-(.+)", head)
            m2 = re.match(r"D-(.+)", head)
            if m:
                arity, label = int(m.group(1)), m.group(2)
            elif m2:
                arity, label = -1, m2.group(1)
            children = []
            while True:
                if arity == 0:
                    break
                if arity < 0:
                    assert len(tail) > 0
                    if tail[0] == ")":
                        _, *tail = tail
                        break
                child, tail = cls._shortlisp_to_tree(tail, arityparent=arity)
                children.append(child)
                arity -= 1
            ret = Tree(label, children=children)
            return ret, tail
        else:
            child = Tree(head, children=[])
            return child, tail


    def fix_lf_string(self, x:str):
        _x = x
        _x = re.sub(r"x\s_\s(\d+)", r"x_\1", _x)
        # _x = re.sub(r"([a-z]+)\s\.\s([a-z]+)", r"\1.\2", _x)
        if "LAMBDA" in _x:
            _x = re.sub(r"LAMBDA\s[a-z]\s\.", "", _x)
        _x = re.sub(r"\s\.\s", ".", _x)
        _x = re.sub(r"\*", "", _x)
        _x = re.sub(r";", " AND ", _x)
        _x = re.sub("\s+", " ", _x).strip()
        # _x = _x.split(" ")
        return _x


# endregion
def try_cfq():
    dsl = CFQDatasetLoader()
    data = dsl.load("mcd1/modent", loadunused=True)


def try_scan():
    scandsl = SCANDatasetLoader()
    # scandsl.build_index_sets()
    ds = scandsl.load("length")
    print(ds[0])


def try_tokenizer_dataset():
    from transformers import BartTokenizer

    ovd = OvernightDatasetLoader().load()
    seqenc = SequenceEncoder(tokenizer=tree_to_lisp_tokens)
    for example in ovd.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=example[2] == "train")
    seqenc.finalize_vocab()
    nl_tokenizer = BartTokenizer.from_pretrained("bart-large")
    def tokenize(x):
        ret = [xe for xe in x]
        ret.append(nl_tokenizer.tokenize(ret[0]))
        ret.append(nl_tokenizer.encode(ret[0], return_tensors="pt"))
        ret.append(seqenc.convert(ret[1], return_what="tensor")[0][None])
        return ret
    ovd = ovd.map(tokenize)
    print(ovd[0])


def try_perturbed_generated_dataset():
    torch.manual_seed(1234)
    ovd = OvernightDatasetLoader().load()
    govd = PCFGDataset(OvernightPCFGBuilder()
                       .build(ovd[(None, None, lambda x: x in {"train", "valid"})]
                              .map(lambda f: f[1]).examples),
                       N=10000)

    print(govd[0])
    # print(govd[lambda x: True][0])

    # print(govd[:])
    # create vocab from pcfg
    vocab = build_vocab_from_pcfg(govd._pcfg)
    seqenc = SequenceEncoder(vocab=vocab, tokenizer=tree_to_lisp_tokens)
    spanmasker = SpanMasker(seed=12345667)
    treemasker = SubtreeMasker(p=.05, seed=2345677)

    perturbed_govd = govd.cache()\
        .map(lambda x: (seqenc.convert(x, "tensor"), x)) \
        .map(lambda x: x + (seqenc.convert(x[-1], "tokens"),)) \
        .map(lambda x: x + (spanmasker(x[-1]),)) \
        .map(lambda x: x + (seqenc.convert(x[-1], "tensor"),)) \
        .map(lambda x: (x[-1], x[0]))

    dl = DataLoader(perturbed_govd, batch_size=10, shuffle=True, collate_fn=pad_and_default_collate)
    batch = next(iter(dl))
    print(batch)
    print(vocab.tostr(batch[0][1]))
    print(vocab.tostr(batch[1][1]))

    tt = q.ticktock()
    tt.tick("first run")
    for i in range(10000):
        y = perturbed_govd[i]
        if i < 10:
            print(f"{y[0]}\n{y[-2]}")
    tt.tock("first run done")
    tt.tick("second run")
    for i in range(10000):
        y = perturbed_govd[i]
        if i < 10:
            print(f"{y[0]}\n{y[-2]}")
    tt.tock("second run done")


def try_top_dataset():
    ds = TOPDatasetLoader().load()
    print(ds[(None, None, "train")][0])
    gds = PCFGDataset(PCFGBuilder()
                      .build(ds[lambda x: x[2] in {"train", "valid"}]
                             .map(lambda f: f[1]).examples))

    def tree_to_str(x:Tree):
        toks = tree_to_lisp_tokens(x, brackets="[]")
        toks = " ".join(toks)
        toks = toks.replace("[ ", "<[").replace("]", "]>")
        return toks

    fl_tokens = {"]>",}
    for example in ds[lambda x: x[-1] == "train"]\
            .map(lambda x: (tree_to_str(x[1]),)).examples:
        example_toks = example[0].split(" ")
        for tok in example_toks:
            if tok.startswith("<[SL:") or tok.startswith("<[IN:"):
                fl_tokens.add(tok)

    print(f"extra tokens ({len(fl_tokens)}):")
    for fl_token in fl_tokens:
        print(fl_token)

    nl_tokenizer = BartTokenizer.from_pretrained("bart-large")
    fl_tokenizer = BartTokenizer.from_pretrained("bart-large")
    fl_tokenizer.add_special_tokens({"additional_special_tokens": list(fl_tokens)})

    pl = Pipeline()
    pl = pl.add(lambda x: (nl_tokenizer.tokenize(" " + x[0]),
                            nl_tokenizer.tokenize(" " + tree_to_str(x[1])),
                            fl_tokenizer.tokenize(" " + tree_to_str(x[1])),
                            x[-1]))
    # print(json.dumps(mds[0], indent=4))
    tds = ds[lambda x: x[-1] == "test"]
    for i in range(10):
        x = tds.map(pl)[i]
        print(i)
        print(x[0])
        print(x[1])
        print(x[2])
        
    pl2 = Pipeline()
    pl2 = pl2.add(pl)
    pl2 = pl2.add(lambda x: (nl_tokenizer.encode(x[0], return_tensors="pt"),
                              nl_tokenizer.encode(x[1], return_tensors="pt"),
                              fl_tokenizer.encode(x[2], return_tensors="pt"),
                              x[-1]))

    for i in range(10):
        x = tds.map(pl2)[i]
        print(i)
        print(nl_tokenizer.decode(x[0][0]))
        print(nl_tokenizer.decode(x[1][0]))
        print(fl_tokenizer.decode(x[2][0]))

    tds = ds[(None, None, "test")]
    xds = tds.map(pl)
    xds2 = tds.map(pl2)
    for i in tqdm(range(len(xds))):
        oinp = tree_to_str(tds[i][1])
        inp = " ".join(xds[i][2])
        inp2 = fl_tokenizer.decode(xds2[i][2][0])
        assert("<pad>" not in inp2)
        assert("<unk>" not in inp2)
        inp2 = inp2.replace("<s>", "").replace("</s>", "").strip()
        if not oinp.lower().replace(" ", "") == inp2.lower().replace(" ", ""):
            _inp2 = fl_tokenizer.decode(xds2[i][2][0])
            assert(oinp.lower().replace(" ", "") == inp2.lower().replace(" ", ""))



    print(nl_tokenizer.get_vocab())
    print(gds._pcfg.productions)


def try_iterable_ds():
    class RandomIterableDataset(IterableDataset):
        def __init__(self, seed=42, **kw):
            super(RandomIterableDataset, self).__init__(**kw)
            self.seed = seed
            self.rng = random.Random(seed)

        def reset_seed(self):
            self.rng = random.Random(self.seed)

        def __next__(self):
            return self.rng.random()

    baseds = RandomIterableDataset()
    # dsiter = iter(baseds)
    print(len(baseds))
    print("random values")
    first = []
    for i in range(10):
        x = next(baseds)
        print(x)
        first.append(x)

    # dsiter = iter(baseds)
    baseds.reset_seed()
    second = []
    print("random values again")
    for i in range(10):
        x = next(baseds)
        print(x)
        second.append(x)

    assert np.allclose(first, second)
    print("reset seed works")

    baseds.reset_seed()
    fds = baseds.filter(lambda x: x > 0.5)
    first = []
    for i in range(10):
        x = next(fds)
        print(x)
        first.append(x)
    print("filtering works")

    baseds.reset_seed()
    mfds = fds.map(lambda x: x - 0.5)
    first = []
    for i in range(10):
        x = next(mfds)
        print(x)
        first.append(x)
    print("mapping works")

    baseds.reset_seed()
    fmfds = mfds.filter(lambda x: x*4 < 0.5)
    first = []
    for i in range(10):
        x = next(fmfds)
        print(x)
        first.append(x)
    print("mapping works")


def try_cogs():
    dl = COGSDatasetLoader()

    # try shortlisp
    treestr = "(@AND@ (person x1) (thing x2) (owner x1 x2) (location x2 (under x3)))"
    tree = lisp_to_tree(treestr)
    print(tree)
    shortlisp = dl.tree_to_shortlisp(tree)
    print(shortlisp)
    rtree = dl.shortlisp_to_tree(shortlisp)
    print(rtree)

    assert(tree == rtree)

    print("done")

    dl.load()




if __name__ == '__main__':
    # import filelock
    # try_tokenizer_dataset()
    # try_perturbed_generated_dataset()
    # try_top_dataset()
    # ovd = OvernightDatasetLoader(usecache=False, simplify_mode="light").load()
    # govd = PCFGDataset(OvernightPCFGBuilder()
    #                    .build(ovd[lambda x: x[2] in {"train", "valid"}]
    #                           .map(lambda f: f[1])
    #                           .examples))
    # print(ovd[0])
    # print(govd[0])
    # print(govd.examples)
    # print(try_multilingual_geoquery_dataset_loader())
    # try_scan()
    try_cfq()
    # try_iterable_ds()
    # try_cogs()