import csv
import json
import os
import random
import timeit
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

from parseq.grammar import lisp_to_pas, pas_to_tree, tree_size, tree_to_lisp, tree_to_lisp_tokens
from parseq.vocab import SequenceEncoder, Vocab
from transformers import BartTokenizer


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
        return self

    def __len__(self):
        return len(self._examples)

    def _filter_inner(self, ex, f, ret):
        if isinstance(f, Callable) and f(ex):
            ret.append(ex)
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
                ret.append(ex)
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
                ret.append(ex)
        return ret

    def filter(self, f):
        ret = []
        for ex in self._examples:
            ret = self._filter_inner(ex, f, ret)
        ret = Dataset(ret)
        return ret

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            ret = self.filter(item)
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


class CachedDataset(object):
    def __init__(self, use_cache=False, **kw):
        super(CachedDataset, self).__init__(**kw)
        self.use_cache = use_cache
        self._examples_cache = {}
        self.baseds = None

    def cache(self):
        """ Enable cache at this level. """
        self.enable_cache()
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

    @property
    def rootds(self):
        return self.baseds.rootds

    def filter(self, f):
        newbase = self.baseds.filter(lambda x: f(self.f(x)))
        ret = newbase.map(self.f)
        return ret

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            ret = self.filter(item)
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
            ret = self._filter_inner(ex, f, ret)
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
            children = [x.label()] + children
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
        if isinstance(yi[0], torch.LongTensor) and yi[0].dim() == 1:
            y[i] = q.pad_tensors(yi, 0, pad_value)
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.Tensor):
            yi = [yij[None] for yij in yi]
            y[i] = torch.cat(yi, 0)
    return y


def pad_and_default_collate(x, pad_value=0):
    y = list(zip(*x))
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.LongTensor) and yi[0].dim() == 1:
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
class OvernightDatasetLoader(object):
    def __init__(self,
                 p="../datasets/overnightData/",
                 pcache="../datasets/overnightCache/",
                 usecache=True,
                 validfrac=.2,
                 simplify_mode="full", **kw):
        super(OvernightDatasetLoader, self).__init__(**kw)
        self._simplify_filters = True        # if True, filter expressions are converted to orderless and-expressions
        self._pcache = pcache if usecache else None
        self._usecache = usecache
        self.validfrac = validfrac
        self._p = p
        self.simplify_mode = simplify_mode      # "full" or "light"

    @property
    def full_simplify(self):
        return self.simplify_mode == "full"

    def load(self, domain:str="restaurants"):
        examples = self._initialize(self._p, domain)
        return Dataset(examples)

    def lines_to_examples(self, lines:List[str]):
        maxsize_before = 0
        avgsize_before = []
        maxsize_after = 0
        avgsize_after = []
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
                            print(f"SAME!: {conditions[i]}, {conditions[j]}")
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
                    return t
            else:
                t[:] = [simplify_furthermore(tc) for tc in t]
                return t

        def simplify_final(t):
            assert(t.label() == "SW:listValue")
            assert(len(t) == 1)
            return t[0]

        ret = []
        ltp = None
        j = 0
        for i, line in enumerate(lines):
            z, ltp = lisp_to_pas(line, ltp)
            if z is not None:
                print(f"Example {j}:")
                ztree = pas_to_tree(z[1][2][1][0])
                maxsize_before = max(maxsize_before, tree_length(ztree))
                avgsize_before.append(tree_length(ztree))
                lf = ztree
                if self.simplify_mode != "none":
                    lf = simplify_tree(lf)
                    lf = simplify_further(lf)
                    lf = simplify_furthermore(lf)
                    lf = simplify_final(lf)
                question = z[1][0][1][0]
                assert(question[0] == '"' and question[-1] == '"')
                ret.append((question[1:-1], lf))
                print(ret[-1][0])
                print(ret[-1][1])
                ltp = None
                maxsize_after = max(maxsize_after, tree_length(lf))
                avgsize_after.append(tree_length(lf))

                print(pas_to_tree(z[1][2][1][0]))
                print()
                j += 1

        avgsize_before = sum(avgsize_before) / len(avgsize_before)
        avgsize_after = sum(avgsize_after) / len(avgsize_after)

        print(f"Simplification results ({j} examples):")
        print(f"\t Max, Avg size before: {maxsize_before}, {avgsize_before}")
        print(f"\t Max, Avg size after: {maxsize_after}, {avgsize_after}")

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

    def _initialize(self, p, domain):
        self.data = {}

        trainexamples, testexamples = None, None
        if self._usecache:
            try:
                trainexamples, testexamples = self._load_cached(domain)
            except (IOError, ValueError) as e:
                pass

        if trainexamples is None:

            trainlines = [x.strip() for x in
                         open(os.path.join(os.path.dirname(__file__), p, f"{domain}.paraphrases.train.examples"), "r").readlines()]
            testlines = [x.strip() for x in
                        open(os.path.join(os.path.dirname(__file__), p, f"{domain}.paraphrases.test.examples"), "r").readlines()]

            trainexamples = self.lines_to_examples(trainlines)
            testexamples = self.lines_to_examples(testlines)

            if self._usecache:
                self._cache(domain, trainexamples, testexamples)

        questions, queries = tuple(zip(*(trainexamples + testexamples)))
        trainlen = int(round((1-self.validfrac) * len(trainexamples)))
        validlen = len(trainexamples) - trainlen
        splits = ["train"] * trainlen + ["valid"] * validlen
        rng = np.random.RandomState(12345678)
        rng.shuffle(splits)
        assert(len(splits) == len(trainexamples))
        splits = splits + ["test"] * len(testexamples)

        examples = list(zip(questions, queries, splits))
        return examples


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


# endregion


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

if __name__ == '__main__':
    # import filelock
    # try_tokenizer_dataset()
    # try_perturbed_generated_dataset()
    # try_top_dataset()
    ovd = OvernightDatasetLoader(usecache=False, simplify_mode="light").load()
    # govd = PCFGDataset(OvernightPCFGBuilder()
    #                    .build(ovd[lambda x: x[2] in {"train", "valid"}]
    #                           .map(lambda f: f[1])
    #                           .examples))
    # print(ovd[0])
    # print(govd[0])
    # print(govd.examples)