import os
import random
from abc import abstractmethod
from typing import List, Tuple, Callable

import nltk
import ujson
from nltk import Tree, Nonterminal
import numpy as np
from nltk.parse.generate import generate

from parseq.grammar import lisp_to_pas, pas_to_tree, tree_size, tree_to_lisp, tree_to_lisp_tokens
from parseq.vocab import SequenceEncoder, Vocab


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

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        if isinstance(item, Callable):
            ret = [rete for rete in self._examples if item(rete)]
            return Dataset(ret)
        else:
            ret = self._examples[item]
            return ret

    def map(self, f):
        """
        Create a MappedDataset that will apply given function f on an example on __getitem__
        """
        ret = MappedDataset(self, f)
        return ret


class MappedDataset(Dataset):
    def __init__(self, baseds, f, use_cache=True, **kw):
        super(MappedDataset, self).__init__(**kw)
        self.baseds = baseds
        self.f = f
        self.use_cache = use_cache
        self._examples_cache = {}

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
        if isinstance(self.baseds, MappedDataset):
            self.baseds.deep_disable_cache()
        return self

    def deep_enable_cache(self):
        self.enable_cache()
        if isinstance(self.baseds, MappedDataset):
            self.baseds.deep_enable_cache()
        return self

    def __len__(self):
        return len(self.baseds)

    def __getitem__(self, item):
        if isinstance(item, str):
            ret = self.baseds[item]
            return MappedDataset(ret, self.f)
        else:
            if item in self._examples_cache and self.use_cache:
                ret = self._examples_cache[item]
            else:
                example = self.baseds[item]
                ret = self.f(example)
                if self.use_cache:
                    self._examples_cache[item] = ret
            return ret


class GeneratedDataset(Dataset):
    def __init__(self, N=np.infty, seed=12345678, **kw):
        super(GeneratedDataset, self).__init__(**kw)
        self.seed = seed
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if isinstance(item, Callable):
            pass    # TODO
            raise NotImplemented()
        else:
            rng = np.random.RandomState(self.seed + item)
            return self.generate(rng=rng)

    @property
    def examples(self):
        for i in range(self.N):
            yield self[i]

    @abstractmethod
    def generate(self, rng=None): pass

    def map(self, f):
        return GeneratedMappedDataset(self, f)


class GeneratedMappedDataset(MappedDataset):
    @property
    def examples(self):
        for i in range(self.N):
            yield self[i]


class PCFGDataset(GeneratedDataset):
    def __init__(self, pcfg, N=np.infty, seed=12345678, **kw):
        super(PCFGDataset, self).__init__(N=N, seed=seed, **kw)
        self._pcfg = pcfg

    def generate(self, start=None, rng=None):
        rng = np.random.RandomState(self.seed) if rng is None else rng
        start = self._pcfg.start() if start is None else start
        productions = self._pcfg.productions(start)
        productions, probs = zip(*[(prod, prod.prob()) for prod in productions])
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
        return ret


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


class SpanMasker(object):
    mask_symbol = "@MASK@"
    def __init__(self, p=.1, lamda=1.2, seed=None, **kw):
        """
        :param p: probability of replacing a span with mask symbol
        :param lamda: parameter for Poisson distribution on the length of span (0 ==> length 1 always)
        :param kw:
        """
        super(SpanMasker, self).__init__(**kw)
        self.lamda = lamda
        self.p = p
        self.rng = np.random.RandomState(seed)

    def __call__(self, tokens:List[str]):
        ret = []
        i = 0
        while i < len(tokens):
            if self.p < np.random.random():
                l = np.random.poisson(self.lamda)
                ret.append(self.mask_symbol)
                i += l
            else:
                ret.append(tokens[i])
                i += 1
        return ret



class OvernightDatasetLoader(object):
    def __init__(self,
                 p="../datasets/overnightData/",
                 pcache="../datasets/overnightCache/",
                 usecache=True,
                 validfrac=.2, **kw):
        super(OvernightDatasetLoader, self).__init__(**kw)
        self._simplify_filters = True        # if True, filter expressions are converted to orderless and-expressions
        self._pcache = pcache if usecache else None
        self._usecache = usecache
        self.validfrac = validfrac
        self._p = p

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
            if t.label() == "call":
                assert(len(t[0]) == 0)
                # if not t[0].label().startswith("SW."):
                #     print(t)
                # assert(t[0].label().startswith("SW."))
                t.set_label(t[0].label())
                del t[0]
            elif t.label() == "string":
                afterstring.update(set([tc.label() for tc in t]))
                assert(len(t) == 1)
                assert(len(t[0]) == 0)
                t.set_label(f"arg:{t[0].label()}")
                del t[0]
            if t.label().startswith("edu.stanford.nlp.sempre.overnight.SimpleWorld."):
                t.set_label("SW:" + t.label()[len("edu.stanford.nlp.sempre.overnight.SimpleWorld."):])
            if t.label() == "SW:getProperty":
                assert(len(t) == 2)
                ret = simplify_tree(t[1])
                ret.append(simplify_tree(t[0]))
                return ret
            elif t.label() == "SW:singleton":
                assert(len(t) == 1)
                assert(len(t[0]) == 0)
                return t[0]
            elif t.label() == "SW:ensureNumericProperty":
                assert(len(t) == 1)
                return simplify_tree(t[0])
            elif t.label() == "SW:ensureNumericEntity":
                assert(len(t) == 1)
                return simplify_tree(t[0])
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
                maxsize_before = max(maxsize_before, tree_size(ztree))
                avgsize_before.append(tree_size(ztree))
                lf = simplify_tree(ztree)
                lf = simplify_further(lf)
                lf = simplify_furthermore(lf)
                lf = simplify_final(lf)
                question = z[1][0][1][0]
                assert(question[0] == '"' and question[-1] == '"')
                ret.append((question[1:-1], lf))
                print(ret[-1][0])
                print(ret[-1][1])
                ltp = None
                maxsize_after = max(maxsize_after, tree_size(lf))
                avgsize_after.append(tree_size(lf))

                print(pas_to_tree(z[1][2][1][0]))
                print()
                j += 1

        avgsize_before = sum(avgsize_before) / len(avgsize_before)
        avgsize_after = sum(avgsize_after) / len(avgsize_after)

        print("Simplification results ({j} examples):")
        print(f"\t Max, Avg size before: {maxsize_before}, {avgsize_before}")
        print(f"\t Max, Avg size after: {maxsize_after}, {avgsize_after}")

        return ret

    def _load_cached(self, domain):
        train_cached = ujson.load(open(os.path.join(self._pcache, f"{domain}.train.json"), "r"))
        trainexamples = [(x, Tree.fromstring(y)) for x, y in train_cached]
        test_cached = ujson.load(open(os.path.join(self._pcache, f"{domain}.test.json"), "r"))
        testexamples = [(x, Tree.fromstring(y)) for x, y in test_cached]
        print("loaded from cache")
        return trainexamples, testexamples

    def _cache(self, trainexamples:List[Tuple[str, Tree]], testexamples:List[Tuple[str, Tree]]):
        train_cached, test_cached = None, None
        if os.path.exists(os.path.join(self._pcache, f"{self._domain}.train.json")):
            try:
                train_cached = ujson.load(open(os.path.join(self._pcache, f"{self._domain}.train.json"), "r"))
                test_cached = ujson.load(open(os.path.join(self._pcache, f"{self._domain}.test.json"), "r"))
            except (IOError, ValueError) as e:
                pass
        trainexamples = [(x, str(y)) for x, y in trainexamples]
        testexamples = [(x, str(y)) for x, y in testexamples]

        if train_cached != trainexamples:
            with open(os.path.join(self._pcache, f"{self._domain}.train.json"), "w") as f:
                ujson.dump(trainexamples, f, indent=4, sort_keys=True)
        if test_cached != testexamples:
            with open(os.path.join(self._pcache, f"{self._domain}.test.json"), "w") as f:
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
                         open(os.path.join(p, f"{domain}.paraphrases.train.examples"), "r").readlines()]
            testlines = [x.strip() for x in
                        open(os.path.join(p, f"{domain}.paraphrases.test.examples"), "r").readlines()]

            trainexamples = self.lines_to_examples(trainlines)
            testexamples = self.lines_to_examples(testlines)

            if self._usecache:
                self._cache(trainexamples, testexamples)

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


class PCFGBuilder(object):
    def __init__(self, orderless=tuple(), **kw):
        super(PCFGBuilder, self).__init__(**kw)
        self.orderless = set(orderless)

    def build(self, examples=tuple()):
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


class OvernightPCFGBuilder(PCFGBuilder):
    def __init__(self, **kw):
        super(OvernightPCFGBuilder, self).__init__(("op:and",), **kw)


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
    ovd = OvernightDatasetLoader().load()
    govd = PCFGDataset(OvernightPCFGBuilder()
                       .build(ovd[lambda x: x[2] in {"train", "valid"}]
                              .map(lambda f: f[1])
                              .examples))
    # create vocab from pcfg
    vocab = build_vocab_from_pcfg(govd._pcfg)
    seqenc = SequenceEncoder(vocab=vocab, tokenizer=tree_to_lisp_tokens)
    masker = SpanMasker(seed=12345)

    perturbed_govd = govd\
        .map(lambda x: tuple(x) + (seqenc.convert(x[1], "tokens"),))\
        .map(lambda x: tuple(x) + (masker(x[-1]),))\
        .map(lambda x: tuple(x) + (seqenc.convert(x[-1], "tensor"),))\
        .deep_disable_cache()

    print(perturbed_govd[0])




if __name__ == '__main__':
    # import filelock
    # try_tokenizer_dataset()
    try_perturbed_generated_dataset()
    # ovd = OvernightDatasetLoader().load()
    # govd = PCFGDataset(OvernightPCFGBuilder()
    #                    .build(ovd[lambda x: x[2] in {"train", "valid"}]
    #                           .map(lambda f: f[1])
    #                           .examples))
    # print(ovd[0])
    # print(govd[0])
    # print(govd.examples)