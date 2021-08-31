import os
import random
from typing import List

from nltk import Nonterminal, ProbabilisticProduction, PCFG, Tree
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import shelve
import numpy as np
import torch
import qelos as q


from parseq.datasets import Dataset, SCANDatasetLoader, CFQDatasetLoader, autocollate, IterableDataset, PCFGBuilder
from parseq.grammar import lisp_to_tree
from parseq.vocab import Vocab

from transformers import AutoTokenizer


class Tokenizer(object):
    def __init__(self, toktype="vanilla", inpvocab:Vocab=None, outvocab:Vocab=None, **kw):
        super(Tokenizer, self).__init__(**kw)
        if toktype.startswith("bert"):
            self.berttok = AutoTokenizer.from_pretrained(toktype)
            assert inpvocab == None
            self._inpvocab = self.berttok.vocab
        else:
            self.berttok = None
            self._inpvocab = inpvocab
        self.outvocab = outvocab

    @property
    def inpvocab(self):
        return self._inpvocab

    @inpvocab.setter
    def inpvocab(self, value):
        if self.berttok is not None:
            pass
        else:
            self._inpvocab = value

    def tokenize(self, inps, outs):
        if inps is not None:
            if self.berttok is not None:
                inptoks = self.berttok.tokenize(inps)
            else:
                inptoks = ["@START@"] + self.get_toks(inps) + ["@END@"]
        else:
            inptoks = None
        if outs is not None:
            # outtoks = ["@START@"] + self.get_toks(outs) + ["@END@"]
            outtoks = self.get_toks(outs)
        else:
            outtoks = None
        return inptoks, outtoks

    def tensorize(self, inptoks, outtoks):
        if inptoks is not None:
            if self.berttok is not None:
                inptensor = self.berttok.encode(inptoks, return_tensors="pt")[0]    # TODO
            else:
                inptensor = self.tensorize_output(inptoks, self._inpvocab)
        else:
            inptensor = None
        if outtoks is not None:
            outtensor = self.tensorize_output(outtoks, self.outvocab)
        else:
            outtensor = None
        return inptensor, outtensor

    def tokenize_and_tensorize(self, inps, outs):
        inptoks, outtoks = self.tokenize(inps, outs)
        inptensors, outtensors = self.tensorize(inptoks, outtoks)
        return inptensors, outtensors
        # if inps is not None:
        #     if self.berttok is not None:
        #         inptoks = self.berttok.tokenize(inps)
        #     else:
        #         inptoks = ["@START@"] + self.get_toks(inps) + ["@END@"]
        #
        #     if self.berttok is not None:
        #         inptensor = self.berttok.encode(inps, return_tensors="pt")[0]
        #     else:
        #         inptensor = self.tensorize_output(inptoks, self._inpvocab)
        # else:
        #     inptoks, inptensor = None, None
        #
        # if outs is not None:
        #     # outtoks = ["@START@"] + self.get_toks(outs) + ["@END@"]
        #     outtoks = self.get_toks(outs)
        #     outtensor = self.tensorize_output(outtoks, self.outvocab)
        # else:
        #     outtoks, outtensor = None, None
        #
        # ret = {"inps": inps, "outs":outs, "inptoks": inptoks, "outtoks": outtoks,
        #        "inptensor": inptensor, "outtensor": outtensor}
        # ret = (ret["inptensor"], ret["outtensor"])
        # return ret

    def get_toks(self, x):
        return x.strip().split(" ")

    def tensorize_output(self, x, vocab):
        ret = [vocab[xe] for xe in x]
        ret = torch.tensor(ret)
        return ret


ORDERLESS = {"@WHERE", "@OR", "@AND", "@QUERY", "(@WHERE", "(@OR", "(@AND", "(@QUERY"}


def load_ds(dataset="scan/random", tokenizer="vanilla", validfrac=0.1, splitseed=42, recompute=False):
    l = locals().copy()
    tt = q.ticktock("data")

    assert tokenizer == "vanilla"

    del l["recompute"]
    tt.tick(f"loading '{dataset}'")
    if dataset.startswith("cfq/") or dataset.startswith("scan/mcd"):
        del l["validfrac"]
        del l["splitseed"]
        print(f"validfrac is ineffective with dataset '{dataset}'")

    key = str(l)

    shelfname = os.path.basename(__file__) + ".ds.cache.shelve"
    if not recompute:
        tt.tick(f"loading from shelf (key '{key}')")
        with shelve.open(shelfname) as shelf:
            if key not in shelf:
                recompute = True
                tt.tock("couldn't load from shelf")
            else:
                shelved = shelf[key]
                trainex, validex, testex, tokenizer, validoverlaps, testoverlaps = \
                    [shelved[x] for x in ["trainex", "validex", "testex", "tokenizer", "validoverlaps", "testoverlaps"]]
                trainds, validds, testds = Dataset(trainex), Dataset(validex), Dataset(testex)
                tt.tock("loaded from shelf")

                tt.msg("Stored overlap stats:")
                tt.msg("Overlap of validation with train:")
                print(json.dumps(validoverlaps, indent=4))
                tt.msg("Overlap of test with train:")
                print(json.dumps(testoverlaps, indent=4))

    if recompute:
        tt.tick("loading data")
        splits = dataset.split("/")
        dataset, splits = splits[0], splits[1:]
        split = "/".join(splits)
        if dataset == "scan":
            ds = SCANDatasetLoader().load(split, validfrac=validfrac, seed=splitseed)
        elif dataset == "cfq":
            ds = CFQDatasetLoader().load(split + "/modent")
        else:
            raise Exception(f"Unknown dataset: '{dataset}'")
        tt.tock("loaded data")

        tt.tick("creating tokenizer")
        tokenizer = Tokenizer(toktype="vanilla")
        tt.tock("created tokenizer")

        print(len(ds))

        tt.tick("dictionaries")
        inpdic = Vocab()
        inplens, outlens = [0], []
        fldic = Vocab()
        for x in ds:
            outtoks = tokenizer.get_toks(x[1])
            outlens.append(len(outtoks))
            for tok in outtoks:
                fldic.add_token(tok, seen=x[2] == "train")
            inptoks = tokenizer.get_toks(x[0])
            for tok in inptoks:
                inpdic.add_token(tok, seen=x[2] == "train")
        inpdic.finalize(min_freq=0, top_k=np.infty)
        fldic.finalize(min_freq=0, top_k=np.infty)
        print(
            f"input avg/max length is {np.mean(inplens):.1f}/{max(inplens)}, output avg/max length is {np.mean(outlens):.1f}/{max(outlens)}")
        print(f"output vocabulary size: {len(fldic.D)} at output, {len(inpdic.D)} at input")
        tt.tock()

        tt.tick("tensorizing")
        if tokenizer.berttok is None:
            tokenizer._inpvocab = inpdic
        tokenizer.outvocab = fldic
        trainds = ds.filter(lambda x: x[-1] == "train").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize_and_tensorize(x[0], x[1])).cache(True)
        validds = ds.filter(lambda x: x[-1] == "valid").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize_and_tensorize(x[0], x[1])).cache(True)
        testds = ds.filter(lambda x: x[-1] == "test").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize_and_tensorize(x[0], x[1])).cache(True)
        # ds = ds.map(lambda x: tokenizer.tokenize(x[0], x[1]) + (x[2],)).cache(True)
        tt.tock("tensorized")

        tt.tock(f"loaded '{dataset}'")
        tt.msg(f"#train={len(trainds)}, #valid={len(validds)}, #test={len(testds)}")

        tt.msg("Overlap of validation with train:")
        validoverlaps = compute_overlaps(trainds, validds)
        print(json.dumps(validoverlaps, indent=4))

        tt.msg("Overlap of test with train:")
        testoverlaps = compute_overlaps(trainds, testds)
        print(json.dumps(testoverlaps, indent=4))

        tt.tick("shelving")
        with shelve.open(shelfname) as shelf:
            shelved = {
                "trainex": trainds.examples,
                "validex": validds.examples,
                "testex": testds.examples,
                "tokenizer": tokenizer,
                "validoverlaps": validoverlaps,
                "testoverlaps": testoverlaps,
            }
            shelf[key] = shelved
        tt.tock("shelved")

    return trainds, validds, testds, tokenizer


def compute_overlaps(train, test):
    inp_overlap = []
    out_overlap = []
    both_overlap = []
    traininps, trainouts, trainboths = set(), set(), set()
    for i in tqdm(range(len(train))):
        ex = train[i]
        inpstr = list(ex[0].cpu().numpy())
        inpstr = " ".join([str(exe) for exe in inpstr])
        outstr = list(ex[1].cpu().numpy())
        outstr = " ".join([str(exe) for exe in outstr])
        traininps.add(inpstr)
        trainouts.add(outstr)
        trainboths.add(inpstr+"|"+outstr)

    for i in tqdm(range(len(test))):
        ex = test[i]
        inpstr = list(ex[0].cpu().numpy())
        inpstr = " ".join([str(exe) for exe in inpstr])
        outstr = list(ex[1].cpu().numpy())
        outstr = " ".join([str(exe) for exe in outstr])
        if inpstr in traininps:
            inp_overlap.append(inpstr)
        if outstr in trainouts:
            out_overlap.append(outstr)
        if inpstr + "|" + outstr in trainboths:
            both_overlap.append(inpstr + "|" + outstr)

    ret = {"inps": len(inp_overlap)/len(test),
           "outs": len(out_overlap)/len(test),
           "both": len(both_overlap)/len(test),}
    return ret


def collate_fn(x, pad_value=0, numtokens=5000):
    lens = [len(xe[1]) for xe in x]
    a = list(zip(lens, x))
    a = sorted(a, key=lambda xe: xe[0], reverse=True)
    maxnum = int(numtokens/max(lens))
    b = a[:maxnum]
    b = [be[1] for be in b]
    ret = autocollate(b, pad_value=pad_value)
    return ret


class Seeded(object):
    def __init__(self, seed=42, **kw):
        super(Seeded, self).__init__(**kw)
        self.seed = seed
        self.reset_seed()

    def reset_seed(self):
        self.rng = np.random.RandomState(self.seed)


class RandomAugDataset(IterableDataset, Seeded):
    def __init__(self, words, probs=None, minlen=1, maxlen=100, seed=42, **kw):
        """
        :param dic:     dictionary mapping tokens to integer ids
        :param probs:   dictionary mapping tokens to probabilities. Default: None = uniform
                        Can be partially filled (not contain all tokens).
                        The remaining probability mass is distributed uniformly over remaining tokens.
        :param minlen: minimum length
        :param maxlen: maximum length
        :param kw:
        """
        super(RandomAugDataset, self).__init__(seed=seed, **kw)
        self.words = list(words)
        self.probs = {k: v for k, v in probs.items()}
        if self.probs is None:
            self.probs = {}

        # redistribute remaining probability mass uniformly over words not mentioned in 'probs'
        totprobs = 0
        wordswithprobs = 0
        for word in self.words:
            if word in self.probs:
                totprobs += self.probs[word]
                wordswithprobs += 1
        assert totprobs <= 1.0
        if wordswithprobs < len(self.words):
            prob = (1 - totprobs) / (len(words) - wordswithprobs)
            for k in self.words:
                if k not in self.probs:
                    self.probs[k] = prob
        totprobs = sum(self.probs.values())
        assert np.isclose(totprobs, 1.0)

        self.minlen = minlen
        self.maxlen = maxlen
        assert self.maxlen > self.minlen

        self._sampled = 0       # number of outside calls to sample()
        self._rejected = 0      # number of rejections while sampling

    def __iter__(self):
        self._sampled = 0
        self._rejected = 0
        self.reset_seed()
        return self

    def __next__(self):
        return self.sample()

    def sample(self):
        self._sampled += 1
        # 1. sample a length
        len = int(round(random.random() * (self.maxlen - self.minlen))) + self.minlen

        # 2. sample a sequence of tokens
        toks = list(self.words)
        probs = [self.probs[tok] for tok in toks]
        seq = np.random.choice(toks, len, True, probs)
        seq = " ".join(seq)
        return seq


class PCFGAugDataset(IterableDataset, Seeded):
    def __init__(self, pcfg:PCFG, maxlen=250, seed=42, **kw):
        super(PCFGAugDataset, self).__init__(seed=seed, **kw)
        self.pcfg = pcfg
        self.maxlen = maxlen
        self._sampled = 0       # number of outside calls to sample()
        self._rejected = 0      # number of rejections while sampling

    def __iter__(self):
        self._sampled = 0
        self._rejected = 0
        self.reset_seed()
        return self

    def __next__(self):
        return self.sample()

    def sample(self, _resample=False)->Tree:
        """
        Samples a tree according to the set 'pcfg'.
        The return tree contains at most 'maxlen' tokens and will not contain non-terminals
        (all samples longer than specified length are rejected and resampled).
        :return: nltk.Tree
        """
        if _resample is False:
            self._sampled += 1
        start = self.pcfg.start()
        seq = [start]
        while True:
            allterminal = True
            newseq = []
            for x in seq:
                if isinstance(x, Nonterminal):
                    productions = self.pcfg.productions(lhs=x)
                    prodprobs = [prod.prob() for prod in productions]
                    sampledprod = np.random.choice(list(range(len(productions))), size=None, replace=False, p=prodprobs)
                    sampledprod = productions[sampledprod]
                    for y in sampledprod.rhs():
                        if isinstance(y, Nonterminal):
                            allterminal = False
                        newseq.append(y)
                else:
                    newseq.append(x)
            seq = newseq
            if allterminal or len(seq) > self.maxlen:
                break
        if len(seq) > self.maxlen:
            self._rejected += 1
            seq = self.sample(_resample=True)
        return " ".join(seq).replace("( ", "(")


class NoiseAdder(object):
    def __init__(self, p=0.2, plen=None, repl="@MASK@", seed=42, **kw):
        super(NoiseAdder, self).__init__(**kw)
        self.p = p
        self.plen = plen if plen is not None else (1.,)
        self.repl = repl
        self.seed = seed
        self.reset_seed()

    def reset_seed(self):
        self.rng = np.random.RandomState(self.seed)

    def __call__(self, xs:List[str]):
        # xs = x.split(" ")
        repl = [self.rng.random() < self.p for _ in xs]
        ys = []
        mask = []
        i = 0
        prevrepl = False
        while i < len(xs):
            if repl[i] == True:
                if prevrepl is False:
                    ys.append(self.repl)
                # sample how many tokens to replace
                howmany = self.rng.choice(list(range(1, len(self.plen) + 1)), None, False, self.plen)
                mask += [True] * howmany
                i += howmany
                prevrepl = True
            else:
                ys.append(xs[i])
                mask.append(False)
                i += 1
                prevrepl = False
        # ret = " ".join(ys)
        return ys, xs, mask


def get_datasets(dataset="scan/mcd1", tokenizer="vanilla", batsize=10, validfrac=0.1,
                 recompute=False, splitseed=42, seed=42, recomputeds=False):
    l = locals().copy()
    tt = q.ticktock("dataload")

    assert tokenizer == "vanilla"

    tt.tick(f"loading '{dataset}'")
    del l["seed"]
    del l["recompute"]
    del l["recomputeds"]
    del l["batsize"]
    if dataset.startswith("cfq/") or dataset.startswith("scan/mcd"):
        del l["validfrac"]
        del l["splitseed"]
        print(f"validfrac and splitseed are ineffective with dataset '{dataset}'")

    key = str(l)

    tt.tick("loading dataset")
    trainds, validds, testds, tokenizer = load_ds(dataset=dataset,
                                                  validfrac=validfrac,
                                                  recompute=recomputeds,
                                                  tokenizer=tokenizer)
    tt.tock("dataset loaded")
    tt.msg(f"TRAIN DATA: {len(trainds)}")
    tt.msg(f"DEV DATA: {len(validds)}")
    tt.msg(f"TEST DATA: {len(testds)}")

    # tt.tick("dataloaders")
    # traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    # validdl = DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    # testdl = DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    # tt.tock()
    # tt.tock("got dataloaders")

    return trainds, validds, testds, tokenizer


def check_pcfg(pcfg:PCFG, ds, tokenizer):
    # TODO: check how many training, valid and test examples can be reconstructed based on the pcfg
    return {'frac': 1.}


def tst_tokenizer():
    tok = Tokenizer("bert-base-uncased")
    print(tok)


def tst_scan_loader():
    trainds, validds, testds, fldic, inpdic = load_ds(dataset="scan/mcd1",
                                                      validfrac=0.1,
                                                      recompute=True)
    trainds, validds, testds, fldic, inpdic = load_ds(dataset="scan/mcd1",
                                                      validfrac=0.1,
                                                      recompute=False)

def tst_data_loader(dataset="scan/mcd1", batsize=10):
    traindl, validdl, testdl, auginpdl, augoutdl, tokenizer \
        = get_datasets(dataset, augmode="random", batsize=batsize, auglen=50)

    print(len(auginpdl))
    i = 0
    for batch in auginpdl:
        print(batch)
        for j in range(len(batch[0])):
            print(tokenizer.inpvocab.tostr(batch[0][j]))
        if i == 10:
            break
        i += 1


def try_aug_dataset():
    d = {"A": 1, "B": 2, "C": 3, "D": 4}
    ds = RandomAugDataset(d)

    print(len(ds))
    iter(ds)
    for i in range(10):
        print(next(ds))
    iter(ds)
    for i in range(10):
        print(next(ds))


def correct_cfq_pcfg(pcfg, D):
    _variables = set([f"?x{i}" for i in range(100)])
    _placeholders = set([f"m{i}" for i in range(100)])

    variables = set()
    placeholders = set()
    rels = set()
    entities = set()
    # rels.add("a")
    for k in D:
        kp = None
        if k in _variables:
            variables.add(k)
            continue
        elif k in _placeholders:
            placeholders.add(k)
            continue
        elif k.startswith("ns:"):
            kp = k[3:]
        elif k.startswith("^ns:"):
            kp = k[4:]
        else:
            kp = None

        if kp is not None:
            if kp.startswith("m."):
                entities.add(k)
            elif len(kp.split(".")) == 2:
                entities.add(k)
            else:
                rels.add(k)

    print(f"{len(placeholders)} placeholders")
    print(f"{len(variables)} variables")
    print(f"{len(rels)} rels")
    print(f"{len(entities)} entities")

    print("Remaining:")
    print(set(D.keys()) - variables - placeholders - entities - rels)

    newproductions = []
    orprob = 0
    aprob = 0
    for production in pcfg.productions():
        if production.lhs() == Nonterminal("NT-@COND-ARG0") \
            or production.lhs() == Nonterminal("NT-@COND-ARG1") \
            or production.lhs() == Nonterminal("NT-@COND-ARG2") \
            or production.lhs() == Nonterminal("NT-filter-ARG0") \
            or production.lhs() == Nonterminal("NT-filter-ARG2") \
            or production.lhs() == Nonterminal("NT-@OR-ARG"):
            pass
            if production.lhs() == Nonterminal("NT-@COND-ARG1") \
                    and Nonterminal("NT-@OR") in production.rhs():
                newproductions.append(production)
                orprob = production.prob()
            elif production.lhs() == Nonterminal("NT-@COND-ARG1") \
                    and "a" in production.rhs():
                newproductions.append(production)
                aprob = production.prob()
        else:
            newproductions.append(production)

    # add cond-arg and filter-arg productions manually
    for entity in entities:
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-@COND-ARG2"), [entity], prob=1/(len(entities)+len(variables)+len(placeholders))))
    for entity in rels:
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-@COND-ARG1"), [entity], prob=(1-orprob-aprob)/(len(rels))))
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-@OR-ARG"), [entity], prob=1/(len(rels))))
    for entity in variables:
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-@COND-ARG0"), [entity],
                                                      prob=1 / (len(variables) + len(placeholders))))
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-@COND-ARG2"), [entity],
                                                      prob=1 / (len(entities) + len(variables) + len(placeholders))))
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-filter-ARG0"), [entity],
                                                      prob=1 / (len(variables) + len(placeholders))))
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-filter-ARG2"), [entity],
                                                      prob=1 / (len(variables) + len(placeholders))))
    for entity in placeholders:
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-@COND-ARG0"), [entity],
                                                      prob=1 / (len(variables) + len(placeholders))))
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-@COND-ARG2"), [entity],
                                                      prob=1 / (len(entities) + len(variables) + len(placeholders))))
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-filter-ARG0"), [entity],
                                                      prob=1 / (len(variables) + len(placeholders))))
        newproductions.append(ProbabilisticProduction(Nonterminal("NT-filter-ARG2"), [entity],
                                                      prob=1 / (len(variables) + len(placeholders))))

    newpcfg = PCFG(pcfg.start(), newproductions)
    return newpcfg


def try_pcfg_builder(dataset="cfq/mcd1", validfrac=0.2, recompute=False):
    assert dataset.startswith("cfq")
    trainds, validds, testds, tokenizer = load_ds(dataset=dataset,
                                                      validfrac=validfrac,
                                                      recompute=recompute)

    ystrs = []
    for ex in tqdm(trainds):
        ystr = tokenizer.outvocab.tostr(ex[1])
        ystrs.append(lisp_to_tree(ystr))
        if len(ystrs) == 10000:     # TODO: remove this to use all examples
            break

    pcfg = PCFGBuilder(("@OR", "@WHERE", "@AND")).build(ystrs)
    pcfg = correct_cfq_pcfg(pcfg, tokenizer.outvocab.D)

    ds = PCFGAugDataset(pcfg, seed=42)
    print(next(iter(ds)))


def tst_data_loader_pcfg(dataset="cfq/mcd1", batsize=10):
    trainds, validds, testds, auginpds, augoutds, tokenizer \
        = get_datasets(dataset, augmode="random-pcfg", batsize=batsize)

    auginpdl = DataLoader(auginpds, batch_size=batsize, collate_fn=autocollate)
    augoutdl = DataLoader(augoutds, batch_size=batsize, collate_fn=autocollate)

    print(len(auginpdl))
    i = 0
    for batch in auginpdl:
        print(batch)
        for j in range(len(batch[0])):
            print(tokenizer.inpvocab.tostr(batch[0][j]))
        if i == 10:
            break
        i += 1

    print(len(augoutdl))
    i = 0
    for batch in augoutdl:
        print(batch)
        for j in range(len(batch[0])):
            print(tokenizer.outvocab.tostr(batch[0][j]))
        if i == 10:
            break
        i += 1

    print(f"Sampled: {augoutdl.dataset.rootds._sampled}")
    print(f"Rejected: {augoutdl.dataset.rootds._rejected}")


if __name__ == '__main__':
    # tst_tokenizer()
    # tst_scan_loader()
    # tst_data_loader()
    # try_aug_dataset()
    # try_pcfg_builder()
    tst_data_loader_pcfg()