import json
import os
import random
from abc import abstractmethod, ABC
from copy import copy
from typing import List, Tuple, Callable, Union

import nltk
import torch
from nltk import Tree, Nonterminal
import numpy as np
from scipy.special import softmax
from torch.utils.data.dataloader import default_collate, DataLoader
from tqdm import tqdm

from parseq.grammar import prolog_to_tree
from transformers import BertTokenizer, AutoTokenizer


class _Vocab(object):
    pass


class Vocab(_Vocab):
    padtoken = "@PAD@"
    unktoken = "@UNK@"
    starttoken = "@START@"
    endtoken = "@END@"
    masktoken = "@MASK@"
    def __init__(self, padid:int=0, unkid:int=1, startid:int=2, endid:int=3, maskid:int=4, **kw):
        self.D = {self.padtoken: padid, self.unktoken: unkid}
        self.D[self.starttoken] = startid
        self.D[self.endtoken] = endid
        self.D[self.masktoken] = maskid
        self.counts = {k: np.infty for k in self.D.keys()}
        self.rare_tokens = set()
        self.rare_ids = set()
        self.RD = {v: k for k, v in self.D.items()}
        self.growing = True

    def set_dict(self, D):
        self.D = D
        self.RD = {v: k for k, v in self.D.items()}

    def nextid(self):
        return max(self.D.values()) + 1

    def finalize(self, min_freq:int=0, top_k:int=np.infty, keep_tokens=None):
        self.growing = False
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)

        if min_freq == 0 and top_k > len(sorted_counts):
            self.rare_tokens = set()
        else:
            if top_k < len(sorted_counts) and sorted_counts[top_k][1] > min_freq:
                i = top_k
            else:
                if top_k < len(sorted_counts):
                    sorted_counts = sorted_counts[:top_k]
                # binary search for min_freq position
                i = 0
                divider = 2
                where = +1
                while True:
                    i += where * len(sorted_counts) // divider
                    if (i == len(sorted_counts)) or (
                            sorted_counts[i][1] <= min_freq - 1 and sorted_counts[i - 1][1] >= min_freq):
                        break  # found
                    elif sorted_counts[i][1] < min_freq:  # go up
                        where = -1
                    elif sorted_counts[i][1] >= min_freq:  # go down
                        where = +1
                    divider *= 2
                    divider = min(divider, len(sorted_counts))
            if keep_tokens is not None:
                if keep_tokens in ("all", "ALL"):
                    self.rare_tokens = set([t[0] for t in sorted_counts[i:]])
                else:
                    sorted_counts = [sc for j, sc in enumerate(sorted_counts) if sc[0] in keep_tokens or j < i]
                    self.rare_tokens = set([t[0] for t in sorted_counts[i:]]) & keep_tokens
            else:
                sorted_counts = sorted_counts[:i]

        nextid = max(self.D.values()) + 1
        for token, cnt in sorted_counts:
            if token not in self.D:
                self.D[token] = nextid
                nextid += 1

        self.RD = {v: k for k, v in self.D.items()}
        if keep_tokens is not None:
            self.rare_ids = set([self[rare_token] for rare_token in self.rare_tokens])

    def add_token(self, token, seen:Union[int,bool]=True):
        assert(self.growing)
        if token not in self.counts:
            self.counts[token] = 0
        if seen > 0:
            self.counts[token] += float(seen)

    def __getitem__(self, item:str) -> int:
        if item not in self.D:
            assert(self.unktoken in self.D)
            item = self.unktoken
        id = self.D[item]
        return id

    def __call__(self, item:int) -> str:
        return self.RD[item]

    def number_of_ids(self, exclude_rare=False):
        if not exclude_rare:
            return max(self.D.values()) + 1
        else:
            return max(set(self.D.values()) - self.rare_ids) + 1

    def reverse(self):
        return {v: k for k, v in self.D.items()}

    def __iter__(self):
        return iter([(k, v) for k, v in self.D.items()])

    def __contains__(self, item:Union[str,int]):
        if isinstance(item, str):
            return item in self.D
        if isinstance(item, int):
            return item in self.RD
        else:
            raise Exception("illegal argument")

    def tostr(self, x:Union[np.ndarray, torch.Tensor], return_tokens=False):
        """
        :param x:   2D LongTensor or array
        :param return_tokens:
        :return:
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = list(np.vectorize(lambda e: self(e))(x))
        x = [e for e in x if e != self.padtoken]
        ret = []
        for xe in x:
            if len(ret) > 0 and ret[-1] == self.endtoken:
                break
            ret.append(xe)
        if return_tokens:
            return ret
        else:
            return " ".join(ret)


class VocabBuilder(ABC):
    @abstractmethod
    def inc_build_vocab(self, x:str, seen:bool=True):
        raise NotImplemented()

    @abstractmethod
    def finalize_vocab(self, min_freq:int=0, top_k:int=np.infty):
        raise NotImplemented()

    @abstractmethod
    def vocabs_finalized(self):
        raise NotImplemented()


class SequenceEncoder(VocabBuilder):
    def __init__(self, tokenizer: Callable[[str], List[str]], vocab: Vocab = None, add_start_token=False,
                 add_end_token=False, **kw):
        super(SequenceEncoder, self).__init__(**kw)
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab is not None else Vocab()
        self.vocab_final = False
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token

    def inc_build_vocab(self, x: str, seen: bool = True):
        if not self.vocab_final:
            tokens = self.tokenizer(x) + []
            if self.add_end_token:
                tokens.append(self.vocab.endtoken)
            for token in tokens:
                self.vocab.add_token(token, seen=seen)
            return tokens
        else:
            return []

    def finalize_vocab(self, min_freq: int = 0, top_k: int = np.infty, keep_tokens=None):
        self.vocab_final = True
        self.vocab.finalize(min_freq=min_freq, top_k=top_k, keep_tokens=keep_tokens)

    def vocabs_finalized(self):
        return self.vocab_final

    def convert(self, x: Union[str, List[str]], return_what: Union[
        str, List[str]] = "tensor"):  # "tensor", "ids", "tokens" or comma-separated combo of all
        return_what = [r.strip() for r in return_what.split(",")] if isinstance(return_what,
                                                                                str) and "," in return_what else return_what
        if isinstance(x, list) and not isinstance(x, Tree) \
                and (x == [] or isinstance(x[0], str)):
            tokens = x
        else:
            tokens = self.tokenizer(x)
        if self.add_start_token and tokens[0] != self.vocab.starttoken:
            tokens.insert(0, self.vocab.starttoken)
        if self.add_end_token and tokens[-1] != self.vocab.endtoken:
            tokens.append(self.vocab.endtoken)
        ret = {"tokens": tokens}

        # returns
        return_single = False
        if isinstance(return_what, str):
            return_single = True
            return_what = [return_what]
        if "ids" in return_what or "tensor" in return_what:
            ret["ids"] = [self.vocab[token] for token in tokens]
        if "tensor" in return_what:
            ret["tensor"] = torch.tensor(ret["ids"], dtype=torch.long)
        ret = [ret[r] for r in return_what]
        if return_single:
            assert (len(ret) == 1)
            ret = ret[0]
        return ret


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
        for ex in self._examples:
            if self._example_fits_filter(ex, f):
                ret.append(ex)
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
    def __init__(self, baseds: Dataset, f, use_cache=False, **kw):
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
        newbase = self.baseds.filter(lambda ex: self._example_fits_filter(self.f(ex), f))
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


def tree_length(x: Union[Tree, str], count_brackets=False):
    if isinstance(x, str):
        return 1
    elif isinstance(x, Tree):
        if len(x) == 0:
            return 1
        else:
            return 1 + sum([tree_length(xe, count_brackets=count_brackets) for xe in x]) + (2 if count_brackets else 0)
    else:
        raise Exception()


def pad_tensors(x:List[torch.Tensor], dim:Union[int,Tuple[int],List[int]]=-2, value=0):
    if isinstance(dim, (tuple, list)):
        if len(dim) > 1:
            x = pad_tensors(x, dim[1:])
        dim = dim[0]
    maxsize = max([xe.size(dim) for xe in x])
    _x = []
    for xe in x:
        xsize = list(xe.size()).copy()
        xsize[dim] = None
        xsize[dim] = maxsize - xe.size(dim)
        _xe = xe
        if xsize[dim] > 0:
            _xe = torch.cat([xe, torch.ones(xsize, device=xe.device, dtype=xe.dtype) * value], dim)
        _x.append(_xe)
    return _x


def autocollate(x, pad_value=0):
    y = list(zip(*x))
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.LongTensor) and yi[0].dim() == 1:
            y[i] = pad_tensors(yi, 0, pad_value)
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.Tensor):
            yi = [yij[None] for yij in yi]
            y[i] = torch.cat(yi, 0)
    return y


def pad_and_default_collate(x, pad_value=0):
    y = list(zip(*x))
    for i, yi in enumerate(y):
        if isinstance(yi[0], torch.LongTensor) and yi[0].dim() == 1:
            y[i] = pad_tensors(yi, 0, pad_value)
    x = list(zip(*y))
    ret = default_collate(x)
    return ret


# region SPECIFIC DATASETS

class MultilingualGeoqueryDatasetLoader(object):
    def __init__(self,
                 p="../../datasets/geo880_multiling/geoquery",
                 validfrac=0.2,
                 **kw):
        super(MultilingualGeoqueryDatasetLoader, self).__init__(**kw)
        self.p = p
        self.validfrac = validfrac

    def load(self, lang: str = "en"):
        with open(os.path.join(self.p, f"geo-{lang}.json")) as f:
            data = json.load(f)
        print(f"{len(data)} examples loaded for language {lang}")
        numtrain = len([x for x in data if x["split"] == "train"])
        istrain = [True] * (int(round(numtrain * (1 - self.validfrac)))) \
                  + [False] * (int(round(numtrain * self.validfrac)))
        random.shuffle(istrain)
        i = 0
        for x in data:
            if x["split"] == "train":
                x["split"] = "valid" if istrain[i] is False else "train"
                i += 1
        return Dataset(data)


def tree_to_lisp_tokens(x:Tree, brackets="()"):
    if isinstance(x, str):
        return [x]
    elif len(x) > 0:
        children = [tree_to_lisp_tokens(xe, brackets=brackets) for xe in x]
        return [brackets[0], x.label()] + [childe for child in children for childe in child] + [brackets[1]]
    else:
        return [x.label()]


def remove_literals(x:Tree, literalparents=("placeid", "countryid", "riverid", "cityid", "stateid")):
    if x.label() in literalparents:
        del x[:]
        return x
    else:
        x[:] = [remove_literals(xe, literalparents) for xe in x]
        return x


def load_geoquery(lang:str="en", nltok_name:str="bert-base-uncased",
                  top_k:int=np.infty, min_freq:int=0,
                  p:str="../../datasets/geo880_multiling/geoquery"):
    ds = MultilingualGeoqueryDatasetLoader(p=p).load(lang)
    bert_tok = AutoTokenizer.from_pretrained(nltok_name)

    ds = ds.map(lambda x: (bert_tok.encode(x["nl"]),
                           tree_to_lisp_tokens(
                               remove_literals(
                                   prolog_to_tree(x["mrl"]))),
                           x["split"]))

    flenc = SequenceEncoder(lambda x: x, add_start_token=True, add_end_token=True)
    for _, fltoks, split in ds.examples:
        flenc.inc_build_vocab(fltoks, seen=split=="train")

    flenc.finalize_vocab(min_freq=min_freq, top_k=top_k)

    ds = ds.map(lambda x: (x[0], flenc.convert(x[1]), x[2]))

    trainds = ds.filter(lambda x: x[2] == "train").map(lambda x: (x[0], x[1])).cache()
    validds = ds.filter(lambda x: x[2] == "valid").map(lambda x: (x[0], x[1])).cache()
    testds = ds.filter(lambda x: x[2] == "test").map(lambda x: (x[0], x[1])).cache()

    return trainds, validds, testds, bert_tok, flenc


def try_multilingual_geoquery_dataset_loader():
    tds, vds, xds, _, flenc = load_geoquery("fa")
    print(tds[0])
    print(flenc.vocab.D)
    print("done")

    dl = DataLoader(tds, batch_size=5, shuffle=True, collate_fn=autocollate)

    batch = next(iter(dl))

    print(batch)
    print(" ".join([flenc.vocab(xe) for xe in batch[1][0].numpy()]))


if __name__ == '__main__':
    print(try_multilingual_geoquery_dataset_loader())