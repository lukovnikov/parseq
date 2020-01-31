import random

import qelos as q
import torch
import numpy as np
import re

from torch.utils.data import DataLoader

from parseq.util import DatasetSplitProxy
from parseq.vocab import SequenceEncoder


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def gen_data(maxlen=10, N=10000):
    ranges = list(range(48,50)) + list(range(97,123)) + list(range(65, 70))
    chars = [chr(r) for r in ranges]

    ret = []
    y = []
    for i in range(N):
        rete = ""
        ye = None
        for j in range(maxlen):
            c = random.choice(chars)
            rete += c
            if ye is None:
                ye = c
            elif re.match("\d", c):
                if not re.match("\d", ye):
                    ye = c
            elif re.match("[A-Z]", c):
                if re.match("[a-z]", ye):
                    ye = c
        y.append(ye)
        ret.append(rete)
    return ret, y


class ConditionalRecallDataset(object):
    def __init__(self, maxlen=10, N=20000, **kw):
        super(ConditionalRecallDataset, self).__init__(**kw)
        self.data = {}
        self.N, self.maxlen = N, maxlen
        self._seqs, self._ys = gen_data(self.maxlen, self.N)
        self.encoder = SequenceEncoder(tokenizer=lambda x: list(x))

        for seq, y in zip(self._seqs, self._ys):
            self.encoder.inc_build_vocab(seq)
            self.encoder.inc_build_vocab(y)

        splits = ["train"] * int(N*0.8) + ["valid"] * int(N*0.1) + ["test"] * int(N*0.1)

        self.encoder.finalize_vocab()
        self.build_data(self._seqs, self._ys, splits)

    def build_data(self, seqs, ys, splits):
        for seq, y, split in zip(seqs, ys, splits):
            seq_tensor = self.encoder.convert(seq, return_what="tensor")
            y_tensor = self.encoder.convert(y, return_what="tensor")
            if split not in self.data:
                self.data[split] = []
            self.data[split].append((seq_tensor, y_tensor))

    def get_split(self, split:str):
        return DatasetSplitProxy(self.data[split])

    def dataloader(self, split:str=None, batsize:int=5, shuffle=None):
        if split is None:   # return all splits
            ret = {}
            for split in self.data.keys():
                ret[split] = self.dataloader(batsize=batsize, split=split, shuffle=shuffle)
            return ret
        else:
            assert(split in self.data.keys())
            shuffle = shuffle if shuffle is not None else split in ("train", "train+valid")
            dl = DataLoader(self.get_split(split), batch_size=batsize, shuffle=shuffle)
            return dl


def run(lr=0.001):
    ds = ConditionalRecallDataset()
    dl = ds.dataloader("train")
    batch = iter(dl).next()
    print(batch)



if __name__ == '__main__':
    q.argprun(run)