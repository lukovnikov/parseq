import collections
import math
import random
from functools import partial

import qelos as q
import torch
import numpy as np
import re
import dgl
from torch.nn import init

from torch.utils.data import DataLoader

from parseq.nn import GRUEncoder, RNNEncoder
from parseq.vocab import SequenceEncoder


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def gen_data(maxlen=10, NperY=2):
    letters = [chr(x) for x in range(97, 123)]
    numbers = [chr(x) for x in range(48, 58)]
    uppercase = [chr(x) for x in range(65, 90)]
    # print(letters)
    # print(numbers)
    # print(uppercase)

    y = [l for i in range(NperY) for l in letters] \
        + [l for i in range(NperY) for l in numbers] \
        + [l for i in range(NperY) for l in uppercase]

    ret = []
    for i in range(len(y)):
        rete = ""
        if y[i] in letters:
            for j in range(maxlen-1):
                rete += random.choice(letters)
            rete = y[i] + rete
        else:
            position = random.choice(list(range(maxlen-3)))    # position at which y occurs
            if y[i] in numbers:
                allowed_before = letters + uppercase
                allowed_after = letters + uppercase + numbers
            else:   # y[i] is uppercase
                allowed_before = letters
                allowed_after = letters + uppercase
            for j in range(maxlen):
                if j < position:
                    rete += random.choice(allowed_before)
                elif j == position:
                    rete += y[i]
                else:
                    rete += random.choice(allowed_after)

        # check
        _y = None
        for j in rete:
            if _y in numbers:
                pass
            elif _y in uppercase:
                if j in numbers:
                    _y = j
            elif _y in letters:
                if j in numbers + uppercase:
                    _y = j
            else:
                assert(_y is None)
                _y = j
        if _y != y[i]:
            assert(_y == y[i])
        ret.append(rete)
    # for _y, _ret in zip(y, ret):
    #     print(_ret, _y)
    return ret, y


class ConditionalRecallDataset(object):
    def __init__(self, maxlen=10, NperY=10, **kw):
        super(ConditionalRecallDataset, self).__init__(**kw)
        self.data = {}
        self.NperY, self.maxlen = NperY, maxlen
        self._seqs, self._ys = gen_data(self.maxlen, self.NperY)
        self.encoder = SequenceEncoder(tokenizer=lambda x: list(x))

        for seq, y in zip(self._seqs, self._ys):
            self.encoder.inc_build_vocab(seq)
            self.encoder.inc_build_vocab(y)

        self.N = len(self._seqs)
        N = self.N

        splits = ["train"] * int(N * 0.8) + ["valid"] * int(N * 0.1) + ["test"] * int(N * 0.1)
        random.shuffle(splits)

        self.encoder.finalize_vocab()
        self.build_data(self._seqs, self._ys, splits)

    def build_data(self, seqs, ys, splits):
        for seq, y, split in zip(seqs, ys, splits):
            seq_tensor = self.encoder.convert(seq, return_what="tensor")
            y_tensor = self.encoder.convert(y, return_what="tensor")
            if split not in self.data:
                self.data[split] = []
            self.data[split].append((seq_tensor[0], y_tensor[0][0]))

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


class BasicGGNNCell(torch.nn.Module):       # RELATIONS: adding vectors
    def __init__(self, hdim, dropout=0., numrels=16, **kw):
        super(BasicGGNNCell, self).__init__(**kw)
        self.hdim = hdim
        self.node_gru = torch.nn.GRUCell(self.hdim, self.hdim)
        self.rellin = torch.nn.Linear(self.hdim, self.hdim, bias=False)
        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.hdim))
        init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

        self.dropout = torch.nn.Dropout(dropout)

    def message_func(self, edges):
        relvecs = self.relvectors[edges.data["id"]]
        msg = edges.src["h"]
        # msg = self.rellin(msg)
        msg = msg + relvecs
        return {"msg": msg}

    def reduce_func(self, nodes):
        red = nodes.mailbox["msg"].sum(1)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = self.node_gru(nodes.data["red"], nodes.data["h"])
        return {"h": h}


class SeqGGNN(torch.nn.Module):
    useposemb = True
    def __init__(self, vocab, embdim, cell, numsteps=10, maxlen=10, **kw):
        super(SeqGGNN, self).__init__(**kw)
        self.vocab = vocab
        self.cell = cell
        self.hdim = cell.hdim
        self.numsteps = numsteps
        self.embdim = embdim
        self.maxlen = maxlen

        self.emb = torch.nn.Embedding(vocab.number_of_ids(), self.embdim)
        self.posemb = torch.nn.Embedding(maxlen, self.embdim)
        self.outlin = torch.nn.Linear(self.hdim, vocab.number_of_ids())

    def forward(self, x):
        # region create graph
        g = dgl.DGLGraph()
        g.add_nodes(x.size(0) * x.size(1))
        embs = self.emb(x)
        if self.useposemb:
            positions = torch.arange(1, self.maxlen+1, device=x.device)
            positions = positions[None, :embs.size(1)].repeat(x.size(0), 1)
            posembs = self.posemb(positions)
            embs = torch.cat([embs, posembs], 2)
            # embs = embs + posembs
        _embs = embs.view(-1, embs.size(-1))
        g.ndata["h"] = torch.cat([_embs,
                                  torch.zeros(_embs.size(0), self.hdim - _embs.size(-1), device=_embs.device)],
                                 -1)
        g.ndata["red"] = torch.zeros(_embs.size(0), self.hdim, device=_embs.device)

        xlen = x.size(1)
        for i in range(x.size(0)):
            for j in range(xlen-1):
                g.add_edge(i * xlen + j, i * xlen + j + 1, {"id": torch.tensor([1], device=x.device)})
                g.add_edge(i * xlen + j + 1, i * xlen + j, {"id": torch.tensor([2], device=x.device)})

        # endregion

        # run updates
        for step in range(self.numsteps):
            g.update_all(self.cell.message_func, self.cell.reduce_func, self.cell.apply_node_func)

        # region extract predictions
        out = g.ndata["h"]
        out = out.view(embs.size(0), embs.size(1), self.hdim)

        lastout = out[:, -1, :]
        pred = self.outlin(lastout)
        # endregion
        return pred


class ClassificationAccuracy(torch.nn.Module):
    def forward(self, probs, target):
        _, pred = probs.max(-1)
        same = pred == target
        ret = same.float().sum() / same.size(0)
        return ret


def run(lr=0.001,
        dropout=0.2,
        embdim=20,
        hdim=50,
        epochs=100,
        seqlen=10,
        batsize=20,
        cuda=False,
        npery=20,
        gpu=0,
        ):
    if cuda is False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)
    ds = ConditionalRecallDataset(maxlen=seqlen, NperY=npery)

    m = SeqGGNN(ds.encoder.vocab, embdim, BasicGGNNCell(hdim, dropout=dropout), numsteps=seqlen+4, maxlen=seqlen+2)

    # dl = ds.dataloader("train", batsize=batsize, shuffle=True)
    # batch = iter(dl).next()
    # print(batch)
    # y = m(batch[0])
    # print(y.size())

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    acc = ClassificationAccuracy()

    optim = torch.optim.Adam(m.parameters(), lr=lr)
    losses = [q.LossWrapper(loss, "CE"), q.LossWrapper(acc, "acc")]
    vlosses = [q.LossWrapper(loss, "CE"), q.LossWrapper(acc, "acc")]

    trainepoch = partial(q.train_epoch, model=m, dataloader=ds.dataloader("train", batsize=batsize, shuffle=True), optim=optim, losses=losses, device=device)
    validepoch = partial(q.test_epoch, model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False), losses=vlosses, device=device)

    q.run_training(trainepoch, validepoch, max_epochs=epochs)


class RNNModel(torch.nn.Module):
    def __init__(self, vocab, embdim, hdim, dropout=0., **kw):
        super(RNNModel, self).__init__(**kw)
        self.vocab = vocab
        self.embdim, self.hdim = embdim, hdim

        self.emb = torch.nn.Embedding(vocab.number_of_ids(), self.embdim)
        self.outlin = torch.nn.Linear(self.hdim, vocab.number_of_ids())

        self.gru = GRUEncoder(self.embdim, self.hdim, dropout=dropout, bidirectional=False)

    def forward(self, x):
        embs = self.emb(x)
        encs, finalenc = self.gru(embs)
        out = self.outlin(finalenc[-1][0])
        return out


def run_gru(lr=0.001,
        embdim=20,
        hdim=50,
        epochs=100,
        numsteps=11,
        batsize=20,
        cuda=False,
        dropout=.2,
        gpu=0,
        npery=20,
        ):
    if cuda is False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)
    ds = ConditionalRecallDataset(maxlen=numsteps-1, NperY=npery)
    print(f"{ds.N} examples in total.")

    m = RNNModel(ds.encoder.vocab, embdim, hdim, dropout=dropout)

    # dl = ds.dataloader("train", batsize=batsize, shuffle=True)
    # batch = iter(dl).next()
    # print(batch)
    # y = m(batch[0])
    # print(y.size())

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    acc = ClassificationAccuracy()

    optim = torch.optim.Adam(m.parameters(), lr=lr)
    losses = [q.LossWrapper(loss, "CE"), q.LossWrapper(acc, "acc")]
    vlosses = [q.LossWrapper(loss, "CE"), q.LossWrapper(acc, "acc")]

    trainepoch = partial(q.train_epoch, model=m, dataloader=ds.dataloader("train", batsize=batsize, shuffle=True), optim=optim, losses=losses, device=device)
    validepoch = partial(q.test_epoch, model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False), losses=vlosses, device=device)

    q.run_training(trainepoch, validepoch, max_epochs=epochs)



if __name__ == '__main__':
    q.argprun(run)
    # q.argprun(run_gru)