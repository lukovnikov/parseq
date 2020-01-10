from unittest import TestCase

import torch

from parseq.nn import TokenEmb, GRUEncoder, LSTMEncoder


class TestTokenEmb(TestCase):
    def test_it(self):
        emb = torch.nn.Embedding(1000, 300, padding_idx=0)
        emb = TokenEmb(emb)

        D = "the cat dog ate , a nice meal".split()
        D = dict(zip(D, range(1, len(D) + 1)))

        emb.load_pretrained(D)

        print(D)


class TestEncoder(TestCase):
    def test_it(self):
        enc = GRUEncoder(10, 10, 2)
        x = torch.nn.Parameter(torch.randn(3, 6, 10))
        mask = torch.tensor([
            [1,1,1,0,0,0],
            [1,1,1,1,1,0],
            [1,1,1,1,0,0],
        ])
        y, h = enc(x, mask)
        print(y.size())
        print(len(h))
        print(len(h[0]))
        print(h[0][0].size())

    def test_grad_final_states(self):
        enc = GRUEncoder(10, 10, 2)
        x = torch.nn.Parameter(torch.randn(3, 6, 10))
        mask = torch.tensor([
            [1,1,1,0,0,0],
            [1,1,1,1,1,0],
            [1,1,1,1,0,0],
        ])
        y, h = enc(x, mask)
        print(y.size())
        print(len(h))
        print(len(h[0]))
        print(h[0][0].size())
        h[0][-1][1].sum().backward()
        print(x.grad[:, :, :2])

    def test_grad_time_states(self):
        enc = LSTMEncoder(10, 10, 2)
        x = torch.nn.Parameter(torch.randn(3, 6, 10))
        mask = torch.tensor([
            [1,1,1,0,0,0],
            [1,1,1,1,1,0],
            [1,1,1,1,0,0],
        ])
        y, h = enc(x, mask)
        print(y.size())
        print(len(h))
        print(len(h[0]))
        print(h[0][0].size())

        y[2].sum().backward()
        print(x.grad[:, :, :2])



