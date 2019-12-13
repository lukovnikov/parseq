from unittest import TestCase

import torch

from parseq.nn import TokenEmb


class TestTokenEmb(TestCase):
    def test_it(self):
        emb = torch.nn.Embedding(1000, 300, padding_idx=0)
        emb = TokenEmb(emb)

        D = "the cat dog ate , a nice meal".split()
        D = dict(zip(D, range(1, len(D) + 1)))

        emb.load_pretrained(D)

        print(D)
