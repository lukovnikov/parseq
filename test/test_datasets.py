from unittest import TestCase
import torch

from parseq.datasets import autocollate


class TestAutocollate(TestCase):
    def test_normal(self):
        a = torch.randint(0, 10, (5, ))
        b = torch.randint(0, 10, (7, ))
        c = torch.randint(0, 10, (3, ))

        y = autocollate([(a,), (b,), (c,)])[0]
        print(a, b, c)
        print(y)
        print(y[0, :len(a)])
        self.assertTrue(torch.all(a == y[0, :len(a)]))
        self.assertTrue(torch.all(b == y[1, :len(b)]))
        self.assertTrue(torch.all(c == y[2, :len(c)]))
        self.assertTrue(torch.all(0 == y[0, len(a):]))
        self.assertTrue(torch.all(0 == y[1, len(b):]))
        self.assertTrue(torch.all(0 == y[2, len(c):]))

    def test_2D(self):
        a = torch.randint(0, 10, (5, ))
        b = torch.randint(0, 10, (7, ))
        c = torch.randint(0, 10, (3, ))

        d = torch.randint(0, 10, (4, 3))
        e = torch.randint(0, 10, (1, 8))
        f = torch.randint(0, 10, (2, 2))

        x = [(a, d), (b, e), (c, f)]
        y = autocollate(x)
        print(x)
        print(y)
        y, z = y
        self.assertTrue(torch.all(a == y[0, :len(a)]))
        self.assertTrue(torch.all(b == y[1, :len(b)]))
        self.assertTrue(torch.all(c == y[2, :len(c)]))
        self.assertTrue(torch.all(0 == y[0, len(a):]))
        self.assertTrue(torch.all(0 == y[1, len(b):]))
        self.assertTrue(torch.all(0 == y[2, len(c):]))
        self.assertTrue(torch.all(d == z[0, :d.size(0), :d.size(1)]))
        self.assertTrue(torch.all(e == z[1, :e.size(0), :e.size(1)]))
        self.assertTrue(torch.all(f == z[2, :f.size(0), :f.size(1)]))



