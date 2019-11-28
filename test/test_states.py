from unittest import TestCase
import numpy as np
import torch

from parseq.states import State


class Test_State(TestCase):
    def test_state_create(self):
        x = State()
        x.set(k=torch.randn(3, 4))
        xsub = State()
        xsub.set(v=torch.rand(3, 2))
        x.set(s=xsub)
        x.set(l=["sqdf", "qdsf", "qdsf"])
        print(x)
        print(x._schema_keys)
        print(x.k)
        print(x.s.v)
        print(x.l)

    def test_state_merge(self):
        x = State()
        x.set(k=torch.randn(3, 4))
        xsub = State()
        xsub.set(v=torch.rand(3, 2))
        x.set(s=xsub)
        x.set(l=["sqdf", "qdsf", "qdsf"])
        y = State()
        y.set(k=torch.randn(2, 4))
        ysub = State()
        ysub.set(v=torch.rand(2, 2))
        y.set(s=ysub)
        y.set(l=["sqdf", "qdsf"])
        z = State.merge([x, y])
        print(z._merge_keys)
        print(z.k)
        print(x.k)
        print(y.k)
        print(z.s.v)
        print(z.l)

    def test_state_getitem(self):
        x = State()
        x.set(k=torch.randn(5, 4))
        xsub = State()
        xsub.set(v=torch.rand(5, 2))
        x.set(s=xsub)
        x.set(l=["sqdf", "qdsf", "qdsf", "qf", "qsd"])
        y = x[2]
        print(x[2].k)
        print(x[2].l)
        print(x[2].s.v)
        print(x[:2].k)
        print(x[:2].l)
        print(x[:2].s.v)

    def test_state_setitem(self):
        x = State()
        x.set(k=torch.randn(5, 4))
        xsub = State()
        xsub.set(v=torch.rand(5, 2))
        x.set(s=xsub)
        x.set(l=["sqdf", "qdsf", "qdsf", "a", "b"])
        y = State()
        y.set(k=torch.ones(2, 4))
        ysub = State()
        ysub.set(v=torch.ones(2, 2))
        y.set(s=ysub)
        y.set(l=["o", "o"])

        x[1:3] = y
        print(x.k)
        print(x.s.v)
        print(x.l)

    def test_slicing_by_indexes(self):
        x = State(k=torch.rand(6, 3), s="a b c d e f".split())
        print(x.k)
        print(x[[1, 2]].k)
        print(x[[1, 2]].s)

        print(x[np.asarray([1, 2])].k)
        print(x[np.asarray([1, 2])].s)

        print(x[torch.tensor([1, 2])].k)
        print(x[torch.tensor([1, 2])].s)

    def test_seting_by_indexes(self):
        x = State(k=torch.rand(6, 3), s="a b c d e f".split())
        y = State(k=torch.zeros(2, 3), s="o o".split())

        x[[1, 3]] = y
        print(x.k)
        print(x.s)

        x = State(k=torch.rand(6, 3), s="a b c d e f".split())
        y = State(k=torch.zeros(2, 3), s="o o".split())

        x[np.asarray([1, 3])] = y
        print(x.k)
        print(x.s)

        x = State(k=torch.rand(6, 3), s="a b c d e f".split())
        y = State(k=torch.zeros(2, 3), s="o o".split())

        x[torch.tensor([1, 3])] = y
        print(x.k)
        print(x.s)

    def test_copy_non_detached(self):
        x = State(k=torch.nn.Parameter(torch.rand(5, 3)))
        y = x.make_copy(detach=False)
        l = y.k.sum()
        l.backward()
        print(x.k.grad)

    def test_copy_detached(self):
        x = State(k=torch.nn.Parameter(torch.rand(5, 3)))
        y = x.make_copy()
        y.k[:] = 0
        print(x.k)
        print(y.k)

    def test_copy_deep(self):
        x = State(k=["a", "b", "c"])
        y = x.make_copy(deep=False)
        y.k[:] = "q"
        print(x.k)
        print(y.k)