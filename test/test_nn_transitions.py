from unittest import TestCase

from parseq.transitions import LSTMCellTransition
import torch


class TestLSTMCellTransition(TestCase):
    def test_init(self):
        t = LSTMCellTransition(torch.nn.LSTMCell(10, 10), dropout=0.)
        init_state = t.get_init_state(2)
        print(init_state)
        print(init_state.get(0).has())
