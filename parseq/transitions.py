from typing import Dict, List

import torch

from parseq.states import State, ListState


class TransitionModel(torch.nn.Module): pass


class LSTMState(State): pass


class MultiLSTMState(ListState): pass


class LSTMCellTransition(TransitionModel):
    def __init__(self, *cells:torch.nn.LSTMCell, dropout:float=0., **kw):
        super(LSTMCellTransition, self).__init__(**kw)
        self.cells = torch.nn.ModuleList(cells)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_rec = torch.nn.Dropout(0.0)

    def get_init_state(self, batsize, device=torch.device("cpu")):
        states = []
        for i in range(len(self.cells)):
            state = LSTMState()
            state.h_dropout = self.dropout_rec(torch.ones(batsize, self.cells[i].hidden_size, device=device))
            state.c_dropout = self.dropout_rec(torch.ones_like(state.h_dropout))
            state.h = torch.zeros_like(state.h_dropout)
            state.c = torch.zeros_like(state.h_dropout)
            states.append(state)
        ret = MultiLSTMState(*states)
        return ret

    def forward(self, inp:torch.Tensor, states:MultiLSTMState):
        x = inp
        for i in range(len(self.cells)):
            _x = self.dropout(x)
            state = states.get(i)
            x, c = self.cells[i](_x, (state.h * state.h_dropout, state.c * state.c_dropout))
            state.h = x
            state.c = c
        return x, states
