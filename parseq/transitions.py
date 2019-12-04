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

    def get_init_state(self, batsize, device=torch.device("cpu")):
        states = []
        for i in range(len(self.cells)):
            state = LSTMState()
            state.set(h_dropout=self.dropout(torch.ones(batsize, self.cells[i].hidden_size, device=device)))
            state.set(c_dropout=self.dropout(torch.ones_like(state.get("h_dropout"))))
            state.set("h", torch.zeros_like(state.get("h_dropout")))
            state.set("c", torch.zeros_like(state.get("h_dropout")))
            states.append(state)
        ret = MultiLSTMState(*states)
        return ret

    def forward(self, inp:torch.Tensor, states:MultiLSTMState):
        x = inp
        for i in range(len(self.cells)):
            _x = self.dropout(x)
            state = states.get(i)
            x, c = self.cells[i](_x, (state.get("h") * state.get("h_dropout"),
                                      state.get("c") * state.get("c_dropout")))
            state.set(h=x)
            state.set(c=c)
        return x, states
