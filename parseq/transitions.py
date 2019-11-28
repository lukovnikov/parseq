from typing import Dict, List

import torch

from parseq.states import State


class TransitionModel(torch.nn.Module): pass


class LSTMState(State): pass


class MultiLSTMState(State):
    def __init__(self, *layersstates:List[LSTMState], **kw):
        super(MultiLSTMState, self).__init__(*layersstates, **kw)


class LSTMCellTransition(TransitionModel):
    def __init__(self, *cells:torch.nn.LSTMCell, dropout:float=0., **kw):
        super(LSTMCellTransition, self).__init__(**kw)
        self.cells = torch.nn.ModuleList(cells)
        self.dropout = torch.nn.Dropout(dropout)

    def get_init_state(self, batsize, device):
        states = []
        for i in range(len(self.cells)):
            state = LSTMState()
            state["h.dropout"] = self.dropout(
                torch.ones(batsize,
                self.cells[i].hidden_size,
                device=device))
            state["c.dropout"] = self.dropout(
                torch.ones_like(state["h.dropout"])
            )
            state["h"] = torch.zeros_like(state["h.dropout"])
            state["c"] = torch.zeros_like(state["h.dropout"])
            states.append(state)
        ret = MultiLSTMState(*states)
        return ret

    def forward(self, inp:torch.Tensor, states:MultiLSTMState):
        x = inp
        for i in range(len(self.cells)):
            _x = self.dropout(x)
            state = states[i]
            x, c = self.cells[i](_x, (state["h"] * state["h.dropout"],
                                      state["c"] * state["c.dropout"]))
            state["h"] = x
            state["c"] = c
        return x
