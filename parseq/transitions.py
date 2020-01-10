from typing import Dict, List

import torch

from parseq.states import State, ListState


class TransitionModel(torch.nn.Module): pass


class LSTMState(State): pass


class MultiLSTMState(ListState): pass


class GRUTransition(TransitionModel):
    def __init__(self, indim, hdim, num_layers=1, dropout:float=0., dropout_rec:float=0., **kw):
        super(GRUTransition, self).__init__(**kw)
        self.indim, self.hdim, self.numlayers, self.dropoutp = indim, hdim, num_layers, dropout
        self.cell = torch.nn.GRU(indim, hdim, num_layers, bias=True,
                                  batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)

    def get_init_state(self, batsize, device=torch.device("cpu")):
        state = State()
        x = torch.ones(batsize, self.numlayers, self.hdim, device=device)
        state.h = torch.zeros_like(x)
        state.h_dropout = self.dropout_rec(torch.ones_like(x)).clamp(0, 1)
        return state

    def forward(self, inp:torch.Tensor, state:State):
        """
        :param inp:     (batsize, indim)
        :param state:   State with .h, .c of shape (numlayers, batsize, hdim)
        :return:
        """
        x = inp
        _x = self.dropout(x)
        h_nm1 = ((state.h * state.h_dropout) if self.dropout_rec.p > 0 else state.h).transpose(0, 1)
        out, h_n = self.cell(_x[:, None, :], h_nm1.contiguous())
        out = out[:, 0, :]
        state.h = h_n.transpose(0, 1)
        return out, state


class LSTMTransition(TransitionModel):
    def __init__(self, indim, hdim, num_layers=1, dropout:float=0., dropout_rec:float=0., **kw):
        super(LSTMTransition, self).__init__(**kw)
        self.indim, self.hdim, self.numlayers, self.dropoutp = indim, hdim, num_layers, dropout
        self.cell = torch.nn.LSTM(indim, hdim, num_layers, bias=True,
                                  batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)

    def get_init_state(self, batsize, device=torch.device("cpu")):
        state = State()
        x = torch.ones(batsize, self.numlayers, self.hdim, device=device)
        state.h = torch.zeros_like(x)
        state.c = torch.zeros_like(x)
        state.h_dropout = self.dropout_rec(torch.ones_like(x)).clamp(0, 1)
        state.c_dropout = self.dropout_rec(torch.ones_like(x)).clamp(0, 1)
        return state

    def forward(self, inp:torch.Tensor, state:State):
        """
        :param inp:     (batsize, indim)
        :param state:   State with .h, .c of shape (numlayers, batsize, hdim)
        :return:
        """
        x = inp
        _x = self.dropout(x)
        h_nm1 = ((state.h * state.h_dropout) if self.dropout_rec.p > 0 else state.h).transpose(0, 1)
        c_nm1 = ((state.c * state.c_dropout) if self.dropout_rec.p > 0 else state.c).transpose(0, 1)
        out, (h_n, c_n) = self.cell(_x[:, None, :], (h_nm1.contiguous(), c_nm1.contiguous()))
        out = out[:, 0, :]
        state.h = h_n.transpose(0, 1)
        state.c = c_n.transpose(0, 1)
        return out, state


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
