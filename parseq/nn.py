import ujson
import os
import re
from typing import Set

import torch
import numpy as np

from parseq.states import State
from parseq.vocab import Vocab
import qelos as q


class TokenEmb(torch.nn.Module):
    def __init__(self, emb:torch.nn.Embedding, adapt_dims=None, rare_token_ids:Set[int]=None, rare_id:int=None, **kw):
        super(TokenEmb, self).__init__(**kw)
        self.emb = emb
        self.rare_token_ids = rare_token_ids
        self.rare_id = rare_id
        self._do_rare()

        self.adapter = None
        if adapt_dims is not None and adapt_dims[0] != adapt_dims[1]:
            self.adapter = torch.nn.Linear(*adapt_dims)

        # self.init_params()

    def init_params(self):
        torch.nn.init.uniform_(self.emb.weight, -0.1, 0.1)
        torch.nn.init.constant_(self.emb.weight[0], 0)

    def _do_rare(self, rare_token_ids:Set[int]=None, rare_id:int=None):
        self.register_buffer("unkmap", None)
        self.rare_token_ids = self.rare_token_ids if rare_token_ids is None else rare_token_ids
        self.rare_id = self.rare_id if rare_id is None else rare_id
        if self.rare_id is not None and self.rare_token_ids is not None:
            # build id mapper
            unkmap = torch.arange(0, self.emb.num_embeddings)
            save = False
            for id in self.rare_token_ids:
                if id < unkmap.size(0):
                    unkmap[id] = self.rare_id
                    save = True
            if save:
                self.register_buffer("unkmap", unkmap)

    def forward(self, x:torch.Tensor):
        numembs = self.emb.num_embeddings
        unkmask = x >= numembs
        if torch.any(unkmask):
            x = x.masked_fill(unkmask, self.rare_id)
        if self.unkmap is not None:
            x = self.unkmap[x]
        ret = self.emb(x)
        if self.adapter is not None:
            ret = self.adapter(ret)
        return ret


def load_pretrained_embeddings(emb, D, p="../data/glove/glove300uncased"):
    W = np.load(p + ".npy")
    with open(p + ".words") as f:
        words = ujson.load(f)
        preD = dict(zip(words, range(len(words))))
    # map D's indexes onto preD's indexes
    select = np.zeros(emb.num_embeddings, dtype="int64") - 1
    covered_words = set()
    covered_word_ids = set()
    for k, v in D.items():
        if k in preD:
            select[v] = preD[k]
            covered_words.add(k)
            covered_word_ids.add(v)
    selectmask = select != -1
    select = select * selectmask.astype("int64")
    subW = W[select, :]
    subW = torch.tensor(subW).to(emb.weight.device)
    selectmask = torch.tensor(selectmask).to(emb.weight.device).to(torch.float)
    emb.weight.data = emb.weight.data * (1-selectmask[:, None]) + subW * selectmask[:, None]        # masked set or something else?
    print("done")
    return covered_words, covered_word_ids


class DGRUCell(torch.nn.Module):
    def __init__(self, dim, bias=True, **kw):
        super(DGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 5, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)

    def forward(self, x, h):
        gates = self.gateW(torch.cat([x, h], 1))
        gates = gates.chunk(5, 1)
        rx = torch.sigmoid(gates[0])
        rh = torch.sigmoid(gates[1])
        z = torch.softmax(torch.stack(gates[2:5], 2), -1)
        u = self.gateU(torch.cat([x * rx, h * rh], 1))
        u = torch.tanh(u)
        h_new = torch.stack([x, h, u], 2) * z
        h_new = h_new.sum(-1)
        return h_new


def try_dgru_cell():
    xs = torch.nn.Parameter(torch.rand(2, 10, 5))
    _hs = torch.nn.Parameter(torch.rand(2, 10, 5))
    hs = [h[:, 0, :] for h in _hs.split(1, 1)]

    m = [DGRUCell(5) for _ in range(10)]
    # m = [torch.nn.GRUCell(5, 5) for _ in range(10)]
    for i in range(xs.size(1)):
        x = xs[:, i]
        for j in range(len(hs)):
            h = hs[j]
            y = m[j](x, h)
            x = y
            hs[j] = y
    print(y)
    y.sum().backward()
    print(xs.grad[:, 0].norm())
    print(_hs.grad[:, 0].norm())


class _Encoder(torch.nn.Module):
    def __init__(self, embdim, hdim, num_layers=1, dropout=0., bidirectional=True, **kw):
        super(_Encoder, self).__init__(**kw)
        self.embdim, self.hdim, self.numlayers, self.bidir, self.dropoutp = embdim, hdim, num_layers, bidirectional, dropout
        self.dropout = torch.nn.Dropout(dropout)
        self.create_rnn()

    def forward(self, x, mask=None):
        x = self.dropout(x)

        if mask is not None:
            _x = torch.nn.utils.rnn.pack_padded_sequence(x, mask.sum(-1), batch_first=True, enforce_sorted=False)
        else:
            _x = x

        _outputs, hidden = self.rnn(_x)

        if mask is not None:
            y, _ = torch.nn.utils.rnn.pad_packed_sequence(_outputs, batch_first=True)
        else:
            y = _outputs

        hidden = (hidden,) if not q.issequence(hidden) else hidden
        hiddens = []
        for _hidden in hidden:
            i = 0
            _hiddens = tuple()
            while i < _hidden.size(0):
                if self.bidir is True:
                    _h = torch.cat([_hidden[i], _hidden[i+1]], -1)
                    i += 2
                else:
                    _h = _hidden[i]
                    i += 1
                _hiddens = _hiddens + (_h,)
            hiddens.append(_hiddens)
        hiddens = tuple(zip(*hiddens))
        return y, hiddens


class RNNEncoder(_Encoder):
    def create_rnn(self):
        self.rnn = torch.nn.RNN(self.embdim, self.hdim, self.numlayers,
                                bias=True, batch_first=True, dropout=self.dropoutp, bidirectional=self.bidir)

    def init_params(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)


class GRUEncoder(_Encoder):
    def create_rnn(self):
        self.rnn = torch.nn.GRU(self.embdim, self.hdim, self.numlayers,
                                bias=True, batch_first=True, dropout=self.dropoutp, bidirectional=self.bidir)
        self.init_params()

    def init_params(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)


class LSTMEncoder(_Encoder):
    def create_rnn(self):
        self.rnn = torch.nn.LSTM(self.embdim, self.hdim, self.numlayers,
                                bias=True, batch_first=True, dropout=self.dropoutp, bidirectional=self.bidir)
        self.init_params()

    def init_params(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)


class BasicGenOutput(torch.nn.Module):
    def __init__(self, h_dim:int, vocab:Vocab=None, dropout:float=0., **kw):
        super(BasicGenOutput, self).__init__(**kw)
        self.gen_lin = torch.nn.Linear(h_dim, vocab.number_of_ids(), bias=True)
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)
        self.dropout = torch.nn.Dropout(dropout)

        self.vocab = vocab

        # rare output tokens
        self.rare_token_ids = vocab.rare_ids
        if len(self.rare_token_ids) > 0:
            out_mask = torch.ones(self.vocab.number_of_ids())
            for rare_token_id in self.rare_token_ids:
                out_mask[rare_token_id] = 0
            self.register_buffer("out_mask", out_mask)
        else:
            self.register_buffer("out_mask", None)

    def forward(self, x:torch.Tensor, out_mask=None, **kw):
        """
        :param x:           (batsize, hdim)
        :param out_mask:    (batsize, vocsize)
        :param kw:
        :return:
        """
        x = self.dropout(x)
        # - generation probs
        gen_probs = self.gen_lin(x)
        if self.out_mask is not None:
            gen_probs = gen_probs + torch.log(self.out_mask)[None, :]
        if out_mask is not None:
            gen_probs = gen_probs + torch.log(out_mask)
        gen_probs = self.logsm(gen_probs)
        return gen_probs


class SumPtrGenOutputOLD(torch.nn.Module):
    def __init__(self, h_dim: int, inp_vocab:Vocab=None, out_vocab:Vocab=None, **kw):
        super(SumPtrGenOutputOLD, self).__init__(**kw)
        # initialize modules
        self.gen_lin = torch.nn.Linear(h_dim, out_vocab.number_of_ids(), bias=True)
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)

        self.inp_vocab, self.out_vocab = inp_vocab, out_vocab

        self.register_buffer("_inp_to_act", torch.zeros(self.inp_vocab.number_of_ids(), dtype=torch.long))
        self.register_buffer("_act_from_inp", torch.zeros(out_vocab.number_of_ids(), dtype=torch.long))

        # for COPY, initialize mapping from input node vocab (sgb.vocab) to output action vocab (qgb.vocab_actions)
        self.build_copy_maps()

        # compute action mask from input: actions that are doable using input copy actions are 1, others are 0
        actmask = torch.zeros(out_vocab.number_of_ids(), dtype=torch.uint8)
        actmask.index_fill_(0, self._inp_to_act, 1)
        self.register_buffer("_inp_actmask", actmask)

        # rare actions
        self.rare_token_ids = out_vocab.rare_ids
        rare_id = 1
        if len(self.rare_token_ids) > 0:
            out_map = torch.arange(self.out_vocab.number_of_ids())
            for rare_token_id in self.rare_token_ids:
                out_map[rare_token_id] = rare_id
            self.register_buffer("out_map", out_map)
        else:
            self.register_buffer("out_map", None)

    def build_copy_maps(self):      # TODO test
        str_action_re = re.compile(r"^<W>\s->\s'(.+)'$")
        string_action_vocab = {}
        for k, v in self.out_vocab:
            if str_action_re.match(k):
                string_action_vocab[str_action_re.match(k).group(1)] = v
        for k, inp_v in self.inp_vocab:
            if k[0] == "@" and k[-1] == "@" and len(k) > 2:
                pass
            elif k[0] == "[" and k[-1] == "]" and len(k) > 2:
                pass
            else:
                # assert (k in self.qgb.vocab_actions)
                if k not in string_action_vocab:
                    print(k)
                assert (k in string_action_vocab)
                out_v = string_action_vocab[k]
                self._inp_to_act[inp_v] = out_v
                self._act_from_inp[out_v] = inp_v

    def forward(self, x:torch.Tensor, statebatch:State, attn_scores:torch.Tensor):  # (batsize, hdim), (batsize, numactions)

        # region build action masks
        actionmasks = []
        action_vocab = self.out_vocab
        for state in statebatch.states:
            # state.get_valid_actions_at(open_node)
            actionmask = torch.zeros(action_vocab.number_of_ids(), device=x.device, dtype=torch.uint8)
            if not state.is_terminated:
                open_node = state.open_nodes[0]
                actionmask = state.get_valid_action_mask_at(open_node).to(actionmask.device)
                # if state.use_gold and not state.is_terminated:
                #     if state.get_gold_action_at(open_node) not in state.get_valid_actions_at(open_node):
                #         print(open_node, state.get_gold_action_at(open_node), state.get_valid_actions_at(open_node))
                #     assert (state.get_gold_action_at(open_node) in state.get_valid_actions_at(open_node))
                # for valid_action in state.get_valid_actions_at(open_node):
                #     actionmask[action_vocab[valid_action]] = 1
            else:
                actionmask.fill_(1)
            actionmasks.append(actionmask)
        actionmask = torch.stack(actionmasks, 0)
        # endregion

        # - generation probs
        gen_scores = self.gen_lin(x)
        if self.out_map is not None:
            gen_scores = gen_scores.index_select(1, self.out_map)

        # - copy probs
        # get distributions over input vocabulary
        ctx_ids = statebatch["inp_tensor"]
        inpdist = torch.zeros(gen_scores.size(0), self.inp_vocab.number_of_ids(), dtype=torch.float,
                              device=gen_scores.device)
        inpdist.scatter_add_(1, ctx_ids, attn_scores)    # TODO use scatter_max

        # map to distribution over output actions
        ptr_scores = torch.zeros(gen_scores.size(0), self.out_vocab.number_of_ids(),
                                 dtype=torch.float, device=gen_scores.device)  # - np.infty
        ptr_scores.scatter_(1, self._inp_to_act.unsqueeze(0).repeat(gen_scores.size(0), 1),
                            inpdist)

        # - mix
        out_scores = gen_scores + ptr_scores
        if actionmask is not None:
            gen_scores = gen_scores + -1e6 * actionmask.float() #torch.log(actionmask.float())
        out_probs = self.sm(out_scores)
        return out_probs, None, gen_scores, attn_scores


class _PtrGenOutput(torch.nn.Module):
    def __init__(self, h_dim: int, vocab:Vocab=None, **kw):
        super(_PtrGenOutput, self).__init__(**kw)
        # initialize modules
        self.gen_lin = torch.nn.Linear(h_dim, vocab.number_of_ids(), bias=True)
        self.copy_or_gen = torch.nn.Linear(h_dim, 2, bias=True)
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)

        self.inp_vocab, self.out_vocab = None, vocab

        self.naningrad = torch.nn.Parameter(torch.zeros(1))
        self.naningrad2 = torch.nn.Parameter(torch.zeros(1))

    # str_action_re=re.compile(r"^<W>\s->\s'(.+)'$")        # <-- for grammar actions
    def build_copy_maps(self, inp_vocab:Vocab, str_action_re=re.compile(r"^([^_].*)$")):
        self.inp_vocab = inp_vocab
        self.register_buffer("_inp_to_act",
             torch.zeros(inp_vocab.number_of_ids(), dtype=torch.long))
        self.register_buffer("_act_to_inp",
             torch.zeros(self.out_vocab.number_of_ids(), dtype=torch.long))

        # for COPY, initialize mapping from input node vocab (sgb.vocab) to output action vocab (qgb.vocab_actions)
        self._build_copy_maps(str_action_re=str_action_re)

        # compute action mask from input: actions that are doable using input copy actions are 1, others are 0
        actmask = torch.zeros(self.out_vocab.number_of_ids(), dtype=torch.uint8)
        actmask.index_fill_(0, self._inp_to_act, 1)
        actmask[0] = 0
        self.register_buffer("_inp_actmask", actmask)

        # rare actions
        self.rare_token_ids = self.out_vocab.rare_ids
        self.register_buffer("gen_mask", None)
        if len(self.rare_token_ids) > 0:
            gen_mask = torch.ones(self.out_vocab.number_of_ids())
            for rare_token_id in self.rare_token_ids:
                gen_mask[rare_token_id] = 0
            self.register_buffer("gen_mask", gen_mask)

    def _build_copy_maps(self, str_action_re):      # TODO test
        if str_action_re is None:
            string_action_vocab = {k: v for k, v in self.out_vocab}
            for k, inp_v in self.inp_vocab:
                if k[0] == "@" and k[-1] == "@" and len(k) > 2:
                    pass
                elif k[0] == "[" and k[-1] == "]" and len(k) > 2:
                    pass
                else:
                    if k in string_action_vocab:
                        out_v = string_action_vocab[k]
                        self._inp_to_act[inp_v] = out_v
                        self._act_to_inp[out_v] = inp_v
        else:
            string_action_vocab = {}
            for k, v in self.out_vocab:
                if str_action_re.match(k):
                    string_action_vocab[str_action_re.match(k).group(1)] = v
            for k, inp_v in self.inp_vocab:
                if k[0] == "@" and k[-1] == "@" and len(k) > 2:
                    pass
                elif k[0] == "[" and k[-1] == "]" and len(k) > 2:
                    pass
                else:
                    if k not in string_action_vocab:
                        print(k)
                    assert (k in string_action_vocab)
                    out_v = string_action_vocab[k]
                    self._inp_to_act[inp_v] = out_v
                    self._act_to_inp[out_v] = inp_v


class PtrGenOutput2(_PtrGenOutput):
    def forward(self, x:torch.Tensor, inptensor:torch.Tensor=None, attn_scores:torch.Tensor=None, out_mask:torch.Tensor=None):  # (batsize, hdim), (batsize, numactions)
        """
        :param x:               (batsize, hdim)
        :param inptensor:       (batsize, seqlen, encdim)
        :param attn_scores:     (batsize, seqlen)
        :param out_mask:        (batsize, outvocsize)
        :return:
        """
        if self.training:       # !!!! removing action mask stuff during training !!! probably want to put back
            out_mask = None

        # - generation probs
        gen_probs = self.gen_lin(x)
        if self.out_map is not None:
            gen_probs = gen_probs.index_select(1, self.out_map)
        if out_mask is not None:
            gen_probs = gen_probs + torch.log(out_mask.float())
        gen_probs = self.logsm(gen_probs)

        if inptensor is None:
            assert(attn_scores is None)
            return gen_probs

        else:
            assert(attn_scores is not None)

            # - point or generate probs
            ptr_or_gen_probs = self.copy_or_gen(x)  # (batsize, 2)
            if out_mask is not None:
                cancopy_mask = self._inp_actmask.unsqueeze(0) * out_mask
                cancopy_mask = cancopy_mask.sum(
                    1) > 0  # if any overlap between allowed actions and actions doable by copy, set mask to 1
                cancopy_mask = torch.stack([torch.ones_like(cancopy_mask), cancopy_mask], 1)
                ptr_or_gen_probs = ptr_or_gen_probs + torch.log(cancopy_mask.float())
            ptr_or_gen_probs = self.logsm(ptr_or_gen_probs)

            # - copy probs
            # self.naningrad = torch.nn.Parameter(self.naningrad[:attn_scores.size(0), :attn_scores.size(1)])
            # attn_scores = attn_scores + self.naningrad
            attn_probs = self.sm(attn_scores)
            # get distributions over input vocabulary
            ctx_ids = inptensor
            inpdist = torch.zeros(gen_probs.size(0), self.inp_vocab.number_of_ids(), dtype=torch.float,
                                  device=gen_probs.device)
            inpdist.scatter_add_(1, ctx_ids, attn_probs)

            # map to distribution over output actions
            ptr_scores = torch.zeros(gen_probs.size(0), self.out_vocab.number_of_ids(),
                                     dtype=torch.float, device=gen_probs.device)  # - np.infty
            ptr_scores.scatter_(1, self._inp_to_act.unsqueeze(0).repeat(gen_probs.size(0), 1),
                                inpdist)
            # ptr_scores_infmask = ptr_scores == 0
            ptr_scores = ptr_scores.masked_fill(ptr_scores == 0, 0)
            # ptr_scores = ptr_scores + self.naningrad2
            ptr_probs = torch.log(ptr_scores)
            # ptr_probs = ptr_probs + self.naningrad2

            # - mix
            out_probs = torch.stack([ptr_or_gen_probs[:, 0:1] + gen_probs, ptr_or_gen_probs[:, 1:2] + ptr_probs], 1)
            out_probs = torch.logsumexp(out_probs, 1)

            # out_probs = out_probs.masked_fill(out_probs == 0, 0)
            return out_probs, ptr_or_gen_probs, gen_probs, attn_probs


class PtrGenOutput(_PtrGenOutput):
    def forward(self, x:torch.Tensor, inptensor:torch.Tensor=None, attn_scores:torch.Tensor=None, out_mask:torch.Tensor=None):  # (batsize, hdim), (batsize, numactions)
        """
        :param x:               (batsize, hdim)
        :param inptensor:       (batsize, seqlen, encdim)
        :param attn_scores:     (batsize, seqlen)
        :param out_mask:        (batsize, outvocsize)
        :return:
        """
        # - generation probs
        gen_probs = self.gen_lin(x)
        fullvocsize = self.out_vocab.number_of_ids()
        if gen_probs.size(1) < fullvocsize:
            gen_probs = torch.cat([gen_probs,
                                   torch.log(torch.zeros_like(gen_probs[:, :1]))
                                        .repeat(1, fullvocsize - gen_probs.size(1))],
                                  1)
        if self.gen_mask is not None: # and False:     # : enable back
            gen_probs = gen_probs + torch.log(self.gen_mask.float()[None, :])
        if out_mask is not None:
            gen_probs = gen_probs + torch.log(out_mask.float())

        if inptensor is None:
            assert(attn_scores is None)
            return self.logsm(gen_probs)

        else:
            assert(attn_scores is not None)

            # - point or generate probs
            ptr_or_gen_probs = self.copy_or_gen(x)  # (batsize, 2)
            # # : remove this -- disables copy
            # ptr_or_gen_mask = torch.zeros_like(ptr_or_gen_probs)
            # ptr_or_gen_mask[:, 0] = 1
            # ptr_or_gen_probs = ptr_or_gen_probs + torch.log(ptr_or_gen_mask)
            if out_mask is not None:
                cancopy_mask = self._inp_actmask.unsqueeze(0).float() * out_mask.float()
                cancopy_mask = cancopy_mask.sum(
                    1) > 0  # if any overlap between allowed actions and actions doable by copy, set mask to 1
                cancopy_mask = torch.stack([torch.ones_like(cancopy_mask), cancopy_mask], 1)
                ptr_or_gen_probs = ptr_or_gen_probs + torch.log(cancopy_mask.float())
            ptr_or_gen_probs = self.sm(ptr_or_gen_probs)

            # - copy probs
            # self.naningrad = torch.nn.Parameter(self.naningrad[:attn_scores.size(0), :attn_scores.size(1)])
            # attn_scores = attn_scores + self.naningrad
            attn_probs = self.sm(attn_scores)
            # get distributions over input vocabulary
            ctx_ids = inptensor
            inpdist = torch.zeros(gen_probs.size(0), self.inp_vocab.number_of_ids(), dtype=torch.float,
                                  device=gen_probs.device)
            inpdist.scatter_add_(1, ctx_ids, attn_probs)

            # map to distribution over output actions
            ptr_scores = torch.zeros(gen_probs.size(0), self.out_vocab.number_of_ids(),
                                     dtype=torch.float, device=gen_probs.device)  # - np.infty
            ptr_scores.scatter_(1, self._inp_to_act.unsqueeze(0).repeat(gen_probs.size(0), 1),
                                inpdist)
            # ptr_scores_infmask = ptr_scores == 0
            # ptr_scores = ptr_scores + self.naningrad2
            # ptr_probs = ptr_probs + self.naningrad2

            # - mix
            out_probs = ptr_or_gen_probs[:, 0:1] * self.sm(gen_probs) + ptr_or_gen_probs[:, 1:2] * ptr_scores
            out_probs = torch.masked_fill(out_probs, out_probs == 0, 0)
            out_probs = torch.log(out_probs)

            # out_probs = out_probs.masked_fill(out_probs == 0, 0)
            return out_probs, ptr_or_gen_probs, self.sm(gen_probs), attn_probs


class PtrGenOutput3(PtrGenOutput):
    def forward(self, x:torch.Tensor, inptensor:torch.Tensor=None, attn_scores:torch.Tensor=None, out_mask:torch.Tensor=None):  # (batsize, hdim), (batsize, numactions)
        """
        :param x:               (batsize, hdim)
        :param inptensor:       (batsize, seqlen, encdim)
        :param attn_scores:     (batsize, seqlen)
        :param out_mask:        (batsize, outvocsize)
        :return:
        """
        if self.training:       # !!!! removing action mask stuff during training !!! probably want to put back
            out_mask = None

        # - generation probs
        gen_probs = self.gen_lin(x)
        if self.out_map is not None:
            gen_probs = gen_probs.index_select(1, self.out_map)
        if out_mask is not None:
            gen_probs = gen_probs + torch.log(out_mask.float())

        if inptensor is None:
            assert(attn_scores is None)
            return self.logsm(gen_probs)
        else:
            assert(attn_scores is not None)

            # - point or generate probs
            ptr_or_gen_scores = self.copy_or_gen(x)  # (batsize, 2)
            if out_mask is not None:
                cancopy_mask = self._inp_actmask.unsqueeze(0) * out_mask
                cancopy_mask = cancopy_mask.sum(
                    1) > 0  # if any overlap between allowed actions and actions doable by copy, set mask to 1
                cancopy_mask = torch.stack([torch.ones_like(cancopy_mask), cancopy_mask], 1)
                ptr_or_gen_scores = ptr_or_gen_scores + torch.log(cancopy_mask.float())

            # - copy probs
            # get distributions over input vocabulary
            ctx_ids = inptensor
            inpdist = torch.zeros(gen_probs.size(0), self.inp_vocab.number_of_ids(), dtype=torch.float,
                                  device=gen_probs.device)
            inpdist.scatter_add_(1, ctx_ids, attn_scores)

            # map to output actions
            ptr_scores = torch.zeros(gen_probs.size(0), self.out_vocab.number_of_ids(),
                                     dtype=torch.float, device=gen_probs.device)  # - np.infty
            ptr_scores.scatter_(1, self._inp_to_act.unsqueeze(0).repeat(gen_probs.size(0), 1),
                                inpdist)

            # - mix
            out_probs = torch.stack([ptr_or_gen_scores[:, 0:1] + gen_probs, ptr_or_gen_scores[:, 1:2] + ptr_scores], 1)
            out_probs = self.logsm(out_probs.sum(1))

            # out_probs = out_probs.masked_fill(out_probs == 0, 0)
            return out_probs, ptr_or_gen_scores, gen_probs, self.sm(attn_scores)


class GatedFF(torch.nn.Module):
    def __init__(self, indim, odim, dropout=0., activation=None, zdim=None, **kw):
        super(GatedFF, self).__init__(**kw)
        self.dim = indim
        self.odim = odim
        self.zdim = self.dim * 4 if zdim is None else zdim

        self.activation = torch.nn.CELU() if activation is None else activation

        self.linA = torch.nn.Linear(self.dim, self.zdim)
        self.linB = torch.nn.Linear(self.zdim, self.odim)
        self.linMix = torch.nn.Linear(self.zdim, self.odim)
        # self.linMix.bias.data.fill_(3.)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *inps):
        h = inps[-1]
        _h = torch.cat(inps, -1)
        _cand = self.linA(self.dropout(_h))
        _cand = self.activation(_cand)
        cand = self.linB(self.dropout(_cand))
        mix = torch.sigmoid(self.linMix(_cand))
        ret = h * mix + cand * (1 - mix)
        return ret


class SGRUCell(torch.nn.Module):
    def __init__(self, dim, bias=True, dropout=0., **kw):
        super(SGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 5, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        inp = self.dropout(inp)
        gates = self.gateW(inp)
        gates = list(gates.chunk(5, 1))
        rx = torch.sigmoid(gates[0])
        rh = torch.sigmoid(gates[1])
        z_gates = gates[2:5]
        # z_gates[2] = z_gates[2] - self.gate_bias
        z = torch.softmax(torch.stack(z_gates, 2), -1)
        inp = torch.cat([x * rx, h * rh], 1)
        inp = self.dropout(inp)
        u = self.gateU(inp)
        u = torch.tanh(u)
        h_new = torch.stack([x, h, u], 2) * z
        h_new = h_new.sum(-1)
        return h_new



if __name__ == '__main__':
    try_dgru_cell()