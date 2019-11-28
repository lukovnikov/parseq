import re
from typing import Set

import torch

from parseq.states import StateBatch
from parseq.vocab import Vocab


class TokenEmb(torch.nn.Module):
    def __init__(self, emb:torch.nn.Embedding, rare_token_ids:Set[int]=None, rare_id:int=None, **kw):
        super(TokenEmb, self).__init__(**kw)
        self.emb = emb
        self.rare_token_ids = rare_token_ids
        self.rare_id = rare_id
        if rare_id is not None and rare_token_ids is not None:
            # build id mapper
            id_mapper = torch.arange(emb.num_embeddings)
            for id in self.rare_token_ids:
                id_mapper[id] = self.rare_id
            self.register_buffer("id_mapper", id_mapper)
        else:
            self.register_buffer("id_mapper", None)

    def forward(self, x:torch.Tensor):
        if self.id_mapper is not None:
            x = self.id_mapper[x]
        ret = self.emb(x)
        return ret


class BasicGenOutput(torch.nn.Module):
    def __init__(self, h_dim:int, vocab:Vocab=None, **kw):
        super(BasicGenOutput, self).__init__(**kw)
        self.gen_lin = torch.nn.Linear(h_dim, vocab.number_of_ids(), bias=True)
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)

        self.vocab = vocab

        # rare output tokens
        self.rare_token_ids = vocab.rare_ids
        rare_id = 1
        if len(self.rare_token_ids) > 0:
            out_map = torch.arange(self.vocab.number_of_ids())
            for rare_token_id in self.rare_token_ids:
                out_map[rare_token_id] = rare_id
            self.register_buffer("out_map", out_map)
        else:
            self.register_buffer("out_map", None)

    def forward(self, x:torch.Tensor):
        # - generation probs
        gen_probs = self.gen_lin(x)
        if self.out_map is not None:
            gen_probs = gen_probs.index_select(1, self.out_map)
        gen_probs = self.logsm(gen_probs)
        return gen_probs


class SumPtrGenOutput(torch.nn.Module):
    def __init__(self, h_dim: int, inp_vocab:Vocab=None, out_vocab:Vocab=None, **kw):
        super(SumPtrGenOutput, self).__init__(**kw)
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

    def forward(self, x:torch.Tensor, statebatch:StateBatch, attn_scores:torch.Tensor):  # (batsize, hdim), (batsize, numactions)

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


class PtrGenOutput(torch.nn.Module):
    def __init__(self, h_dim: int, inp_vocab:Vocab=None, out_vocab:Vocab=None, **kw):
        super(PtrGenOutput, self).__init__(**kw)
        # initialize modules
        self.gen_lin = torch.nn.Linear(h_dim, out_vocab.number_of_ids(), bias=True)
        self.copy_or_gen = torch.nn.Linear(h_dim, 2, bias=True)
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

    def forward(self, x:torch.Tensor, statebatch:StateBatch, attn_scores:torch.Tensor):  # (batsize, hdim), (batsize, numactions)

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
        if self.training:       # !!!! removing action mask stuff during training !!! probably want to put back
            actionmask = None
        # endregion

        # - point or generate probs
        ptr_or_gen_probs = self.copy_or_gen(x)  # (batsize, 2)
        if actionmask is not None:
            cancopy_mask = self._inp_actmask.unsqueeze(0) * actionmask
            cancopy_mask = cancopy_mask.sum(
                1) > 0  # if any overlap between allowed actions and actions doable by copy, set mask to 1
            cancopy_mask = torch.stack([torch.ones_like(cancopy_mask), cancopy_mask], 1)
            ptr_or_gen_probs = ptr_or_gen_probs + torch.log(cancopy_mask.float())
        ptr_or_gen_probs = self.logsm(ptr_or_gen_probs)

        # - generation probs
        gen_probs = self.gen_lin(x)
        if self.out_map is not None:
            gen_probs = gen_probs.index_select(1, self.out_map)
        if actionmask is not None:
            gen_probs = gen_probs + torch.log(actionmask.float())
        gen_probs = self.logsm(gen_probs)

        # - copy probs
        attn_probs = self.sm(attn_scores)
        # get distributions over input vocabulary
        ctx_ids = statebatch["inp_tensor"]
        inpdist = torch.zeros(gen_probs.size(0), self.inp_vocab.number_of_ids(), dtype=torch.float,
                              device=gen_probs.device)
        inpdist.scatter_add_(1, ctx_ids, attn_probs)

        # map to distribution over output actions
        ptr_scores = torch.zeros(gen_probs.size(0), self.out_vocab.number_of_ids(),
                                 dtype=torch.float, device=gen_probs.device)  # - np.infty
        ptr_scores.scatter_(1, self._inp_to_act.unsqueeze(0).repeat(gen_probs.size(0), 1),
                            inpdist)
        ptr_probs = torch.log(ptr_scores)

        # - mix
        out_probs = ptr_or_gen_probs[:, 0:1] * gen_probs + ptr_or_gen_probs[:, 1:2] * ptr_probs

        out_probs = out_probs.masked_fill(out_probs == 0, 0)
        return out_probs, ptr_or_gen_probs, gen_probs, attn_probs