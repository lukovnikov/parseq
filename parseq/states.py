import re
import typing
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy, copy
from typing import Union, List, Dict, Any
import qelos as q

import torch
from nltk import Tree

from parseq.grammar import ActionTree, AlignedActionTree
from parseq.vocab import SentenceEncoder, FuncQueryEncoder


class State(object):
    """
    A State object represents the state of a batch of examples.
    """
    def __init__(self, **kwdata):
        super(State, self).__init__()
        self._schema_keys = set()
        self._length = None
        if len(kwdata) > 0:
            self.set(**kwdata)

    def set(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k) and not k in self._schema_keys:
                raise AttributeError(f"Key {k} cannot be assigned to this {type(self)}, attribute taken.")
            if not (isinstance(v, (list, torch.Tensor, State, type(None)))):
                raise Exception(f"argument {k} has type {type(v)}. Only list, torch.Tensor and State are allowed.")
            self._length = len(v) if self._length is None and v is not None else self._length
            assert(v is None or self._length == len(v))
            setattr(self, k, v)
            self._schema_keys.add(k)

    def get(self, k:str):
        return getattr(self, k)

    def make_copy(self, ret=None, detach=None, deep=True):
        detach = deep if detach is None else detach
        ret = type(self)() if ret is None else ret
        for k in self._schema_keys:
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                ret.set(**{k:v.clone().detach() if detach else v.clone()})
            elif isinstance(v, State):
                ret.set(**{k: v.make_copy(detach=detach, deep=deep)})
            else:
                ret.set(**{k: deepcopy(v) if deep else copy(v)})
        return ret

    def __len__(self):
        return self._length

    @classmethod
    def merge(cls, states:List['State'], ret=None):
        """
        merge states into single state
        """
        # create new object of this type if one is not given yet
        ret = cls() if ret is None else ret
        # try merge tensors and substates automatically

        merge_schema = None
        for state in states:
            merge_schema = state._schema_keys if merge_schema is None else merge_schema
            assert(merge_schema == state._schema_keys)

        for k in merge_schema:
            vs = [state.get(k) for state in states]

            if all([isinstance(vse, (list,)) for vse in vs]):
                v = [vsee for vse in vs for vsee in vse]
            elif all([isinstance(vse, (torch.Tensor,)) for vse in vs]):
                v = torch.cat(vs, 0)
            elif all([isinstance(vse, (State,)) for vse in vs]):
                assert(all([type(vs[0]) == type(vse) for vse in vs]))
                v = type(vs[0]).merge(vs)
            else:
                raise Exception("Can not merge given states. Schema mismatch.")
            ret.set(**{k:v})

        return ret

    def __getitem__(self, item:Union[int, slice, List[int], np.ndarray, torch.Tensor]):    # slicing and getitem
        # TODO: make more efficient
        if isinstance(item, int):
            item = slice(item, item+1)
        ret = type(self)()
        for k in self._schema_keys:
            v = getattr(self, k)
            if not isinstance(item, slice) and isinstance(v, list):
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                if isinstance(item, np.ndarray):
                    item = list(item)
                vslice = [v[item_e] for item_e in item]
            else:
                vslice = v[item]
            ret.set(**{k:vslice})
        return ret

    def __setitem__(self, item:Union[int, slice, List[int], np.ndarray, torch.Tensor],
                    value:'State'):     # value should be of same type as self
        # TODO: make more efficient
        assert(type(self) == type(value))
        assert(self._schema_keys == value._schema_keys)
        if isinstance(item, int):
            item = slice(item, item+1)
        ret = self[item]
        assert(len(ret) == len(value))
        for k in self._schema_keys:
            v = getattr(self, k)
            assert(True)        # TODO: assert type match
            if not isinstance(item, slice) and isinstance(v, list):
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                if isinstance(item, np.ndarray):
                    item = list(item)
                for i, item_e in enumerate(item):
                    v[item_e] = value.get(k)[i]
            else:
                v[item] = value.get(k)
        return ret

    def to(self, device):
        for k in self._schema_keys:
            v = getattr(self, k)
            if isinstance(v, list):
                if len(set([type(ve) for ve in v]) & {torch.Tensor, State}) > 0:
                    for i in range(len(v)):
                        v[i] = v[i].to(device)
            else:
                setattr(self, k, v.to(device))
        return self


class ListState(State):
    """ State that contains a list of stateful things. Useful for list of states. """
    def __init__(self, *elems:Union[State, torch.Tensor]):
        super(ListState, self).__init__()
        self._list = []
        self._length = None
        for elem in elems:
            self.append(elem)

    def append(self, x):
        if self._length is not None:
            assert(len(x) == self._length)
        else:
            self._length = len(x)
        self._list.append(x)

    def set(self, i:Union[int, slice, List[int]], x:Union[State, torch.Tensor, List[Union[State, torch.Tensor]]]):
        _x = [x] if not isinstance(x, list) else x
        for _xe in _x:
            if self._length is not None:
                assert(len(_xe) == self._length)
            else:
                self._length = len(_xe)
        if isinstance(i, list):
            for ie, xe in zip(i, x):
                self._list[ie] = xe
        else:
            self._list[i] = x

    def get(self, i:Union[int, slice, List[int]]):
        if isinstance(i, list):
            ret = []
            for ie in i:
                ret.append(self._list[ie])
            return ret
        else:
            return self._list[i]

    @classmethod
    def merge(cls, states:List['ListState'], ret=None):
        ret = cls() if ret is None else ret
        listlen = None
        for state in states:
            listlen = len(state._list) if listlen is None else listlen
            assert(listlen == len(state._list))

        for i in range(listlen):
            vs = [state.get(i) for state in states]

            if all([isinstance(vse, (list,)) for vse in vs]):
                v = [vsee for vse in vs for vsee in vse]
            elif all([isinstance(vse, (torch.Tensor,)) for vse in vs]):
                v = torch.cat(vs, 0)
            elif all([isinstance(vse, (State,)) for vse in vs]):
                assert (all([type(vs[0]) == type(vse) for vse in vs]))
                v = type(vs[0]).merge(vs)
            else:
                raise Exception("Can not merge given states. Schema mismatch.")

            return ret

    def __getitem__(self, item:Union[int, slice, List[int], np.ndarray, torch.Tensor]):    # slicing and getitem
        if isinstance(item, int):
            item = slice(item, item+1)
        ret = type(self)()
        for k in range(len(self._list)):
            v = self.get(k)

            if not isinstance(item, slice) and isinstance(v, list):
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                if isinstance(item, np.ndarray):
                    item = list(item)
                vslice = [v[item_e] for item_e in item]
            else:
                vslice = v[item]

            ret.append(vslice)
        return ret

    def __setitem__(self, item:Union[int, slice, List[int], np.ndarray, torch.Tensor],
                    value:'ListState'):     # value should be of same type as self
        assert(type(self) == type(value))
        assert(len(self._list) == len(value._list))
        if isinstance(item, int):
            item = slice(item, item+1)
        ret = self[item]
        assert(len(ret) == len(value))

        for k in range(len(self._list)):
            v = self.get(k)
            assert(True)        # TODO: assert type match
            if not isinstance(item, slice) and isinstance(v, list):
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                if isinstance(item, np.ndarray):
                    item = list(item)
                for i, item_e in enumerate(item):
                    v[item_e] = value.get(k)[i]
            else:
                v[item] = value.get(k)
        return ret

    def to(self, device):
        for k in range(len(self._list)):
            v = self.get(k)
            if isinstance(v, list):
                if len(set([type(ve) for ve in v]) & {torch.Tensor, State}) > 0:
                    for i in range(len(v)):
                        v[i] = v[i].to(device)
            else:
                self.set(k, v.to(device))
        return self


class DecodableState(State, ABC):
    """
    Subclasses must have "out_probs" substate and a "gold_actions" attribute for use with TF/Free/..-transitions in decoding
    """
    @abstractmethod
    def start_decoding(self):
        pass

    @abstractmethod
    def is_terminated(self)-> List[bool]:
        """
        Must return a list of booleans whether each elem in this state is terminated.
        """
        pass

    @abstractmethod
    def step(self, action:Union[torch.Tensor, List[Union[str, torch.Tensor]]]=None):
        pass

    def all_terminated(self)-> bool:
        return all(self.is_terminated())


class TrainableState(State, ABC):
    @abstractmethod
    def get_gold(self, i:int) -> torch.Tensor:
        pass


class TrainableDecodableState(TrainableState, DecodableState): pass


def batchstack(x:List[torch.Tensor]):
    """ Stack given tensors along dim but equalize their dimensions first. """
    maxlen = 0
    for xe in x:
        maxlen = max(maxlen, xe.size(0))
    othersizes = x[0].size()[1:]
    x = [torch.cat([xe, xe.new_zeros(maxlen - xe.size(0), *othersizes)]) for xe in x]
    ret = torch.stack(x, 0)
    return ret


class BasicDecoderState(TrainableDecodableState):
    """
    Basic state object for seq2seq
    """
    endtoken = "@END@"
    def __init__(self,
                 inp_strings:List[str]=None,
                 gold_strings:List[str]=None,
                 sentence_encoder:SentenceEncoder=None,
                 query_encoder:SentenceEncoder=None,
                 **kw):
        if inp_strings is None:
            super(BasicDecoderState, self).__init__(**kw)
        else:
            kw = kw.copy()
            kw.update({"inp_strings": inp_strings, "gold_strings": gold_strings})
            super(BasicDecoderState, self).__init__(**kw)

            self.sentence_encoder = sentence_encoder
            self.query_encoder = query_encoder

            self.set(followed_actions = [[] for _ in self.inp_strings])
            self.set(_is_terminated = [False for _ in self.inp_strings])
            self.set(_timesteps = [0 for _ in self.inp_strings])

            if sentence_encoder is not None:
                x = [sentence_encoder.convert(x, return_what="tensor,tokens") for x in self.inp_strings]
                x = list(zip(*x))
                x = {"inp_tensor": batchstack(x[0]),
                     "inp_tokens": list(x[1])}
                self.set(**x)
            if self.gold_strings is not None:
                if query_encoder is not None:
                    x = [query_encoder.convert(x, return_what="tensor,tokens") for x in self.gold_strings]
                    x = list(zip(*x))
                    x = {"gold_tensor": batchstack(x[0]),
                         "gold_tokens": list(x[1])}
                    self.set(**x)

    # State API override implementation
    def make_copy(self, ret=None, detach=None, deep=True):
        ret = super(BasicDecoderState, self).make_copy(ret=ret, detach=detach, deep=deep)
        ret.sentence_encoder = self.sentence_encoder
        ret.query_encoder = self.query_encoder
        return ret

    @classmethod
    def merge(cls, states:List['BasicDecoderState'], ret=None):
        assert(all([state.sentence_encoder == states[0].sentence_encoder and state.query_encoder == states[0].query_encoder for state in states]))
        ret = super(BasicDecoderState, cls).merge(states, ret=ret)
        ret.sentence_encoder = states[0].sentence_encoder
        ret.query_encoder = states[0].query_encoder
        return ret

    def __getitem__(self, item):
        ret = super(BasicDecoderState, self).__getitem__(item)
        ret.sentence_encoder = self.sentence_encoder
        ret.query_encoder = self.query_encoder
        return ret

    def __setitem__(self, key, value:'BasicDecoderState'):
        assert(value.sentence_encoder == self.sentence_encoder and value.query_encoder == self.query_encoder)
        ret = super(BasicDecoderState, self).__setitem__(key, value)
        return ret

    # DecodableState API implementation
    def is_terminated(self):
        return self._is_terminated

    def start_decoding(self):
        # initialize prev_action
        qe = self.query_encoder
        self.set(prev_actions=torch.tensor([qe.vocab[qe.vocab.starttoken] for _ in self.inp_strings],
                                                   device=self.inp_tensor.device, dtype=torch.long))

    def step(self, tokens:Union[torch.Tensor, List[Union[str, torch.Tensor]]]):
        # TODO: make more efficient
        qe = self.query_encoder
        assert(len(tokens) == len(self))
        for i, token in enumerate(tokens):
            if isinstance(token, torch.Tensor):
                token = qe.vocab(int(token.detach().cpu().item()))
            if not self.is_terminated()[i]:
                self.prev_actions[i] = qe.vocab[token] if isinstance(token, str) else token.to(self.inp_tensor.device)
                self.followed_actions[i].append(token)
                self._timesteps[i] += 1
                self._is_terminated[i] |= token == self.endtoken

    def get_gold(self, i):
        return self.gold_tensor[:, i]


# class _StateBase(object):
#     def __init__(self):
#         super(_StateBase, self).__init__()
#         self._list = []
#         self._attr_keys = set()
#
#     @property
#     def all_attr_keys(self):
#         return set(range(len(self._list))) | set(self._attr_keys)
#
#     def __getitem__(self, item):
#         if isinstance(item, str):
#             return getattr(self, item)
#         elif isinstance(item, int):
#             return self._list[item]
#
#     def __setitem__(self, item, v):
#         if isinstance(item, str):
#             if item in self.__dict__:
#                 assert (item in self._attr_keys)
#             self._attr_keys.add(item)
#             setattr(self, item, v)
#         elif isinstance(item, int):
#             self._list[item] = v
#
#     def __contains__(self, item):
#         return item in self.all_attr_keys
#
#     def append(self, item):
#         self._list.append(item)
#
#     def listlen(self):
#         return len(self._list)
#
#     def dictlen(self):
#         return len(self._attr_keys)
#
#     def alllen(self):
#         return self.listlen() + self.dictlen()
#
#
# class State(_StateBase):
#     """
#     Describes a single state, corresponding to one example.
#     May contain other states as its immediate arguments. These states are taken care of automatically in batching.
#
#     Attributes passed during construction or assigned using __setitem__
#         are registered as batchable, copyable and movable attributes.
#         These attributes are automatically copied, batched and moved to a different device
#         when .make_copy(), batching calls or .to() are done.
#
#     All other attributes on the State object are not automatically handled in copy, batching, or moving
#     by State and are considered the responsibility of subclasses.
#     """
#     def __init__(self, *argattrs, **kwattrs):
#         super(State, self).__init__()
#         self._list = []
#         self._attr_keys = set(kwattrs.keys())
#         for k, v in kwattrs.items():
#             self[k] = v
#         for i, v in enumerate(argattrs):
#             self.append(v)
#
#     def make_copy(self, ret=None):
#         """
#         Make and return a deep copy of this state.
#         Only deepcopies keys in self.all_attr_keys.
#         Doesn't copy anything else.
#         """
#         ret = type(self)() if ret is None else ret
#         for k in self.all_attr_keys:
#             ret[k] = make_copy(self[k])
#         return ret
#
#     def to(self, device):
#         """
#         Move all torch Tensors to given device.
#         """
#         # check dict for any tensors or other states, and call .to on them.
#         for k in self.all_attr_keys:
#             if isinstance(self[k], (torch.Tensor, State)):
#                 self[k] = self[k].to(device)
#         return self
#
#     def get_batch_class(self): return StateBatch
#
#
# class StateBatch(_StateBase):
#     def __init__(self, states:List=None, **kw):
#         super(StateBatch, self).__init__(**kw)
#         self.states = states if states is not None else []
#         self._list = []
#         self._attr_keys = set()
#
#     def to(self, device):
#         """
#         Move all torch tensors onto given device.
#         """
#         # 2. move any attached statebatches to device
#         for k in self.all_attr_keys:
#             if isinstance(self[k], (torch.Tensor, StateBatch)):
#                 self[k] = self[k].to(device)
#         # 3. move anything else to device
#         self.to_own(device)
#         return self
#
#     def to_own(self, device):
#         pass
#
#     def make_copy(self):
#         """
#         Make and return a copy of this state batch.
#         :return:
#         """
#         states = [state.make_copy() for state in self.states]
#         ret = type(self)(states)
#         return ret
#
#     def __len__(self):
#         return len(self.states)


# def batch(states:Union[List[State], List[torch.Tensor]]):
#     if isinstance(states[0], torch.Tensor):
#         return torch.stack(states, 0)
#     elif isinstance(states[0], State):
#         return states[0].get_batch_class()(states)
#     else:
#         raise Exception(f"unsupported type: '{type(states[0])}'")
#
#
# def unbatch(statebatch:Union[StateBatch, torch.Tensor]):
#     if isinstance(statebatch, torch.Tensor):
#         return [x[0] for x in torch.split(statebatch, 1, 0)]
#     elif isinstance(statebatch, StateBatch):
#         return statebatch.unbatch()
#     else:
#         raise Exception(f"unsupported type: '{type(statebatch)}'")
#
#
# def make_copy(v):
#     if isinstance(v, torch.Tensor):
#         return v.clone().detach()
#     elif isinstance(v, State):
#         return v.make_copy()
#     else:
#         return deepcopy(v)


# class DecodableState(State, ABC):
#     """
#     Subclasses must have "out_probs" substate and a "gold_actions" attribute for use with TF/Free/..-transitions in decoding
#     """
#     @abstractmethod
#     def start_decoding(self):
#         pass
#
#     @abstractmethod
#     def is_terminated(self):
#         pass
#
#     @abstractmethod
#     def get_decoding_step(self):
#         pass
#
#     @abstractmethod
#     def step(self, action:Union[torch.Tensor, str]=None):
#         pass
#
#     @abstractmethod
#     def get_action_scores(self):
#         pass
#
#     def get_batch_class(self): return DecodableStateBatch
#
#
# class DecodableStateBatch(StateBatch):
#     def start_decoding(self):
#         self.unbatch()
#         for state in self.states:
#             state.start_decoding()
#         self.batch()
#
#     def all_terminated(self):
#         return all([state.is_terminated() for state in self.states])
#
#     def step(self, actions:Union[torch.Tensor, List[Union[torch.Tensor, str]], List[str]]=None):
#         self.unbatch()
#         if actions is not None:
#             [state.step(action) for state, action in zip(self.states, actions)]
#         else:
#             [state.step() for state in self.states]
#         self.batch()
#
#     def get_action_scores(self):
#         return torch.stack([state.get_action_scores() for state in self.states], 0)
#
#
# class BasicState(DecodableState):
#     """
#     Basic state object for seq2seq
#     """
#     endtoken = "@END@"
#     def __init__(self, inp:str=None, out:str=None,
#                  sentence_encoder:SentenceEncoder=None,
#                  query_encoder:SentenceEncoder=None, **kw):
#         super(BasicState, self).__init__(**kw)
#         self.inp_string, self.gold_string = inp, out
#         self.sentence_encoder, self.query_encoder = sentence_encoder, query_encoder
#         self.inp_tensor = None
#         self._timestep = 0
#
#         self.followed_actions = []
#         self._is_terminated = False
#
#         self["out_probs"] = State()
#         self["nn_state"] = State()
#
#         self.initialize()
#
#     def initialize(self):
#         if self.sentence_encoder is not None:
#             self.inp_tensor, self.inp_actions = self.sentence_encoder.convert(self.inp_string, return_what="tensor,tokens")
#         if self.gold_string is not None:
#             if self.query_encoder is not None:
#                 self["gold_tensor"], self.gold_actions = self.query_encoder.convert(self.gold_string, return_what="tensor,tokens")
#         if self.inp_tensor is not None:
#             self.nn_state["inp_tensor"] = self.inp_tensor
#
#
#     def make_copy(self, ret=None):
#         ret = type(self)(deepcopy(self.inp_string), deepcopy(self.gold_string)) if ret is None else ret
#         ret = super(BasicState, self).make_copy(ret)
#         do_shallow = {"sentence_encoder", "query_encoder"}
#         for k, v in self.__dict__.items():
#             if k in self.all_attr_keys:
#                 pass        # already copied by State
#             elif k in do_shallow:
#                 setattr(ret, k, v)  # shallow copy
#             else:
#                 setattr(ret, k, deepcopy(v))
#         return ret
#
#     def is_terminated(self):
#         return self._is_terminated
#
#     def start_decoding(self):
#         # initialize prev_action
#         self.nn_state["prev_action"] = torch.tensor(self.query_encoder.vocab[self.query_encoder.vocab.starttoken],
#                                                    device=self.nn_state["inp_tensor"].device, dtype=torch.long)
#
#     def get_decoding_step(self):
#         return self._timestep
#
#     def step(self, token:Union[torch.Tensor, str]):
#         if isinstance(token, torch.Tensor):
#             token = self.query_encoder.vocab(int(token.detach().cpu().numpy()))
#         if not self.is_terminated():
#             self.nn_state["prev_action"] = torch.tensor(self.query_encoder.vocab[token],
#                                                        device=self.nn_state["inp_tensor"].device, dtype=torch.long)
#             self.followed_actions.append(token)
#             self._timestep += 1
#             if token == self.endtoken:
#                 self._is_terminated = True
#
#     def get_action_scores(self):
#         return self.out_probs[-1]
#
#
# class FuncTreeState(State):
#     """
#     State object containing
#     """
#     def __init__(self, inp:str, out:str=None,
#                  sentence_encoder:SentenceEncoder=None,
#                  query_encoder:FuncQueryEncoder=None, **kw):
#         super(FuncTreeState, self).__init__(**kw)
#         self.inp_string, self.out_string = inp, out
#         self.sentence_encoder, self.query_encoder = sentence_encoder, query_encoder
#         self.inp_tensor = None
#
#         self.has_gold = False
#         self.use_gold = False
#
#         self.out_tree = None
#         self.out_rules = None
#         self.nn_state = State()
#         self.open_nodes = []
#
#         self.initialize()
#
#     def initialize(self):
#         if self.sentence_encoder is not None:
#             self.inp_tensor, self.inp_tokens = self.sentence_encoder.convert(self.inp_string, return_what="tensor,tokens")
#         if self.out_string is not None:
#             self.has_gold = True
#             self.use_gold = self.has_gold
#             if self.query_encoder is not None:
#                 self.gold_tensor, self.gold_tree, self.gold_rules = self.query_encoder.convert(self.out_string, return_what="tensor,tree,actions")
#                 assert(self.gold_tree.action() is not None)
#         if self.inp_tensor is not None:
#             self.nn_state["inp_tensor"] = self.inp_tensor
#
#     def make_copy(self, ret=None):
#         ret = type(self)(deepcopy(self.inp_string), deepcopy(self.out_string)) if ret is None else ret
#         ret = super(FuncTreeState, self).make_copy(ret)
#         do_shallow = {"sentence_encoder", "query_encoder"}
#         for k, v in self.__dict__.items():
#             if k in self.all_attr_keys:
#                 pass        # already copied by State
#             elif k in do_shallow:
#                 setattr(ret, k, v)  # shallow copy exclusions
#             else:
#                 setattr(ret, k, deepcopy(v))
#         return ret
#
#     def is_terminated(self):
#         return len(self.open_nodes) == 0
#
#     def get_open_nodes(self, tree=None):
#         if self.out_tree is None and tree is None:
#             return []
#         tree = tree if tree is not None else self.out_tree
#         ret = []
#         for child in tree:
#             ret = ret + self.get_open_nodes(child)
#         if tree.label() in self.query_encoder.grammar.rules_by_type:  # non terminal
#             ret = ret + [tree]
#         return ret
#
#     def start_decoding(self):
#         start_type = self.query_encoder.grammar.start_type
#         self.out_tree = AlignedActionTree(start_type, children=[])
#         self.out_rules = []
#         self.open_nodes = [self.out_tree] + self.open_nodes
#         # align to gold
#         if self.use_gold:
#             self.out_tree._align = self.gold_tree
#         # initialize prev_action
#         self.nn_state["prev_action"] = torch.tensor(self.query_encoder.vocab_actions[self.query_encoder.start_action],
#                                                     device=self.nn_state["inp_tensor"].device, dtype=torch.long)
#
#     def apply_rule(self, node:AlignedActionTree, rule:Union[str, int]):
#         # if node.label() not in self.query_encoder.grammar.rules_by_type \
#         #         or rule not in self.query_encoder.grammar.rules_by_type[node.label()]:
#         #     raise Exception("something wrong")
#         #     return
#         self.nn_state["prev_action"] = torch.ones_like(self.nn_state["prev_action"]) \
#                                        * self.query_encoder.vocab_actions[rule]
#         self.out_rules.append(rule)
#         assert(node == self.open_nodes[0])
#         if isinstance(rule, str):
#             ruleid = self.query_encoder.vocab_actions[rule]
#             rulestr = rule
#         elif isinstance(rule, int):
#             ruleid = rule
#             rulestr = self.query_encoder.vocab_actions(rule)
#
#         head, body = rulestr.split(" -> ")
#         func_splits = body.split(" :: ")
#         sibl_splits = body.split(" -- ")
#
#         if len(sibl_splits) > 1:
#             raise Exception("sibling rules no longer supported")
#
#         self.open_nodes.pop(0)
#
#         if node.label()[-1] in "*+" and body != f"{head}:END@":  # variable number of children
#             # create new sibling node
#             parent = node.parent()
#             i = len(parent)
#
#             new_sibl_node = AlignedActionTree(node.label(), [])
#             parent.append(new_sibl_node)
#
#             # manage open nodes
#             self.open_nodes = ([new_sibl_node]
#                                if (new_sibl_node.label() in self.query_encoder.grammar.rules_by_type
#                                    or new_sibl_node.label()[:-1] in self.query_encoder.grammar.rules_by_type)
#                                else []) \
#                               + self.open_nodes
#
#             if self.use_gold:
#                 gold_child = parent._align[i]
#                 new_sibl_node._align = gold_child
#
#         if len(func_splits) > 1 :
#             rule_arg, rule_inptypes = func_splits
#             rule_inptypes = rule_inptypes.split(" ")
#
#             # replace label of tree
#             node.set_label(rule_arg)
#             node.set_action(rule)
#
#             # align to gold
#             if self.use_gold:
#                 gold_children = node._align[:]
#
#             # create children nodes as open non-terminals
#             for i, child in enumerate(rule_inptypes):
#                 child_node = AlignedActionTree(child, [])
#                 node.append(child_node)
#
#                 if self.use_gold:
#                     child_node._align = gold_children[i]
#
#             # manage open nodes
#             self.open_nodes = [child_node for child_node in node if child_node.label() in self.query_encoder.grammar.rules_by_type]\
#                               + self.open_nodes
#         else:   # terminal
#             node.set_label(body)
#             node.set_action(rule)
#
#     def copy_token(self, node:AlignedActionTree, inp_position:int):
#         inplabel = self.inp_tokens[inp_position]
#         rule = f"<W> -> '{inplabel}'"
#         self.apply_rule(node, rule)
#
#     def get_valid_actions_at(self, node:AlignedActionTree):
#         action_mask = self.get_valid_action_mask_at(node)
#         valid_action_ids = action_mask.nonzero().cpu().numpy()
#         # TODO: finish: translate back to strings
#
#         # node_label = node.label()
#         # if node_label in self.query_encoder.grammar.rules_by_type:
#         #     if node_label[-1] in "*+":
#         #         ret = self.query_encoder.grammar.rules_by_type[node_label]
#         #         ret += self.query_encoder.grammar.rules_by_type[node_label[:-1]]
#         #         return ret
#         #     else:
#         #         return self.query_encoder.grammar.rules_by_type[node_label]
#         # else:
#         #     return [self.query_encoder.none_action]
#
#     def get_valid_action_mask_at(self, node:AlignedActionTree):
#         node_label = node.label()
#         ret = self.query_encoder.get_action_mask_for(node_label)
#         return ret
#
#     def get_gold_action_at(self, node:AlignedActionTree):
#         assert(self.use_gold)
#         return node._align.action()
#
#     def apply_action(self, node:AlignedActionTree, action:str):
#         # self.out_actions.append(action)
#         copyre = re.compile("COPY\[(\d+)\]")
#         if copyre.match(action):
#             self.copy_token(node, int(copyre.match(action).group(1)))
#         else:
#             self.apply_rule(node, action)


if __name__ == '__main__':
    pass