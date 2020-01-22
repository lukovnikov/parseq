import re
import typing
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy, copy
from typing import Union, List, Dict, Any
import qelos as q

import torch
from nltk import Tree

from parseq.grammar import ActionTree
from parseq.vocab import SequenceEncoder, FuncQueryEncoder, Vocab


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

    def is_empty(self):
        return len(self._schema_keys) == 0

    def set(self, k:str=None, v=None, **kw):
        if k is not None:
            if k in kw:
                raise Exception("Key {k} already in kwargs.")
            kw = kw.copy()
            kw.update({k: v})
        for k, v in kw.items():
            assert("." not in k)
            if hasattr(self, k) and not k in self._schema_keys:
                raise AttributeError(f"Key {k} cannot be assigned to this {type(self)}, attribute taken.")
            if not (isinstance(v, (np.ndarray, torch.Tensor, State, type(None)))):
                raise Exception(f"argument {k} has type {type(v)}. Only list, torch.Tensor and State are allowed.")
            if isinstance(v, State):
                self._length = v._length if self._length is None and v is not None else self._length
                assert(len(v) == self._length or v.is_empty())
                if v.is_empty():
                    v._length = self._length
            else:
                self._length = len(v) if self._length is None and v is not None else self._length
                assert(v is None or self._length == len(v))
            self.__dict__[k] = v
            self._schema_keys.add(k)

    def __setattr__(self, key, value):
        if isinstance(value, (np.ndarray, torch.Tensor, State)):
            self.set(key, value)
        else:
            self.__dict__[key] = value

    def get(self, k:str):
        return getattr(self, k)

    def has(self, k:str=None)->Union[bool, typing.Set[str]]:
        if k is not None:
            return k in self._schema_keys
        else:
            return self._schema_keys

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
        return self._length if self._length is not None else 0

    def __contains__(self, item):
        if isinstance(item, str):
            return self.has(item)
        else:
            raise Exception(f"Unsupported item type: {type(item)}")

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

            if all([isinstance(vse, (np.ndarray,)) for vse in vs]):
                v = np.concatenate(vs, 0)
            elif all([isinstance(vse, (torch.Tensor,)) for vse in vs]):
                v = torch.cat(vs, 0)
            elif all([isinstance(vse, (State,)) for vse in vs]):
                assert(all([type(vs[0]) == type(vse) for vse in vs]))
                v = type(vs[0]).merge(vs)
            else:
                raise Exception("Can not merge given states. Schema mismatch.")
            ret.set(**{k:v})

        return ret

    def __getitem__(self, item:Union[str, int, slice, List[int], np.ndarray, torch.Tensor]):    # slicing and getitem
        if isinstance(item, str):
            return self.get(item)
        if isinstance(item, int):
            item = slice(item, item+1)
        ret = type(self)()
        item_cpu = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
        for k in self._schema_keys:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                item = item_cpu
            vslice = v[item]
            ret.set(**{k:vslice})
        return ret

    def __setitem__(self, item:Union[str, int, slice, List[int], np.ndarray, torch.Tensor],
                    value:'State'):     # value should be of same type as self
        if isinstance(item, str):
            ret = self.get(item)
            self.set(item, value)
            return ret
        else:
            assert(type(self) == type(value))
            assert(self._schema_keys == value._schema_keys)
            if isinstance(item, int):
                item = slice(item, item+1)
            ret = self[item]
            assert(len(ret) == len(value))
            item_cpu = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
            for k in self._schema_keys:
                v = getattr(self, k)
                assert(True)        # TODO: assert type match
                if isinstance(v, np.ndarray):
                    item = item_cpu
                v[item] = value.get(k)
            return ret

    def to(self, device):
        for k in self._schema_keys:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
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

    def make_copy(self, ret=None, detach=None, deep=True):
        detach = deep if detach is None else detach
        ret = type(self)() if ret is None else ret
        for i in range(len(self._list)):
            v = self._list[i]
            if isinstance(v, torch.Tensor):
                ret.append(v.clone.detach() if detach else v.clone())
            elif isinstance(v, State):
                ret.append(v.make_copy(detach=detach, deep=deep))
            else:
                ret.append(deepcopy(v) if deep else copy(v))
            assert(len(ret._list) == i + 1)
        return ret

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
    def get_gold(self, i:int=None) -> torch.Tensor:
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


class BeamState(DecodableState):
    def __init__(self, states:List[DecodableState]=None, scores:torch.Tensor=None,
                 actionprobs:List[torch.Tensor]=None, predactions:torch.Tensor=None):
        if states is not None:
            bstates = ListState(*states)
            batsize, beamsize = len(bstates), len(states)
            bscores = torch.zeros(batsize, beamsize) if scores is None else scores
            kw = {"bstates": bstates, "bscores": bscores}
            if actionprobs is not None:
                actionprobs = ListState(*actionprobs)
                kw["actionprobs"] = actionprobs
            kw["predactions"] = predactions
            super(BeamState, self).__init__(**kw)
        else:
            super(BeamState, self).__init__()

    def start_decoding(self):
        raise Exception("states must have already been started decoding.")

    def is_terminated(self)-> List[List[bool]]:
        """
        Returns a list (over beam elements) of lists (over batch size) of booleans whether each elem in this state is terminated.
        """
        return [x.is_terminated() for x in self.bstates._list]

    def all_terminated(self):
        return all([all(ist) for ist in self.is_terminated()])

    def step(self, action:Union[torch.Tensor, List[Union[str, torch.Tensor]]]=None):
        raise Exception("this should not be used")


class BasicDecoderState(TrainableDecodableState):
    """
    Basic state object for seq2seq
    """
    endtoken = "@END@"
    def __init__(self,
                 inp_strings:List[str]=None,
                 gold_strings:List[str]=None,
                 inp_tensor:torch.Tensor=None,
                 gold_tensor:torch.Tensor=None,
                 inp_tokens:List[List[str]]=None,
                 gold_tokens:List[List[str]]=None,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 **kw):
        if inp_strings is None:
            super(BasicDecoderState, self).__init__(**kw)
        else:
            kw = kw.copy()
            kw.update({"inp_strings": np.asarray(inp_strings), "gold_strings": np.asarray(gold_strings)})
            super(BasicDecoderState, self).__init__(**kw)

            self.sentence_encoder = sentence_encoder
            self.query_encoder = query_encoder

            # self.set(followed_actions_str = np.asarray([None for _ in self.inp_strings]))
            # for i in range(len(self.followed_actions_str)):
            #     self.followed_actions_str[i] = []
            self.set(followed_actions = torch.zeros(len(inp_strings), 0, dtype=torch.long))
            self.set(_is_terminated = np.asarray([False for _ in self.inp_strings]))
            self.set(_timesteps = np.asarray([0 for _ in self.inp_strings]))

            if sentence_encoder is not None:
                x = [sentence_encoder.convert(x, return_what="tensor,tokens") for x in self.inp_strings]
                x = list(zip(*x))
                inp_tokens = np.asarray([None for _ in range(len(x[1]))], dtype=np.object)
                for i, inp_tokens_e in enumerate(x[1]):
                    inp_tokens[i] = tuple(inp_tokens_e)
                x = {"inp_tensor": batchstack(x[0]),
                     "inp_tokens": inp_tokens}
                self.set(**x)
            if self.gold_strings is not None:
                if query_encoder is not None:
                    x = [query_encoder.convert(x, return_what="tensor,tokens") for x in self.gold_strings]
                    x = list(zip(*x))
                    gold_tokens = np.asarray([None for _ in range(len(x[1]))])
                    for i, gold_tokens_e in enumerate(x[1]):
                        gold_tokens[i] = tuple(gold_tokens_e)
                    x = {"gold_tensor": batchstack(x[0]),
                         "gold_tokens": gold_tokens}
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

    def step(self, tokens:Union[torch.Tensor, np.ndarray, List[Union[str, np.ndarray, torch.Tensor]]]):
        qe = self.query_encoder
        assert(len(tokens) == len(self))
        tokens_np = np.zeros((len(tokens),), dtype="int64")

        if isinstance(tokens, list):
            types = [type(token) for token in tokens]
            if all([tokentype == torch.Tensor for tokentype in types]):
                tokens = torch.stack(tokens, 0)
            if all([tokentype == np.ndarray for tokentype in types]):
                tokens = np.stack(tokens, 0)
        if isinstance(tokens, list):
            for i, token in enumerate(tokens):
                if isinstance(token, str):
                    tokens_np[i] = qe.vocab[token]
                elif isinstance(token, np.ndarray):
                    assert(token.shape == (1,))
                    tokens_np[i] = token[0]
                elif isinstance(token, torch.Tensor):
                    assert(token.size() in [(1,), tuple()])
                    tokens_np[i] = token.detach().cpu().item()
        elif isinstance(tokens, torch.Tensor):
            tokens_np = tokens.detach().cpu().numpy()
        tokens_np = tokens_np * (~self._is_terminated).astype("int64")
        tokens_pt = torch.tensor(tokens_np).to(self.inp_tensor.device)
        # tokens_str = np.vectorize(lambda x: qe.vocab(x))(tokens_np)

        mask = torch.tensor(self._is_terminated).to(self.prev_actions.device)
        self.prev_actions = self.prev_actions * (mask).long() + tokens_pt * (~mask).long()
        self._timesteps[~self._is_terminated] = self._timesteps[~self._is_terminated] + 1

        # for i, token in enumerate(tokens_str):
        #     if not self.is_terminated()[i]:
        #         self.followed_actions_str[i] = self.followed_actions_str[i] + [token]

        self.followed_actions = torch.cat([self.followed_actions, tokens_pt[:, None]], 1)

        # self._is_terminated |= tokens_str == self.endtoken
        self._is_terminated = self._is_terminated | (tokens_np == self.query_encoder.vocab[self.endtoken])

    def get_gold(self, i:int=None):
        if i is None:
            return self.gold_tensor
        else:
            return self.gold_tensor[:, i]


class TreeDecoderState(TrainableDecodableState):
    endtoken = "@END@"
    reducetoken = ")"
    pushtoken = "("
    def __init__(self,
                 inp_strings:List[str]=None,
                 gold_trees:List[ActionTree]=None,
                 inp_tensor:torch.Tensor=None,
                 gold_tensor:torch.Tensor=None,
                 inp_tokens:List[List[str]]=None,
                 gold_tokens:List[List[str]]=None,
                 sentence_vocab:Vocab=None,
                 query_vocab:Vocab=None,
                 token_specs=None,
                 **kw):
        if inp_strings is None:
            super(TreeDecoderState, self).__init__(**kw)
        else:
            kw = kw.copy()

            gold_trees_ = np.array(range(len(gold_trees)), dtype="object")
            inp_tokens_ = np.array(range(len(inp_tokens)), dtype="object")
            gold_tokens_ = np.array(range(len(gold_tokens)), dtype="object")
            child_counts_ = np.array(range(len(gold_tokens)), dtype="object")   # stack of child counts
            followed_tokens_ = np.array(range(len(gold_tokens)), dtype="object")
            for i in range(len(gold_trees)):
                gold_trees_[i] = gold_trees[i]
                inp_tokens_[i] = inp_tokens[i]
                gold_tokens_[i] = gold_tokens[i]
                child_counts_[i] = []
                followed_tokens_[i] = []

            kw.update({"inp_strings": np.asarray(inp_strings),
                       "gold_trees": gold_trees_,
                       "inp_tokens": inp_tokens_,
                       "gold_tokens": gold_tokens_,
                       "inp_tensor": inp_tensor,
                       "gold_tensor": gold_tensor,
                       "child_counts": child_counts_,
                       "followed_tokens": followed_tokens_})
            super(TreeDecoderState, self).__init__(**kw)

            self.sentence_vocab = sentence_vocab
            self.query_vocab = query_vocab
            self.token_specs = token_specs
            self.token_groups = None
            if token_specs is not None:
                self.token_groups = {
                    "leaf": set(),
                    "nonleaf": set(),
                }
                for t, (min_c, max_c) in self.token_specs.items():
                    if min_c == 0 and max_c == 0:
                        self.token_groups["leaf"].add(t)
                    else:
                        self.token_groups["nonleaf"].add(t)

            self.set(followed_actions = torch.zeros(len(inp_strings), 0, dtype=torch.long))
            self.set(_is_terminated = np.asarray([False for _ in self.inp_strings]))
            self.set(_timesteps = np.asarray([0 for _ in self.inp_strings]))


    # State API override implementation
    def make_copy(self, ret=None, detach=None, deep=True):
        ret = super(TreeDecoderState, self).make_copy(ret=ret, detach=detach, deep=deep)
        ret.sentence_vocab = self.sentence_vocab
        ret.query_vocab = self.query_vocab
        ret.token_specs = self.token_specs
        ret.token_groups = self.token_groups
        return ret

    @classmethod
    def merge(cls, states:List['TreeDecoderState'], ret=None):
        assert(all([state.sentence_vocab == states[0].sentence_vocab and state.query_vocab == states[0].query_vocab for state in states]))
        ret = super(TreeDecoderState, cls).merge(states, ret=ret)
        ret.sentence_vocab = states[0].sentence_vocab
        ret.query_vocab = states[0].query_vocab
        ret.token_specs = states[0].token_specs
        ret.token_groups = states[0].token_groups
        return ret

    def __getitem__(self, item):
        ret = super(TreeDecoderState, self).__getitem__(item)
        ret.sentence_vocab = self.sentence_vocab
        ret.query_vocab = self.query_vocab
        ret.token_specs = self.token_specs
        ret.token_groups = self.token_groups
        return ret

    def __setitem__(self, key, value:'TreeDecoderState'):
        assert(value.sentence_vocab == self.sentence_vocab and value.query_vocab == self.query_vocab)
        ret = super(TreeDecoderState, self).__setitem__(key, value)
        return ret

    # DecodableState API implementation
    def is_terminated(self):
        return self._is_terminated

    def start_decoding(self):
        # initialize prev_action
        qe = self.query_vocab
        self.set(prev_actions=torch.tensor([qe[qe.starttoken] for _ in self.inp_strings],
                                                   device=self.inp_tensor.device, dtype=torch.long))

    def step(self, tokens:Union[torch.Tensor, np.ndarray, List[Union[str, np.ndarray, torch.Tensor]]]):
        qe = self.query_vocab
        assert(len(tokens) == len(self))
        tokens_np = np.zeros((len(tokens),), dtype="int64")

        if isinstance(tokens, list):
            types = [type(token) for token in tokens]
            if all([tokentype == torch.Tensor for tokentype in types]):
                tokens = torch.stack(tokens, 0)
            if all([tokentype == np.ndarray for tokentype in types]):
                tokens = np.stack(tokens, 0)
        if isinstance(tokens, list):
            for i, token in enumerate(tokens):
                if isinstance(token, str):
                    tokens_np[i] = qe.vocab[token]
                elif isinstance(token, np.ndarray):
                    assert(token.shape == (1,))
                    tokens_np[i] = token[0]
                elif isinstance(token, torch.Tensor):
                    assert(token.size() in [(1,), tuple()])
                    tokens_np[i] = token.detach().cpu().item()
        elif isinstance(tokens, torch.Tensor):
            tokens_np = tokens.detach().cpu().numpy()
        tokens_np = tokens_np * (~self._is_terminated).astype("int64")
        tokens_pt = torch.tensor(tokens_np).to(self.inp_tensor.device)
        # tokens_str = np.vectorize(lambda x: qe.vocab(x))(tokens_np)

        mask = torch.tensor(self._is_terminated).to(self.prev_actions.device)
        self.prev_actions = self.prev_actions * (mask).long() + tokens_pt * (~mask).long()
        self._timesteps[~self._is_terminated] = self._timesteps[~self._is_terminated] + 1

        # for i, token in enumerate(tokens_str):
        #     if not self.is_terminated()[i]:
        #         self.followed_actions_str[i] = self.followed_actions_str[i] + [token]

        self.followed_actions = torch.cat([self.followed_actions, tokens_pt[:, None]], 1)
        token_strs = [self.query_vocab(tokenid) for tokenid in tokens_np]
        for followed_tokens_i, token_str in zip(self.followed_tokens, token_strs):
            followed_tokens_i.append(token_str)

        # self._is_terminated |= tokens_str == self.endtoken
        self._is_terminated = self._is_terminated | (tokens_np == self.query_vocab[self.endtoken])
        #
        # if self.token_specs is not None:
        #     for token_str, child_count in zip(token_strs, self.child_counts):
        #         token_spec = self.token_specs[token_str]
        #         if token_spec == "PUSH":
        #             cc = child_count[-1]
        #             child_count[-1] = (cc[0] - 1, cc[1])
        #             child_count.append(None)
        #         elif token_spec == "RED":
        #             assert(child_count[-1] in ((0, "="), (0, "+")))
        #             child_count.pop(-1)
        #         else:
        #             if token_spec not in ((0, "="), (0, "+")):
        #                 assert(child_count[-1] is None)   # must have been preceded by a PUSH
        #                 child_count[-1] = token_spec

    def get_out_mask(self, device=torch.device("cpu")):
        rets = []
        for history in self.followed_tokens:
            if len(history) == 0:
                ret = {"("}
            elif history[-1] in ("@END@", "@PAD@"):
                ret = {"@PAD@"}
            else:
                child_counts = [0]
                ancestors = []
                for t in history:
                    if t == self.pushtoken:
                        child_counts.append(-1)
                        ancestors.append(None)
                    elif t == self.reducetoken:
                        child_counts.pop(-1)
                        ancestors.pop(-1)
                        child_counts[-1] += 1
                    else:
                        child_counts[-1] += 1
                        if len(ancestors) > 0 and ancestors[-1] is None:
                            ancestors[-1] = t
                ret = {self.reducetoken, self.pushtoken}
                if len(ancestors) == 0:
                    ret = {"@END@"}
                elif child_counts[-1] == -1:
                    ret |= self.token_groups["nonleaf"]
                    ret -= {self.pushtoken}
                else:
                    ret |= self.token_groups["leaf"]
                if len(ancestors) > 0 and ancestors[-1] in self.token_specs:
                    parent_min_children, parent_max_children = self.token_specs[ancestors[-1]]
                    if child_counts[-1] == parent_max_children:
                        ret = {self.reducetoken}
                    elif child_counts[-1] < parent_min_children:
                        ret -= {self.reducetoken}
            ret_ = torch.zeros(self.query_vocab.number_of_ids(), device=device)
            for ret_i in ret:
                ret_[self.query_vocab[ret_i]] = 1
            rets.append(ret_)
        return torch.stack(rets, 0)

    def get_gold(self, i:int=None):
        if i is None:
            return self.gold_tensor
        else:
            return self.gold_tensor[:, i]


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