from abc import ABC, abstractmethod
from typing import Union, Callable, List, Dict
import numpy as np

import torch

from parseq.grammar import FuncGrammar


class _Vocab(object):
    pass


class Vocab(_Vocab):
    padtoken = "@PAD@"
    unktoken = "@UNK@"
    starttoken = "@START@"
    endtoken = "@END@"
    def __init__(self, padid:int=0, unkid:int=1, startid:int=2, endid:int=3, **kw):
        self.D = {self.padtoken: padid, self.unktoken: unkid}
        self.D[self.starttoken] = startid
        self.D[self.endtoken] = endid
        self.counts = {k: np.infty for k in self.D.keys()}
        self.rare_tokens = set()
        self.rare_ids = set()
        self.RD = {v: k for k, v in self.D.items()}
        self.growing = True

    def set_dict(self, D):
        self.D = D
        self.RD = {v: k for k, v in self.D.items()}

    def nextid(self):
        return max(self.D.values()) + 1

    def finalize(self, min_freq:int=0, top_k:int=np.infty, keep_tokens=None):
        self.growing = False
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)

        if min_freq == 0 and top_k > len(sorted_counts):
            self.rare_tokens = set()
        else:
            if top_k < len(sorted_counts) and sorted_counts[top_k][1] > min_freq:
                i = top_k
            else:
                if top_k < len(sorted_counts):
                    sorted_counts = sorted_counts[:top_k]
                # binary search for min_freq position
                i = 0
                divider = 2
                where = +1
                while True:
                    i += where * len(sorted_counts) // divider
                    if (i == len(sorted_counts)) or (
                            sorted_counts[i][1] <= min_freq - 1 and sorted_counts[i - 1][1] >= min_freq):
                        break  # found
                    elif sorted_counts[i][1] < min_freq:  # go up
                        where = -1
                    elif sorted_counts[i][1] >= min_freq:  # go down
                        where = +1
                    divider *= 2
                    divider = min(divider, len(sorted_counts))
            if keep_tokens is not None:
                if keep_tokens in ("all", "ALL"):
                    self.rare_tokens = set([t[0] for t in sorted_counts[i:]])
                else:
                    sorted_counts = [sc for j, sc in enumerate(sorted_counts) if sc[0] in keep_tokens or j < i]
                    self.rare_tokens = set([t[0] for t in sorted_counts[i:]]) & keep_tokens
            else:
                sorted_counts = sorted_counts[:i]

        nextid = max(self.D.values()) + 1
        for token, cnt in sorted_counts:
            if token not in self.D:
                self.D[token] = nextid
                nextid += 1

        self.RD = {v: k for k, v in self.D.items()}
        if keep_tokens is not None:
            self.rare_ids = set([self[rare_token] for rare_token in self.rare_tokens])

    def add_token(self, token, seen:Union[int,bool]=True):
        assert(self.growing)
        if token not in self.counts:
            self.counts[token] = 0
        if seen > 0:
            self.counts[token] += float(seen)

    def __getitem__(self, item:str) -> int:
        if item not in self.D:
            assert(self.unktoken in self.D)
            item = self.unktoken
        id = self.D[item]
        return id

    def __call__(self, item:int) -> str:
        return self.RD[item]

    def number_of_ids(self, exclude_rare=False):
        if not exclude_rare:
            return max(self.D.values()) + 1
        else:
            return max(set(self.D.values()) - self.rare_ids) + 1

    def reverse(self):
        return {v: k for k, v in self.D.items()}

    def __iter__(self):
        return iter([(k, v) for k, v in self.D.items()])

    def __contains__(self, item:Union[str,int]):
        if isinstance(item, str):
            return item in self.D
        if isinstance(item, int):
            return item in self.RD
        else:
            raise Exception("illegal argument")

    def tostr(self, x:Union[np.ndarray, torch.Tensor], return_tokens=False):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = list(np.vectorize(lambda e: self(e))(x))
        x = [e for e in x if e != self.padtoken]
        ret = []
        for xe in x:
            if len(ret) > 0 and ret[-1] == self.endtoken:
                break
            ret.append(xe)
        if return_tokens:
            return ret
        else:
            return " ".join(ret)


class FixedVocab(Vocab):
    def __init__(self, padid:int=0, unkid:int=1, vocab:Dict=None, **kw):
        super(FixedVocab, self).__init__(padid, unkid, **kw)
        self.D = vocab
        self.growing = False

    def add_token(self, token, seen=True):
        print("Warning: trying to add token to fixed vocab")
        pass

    def do_rare(self, min_freq=0, top_k=np.infty):
        print("Warning: trying to do rare on fixed vocab")
        pass


def try_vocab():
    vocab = Vocab()
    tokens = "a b c d e a b c d a b c a b a a a a b e d g m o i p p x x i i b b ai ai bi bi bb bb abc abg abh abf".split()
    for t in tokens:
        vocab.add_token(t)
    vocab.do_rare(min_freq=2, top_k=15)
    print(vocab.rare_tokens)
    print(vocab.rare_ids)


class VocabBuilder(ABC):
    @abstractmethod
    def inc_build_vocab(self, x:str, seen:bool=True):
        raise NotImplemented()

    @abstractmethod
    def finalize_vocab(self, min_freq:int=0, top_k:int=np.infty):
        raise NotImplemented()

    @abstractmethod
    def vocabs_finalized(self):
        raise NotImplemented()

    
class SequenceEncoder(VocabBuilder):
    endtoken = "@END@"
    def __init__(self, tokenizer:Callable[[str], List[str]], vocab:Vocab=None, add_end_token=False, **kw):
        super(SequenceEncoder, self).__init__(**kw)
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab is not None else Vocab()
        self.vocab_final = False
        self.add_end_token = add_end_token
        
    def inc_build_vocab(self, x:str, seen:bool=True):
        if not self.vocab_final:
            tokens = self.tokenizer(x)
            if self.add_end_token:
                tokens.append(self.endtoken)
            for token in tokens:
                self.vocab.add_token(token, seen=seen)
            return tokens
        else:
            return []
    
    def finalize_vocab(self, min_freq:int=0, top_k:int=np.infty, keep_tokens=None):
        self.vocab_final = True
        self.vocab.finalize(min_freq=min_freq, top_k=top_k, keep_tokens=keep_tokens)
        
    def vocabs_finalized(self):
        return self.vocab_final
    
    def convert(self, x:Union[str, List[str]], return_what="tensor"):     # "tensor", "ids", "tokens" or comma-separated combo of all
        rets = [r.strip() for r in return_what.split(",")]
        if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], str)):
            tokens = x
        else:
            tokens = self.tokenizer(x)
        if self.add_end_token and tokens[-1] != self.endtoken:
            tokens.append(self.endtoken)
        ret = {"tokens": tokens}
        if "ids" in rets or "tensor" in rets:
            ret["ids"] = [self.vocab[token] for token in tokens]
        if "tensor" in rets:
            ret["tensor"] = torch.tensor(ret["ids"], dtype=torch.long)
        ret = [ret[r] for r in rets]
        return ret
    
    
class FuncQueryEncoder(VocabBuilder):
    def __init__(self, grammar:FuncGrammar=None, vocab_tokens:Vocab=None, vocab_actions:Vocab=None,
                 sentence_encoder:SequenceEncoder=None, format:str= "prolog", **kw):
        super(FuncQueryEncoder, self).__init__(**kw)
        self.vocab_final = False
        self.vocab_tokens = vocab_tokens if vocab_tokens is not None else Vocab()
        self.vocab_actions = vocab_actions if vocab_actions is not None else Vocab()

        self.none_action = "[NONE]"
        self.start_action = "@START@"
        self.vocab_actions.add_token(self.none_action, seen=np.infty)
        self.vocab_actions.add_token(self.start_action, seen=np.infty)

        self.grammar = grammar
        self.sentence_encoder = sentence_encoder

        self.format = format

        # prebuild valid action masks
        self._valid_action_mask_by_type = {}

    def prebuild_valid_action_masks(self):
        for typestr, rules in self.grammar.rules_by_type.items():
            action_mask = torch.zeros(self.vocab_actions.number_of_ids(), dtype=torch.uint8)
            for rule in rules:
                action_mask[self.vocab_actions[rule]] = 1
            if typestr[-1] in "*+":
                rules = self.grammar.rules_by_type[typestr[:-1]]
                for rule in rules:
                    action_mask[self.vocab_actions[rule]] = 1
            self._valid_action_mask_by_type[typestr] = action_mask
        # self._valid_action_mask_by_type["_@unk@_"] = torch.tensor(self.vocab_actions.number_of_ids(), dtype=torch.uint8)

    def get_action_mask_for(self, typ:str):
        if typ in self._valid_action_mask_by_type:
            return self._valid_action_mask_by_type[typ]
        else:
            ret = torch.zeros(self.vocab_actions.number_of_ids(), dtype=torch.uint8)
            ret[self.vocab_actions[self.none_action]] = 1
            return ret

    def vocabs_finalized(self):
        return self.vocab_final

    def inc_build_vocab(self, x:str, seen:bool=True):
        if not self.vocab_final:
            actions = self.grammar.actions_for(x, format=self.format)
            self._add_to_vocabs(actions, seen=seen)

    def _add_to_vocabs(self, actions:List[str], seen:bool=True):
        for action in actions:
            self.vocab_actions.add_token(action, seen=seen)
            head, body = action.split(" -> ")
            self.vocab_tokens.add_token(head, seen=seen)
            body = body.split(" ")
            body = [x for x in body if x not in ["::", "--"]]
            for x in body:
                self.vocab_tokens.add_token(x, seen=seen)

    def finalize_vocab(self, min_freq:int=0, top_k:int=np.infty):
        for out_type, rules in self.grammar.rules_by_type.items():
            self._add_to_vocabs(rules, seen=False)

        self.vocab_tokens.stopgrowth()
        self.vocab_actions.stopgrowth()

        self.vocab_final = True
        self.vocab_tokens.do_rare(min_freq=min_freq, top_k=top_k)
        self.vocab_actions.do_rare(min_freq=min_freq, top_k=top_k)

        self.prebuild_valid_action_masks()

    def convert(self, x:str, return_what="tensor"):     # "tensor", "ids", "actions", "tree"
        rets = [r.strip() for r in return_what.split(",")]
        ret = {}
        actions = self.grammar.actions_for(x, format=self.format)
        ret["actions"] = actions
        tree = self.grammar.actions_to_tree(actions)
        ret["tree"] = tree
        actionids = [self.vocab_actions[action] for action in actions]
        ret["ids"] = actionids
        tensor = torch.tensor(actionids, dtype=torch.long)
        ret["tensor"] = tensor
        ret = [ret[r] for r in rets]
        return ret


def try_func_query_encoder():
    fg = FuncGrammar("<R>")
    fg.add_rule("<R> -> wife :: <R>")
    fg.add_rule("<R> -> president :: <R>")
    fg.add_rule("<R> -> BO")
    fg.add_rule("<R> -> stateid :: <W>*")
    fg.add_rule("<W>* -> @W:END@")
    fg.add_rule("<W>* -> 'bo' -- <W>*")
    fg.add_rule("<W>* -> 'us' -- <W>*")
    fg.add_rule("<W>* -> 'a' -- <W>*")
    fg.add_rule("<R> -> countryid :: <W>*")
    fg.add_rule("<R> -> country :: <R>")
    fg.add_rule("<R> -> US")

    query = "wife(president(countryid('bo', 'us', 'a', @W:END@)))"
    actions = fg.actions_for(query, format="pred")
    print(actions)
    tree = fg.actions_to_tree(actions)
    print(tree)

    print(fg.rules_by_type)

    gb = FuncQueryEncoder(grammar=fg, format="pred")
    gb.finalize_vocab()

    g = gb.convert(query, return_what="tensor,tree")
    print(g[0])
    print(g[1])
    print(g[1]._action)


if __name__ == '__main__':
    try_vocab()
    # try_func_query_encoder()