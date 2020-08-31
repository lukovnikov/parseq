import json
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Tuple, Iterable, List, Dict

import numpy as np
import torch
from nltk import Tree
from torch.utils.data import DataLoader

import qelos as q
from parseq.datasets import OvernightDatasetLoader, autocollate
from parseq.eval import TreeAccuracy, make_array_of_metrics
from parseq.grammar import tree_to_lisp_tokens, tree_to_prolog, are_equal_trees, tree_size
from parseq.transformer import TransformerConfig, TransformerModel, TransformerStack
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


class ATree(Tree):
    def __init__(self, label, children:List['ATree']=None,
                 is_open:bool=True, gold_actions=None, **kw):
        if children is None:
            children = []
        super(ATree, self).__init__(label, children)
        self.is_open = is_open
        self.gold_actions = gold_actions if gold_actions is not None else set()

    def __deepcopy__(self, memo):
        children = [deepcopy(child) for child in self]
        ret = type(self)(self._label, children,
                         is_open=self.is_open,
                         gold_actions=deepcopy(self.gold_actions))
        return ret


def add_parentheses(x:ATree):
    """ adds parentheses into the tree as actual children """
    ret = deepcopy(x)
    queue = [ret]
    while len(queue) > 0:
        first = queue.pop(0)
        if len(first) > 0:
            queue += first[:]
            first.insert(0, ATree("(", []))
            first.append(ATree(")", []))
    return ret


def all_terminated(x:ATree):
    return not x.is_open and all([all_terminated(xe) for xe in x])


def add_descendants_ancestors(x:ATree, ancestors:Iterable[ATree]=tuple()):
    x._ancestors = ancestors
    x._descendants = []
    if x.label() not in ("(", ")"):
        for ancestor in ancestors:
            ancestor._descendants.append(x)
        for i, xe in enumerate(x):
            add_descendants_ancestors(xe, ancestors+(x,))
            xe.parent = x
    return x


def child_number_of(x:ATree):
    p = x.parent
    for i in range(len(p)):
        if p[i] is x:
            return i
    return None     # not a child


def assign_gold_actions(x:ATree, mode="default"):
    """
    :param x:
    :param mode:    "default" (all) or "ltr" (only first one)
    :return:
    """
    """ assigns actions that can be taken at every node of the given tree """
    for xe in x:
        assign_gold_actions(xe, mode=mode)
    if not x.is_open:
        x.gold_actions = []
    else:
        if x.label() == ")" or x.label() == "(":
            x.is_open = False
            x.gold_actions = []
        elif x.label() == "@SLOT@":
            if len(x.parent) == 1:
                raise Exception()
            # get this slots's siblings
            x.gold_actions = []
            xpos = child_number_of(x)
            if xpos == 0:
                leftsibling = None
                leftsibling_nr = None
            else:
                leftsibling = x.parent[xpos - 1]
                leftsibling_nr = child_number_of(leftsibling.align)
            if xpos == len(x.parent) - 1:
                rightsibling = None
                rightsibling_nr = None
            else:
                rightsibling = x.parent[xpos + 1]
                rightsibling_nr = child_number_of(rightsibling.align)

            if leftsibling is None and rightsibling is None:
                # slot is only child, can use any descendant
                x.gold_actions = x.parent.align.descendants
                if mode == "ltr" and len(x.gold_actions) > 0:
                    x.gold_actions = [x.gold_actions[0]]
                assert(False)   # should not happen if deletion actions are not used
            else:
                p = leftsibling.align.parent if leftsibling is not None else rightsibling.align.parent
                slicefrom = leftsibling_nr + 1 if leftsibling_nr is not None else None
                slicer = slice(slicefrom, rightsibling_nr)
                x.gold_actions = p[slicer]
                if mode == "ltr" and len(x.gold_actions) > 0:
                    x.gold_actions = [x.gold_actions[0]]
            if len(x.gold_actions) == 0:
                x.gold_actions = ["@CLOSE@"]
        else:       # not a sibling slot ("@SLOT@"), not a "(" or ")"
            x.gold_actions = []
            if len(x) == 0:
                x.gold_actions = list(x.align._descendants)
                if mode == "ltr" and len(x.gold_actions) > 0:
                    x.gold_actions = [x.gold_actions[0]]
            else:
                realchildren = [xe for xe in x if xe.label() != "@SLOT@"]
                childancestors = realchildren[0].align._ancestors[::-1]
                for child in realchildren:
                    assert(childancestors == child.align._ancestors[::-1])
                for ancestor in childancestors:
                    if ancestor is x.align:
                        break
                    else:
                        x.gold_actions.append(ancestor)
                if mode == "ltr" and len(x.gold_actions) > 0:
                    x.gold_actions = [x.gold_actions[0]]
        if len(x.gold_actions) == 0 and x.is_open:
            x.gold_actions = ["@CLOSE@"]

    if len(x.gold_actions) > 0:
        # x._chosen_action = x.gold_actions[0]
        x._chosen_action = random.choice(x.gold_actions)
    else:
        x._chosen_action = None
    return x


def adjust_gold(x:ATree, mode:str="single"):
    if mode == "ltr":
        queue = [x]
        nextnode = None
        while len(queue) > 0:
            first = queue.pop(0)
            if nextnode is not None:
                first.gold_actions = []
            else:
                if first.is_open:
                    if len(
                            first) == 0 and first.label() != "@SLOT@":  # if leaf but not closed --> this is the next real actions
                        nextnode = first
                else:
                    assert(first.gold_actions == [])
                    first.gold_actions = []

            queue = first[:] + queue
    return x


def mark_for_execution(x:ATree, mode:str="single"):     # "all", "parallel:100%", "single", "ltr"
    """ Marks only a selection of all nodes in the given tree for execution
        by setting ._chosen_action of other nodes to None """
    nodes_with_actions = []
    queue = [x]
    while len(queue) > 0:
        head = queue.pop(0)
        if hasattr(head, "_chosen_action") and head._chosen_action is not None:
            nodes_with_actions.append(head)
        queue += head[:]

    if mode == "single":    # leave only one node for execution
        selected = random.choice(nodes_with_actions)
        for node in nodes_with_actions:
            if node is not selected:
                node._chosen_action = None
    elif mode == "ltr":
        queue = [x]
        nextnode = None
        while len(queue) > 0:
            first = queue.pop(0)
            if nextnode is not None:
                first._chosen_action = None
            else:
                if first.is_open:
                    if len(first) == 0 and first.label() != "@SLOT@":     # if leaf but not closed --> this is the next real actions
                        nextnode = first
                else:
                    first._chosen_action = None

            queue = first[:] + queue
    elif mode == "full" or mode == "all" or mode == "parallel:100%":
        pass
    return x


def execute_chosen_actions(x:ATree, _budget=[np.infty], mode="full"):
    if x._chosen_action is None or not x.is_open:
        iterr = list(x)
        for xe in iterr:
            execute_chosen_actions(xe, _budget=_budget, mode=mode)
        return x
    if x.label() == "(":    # insert a parent before current parent
        pass
    elif x.label() == ")":
        pass
    elif x.label() == "@SLOT@":
        if x._chosen_action == "@CLOSE@":
            del x.parent[child_number_of(x)]
            # if parentheses became empty, remove
            _budget[0] += 1
            if len(x.parent) == 2 and x.parent[0].label() == "(" and x.parent[1].label() == ")":
                x.parent[:] = []
        else:
            if _budget[0] <= 0:
                return x
            if isinstance(x._chosen_action, Tree):
                x.set_label(x._chosen_action.label())
            else:
                x.set_label(x._chosen_action)

            if isinstance(x._chosen_action, Tree):
                x.align = x._chosen_action
            x.is_open = True

            leftslot = ATree("@SLOT@", [], is_open=True)
            leftslot.parent = x.parent
            rightslot = ATree("@SLOT@", [], is_open=True)
            rightslot.parent = x.parent

            if mode != "ltr":
                x.parent.insert(child_number_of(x), leftslot)
                _budget[0] -= 1

            x.parent.insert(child_number_of(x)+1, rightslot)
            _budget[0] -= 1

    else:
        iterr = list(x)
        for xe in iterr:
            execute_chosen_actions(xe, _budget=_budget, mode=mode)
        if _budget[0] <= 0:
            return x
        if x._chosen_action == "@CLOSE@":
            x.is_open = False       # this node can't generate children anymore
        else:
            # X(A, B, C) -> X _( _@SLOT _Y (A, B, C) [_@SLOT] _)
            # add child, with "(" and ")" and "@SLOT@" nodes

            if isinstance(x._chosen_action, Tree):
                newnode = ATree(x._chosen_action.label(), [])
            else:
                newnode = ATree(x._chosen_action, [])

            newnode.is_open = True
            if mode == "ltr":
                x.is_open = False

            if isinstance(x._chosen_action, Tree):
                newnode.align = x._chosen_action
            newnode.parent = x

            newnode[:] = x[:]
            for xe in newnode:
                xe.parent = newnode

            leftslot = ATree("@SLOT@", [], is_open=True)
            leftslot.parent = newnode.parent

            rightslot = ATree("@SLOT@", [], is_open=True)
            rightslot.parent = newnode.parent

            if mode != "ltr":
                x[:] = [leftslot, newnode, rightslot]
                _budget[0] -= 3
            else:
                x[:] = [newnode, rightslot]
                _budget[0] -= 2
    return x


def uncomplete_tree_parallel(x:ATree, mode="full"):
    """ Input is tuple (nl, fl, split)
        Output is a randomly uncompleted tree,
            every node annotated whether it's terminated and what actions are good at that node
    """
    fl = x

    fl.parent = None
    add_descendants_ancestors(fl)

    y = ATree("@START@", [])
    y.align = fl
    y.is_open = True

    i = 0
    y = assign_gold_actions(y, mode=mode)
    choices = [deepcopy(y)]         # !! can't cache because different choices !
    while not all_terminated(y):
        y = mark_for_execution(y, mode=mode)
        y = execute_chosen_actions(y, mode=mode)
        y = assign_gold_actions(y, mode=mode)
        y = adjust_gold(y, mode=mode)
        choices.append(deepcopy(y))
        i += 1

    ret = random.choice(choices[:-1])
    return ret


def extract_info(x:ATree, onlytokens=False, nogold=False):
    """ Receives an annotated tree (with parentheses) and returns:
            - a sequence of tokens derived from that tree
            - a sequence of whether the token is terminated
            - a sequence of sets of gold labels
    """
    tokens, openmask, golds = [], [], []
    queue = ["(", x, ")"]
    while len(queue) > 0:
        first = queue.pop(0)
        if isinstance(first, str):
            tokens.append(first)
            if not onlytokens:
                openmask.append(False)
            if not onlytokens and not nogold:
                gold = set()
                golds.append(gold)
        else:
            tokens.append(first.label())
            if not onlytokens:
                openmask.append(first.is_open if hasattr(first, "is_open") else False)
            if not onlytokens and not nogold:
                gold = set()
                if hasattr(first, "gold_actions"):
                    for golde in first.gold_actions:
                        gold.add(golde.label() if isinstance(golde, Tree) else golde)
                golds.append(gold)
            queueprefix = []
            for fc in first:
                if len(fc) == 0:
                    queueprefix.append(fc)
                else:
                    queueprefix += ["(", fc, ")"]

            queue = queueprefix + queue

    if onlytokens:
        return tokens
    elif nogold:
        return tokens, openmask
    else:
        return tokens, openmask, golds


def load_ds(domain="restaurants", mode="full", nl_mode="bert-base-uncased", trainonvalid=False):
    """
    Creates a dataset of examples which have
    * NL question and tensor
    * original FL tree
    * reduced FL tree with slots (this is randomly generated)
    * tensor corresponding to reduced FL tree with slots
    * mask specifying which elements in reduced FL tree are terminated
    * 2D gold that specifies whether a token/action is in gold for every position (compatibility with MML!)
    """
    orderless = {"op:and", "SW:concat"}     # only use in eval!!

    ds = OvernightDatasetLoader().load(domain=domain, trainonvalid=trainonvalid)
    ds = ds.map(lambda x: (x[0], ATree("@START@", [x[1]]), x[2]))

    vocab = Vocab(padid=0, startid=2, endid=3, unkid=1)
    vocab.add_token("@START@", seen=np.infty)
    vocab.add_token("@CLOSE@", seen=np.infty)        # only here for the action of closing an open position, will not be seen at input
    vocab.add_token("@OPEN@", seen=np.infty)         # only here for the action of opening a closed position, will not be seen at input
    vocab.add_token("@REMOVE@", seen=np.infty)       # only here for deletion operations, won't be seen at input
    vocab.add_token("@REMOVESUBTREE@", seen=np.infty)       # only here for deletion operations, won't be seen at input
    vocab.add_token("@SLOT@", seen=np.infty)         # will be seen at input, can't be produced!

    nl_tokenizer = BertTokenizer.from_pretrained(nl_mode)
    # for tok, idd in nl_tokenizer.vocab.items():
    #     vocab.add_token(tok, seen=np.infty)          # all wordpieces are added for possible later generation

    tds, vds, xds = ds[lambda x: x[2] == "train"], \
                    ds[lambda x: x[2] == "valid"], \
                    ds[lambda x: x[2] == "test"]

    seqenc = SequenceEncoder(vocab=vocab, tokenizer=lambda x: extract_info(x, onlytokens=True),
                             add_start_token=False, add_end_token=False)
    for example in tds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=True)
    for example in vds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=False)
    for example in xds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=False)
    seqenc.finalize_vocab(min_freq=0)

    def mapper(x):
        nl = x[0]
        fl = uncomplete_tree_parallel(x[1], mode=mode)
        fltoks, openmask, gold_sets = extract_info(fl)

        seq = seqenc.convert(fltoks, return_what="tensor")
        golds = torch.zeros(seq.size(0), seqenc.vocab.number_of_ids())
        for i, gold in enumerate(gold_sets):
            for golde in gold:
                golds[i, seqenc.vocab[golde]] = 1
        ret = (nl_tokenizer.encode(nl, return_tensors="pt")[0],
               seq,
               torch.tensor(openmask),
               golds,)
               # seqenc.convert(x[4], return_what="tensor"))
               # x[4],
               # x[0], x[1], x[3])
        return ret

    def mapper2(x):
        nl = x[0]
        fl = x[1]
        fltoks = extract_info(fl, onlytokens=True)
        seq = seqenc.convert(fltoks, return_what="tensor")
        ret = (nl_tokenizer.encode(nl, return_tensors="pt")[0],
               seq)
        return ret

    _tds = tds.map(mapper)
    _vds = vds.map(mapper)
    _xds = xds.map(mapper)

    tds_seq = tds.map(mapper2)
    vds_seq = vds.map(mapper2)
    xds_seq = xds.map(mapper2)
    return _tds, _vds, _xds, tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc, orderless


def collate_fn(x, pad_value=0):
    y = list(zip(*x))
    assert(len(y) == 4)

    y[0] = torch.stack(q.pad_tensors(y[0], 0, pad_value), 0)
    y[1] = torch.stack(q.pad_tensors(y[1], 0, pad_value), 0)
    y[2] = torch.stack(q.pad_tensors(y[2], 0, False), 0)
    y[3] = torch.stack(q.pad_tensors(y[3], 0, pad_value), 0)

    return y


class TreeInsertionTagger(ABC, torch.nn.Module):
    """ A tree insertion tagging model takes a sequence representing a tree
        and produces distributions over tree modification actions for every (non-terminated) token.
    """
    @abstractmethod
    def forward(self, tokens:torch.Tensor, openmask:torch.Tensor=None, **kw):
        """
        :param tokens:      (batsize, seqlen)
        :param openmask:    (batsize,) - True if token is terminated
        :return:
        """
        pass

    @abstractmethod
    def get_init_state(self, **kw)-> Tuple[ATree, Dict]:
        """ Run encoding on context etc.
        Return starting trees and a dictionary of context variables that that will be used as the kwargs in .forward() of the tagger during decoding."""
        pass


class MultiCELoss(torch.nn.Module):
    def __init__(self, seqaggmode:str="mean", mode:str="logits", **kw):
        super(MultiCELoss, self).__init__(**kw)
        self.mode = mode
        self.seqaggmode = seqaggmode

    def forward(self, probs, golds, mask=None):
        """
        :param probs:       (batsize, seqlen, vocsize) - distributions over tokens
        :param golds:       (batsize, seqlen, vocsize) - 1 if token is gold, 0 otherwise
        :param mask:        (batsize, seqlen)
        :return:
        """
        golds = golds.float()
        if self.mode == "logits":
            probs = torch.softmax(probs, -1)
        elif self.mode == "logprobs":
            probs = torch.exp(probs)

        zeromask = (golds.sum(-1) != 0).float()
        extragolds = torch.zeros_like(golds)
        extragolds[:, :, 0] = 1
        golds = golds * (zeromask.unsqueeze(-1)) + extragolds * (1 - zeromask.unsqueeze(-1))

        selectedprobs = (golds * probs).sum(-1)
        selectedloss = -torch.log(selectedprobs.clamp_min(1e-4))
        selectedloss = selectedloss * zeromask * mask.float()

        if self.seqaggmode == "mean":
            a = selectedloss.sum(-1)
            b = (zeromask * mask.float()).sum(-1).clamp_min(1e-6)
            loss = a/b
        else:
            raise Exception("unknown seqaggmode")
        return loss


class Recall(torch.nn.Module):
    def forward(self, probs, golds, mask=None):
        """
        :param probs:       (batsize, seqlen, vocsize) - distributions over tokens
        :param golds:       (batsize, seqlen, vocsize) - 1 if token is gold, 0 otherwise
        :param mask:        (batsize, seqlen)
        :return:
        """
        golds = golds.float()

        zeromask = (golds.sum(-1) != 0).float()
        extragolds = torch.zeros_like(golds)
        extragolds[:, :, 0] = 1     # make padding available as gold if gold is all-zero
        golds = golds * (zeromask.unsqueeze(-1)) + extragolds * (1 - zeromask.unsqueeze(-1))

        _, best = probs.max(-1)
        bestingold = golds.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        # (batsize, seqlen)

        _mask = zeromask * mask.float()

        seqrecall = ((bestingold >= 1) | ~(_mask >= 1)).all(-1).float()
        anyrecall = ((bestingold >= 1) & (_mask >= 1)).any(-1).float()

        elemrecall = bestingold.sum(-1)
        b = _mask.sum(-1).clamp_min(1e-6)
        elemrecall = elemrecall / b
        return elemrecall, seqrecall, anyrecall


def test_losses():
    probs = torch.nn.Parameter(torch.randn(5, 3, 10))
    golds = torch.zeros(5, 3, 10)
    golds[0, :, 1] = 1
    golds[1, :2, 2] = 1
    golds[2, :1, 2] = 1
    golds[3, 0, 2] = 1
    golds[3, 1, 3] = 1
    golds[3, 2, 4] = 1
    golds[4, :, 4] = 1
    mask = torch.ones(5, 3)
    mask[0, :] = 0
    mask[4, 1:] = 0

    m = MultiCELoss()
    l = m(probs, golds, mask=mask)

    print(l)
    l.backward()
    print(probs.grad)

    m = Recall()
    l = m(probs, golds, mask=mask)
    print(l)


def build_atree(x:Iterable[str], open:Iterable[bool]=None, chosen_actions:Iterable[str]=None):
    open = [False for _ in x] if open is None else open
    chosen_actions = [None for _ in x] if chosen_actions is None else chosen_actions
    nodes = []
    for xe, opene, chosen_action in zip(x, open, chosen_actions):
        if xe == "(" or xe == ")":
            nodes.append(xe)
            assert(opene == False)
            # assert(chosen_action is None)
        else:
            a = ATree(xe, [], is_open=opene)
            a._chosen_action = chosen_action
            nodes.append(a)

    buffer = list(nodes)
    stack = []
    keepgoing = len(buffer) > 0
    while keepgoing:
        if len(stack) > 0 and stack[-1] == ")":
                stack.pop(-1)
                acc = []
                while len(acc) == 0 or not stack[-1] == "(":
                    acc.append(stack.pop(-1))
                stack.pop(-1)
                node = acc.pop(-1)
                node[:] = reversed(acc)
                for nodechild in node:
                    nodechild.parent = node
                stack.append(node)
        else:
            if len(buffer) == 0:
                keepgoing = False
            else:
                stack.append(buffer.pop(0))
    assert(len(stack) == 1)
    return stack[0]


def tensors_to_tree(x, openmask=None, actions=None, D:Vocab=None):
    # x: 1D int tensor
    x = list(x.detach().cpu().numpy())
    x = [D(xe) for xe in x]
    x = [xe for xe in x if xe != D.padtoken]

    if openmask is not None:
        openmask = list(openmask.detach().cpu().numpy())
    if actions is not None:
        actions = list(actions.detach().cpu().numpy())
        actions = [D(xe) for xe in actions]
        actions = [xe for xe in actions if xe != D.padtoken]

    tree = build_atree(x, open=openmask, chosen_actions=actions)
    return tree


def test_tensors_to_tree():
    tds, vds, xds, tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds("restaurants")
    tdl = DataLoader(tds, batch_size=5, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(tdl))

    trees = [tensors_to_tree(seqe, openmask=openmaske, actions=beste, D=flenc.vocab)
             for seqe, openmaske, beste in
             zip(list(batch[1]), list(batch[2]), list(batch[3].max(-1)[1]))]


class TreeInsertionTaggerModel(torch.nn.Module):
    """ A tree insertion model used for training and inference.
        Receives both input to the tagging model as well as gold.
        Computes loss during training.
        Performs decoding during testing.
    """
    def __init__(self, tagger:TreeInsertionTagger, **kw):
        super(TreeInsertionTaggerModel, self).__init__(**kw)
        self.tagger = tagger
        self.ce = MultiCELoss()
        self.recall = Recall()

    def forward(self, inpseq:torch.Tensor=None, tokens:torch.Tensor=None, openmask:torch.Tensor=None, gold:torch.Tensor=None, **kw):
        """
        Used only to train and test the tagger (this is one step of the decoding process)
        :param tokens:      (batsize, seqlen) - token ids
        :param openmask:    (batsize, seqlen) - True if token is open (not terminated)
        :param gold:        (batsize, seqlen, vocsize) - which of the possible actions are gold at every token.
        :return:
        """
        _, ctx = self.tagger.get_init_state(inpseqs=inpseq)
        probs = self.tagger(tokens, openmask=openmask, **ctx)    # (batsize, seqlen, vocsize)

        ce = self.ce(probs, gold, mask=openmask)
        elemrecall, seqrecall, anyrecall = self.recall(probs, gold, mask=openmask)
        return {"loss": ce, "ce": ce, "elemrecall": elemrecall, "allrecall": seqrecall, "anyrecall": anyrecall}, probs


def try_tree_insertion_model_tagger(batsize=10):
    tt = q.ticktock()
    tt.tick("loading")
    tds, vds, xds, tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds("restaurants")
    tt.tock("loaded")

    tdl = DataLoader(tds, batch_size=batsize, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(tdl))

    class DummyTreeInsertionTagger(TreeInsertionTagger):
        exclude = {"@PAD@", "@UNK@", "@START@", "@END@", "@MASK@", "@OPEN@", "@REMOVE@", "@REMOVESUBTREE@", "@SLOT@", "(", ")"}
        def __init__(self, vocab:Vocab, **kw):
            super(DummyTreeInsertionTagger, self).__init__(**kw)
            self.vocab = vocab
            self.vocabsize = self.vocab.number_of_ids()
            vocab_mask = torch.ones(self.vocabsize)
            for excl_token in self.exclude:
                if excl_token in self.vocab:
                    vocab_mask[self.vocab[excl_token]] = 0
            self.register_buffer("vocab_mask", vocab_mask)

        def forward(self, tokens:torch.Tensor, openmask:torch.Tensor, **kw):
            print(tokens)
            ret = torch.randn(tokens.size(0), tokens.size(1), self.vocabsize, device=tokens.device)
            ret = ret + torch.log(self.vocab_mask[None, None, :])
            ret = torch.softmax(ret, -1)
            return ret

        def get_init_state(self, inpseqs=None, y_in=None):
            batsize = inpseqs.size(0)
            trees = [ATree("@START@", [])] * batsize
            return trees, {}

    cell = DummyTreeInsertionTagger(flenc.vocab)
    m = TreeInsertionTaggerModel(cell)

    m(batch[0], batch[1], batch[2], batch[3])


ORDERLESS = {"op:and", "SW:concat"}


def simplify_tree_for_eval(x:Tree):   # removes @SLOT@'s and @START@
    children = [simplify_tree_for_eval(xe) for xe in x]
    children = [child for child in children if child is not None]
    x[:] = children
    if x.label() == "@SLOT@":
        return None
    else:
        return x


class TreeInsertionDecoder(torch.nn.Module):
    def __init__(self, tagger:TreeInsertionTagger, seqenc:SequenceEncoder=None,
                 maxsteps:int=50, max_tree_size:int=100,
                 mode:str="full", device=None, **kw):
        super(TreeInsertionDecoder, self).__init__(**kw)
        self.tagger = tagger
        self.maxsteps = maxsteps
        self.max_tree_size = max_tree_size
        self.mode = mode
        self.seqenc = seqenc

    def forward(self, inpseqs:torch.Tensor=None, y_in:torch.Tensor=None, gold:torch.Tensor=None,
                mode:str=None, maxsteps:int=None, **kw):
        """

        """
        maxsteps = maxsteps if maxsteps is not None else self.maxsteps
        mode = mode if mode is not None else self.mode
        device = next(self.parameters()).device

        trees, context = self.tagger.get_init_state(inpseqs=inpseqs, y_in=y_in)

        i = 0
        while not all([all_terminated(tree) for tree in trees]) and i < maxsteps:
            # go from tree to tensors,
            tensors = []
            masks = []
            for tree in trees:
                fltoks, openmask = extract_info(tree, nogold=True)
                seq = self.seqenc.convert(fltoks, return_what="tensor")
                tensors.append(seq)
                masks.append(torch.tensor(openmask))
            seq = torch.stack(q.pad_tensors(tensors, 0), 0).to(device)
            openmask = torch.stack(q.pad_tensors(masks, 0, False), 0).to(device)

            #  feed to tagger,
            probs = self.tagger(seq, openmask=openmask, **context)

            #  get best predictions,
            _, best = probs.max(-1)

            #  convert to trees,
            trees = [tensors_to_tree(seqe, openmask=openmaske, actions=beste, D=self.seqenc.vocab)
                     for seqe, openmaske, beste
                     in zip(list(seq), list(openmask), list(best))]

            #  and execute,
            trees_ = []
            for tree in trees:
                if tree_size(tree) < self.max_tree_size:
                    tree = mark_for_execution(tree, mode=mode)
                    budget = [self.max_tree_size - tree_size(tree)]
                    tree = execute_chosen_actions(tree, _budget=budget, mode=mode)
                trees_.append(tree)

            trees = trees_
            i += 1
            #  then repeat until all terminated

        # after done decoding, if gold is given, run losses, else return just predictions

        ret = {}

        if gold is not None:
            goldtrees = [tensors_to_tree(seqe, D=self.seqenc.vocab) for seqe in list(gold)]
            goldtrees = [simplify_tree_for_eval(x) for x in goldtrees]
            predtrees = [simplify_tree_for_eval(x) for x in trees]
            ret["treeacc"] = [float(are_equal_trees(gold_tree, pred_tree,
                            orderless=ORDERLESS, unktoken="@UNK@"))
                   for gold_tree, pred_tree in zip(goldtrees, predtrees)]
            ret["treeacc"] = torch.tensor(ret["treeacc"]).to(device)

        return ret, trees


class TreeInsertionDecoderTrainModel(torch.nn.Module):
    def __init__(self, model:TreeInsertionDecoder, **kw):
        super(TreeInsertionDecoderTrainModel, self).__init__(**kw)
        self.model = model

    def forward(self, inpseqs:torch.Tensor=None, gold:torch.Tensor=None, **kw):
        ret = self.model(inpseqs, None, gold)
        return ret


def try_tree_insertion_model_decode(batsize=10):
    tt = q.ticktock()
    tt.tick("loading")
    tds, vds, xds, tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds("restaurants")
    tt.tock("loaded")

    tdl = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    batch = next(iter(tdl))

    class DummyTreeInsertionTagger(TreeInsertionTagger):
        exclude = {"@PAD@", "@UNK@", "@START@", "@END@", "@MASK@", "@OPEN@", "@REMOVE@","@REMOVESUBTREE@", "@SLOT@", "(", ")"}
        def __init__(self, vocab:Vocab, **kw):
            super(DummyTreeInsertionTagger, self).__init__(**kw)
            self.vocab = vocab
            self.vocabsize = self.vocab.number_of_ids()
            vocab_mask = torch.ones(self.vocabsize)
            for excl_token in self.exclude:
                if excl_token in self.vocab:
                    vocab_mask[self.vocab[excl_token]] = 0
            self.register_buffer("vocab_mask", vocab_mask)

        def forward(self, tokens:torch.Tensor, openmask:torch.Tensor, **kw):
            print(tokens)
            ret = torch.randn(tokens.size(0), tokens.size(1), self.vocabsize, device=tokens.device)
            ret = ret + torch.log(self.vocab_mask[None, None, :])
            ret = torch.softmax(ret, -1)
            return ret

        def get_init_state(self, inpseqs=None, y_in=None):
            batsize = inpseqs.size(0)
            trees = [ATree("@START@", [])] * batsize
            return trees, {}

    cell = DummyTreeInsertionTagger(flenc.vocab)
    m = TreeInsertionDecoder(cell, seqenc=flenc)

    m(batch[0], None, batch[1], maxsteps=3)


class TransformerTagger(TreeInsertionTagger):
    exclude = {"@PAD@", "@UNK@", "@START@", "@END@", "@MASK@", "@OPEN@", "@REMOVE@", "@REMOVESUBTREE@", "@SLOT@", "(", ")"}
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(TransformerTagger, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        config = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                   num_layers=numlayers, num_heads=numheads, dropout_rate=dropout,
                                   use_relative_position=True)

        self.emb = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.posemb = torch.nn.Embedding(maxpos, config.d_model)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = False
        self.decoder = TransformerStack(decoder_config)

        self.out = torch.nn.Linear(self.dim, self.vocabsize)

        vocab_mask = torch.ones(self.vocabsize)
        for excl_token in self.exclude:
            if excl_token in self.vocab:
                vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname)
        def set_dropout(m:torch.nn.Module):
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout
        self.bert_model.apply(set_dropout)

        self.adapter = None
        if self.bert_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.bert_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, openmask:torch.Tensor=None,
                enc=None, encmask=None):
        padmask = (tokens != 0)
        embs = self.emb(tokens)
        # posembs = self.posemb(torch.arange(tokens.size(1), device=tokens.device))[None]
        # embs = embs + posembs
        ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=False)
        logits = self.out(ret[0])
        logits = logits + torch.log(self.vocab_mask[None, None, :])
        return logits

    def get_init_state(self, inpseqs=None, y_in=None) -> Tuple[ATree, Dict]:
        """ Encodes inpseqs and creates new states """
        assert(y_in is None)    # starting decoding from non-vanilla is not supported yet
        # encode inpseqs
        encmask = (inpseqs != 0)
        encs = self.bert_model(inpseqs)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)

        # create trees
        batsize = inpseqs.size(0)
        trees = [ATree("@START@", [])] * batsize
        return trees, {"enc": encs, "encmask": encmask}


def try_real_tree_insertion_model_tagger(batsize=10):
    tt = q.ticktock()
    tt.tick("loading")
    tds, vds, xds, tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds("restaurants")
    tt.tock("loaded")

    tdl = DataLoader(tds, batch_size=batsize, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(tdl))

    cell = TransformerTagger(512, flenc.vocab)
    m = TreeInsertionTaggerModel(cell)

    m(batch[0], batch[1], batch[2], batch[3])


def run(lr=0.001,
        enclrmul=0.1,
        hdim=768,
        numlayers=8,
        numheads=12,
        dropout=0.1,
        wreg=0.,
        batsize=10,
        epochs=100,
        warmup=0,
        sustain=0,
        cosinelr=False,
        gradacc=1,
        gradnorm=100,
        patience=5,
        validinter=3,
        seed=87646464,
        gpu=-1,
        mode="full",    # "full", "ltr" (left to right), "single"
        ):
    settings = locals().copy()
    print(json.dumps(settings, indent=4))
    datamode = "single" if mode in ("full", "single") else mode

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading")
    tds, vds, xds, tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds("restaurants", mode=datamode)
    tt.tock("loaded")

    tdl = DataLoader(tds, batch_size=batsize, shuffle=True, collate_fn=collate_fn)
    vdl = DataLoader(vds, batch_size=batsize, shuffle=False, collate_fn=collate_fn)
    xdl = DataLoader(xds, batch_size=batsize, shuffle=False, collate_fn=collate_fn)

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model
    tagger = TransformerTagger(hdim, flenc.vocab, numlayers, numheads, dropout)
    tagmodel = TreeInsertionTaggerModel(tagger)
    decodermodel = TreeInsertionDecoder(tagger, seqenc=flenc, maxsteps=50, max_tree_size=30,
                                        mode=mode)
    decodermodel = TreeInsertionDecoderTrainModel(decodermodel)

    # batch = next(iter(tdl))
    # out = tagmodel(*batch)

    tmetrics = make_array_of_metrics("loss", "elemrecall", "allrecall", "anyrecall", reduction="mean")
    vmetrics = make_array_of_metrics("loss", "elemrecall", "allrecall", "anyrecall", reduction="mean")
    tseqmetrics = make_array_of_metrics("treeacc", reduction="mean")
    vseqmetrics = make_array_of_metrics("treeacc", reduction="mean")
    xmetrics = make_array_of_metrics("treeacc", reduction="mean")

    # region parameters
    def get_parameters(m, _lr, _enclrmul):
        bertparams = []
        otherparams = []
        for k, v in m.named_parameters():
            if "bert_model." in k:
                bertparams.append(v)
            else:
                otherparams.append(v)
        if len(bertparams) == 0:
            raise Exception("No encoder parameters found!")
        paramgroups = [{"params": bertparams, "lr": _lr * _enclrmul},
                       {"params": otherparams}]
        return paramgroups
    # endregion

    def get_optim(_m, _lr, _enclrmul, _wreg=0):
        paramgroups = get_parameters(_m, _lr=lr, _enclrmul=_enclrmul)
        optim = torch.optim.Adam(paramgroups, lr=lr, weight_decay=_wreg)
        return optim

    def clipgradnorm(_m=None, _norm=None):
        torch.nn.utils.clip_grad_norm_(_m.parameters(), _norm)

    eyt = q.EarlyStopper(vseqmetrics[0], patience=patience, min_epochs=30, more_is_better=True, remember_f=lambda: deepcopy(tagger))
    # def wandb_logger():
    #     d = {}
    #     for name, loss in zip(["loss", "elem_acc", "seq_acc", "tree_acc"], metrics):
    #         d["train_"+name] = loss.get_epoch_error()
    #     for name, loss in zip(["seq_acc", "tree_acc"], vmetrics):
    #         d["valid_"+name] = loss.get_epoch_error()
    #     wandb.log(d)
    t_max = epochs
    optim = get_optim(tagger, lr, enclrmul, wreg)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=t_max-warmup) >> 0.
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, gradient_accumulation_steps=gradacc,
                                        on_before_optim_step=[lambda : clipgradnorm(_m=tagger, _norm=gradnorm)])

    trainepoch = partial(q.train_epoch, model=tagmodel,
                                        dataloader=tdl,
                                        optim=optim,
                                        losses=tmetrics,
                                        device=device,
                                        _train_batch=trainbatch,
                                        on_end=[lambda: lr_schedule.step()])

    trainseqepoch = partial(q.test_epoch,
                         model=decodermodel,
                         losses=tseqmetrics,
                         dataloader=tdl_seq,
                         device=device)

    validepoch = partial(q.test_epoch,
                         model=decodermodel,
                         losses=vseqmetrics,
                         dataloader=vdl_seq,
                         device=device,
                         on_end=[lambda: eyt.on_epoch_end()])

    # validepoch()        # TODO: remove this after debugging

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=[trainseqepoch, validepoch],
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    # inspect predictions
    validepoch = partial(q.test_epoch,
                            model=tagmodel,
                            losses=vmetrics,
                            dataloader=vdl,
                            device=device)
    print(validepoch())
    inps, outs = q.eval_loop(tagmodel, vdl, device=device)

    # print(outs)

    doexit = False
    for i in range(len(inps[0])):
        for j in range(len(inps[0][i])):
            ui = input("next? (ENTER for next/anything else to exit)>>>")
            if ui != "":
                doexit = True
                break
            question = " ".join(nltok.convert_ids_to_tokens(inps[0][i][j]))
            out_toks = flenc.vocab.tostr(inps[1][i][j].detach().cpu().numpy()).split(" ")

            iscorrect = True

            lines = []
            for k, out_tok in enumerate(out_toks):
                gold_toks_for_k = inps[3][i][j][k].detach().cpu().nonzero()[:, 0]
                if len(gold_toks_for_k) > 0:
                    gold_toks_for_k = flenc.vocab.tostr(gold_toks_for_k).split(" ")
                else:
                    gold_toks_for_k = [""]

                isopen = inps[2][i][j][k]
                isopen = isopen.detach().cpu().item()

                pred_tok = outs[1][i][j][k].max(-1)[1].detach().cpu().item()
                pred_tok = flenc.vocab(pred_tok)

                pred_tok_correct = pred_tok in gold_toks_for_k or not isopen
                if not pred_tok_correct:
                    iscorrect = False

                entropy = torch.softmax(outs[1][i][j][k], -1).clamp_min(1e-6)
                entropy = -(entropy * torch.log(entropy)).sum().item()
                lines.append(f"{out_tok:25} [{isopen:1}] >> {f'{pred_tok} ({entropy:.3f})':35} {'!!' if not pred_tok_correct else '  '} [{','.join(gold_toks_for_k) if isopen else ''}]")

            print(f"{question} {'!!WRONG!!' if not iscorrect else ''}")
            for line in lines:
                print(line)

        if doexit:
            break



if __name__ == '__main__':
    # test_multi_celoss()
    # test_tensors_to_tree()
    # try_tree_insertion_model_decode()
    # try_tree_insertion_model_tagger()
    # try_real_tree_insertion_model_tagger()
    q.argprun(run)