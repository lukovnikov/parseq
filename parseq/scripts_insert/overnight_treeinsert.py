import random
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Tuple, Iterable, List

import numpy as np
import torch
from nltk import Tree
from torch.utils.data import DataLoader

import qelos as q
from parseq.datasets import OvernightDatasetLoader, autocollate
from parseq.grammar import tree_to_lisp_tokens, tree_to_prolog
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer


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


def compute_gold_actions(x:ATree):
    """ x must be an aligned tree.
        this method will assign to every non-terminated node in this tree the gold actions according to one of the possible alignments """


def add_descendants_ancestors(x:ATree, ancestors:Iterable[ATree]=tuple(), orderless=set()):
    x._ancestors = ancestors
    x._descendants = []
    if x.label() not in ("(", ")"):
        for ancestor in ancestors:
            ancestor._descendants.append(x)
        for i, xe in enumerate(x):
            add_descendants_ancestors(xe, ancestors+(x,))
            xe._parent = x
    return x


def child_number_of(x:ATree):
    p = x._parent
    for i in range(len(p)):
        if p[i] is x:
            return i
    return None     # not a child


def assign_gold_actions(x:ATree, orderless=None):
    """ assigns actions that can be taken at every node of the given tree """
    orderless = set() if orderless is None else orderless
    for xe in x:
        assign_gold_actions(xe, orderless)
    if not x.is_open:
        x.gold_actions = []
    else:
        if x.label() == "(":        # can decode any ancestor in between here and its parent
            x.gold_actions = []
            for ancestor in x.align._ancestors[::-1]:
                if ancestor is x._parent.align:
                    break
                else:
                    x.gold_actions.append(ancestor)
            if len(x.gold_actions) == 0:
                x.gold_actions.append("@CLOSE@")
        elif x.label() == ")":
            x.is_open = False
            x.gold_actions = []
        elif x.label() == "@SLOT@":
            # get this slots's siblings
            x.gold_actions = []
            if x._parent.label() in orderless:
                # retrieve aligned nodes which have not been decoded yet
                decoded = [xe.align for xe in x._parent if xe.label() != "@SLOT@"]
                for xe in x._parent.align:
                    if not any([decodede is xe for decodede in decoded]):
                        x.gold_actions.append(xe)
            else:
                leftsibling = x._parent[child_number_of(x) - 1]
                rightsibling = x._parent[child_number_of(x) + 1]
                leftsibling_nr = child_number_of(leftsibling.align)
                rightsibling_nr = child_number_of(rightsibling.align)
                p = leftsibling.align._parent
                for i in range(leftsibling_nr+1, rightsibling_nr):
                    x.gold_actions.append(p[i])
            if len(x.gold_actions) == 0:
                x.gold_actions = ["@CLOSE@"]
        else:       # not a sibling slot ("@SLOT@"), not a "(" or ")"
            if len(x) == 0:      # if it has children: can't do anything
                if len(x.align._descendants) > 0:
                    x.gold_actions = x.align._descendants
                else:
                    x.gold_actions = ["@CLOSE@"]

    if len(x.gold_actions) > 0:
        # x._chosen_action = x.gold_actions[0]
        x._chosen_action = random.choice(x.gold_actions)
    else:
        x._chosen_action = None
    return x


def mark_for_execution(x:ATree, mode:str="single"):
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
    return x


def execute_chosen_actions(x:ATree, orderless=None):
    orderless = set() if orderless is None else orderless
    if x._chosen_action is None or not x.is_open:
        iterr = list(x)
        for xe in iterr:
            execute_chosen_actions(xe, orderless=orderless)
        return x
    if x.label() == "(":    # insert a parent before current parent
        if x._chosen_action == "@CLOSE@":
            x.is_open = False
        else:
            p = x._parent
            if isinstance(x._chosen_action, Tree):
                newnode = ATree(x._chosen_action.label(), p[:])
            else:       # string
                newnode = ATree(x._chosen_action, p[:])
            for xe in p:
                xe._parent = newnode
            newnode.is_open = False
            if isinstance(x._chosen_action, Tree):
                newnode.align = x._chosen_action
            newnode._parent = p

            if newnode.label() in orderless and p.label() not in orderless:
                # convert from ordered to orderless
                # remove all @SLOT@'s, add one @SLOT@ if there was any
                newchildren = [child for child in newnode if child.label() != "@SLOT@"]
                anyslot = len(newnode) - len(newchildren)
                if anyslot > 1:
                    leftslot = ATree("@SLOT@", [])
                    leftslot.is_open = True
                    leftslot._parent = newnode
                    newchildren.insert(1, leftslot)
                    newnode[:] = newchildren

            elif newnode.label() not in orderless and p.label() in orderless:
                # convert form orderless to ordered
                # remove the @SLOT@ and insert @SLOT@ at every second position
                newchildren = [child for child in newnode if child.label() != "@SLOT@"]
                anyslot = len(newchildren) < len(newnode)
                assert(len(newchildren) >= len(newnode) - 1)
                if anyslot:
                    newnode[:] = [newchildren.pop(0)]
                    while len(newchildren) > 0:
                        leftslot = ATree("@SLOT@", [])
                        leftslot.is_open = True
                        leftslot._parent = newnode
                        newnode.append(leftslot)
                        newnode.append(newchildren.pop(0))

            leftbracket = ATree("(", [])
            leftbracket.is_open = True
            if isinstance(x._chosen_action, Tree):
                leftbracket.align = newnode.align._parent[0]
            leftbracket._parent = newnode._parent
            rightbracket = ATree(")", [])
            rightbracket.is_open = False

            if isinstance(x._chosen_action, Tree):
                rightbracket.align = newnode.align._parent[-1]
            rightbracket._parent = newnode._parent
            leftslot = ATree("@SLOT@", [])
            leftslot.is_open = True
            leftslot._parent = newnode._parent
            rightslot = ATree("@SLOT@", [])
            rightslot.is_open = True
            rightslot._parent = newnode._parent

            if p.label() in orderless:
                p[:] = [leftbracket, leftslot, newnode, rightbracket]
            else:
                p[:] = [leftbracket, leftslot, newnode, rightslot, rightbracket]
    elif x.label() == ")":
        pass
    elif x.label() == "@SLOT@":
        if x._chosen_action == "@CLOSE@":
            del x._parent[child_number_of(x)]
        else:
            if isinstance(x._chosen_action, Tree):
                x.set_label(x._chosen_action.label())
            else:
                x.set_label(x._chosen_action)

            if isinstance(x._chosen_action, Tree):
                x.align = x._chosen_action
            x.is_open = True

            leftslot = ATree("@SLOT@", [])
            leftslot.is_open = True
            leftslot._parent = x._parent
            rightslot = ATree("@SLOT@", [])
            rightslot.is_open = True
            rightslot._parent = x._parent

            if x._parent.label() in orderless:
                x._parent.insert(1, leftslot)
            else:
                x._parent.insert(child_number_of(x), leftslot)
                x._parent.insert(child_number_of(x)+1, rightslot)
    else:
        iterr = list(x)
        for xe in iterr:
            execute_chosen_actions(xe, orderless=orderless)
        if x._chosen_action == "@CLOSE@":
            x.is_open = False
        else:
            x.is_open = False
            # add child, with "(" and ")" and "@SLOT@" nodes

            if isinstance(x._chosen_action, Tree):
                newnode = ATree(x._chosen_action.label(), [])
            else:
                newnode = ATree(x._chosen_action, [])

            newnode.is_open = True

            if isinstance(x._chosen_action, Tree):
                newnode.align = x._chosen_action
            newnode._parent = x

            leftbracket = ATree("(", [])
            leftbracket.is_open = True

            if isinstance(x._chosen_action, Tree):
                leftbracket.align = newnode.align._parent[0]
            leftbracket._parent = newnode._parent
            rightbracket = ATree(")", [])
            rightbracket.is_open = False

            if isinstance(x._chosen_action, Tree):
                rightbracket.align = newnode.align._parent[-1]
            rightbracket._parent = newnode._parent

            leftslot = ATree("@SLOT@", [])
            leftslot.is_open = True
            leftslot._parent = newnode._parent
            rightslot = ATree("@SLOT@", [])
            rightslot.is_open = True
            rightslot._parent = newnode._parent

            if x.label() in orderless:
                x[:] = [leftbracket, leftslot, newnode, rightbracket]
            else:
                x[:] = [leftbracket, leftslot, newnode, rightslot, rightbracket]
    return x


def uncomplete_tree_parallel(x:ATree, orderless=None):
    """ Input is tuple (nl, fl, split)
        Output is a randomly uncompleted tree,
            every node annotated whether it's terminated and what actions are good at that node
    """
    orderless = set() if orderless is None else orderless
    # region 1. initialize annotations
    fl = x

    fl._parent = None
    add_descendants_ancestors(fl)

    # region 2.
    y = ATree("@START@", [])
    y.align = fl
    y.is_open = True


    i = 0
    choices = [deepcopy(y)]
    y = assign_gold_actions(y, orderless=orderless)
    while not all_terminated(y):
        y = mark_for_execution(y, mode="single")
        y = execute_chosen_actions(y, orderless=orderless)
        y = assign_gold_actions(y, orderless=orderless)
        choices.append(deepcopy(y))
        i += 1

    ret = random.choice(choices)
    return ret


def extract_info(x:ATree, onlytokens=False, nogold=False):
    """ Receives an annotated tree (with parentheses) and returns:
            - a sequence of tokens derived from that tree
            - a sequence of whether the token is terminated
            - a sequence of sets of gold labels
    """
    tokens, openmask, golds = [], [], []
    queue = [x]
    while len(queue) > 0:
        first = queue.pop(0)
        tokens.append(first.label())
        if not onlytokens:
            openmask.append(first.is_open if hasattr(first, "is_open") else False)
        if not onlytokens and not nogold:
            gold = set()
            if hasattr(first, "gold_actions"):
                for golde in first.gold_actions:
                    gold.add(golde.label() if isinstance(golde, Tree) else golde)
            golds.append(gold)
        queue = first[:] + queue

    if onlytokens:
        return tokens
    elif nogold:
        return tokens, openmask
    else:
        return tokens, openmask, golds


def load_ds(domain="restaurants", nl_mode="bert-base-uncased", trainonvalid=False):
    """
    Creates a dataset of examples which have
    * NL question and tensor
    * original FL tree
    * reduced FL tree with slots (this is randomly generated)
    * tensor corresponding to reduced FL tree with slots
    * mask specifying which elements in reduced FL tree are terminated
    * 2D gold that specifies whether a token/action is in gold for every position (compatibility with MML!)
    """
    orderless = {"op:and", "SW:concat"}

    ds = OvernightDatasetLoader().load(domain=domain, trainonvalid=trainonvalid)
    ds = ds.map(lambda x: (x[0], add_parentheses(ATree("@START@", [x[1]])), x[2]))

    vocab = Vocab(padid=0, startid=2, endid=3, unkid=1)
    vocab.add_token("@START@", seen=np.infty)
    vocab.add_token("@CLOSE@", seen=np.infty)        # only here for the action of closing an open position, will not be seen at input
    vocab.add_token("@OPEN@", seen=np.infty)         # only here for the action of opening a closed position, will not be seen at input
    vocab.add_token("@REMOVE@", seen=np.infty)       # only here for deletion operations, won't be seen at input
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
        fl = uncomplete_tree_parallel(x[1], orderless=orderless)
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
    return _tds, _vds, _xds, tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc


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
    def forward(self, tokens:torch.Tensor, openmask:torch.Tensor, **kw):
        """
        :param tokens:      (batsize, seqlen)
        :param openmask:    (batsize,) - True if token is terminated
        :return:
        """
        pass

    @abstractmethod
    def get_init_state(self, **kw):
        """ Run encoding on context etc.
        Return a dictionary that will be used as the kwargs in .forward() of the tagger during decoding."""
        pass


class MultiCELoss(torch.nn.Module):
    def __init__(self, aggmode:str="mean", mode:str="logits", **kw):
        super(MultiCELoss, self).__init__(**kw)
        self.mode = mode
        self.aggmode = aggmode

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

        if self.aggmode == "mean":
            a = selectedloss.sum()
            b = (zeromask * mask.float()).sum()
            loss = a/b
        else:
            raise Exception("unknown aggmode")
        return loss


class Recall(torch.nn.Module):
    def __init__(self, aggmode:str="mean", **kw):
        super(Recall, self).__init__(**kw)
        self.aggmode = aggmode

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
        extragolds[:, :, 0] = 1
        golds = golds * (zeromask.unsqueeze(-1)) + extragolds * (1 - zeromask.unsqueeze(-1))

        _, best = probs.max(-1)
        bestingold = golds.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        # (batsize, seqlen)

        _mask = zeromask * mask.float()
        elemrecall = bestingold.sum()
        seqrecall = ((bestingold >= 1) | ~(_mask >= 1)).all(-1).float()
        b = _mask.sum()
        if self.aggmode == "mean":
            elemrecall = elemrecall / b
            seqrecall = seqrecall.mean()
        else:
            raise Exception("unknown aggmode")

        return elemrecall, seqrecall


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
    print(x)
    open = [False for _ in x] if open is None else open
    chosen_actions = [None for _ in x] if chosen_actions is None else chosen_actions
    nodes = [ATree(xe, [], is_open=opene) for xe, opene in zip(x, open)]
    for node, chosen_action in zip(nodes, chosen_actions):
        node._chosen_action = chosen_action

    buffer = list(nodes)
    stack = []
    keepgoing = len(buffer) > 0
    while keepgoing:
        if len(stack) > 0 and stack[-1].label() == ")":    # closing -> create subtree and push to stack
            # find opening tag
            acc = [stack.pop(-1)]
            while not acc[-1].label() == "(":
                acc.append(stack.pop(-1))
            node = stack.pop(-1)
            node[:] = acc[::-1]
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
    tds, vds, xds, tds_seq, vds_seq, xds_seq, nltok, flenc = load_ds("restaurants")
    tdl = DataLoader(tds, batch_size=5, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(tdl))

    trees = [tensors_to_tree(seqe, openmask=openmaske, actions=beste, D=flenc.vocab)
             for seqe, openmaske, beste in
             zip(list(batch[1]), list(batch[2]), list(batch[3].max(-1)[1]))]




class TreeInsertionModel(torch.nn.Module):
    """ A tree insertion model used for training and inference.
        Receives both input to the tagging model as well as gold.
        Computes loss during training.
        Performs decoding during testing.
    """
    def __init__(self, tagger:TreeInsertionTagger, **kw):
        super(TreeInsertionModel, self).__init__(**kw)
        self.tagger = tagger
        self.ce = MultiCELoss()
        self.recall = Recall()

    def forward(self, tokens:torch.Tensor, openmask:torch.Tensor, gold:torch.Tensor, **kw):
        """
        Used only to train and test the tagger (this is one step of the decoding process)
        :param tokens:      (batsize, seqlen) - token ids
        :param openmask:    (batsize, seqlen) - True if token is open (not terminated)
        :param gold:        (batsize, seqlen, vocsize) - which of the possible actions are gold at every token.
        :return:
        """
        initstate = self.tagger.get_init_state(**kw["context"])
        probs = self.tagger(**initstate)    # (batsize, seqlen, vocsize)

        ce = self.ce(probs, gold, mask=openmask)
        elemrecall, seqrecall = self.recall(probs, gold, mask=openmask)
        return {"loss": ce, "ce": ce, "elemrecall": elemrecall, "seqrecall": seqrecall}

    def decode(self, batsize, mode="parallel:100%", maxsteps=100, **kw):
        """
        Generates a tree using the trained tagging model.
        Used for testing.
        :param mode:    string specifying how to decode
        """
        initstate = self.tagger.get_init_state(**kw["context"])
        trees = [ATree("@START@", [])] * batsize

        # go from tree to tensors,
        tensors = []
        masks = []
        for tree in trees:
            fltoks, openmask = extract_info(tree, onlytokens=True)
            seq = self.seqenc.convert(fltoks, return_what="tensor")
            tensors.append(seq)
            masks.append(openmask)
        seq = torch.stack(q.pad_tensors(tensors, 0), 0)
        openmask = torch.stack(q.pad_tensors(masks, 0, False), 0)

        #  feed to tagger,
        probs = self.tagger(seq, openmask=openmask)

        #  get best predictions,
        _, best = probs.max(-1)

        #  convert to trees
        trees = [tensors_to_tree(seqe, openmask=openmaske, actions=beste, D=self.seqenc.vocab)
                 for seqe, openmaske, beste
                 in zip(list(seq), list(openmask), list(best))]
        #  and execute,
        #  then repeat until all terminated




def run(lr=0.001,
        batsize=10):
    tt = q.ticktock("script")
    tt.tick("loading")
    tds, vds, xds, tds_seq, vds_seq, xds_seq, nltok, flenc = load_ds("restaurants")
    tt.tock("loaded")

    tdl = DataLoader(tds, batch_size=batsize, shuffle=True, collate_fn=collate_fn)
    vdl = DataLoader(vds, batch_size=batsize, shuffle=False, collate_fn=collate_fn)
    xdl = DataLoader(xds, batch_size=batsize, shuffle=False, collate_fn=collate_fn)

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    tt.tick("creating one batch")
    example = next(iter(tdl))
    tt.tock("created one batch")
    print(example)

    print(next(iter(tdl_seq)))


if __name__ == '__main__':
    # test_multi_celoss()
    test_tensors_to_tree()
    # q.argprun(run)