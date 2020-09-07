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
from parseq.scripts_insert.overnight_treeinsert import ATree, child_number_of, extract_info, collate_fn, \
    TreeInsertionTagger, MultiCELoss, Recall, add_descendants_ancestors, all_terminated
from parseq.transformer import TransformerConfig, TransformerModel, TransformerStack
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


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


def mark_for_execution(x:ATree, mode:str="random", uniformfactor=0.3, entropylimit=0.01):     # "all", "parallel:100%", "single", "ltr", "entropy-single"
    """ Marks only a selection of all nodes in the given tree for execution
        by setting ._chosen_action of other nodes to None """
    nodes_with_actions = []
    queue = [x]
    while len(queue) > 0:
        head = queue.pop(0)
        if head.is_open and hasattr(head, "_chosen_action") and head._chosen_action is not None:
            nodes_with_actions.append(head)
        queue += head[:]

    if len(nodes_with_actions) == 0:
        return x

    if mode == "random" or mode == "train-random":    # leave only one node for execution
        selected = random.choice(nodes_with_actions)
        for node in nodes_with_actions:
            if node is not selected:
                node._chosen_action = None
    elif mode == "leastentropy":
        selected = []
        if entropylimit > 0.:
            selected = [node for node in nodes_with_actions if node._entropy < entropylimit]
        if len(selected) == 0:
            selected = [sorted(nodes_with_actions, key=lambda x: x._entropy)[0]]
        for node in nodes_with_actions:
            node_in_selected = False
            for othernode in selected:
                if node is othernode:
                    node_in_selected = True
                    break
            if not node_in_selected:
                node._chosen_action = None
    elif mode == "train-leastentropy":
        selected = []
        if entropylimit > 0.:
            selected = [node for node in nodes_with_actions if node._entropy < entropylimit]
        if len(selected) == 0:
            do_random = random.random() < uniformfactor
            if do_random:
                selected = [random.choice(nodes_with_actions)]
            else:
                entropies = torch.tensor([node._entropy for node in nodes_with_actions]).detach().cpu()
                selected = torch.multinomial(1/entropies, 1).item()
                selected = [nodes_with_actions[selected]]
        for node in nodes_with_actions:
            node_in_selected = False
            for othernode in selected:
                if node is othernode:
                    node_in_selected = True
                    break
            if not node_in_selected:
                node._chosen_action = None
    return x


def convert_chosen_actions_from_str_to_node(x:ATree, c=0):
    """ a tree with chosen actions as strings and gold is transformed to chosen actions from gold """
    if x.is_open and x._chosen_action is not None:
        assert(isinstance(x._chosen_action, str))
        for gold_action in x.gold_actions:
            if isinstance(gold_action, ATree):
                if gold_action.label() == x._chosen_action:
                    x._chosen_action = gold_action
                    break
            elif isinstance(gold_action, str):
                if gold_action == x._chosen_action:
                    x._chosen_action = gold_action
                    break
        # assert(isinstance(x._chosen_action, ATree))
        c = c + 1
    children = []
    for child in x:
        child, c = convert_chosen_actions_from_str_to_node(child, c)
        children.append(child)
    x[:] = children
    if c > 1:
        assert(c <= 1)
    return x, c


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


def reorder_tree(x:Tree, orderless=None):
    if orderless is None or len(orderless) == 0 or len(x) == 0:
        return x
    else:
        children = [reorder_tree(xe, orderless=orderless) for xe in x]
        if x.label() in orderless:
            # do type first
            types = [xe for xe in children if xe.label() == "arg:~type"]
            types = sorted(types, key=lambda _xe: str(_xe))
            otherchildren = [xe for xe in children if xe.label() != "arg:~type"]
            otherchildren = sorted([xe for xe in otherchildren], key=lambda _xe: str(_xe))
            children = types + otherchildren
        x[:] = children
        return x


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
    orderless = {"op:and", "SW:concat"}     # only use in eval!!

    ds = OvernightDatasetLoader().load(domain=domain, trainonvalid=trainonvalid)
    ds = ds.map(lambda x: (x[0], ATree("@START@", [x[1]]), x[2]))

    ds = ds.map(lambda x: (x[0], reorder_tree(x[1], orderless=orderless), x[2]))

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
        fl = x[1]
        fltoks = extract_info(fl, onlytokens=True)
        seq = seqenc.convert(fltoks, return_what="tensor")
        ret = (nl_tokenizer.encode(nl, return_tensors="pt")[0],
               seq)
        return ret

    tds_seq = tds.map(mapper)
    vds_seq = vds.map(mapper)
    xds_seq = xds.map(mapper)
    return tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc, orderless


def build_atree(x:Iterable[str], open:Iterable[bool]=None, chosen_actions:Iterable[str]=None, entropies=None):
    open = [False for _ in x] if open is None else open
    chosen_actions = [None for _ in x] if chosen_actions is None else chosen_actions
    entropies = [0 for _ in x] if entropies is None else entropies
    nodes = []
    for xe, opene, chosen_action, entropy in zip(x, open, chosen_actions, entropies):
        if xe == "(" or xe == ")":
            nodes.append(xe)
            assert(opene == False)
            # assert(chosen_action is None)
        else:
            a = ATree(xe, [], is_open=opene)
            a._chosen_action = chosen_action
            a._entropy = entropy
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


def tensors_to_tree(x, openmask=None, actions=None, D:Vocab=None, entropies=None):
    # x: 1D int tensor
    x = list(x.detach().cpu().numpy())
    x = [D(xe) for xe in x]
    x = [xe for xe in x if xe != D.padtoken]

    if openmask is not None:
        openmask = list(openmask.detach().cpu().numpy())
    if actions is not None:
        actions = list(actions.detach().cpu().numpy())
        actions = [D(xe) for xe in actions][:len(x)]
    if entropies is not None:
        entropies = list(entropies.detach().cpu().numpy())

    tree = build_atree(x, open=openmask, chosen_actions=actions, entropies=entropies)
    return tree


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
                 mode:str="random", device=None, **kw):
        super(TreeInsertionDecoder, self).__init__(**kw)
        self.tagger = tagger
        self.maxsteps = maxsteps
        self.max_tree_size = max_tree_size
        self.mode = mode
        self.seqenc = seqenc
        self.uniformfactor = 0.25

        self.ce = MultiCELoss()
        self.recall = Recall()

        self.entropylimit = 0.

    def attach_info_to_tree(self, x:ATree, **kw):
        queue = ["(", x, ")"]
        i = 0
        while len(queue) > 0:
            first = queue.pop(0)
            if isinstance(first, ATree):
                for k, v in kw.items():
                    setattr(first, k, v[i])
                queueprefix = []
                for fc in first:
                    if len(fc) == 0:
                        queueprefix.append(fc)
                    else:
                        queueprefix += ["(", fc, ")"]
                queue = queueprefix + queue
            i += 1
        return x

    def forward(self, inpseqs:torch.Tensor=None, gold:torch.Tensor=None,
                mode:str=None, maxsteps:int=None, **kw):
        """

        """
        maxsteps = maxsteps if maxsteps is not None else self.maxsteps
        mode = mode if mode is not None else self.mode
        device = next(self.parameters()).device

        trees, context = self.tagger.get_init_state(inpseqs=inpseqs, y_in=None)
        if gold is not None:
            goldtrees = [tensors_to_tree(seqe, D=self.seqenc.vocab) for seqe in list(gold)]
            goldtrees = [add_descendants_ancestors(goldtree) for goldtree in goldtrees]
            for i in range(len(trees)):
                trees[i].align = goldtrees[i]

        i = 0

        if self.training:
            trees = [assign_gold_actions(tree, mode=mode) for tree in trees]
        # choices = [deepcopy(trees)]
        numsteps = [0 for _ in range(len(trees))]
        treesizes = [0 for _ in range(len(trees))]
        seqlens = [0 for _ in range(len(trees))]
        allmetrics = {}
        allmetrics["loss"] = []
        allmetrics["ce"] = []
        allmetrics["elemrecall"] = []
        allmetrics["allrecall"] = []
        allmetrics["anyrecall"] = []
        allmetrics["lowestentropyrecall"] = []
        examplemasks = []
        while not all([all_terminated(tree) for tree in trees]) and i < maxsteps:
            # go from tree to tensors,
            seq = []
            openmask = []
            stepgold = []
            for j, tree in enumerate(trees):
                if not all_terminated(tree):
                    numsteps[j] += 1
                treesizes[j] = tree_size(tree)

                fltoks, openmask_e, stepgold_e = extract_info(tree)
                seqlens[j] = len(fltoks)
                seq_e = self.seqenc.convert(fltoks, return_what="tensor")
                seq.append(seq_e)
                openmask.append(torch.tensor(openmask_e))
                if self.training:
                    stepgold_e_tensor = torch.zeros(seq_e.size(0), self.seqenc.vocab.number_of_ids())
                    for j, stepgold_e_i in enumerate(stepgold_e):
                        for golde in stepgold_e_i:
                            stepgold_e_tensor[j, self.seqenc.vocab[golde]] = 1
                    stepgold.append(stepgold_e_tensor)
            seq = torch.stack(q.pad_tensors(seq, 0, 0), 0).to(device)
            openmask = torch.stack(q.pad_tensors(openmask, 0, False), 0).to(device)
            if self.training:
                stepgold = torch.stack(q.pad_tensors(stepgold, 0, 0), 0).to(device)

            if self.training:
                examplemask = (stepgold != 0).any(-1).any(-1)
                examplemasks.append(examplemask)

            #  feed to tagger,
            probs = self.tagger(seq, openmask=openmask, **context)

            if self.training:
                # stepgold is (batsize, seqlen, vocsize) with zeros and ones (ones for good actions)
                ce = self.ce(probs, stepgold, mask=openmask)
                elemrecall, allrecall, anyrecall, lowestentropyrecall = self.recall(probs, stepgold, mask=openmask)
                allmetrics["loss"].append(ce)
                allmetrics["ce"].append(ce)
                allmetrics["elemrecall"].append(elemrecall)
                allmetrics["allrecall"].append(allrecall)
                allmetrics["anyrecall"].append(anyrecall)
                allmetrics["lowestentropyrecall"].append(lowestentropyrecall)

            #  get best predictions,
            _, best_actions = probs.max(-1)
            entropies = torch.softmax(probs, -1).clamp_min(1e-6)
            entropies = - (entropies * torch.log(entropies)).sum(-1)

            if self.training:
                newprobs = torch.softmax(probs, -1) * stepgold              # mask using gold
                newprobs = newprobs / newprobs.sum(-1)[:, :, None].clamp_min(1e-6)  # renormalize
                uniform = stepgold
                uniform = uniform / uniform.sum(-1)[:, :, None].clamp_min(1e-6)
                newprobs = newprobs * (1 - q.v(self.uniformfactor)) + uniform * q.v(self.uniformfactor)

                noprobsmask = (newprobs != 0).any(-1, keepdim=True).float()
                zeroprobs = torch.zeros_like(newprobs)
                zeroprobs[:, :, 0] = 1
                newprobs = newprobs * noprobsmask + (1-noprobsmask) * zeroprobs

                sampled_gold_actions = torch.distributions.categorical.Categorical(probs=newprobs.view(-1, newprobs.size(-1)))\
                    .sample().view(newprobs.size(0), newprobs.size(1))
                taken_actions = sampled_gold_actions
            else:
                taken_actions = best_actions

            #  attach chosen actions and entropies to existing trees,
            for tree, actions_e, entropies_e in zip(trees, taken_actions, entropies):
                actions_e = list(actions_e.detach().cpu().numpy())
                actions_e = [self.seqenc.vocab(xe) for xe in actions_e]
                entropies_e = list(entropies_e.detach().cpu().numpy())
                self.attach_info_to_tree(tree,
                    _chosen_action=actions_e,
                    _entropy=entropies_e)
            # trees = [tensors_to_tree(seqe, openmask=openmaske, actions=actione, D=self.seqenc.vocab, entropies=entropies_e)
            #          for seqe, openmaske, actione, entropies_e
            #          in zip(list(seq), list(openmask), list(taken_actions), list(entropies))]

            #  and execute,
            trees_ = []
            for tree in trees:
                if tree_size(tree) < self.max_tree_size:
                    markmode = mode if not self.training else "train-"+mode
                    tree = mark_for_execution(tree, mode=markmode, entropylimit=self.entropylimit)
                    budget = [self.max_tree_size - tree_size(tree)]
                    if self.training:
                        tree, _ = convert_chosen_actions_from_str_to_node(tree)
                    tree = execute_chosen_actions(tree, _budget=budget, mode=mode)
                    if self.training:
                        tree = assign_gold_actions(tree, mode=mode)
                trees_.append(tree)

            trees = trees_
            i += 1
            #  then repeat until all terminated

        # after done decoding, if gold is given, run losses, else return just predictions

        ret = {}

        ret["seqlens"] = torch.tensor(seqlens).float()
        ret["treesizes"] = torch.tensor(treesizes).float()
        ret["numsteps"] = torch.tensor(numsteps).float()

        if self.training:
            assert(len(examplemasks) > 0)
            assert(len(allmetrics["loss"]) > 0)
            allmetrics = {k: torch.stack(v, 1) for k, v in allmetrics.items()}
            examplemasks = torch.stack(examplemasks, 1).float()
            _allmetrics = {}
            for k, v in allmetrics.items():
                _allmetrics[k] = (v * examplemasks).sum(1) / examplemasks.sum(1).clamp_min(1e-6)
            allmetrics = _allmetrics
            ret.update(allmetrics)

        if gold is not None:
            goldtrees = [tensors_to_tree(seqe, D=self.seqenc.vocab) for seqe in list(gold)]
            goldtrees = [simplify_tree_for_eval(x) for x in goldtrees]
            predtrees = [simplify_tree_for_eval(x) for x in trees]
            ret["treeacc"] = [float(are_equal_trees(gold_tree, pred_tree,
                            orderless=ORDERLESS, unktoken="@UNK@"))
                   for gold_tree, pred_tree in zip(goldtrees, predtrees)]
            ret["treeacc"] = torch.tensor(ret["treeacc"]).to(device)

        return ret, trees


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
        trees = [ATree("@START@", []) for _ in range(batsize)]
        return trees, {"enc": encs, "encmask": encmask}


def run(lr=0.001,
        minlr=0.000001,
        enclrmul=0.1,
        hdim=768,
        numlayers=8,
        numheads=12,
        dropout=0.1,
        wreg=0.,
        batsize=10,
        epochs=100,
        warmup=0,
        cosinelr=False,
        sustain=0,
        cooldown=0,
        gradacc=1,
        gradnorm=100,
        patience=5,
        validinter=1,
        seed=87646464,
        gpu=-1,
        mode="leastentropy",
        trainonvalid=False,
        entropylimit=0.,
        # datamode="single",
        # decodemode="single",    # "full", "ltr" (left to right), "single", "entropy-single"
        ):
    settings = locals().copy()
    print(json.dumps(settings, indent=4))

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading")
    tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds("restaurants", trainonvalid=trainonvalid)
    tt.tock("loaded")

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model
    tagger = TransformerTagger(hdim, flenc.vocab, numlayers, numheads, dropout)
    decodermodel = TreeInsertionDecoder(tagger, seqenc=flenc, maxsteps=70, max_tree_size=30, mode=mode)
    decodermodel.entropylimit = entropylimit

    # batch = next(iter(tdl))
    # out = tagmodel(*batch)

    tmetrics = make_array_of_metrics("loss", "elemrecall", "allrecall", "lowestentropyrecall", reduction="mean")
    tvmetrics = make_array_of_metrics("treesizes", "seqlens", "numsteps", "treeacc", reduction="mean")
    vmetrics = make_array_of_metrics("treesizes", "seqlens", "numsteps", "treeacc", reduction="mean")
    xmetrics = make_array_of_metrics("treesizes", "seqlens", "numsteps", "treeacc", reduction="mean")

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

    eyt = q.EarlyStopper(vmetrics[-1], patience=patience, min_epochs=30, more_is_better=True, remember_f=lambda: deepcopy(tagger))
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
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Constant(1., steps=sustain) >> q.sched.Cosine(low=minlr, high=lr, steps=t_max-warmup-sustain-cooldown) >> q.sched.Constant(minlr, steps=cooldown)
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, gradient_accumulation_steps=gradacc,
                                        on_before_optim_step=[lambda : clipgradnorm(_m=tagger, _norm=gradnorm)])

    trainepoch = partial(q.train_epoch, model=decodermodel,
                                        dataloader=tdl_seq,
                                        optim=optim,
                                        losses=tmetrics,
                                        device=device,
                                        _train_batch=trainbatch,
                                        on_end=[lambda: lr_schedule.step()])

    trainvalidepoch = partial(q.test_epoch,
                         model=decodermodel,
                         losses=tvmetrics,
                         dataloader=tdl_seq,
                         device=device,)

    validepoch = partial(q.test_epoch,
                         model=decodermodel,
                         losses=vmetrics,
                         dataloader=vdl_seq,
                         device=device,
                         on_end=[lambda: eyt.on_epoch_end()])

    # validepoch()        # TODO: remove this after debugging

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=[trainvalidepoch, validepoch],
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    if eyt.remembered is not None:
        decodermodel.tagger = eyt.remembered
    tt.msg("reloaded best")

    tt.tick("trying different entropy limits")
    vmetrics2 = make_array_of_metrics("treesizes", "seqlens", "numsteps", "treeacc", reduction="mean")
    validepoch = partial(q.test_epoch,
                         model=decodermodel,
                         losses=vmetrics2,
                         dataloader=vdl_seq,
                         device=device)
    entropylimits = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 10][::-1]
    for _entropylimit in entropylimits:
        tt.msg(f"entropy limit {_entropylimit}")
        decodermodel.entropylimit = _entropylimit
        tt.msg(validepoch())
    tt.tock("done trying entropy limits")

    tt.tick("testing on test")
    testepoch = partial(q.test_epoch,
                         model=decodermodel,
                         losses=xmetrics,
                         dataloader=xdl_seq,
                         device=device,)
    print(testepoch())
    tt.tock("tested on test")




if __name__ == '__main__':
    # test_multi_celoss()
    # test_tensors_to_tree()
    # try_tree_insertion_model_decode()
    # try_tree_insertion_model_tagger()
    # try_real_tree_insertion_model_tagger()
    q.argprun(run)