import math
import re

from prompt_toolkit.formatted_text import PygmentsTokens

import json
import random
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Dict, List, Union, Iterable, Tuple

import torch
import wandb
from nltk import Tree
import numpy as np
from torch.utils.data import DataLoader

import qelos as q

from parseq.datasets import OvernightDatasetLoader, autocollate
from parseq.eval import make_array_of_metrics
from parseq.grammar import tree_to_lisp_tokens, are_equal_trees, lisp_to_tree, tree_size
from parseq.scripts_insert.overnight_treeinsert import extract_info
from parseq.scripts_insert.util import reorder_tree, flatten_tree
from parseq.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


ORDERLESS = {"op:and", "SW:concat", "filter"}


def tree_to_seq_special(x:Tree):
    childseqs = [tree_to_seq_special(xe) for xe in x]
    if len(childseqs) > 0:
        xseq = ["(", x.label(), "|" ] + [childseq_e for childseq in childseqs for childseq_e in childseq] + [")"]
    else:
        xseq = ["(", x.label(), ")"]
    return xseq


def special_seq_to_tree(x:List[str]):
    xseq = [xe for xe in x if xe != "|"]
    xstr = " ".join(xseq)
    ret = lisp_to_tree(xstr)
    return ret


def special_str_to_tree(x:str):
    xstr = x.replace("|","").replace("\s+", " ").strip()
    ret = lisp_to_tree(xstr)
    return ret


def compute_centralities(distances):
    centralities = {}
    for src in distances:
        acc = sum(distances[src].values())
        centralities[src] = 1 / max(acc, 1e-3)
    return centralities


def compute_distances_graph(nodes, edges):
    """ Run message passing to compute minimum distances.
        Takes a list of nodes with nodeids and edges (expressed using those nodeids)
        Returns a dictionary mapping from nodeid u to a dictionary that maps from a nodeid v
        to an integer that is the minimum distance between u and v in the graph.
    """
    _nodes = nodes
    nodes = [n.nodeid for n in nodes]
    # remap nodes which are trees to numbers
    maxsteps = len(nodes)
    # compute which nodes are neighbours
    neighbours = {n: set() for n in nodes}
    for src, dst, _ in edges:
        neighbours[src].add(dst)

    # initialize messages
    messages = {n: {n: 0} for n in nodes}
    inboxes = {n: [] for n in nodes}

    i = 0
    allreached = False
    while i < maxsteps and not allreached:
        allreached = True
        # distribute messages:
        for n in nodes:                    # for every node n
            for neighbour in neighbours[n]:     # for every neighbour reachable from n
                inboxes[neighbour].append(messages[n])  # add current node message to n's inbox

        # aggregate incoming messages into new message for a node
        for n in nodes:
            message = {}
            for m in inboxes[n]:
                for k, v in m.items():
                    if k not in message:
                        message[k] = v
                    else:
                        message[k] = min(message[k], v)
            message = {k: v+1 for k, v in message.items()}
            message[n] = 0
            if set(message.keys()) != set(nodes):
                allreached = False
            messages[n] = message
        i += 1

    return messages


# def _run_bottom_up(x:Tree):
#     message = {x.nodeid: 0}
#     msgs = {}
#     for xe in x:
#         _m = _run_bottom_up(xe)
#         for _me in _m:
#             assert(_me not in msgs)
#             msgs[_me] = _m[_me]
#         m = _m[xe.nodeid]
#         for k, v in m.items():
#             if k not in message:
#                 message[k] = v+1
#             else:
#                 message[k] = min(message[k], v+1)
#     msgs[x.nodeid] = message
#     return msgs
#
#
# def _run_top_down(x:Tree, parent_msg, msgs):
#     message = msgs[x.nodeid]
#     for k, v in parent_msg.items():
#         if k not in message:
#             message[k] = v + 1
#         else:
#             message[k] = min(message[k], v + 1)
#
#     for i, child in enumerate(x):
#         _run_top_down(child, message, msgs)
#
#     for i in range(len(x) - 1):
#         l, r = x[i], x[i+1]
#
#
# def compute_distances_tree(x:Tree):
#     # bottom up
#     msgs = _run_bottom_up(x)
#     # top down
#     _run_top_down(x, {}, msgs)
#     return msgs


def _gather_descendants(tree:Tree, _top=True):
    ret = []
    for child in tree:
        for n in _gather_descendants(child, _top=False):
            ret.append(n)
    if _top is False:
        ret.append(tree)
    return ret


def assign_dfs_nodeids(x:Tree):
    nodes = []
    queue = [x]
    while len(queue) > 0:
        e = queue.pop(0)
        nodes.append(e)
        e.nodeid = len(nodes) - 1
        queue = e[:] + queue
    return x


def assign_f_nodeids(x:Tree, f=None):
    nodes = []
    queue = [x]
    while len(queue) > 0:
        e = queue.pop(0)
        nodes.append(e)
        e.nodeid = f(e)
        queue = e[:] + queue
    return x


def build_graph(x:Tree):
    """ Build graph for given tree """
    nodes = []
    edges = []

    queue = [x]
    while len(queue) > 0:
        e = queue.pop(0)
        nodes.append(e)
        # assert(e.nodeid == len(nodes) - 1)
        if len(e) > 0:
            children = e[:]
            for i, child in enumerate(children):
                edges.append((e.nodeid, child.nodeid, f"child-{i+1}"))
                edges.append((child.nodeid, e.nodeid, f"parent-{i+1}"))
            for i in range(len(children) - 1):
                edges.append((children[i].nodeid, children[i + 1].nodeid, "next"))
                edges.append((children[i + 1].nodeid, children[i].nodeid, "prev"))
            queue = children + queue
    return nodes, edges


def deepcopy_tree(x:Tree):
    ychildren = [deepcopy_tree(xe) for xe in x]
    y = Tree(x.label(), children=ychildren)
    if hasattr(x, "nodeid"):
        y.nodeid = x.nodeid
    return y


def get_supervision_sets(tree:Tree, node:Tree):
    """
    Should return T, Tprime, B, L, R:
        T is top part of 'tree', where the subtree of the parent of 'node' is removed,
        Tprime is a list of nodes in T that form the ancestors of 'node'
        B, L, R are lists of subtrees of tree.
    The provided 'tree' is modified in place, so pass a deepcopy to be safe!
    """
    if tree is node:    # if top-level recursion: return
        return None, [], tree[:], [], []

    A = []
    B, L = None, None
    i = 0
    while i < len(tree):
        child = tree[i]
        if child is node:   # found it
            B = child[:]
            L = A
            A = []
            if isinstance(tree, Tree):
                Tprime = [tree]
            else:
                Tprime = []
        else:
            A.append(child)
        i += 1

    if B is None:   # not found --> go deeper
        i = 0
        while i < len(tree):
            child = tree[i]
            ret = get_supervision_sets(child, node)
            if ret is None or len(ret) == 1:   # not found here
                pass
            elif len(ret) == 5:
                T, Tprime, B, L, R = ret
                if isinstance(tree, Tree):
                    Tprime.append(tree)
                return tree, Tprime, B, L, R
            i += 1
        return None
    else:           # found --> return
        del tree[:]
        return tree, Tprime, B, L, A


def filter_graph(nodes:List[Tree], edges:List[Tuple], trees:Union[Tree, Iterable[Tree]]):
    """
    :param nodes:   List of nodes with nodeids
    :param edges:   List of edges expressed using nodeids
    :param trees:   Node(s) (with nodeids from the same space as 'nodes')
    :return: list of nodes and edges that are contained in the subgraph covered by nodes specified in 'trees'
    """
    if isinstance(trees, Tree):
        trees = [trees]

    # collect all nodes contained in 'trees'
    treenodes = []
    for tree in trees:
        treenodes = treenodes + _gather_descendants(tree, _top=False)
    # and map them to nodeids
    treenodeids = set([tn.nodeid for tn in treenodes])

    # filter
    _nodes = []
    _edges = []

    for node in nodes:
        if node.nodeid in treenodeids:
            _nodes.append(node)

    for edge in edges:
        if edge[0] in treenodeids and edge[1] in treenodeids:
            _edges.append(edge)

    return _nodes, _edges


def test_tree_oracle(x="x"):
    # treestr = "(A ( B (C E F) (D G H)))"
    # treestr = "(A ( B (C (E (F (X (I J))))) D ))"
    treestr = "(1 (11 (111 1111 1112 1113) (112 1121 1122 1123) (113 1131 1132 1133)) (12 (121 1211 1212 1213) (122 1221 (1222 (12222 (122222 1222222))) 1223) (123 1231 1232 1233)) (13 (131 1311 1312 1313) (132 1321 1322 1323) (133 1331 1332 1333)))"
    tree = lisp_to_tree(treestr)
    print(tree)
    print(" ".join(tree_to_seq_special(tree)))

    tree = assign_dfs_nodeids(tree)
    queue = [tree]
    while len(queue) > 0:
        xe = queue.pop(0)
        print(f"{xe.nodeid}: {xe}")
        queue = xe[:] + queue

    nodes, edges = build_graph(tree)
    for i, n in enumerate(nodes):
        print(f"{i}: {n}")
    print("\n".join([str((str(n[0]), str(n[1]), n[2])) for n in edges]))

    distances = compute_distances_graph(nodes, edges)
    centralities = compute_centralities(distances)

    # distances = compute_distances_tree(tree)
    # print(json.dumps(distances, indent=3))
    #
    # centralities = compute_centralities(distances)

    print(json.dumps(centralities, indent=3))

    bestnode = None
    bestcentrality = 0
    queue = [tree]
    while len(queue) > 0:
        node = queue.pop(0)
        print(node)
        print(centralities[node.nodeid])
        if centralities[node.nodeid] > bestcentrality:
            bestnode = node
            bestcentrality = centralities[node.nodeid]
        queue = queue + node[:]
    print(f"best node: {bestnode}")

    _tree = deepcopy_tree(tree)
    T, Tprime, B, L, R = get_supervision_sets(tree, bestnode)
    print(T)
    print(Tprime)
    print(B)
    print(L)
    print(R)

    # filter nodes and edges using a set B(ottom) for example and rerun centralities etc
    _nodes, _edges = filter_graph(nodes, edges, B)
    distances = compute_distances_graph(_nodes, _edges)
    centralities = compute_centralities(distances)

    print(json.dumps(centralities, indent=3))

    bestnode = None
    bestcentrality = 0
    queue = [be for be in B]
    while len(queue) > 0:
        node = queue.pop(0)
        print(node)
        print(centralities[node.nodeid])
        if centralities[node.nodeid] > bestcentrality:
            bestnode = node
            bestcentrality = centralities[node.nodeid]
        queue = queue + node[:]
    print(f"best node: {bestnode}")

    Bcopy = [deepcopy_tree(be) for be in B]

    T, Tprime, _B, L, R = get_supervision_sets(B, bestnode)
    print(T)
    print(Tprime)
    print(_B)
    print(L)
    print(R)


    print(f"best node 2: {Bcopy[1][1]}")
    T, Tprime, _B, L, R = get_supervision_sets(Bcopy, Bcopy[1][1])
    print(T)
    print(Tprime)
    print(_B)
    print(L)
    print(R)

    tree = _tree
    # bestnode = tree[0]
    # T, Tprime, B, L, R = get_supervision_sets(tree, bestnode)
    # print(T)
    # print(Tprime)
    # print(B)
    # print(L)
    # print(R)


def sample_partial_tree(x:Tree):
    """ Takes a full tree and samples a partial tree.
        Root is always taken as part of the sampled partial tree.
    """
    try:
        # uniformly sample length
        xlen = tree_size(x)
        ylen = random.choice(list(range(xlen)))

        # select nodes
        nodes = _gather_descendants(x)
        random.shuffle(nodes)

        selectednodes = nodes[:ylen] + [x]
        selectednodeids = set([sn.nodeid for sn in selectednodes])

        # build partial tree
        ptree = build_partial_tree(x, selectednodeids)
        ptree = ptree[0]
        return ptree
    except PartialTreeImpossibleException as e:
        return sample_partial_tree(x)


class PartialTreeImpossibleException(Exception): pass


def build_partial_tree(tree, sel):
    children = []
    for k in tree:
        child = build_partial_tree(k, sel)
        children.append(child)
    if not (len(children) <= 1 or all([len(k) <= 1 for k in children])):
        raise PartialTreeImpossibleException()
    children = [child for k in children for child in k]
    if tree.nodeid in sel:
        ret = Tree(tree.label(), children=children)
        ret.nodeid = tree.nodeid
        return [ret]
    else:
        return children


def build_supervision_sets(ptree, tree):
    """
    Receives a partial tree 'ptree' and the original tree 'tree'.
    Has to compute the insertion sets (T(op), B(ottom), L(eft), R(ight)) for every node in 'ptree'.
    (! Single children should always have an empty T set)
    (! )
    """
    if len(ptree) == 0:
        ret = Tree(ptree.label(), [])
        ret.nodeid = ptree.nodeid
        ret.insert_ancestors = []
        assert ptree.nodeid == tree.nodeid
        ret.insert_descendants = _gather_descendants(tree)
        ret.insert_siblings = []
        return ret
    # find lowest common ancestor
    pchildren = ptree[:]
    pchildrenids = set([pk.nodeid for pk in pchildren])
    lca = _rec_find_lca(tree, pchildrenids)

    # build sets and do deeper recurrence
    bottom, ancestors, siblings, pchildren_ss = _rec_build_sets(tree, lca, pchildren)

    ret = Tree(ptree.label(), pchildren_ss)
    ret.nodeid = ptree.nodeid
    ret.insert_ancestors = []
    ret.insert_descendants = bottom
    ret.insert_siblings = siblings

    assert len(ptree) == len(ancestors)
    for child, child_ancestors in zip(pchildren_ss, ancestors):
        child.insert_ancestors = child_ancestors

    return ret


def _rec_build_sets(tree, lca, pchildren):
    if lca is not None:
        if tree is lca or tree.nodeid == lca.nodeid:    # 'tree' is lca --> gather
            _lca = None
            child_ancestors = []
            insert_siblings = [[]]
            pchildren_ss = []
            for k in tree:
                pchild = pchildren[0] if len(pchildren) > 0 else None
                found, l, pchild_ss = _rec_build_sets(k, None, pchild)        # try to find first pchild in this child
                if found is True:
                    pchildren.pop(0)
                    child_ancestors.append(l)
                    insert_siblings.append([])
                    pchildren_ss.append(pchild_ss)
                else:
                    insert_siblings[-1] += l
            assert len(pchildren) == 0, "not all children in partial tree found!"
            return [], child_ancestors, insert_siblings, pchildren_ss
        else:       # go deeper
            if len(tree) == 0:      # leaf, dead-end
                return False
            else:
                for k in tree:
                    ret = _rec_build_sets(k, lca, pchildren)
                    if ret is False:    # this whole branch is dead end, do nothing
                        pass
                    else:
                        push_down, child_ancestors, insert_siblings, pchildren_ss = ret
                        push_down = push_down + [k]
                        return push_down, child_ancestors, insert_siblings, pchildren_ss
                return False

    else:   # we reached lca above  --> look out for children
        pchild = pchildren
        if pchild is not None and tree.nodeid == pchild.nodeid:        # found child
            pchild_ss = build_supervision_sets(pchild, tree)
            return True, [], pchild_ss
        elif len(tree) == 0:        # no more children left
            return False, [tree], None
        else:                       # some children present
            nodes = []
            for k in tree:
                found, l, pchild_ss = _rec_build_sets(k, None, pchild)
                if found is True:
                    l = l + [tree]
                    return found, l, pchild_ss
                nodes = nodes + l
            nodes = nodes + [tree]
            return found, nodes, None


def _rec_find_lca(tree, pchildrenids):
    found = set()
    if len(pchildrenids) > 0 and tree.nodeid in pchildrenids:
        pass
        found.add(tree.nodeid)
        return found
    elif len(tree) > 0:
        for k in tree:
            ret = _rec_find_lca(k, pchildrenids)
            if isinstance(ret, Tree):
                return ret
            else:
                found = found | ret
        if found == pchildrenids:       # all children found
            return tree
        else:
            return found
    else:
        return found


class SpecialTokens():
    def __init__(self, **kw):
        super(SpecialTokens, self).__init__()
        for k, v in kw.items():
            setattr(self, k, v)


class DefaultSpecialTokens():
    def __init__(self,
                 root_token="@R@",
                 ancestor_slot="@^",
                 descendant_slot="@v",
                 sibling_slot="@--",
                 parent_separator="|",
                 opening_parentheses="(",
                 closing_parentheses=")",
                 keep_action="@KEEP@",
                 close_action="@CLOSE@"):
        super(DefaultSpecialTokens, self).__init__()
        self.root_token = root_token
        self.ancestor_slot = ancestor_slot
        self.descendant_slot = descendant_slot
        self.sibling_slot = sibling_slot
        self.parent_separator = parent_separator
        self.opening_parentheses = opening_parentheses
        self.closing_parentheses = closing_parentheses
        self.keep_action = keep_action
        self.close_action = close_action


def linearize_ptree(x, is_root=True, only_child=True, specialtokens=DefaultSpecialTokens()):
    if len(x) == 0:     # no children
        # ret = [open_parentheses, ancestor_slot, x.label(), descendant_slot, closed_parentheses]
        ret = [specialtokens.ancestor_slot, x, specialtokens.descendant_slot]
        if only_child:
            del ret[0]
        if is_root:
            ret = [specialtokens.opening_parentheses] + ret + [specialtokens.closing_parentheses]
    else:
        childrets = [specialtokens.sibling_slot]
        i = 0
        for k in x:
            ret = linearize_ptree(k, only_child=len(x) == 1, is_root=False, specialtokens=specialtokens)
            childrets = childrets + ret + [specialtokens.sibling_slot]
            i += 1
        ret = [specialtokens.opening_parentheses, specialtokens.ancestor_slot, x, specialtokens.descendant_slot, specialtokens.parent_separator] + childrets + [specialtokens.closing_parentheses]
        if only_child:
            del ret[1]
    return ret


def linearize_supervised_ptree(x, is_root=True, only_child=True, specialtokens=DefaultSpecialTokens()):
    if len(x) == 0:     # no children
        # assert len(x.insert_descendants) == 0
        assert len(x.insert_siblings) == 0
        # ret = [open_parentheses, ancestor_slot, x.label(), descendant_slot, closed_parentheses]
        ret = [specialtokens.ancestor_slot, x, specialtokens.descendant_slot]
        # ret_sup = [None, x.insert_ancestors, None, [], None]
        ret_sup = [x.insert_ancestors, None, x.insert_descendants]
        if only_child:
            assert len(ret_sup[0]) == 0
            del ret[0]
            del ret_sup[0]
        if is_root:
            ret = [specialtokens.opening_parentheses] + ret + [specialtokens.closing_parentheses]
            ret_sup = [None] + ret_sup + [None]
    else:
        childrets = [specialtokens.sibling_slot]
        i = 0
        childret_sups = [x.insert_siblings[i]]
        for k in x:
            ret, ret_sup = linearize_supervised_ptree(k, only_child=len(x) == 1, specialtokens=specialtokens)
            childrets = childrets + ret + [specialtokens.sibling_slot]
            i += 1
            childret_sups = childret_sups + ret_sup + [x.insert_siblings[i]]
        ret = [specialtokens.opening_parentheses, specialtokens.ancestor_slot, x, specialtokens.descendant_slot, specialtokens.parent_separator] + childrets + [specialtokens.closing_parentheses]
        ret_sup = [None, x.insert_ancestors, None, x.insert_descendants, None] + childret_sups + [None]
        if only_child:
            assert len(ret_sup[1]) == 0
            del ret[1]
            del ret_sup[1]
    return ret, ret_sup


def compute_centralities_ptree(ptree, tree):
    """
    :param ptree:   partial tree with nodeids as well as supervision sets
    :param tree:    original tree with the same nodeid space
    :return:        dictionary mapping nodeids to centralities
    """
    # 1. break up the graph of 'tree'
    #       by removing edges of nodes that are already in ptree
    #       as well as separating all children of LCA nodes with more than one child
    nodes, edges = build_graph(tree)
    break_edges = find_break_edges(ptree)


    break_nodeids = set()
    queue = [ptree]
    while len(queue) > 0:
        pchild = queue.pop(0)
        break_nodeids.add(pchild.nodeid)
        queue = pchild[:] + queue

    edges_ = []
    for edge in edges:
        if (edge[0], edge[1]) in break_edges:
            continue
        elif edge[0] in break_nodeids or edge[1] in break_nodeids:
            continue
        else:
            edges_.append(edge)

    # print(len(edges), len(edges_))
    # 2. compute distances
    distances = compute_distances_graph(nodes, edges_)
    # 3. compute centralities
    centralities = compute_centralities(distances)
    return centralities


def find_break_edges(ptree):
    """ Find edges which to remove from the graph for the original tree behind this ptree.
        ==> edges between adjac
    """
    ret = set()
    if len(ptree.insert_descendants) > 0:
        lca = ptree.insert_descendants[0]
        for lca_child in lca:
            ret.add((lca.nodeid, lca_child.nodeid))
            ret.add((lca_child.nodeid, lca.nodeid))

    ancestors_of_children = []
    for pchild in ptree:
        ancestors_of_children.append([pchild.insert_ancestors[-1]] if len(pchild.insert_ancestors) > 0 else [])
        ret_ = find_break_edges(pchild)
        ret |= ret_

    if len(ptree) > 1:
        s = [ptree.insert_siblings[0]]
        for i in range(len(ancestors_of_children)):
            s.append(ancestors_of_children[i])
            s.append(ptree.insert_siblings[i+1])
        s = [se for se in s if len(se) > 0]
        for i in range(len(s)-1):
            ret.add((s[i][-1].nodeid, s[i+1][0].nodeid))
            ret.add((s[i+1][0].nodeid, s[i][-1].nodeid))
    return ret


def _collect_lca_nodeids(tree, ptree):
    lcas = set()
    if len(ptree) > 1:
        pchildren = ptree[:]
        pchildrenids = set([pk.nodeid for pk in pchildren])
        lca = _rec_find_lca(tree, pchildrenids)
        lcas.add(lca.nodeid)
    for pchild in ptree:
        _lcas = _collect_lca_nodeids_2(tree, pchild)
        lcas = lcas | _lcas
    return lcas


def _collect_lca_nodeids_2(tree, pchild):
    lcas = set()
    if tree.nodeid == pchild.nodeid:
        _lcas = _collect_lca_nodeids(tree, pchild)
        lcas = lcas | _lcas
    else:
        for child in tree:
            _lcas = _collect_lca_nodeids_2(child, pchild)
            lcas = lcas | _lcas
    return lcas


def compute_target_distribution_data(x, centrs, end_token="@END@"):
    # compute ranks
    if len(x) > 0:
        sorted_retsupe = sorted([(centrs[e.nodeid], e) for e in x],
                                key=cmp_to_key(retsup_cmp))
        r = []
        rank = 0
        prev_e = sorted_retsupe[0]
        r.append((rank, prev_e[1].label()))
        for e in sorted_retsupe[1:]:
            if retsup_cmp(e, prev_e) != 0:
                assert retsup_cmp(e, prev_e) > 0
                rank += 1
            r.append((rank, e[1].label()))
            prev_e = e
    else:
        r = [(1., end_token)]
    # compute softmax over ranks
    tau = 1.
    d = sum([np.exp(-e[0] / tau) for e in r])
    r = [(np.exp(-e[0] / tau)/d, e[1]) for e in r]
    return r


def retsup_cmp(a:Tuple[float, Tree], b:Tuple[float, Tree]):
    """
    :param a, b:        tuple consisting of a float and a Tree: (float, Tree) where the float is a centrality score for the tree (higher is better)
    :return:
    """
    ret = b[0] - a[0]
    if ret == 0:
        ret = +1* (tree_size(a[1]) - tree_size(b[1]))
    # if ret == 0:
    #     al = a[1].label()
    #     bl = b[1].label()
    #     if al == bl:
    #         ret = 0
    #     elif al < bl:
    #         ret = -1
    #     else:
    #         ret = 1
    # if ret == 0:
    #     al = str(a[1])
    #     bl = str(b[1])
    #     if al == bl:
    #         ret = 0
    #     elif al < bl:
    #         ret = -1
    #     else:
    #         ret = 1
    return ret


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def perform_decoding_step(xseq:List[str], tagseq:List[str], specialtokens=DefaultSpecialTokens()):
    # build special tree where siblings slots are nodes and other nodes have annotations what slots are available and what is being inserted there
    assert len(xseq) == len(tagseq)

    class State():
        def __setattr__(self, key, value):
            self.__dict__[key] = value
        def __getattr__(self, key):
            return self.__dict__[key]

    state = State()
    state.parentnode = None
    state.curnode = None
    state.next_is_parent = False
    state.ancestor_insert = None
    state.node = None
    state.descendant_insert = None

    def close_node():
        if state.node is not None:
            state.node.ancestor_insert = state.ancestor_insert
            state.node.descendant_insert = state.descendant_insert
            state.curnode = state.node
            if state.parentnode is not None:
                state.parentnode.append(state.curnode)
                state.curnode.parent = state.parentnode
        reset_vars()

    def reset_vars():
        state.ancestor_insert = None
        state.descendant_insert = None
        state.node = None

    for xe, ye in zip(xseq, tagseq):
        if xe == specialtokens.opening_parentheses:
            # next_is_parent = True
            assert state.ancestor_insert is None
            assert state.node is None
            assert state.descendant_insert is None
            reset_vars()
        elif xe == specialtokens.closing_parentheses:     # closing a parent
            close_node()
            realchildren = [pn_child for pn_child in state.parentnode if pn_child.label() != specialtokens.sibling_slot]
            if len(realchildren) == 1:      # single child  --> preserve child's ancestral slot
                assert realchildren[0].ancestor_insert == None
                realchildren[0].ancestor_insert = specialtokens.keep_action
            # go up
            if state.parentnode.parent is not None:
                state.parentnode = state.parentnode.parent
            else:
                break
        elif xe == specialtokens.parent_separator:
            # parent itself complete --> proceed to children
            close_node()
            state.parentnode = state.curnode
        elif xe == specialtokens.ancestor_slot:
            close_node()
            state.ancestor_insert = ye
        elif xe == specialtokens.descendant_slot:
            state.descendant_insert = ye
        elif xe == specialtokens.sibling_slot:
            close_node()
            node = Tree(xe, [])
            node.insert_node = ye
            state.parentnode.append(node)
            node.parent = state.parentnode
        else:       # real token
            if state.node is not None:    # unfinished node is open
                close_node()
            state.node = Tree(xe, [])
            state.node.parent = None

    # execute annotated slotted tree
    executed_tree = _execute_slotted_tree(state.parentnode, specialtokens=specialtokens)
    assert len(executed_tree) == 1
    executed_tree = executed_tree[0]

    # linearize executed slotted tree to special slotted sequence of tokens
    outseq = _linearize_slotted_tree(executed_tree, specialtokens=specialtokens)

    return outseq


def _linearize_slotted_tree(x, specialtokens=DefaultSpecialTokens()):
    ownseq = []
    if x.label() == specialtokens.sibling_slot:
        ownseq.append(x.label())
    else:
        if x.ancestor_insert is not None:
            ownseq.append(specialtokens.ancestor_slot)
        ownseq.append(x.label())
        if x.descendant_insert is not None:
            ownseq.append(specialtokens.descendant_slot)

    if len(x) > 0:
        seq = [specialtokens.opening_parentheses] + ownseq + [specialtokens.parent_separator]

        for k in x:
            subseq = _linearize_slotted_tree(k, specialtokens=specialtokens)
            seq = seq + subseq

        seq = seq + [specialtokens.closing_parentheses]
    else:
        seq = ownseq
    return seq


def _execute_slotted_tree(x, specialtokens=DefaultSpecialTokens()):
    if x.label() == specialtokens.sibling_slot:
        if x.insert_node == specialtokens.keep_action:
            node = Tree(x.label(), [])
            return [node]
        elif x.insert_node == specialtokens.close_action or x.insert_node is None:
            return []
        else:
            node = Tree(x.insert_node, [])
            node.ancestor_insert = specialtokens.keep_action
            node.descendant_insert = specialtokens.keep_action
            ret = [Tree(specialtokens.sibling_slot, []), node, Tree(specialtokens.sibling_slot, [])]
            return ret
    else:
        node = Tree(x.label(), [])
        node.ancestor_insert = specialtokens.keep_action
        node.descendant_insert = specialtokens.keep_action
        ancestor = node

        if x.ancestor_insert == specialtokens.keep_action:
            node.ancestor_insert = specialtokens.keep_action
        elif x.ancestor_insert == specialtokens.close_action or x.ancestor_insert is None:
            node.ancestor_insert = None
        else:
            ancestor = Tree(x.ancestor_insert, [Tree(specialtokens.sibling_slot, []),
                                                node,
                                                Tree(specialtokens.sibling_slot, [])])
            node.ancestor_insert = None
            ancestor.ancestor_insert = specialtokens.keep_action
            ancestor.descendant_insert = specialtokens.keep_action

        if x.descendant_insert == specialtokens.keep_action:
            node.descendant_insert = specialtokens.keep_action
        elif x.descendant_insert == specialtokens.close_action or x.descendant_insert is None:
            node.descendant_insert = None
        else:
            descendant = Tree(x.descendant_insert, [])
            descendant.ancestor_insert = None
            descendant.descendant_insert = specialtokens.keep_action
            node.append(Tree(specialtokens.sibling_slot, []))
            node.append(descendant)
            node.append(Tree(specialtokens.sibling_slot, []))
            node = descendant

        for child in x:
            childrets = _execute_slotted_tree(child, specialtokens=specialtokens)
            for childret in childrets:
                node.append(childret)
        return [ancestor]


def test_decode(n=1):

    # test basic:
    print("basic test")
    r = Tree("R", [Tree("B", [])])
    rl = [re.label() if isinstance(re, Tree) else re for re in linearize_ptree(r)]
    r_str = " ".join(rl)
    print(r_str)
    rl_tags = [None, None, "D", None, "E", None, "G", "H", None]
    print(rl_tags)
    rl_tp1 = perform_decoding_step(rl, rl_tags)
    print(" ".join(rl_tp1))
    assert " ".join(rl_tp1) == "( R @v | @-- ( D @v | @-- @^ E @v @-- ( @^ B @v | @-- G @v @-- ) @-- @^ H @v @-- ) @-- )"


    # test different slots
    print("test with different slots")
    r = Tree("R", [Tree("A", [Tree("C", [])]), Tree("B", [])])
    rl = [re.label() if isinstance(re, Tree) else re for re in linearize_ptree(r)]
    r_str = " ".join(rl)
    print(r_str)
    #            (    R    @v     |   @--    (   @^     A   @v    |    @--    C   @v    @--  )    @--   @^    B   @v   @--   )
    rl_tags = [None, None, "D", None, "E", None, "F", None, "G", None, "H", None, "I", "J", None, "K", "L", None, "M", "N", None]
    print(rl_tags)
    rl_tp1 = perform_decoding_step(rl, rl_tags)
    print(" ".join(rl_tp1))
    assert " ".join(rl_tp1) == "( R @v | @-- ( D @v | @-- @^ E @v @-- ( @^ F @v | @-- ( A @v | @-- ( G @v | @-- @^ H @v @-- ( @^ C @v | @-- I @v @-- ) @-- @^ J @v @-- ) @-- ) @-- ) @-- @^ K @v @-- ( @^ L @v | @-- ( B @v | @-- M @v @-- ) @-- ) @-- @^ N @v @-- ) @-- )"
                           #"( R @v | @-- ( D @v | @-- @^ E @v @-- ( @^ F @v | @-- ( @^ A @v | @-- ( G @v | @-- @^ H @v @-- ( @^ C @v | @-- I @v @-- ) @-- @^ J @v @-- ) @-- ) @-- ) @-- @^ K @v @-- ( @^ L @v | @-- ( @^ B @v | @-- M @v @-- ) @-- ) @-- @^ N @v @-- ) @-- )
                              #"( R @v | @-- ( D @v | @-- @^ E @v @-- ( F @v | @-- ( @^ A @v | @-- ( G @v | @-- @^ H @v @-- ( @^ C @v | @-- I @v @-- ) @-- @^ J @v @-- ) @-- ) @-- ) @-- @^ K @v @-- ( L @v | @-- ( @^ B @v | @-- M @v @-- ) @-- ) @-- @^ N @v @-- ) @-- )"
    # test with closed slots I
    print("test with closed slots I")
    # r = Tree("R", [Tree("A", [Tree("C", [])]), Tree("B", [])])
    # rl = [re.label() if isinstance(re, Tree) else re for re in linearize_ptree(r)]
    r_str = "( R @v | ( A @v | @-- B C @v @-- ) @-- )"
    print(r_str)
    rl = r_str.split(" ")
    #           (      R    @v    |    (     A    @v     |   @--    B     C   @v   @--    )   @--  )
    rl_tags = [None, None, "D", None, None, None, "E", None, "F", None, None, "G", "H", None, "I", None]
    print(rl_tags)
    rl_tp1 = perform_decoding_step(rl, rl_tags)
    print(" ".join(rl_tp1))
    assert " ".join(
        rl_tp1) == "( R @v | @-- ( D @v | ( @^ A @v | @-- ( E @v | @-- @^ F @v @-- B ( C @v | @-- G @v @-- ) @-- @^ H @v @-- ) @-- ) @-- @^ I @v @-- ) @-- )"


def test_tree_sampling_random(n=100):
    # treestr = "(A ( B (C E F) K (D (G H I J))))"
    # treestr = "(A ( B (C (E (F (X (I J))))) D ))"
    treestr = "(1 (11 (111 1111 1112 1113) (112 1121 1122 1123) (113 1131 1132 1133)) (12 (121 1211 1212 1213) (122 1221 (1222 (12222 (122222 1222222))) 1223) (123 1231 1232 1233)) (13 (131 1311 1312 1313) (132 1321 1322 1323) (133 1331 1332 1333)))"
    tree = lisp_to_tree(treestr)
    print(tree)
    print(" ".join(tree_to_seq_special(tree)))

    tree = assign_dfs_nodeids(tree)

    for _ in range(n):
        print("\n")
        ptree = sample_partial_tree(tree)

        print(ptree)

        ptree_ret = build_supervision_sets(ptree, tree)

        ret, retsup = linearize_supervised_ptree(ptree_ret)

        print(" ".join(ret))

        for rete, retsupe in zip(ret, retsup):
            if retsupe != None:
                supstr = ", ".join([str(e) for e in retsupe])
                print(f"{rete}: [{supstr}]")
            else:
                print(f"{rete}")


def test_tree_sampling(x="x"):
    # treestr = "(R (T X (A ( B (C E F) (D (G H I J)) N))))"
    treestr = "(A ( B (C E F) K (D (G H I J))))"
    # treestr = "(A ( B (C (E (F (X (I J))))) D ))"
    # treestr = "(1 (11 (111 1111 1112 1113) (112 1121 1122 1123) (113 1131 1132 1133)) (12 (121 1211 1212 1213) (122 1221 (1222 (12222 (122222 1222222))) 1223) (123 1231 1232 1233)) (13 (131 1311 1312 1313) (132 1321 1322 1323) (133 1331 1332 1333)))"
    tree = lisp_to_tree(treestr)
    print(tree)
    print(" ".join(tree_to_seq_special(tree)))

    tree = assign_f_nodeids(tree, lambda x: ord(x.label()))

    ptree = sample_partial_tree(tree)

    print(ptree)


    print("\n")
    # ptree = Tree("R", children=[Tree("D", [])])       # ptree = Tree("R", children=[Tree("I", [])])
    ptree = Tree("A", children=[Tree("E", []), Tree("G", [])])
    print(f"Ptree: {ptree}")
    ptree = assign_f_nodeids(ptree, lambda x: ord(x.label()))

    ptree_ret = build_supervision_sets(ptree, tree)

    ret, retsup = linearize_supervised_ptree(ptree_ret)

    centralities = compute_centralities_ptree(ptree_ret, tree)

    print(" ".join([e.label() if isinstance(e, Tree) else e for e in ret]))

    for rete, retsupe in zip(ret, retsup):
        if retsupe != None:
            r = compute_target_distribution_data(retsupe, centralities)

            supstr = ", ".join([f"{c:.3f}:{e}" for c, e in r])
            print(f"{rete}: [{supstr}]")
        else:
            print(f"{rete.label() if isinstance(rete, Tree) else rete}")


def test_tree_sampling_(x="x"):
    treestr = "(A ( B (C E F) K (D (G H I J))))"
    # treestr = "(A ( B (C (E (F (X (I J))))) D ))"
    # treestr = "(1 (11 (111 1111 1112 1113) (112 1121 1122 1123) (113 1131 1132 1133)) (12 (121 1211 1212 1213) (122 1221 (1222 (12222 (122222 1222222))) 1223) (123 1231 1232 1233)) (13 (131 1311 1312 1313) (132 1321 1322 1323) (133 1331 1332 1333)))"
    tree = lisp_to_tree(treestr)
    print(tree)
    print(" ".join(tree_to_seq_special(tree)))

    tree = assign_dfs_nodeids(tree)

    ptree = sample_partial_tree(tree)

    print(ptree)


    print("\n")
    ptree = Tree("A", children=[Tree("E", []), Tree("I", [])])
    print(f"Ptree: {ptree}")
    ptree.nodeid = 0
    ptree[0].nodeid = 3
    ptree[1].nodeid = 9
    ptree_ret = build_supervision_sets(ptree, tree)

    ret, retsup = linearize_supervised_ptree(ptree_ret)
    compute_centralities_ptree(ptree_ret, tree)

    print(" ".join(ret))

    for rete, retsupe in zip(ret, retsup):
        if retsupe != None:
            supstr = ", ".join([str(e) for e in retsupe])
            print(f"{rete}: [{supstr}]")
        else:
            print(f"{rete}")

    print("\n")
    ptree = Tree("A", children=[Tree("I", []), Tree("J", [])])
    print(f"Ptree: {ptree}")
    ptree.nodeid = 0
    ptree[0].nodeid = 9
    ptree[1].nodeid = 10
    ptree_ret = build_supervision_sets(ptree, tree)

    ret, retsup = linearize_supervised_ptree(ptree_ret)

    print(" ".join(ret))

    for rete, retsupe in zip(ret, retsup):
        if retsupe != None:
            supstr = ", ".join([str(e) for e in retsupe])
            print(f"{rete}: [{supstr}]")
        else:
            print(f"{rete}")

    print("\n")
    ptree = Tree("A", children=[Tree("I", [])])
    print(f"Ptree: {ptree}")
    ptree.nodeid = 0
    ptree[0].nodeid = 9
    ptree_ret = build_supervision_sets(ptree, tree)

    ret, retsup = linearize_supervised_ptree(ptree_ret)

    print(" ".join(ret))

    for rete, retsupe in zip(ret, retsup):
        if retsupe != None:
            supstr = ", ".join([str(e) for e in retsupe])
            print(f"{rete}: [{supstr}]")
        else:
            print(f"{rete}")

    print("\n")
    ptree = Tree("A", children=[Tree("E", []), Tree("D", [Tree("I", [])])])
    print(f"Ptree: {ptree}")
    ptree.nodeid = 0
    ptree[0].nodeid = 3
    ptree[1].nodeid = 6
    ptree[1][0].nodeid = 9
    ptree_ret = build_supervision_sets(ptree, tree)

    ret, retsup = linearize_supervised_ptree(ptree_ret)


    print(" ".join(ret))

    for rete, retsupe in zip(ret, retsup):
        if retsupe != None:
            supstr = ", ".join([str(e) for e in retsupe])
            print(f"{rete}: [{supstr}]")
        else:
            print(f"{rete}")


    # TODO: compute centralities within supervision segments

    # TODO: compute weights from centralities <-- rank nodes, use softmax with temp

    # TODO: generate tensors for input and target distributions











def tree_to_seq(x:Tree):
    xstr = tree_to_lisp_tokens(x)
    # xstr = ["@BOS@"] + xstr + ["@EOS@"]
    return xstr


def make_numbered_tokens(x:List[str]):
    counts = {}
    y = []
    for xe in x:
        if xe not in counts:
            counts[xe] = 0
        counts[xe] += 1
        y.append(f"{xe}::{counts[xe]}")
    return y


def load_ds(domain="restaurants", nl_mode="bert-base-uncased",
            trainonvalid=False, noreorder=False, numbered=False):
    """
    Creates a dataset of examples which have
    * NL question and tensor
    * original FL tree
    * reduced FL tree with slots (this is randomly generated)
    * tensor corresponding to reduced FL tree with slots
    * mask specifying which elements in reduced FL tree are terminated
    * 2D gold that specifies whether a token/action is in gold for every position (compatibility with MML!)
    """
    # orderless = {"op:and", "SW:concat"}     # only use in eval!!
    orderless = ORDERLESS

    ds = OvernightDatasetLoader(simplify_mode="none").load(domain=domain, trainonvalid=trainonvalid)
    # ds contains 3-tuples of (input, output tree, split name)

    if not noreorder:
        ds = ds.map(lambda x: (x[0], reorder_tree(x[1], orderless=orderless), x[2]))
    ds = ds.map(lambda x: (x[0], tree_to_seq(x[1]), x[2]))

    if numbered:
        ds = ds.map(lambda x: (x[0], make_numbered_tokens(x[1]), x[2]))

    vocab = Vocab(padid=0, startid=2, endid=3, unkid=1)
    vocab.add_token("@BOS@", seen=np.infty)
    vocab.add_token("@EOS@", seen=np.infty)
    vocab.add_token("@STOP@", seen=np.infty)

    nl_tokenizer = BertTokenizer.from_pretrained(nl_mode)

    tds, vds, xds = ds[lambda x: x[2] == "train"], \
                    ds[lambda x: x[2] == "valid"], \
                    ds[lambda x: x[2] == "test"]

    seqenc = SequenceEncoder(vocab=vocab, tokenizer=lambda x: x,
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
        seq = seqenc.convert(x[1], return_what="tensor")
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0], seq)
        return ret

    tds_seq = tds.map(mapper)
    vds_seq = vds.map(mapper)
    xds_seq = xds.map(mapper)
    return tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc, orderless


class SeqInsertionTagger(torch.nn.Module):
    """ A tree insertion tagging model takes a sequence representing a tree
        and produces distributions over tree modification actions for every (non-terminated) token.
    """
    @abstractmethod
    def forward(self, tokens:torch.Tensor, **kw):
        """
        :param tokens:      (batsize, seqlen)       # all are open!
        :return:
        """
        pass


class TransformerTagger(SeqInsertionTagger):
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", baseline=False, **kw):
        super(TransformerTagger, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.baseline = baseline
        config = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                   num_layers=numlayers, num_heads=numheads, dropout_rate=dropout,
                                   use_relative_position=False)

        self.emb = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.posemb = torch.nn.Embedding(maxpos, config.d_model)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = baseline
        self.decoder = TransformerStack(decoder_config)

        if baseline:
            self.out = torch.nn.Linear(self.dim, self.vocabsize)
        else:
            self.out = torch.nn.Linear(self.dim * 2, self.vocabsize)
        # self.out = MOS(self.dim, self.vocabsize, K=mosk)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname)
        # def set_dropout(m:torch.nn.Module):
        #     if isinstance(m, torch.nn.Dropout):
        #         m.p = dropout
        # self.bert_model.apply(set_dropout)

        self.adapter = None
        if self.bert_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.bert_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
        encs = self.bert_model(x)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
        padmask = (tokens != 0)
        # if not self.baseline:
        #     padmask = padmask[:, 1:]
        embs = self.emb(tokens)
        posembs = self.posemb(torch.arange(tokens.size(1), device=tokens.device))[None]
        embs = embs + posembs
        use_cache = False
        if self.baseline:
            use_cache = True
        if cache is not None:
            embs = embs[:, -1:, :]
        _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=use_cache,
                            past_key_value_states=cache)
        ret = _ret[0]
        cache = None
        if self.baseline:
            c = ret
            cache = _ret[1]
        else:
            c = torch.cat([ret[:, 1:], ret[:, :-1]], -1)
        logits = self.out(c)
        # logits = logits + torch.log(self.vocab_mask[None, None, :])
        if self.baseline:
            return logits, cache
        else:
            return logits
        # probs = self.out(ret[0], self.vocab_mask[None, None, :])
        # return probs


class SeqInsertionDecoder(torch.nn.Module):
    # default_termination_mode = "slot"
    # default_decode_mode = "parallel"

    def __init__(self, tagger:SeqInsertionTagger,
                 vocab=None,
                 prob_threshold=0.,
                 max_steps:int=20,
                 max_size:int=100,
                 end_offset=0.,
                 **kw):
        super(SeqInsertionDecoder, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_steps = max_steps
        self.max_size = max_size
        self.kldiv = torch.nn.KLDivLoss(reduction="none")
        self.logsm = torch.nn.LogSoftmax(-1)
        self.prob_threshold = prob_threshold
        self.end_offset = end_offset

        # self.termination_mode = self.default_termination_mode if termination_mode == "default" else termination_mode
        # self.decode_mode = self.default_decode_mode if decode_mode == "default" else decode_mode

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

    @abstractmethod
    def extract_training_example(self, x, y):
        pass

    def compute_loss(self, logits, tgt, mask=None):
        """
        :param logits:      (batsize, seqlen, vocsize)
        :param tgt:         (batsize, seqlen, vocsize)
        :param mask:        (batsize, seqlen)
        :return:
        """
        logprobs = self.logsm(logits)
        kl = self.kldiv(logprobs, tgt)      # (batsize, seqlen, vocsize)
        kl = kl.sum(-1)                     # (batsize, seqlen)
        if mask is not None:
            kl = kl * mask
        kl = kl.sum(-1)
        return kl

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits = self.tagger(tokens=newy, enc=enc, encmask=encmask)
        # compute loss: different versions do different masking and different targets
        loss = self.compute_loss(logits, tgt[:, :-1], mask=tgtmask[:, :-1])
        return {"loss": loss}, logits

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.zeros(x.size(0), 2, device=x.device, dtype=torch.long)
        y[:, 0] = self.vocab["@BOS@"]
        y[:, 1] = self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.tagger.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        while step < self.max_steps and not torch.all(ended): #(newy is None or not (y.size() == newy.size() and torch.all(y == newy))):
            y = newy if newy is not None else y
            # run tagger
            logits = self.tagger(tokens=y, enc=enc, encmask=encmask)
            # logprobs = torch.log_softmax(logits, -1)
            # logprobs[:, :, self.vocab["@END@"]] = logprobs[:, :, self.vocab["@END@"]] - self.end_offset
            # probs = torch.exp(logprobs)
            probs = torch.softmax(logits, -1)
            probs[:, :, self.vocab["@END@"]] = probs[:, :, self.vocab["@END@"]] - self.end_offset
            predprobs, preds = probs.max(-1)
            predprobs, preds = predprobs.cpu().detach().numpy(), preds.cpu().detach().numpy()
            _y = y.cpu().detach().numpy()
            newy = torch.zeros(y.size(0), min(self.max_size, y.size(1) * 2), device=y.device, dtype=torch.long)
            _ended = torch.zeros_like(y[:, 0]).bool()
            # update sequences
            for i in range(len(y)):
                k = 0
                p_i = preds[i]
                pp_mask = p_i != self.vocab["@END@"]
                pp_i = predprobs[i] * pp_mask
                pp_mask = _y[i] == self.vocab["@EOS@"]
                pp_mask = np.cumsum(pp_mask, -1)
                pp_i = pp_i * (pp_mask[:-1] == 0)
                # pp_i = (pp_i > 0) * np.random.rand(*pp_i.shape)
                prob_thresh = min(self.prob_threshold, max(pp_i))
                terminated = True
                for j in range(len(y[i])):          # loop, advance j = j+1
                    if k >= newy.size(1):
                        break
                    newy[i, k] = y[i, j]            # copy from previous target sequence
                    k += 1                          # advance newy pointer to next position
                    y_ij = _y[i, j]
                    if y_ij == self.vocab["@EOS@"]: # if token was EOS, terminate generation
                        break  # stop
                    if j >= len(p_i):               # if we reached beyond the length of predictions, terminate
                        break
                    p_ij = p_i[j]                   # token predicted between j-th and j+1-st position in previous sequence
                    pp_ij = pp_i[j]                 # probability assigned to that token
                    if pp_ij >= prob_thresh:        # skip if probability was lower than threshold
                        if p_ij == self.vocab["@END@"]:         # if predicted token is @END@, do nothing
                            pass  # don't insert anything
                        else:  # insert what was predicted
                            if k >= newy.size(1):
                                break
                            newy[i, k] = preds[i, j]
                            k += 1                  # advance newy pointer to next position
                            terminated = False      # sequence changed so don't terminate
                _ended[i] = terminated

            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)  # prevent terminated examples from changing

            maxlen = (newy != 0).long().sum(-1).max()
            newy = newy[:, :maxlen]
            step += 1
            ended = ended | _ended
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
            lens = (newy != 0).long().sum(-1)
            maxlenreached = (lens == self.max_size)
            if torch.all(maxlenreached):
                break
        return newy, steps_used.float()

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        preds, stepsused = self.get_prediction(x)

        def tensor_to_trees(x, vocab:Vocab):
            xstrs = [vocab.tostr(x[i]).replace("@BOS@", "").replace("@EOS@", "") for i in range(len(x))]
            xstrs = [re.sub("::\d+", "", xstr) for xstr in xstrs]
            trees = []
            for xstr in xstrs:
                # drop everything after @END@, if present
                xstr = xstr.split("@END@")
                xstr = xstr[0]
                # add an opening parentheses if not there
                xstr = xstr.strip()
                if len(xstr) == 0 or xstr[0] != "(":
                    xstr = "(" + xstr
                # balance closing parentheses
                parenthese_imbalance = xstr.count("(") - xstr.count(")")
                xstr = xstr + ")" * max(0, parenthese_imbalance)        # append missing closing parentheses
                xstr = "(" * -min(0, parenthese_imbalance) + xstr       # prepend missing opening parentheses
                try:
                    tree = lisp_to_tree(xstr)
                    if isinstance(tree, tuple) and len(tree) == 2 and tree[0] is None:
                        tree = None
                except Exception as e:
                    tree = None
                trees.append(tree)
            return trees

        # compute loss and metrics
        gold_trees = tensor_to_trees(gold, vocab=self.vocab)
        pred_trees = tensor_to_trees(preds, vocab=self.vocab)
        treeaccs = [float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device), "stepsused": stepsused}
        return ret, pred_trees


class SeqInsertionDecoderUniform(SeqInsertionDecoder):
    # decode modes: "parallel", "serial" or "semiparallel":
    #    - parallel: execute actions at all slots simultaneously
    #           --> prob threshold = 0.
    #    - serial: execute action with highest probability across all slots, unless the action is an @END@
    #           --> prob threshold > 1.
    #    - semiparallel: execute actions for all slots where the highest probability is above a certain threshold,
    #                    unless the action is an @END@.
    #                    if there are no slots with highest probability higher than threshold, fall back to serial mode for this decoding step
    #           --> prob threshold between 0. and 1.
    # --> all modes naturally terminate once all slots predict an @END@
    # default_termination_mode = "slot"
    # default_decode_mode = "parallel"

    def get_slot_value_probs(self, slotvalues):     # uniform
        # uniformly distribute probability over unique tokens
        # then distribute uniformly over every position a token occurs in
        # example: A B B C D --> [0.25, 0.125, 0.125, 0.25, 0.25]
        # this way, when tgt is accumulated in training code, the distribution over tokens will be uniform
        prob = 1./len(set(slotvalues))
        token_freqs = {}
        for slotvalue in slotvalues:
            if slotvalue not in token_freqs:
                token_freqs[slotvalue] = 0
            token_freqs[slotvalue] += 1
        probs = [prob/token_freqs[slotvalue] for slotvalue in slotvalues]
        return probs

    def extract_training_example(self, x: torch.Tensor, y: torch.Tensor):
        # y: (batsize, seqlen) ids, padded with zeros
        ymask = (y != 0).float()
        ytotallens = ymask.sum(1)
        ylens = torch.rand(ytotallens.size(), device=ytotallens.device)
        ylens = (ylens * ytotallens).round().long()
        _ylens = ylens.cpu().numpy()
        # ylens contains the sampled lengths

        # for LTR: take 'ylens' leftmost tokens
        # for Uniform/Binary: randomly select 'ylens' tokens
        newy = torch.zeros(y.size(0), y.size(1) + 2, device=y.device).long()
        newy[:, 0] = self.vocab["@BOS@"]
        tgt = torch.zeros(y.size(0), y.size(1) + 2, self.vocab.number_of_ids(), device=y.device)
        # 'tgt' contains target distributions
        for i in range(newy.size(0)):
            perm = torch.randperm(ytotallens[i].long().cpu().item())
            perm = perm[:ylens[i].long().cpu().item()]
            select, _ = perm.sort(-1)
            select = list(select.cpu().numpy())
            k = 1  # k is where in the new sampled sequence we're at

            slotvalues_acc = []
            slotvalues = []
            for j in range(int(ytotallens[i].cpu().item())):
                y_ij = y[i, j].cpu().item()
                if k <= len(select) and j == select[k - 1]:  # if j-th token in y should be k-th in newy
                    newy[i, k] = y[i, j]
                    slotvalues_acc.append(slotvalues)
                    slotvalues = []
                    k += 1
                else:  # otherwise, add
                    slotvalues.append(y_ij)
                    # tgt[i, k - 1, y_ij] = 1
            slotvalues_acc.append(slotvalues)
            newy[i, k] = self.vocab["@EOS@"]

            for j, slotvalues in enumerate(slotvalues_acc):
                if len(slotvalues) == 0:
                    tgt[i, j, self.vocab["@END@"]] = 1
                else:
                    for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
                        tgt[i, j, slotvalue] += float(valueprob)

        # normalize
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-6)
        tgtmask = (tgt.sum(-1) != 0).float()
        # make uniform for masked positions
        newymask = (newy != 0).float()
        uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
        tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
        # cut unnecessary padded elements from the right of newy
        newlen = newymask.sum(-1).max()
        newy = newy[:, :int(newlen)]
        tgt = tgt[:, :int(newlen)]
        tgtmask = tgtmask[:, :int(newlen)]

        return x, newy, tgt, tgtmask


class SeqInsertionDecoderBinary(SeqInsertionDecoderUniform):
    """ Differs from Uniform only in computing and using non-uniform weights for gold output distributions """
    def __init__(self, tagger:SeqInsertionTagger,
                 vocab=None,
                 prob_threshold=0.,
                 max_steps:int=20,
                 max_size:int=100,
                 tau=1.,
                 **kw):
        super(SeqInsertionDecoderBinary, self).__init__(tagger, vocab=vocab,
                                                        max_steps=max_steps,
                                                        max_size=max_size,
                                                        prob_threshold=prob_threshold,
                                                        **kw)
        self.tau = tau

    def get_slot_value_probs(self, slotvalues):
        # assign higher probability to tokens closer to centre
        # when multiple tokens of the same type are present: keep score of the closest one to center
        # set distance for all other ones of the same token to infinity
        center = len(slotvalues) / 2 - 0.5
        distances = [abs(x - center) for x in range(len(slotvalues))]
        mindist_per_token = {}
        for slotvalue, distance in zip(slotvalues, distances):
            if slotvalue not in mindist_per_token:
                mindist_per_token[slotvalue] = +9999
            mindist_per_token[slotvalue] = min(mindist_per_token[slotvalue], distance)
        mindistances = [d for d in distances]
        for i, (slotvalue, distance) in enumerate(zip(slotvalues, distances)):
            if distance == mindist_per_token[slotvalue]:
                mindistances[i] = distance
            else:
                mindistances[i] = +99999

        probs = torch.softmax(-torch.tensor(mindistances)/self.tau, -1)
        probs = probs.numpy()
        return probs


def get_slotvalues_maxspan(selected, yi):
    if len(selected) == 0:
        ret = [list(yi.cpu().detach().numpy())]
        return ret
    middle_j = int(len(selected)/2)
    yilen = (yi > 0).sum()
    yi = yi[:yilen]
    middle_k = int(len(yi)/2)
    left_selected, right_selected = selected[:middle_j], selected[middle_j+1:]
    # search for the midmost element in selected in yi
    yi_left, yi_right = None, None
    foundit = 0

    # compute earliest possible position for the middle element of selected
    _left_selected = list(left_selected.cpu().detach().numpy())
    _right_selected = list(right_selected.cpu().detach().numpy())
    _yi = list(yi.cpu().detach().numpy())

    i = 0
    for e in _left_selected:
        while e != _yi[i]:
            i += 1
        i += 1
    earliest_pos = i

    _yi = _yi[::-1]
    i = 0
    for e in _right_selected[::-1]:
        while e != _yi[i]:
            i += 1
        i += 1
    latest_pos = len(_yi) - i

    # compute latest possible position for the middle
    for l in range(math.ceil(len(yi)/2)+1):
        foundit = 0
        if len(yi) == 0 or len(selected) == 0:
            print("something wrong")
        if middle_k - l >= earliest_pos and middle_k - l < latest_pos and yi[middle_k - l] == selected[middle_j]:
            foundit = -1
        elif middle_k + l >= earliest_pos and middle_k + l < latest_pos and yi[middle_k + l] == selected[middle_j]:
            foundit = +1

        if foundit != 0:
            splitpoint = middle_k + l*foundit
            yi_left, yi_right = yi[:splitpoint], yi[splitpoint + 1:]
            if len(left_selected) == 0:
                left_slotvalueses = [list(yi_left.cpu().detach().numpy())]
            else:
                left_slotvalueses = get_slotvalues_maxspan(left_selected, yi_left)

            if len(right_selected) == 0:
                right_slotvalueses = [list(yi_right.cpu().detach().numpy())]
            else:
                right_slotvalueses = get_slotvalues_maxspan(right_selected, yi_right)

            if left_slotvalueses is None or right_slotvalueses is None:
                continue

            ret = left_slotvalueses + right_slotvalueses
            return ret
    if foundit == 0:
        return None


class SeqInsertionDecoderMaxspanBinary(SeqInsertionDecoderBinary):
    def extract_training_example(self, x: torch.Tensor, y: torch.Tensor):
        # y: (batsize, seqlen) ids, padded with zeros
        ymask = (y != 0).float()
        ytotallens = ymask.sum(1)
        ylens = torch.rand(ytotallens.size(), device=ytotallens.device)
        ylens = (ylens * ytotallens).round().long()
        _ylens = ylens.cpu().numpy()
        # ylens contains the sampled lengths

        # for LTR: take 'ylens' leftmost tokens
        # for Uniform/Binary: randomly select 'ylens' tokens
        newy = torch.zeros(y.size(0), y.size(1) + 2, device=y.device).long()
        newy[:, 0] = self.vocab["@BOS@"]
        tgt = torch.zeros(y.size(0), y.size(1) + 2, self.vocab.number_of_ids(), device=y.device)
        # 'tgt' contains target distributions
        for i in range(newy.size(0)):
            perm = torch.randperm(ytotallens[i].long().cpu().item())
            perm = perm[:ylens[i].long().cpu().item()]
            select, _ = perm.sort(-1)
            select = list(select.cpu().numpy())
            selected = y[i][select]

            slotvalues_acc = get_slotvalues_maxspan(selected, y[i])

            # k = 1  # k is where in the new sampled sequence we're at
            #
            # slotvalues_acc = []
            # slotvalues = []
            # for j in range(int(ytotallens[i].cpu().item())):
            #     y_ij = y[i, j].cpu().item()
            #     if k <= len(select) and j == select[k - 1]:  # if j-th token in y should be k-th in newy
            #         newy[i, k] = y[i, j]
            #         slotvalues_acc.append(slotvalues)
            #         slotvalues = []
            #         k += 1
            #     else:  # otherwise, add
            #         slotvalues.append(y_ij)
            #         # tgt[i, k - 1, y_ij] = 1
            # slotvalues_acc.append(slotvalues)
            k = 1
            j = len(slotvalues_acc[0])
            for slotvalues in slotvalues_acc[1:]:
                newy[i, k] = y[i, j]
                k += 1
                j += len(slotvalues) + 1
            newy[i, k] = self.vocab["@EOS@"]

            for j, slotvalues in enumerate(slotvalues_acc):
                if len(slotvalues) == 0:
                    tgt[i, j, self.vocab["@END@"]] = 1
                else:
                    for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
                        tgt[i, j, slotvalue] += float(valueprob)

        # normalize
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-6)
        tgtmask = (tgt.sum(-1) != 0).float()
        # make uniform for masked positions
        newymask = (newy != 0).float()
        uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
        tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
        # cut unnecessary padded elements from the right of newy
        newlen = newymask.sum(-1).max()
        newy = newy[:, :int(newlen)]
        tgt = tgt[:, :int(newlen)]
        tgtmask = tgtmask[:, :int(newlen)]

        return x, newy, tgt, tgtmask


class SeqInsertionDecoderAny(SeqInsertionDecoderUniform):
    def get_slot_value_probs(self, slotvalues):     # uniform
        probs = [1. for _ in slotvalues]
        return probs

    def compute_loss(self, logits, tgt, mask=None):
        """
        :param logits:
        :param tgt:         will have a non-zero for every correct token
        :param mask:        will be zero for positions that don't need insertion
        :return:
        """
        probs = torch.softmax(logits, -1)
        nonzero_tgt = (tgt > 0).float()
        m = probs * nonzero_tgt
        m = m.sum(-1)       # (batsize, seqlen)
        loss = - torch.log(m.clamp_min(1e-6))
        if mask is not None:
            loss = loss * mask.float()
        loss = loss.sum(-1)
        return loss


class SeqInsertionDecoderPredictiveBinary(SeqInsertionDecoderBinary):
    ### Follows gold policy !!

    def train_forward(self, x:torch.Tensor, gold:torch.Tensor):
        enc, encmask = self.tagger.encode_source(x)
        goldlens = (gold != 0).sum(-1)

        y = torch.zeros(x.size(0), 2, device=x.device, dtype=torch.long)
        y[:, 0] = self.vocab["@BOS@"]
        y[:, 1] = self.vocab["@EOS@"]
        ylens = (y != 0).sum(-1)

        gold = torch.cat([y[:, 0:1], gold, torch.zeros_like(y[:, 1:2])], 1)
        gold = gold.scatter(1, goldlens[:, None]+1, self.vocab["@EOS@"])
        goldlens = (gold != 0).sum(-1)

        yalign = torch.zeros_like(y)
        yalign[:, 0] = 0
        yalign[:, 1] = goldlens - 1

        logitsacc = []
        lossacc = torch.zeros(y.size(0), device=y.device)

        newy = None
        newyalign = None
        ended = torch.zeros_like(y[:, 0]).bool()
        while not torch.all(ended): #torch.any(ylens < goldlens):
            # make newy the previous y
            y = newy if newy is not None else y
            yalign = newyalign if newyalign is not None else yalign

            # region TRAIN
            # compute target distribution and mask
            tgt = torch.zeros(y.size(0), y.size(1), self.vocab.number_of_ids(),
                              device=y.device)
            _y = y.cpu().detach().numpy()
            _yalign = yalign.cpu().detach().numpy()
            _gold = gold.cpu().detach().numpy()

            slotvalues_acc = []
            for i in range(len(_y)):
                all_slotvalues = []
                for j in range(len(_y[i])):
                    slotvalues = []
                    if _y[i, j] == self.vocab["@EOS@"]:
                        break
                    # y_ij = _y[i, j]
                    k = _yalign[i, j]   # this is where in the gold we're at
                    k = k + 1
                    # if j + 1 >= len(_yalign[i]):
                    #     print("too large")
                    #     pass
                    while k < _yalign[i, j + 1]:
                        slotvalues.append(_gold[i, k])
                        k += 1
                    if len(slotvalues) == 0:
                        tgt[i, j, self.vocab["@END@"]] = 1
                    else:
                        for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
                            tgt[i, j, slotvalue] += float(valueprob)
                    all_slotvalues.append((slotvalues, self.get_slot_value_probs(slotvalues)))
                slotvalues_acc.append(all_slotvalues)
            tgtmask = (tgt.sum(-1) != 0).float()
            uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
            tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
            # run tagger on y
            logits = self.tagger(tokens=y, enc=enc, encmask=encmask)
            # do loss and backward
            loss = self.compute_loss(logits, tgt[:, :-1], tgtmask[:, :-1])
            loss.mean().backward(retain_graph=True)

            lossacc = lossacc + loss.detach()
            logitsacc.append(logits.detach().clone())
            # endregion

            # region STEP
            # get argmax predicted token to insert at every slot
            _logits = logits.cpu().detach().numpy()

            newy = torch.zeros(y.size(0), y.size(1) + 1, device=y.device).long()
            newyalign = torch.zeros(yalign.size(0), yalign.size(1) + 1, device=yalign.device).long()
            _ended = torch.zeros_like(y[:, 0]).bool()
            for i in range(len(y)):
                k = 0
                # randomly choose which slot to develop
                chosen_js = []
                for j in range(len(slotvalues_acc[i])):     # for every slot
                    if _y[i, j] == self.vocab["@EOS@"]:
                        break
                    slotvalues = slotvalues_acc[i][j][0]
                    if len(slotvalues) != 0:        # consider only positions where a real token must be predicted so not @END@
                        chosen_js.append(j)

                terminated = True
                if len(chosen_js) == 0:     # if all slots terminated
                    newyalign[i, :yalign.size(1)] = yalign[i]
                    newy[i, :y.size(1)] = y[i]
                else:
                    chosen_j = random.choice(chosen_js)

                    for j in range(len(y[i])):
                        if k >= newy.size(1):
                            break
                        newy[i, k] = y[i, j]        # copy
                        newyalign[i, k] = yalign[i, j]
                        k += 1
                        y_ij = _y[i, j]
                        if y_ij == self.vocab["@EOS@"]:  # if token was EOS, terminate generation
                            break  # stop
                        # if j >= len(p_i):  # if we reached beyond the length of predictions, terminate
                        #     break

                        if j == chosen_j:       # if we're at the chosen insertion slot:
                            # get the most probable correct token
                            logits_ij = logits[i, j]  # (vocabsize,)
                            newprobs_ij = torch.zeros_like(logits_ij)
                            slv = list(set(slotvalues_acc[i][j][0]))
                            if len(slv) == 0:       # this slot is completed
                                newprobs_ij[self.vocab["@END@"]] = 1
                            else:
                                for slv_item, slv_prob in zip(*slotvalues_acc[i][j]):
                                    newprobs_ij[slv_item] += slv_prob
                                # newlogits_ij[slv] = logits_ij[slv]
                            # pp_ij, p_ij = newlogits_ij.max(-1)
                            p_ij = torch.multinomial(newprobs_ij, 1)[0]
                            p_ij = p_ij.cpu().detach().item()
                            if p_ij == self.vocab["@END@"]:  # if predicted token is @END@, do nothing
                                pass  # don't insert anything
                            else:  # insert what was predicted
                                if k >= newy.size(1):
                                    break
                                # insert token
                                newy[i, k] = p_ij
                                # align inserted token to gold
                                slotvalues = list(zip(*(slotvalues_acc[i][j] + (range(len(slotvalues_acc[i][j][0])),))))
                                slotvalues = sorted(slotvalues, key=lambda x: x[1], reverse=True)
                                aligned_pos = None
                                for slv, slp, sll in slotvalues:
                                    if slv == p_ij:
                                        aligned_pos = sll + newyalign[i, j] + 1
                                        break
                                newyalign[i, k] = aligned_pos
                                k += 1  # advance newy pointer to next position
                                terminated = False  # sequence changed so don't terminate
                _ended[i] = terminated

            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)  # prevent terminated examples from changing

            maxlen = (newy != 0).long().sum(-1).max()
            newy = newy[:, :maxlen]
            # step += 1
            ended = ended | _ended
            # steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
            lens = (newy != 0).long().sum(-1)
            maxlenreached = (lens == self.max_size)
            if torch.all(maxlenreached):
                break

            # select a random slot by using mask
            # find the most central token that is the same as predicted token for every slot and advance state

            # endregion

            logits = None
            loss = None
        lossacc.requires_grad = True
        return {"loss": lossacc}, logitsacc


class OldSeqInsertionDecoderPredictiveBinary(SeqInsertionDecoderBinary):
    def train_forward(self, x:torch.Tensor, gold:torch.Tensor):
        enc, encmask = self.tagger.encode_source(x)
        goldlens = (gold != 0).sum(-1)

        y = torch.zeros(x.size(0), 2, device=x.device, dtype=torch.long)
        y[:, 0] = self.vocab["@BOS@"]
        y[:, 1] = self.vocab["@EOS@"]
        ylens = (y != 0).sum(-1)

        gold = torch.cat([y[:, 0:1], gold, torch.zeros_like(y[:, 1:2])], 1)
        gold = gold.scatter(1, goldlens[:, None]+1, self.vocab["@EOS@"])
        goldlens = (gold != 0).sum(-1)

        yalign = torch.zeros_like(y)
        yalign[:, 0] = 0
        yalign[:, 1] = goldlens - 1

        logitsacc = []
        lossacc = torch.zeros(y.size(0), device=y.device)

        newy = None
        newyalign = None
        ended = torch.zeros_like(y[:, 0]).bool()
        while not torch.all(ended): #torch.any(ylens < goldlens):
            # make newy the previous y
            y = newy if newy is not None else y
            yalign = newyalign if newyalign is not None else yalign

            # region TRAIN
            # compute target distribution and mask
            tgt = torch.zeros(y.size(0), y.size(1), self.vocab.number_of_ids(),
                              device=y.device)
            _y = y.cpu().detach().numpy()
            _yalign = yalign.cpu().detach().numpy()
            _gold = gold.cpu().detach().numpy()

            slotvalues_acc = []
            for i in range(len(_y)):
                all_slotvalues = []
                for j in range(len(_y[i])):
                    slotvalues = []
                    if _y[i, j] == self.vocab["@EOS@"]:
                        break
                    # y_ij = _y[i, j]
                    k = _yalign[i, j]   # this is where in the gold we're at
                    k = k + 1
                    # if j + 1 >= len(_yalign[i]):
                    #     print("too large")
                    #     pass
                    while k < _yalign[i, j + 1]:
                        slotvalues.append(_gold[i, k])
                        k += 1
                    if len(slotvalues) == 0:
                        tgt[i, j, self.vocab["@END@"]] = 1
                    else:
                        for slotvalue, valueprob in zip(slotvalues, self.get_slot_value_probs(slotvalues)):
                            tgt[i, j, slotvalue] += float(valueprob)
                    all_slotvalues.append((slotvalues, self.get_slot_value_probs(slotvalues)))
                slotvalues_acc.append(all_slotvalues)
            tgtmask = (tgt.sum(-1) != 0).float()
            uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
            tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
            # run tagger on y
            logits = self.tagger(tokens=y, enc=enc, encmask=encmask)
            # do loss and backward
            loss = self.compute_loss(logits, tgt[:, :-1], tgtmask[:, :-1])
            loss.mean().backward(retain_graph=True)

            lossacc = lossacc + loss.detach()
            logitsacc.append(logits.detach().clone())
            # endregion

            # region STEP
            # get argmax predicted token to insert at every slot
            # TODO: must predict the most probable correct tokens
            # predprobs, preds = logits.max(-1)
            # predprobs, preds = predprobs.cpu().detach().numpy(), preds.cpu().detach().numpy()
            _logits = logits.cpu().detach().numpy()

            newy = torch.zeros(y.size(0), y.size(1) + 1, device=y.device).long()
            newyalign = torch.zeros(yalign.size(0), yalign.size(1) + 1, device=yalign.device).long()
            _ended = torch.zeros_like(y[:, 0]).bool()
            for i in range(len(y)):
                k = 0
                # randomly choose which slot to develop
                chosen_js = []
                for j in range(len(slotvalues_acc[i])):     # for every slot
                    if _y[i, j] == self.vocab["@EOS@"]:
                        break
                    slotvalues = slotvalues_acc[i][j][0]
                    if len(slotvalues) != 0:        # consider only positions where a real token must be predicted so not @END@
                        chosen_js.append(j)

                terminated = True
                if len(chosen_js) == 0:     # if all slots terminated
                    newyalign[i, :yalign.size(1)] = yalign[i]
                    newy[i, :y.size(1)] = y[i]
                else:
                    chosen_j = random.choice(chosen_js)

                    for j in range(len(y[i])):
                        if k >= newy.size(1):
                            break
                        newy[i, k] = y[i, j]        # copy
                        newyalign[i, k] = yalign[i, j]
                        k += 1
                        y_ij = _y[i, j]
                        if y_ij == self.vocab["@EOS@"]:  # if token was EOS, terminate generation
                            break  # stop
                        # if j >= len(p_i):  # if we reached beyond the length of predictions, terminate
                        #     break

                        if j == chosen_j:       # if we're at the chosen insertion slot:
                            # get the most probable correct token
                            logits_ij = logits[i, j]  # (vocabsize,)
                            newlogits_ij = torch.zeros_like(logits_ij) - np.infty
                            slv = list(set(slotvalues_acc[i][j][0]))
                            if len(slv) == 0:       # this slot is completed
                                newlogits_ij[self.vocab["@END@"]] = 1
                            else:
                                newlogits_ij[slv] = logits_ij[slv]
                            pp_ij, p_ij = newlogits_ij.max(-1)
                            p_ij = p_ij.cpu().detach().item()
                            if p_ij == self.vocab["@END@"]:  # if predicted token is @END@, do nothing
                                pass  # don't insert anything
                            else:  # insert what was predicted
                                if k >= newy.size(1):
                                    break
                                # insert token
                                newy[i, k] = p_ij
                                # align inserted token to gold
                                slotvalues = list(zip(*(slotvalues_acc[i][j] + (range(len(slotvalues_acc[i][j][0])),))))
                                slotvalues = sorted(slotvalues, key=lambda x: x[1], reverse=True)
                                aligned_pos = None
                                for slv, slp, sll in slotvalues:
                                    if slv == p_ij:
                                        aligned_pos = sll + newyalign[i, j] + 1
                                        break
                                newyalign[i, k] = aligned_pos
                                k += 1  # advance newy pointer to next position
                                terminated = False  # sequence changed so don't terminate
                _ended[i] = terminated

            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)  # prevent terminated examples from changing

            maxlen = (newy != 0).long().sum(-1).max()
            newy = newy[:, :maxlen]
            # step += 1
            ended = ended | _ended
            # steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
            lens = (newy != 0).long().sum(-1)
            maxlenreached = (lens == self.max_size)
            if torch.all(maxlenreached):
                break

            # select a random slot by using mask
            # find the most central token that is the same as predicted token for every slot and advance state

            # endregion

            logits = None
            loss = None
        lossacc.requires_grad = True
        return {"loss": lossacc}, logitsacc

#
# class SeqInsertionDecoderBinaryPredictive(SeqInsertionDecoderPredictive, SeqInsertionDecoderBinary): pass
# class SeqInsertionDecoderUniformPredictive(SeqInsertionDecoderPredictive, SeqInsertionDecoderUniform): pass


class SeqDecoderBaseline(SeqInsertionDecoder):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits, cache = self.tagger(tokens=newy, enc=enc, encmask=encmask, cache=None)
        # compute loss: different versions do different masking and different targets
        loss = self.compute_loss(logits, tgt, mask=tgtmask)
        return {"loss": loss}, logits

    def extract_training_example(self, x, y):
        ymask = (y != 0).float()
        ylens = ymask.sum(1).long()
        newy = y
        newy = torch.cat([torch.ones_like(newy[:, 0:1]) * self.vocab["@BOS@"], newy], 1)
        newy = torch.cat([newy, torch.zeros_like(newy[:, 0:1])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            newy[i, ylen+1] = self.vocab["@END@"]

        goldy = newy[:, 1:]
        tgt = torch.zeros(goldy.size(0), goldy.size(1), self.vocab.number_of_ids(), device=goldy.device)
        tgt = tgt.scatter(2, goldy[:, :, None], 1.)
        tgtmask = (goldy != 0).float()

        newy = newy[:, :-1]
        return x, newy, tgt, tgtmask

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@BOS@"]
        # yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.tagger.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        cache = None
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run tagger
            # y = torch.cat([y, yend], 1)
            logits, cache = self.tagger(tokens=y, enc=enc, encmask=encmask, cache=cache)
            _, preds = logits.max(-1)
            preds = preds[:, -1]
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        return newy, steps_used.float()


class SeqInsertionDecoderLTR(SeqInsertionDecoder):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    # def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of tagger
    #     # extract a training example from y:
    #     x, newy, tgt, tgtmask = self.extract_training_example(x, y)
    #     enc, encmask = self.tagger.encode_source(x)
    #     # run through tagger: the same for all versions
    #     logits = self.tagger(tokens=newy, enc=enc, encmask=encmask)
    #     # compute loss: different versions do different masking and different targets
    #     loss = self.compute_loss(logits, tgt, mask=tgtmask)
    #     return {"loss": loss}, logits

    def extract_training_example(self, x, y):
        # y: (batsize, seqlen) ids, padded with zeros
        ymask = (y != 0).float()
        ytotallens = ymask.sum(1)
        ylens = torch.rand(ytotallens.size(), device=ytotallens.device)
        ylens = (ylens * ytotallens).round().long()
        _ylens = ylens.cpu().numpy()
        # ylens contains the sampled lengths

        # mask randomly chosen tails
        z = torch.arange(y.size(1), device=y.device)
        _y = torch.where(z[None, :] < ylens[:, None], y, torch.zeros_like(y))
        _y = torch.cat([_y, torch.zeros_like(_y[:, 0:1])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            _y[i, ylen] = self.vocab["@EOS@"]
        # prepend BOS
        newy = torch.cat([torch.ones_like(y[:, 0:1]) * self.vocab["@BOS@"], _y], 1)

        _y = torch.cat([y, torch.zeros_like(y[:, 0:1])], 1)
        golds = _y.gather(1, ylens[:, None]).squeeze(1)       # (batsize,)
        golds = torch.where(golds != 0, golds, torch.ones_like(golds) * self.vocab["@END@"])        # when full sequence has been fed, and mask is what remains, make sure that we have @EOS@ instead
        tgt = torch.zeros(newy.size(0), newy.size(1), self.vocab.number_of_ids(), device=newy.device)

        for i, tgt_pos, tgt_val in zip(range(len(ylens)), ylens, golds):
            tgt[i, tgt_pos, tgt_val] = 1

        # normalize
        tgt = tgt / tgt.sum(-1).clamp_min(1e-6)[:, :, None]
        tgtmask = (tgt.sum(-1) != 0).float()
        # make uniform for masked positions
        newymask = (newy != 0).float()
        uniform_tgt = torch.ones_like(tgt) / tgt.size(-1)
        tgt = torch.where(tgtmask[:, :, None].bool(), tgt, uniform_tgt)
        # cut unnecessary padded elements from the right of newy
        newlen = newymask.sum(-1).max()
        newy = newy[:, :int(newlen)]
        tgt = tgt[:, :int(newlen)]
        tgtmask = tgtmask[:, :int(newlen)]

        return x, newy, tgt, tgtmask

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_steps
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@BOS@"]
        yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.tagger.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run tagger
            y_ = torch.cat([y, yend], 1)
            logits = self.tagger(tokens=y_, enc=enc, encmask=encmask)
            _, preds = logits[:, -1].max(-1)
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            step += 1
            steps_used = torch.min(steps_used, torch.where(_ended, torch.ones_like(steps_used) * step, steps_used))
        return newy, steps_used.float()


def run(domain="restaurants",
        mode="baseline",         # "baseline", "ltr", "uniform", "binary"
        probthreshold=0.,       # 0. --> parallel, >1. --> serial, 0.< . <= 1. --> semi-parallel
        lr=0.0001,
        enclrmul=0.1,
        batsize=50,
        epochs=1000,
        hdim=366,
        numlayers=6,
        numheads=6,
        dropout=0.1,
        noreorder=False,
        trainonvalid=False,
        seed=87646464,
        gpu=-1,
        patience=-1,
        gradacc=1,
        cosinelr=False,
        warmup=20,
        gradnorm=3,
        validinter=10,
        maxsteps=20,
        maxsize=75,
        testcode=False,
        numbered=False,
        ):

    settings = locals().copy()
    q.pp_dict(settings)
    wandb.init(project=f"seqinsert_overnight_v2", config=settings, reinit=True)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading")
    tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds(domain, trainonvalid=trainonvalid, noreorder=noreorder, numbered=numbered)
    tt.tock("loaded")

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model
    tagger = TransformerTagger(hdim, flenc.vocab, numlayers, numheads, dropout, baseline=mode=="baseline")

    if mode == "baseline":
        decoder = SeqDecoderBaseline(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize)
    elif mode == "ltr":
        decoder = SeqInsertionDecoderLTR(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize)
    elif mode == "uniform":
        decoder = SeqInsertionDecoderUniform(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)
    elif mode == "binary":
        decoder = SeqInsertionDecoderBinary(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)
    elif mode == "maxspanbinary":
        decoder = SeqInsertionDecoderMaxspanBinary(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)
    elif mode == "predictivebinary":
        decoder = SeqInsertionDecoderPredictiveBinary(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)
    elif mode == "any":
        decoder = SeqInsertionDecoderAny(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, prob_threshold=probthreshold)

    # test run
    if testcode:
        batch = next(iter(tdl_seq))
        # out = tagger(batch[1])
        # out = decoder(*batch)
        decoder.train(False)
        out = decoder(*batch)

    tloss = make_array_of_metrics("loss", reduction="mean")
    tmetrics = make_array_of_metrics("treeacc", "stepsused", reduction="mean")
    vmetrics = make_array_of_metrics("treeacc", "stepsused", reduction="mean")
    xmetrics = make_array_of_metrics("treeacc", "stepsused", reduction="mean")


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

    if patience < 0:
        patience = epochs
    eyt = q.EarlyStopper(vmetrics[0], patience=patience, min_epochs=30, more_is_better=True, remember_f=lambda: deepcopy(tagger))

    def wandb_logger():
        d = {}
        for name, loss in zip(["CE"], tloss):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc", "stepsused"], tmetrics):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["tree_acc", "stepsused"], vmetrics):
            d["valid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    t_max = epochs
    optim = get_optim(tagger, lr, enclrmul)
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=t_max-warmup) >> 0.
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.
    lr_schedule = q.sched.LRSchedule(optim, lr_schedule)

    trainbatch = partial(q.train_batch, gradient_accumulation_steps=gradacc,
                                        on_before_optim_step=[lambda : clipgradnorm(_m=tagger, _norm=gradnorm)])

    trainepoch = partial(q.train_epoch, model=decoder,
                                        dataloader=tdl_seq,
                                        optim=optim,
                                        losses=tloss,
                                        device=device,
                                        _train_batch=trainbatch,
                                        on_end=[lambda: lr_schedule.step()])

    trainevalepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=tmetrics,
                         dataloader=tdl_seq,
                         device=device)

    on_end_v = [lambda: eyt.on_epoch_end(), lambda: wandb_logger()]

    validepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=vmetrics,
                         dataloader=vdl_seq,
                         device=device,
                         on_end=on_end_v)

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch,
                   # run_valid_epoch=[trainevalepoch, validepoch], #[validepoch],
                   run_valid_epoch=[validepoch],
                   max_epochs=epochs,
                   check_stop=[lambda: eyt.check_stop()],
                   validinter=validinter)
    tt.tock("done training")

    if eyt.remembered is not None and not trainonvalid:
        tt.msg("reloading best")
        decoder.tagger = eyt.remembered
        tagger = eyt.remembered

        tt.tick("rerunning validation")
        validres = validepoch()
        tt.tock(f"Validation results: {validres}")

    tt.tick("running train")
    trainres = trainevalepoch()
    print(f"Train tree acc: {trainres}")
    tt.tock()

    tt.tick("running test")
    testepoch = partial(q.test_epoch,
                         model=decoder,
                         losses=xmetrics,
                         dataloader=xdl_seq,
                         device=device)
    testres = testepoch()
    print(f"Test tree acc: {testres}")
    tt.tock()

    settings.update({"final_train_CE": tloss[0].get_epoch_error()})
    settings.update({"final_train_tree_acc": tmetrics[0].get_epoch_error()})
    settings.update({"final_valid_tree_acc": vmetrics[0].get_epoch_error()})
    settings.update({"final_test_tree_acc": xmetrics[0].get_epoch_error()})
    settings.update({"final_train_steps_used": tmetrics[1].get_epoch_error()})
    settings.update({"final_valid_steps_used": vmetrics[1].get_epoch_error()})
    settings.update({"final_test_steps_used": xmetrics[1].get_epoch_error()})

    if mode != "baseline":
        calibrate_end = False
        if calibrate_end:
            # calibrate END offset
            tt.tick("running termination calibration")
            end_offsets = [0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            decoder.prob_threshold = 1.

            end_offset_values = []

            best_offset = 0.
            best_offset_value = 0.
            for end_offset in end_offsets:
                tt.tick("rerunning validation")
                decoder.end_offset = end_offset
                validres = validepoch()
                tt.tock(f"Validation results: {validres}")
                end_offset_values.append(vmetrics[0].get_epoch_error())
                if vmetrics[0].get_epoch_error() > best_offset_value:
                    best_offset = end_offset
                    best_offset_value = vmetrics[0].get_epoch_error()
                tt.tock("done")
            print(f"offset results: {dict(zip(end_offsets, end_offset_values))}")
            print(f"best offset: {best_offset}")

            decoder.end_offset = best_offset

        # run different prob_thresholds:
        # thresholds = [0., 0.3, 0.5, 0.6, 0.75, 0.85, 0.9, 0.95,  1.]
        thresholds = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1.]
        for threshold in thresholds:
            tt.tick("running test for threshold " + str(threshold))
            decoder.prob_threshold = threshold
            testres = testepoch()
            print(f"Test tree acc for threshold {threshold}: testres: {testres}")
            settings.update({f"_thr{threshold}_acc": xmetrics[0].get_epoch_error()})
            settings.update({f"_thr{threshold}_len": xmetrics[1].get_epoch_error()})
            tt.tock("done")


    wandb.config.update(settings)
    q.pp_dict(settings)

# TODO: EOS balancing ?!

# TODO: model that follows predictive distribution during training and uses AnyToken loss

def run_experiment(domain="default",    #
                   mode="baseline",         # "baseline", "ltr", "uniform", "binary"
                   probthreshold=-1.,
        lr=-1.,
        enclrmul=-1.,
        batsize=-1,
        epochs=-1,
        hdim=-1,
        numlayers=-1,
        numheads=-1,
        dropout=-1.,
        noreorder=False,
        trainonvalid=False,
        seed=-1,
        gpu=-1,
        patience=-1,
        gradacc=-1,
        cosinelr=False,
        warmup=-1,
        gradnorm=3.,
        validinter=-1,
        maxsteps=90,
        maxsize=90,
        numbered=False,
                   testcode=False,
        ):

    settings = locals().copy()

    ranges = {
        "domain": ["socialnetwork", "blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "epochs": [121],
        "batsize": [50],
        "hdim": [366, 768],
        "numheads": [6, 12],
        "numlayers": [6, 8, 12],
        "lr": [0.0001, 0.000025],
        "enclrmul": [1., 0.1],                  # use 1.
        "seed": [87646464],
        "patience": [-1],
        "warmup": [20],
        "validinter": [15],
        "gradacc": [1],
    }

    if mode == "baseline":        # baseline
        ranges["validinter"] = [5]
    elif mode.startswith("predictive"):
        ranges["validinter"] = [1]
        ranges["lr"] = [0.0001]
        ranges["enclrmul"] = [1.]
        ranges["dropout"] = [0.0, 0.1, 0.3]     # use 0.
        ranges["hdim"] = [768]
        ranges["numlayers"] = [6]
        ranges["numheads"] = [12]
        ranges["numbered"] = [False]
    else:
        # ranges["domain"] = ["blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"]
        # ranges["domain"] = ["calendar", "publications", "recipes"]
        ranges["batsize"] = [30]
        ranges["dropout"] = [0.0, 0.1, 0.2]     # use 0.
        # ranges["lr"] = [0.0001]                 # use 0.000025
        ranges["validinter"] = [20]
        ranges["epochs"] = [401]
        ranges["hdim"] = [768]
        ranges["numlayers"] = [6]
        ranges["numheads"] = [12]
        ranges["probthreshold"] = [0.]
        ranges["lr"] = [0.000025]
        ranges["enclrmul"] = [1.]
        ranges["numbered"] = [True]

    if mode == "ltr":
        ranges["lr"] = [0.0001, 0.000025]
        ranges["warmup"] = [50]
        ranges["epochs"] = [501]
        ranges["validinter"] = [25]
        ranges["gradacc"] = [10]
        ranges["hdim"] = [768]
        ranges["numlayers"] = [6]
        ranges["numheads"] = [12]

    for k in ranges:
        if k in settings:
            if isinstance(settings[k], str) and settings[k] != "default":
                ranges[k] = [settings[k]]
            elif isinstance(settings[k], (int, float)) and settings[k] >= 0:
                ranges[k] = [settings[k]]
            else:
                pass
                # raise Exception(f"something wrong with setting '{k}'")
            del settings[k]

    def checkconfig(spec):
        if spec["hdim"] == 366 and spec["numheads"] != 6:
            return False
        if spec["hdim"] == 768 and spec["numheads"] != 12:
            return False
        return True

    print(__file__)
    p = __file__ + f".baseline.{domain}"
    q.run_experiments_random(
        run, ranges, path_prefix=p, check_config=checkconfig, **settings)



if __name__ == '__main__':
    # q.argprun(test_tree_oracle)
    # q.argprun(test_tree_sampling)
    q.argprun(test_decode)
    # q.argprun(test_tree_sampling_random)

    # DONE: fix orderless for no simplification setting used here
    # DONE: make baseline decoder use cached decoder states
    # DONE: in unsimplified Overnight, the filters are nested but are interchangeable! --> use simplify filters ?!
    # python overnight_seqinsert.py -gpu 0 -domain ? -lr 0.0001 -enclrmul 1. -hdim 768 -dropout 0.3 -numlayers 6 -numheads 12