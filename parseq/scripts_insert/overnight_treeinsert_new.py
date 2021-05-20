import math
import re
import qelos as q
# from multiprocessing import Pool, cpu_count

# from prompt_toolkit.formatted_text import PygmentsTokens

import json
import random
from abc import abstractmethod, ABC
from copy import deepcopy
from functools import partial
from typing import Dict, List, Union, Iterable, Tuple

import torch
import wandb
from nltk import Tree
import numpy as np
from torch.utils.data import DataLoader


from parseq.datasets import OvernightDatasetLoader, autocollate
from parseq.eval import make_array_of_metrics
from parseq.grammar import tree_to_lisp_tokens, are_equal_trees, lisp_to_tree, tree_size
from parseq.scripts_insert.overnight_treeinsert import extract_info
from parseq.scripts_insert.transformer import TransformerConfig, TransformerStack
from parseq.scripts_insert.util import reorder_tree, flatten_tree
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


ORDERLESS = {"op:and", "SW:concat", "filter", "call-SW:concat"}


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


class SpecialTokens(ABC):
    def __init__(self, **kw):
        super(SpecialTokens, self).__init__()
        for k, v in kw.items():
            setattr(self, k, v)

    def __contains__(self, item):
        return item in self.__dict__.values()

    @abstractmethod
    def removeable_tokens(self):
        pass


class DefaultSpecialTokens(SpecialTokens):
    def __init__(self,
                 root_token="@R@",
                 ancestor_slot="@^",
                 descendant_slot="@v",
                 sibling_slot="@-",
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

    def removeable_tokens(self):
        return {self.ancestor_slot, self.descendant_slot, self.sibling_slot, self.parent_separator}


def linearize_ptree(x, is_root=True, only_child=True, specialtokens=DefaultSpecialTokens()):
    # 'x' is normal tree !!
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
    # 'x' is tree with .insert_siblings, .insert_ancestors and .insert_descendants
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
            ret, ret_sup = linearize_supervised_ptree(k, only_child=len(x) == 1, is_root=False, specialtokens=specialtokens)
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


def compute_target_distribution_data(x, centrs, end_token="@END@", tau=1., cmp_use_labels=True):
    # compute ranks
    if len(x) > 0:
        sorted_retsupe = sorted([(centrs[e.nodeid], e) for e in x],
                                key=cmp_to_key(partial(retsup_cmp, uselabels=cmp_use_labels)))
        r = []
        rank = 0
        prev_e = sorted_retsupe[0]
        r.append((rank, prev_e[1].label()))
        for e in sorted_retsupe[1:]:
            if retsup_cmp(e, prev_e, uselabels=cmp_use_labels) != 0:
                if retsup_cmp(e, prev_e, uselabels=cmp_use_labels) < 0:
                    assert retsup_cmp(e, prev_e, uselabels=cmp_use_labels) >= 0
                rank += 1
            else:
                if rank == 0:
                    pass
                    # assert False
            r.append((rank, e[1].label()))
            prev_e = e
    else:
        r = [(1., end_token)]
    # compute softmax over ranks
    d = sum([np.exp(-e[0] / tau) for e in r])
    r = [(np.exp(-e[0] / tau)/d, e[1]) for e in r]
    return r


def retsup_cmp(a:Tuple[float, Tree], b:Tuple[float, Tree], uselabels=True):
    """
    :param a, b:        tuple consisting of a float and a Tree: (float, Tree) where the float is a centrality score for the tree (higher is better)
    :return:
    """
    ret = b[0] - a[0]
    if ret == 0:
        ret = +1* (tree_size(a[1]) - tree_size(b[1]))
    if uselabels and ret == 0:
        al = a[1].label()
        bl = b[1].label()
        if al == bl:
            ret = 0
        elif al < bl:
            ret = -1
        else:
            ret = 1
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


    for i, (xe, ye) in enumerate(zip(xseq, tagseq)):
        if ye in {specialtokens.opening_parentheses,
                  specialtokens.closing_parentheses,
                  specialtokens.parent_separator,
                  specialtokens.sibling_slot,
                  specialtokens.ancestor_slot,
                  specialtokens.descendant_slot,}:
            ye = specialtokens.keep_action
        if xe == specialtokens.opening_parentheses:
            # next_is_parent = True
            assert state.ancestor_insert is None
            if state.node is not None:
                close_node()
                assert state.node is None
            assert state.descendant_insert is None
            reset_vars()
        elif xe == specialtokens.closing_parentheses:     # closing a parent
            close_node()
            if state.parentnode is None:
                state.parentnode = state.curnode
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

    annotated_tree = state.parentnode
    # execute annotated slotted tree
    executed_tree = _execute_slotted_tree(annotated_tree, specialtokens=specialtokens)
    assert len(executed_tree) == 1
    executed_tree = executed_tree[0]

    # linearize executed slotted tree to special slotted sequence of tokens
    outseq = _linearize_slotted_tree(executed_tree, specialtokens=specialtokens)

    return outseq, executed_tree


def _linearize_slotted_tree(x, _root=True, specialtokens=DefaultSpecialTokens()):
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
            subseq = _linearize_slotted_tree(k, _root=False, specialtokens=specialtokens)
            seq = seq + subseq

        seq = seq + [specialtokens.closing_parentheses]
    else:
        seq = ownseq
        if _root:
            seq = [specialtokens.opening_parentheses] + seq + [specialtokens.closing_parentheses]
    return seq


def _execute_slotted_tree(x, specialtokens=DefaultSpecialTokens()):
    if x is None:
        print("x is None")
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

        realchildren = [child for child in node if child.label() != specialtokens.sibling_slot]
        if len(realchildren) == 1:
            realchildren[0].ancestor_insert = None

        return [ancestor]


def run_oracle_tree_decoding(tree, specialtokens=DefaultSpecialTokens(), use_slot_removal=False, explore_top=False):
    assert hasattr(tree, "nodeid")
    xtree = lisp_to_tree(f"({tree.label()} )")

    intermediate_decoding_states = []
    maxiter = 100
    prev_xtree_str = None
    new_xtree_seq = None
    while prev_xtree_str != str(xtree) and maxiter > 0:  # while xtree is changing
        # align nodeids of xtree to tree
        xtree = _align_nodeids(xtree, tree, specialtokens=specialtokens)
        _xtree = xtree
        # build supervision sets
        xtree = build_supervision_sets(xtree, tree)
        centralities = compute_centralities_ptree(xtree, tree)

        # compute best tokens to insert
        ret, retsup = linearize_supervised_ptree(xtree, specialtokens=specialtokens)
        retsup_sup = _get_best_insertions(ret, retsup, centralities, specialtokens=specialtokens, use_slot_removal=use_slot_removal, explore_top=False)
        if explore_top:
            retsup_step = _get_best_insertions(ret, retsup, centralities, specialtokens=specialtokens,
                                               use_slot_removal=use_slot_removal, explore_top=True)
        else:
            retsup_step = retsup_sup

        # remove slots from ret that are not in new_xtree_seq
        if new_xtree_seq is not None:
            i, j = 0, 0
            while i < len(ret):
                ret_i = ret[i].label() if isinstance(ret[i], Tree) else ret[i]
                if ret_i == new_xtree_seq[j]:
                    i += 1; j += 1
                else:
                    assert ret_i in {specialtokens.sibling_slot, specialtokens.ancestor_slot, specialtokens.descendant_slot}
                    assert retsup[i] in {specialtokens.keep_action, specialtokens.close_action}
                    del ret[i]
                    del retsup[i]

        intermediate_decoding_state = (ret, retsup_sup)

        # execute insertions
        ret = [rete.label() if isinstance(rete, Tree) else rete for rete in ret]
        new_xtree_seq, _ = perform_decoding_step(ret, retsup_step, specialtokens=specialtokens)

        # new_xtree = _clean_up_xtree(new_xtree, specialtokens=specialtokens)
        # convert new xtree seq into the new xtree
        _new_xtree_seq = [xe for xe in new_xtree_seq if not (xe in specialtokens.removeable_tokens())]
        new_xtree_str = " ".join(_new_xtree_seq)
        new_xtree = lisp_to_tree(new_xtree_str)
        intermediate_decoding_state = intermediate_decoding_state + (_xtree, new_xtree,)

        prev_xtree_str = str(xtree)
        xtree = new_xtree
        maxiter -= 1

        intermediate_decoding_states.append(intermediate_decoding_state)
    assert str(tree) == str(xtree)
    return intermediate_decoding_states


def _clean_up_xtree(x:Tree, specialtokens=DefaultSpecialTokens()):
    _xes = []
    for xe in x:
        if xe.label() == specialtokens.sibling_slot:
            pass
        else:
            _xe = _clean_up_xtree(xe)
            _xes.append(_xe)
    x[:] = _xes
    return x


def _get_best_insertions(ret, retsup, centralities, use_slot_removal=False, specialtokens=DefaultSpecialTokens(), explore_top=False):
    # if end_token is "@KEEP@", then normal termination is followed,
    # if end_token is "@CLOSE@", then slot removal is used
    end_token = specialtokens.close_action if use_slot_removal is True else specialtokens.keep_action
    _retsup = []
    for rete, retsupe in zip(ret, retsup):
        if retsupe != None:
            r = compute_target_distribution_data(retsupe, centralities, end_token=end_token, cmp_use_labels=not explore_top)
            best_score = r[0][0]
            _r = [r[0]]
            for r_i in r[1:]:
                if r_i[0] == best_score:
                    _r.append(r_i)
                else:
                    break
            _r = random.choice(_r)
            _retsup.append(_r[1])
        else:
            _retsup.append(end_token)
    return _retsup


def _align_nodeids(xtree, tree, specialtokens=DefaultSpecialTokens()):
    tree_nodes = _gather_descendants(tree, _top=False)
    tree_node_labels = [tree_node.label() for tree_node in tree_nodes]
    assert len(tree_node_labels) == len(set(tree_node_labels))      # assert all nodelabels unique

    xtree_nodes = _gather_descendants(xtree, _top=False)
    for xtree_node in xtree_nodes:
        if xtree_node.label() == specialtokens.sibling_slot:
            continue
        for tree_node in tree_nodes:
            if xtree_node.label() == tree_node.label():
                xtree_node.nodeid = tree_node.nodeid
    return xtree


def run_oracle_seq_decoder(tree):
    seq = tree_to_seq(tree[0])      # don't use the first element, which is root
    decoding_steps = _run_seq_decoder_rec(seq)
    maxdepth = _max_seq_decoder_depth(decoding_steps) + 1
    return decoding_steps, maxdepth


def _max_seq_decoder_depth(x):
    if len(x) <= 1:
        return 1
    maxdepth_left = _max_seq_decoder_depth(x[0])
    maxdepth_right= _max_seq_decoder_depth(x[2])
    ret = max(maxdepth_left, maxdepth_right) + 1
    return ret


def _run_seq_decoder_rec(seq):
    if len(seq) <= 1:
        return seq[:]
    # take middle, call recursive on halves
    middle = int(round((len(seq) - 1) / 2))
    left = seq[:middle]
    right = seq[middle+1:]
    out = seq[middle]
    left_out = _run_seq_decoder_rec(left)
    right_out = _run_seq_decoder_rec(right)
    return [left_out, out, right_out]


def test_oracle_decoder(n=1):
    for ni in range(n):
        # tree = lisp_to_tree(f"( R (A (B (C D E) (F (G H I J)))))")
        # tree = lisp_to_tree(f"( R (A (B (C (D E K ) ) L M N (F (G H I J)))))")
        # tree = lisp_to_tree(f"(1 (11 (111 1111 1112 1113) (112 1121 1122 1123) (113 1131 1132 1133)) (12 (121 1211 1212 1213) (122 1221 (1222 (12222 (122222 1222222))) 1223) (123 1231 1232 1233)) (13 (131 1311 1312 1313) (132 1321 1322 1323) (133 1331 1332 1333)))")
        # tree = lisp_to_tree(f"(R (B (C (D (E (F (G (H I J K L M N O))))))))")
        # tree = lisp_to_tree(f"(R (A B (C D (E F (G H (I J (K L (M N (O P Q)))))))))")
        tree = lisp_to_tree(f"(R (A (B C D E) (F G H I) (J K L M)))")

        print("Tree to decode: ", tree)
        print(f"Tree size: {tree_size(tree)}")

        # assign nodeids to original tree
        tree = assign_dfs_nodeids(tree)

        decoding_steps = run_oracle_tree_decoding(tree)
        print(f"number of steps: {len(decoding_steps)}")
        for i, decoding_step in enumerate(decoding_steps):
            print(f"Step {i+1}: {decoding_step[2]}")
        print("done")

        seq = tree_to_seq(tree)
        print("Seq to decode: ", " ".join(seq))

        decoding_steps, maxdepth = run_oracle_seq_decoder(tree)
        print(maxdepth)


def test_decode(n=1):
    # superbasic test:
    print("superbasic test")
    r_str = "( R @v | A @v )"
    print(r_str)
    rl = r_str.split(" ")
    rl_tags = [None, None, "D", None, None, "E", None]
    print(rl_tags)
    rl_tp1 = perform_decoding_step(rl, rl_tags)
    print(" ".join(rl_tp1))
    assert " ".join(rl_tp1) == "( R @v | @- ( D @v | ( A @v | @- E @v @- ) ) @- )"
                              #"( R @v | @- ( D @v | ( @^ A @v | @- E @v @- ) ) @- )"

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
    assert " ".join(rl_tp1) == "( R @v | @- ( D @v | @- @^ E @v @- ( @^ B @v | @- G @v @- ) @- @^ H @v @- ) @- )"

    # test different slots
    print("test with different slots")
    r = Tree("R", [Tree("A", [Tree("C", [])]), Tree("B", [])])
    rl = [re.label() if isinstance(re, Tree) else re for re in linearize_ptree(r)]
    r_str = " ".join(rl)
    print(r_str)
    #            (    R    @v     |   @-    (   @^     A   @v    |    @-    C   @v    @-  )    @-   @^    B   @v   @-   )
    rl_tags = [None, None, "D", None, "E", None, "F", None, "G", None, "H", None, "I", "J", None, "K", "L", None, "M", "N", None]
    print(rl_tags)
    rl_tp1 = perform_decoding_step(rl, rl_tags)
    print(" ".join(rl_tp1))
    assert " ".join(rl_tp1) == "( R @v | @- ( D @v | @- @^ E @v @- ( @^ F @v | @- ( A @v | @- ( G @v | @- @^ H @v @- ( @^ C @v | @- I @v @- ) @- @^ J @v @- ) @- ) @- ) @- @^ K @v @- ( @^ L @v | @- ( B @v | @- M @v @- ) @- ) @- @^ N @v @- ) @- )"

    
    # test with closed slots I
    print("test with closed slots I")
    # r = Tree("R", [Tree("A", [Tree("C", [])]), Tree("B", [])])
    # rl = [re.label() if isinstance(re, Tree) else re for re in linearize_ptree(r)]
    r_str = "( R @v | ( A @v | @- B C @v @- ) @- )"
    print(r_str)
    rl = r_str.split(" ")
    #           (      R    @v    |    (     A    @v     |   @-    B     C   @v   @-    )   @-  )
    rl_tags = [None, None, "D", None, None, None, "E", None, "F", None, None, "G", "H", None, "I", None]
    print(rl_tags)
    rl_tp1 = perform_decoding_step(rl, rl_tags)
    print(" ".join(rl_tp1))
    assert " ".join(
        rl_tp1) == "( R @v | @- ( D @v | ( @^ A @v | @- ( E @v | @- @^ F @v @- B ( C @v | @- G @v @- ) @- @^ H @v @- ) @- ) @- @^ I @v @- ) @- )"


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
    centralities = compute_centralities_ptree(ptree_ret, tree)

    ret, retsup = linearize_supervised_ptree(ptree_ret)

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


def run_decoding_oracle(domain="publications", verbose=False, removeslots=False):
    tt = q.ticktock("oracle")
    # load overnight dataset
    tt.tick(f"loading '{domain}' domain")
    tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc, orderless = load_ds(domain=domain, numbered_nodes=True, output="original")
    tt.tock(f"loaded")
    tt.msg(f"{len(xds_seq)} test examples")

    seq_insert_steps = []
    tree_insert_steps = []

    # for i, x in enumerate(xds_seq):
    for i, x in enumerate(tds_seq):
        print(f"Example {i}")
        if verbose:
            print(f"{x[0]}")
            print(f"{x[1]}")

        # run oracle decoders and log the ideal number of decoding steps for seq-insert and tree-insert
        tree = x[1]
        if verbose:
            print(f"Tree size: {tree_size(tree)}")

        # assign nodeids to original tree
        tree = assign_dfs_nodeids(tree)

        decoding_steps = run_oracle_tree_decoding(tree, use_slot_removal=removeslots)
        tree_insert_steps.append(len(decoding_steps))
        if verbose:
            print(f"number of steps: {len(decoding_steps)}")
            for i, decoding_step in enumerate(decoding_steps):
                print(f"Step {i + 1}: {decoding_step[2]}")
            print("done")

        seq = tree_to_seq(tree)
        if verbose:
            print("Seq to decode: ", " ".join(seq))

        decoding_steps, maxdepth = run_oracle_seq_decoder(tree)
        seq_insert_steps.append(maxdepth)

        if verbose:
            print(f"Number of seq-insertion steps: {maxdepth}")

    avg_seq_insertion_steps = np.mean(seq_insert_steps)
    avg_tree_insertion_steps = np.mean(tree_insert_steps)

    tt.msg(f"Avg seq insertion steps for {domain}: {avg_seq_insertion_steps}")
    tt.msg(f"Avg tree insertion steps for {domain}: {avg_tree_insertion_steps}")

    return avg_seq_insertion_steps, avg_tree_insertion_steps


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


def make_numbered_nodes(x:Tree, counts=None):
    if counts is None:      # root
        counts = {}
    xe = x.label()
    if xe not in counts:
        counts[xe] = 0
    counts[xe] += 1
    xe = f"{xe}::{counts[xe]}"
    children = []
    for child in x:
        _child, counts = make_numbered_nodes(child, counts)
        children.append(_child)
    ret = Tree(xe, children)
    return ret, counts


def make_numbered_nodes_list(x:List[str]):
    counts = {}
    y = []
    for xe in x:
        if xe in {"(", ")"}:
            y.append(xe)
        else:
            if xe not in counts:
                counts[xe] = 0
            counts[xe] += 1
            y.append(f"{xe}::{counts[xe]}")
    return y


def load_ds(domain="restaurants", nl_mode="bert-base-uncased",
            trainonvalid=False, noreorder=False, numbered=False, numbered_nodes=False,
            output="special", specialtokens=DefaultSpecialTokens()):
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

    if numbered_nodes:
        ds = ds.map(lambda x: (x[0], make_numbered_nodes(x[1])[0], x[2]))

    ds = ds.map(lambda x: (x[0], tree_to_seq(x[1]), x[1], x[2]))

    if numbered:
        ds = ds.map(lambda x: (x[0], make_numbered_tokens(x[1]), x[2], x[3]))

    vocab = Vocab(padid=0, startid=2, endid=3, unkid=1)
    for _, specialtoken_v in specialtokens.__dict__.items():
        vocab.add_token(specialtoken_v, seen=np.infty)


    tds, vds, xds = ds[lambda x: x[-1] == "train"], \
                    ds[lambda x: x[-1] == "valid"], \
                    ds[lambda x: x[-1] == "test"]

    nl_tokenizer = BertTokenizer.from_pretrained(nl_mode)
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

    if output == "tensors":
        def mapper(x):
            seq = seqenc.convert(x[1], return_what="tensor")
            ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0], seq)
            return ret
    elif output == "original":
        def mapper(x):
            tr = lisp_to_tree(" ".join(x[1]))
            ret = (x[0], tr)
            return ret
    elif output == "special":       # input as tensors, outputs as rerooted trees
        def mapper(x):
            inp = nl_tokenizer.encode(x[0], return_tensors="pt")[0]
            out = Tree(specialtokens.root_token, [x[2]])
            return (inp, out)
    else:
        mapper = None

    if mapper is not None:
        tds_seq = tds.map(mapper)
        vds_seq = vds.map(mapper)
        xds_seq = xds.map(mapper)
    else:
        tds_seq, vds_seq, xds_seq = tds, vds, xds
    return tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc, orderless


class TreeInsertionTagger(torch.nn.Module):
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


class RelPosDict(object):
    def __init__(self, maxv=10, maxh=10, **kw):
        super(RelPosDict, self).__init__(**kw)
        self.maxv, self.maxh = maxv, maxh
        self.build_dict()

    def build_dict(self):
        self.D = {}
        self.D["@PAD@"] = 0
        self.D["@UNK@"] = 1
        nextid = max(self.D.values()) + 1

        for t in ["self", "descslot", "ancslot", "left", "right", "parent"]:
            self.D[t] = nextid
            nextid += 1

        for i in range(self.maxv):
            self.D[f"^{i}"] = nextid
            nextid += 1

        for i in range(self.maxv):
            self.D[f"v{i}"] = nextid
            nextid += 1

        for i in range(self.maxh):
            self.D[f"<{i}"] = nextid
            nextid += 1

        for i in range(self.maxh):
            self.D[f">{i}"] = nextid
            nextid += 1

        for i in range(self.maxh):
            self.D[f"child-{i}"] = nextid
            nextid += 1

        for i in range(self.maxh):
            self.D[f"child-{i}-of"] = nextid
            nextid += 1

        self.RD = {v: k for k, v in self.D.items()}

    def __getitem__(self, item:str):
        return self.D[item]

    def __call__(self, item:int):
        return self.RD[item]

    def __len__(self):
        return self.num_tokens()

    def num_tokens(self):
        return max(self.D.values()) + 1


class RelPosEmb(torch.nn.Module):
    ## Note: Even if this is shared across layers, keep the execution separate between layers because attention weights are different
    def __init__(self, dim, D=RelPosDict(), mode="sum", **kw):
        super(RelPosEmb, self).__init__(**kw)
        self.D, self.mode = D, mode

        self.emb = torch.nn.Embedding(len(self.D), dim, padding_idx=0)
        self.emb_v = torch.nn.Embedding(len(self.D), dim, padding_idx=0)

    def compute_scores(self, query, relpos):
        """
        :param q:       (batsize, numheads, qlen, dimperhead)
        :param relpos:  (batsize, qlen, klen, n)
        :return:
        """
        retscores = None
        for n in range(relpos.size(-1)):
            indexes = torch.arange(0, self.emb.num_embeddings, device=query.device).long()
            embs = self.emb(indexes)# (numindexes, dim)
            embs = embs.view(embs.size(0), query.size(1), query.size(-1))        # (numindexes, numheads, dimperhead)
            relpos_ = relpos[:, :, :, n]
            scores = torch.einsum("bhqd,nhd->bhqn", query, embs)  # (batsize, numheads, qlen, numindexes)
            relpos_ = relpos_[:, None, :, :].repeat(1, scores.size(1), 1, 1)  # (batsize, numheads, qlen, klen)
            scores_ = torch.gather(scores, 3, relpos_)  # (batsize, numheads, qlen, klen)
            if retscores is None:
                retscores = torch.zeros_like(scores_)
            retscores = retscores + scores_
        return retscores        # (batsize, numheads, qlen, klen)

    def compute_context(self, weights, relpos):
        """
        :param weights: (batsize, numheads, qlen, klen)
        :param relpos:  (batsize, qlen, klen, ...)
        :return:    # weighted sum over klen (batsize, numheads, qlen, dimperhead)
        """
        ret = None
        batsize = weights.size(0)
        numheads = weights.size(1)
        qlen = weights.size(2)
        device = weights.device

        # Naive implementation builds matrices of (batsize, numheads, qlen, klen, dimperhead)
        # whereas normal transformer only (batsize, numheads, qlen, klen) and (batsize, numheads, klen, dimperhead)
        for n in range(relpos.size(-1)):
            relpos_ = relpos[:, :, :, n]

            # map relpos_ to compact integer space of unique relpos_ entries
            relpos_unique = relpos_.unique()
            mapper = torch.zeros(relpos_unique.max() + 1, device=device, dtype=torch.long)  # mapper is relpos_unique but the other way around
            mapper[relpos_unique] = torch.arange(0, relpos_unique.size(0), device=device).long()
            relpos_mapped = mapper[relpos_]     # (batsize, qlen, klen) but ids are from 0 to number of unique relposes

            # sum up the attention weights which refer to the same relpos id
            # scatter: src is weights, index is relpos_mapped[:, None, :, :]
            # scatter: gathered[batch, head, qpos, relpos_mapped[batch, head, qpos, kpos]]
            #               += weights[batch, head, qpos, kpos]
            gathered = torch.zeros(batsize, numheads, qlen, relpos_unique.size(0), device=device)
            gathered = torch.scatter_add(gathered, -1, relpos_mapped[:, None, :, :].repeat(1, numheads, 1, 1), weights)
            # --> (batsize, numheads, qlen, numunique): summed attention weights

            # get embeddings and update ret
            embs = self.emb_v(relpos_unique).view(relpos_unique.size(0), numheads, -1)        # (numunique, numheads, dimperhead)
            relposemb = torch.einsum("bhqn,nhd->bhqd", gathered, embs)
            if ret is None:
                ret = torch.zeros_like(relposemb)
            ret  = ret + relposemb

        # for n in range(relpos.size(-1)):
        #     relpos_ = relpos[:, :, :, n]
        #     relposemb_ = self.emb_v(relpos_)      # (batsize, qlen, klen, dim)
        #     if relposemb is None:
        #         relposemb = torch.zeros_like(relposemb_)
        #     relposemb = relposemb + relposemb_
        # relposemb = relposemb.view(relposemb.size(0), relposemb.size(1), relposemb.size(2), numheads, -1)   # (batsize, qlen, klen, numheads, dimperhead)
        # relposemb = relposemb.permute(0, 3, 1, 2, 4)
        # ret = (relposemb * weights.unsqueeze(-1)).sum(3)
        return ret


class TransformerTagger(TreeInsertionTagger):
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6,
                 dropout:float=0., maxpos=1024, bertname="bert-base-uncased",
                 rel_pos=False, abs_pos=True, **kw):
        super(TransformerTagger, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        config = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                   num_layers=numlayers, num_heads=numheads, dropout_rate=dropout,
                                   use_relative_position=False)

        self.rel_pos = rel_pos

        self.emb = torch.nn.Embedding(config.vocab_size, config.d_model)

        if self.rel_pos is True:
            self.rel_pos = RelPosEmb(self.dim, D=RelPosDict())

        self.abs_pos = abs_pos
        if self.rel_pos is None or self.rel_pos is False or self.abs_pos is True:
            self.abs_pos = torch.nn.Embedding(maxpos, config.d_model)

        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = False
        self.decoder = TransformerStack(decoder_config, rel_emb=self.rel_pos)

        self.out = torch.nn.Linear(self.dim, self.vocabsize)
        # self.out = MOS(self.dim, self.vocabsize, K=mosk)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname,
                                                    hidden_dropout_prob=min(dropout, 0.1),
                                                    attention_probs_dropout_prob=min(dropout, 0.1))

        self.adapter = None
        if self.bert_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.bert_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
        encs = self.bert_model(x, attention_mask=encmask)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None, relpos=None, attnmask=None):
        if attnmask is not None:
            # compress inputs
            attnmask_q = attnmask.sum(1) > 0
            attnmask_k = attnmask.sum(2) > 0
            compress_mask = attnmask_q | attnmask_k
        attnmask = (tokens != 0) if attnmask is None else attnmask
        # if not self.baseline:
        #     padmask = padmask[:, 1:]
        embs = self.emb(tokens)
        if self.abs_pos is not None and self.abs_pos is not False:
            posembs = self.abs_pos(torch.arange(tokens.size(1), device=tokens.device))[None]
            embs = embs + posembs
        use_cache = False
        if cache is not None:
            embs = embs[:, -1:, :]

        _ret = self.decoder(inputs_embeds=embs, attention_mask=attnmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=use_cache,
                     past_key_value_states=cache,
                     relpos=relpos)
        ret = _ret[0]
        cache = None

        c = ret

        logits = self.out(c)
        return logits



def compute_relpos_and_attn_mask(seq:List[str], specialtokens=DefaultSpecialTokens(), D=RelPosDict(), usechildrels=True):
    """
    :param x:   the current batch of trees
    :return:    (batsize, qlen, klen, 4) -- how to reach every token from every other token in the tree.
                    First 3 (out of 4) coordinates specify how to move between real tree nodes.
                    Last 4th (out of 4) is a different special index space for slot tokens:
                        sibling slot is connected to parent using child-N-of relative position,
                            to left and right using special relposes too
                        descendant slot is connected to its node using its special relpos
                        ancestor slot is connected to its node
                    Structural tokens ("(", ")", "|") should be ignored in relpos

                (batsize, qlen, klen) -- attention mask that ignores structural tokens and removes slot tokens from keys
                            and leaves only immediate neighbours as keys to which slot tokens can attend to as queries

                ??? TO DO ?: ^0, v0, <0, >0 should all map to a zero-vector (@PAD@ instead)
    """
    badtokens = {specialtokens.opening_parentheses,
                 specialtokens.closing_parentheses,
                 specialtokens.parent_separator,
                 specialtokens.descendant_slot,
                 specialtokens.ancestor_slot,
                 specialtokens.sibling_slot}

    # attnmask and relpos_slots
    attnmask = torch.ones(len(seq), len(seq))       # (qlen, klen)
    relpos_slots = torch.zeros(len(seq), len(seq))
    for i, x in enumerate(seq):
        if x in badtokens:
            attnmask[:, i] = 0      # can't be attended to from anywhere
            attnmask[i, :] = 0      # can't attend to anything either, ...

        if x not in {specialtokens.opening_parentheses,
                     specialtokens.closing_parentheses,
                     specialtokens.parent_separator}:
            attnmask[i, i] = 1
            relpos_slots[i, i] = D["self"]

        if x == specialtokens.descendant_slot:
            attnmask[i, i-1] = 1    # ... except to its node
            relpos_slots[i, i-1] = D["descslot"]
        elif x == specialtokens.ancestor_slot:
            attnmask[i, i+1] = 1    # ... except to its node
            relpos_slots[i, i+1] = D["ancslot"]
        elif x == specialtokens.sibling_slot:
            pass
    relpos = torch.zeros(len(seq), len(seq), 4)
    relpos[:, :, -1] = relpos_slots

    # do relative position coordinates
    do_relpos(seq, relpos, attnmask, D=D, specialtokens=specialtokens, usechildrels=usechildrels)
    return relpos, attnmask


def do_relpos(seq, relpos, attnmask, specialtokens=DefaultSpecialTokens(), D=RelPosDict(), usechildrels=True):

    # region build tree
    buffer = seq[:]
    stack = [[]]
    i = 0

    nextisparent = False

    while len(buffer) > 0:
        first = buffer.pop(0)
        if first == specialtokens.opening_parentheses:
            nextisparent = True
        elif first == specialtokens.closing_parentheses:
            try:
                bottom = stack.pop(-1)
                stack[-1][-1][:] = bottom
            except Exception as e:
                print(e)
        elif first in {
            specialtokens.parent_separator,
            # specialtokens.sibling_slot,
            specialtokens.ancestor_slot,
            specialtokens.descendant_slot,
        }:
            pass
        else:
            node = Tree(first, [])
            node.seqid = i
            stack[-1].append(node)
            if nextisparent:
                stack.append([])
                nextisparent = False

        i += 1

    assert len(stack) == 1 and len(stack[0]) == 1
    tree = stack[0][0]
    # endregion

    ret = do_relpos_rec(tree, relpos, attnmask, D=D, specialtokens=specialtokens, usechildrels=usechildrels)
    return tree


def do_relpos_rec(tree, relpos, attnmask, D=RelPosDict(), specialtokens=DefaultSpecialTokens(), usechildrels=True):
    rets = []
    ownret = []
    realchildren = []

    for i, child in enumerate(tree):
        if child.label() == specialtokens.sibling_slot:
            leftsibling = tree[i-1] if i-1 >= 0 else None
            rightsibling = tree[i+1] if i+1 < len(tree) else None
            parentnode = tree
            if leftsibling is not None:
                relpos[child.seqid, leftsibling.seqid, 3] = D["left"]
                attnmask[child.seqid, leftsibling.seqid] = 1
            if rightsibling is not None:
                relpos[child.seqid, rightsibling.seqid, 3] = D["right"]
                attnmask[child.seqid, rightsibling.seqid] = 1
            relpos[child.seqid, parentnode.seqid, 3] = D["parent"]
            attnmask[child.seqid, parentnode.seqid] = 1
        else:
            realchildren.append(child)
            ret = do_relpos_rec(child, relpos, attnmask, D=D, usechildrels=usechildrels)      # ret is a list of tuples (Tree, depth) of all descendants
            rets.append(ret)
            # record relations between parent and children
            if usechildrels is True:
                relpos[tree.seqid, child.seqid, 3] = D[f"child-{min(i, D.maxh-1)}"]
                relpos[child.seqid, tree.seqid, 3] = D[f"child-{min(i, D.maxh-1)}-of"]
            else:
                relpos[tree.seqid, child.seqid, 0] = D[f"^1"]
                relpos[child.seqid, tree.seqid, 2] = D[f"v1"]
            # make ownret
            ownret.append((child, 1))
            for rete in ret:
                # record relations between this node and its descendants
                relpos[rete[0].seqid, tree.seqid, 0] = D[f"^{min(rete[1]+1, D.maxv-1)}"]
                relpos[tree.seqid, rete[0].seqid, 2] = D[f"v{min(rete[1]+1, D.maxv-1)}"]
                # build ownret
                ownret.append((rete[0], rete[1]+1))


    for i in range(len(rets)):
        for j in range(i+1, len(rets)):
            # record relations between siblings
            # record relations between descendants of siblings
            for (lret, ldepth) in rets[i] + [(realchildren[i], 0)]:
                for (rret, rdepth) in rets[j] + [(realchildren[j], 0)]:
                    relpos[lret.seqid, rret.seqid, 0] = D[f"^{min(ldepth, D.maxv-1)}"] if ldepth > 0 else D["@PAD@"]
                    relpos[rret.seqid, lret.seqid, 2] = D[f"v{min(ldepth, D.maxv-1)}"] if ldepth > 0 else D["@PAD@"]
                    relpos[lret.seqid, rret.seqid, 1] = D[f">{min(j-i, D.maxh-1)}"]
                    relpos[rret.seqid, lret.seqid, 1] = D[f"<{min(j-i, D.maxh-1)}"]
                    relpos[rret.seqid, lret.seqid, 0] = D[f"^{min(rdepth, D.maxv-1)}"] if rdepth > 0 else D["@PAD@"]
                    relpos[lret.seqid, rret.seqid, 2] = D[f"v{min(rdepth, D.maxv-1)}"] if rdepth > 0 else D["@PAD@"]

    return ownret


def test_relpos_and_attnmask(n=1):
    tree = lisp_to_tree("(A (B (C F) (D G) (E H I)))")
    ret = linearize_ptree(tree)
    ret = [rete.label() if isinstance(rete, Tree) else rete for rete in ret]
    relpos, attnmask = compute_relpos_and_attn_mask(ret)
    for i, rete in enumerate(ret):
        print(f"{i}: {rete}")

    D = RelPosDict()

    for i in range(len(relpos)):
        for j in range(len(relpos[i])):
            if relpos[i, j].sum() > 0:
                s = []
                for k in range(len(relpos[i, j])):
                    if relpos[i, j, k] != 0:
                        s.append(D(relpos[i, j, k].item()))
                print(f"{ret[i]} -> {ret[j]}: {', '.join(s)}")

    print(attnmask)
    relpos_ = torch.zeros_like(relpos)

    attnmask_ = (relpos.sum(-1) > 0).float()
    print((attnmask_- attnmask).nonzero())
    assert torch.allclose(attnmask_, attnmask)

    relpos_[1, 12, 2] = D["v2"]
    relpos_[1, 16, 2] = D["v3"]
    relpos_[1, 23, 2] = D["v2"]
    relpos_[1, 27, 2] = D["v3"]
    relpos_[1, 34, 2] = D["v2"]
    relpos_[1, 39, 2] = D["v3"]
    relpos_[1, 43, 2] = D["v3"]
    relpos_[6, 16, 2] = D["v2"]
    relpos_[6, 27, 2] = D["v2"]
    relpos_[6, 39, 2] = D["v2"]
    relpos_[6, 43, 2] = D["v2"]
    relpos_[16, 27, 2] = D["v1"]
    relpos_[27, 16, 2] = D["v1"]
    relpos_[27, 39, 2] = D["v1"]
    relpos_[39, 27, 2] = D["v1"]
    relpos_[27, 43, 2] = D["v1"]
    relpos_[43, 27, 2] = D["v1"]
    relpos_[16, 39, 2] = D["v1"]
    relpos_[39, 16, 2] = D["v1"]
    relpos_[16, 43, 2] = D["v1"]
    relpos_[43, 16, 2] = D["v1"]
    relpos_[12, 27, 2] = D["v1"]
    relpos_[12, 39, 2] = D["v1"]
    relpos_[12, 43, 2] = D["v1"]
    relpos_[23, 16, 2] = D["v1"]
    relpos_[23, 39, 2] = D["v1"]
    relpos_[23, 43, 2] = D["v1"]
    relpos_[34, 16, 2] = D["v1"]
    relpos_[34, 27, 2] = D["v1"]


    # print((relpos_[:, :, 2] - relpos[:, :, 2]).nonzero())
    assert torch.allclose(relpos_[:, :, 2], relpos[:, :, 2])

    assert attnmask[1, 6] == 1
    assert attnmask[6, 1] == 1
    assert attnmask[6, 7] == 0
    assert attnmask[7, 6] == 1
    attnmask7_ = torch.zeros_like(attnmask[7])
    attnmask7_[6] = 1
    attnmask7_[7] = 1
    # print(attnmask7_, attnmask[7])
    assert torch.allclose(attnmask[7], attnmask7_)

    attnmask20_ = torch.zeros_like(attnmask[20])
    attnmask20_[20] = 1
    attnmask20_[12] = 1
    attnmask20_[23] = 1
    attnmask20_[6] = 1
    print(attnmask20_.nonzero(), attnmask[20].nonzero())
    assert torch.allclose(attnmask[20], attnmask20_)

    attnmaskx_ = torch.zeros_like(attnmask[41])
    attnmaskx_[41] = 1
    attnmaskx_[39] = 1
    attnmaskx_[43] = 1
    attnmaskx_[34] = 1
    print(attnmaskx_.nonzero(), attnmask[41].nonzero())
    assert torch.allclose(attnmask[41], attnmaskx_)

    attnmaskx_ = torch.zeros_like(attnmask[45])
    attnmaskx_[45] = 1
    attnmaskx_[43] = 1
    attnmaskx_[34] = 1
    print(attnmaskx_.nonzero(), attnmask[45].nonzero())
    assert torch.allclose(attnmask[45], attnmaskx_)

    attnmaskx_ = torch.zeros_like(attnmask[49])
    attnmaskx_[49] = 1
    attnmaskx_[1] = 1
    attnmaskx_[6] = 1
    print(attnmaskx_.nonzero(), attnmask[49].nonzero())
    assert torch.allclose(attnmask[49], attnmaskx_)

    print("\n ||   WORKING CORRECTLY || \n")


def sample_remove_ended_slots(seq, seqsup, specialtokens=DefaultSpecialTokens()):
    """
    :param seq:     sequence of str or Tree
    :param seqsup:  sequence of lists of tuples (str, prob) that specify target output distribution at every token
    :return: new seq and seqsup where a random portion of ended slots have been removed
    """
    # find the positions of all slots that are ended (where nothing will be inserted in any of the following steps)
    ended_slots = []
    for i in range(len(seq)):
        seqi = seq[i].label() if isinstance(seq[i], Tree) else seq[i]
        if seqi in {specialtokens.ancestor_slot, specialtokens.descendant_slot, specialtokens.sibling_slot}:
            if seqsup[i] is None or len(seqsup[i]) == 0 or (len(seqsup[i]) == 1 and seqsup[i][0][1] == specialtokens.close_action):
                ended_slots.append(i)

    if len(ended_slots) > 0:
        # uniformly sample a number of how many will be sampled as already previously closed and randomly pick the slots
        num_ended_slots = random.choice(list(range(len(ended_slots))))
        random.shuffle(ended_slots)
        end_slots = set(ended_slots[:num_ended_slots])
    else:
        end_slots = {}

    # copy the input data, but omit those positions where the slots are that are sampled as already previous closed
    retseq = []
    retseqsup = []
    for i in range(len(seq)):
        if i not in end_slots:
            retseq.append(seq[i])
            retseqsup.append(seqsup[i])

    return retseq, retseqsup


class TreeInsertionDecoder(torch.nn.Module):
    # default_termination_mode = "slot"
    # default_decode_mode = "parallel"

    def __init__(self, tagger:TreeInsertionTagger,
                 vocab=None,
                 prob_threshold=0.,
                 max_steps:int=20,
                 max_size:int=200,
                 end_offset=0.,
                 tau=1.,
                 removeslots=False,
                 use_rel_pos=False,
                 oracle_mix=0.,             # 0. --> sampled from scratch, 1. --> sampled from oracle
                 specialtokens=DefaultSpecialTokens(),
                 numtraj=5,
                 usechildrels=True,
                 **kw):
        super(TreeInsertionDecoder, self).__init__(**kw)
        self.tagger = tagger
        self.vocab = vocab
        self.max_steps = max_steps
        self.max_size = max_size
        self.kldiv = torch.nn.KLDivLoss(reduction="none")
        self.logsm = torch.nn.LogSoftmax(-1)
        self.prob_threshold = prob_threshold
        self.end_offset = end_offset
        self.tau = tau

        self.oracle_mix = oracle_mix

        self.specialtokens = specialtokens
        self.removeslots = removeslots
        self.slot_termination_action = self.specialtokens.keep_action if not self.removeslots else self.specialtokens.close_action

        # self.pool = [Pool(cpu_count())]
        # self.termination_mode = self.default_termination_mode if termination_mode == "default" else termination_mode
        # self.decode_mode = self.default_decode_mode if decode_mode == "default" else decode_mode

        self.use_rel_pos = use_rel_pos

        self.numtraj = numtraj
        self.usechildrels = usechildrels

        self._gold_data_cache = {}

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

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

        best_pred = logits.max(-1)[1]   # (batsize, seqlen)
        best_gold = tgt.max(-1)[1]
        same = best_pred == best_gold
        if mask is not None:
            same = same | ~(mask.bool())
        acc = same.all(-1)  # (batsize,)

        # get probability of best predictions
        tgt_probs = torch.gather(tgt, -1, best_pred.unsqueeze(-1))  # (batsize, seqlen, 1)
        recall = (tgt_probs > 0).squeeze(-1)
        if mask is not None:
            recall = recall | ~(mask.bool())
        recall = recall.all(-1)
        return kl, acc.float(), recall.float()

    def train_forward(self, x:torch.Tensor, y:Tuple[Tree]):  # --> implement one step training of tagger
        # extract a training example from y:
        x, newy, tgt, tgtmask, relpos, attnmask = self.extract_training_example(x, y)
        enc, encmask = self.tagger.encode_source(x)
        # run through tagger: the same for all versions
        logits = self.tagger(tokens=newy, enc=enc, encmask=encmask, relpos=relpos, attnmask=attnmask)
        # compute loss: different versions do different masking and different targets
        loss, acc, recall = self.compute_loss(logits, tgt, mask=tgtmask)
        return {"loss": loss, "acc": acc, "recall": recall}, logits

    def extract_training_example_mapf(self, tree, ptree_choice=None, num_top_explore=5):
        tree = assign_dfs_nodeids(tree)

        use_oracle = random.random() < self.oracle_mix
        if use_oracle:
            if str(tree) not in self._gold_data_cache:
                ptrees = []
                for _ in range(max(1, num_top_explore)):
                    ptreese = run_oracle_tree_decoding(tree, self.specialtokens, use_slot_removal=self.removeslots, explore_top=num_top_explore > 0)
                    ptrees.append(ptreese)
                self._gold_data_cache[str(tree)] = ptrees
            ptrees = random.choice(self._gold_data_cache[str(tree)])
            # if ptree_choice is None:
            ptree_choice = random.choice(list(range(len(ptrees))))
            _ptree_choice = min(ptree_choice, len(ptrees)-1)
            ptree = ptrees[_ptree_choice][2]
        else:
            ptree = sample_partial_tree(tree)

        ptree = build_supervision_sets(ptree, tree)
        centralities = compute_centralities_ptree(ptree, tree)
        ret, retsup = linearize_supervised_ptree(ptree)

        _retsup = []
        for rete, retsupe in zip(ret, retsup):
            if retsupe != None:
                r = compute_target_distribution_data(retsupe, centralities, tau=self.tau,
                                                     end_token=self.slot_termination_action)
                _retsup.append(r)
            else:
                _retsup.append(None)

        if self.removeslots and not self.use_oracle:
            ret, _retsup = sample_remove_ended_slots(ret, _retsup)

        relpos, attnmask = None, None
        if self.use_rel_pos:
            ret_ = [rete.label() if isinstance(rete, Tree) else rete for rete in ret]
            relpos, attnmask = compute_relpos_and_attn_mask(ret_, usechildrels=self.usechildrels)

        return ret, _retsup, relpos, attnmask, ptree_choice

    def extract_training_example(self, x: torch.Tensor, y:Tuple[Tree]):
        """
        Samples a partial tree, generates output distributions (tgt) and mask (tgtmask).
        :return: x, newy (batsize, seqlen), tgt (batsize, seqlen, vocsize), tgtmask (batsize, seqlen)
        """
        # sample partial tree
        parallel = False
        if parallel:
            assert False
            ret = self.pool[0].map(self.extract_training_example_mapf, y)
            ret, retsup, relpos, attnmask = list(zip(*ret))
        else:
            outputs = []
            ptree_choice = None
            for ye in y:
                output = self.extract_training_example_mapf(ye, ptree_choice=ptree_choice, num_top_explore=self.numtraj)
                ptree_choice = output[-1]
                outputs.append(output[:-1])
            ret, retsup, relpos, attnmask = list(zip(*outputs))
            # ret, retsup, relpos, attnmask = list(zip(*[self.extract_training_example_mapf(ye) for ye in y]))

        maxlen = max([len(rete) for rete in ret])
        _y = y
        newy = torch.zeros(len(y), maxlen, dtype=torch.long, device=x.device)
        for i, rete in enumerate(ret):
            for j, retei in enumerate(rete):
                retei = retei.label() if isinstance(retei, Tree) else retei
                assert isinstance(retei, str)
                newy[i, j] = self.vocab[retei]
        tgtmask = (newy != 0).float()

        tgt = torch.zeros(newy.size(0), newy.size(1), self.vocab.number_of_ids(), device=newy.device)
        # 'tgt' contains target distributions
        for i, retsupe in enumerate(retsup):
            for j, retsupei in enumerate(retsupe):
                if retsupei is None:
                    tgtmask[i, j] = 0
                    tgt[i, j, self.vocab[self.slot_termination_action]] = 1
                elif len(retsupei) == 0:
                    tgt[i, j, self.vocab[self.slot_termination_action]] = 1
                else:
                    for retsupei_v, retsupei_k in retsupei:
                        tgt[i, j, self.vocab[retsupei_k]] = retsupei_v

        relposes, attnmasks = None, None
        if self.use_rel_pos:
            relposes = torch.zeros(newy.size(0), maxlen, maxlen, relpos[0].size(2), dtype=torch.long, device=x.device)
            attnmasks = torch.zeros(newy.size(0), maxlen, maxlen, dtype=torch.float, device=x.device)
            for i, relpose in enumerate(relpos):
                relposes[i, :relpose.size(0), :relpose.size(1), :] = relpose
                attnmasks[i, :attnmask[i].size(0), :attnmask[i].size(1)] = attnmask[i]

        return x, newy, tgt, tgtmask, relposes, attnmasks

    def get_prediction(self, x:torch.Tensor):
        """
        Takes tensor of input sequence.
        Return output sequence which is a sequence of tokens as strings.
        """
        ptrees = [Tree(self.specialtokens.root_token, []) for _ in range(len(x))]
        rls = [[r.label() if isinstance(r, Tree) else r for r in linearize_ptree(ptree)] for ptree in ptrees]

        steps_used = [0 for _ in rls]

        # run encoder
        enc, encmask = self.tagger.encode_source(x)

        step = 0
        prevstrs = [" ".join(rle) for rle in rls]
        finalpreds = [None for _ in rls]
        while step < self.max_steps and any([finalpred is None for finalpred in finalpreds]): #(newy is None or not (y.size() == newy.size() and torch.all(y == newy))):
            maxlen = max([len(rl) for rl in rls])
            y = torch.zeros(len(rls), maxlen, device=x.device, dtype=torch.long)
            for i, rl in enumerate(rls):
                for j, rli in enumerate(rl):
                    y[i, j] = self.vocab[rli]

            relpos, attnmask = None, None
            if self.use_rel_pos:
                relposes, attnmasks = list(zip(*[compute_relpos_and_attn_mask(rete, usechildrels=self.usechildrels) for rete in rls]))
                relpos = torch.zeros(len(relposes), maxlen, maxlen, relposes[0].size(2), dtype=torch.long,
                                       device=x.device)
                attnmask = torch.zeros(len(relposes), maxlen, maxlen, dtype=torch.float, device=x.device)
                for i, relpose in enumerate(relposes):
                    relpos[i, :relpose.size(0), :relpose.size(1), :] = relpose
                    attnmask[i, :attnmasks[i].size(0), :attnmasks[i].size(1)] = attnmasks[i]

            # run tagger
            logits = self.tagger(tokens=y, enc=enc, encmask=encmask, relpos=relpos, attnmask=attnmask)

            # get probs
            probs = torch.softmax(logits, -1)
            probs[:, :, self.vocab[self.slot_termination_action]] = \
                probs[:, :, self.vocab[self.slot_termination_action]] - self.end_offset
            predprobs, preds = probs.max(-1)
            predprobs, preds = predprobs.cpu().detach().numpy(), preds.cpu().detach().numpy()

            # translate preds to tokens
            tagses = list(np.vectorize(lambda x: self.vocab(x))(preds))

            newrls = []
            # execute tags on the original sequence
            for i, (rle, tags) in enumerate(zip(rls, tagses)):
                if finalpreds[i] is None:
                    new_rle, _ = perform_decoding_step(rle, list(tags)[:len(rle)], specialtokens=self.specialtokens)
                    if len(new_rle) >= self.max_size or step >= self.max_steps or prevstrs[i] == " ".join(new_rle):
                        finalpreds[i] = new_rle[:min(len(new_rle), self.max_size)]
                        new_rle = [self.specialtokens.opening_parentheses,
                                   self.specialtokens.root_token,
                                   self.specialtokens.closing_parentheses]
                    steps_used[i] += 1
                    newrls.append(new_rle)
                    prevstrs[i] = " ".join(newrls[i])
                else:
                    newrls.append(rls[i])

            rls = newrls
            step += 1
        #     # take care of termination: terminate if (1) none of the examples changed (use prevstrs)
        #     #                                     OR (2) max_steps reached
        #     # ! if an example reached maximum allowable length, don't change it and consider it terminated
        #     # ! for efficiency, don't run terminated examples through tagger
        return finalpreds, torch.tensor(steps_used).float().to(x.device)

    def test_forward(self, x:torch.Tensor, gold:Tuple[Tree]):   # --> implement how decoder operates end-to-end
        """
        :param x:       tensor for BERT
        :param gold:    tuple of rerooted gold trees
        :return:
        """
        preds, stepsused = self.get_prediction(x)

        # compute loss and metrics
        gold_trees = gold
        pred_trees = []
        omit = {self.specialtokens.parent_separator, self.specialtokens.descendant_slot, self.specialtokens.ancestor_slot, self.specialtokens.sibling_slot}
        for pred in preds:
            pred_tree = None
            if pred is not None:
                # pred = [re.sub("::\d+", "", pred_token) for pred_token in pred]
                pred_tree = " ".join([pred_token for pred_token in pred if pred_token not in omit])
                pred_tree = lisp_to_tree(pred_tree)
                if isinstance(pred_tree, tuple) and pred_tree[0] is None:
                    pred_tree = None
            pred_trees.append(pred_tree)
        # pred_trees = [lisp_to_tree(" ".join(pred)) if pred is not None else None for pred in preds]
        # pred_trees = [predtree if not (isinstance(predtree, tuple) and predtree[0] is None) else None for predtree in pred_trees]

        treeaccs = []
        for gold_tree, pred_tree in zip(gold_trees, pred_trees):
            gold_tree = remove_number_from_tree_labels(gold_tree)
            pred_tree = remove_number_from_tree_labels(pred_tree)
            treeacc = float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
            treeaccs.append(treeacc)
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device), "stepsused": stepsused}
        return ret, pred_trees


def remove_number_from_tree_labels(x:Tree):
    x_ = None
    if x is not None:
        x_ = []
        for xe in x:
            xe_ = remove_number_from_tree_labels(xe)
            x_.append(xe_)
        x_ = Tree(re.sub("::\d+", "", x.label()), x_)
        # x.set_label(re.sub("::\d+", "", x.label()))
    return x_


def run(domain="restaurants",
        goldtemp=1.,        # 10000 --> towards a more uniform gold distribution
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
        userelpos=False,
        useabspos=False,
        evaltrain=False,
        removeslots=False,
        oraclemix=0.,
        numtraj=5,
        usechildrels=True,
        ):

    settings = locals().copy()
    q.pp_dict(settings)
    settings["version"] = "vcr"
    wandb.init(project=f"treeinsert_overnight_v2", config=settings, reinit=True)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading")
    tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds(domain, trainonvalid=trainonvalid,
                                                                 noreorder=noreorder, numbered_nodes=numbered)
    tt.tock("loaded")

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model
    tagger = TransformerTagger(hdim, flenc.vocab, numlayers, numheads, dropout, rel_pos=userelpos, abs_pos=useabspos)

    decoder = TreeInsertionDecoder(tagger, flenc.vocab, max_steps=maxsteps, max_size=maxsize, removeslots=removeslots,
                                   prob_threshold=probthreshold, tau=goldtemp, use_rel_pos=userelpos, oracle_mix=oraclemix,
                                   usechildrels=usechildrels, numtraj=numtraj)

    # print(decoder)
    # test run
    if testcode:
        batch = next(iter(tdl_seq))
        # out = tagger(batch[1])
        # out = decoder(*batch)
        decoder.train(False)
        out = decoder(*batch)

    tloss = make_array_of_metrics("loss", "acc", "recall", reduction="mean")
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
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(high=1., low=0.1, steps=t_max-warmup) >> 0.1
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

    if evaltrain:
        valid_epoch_fs = [trainevalepoch, validepoch]
    else:
        valid_epoch_fs = [validepoch]
    q.run_training(run_train_epoch=trainepoch,
                   run_valid_epoch=valid_epoch_fs,
                   # run_valid_epoch=[trainevalepoch, validepoch], #[validepoch],
                   # run_valid_epoch=[validepoch],
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

    wandb.config.update(settings)
    q.pp_dict(settings)


def run_experiment(domain="default",    #
                   goldtemp=-1.,
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
        maxsize=250,
        numbered=False,
        userelpos=False,
        removeslots=False,
        useabspos=False,
        evaltrain=False,
        oraclemix=0.,
        usechildrels="default",
        numtraj=-1,
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
        "seed": [87646464, 42, 456852],
        "patience": [-1],
        "warmup": [20],
        "validinter": [15],
        "gradacc": [1],
    }

    if settings["domain"] != "default":
        domains = settings["domain"].split(",")
        ranges["domain"] = domains
        settings["domain"] = "default"
    else:
        # ranges["domain"] = ["socialnetwork", "blocks", "calendar", "housing", "restaurants", "publications", "recipes", "basketball"]
        ranges["domain"] = ["calendar", "publications"]
        ranges["domain"] = ["housing", "blocks", "basketball"]
    ranges["batsize"] = [16]
    # ranges["dropout"] = [0.0, 0.1, 0.2, 0.5]  # use 0.   (or 0.1?)
    ranges["dropout"] = [0.2, 0.4]
    # ranges["lr"] = [0.0001]                 # use 0.000025
    ranges["validinter"] = [10]
    ranges["epochs"] = [251]
    ranges["hdim"] = [768]
    ranges["numlayers"] = [6]
    ranges["numheads"] = [12]
    ranges["probthreshold"] = [0.]
    ranges["lr"] = [0.00001, 0.000025]
    # ranges["lr"] = [0.00005]
    ranges["enclrmul"] = [1.]
    ranges["goldtemp"] = [1., 0.1, 10.]     # results use 0.1, default is 1.

    if usechildrels == "default":
        ranges["usechildrels"] = [True, False]
    else:
        ranges["usechildrels"] = [True] if usechildrels in ("True", "true", "1") else [False]

    ranges["numtraj"] = [0, 1, 10]      # default 5

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


# recipes:       overnight_treeinsert_new.py -gpu 1 -numbered -userelpos -noabspos -useoracle -evaltrain -lr 0.00001 -goldtemp 0.1 -dropout 0.2 -domain recipes
# restaurants:   overnight_treeinsert_new.py -gpu 0 -numbered -batsize 10 -userelpos -domain restaurants,recipes,housing,blocks -lr 0.00005 -oraclemix 1. -evaltrain -goldtemp 0.1 -cosinelr -epochs 201 -dropout 0.2
# housing:       overnight_treeinsert_new.py -gpu 0 -numbered -batsize 10 -userelpos -domain restaurants,recipes,housing,blocks -lr 0.00005 -oraclemix 1. -evaltrain -goldtemp 0.1 -cosinelr -epochs 201 -dropout 0.2
# socialnetwork: overnight_treeinsert_new.py -gpu 0 -numbered -userelpos -evaltrain -lr 0.00005 -cosinelr -goldtemp 0.1 -oraclemix 1.0 -domain socialnetwork -dropout 0.2 -epochs 201
# basketball:    overnight_treeinsert_new.py -gpu 0 -numbered -batsize 10 -userelpos -domain basketball -lr 0.00005 -oraclemix 1. -evaltrain -goldtemp 0.1 -cosinelr -epochs 201 -dropout 0.2
# blocks:        overnight_treeinsert_new.py -gpu 0 -numbered -batsize 10 -userelpos -domain restaurants,recipes,housing,blocks -lr 0.00005 -oraclemix 1. -evaltrain -goldtemp 0.1 -cosinelr -epochs 201 -dropout 0.2


if __name__ == '__main__':
    # q.argprun(test_tree_oracle)
    # q.argprun(test_tree_sampling)
    # q.argprun(test_decode)
    # test_oracle_decoder()
    # q.argprun(run_decoding_oracle)
    # q.argprun(test_tree_sampling_random)
    # test_relpos_and_attnmask()
    q.argprun(run_experiment)             #     -gpu 0 -numbered -batsize 10 -userelpos -noabspos -domain publications -lr 0.00005 -useoracle -dropout 0 -evaltrain -goldtemp 0.1

    # DONE: fix orderless for no simplification setting used here
    # DONE: make baseline decoder use cached decoder states
    # DONE: in unsimplified Overnight, the filters are nested but are interchangeable! --> use simplify filters ?!
    # python overnight_seqinsert.py -gpu 0 -domain ? -lr 0.0001 -enclrmul 1. -hdim 768 -dropout 0.3 -numlayers 6 -numheads 12