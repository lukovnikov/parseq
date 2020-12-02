from nltk import Tree

from parseq.grammar import tree_to_lisp_tokens


def reorder_tree(x:Tree, orderless=None):
    """
    Reorders given tree 'x' such that if a parent label is in 'orderless', the order of the children is always as follows:
    - arg:~type goes first
    - other children are ordered alphabetically
    This function applies itself recursively.
    """
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


def flatten_tree(x: Tree):
    assert(x.label() == "@START@")
    assert(len(x) == 1)
    xstr = tree_to_lisp_tokens(x[0])
    nodes = [Tree(xe if xe not in "()" else "|"+xe, []) for xe in xstr]
    y = Tree(x.label(), nodes)
    return y