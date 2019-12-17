from functools import partial
from unittest import TestCase

import torch

from parseq.eval import TreeAccuracy
from parseq.grammar import lisp_to_tree
from parseq.vocab import Vocab


def tensor2tree(x, D:Vocab=None):
    # x: 1D int tensor
    x = list(x.detach().cpu().numpy())
    x = [D(xe) for xe in x]
    x = [xe for xe in x if xe != D.padtoken]
    # find first @END@ and cut off
    parentheses_balance = 0
    for i in range(len(x)):
        if x[i] ==D.endtoken:
            x = x[:i]
            break
        elif x[i] == "(" or x[i][-1] == "(":
            parentheses_balance += 1
        elif x[i] == ")":
            parentheses_balance -= 1
        else:
            pass

    # balance parentheses
    while parentheses_balance > 0:
        x.append(")")
        parentheses_balance -= 1
    i = len(x) - 1
    while parentheses_balance < 0 and i > 0:
        if x[i] == ")":
            x.pop(i)
            parentheses_balance += 1
        i -= 1

    # convert to nltk.Tree
    try:
        tree, parsestate = lisp_to_tree(" ".join(x), "empty")
    except Exception as e:
        tree = None
    return tree


class TestTreeAccuracy(TestCase):
    def test_normal(self):
        x = ["( and ( has service ) ( has money ) ( and ( got thatstyle ) ( got thatsmile ) ) )",
             "( and ( has service ) ( has service ) ( and ( got thatstyle ) ( got thatsmile ) ) )",
             "( and ( has money ) ( has service ) ( and ( got thatsmile ) ( got thatstyle ) ) )"]
        D = Vocab()
        for xe in x:
            for xes in xe.split():
                D.add_token(xes, seen=True)
        print(D.D)
        acc = TreeAccuracy(tensor2tree=partial(tensor2tree, D=D), orderless={"and"})
        x = [[D[xes] for xes in xe.split()] for xe in x]
        x = torch.tensor(x)
        print(x)

        a = acc(None, x[0:1], x[1:2])
        self.assertEqual(a["tree_acc"], 0)
        print(a)
        a = acc(None, x[0:1], x[2:3])
        self.assertEqual(a["tree_acc"], 1.)
        print(a)

    def test_beam(self):
        x = ["( and ( got the walk ) ( got the talk ) ( and ( got thatstyle ) ( got thatsmile ) ) )",
             "( and ( got the walk ) ( got the walk ) ( and ( got thatstyle ) ( got thatsmile ) ) )",
             "( and ( got the talk ) ( got the walk ) ( and ( got thatsmile ) ( got thatstyle ) ) )",
             "( too_bad ( she ( has ( a penis ) ) ) )"]
        D = Vocab()
        for xe in x:
            for xes in xe.split():
                D.add_token(xes, seen=True)
        print(D.D)
        acc = TreeAccuracy(tensor2tree=partial(tensor2tree, D=D), orderless={"and"})
        x = [[D[xes] for xes in xe.split()] for xe in x]
        # equalize dims
        maxlen = max([len(xe) for xe in x])
        x = [xe + [0]*(maxlen - len(xe)) for xe in x]
        x = torch.tensor(x)
        print(x)

        a = acc(None, x[torch.tensor([1, 3, 2, 0])][None, :, :], x[0:1])
        print(a)
        self.assertTrue(a["tree_acc"] == 0)
        self.assertTrue(a["tree_acc_at1"] == 0)
        self.assertTrue(a["tree_acc_at2"] == 0)
        self.assertTrue(a["tree_acc_at3"] == 1)
        self.assertTrue(a["tree_acc_at4"] == 1)
        self.assertTrue(a["tree_acc_at_last"] == 1)

