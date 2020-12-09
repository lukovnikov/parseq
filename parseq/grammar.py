import re
from abc import ABC, abstractmethod
from typing import List

from nltk import Tree, ParentedTree


def pas_to_str(x):
    if isinstance(x, tuple):    # has children
        head = x[0]
        children = [pas_to_str(e) for e in x[1]]
        return f"{head}({', '.join(children)})"
    else:
        return x


class TreeStrParser(ABC):           # TODO: what if multiple trees in one string?
    def __init__(self, x:str=None, brackets="()"):
        super(TreeStrParser, self).__init__()
        self.stack = [[]]
        self.curstring = None
        self.stringmode = None
        self.prevescape = 0
        self.next_is_sibling = False
        self.nameless_func = "@NAMELESS@"
        self.brackets = brackets

        if x is not None:
            self.feed(x)

    @abstractmethod
    def add_level(self):
        pass

    @abstractmethod
    def close_level(self):
        pass

    @abstractmethod
    def add_sibling(self, next_token):
        pass

    def feed(self, x:str):
        xsplits = re.split("([\(\)\s'\"])", x)
        queue = list(xsplits)
        while len(queue) > 0:
            next_token = queue.pop(0)
            if self.curstring is not None:
                if next_token == "\\":
                    self.prevescape = 2
                elif next_token == "":
                    continue
                self.curstring += next_token
                if self.curstring[-1] == self.stringmode and self.prevescape == 0:  # closing string
                    self.stack[-1].append(self.curstring)
                    self.curstring = None
                    self.stringmode = None
                self.prevescape = max(self.prevescape - 1, 0)
            else:
                self.next_is_sibling = False
                next_token = next_token.strip()
                self.prevescape = False
                if next_token == self.brackets[0]:
                    # add one level on stack
                    self.add_level()
                elif next_token == self.brackets[1]:
                    # close last level on stack, merge into subtree
                    self.close_level()
                elif next_token == "" or next_token == " ":
                    pass  # do nothing
                elif next_token == "'":
                    self.curstring = next_token
                    self.stringmode = "'"
                elif next_token == '"':
                    self.curstring = next_token
                    self.stringmode = '"'
                elif next_token == ",":
                    self.next_is_sibling = True
                else:
                    self.add_sibling(next_token)
        if len(self.stack) != 1 or len(self.stack[-1]) != 1:
            return None
        else:
            return self.stack[-1][-1]


class PrologToPas(TreeStrParser):

    def add_level(self):
        if self.next_is_sibling:
            self.stack[-1].append(self.nameless_func)
        self.stack.append([])

    def close_level(self):
        siblings = self.stack.pop(-1)
        # self.stack[-1].append((siblings[0], siblings[1:]))
        self.stack[-1][-1] = (self.stack[-1][-1], siblings)

    def add_sibling(self, next_token):
        self.stack[-1].append(next_token)


class PrologToTree(TreeStrParser):
    def add_level(self):
        self.stack.append([])

    def close_level(self):
        siblings = self.stack.pop(-1)
        self.stack[-1][-1].extend(siblings)

    def add_sibling(self, next_token):
        self.stack[-1].append(Tree(next_token, []))


class LispToPas(TreeStrParser):
    def add_level(self):
        self.stack.append([])

    def close_level(self):
        siblings = self.stack.pop(-1)
        if len(self.stack[-1]) == 0 and siblings[0] == "lambda":
            self.stack[-1].append("@LAMBDA@")
        self.stack[-1].append((siblings[0], siblings[1:]))

    def add_sibling(self, next_token):
        self.stack[-1].append(next_token)


class LispToTree(TreeStrParser):
    def add_level(self):
        self.stack.append([])

    def close_level(self):
        siblings = self.stack.pop(-1)
        if len(siblings) > 0:
            assert (len(siblings[0]) == 0)
            siblings[0].extend(siblings[1:])
            self.stack[-1].append(siblings[0])

    def add_sibling(self, next_token):
        self.stack[-1].append(Tree(next_token, []))


def _inc_convert_treestr(x, cls, self=-1, brackets="()"):
    """
    :param x: lisp-style string
    strings must be surrounded by single quotes (') and may not contain anything but single quotes
    :return:
    """
    if isinstance(self, cls):
        ret = self.feed(x)
        return ret, self
    else:
        _self = cls(x, brackets=brackets) if not isinstance(self, cls) else self
        ret = _self.feed("")
        if ret is None:
            return None, _self
        else:
            if self is None:
                return ret, _self
            else:
                return ret


def lisp_to_pas(x:str, self:LispToPas=-1, brackets="()"):
    return _inc_convert_treestr(x, LispToPas, self=self, brackets=brackets)


def prolog_to_pas(x:str, self:PrologToPas=-1, brackets="()"):
    return _inc_convert_treestr(x, PrologToPas, self=self, brackets=brackets)


def lisp_to_tree(x:str, self:LispToTree=-1, brackets="()"):
    return _inc_convert_treestr(x, LispToTree, self=self, brackets=brackets)


def prolog_to_tree(x: str, self:PrologToTree = -1, brackets="()"):
    return _inc_convert_treestr(x, PrologToTree, self=self, brackets=brackets)


def pas_to_lisp(x, brackets="()"):
    if isinstance(x, tuple):    # has children
        head = x[0]
        children = [pas_to_lisp(e, brackets=brackets) for e in x[1]]
        return f"{brackets[0]}{head} {' '.join(children)}{brackets[1]}"
    else:
        return x


def pas_to_prolog(x, brackets="()"):
    if isinstance(x, tuple):
        head = x[0]
        children = [pas_to_prolog(e, brackets=brackets) for e in x[1]]
        return f"{head} {brackets[0]} {', '.join(children)} {brackets[1]}"
    else:
        return x


def pas_to_expression(x):
    # flatten out the lists, replace tuples with lists
    if isinstance(x, tuple):
        return [x[0]] + [pas_to_expression(xe) for xe in x[1]]
    else:
        return x


def try_str_to_pas():
    x = "wife(president('US'))"
    pas = prolog_to_pas(x)
    print(x)
    print(pas)

    x = "wife ( president  ( 'united states' ) )  "
    print(x)
    print(prolog_to_pas(x))

    x = "   wife ( president  ( 'united states (country , ,,) ' ) )  "
    print(x)
    print(prolog_to_pas(x))

    x = "  ( wife ( president 'united states (country , ,,) ' ) )  "
    print(x)
    print(prolog_to_pas(x))

    x = "(wife(president 'united states (country , ,,) '))"
    print(x)
    print(prolog_to_pas(x))

    x = "(wife(president 'united states (country , ,,) '))"
    print(x)
    print(prolog_to_pas(x))


def try_lisp_to_tree():
    a = "(wife (president ((us))))"
    b = "(wife (president us))"
    c = "(wife (president (us)))"
    at = lisp_to_tree(a)
    bt = lisp_to_tree(b)
    ct = lisp_to_tree(c)
    print(at)
    print(bt)
    print(ct)



def pas_to_tree(x):
    if isinstance(x, tuple):    # has children
        node = Tree(x[0], [])
        for child in x[1]:
            childnode = pas_to_tree(child)
            node.append(childnode)
    else:
        node = Tree(x, [])
    return node


def tree_to_lisp(x:Tree, brackets="()"):
    if len(x) > 0:
        children = [tree_to_lisp(xe, brackets=brackets) for xe in x]
        return f"{brackets[0]}{x.label()} {' '.join(children)}{brackets[1]}"
    else:
        return x.label()


def tree_to_lisp_tokens(x:Tree, brackets="()"):
    if len(x) > 0:
        children = [tree_to_lisp_tokens(xe, brackets=brackets) for xe in x]
        return [brackets[0], x.label()] + [childe for child in children for childe in child] + [brackets[1]]
    else:
        return [x.label()]


def tree_to_prolog(x:Tree, brackets="()"):
    if len(x) > 0:
        children = [tree_to_prolog(tc, brackets=brackets) for tc in x]
        return f"{x.label()} {brackets[0]} {', '.join(children)} {brackets[1]}"
    else:
        return x.label()


def tree_size(x:Tree):
    ret = sum([tree_size(xe) for xe in x])
    ret += 1
    return ret


class ActionTree(ParentedTree):
    orderless = ["and", "or"]
    singlechildcollapse = ["and", "or"]
    def __init__(self, node, children=None):
        super(ActionTree, self).__init__(node, children=children)
        self._action = None
        self._align = None

    def action(self):
        return self._action

    def set_action(self, action:str):
        self._action = action

    def simplify(self):     # TODO
        for i in range(len(self)):
            child = self[i]
            if child._label in self.singlechildcollapse and len(child) == 2:
                self[i] = child[0]
        for child in self:
            child.simplify_tree_for_eval()

    @classmethod
    def convert(cls, tree):
        """
        Convert a tree between different subtypes of Tree.  ``cls`` determines
        which class will be used to encode the new tree.

        :type tree: Tree
        :param tree: The tree that should be converted.
        :return: The new Tree.
        """
        if isinstance(tree, Tree):
            children = [cls.convert(child) for child in tree]
            ret = cls(tree._label, children)
            ret._action = tree._action
            return ret
        else:
            return tree

    def eq(self, other):
        assert(isinstance(other, ActionTree))
        if self._label != other._label:
            return False
        if self._label in self.orderless:
            # check if every child is in other and other contains no more
            if len(self) != len(other):
                return False
            selfchildren = [selfchild for selfchild in self]
            otherchildren = [otherchild for otherchild in other]
            if not selfchildren[-1].eq(otherchildren[-1]):
                return False    # terminator must be same and in the end
            else:
                selfchildren = selfchildren[:-1]
                otherchildren = otherchildren[:-1]
            i = 0
            while i < len(selfchildren):
                selfchild = selfchildren[i]
                j = 0
                unbroken = True
                while j < len(otherchildren):
                    otherchild = otherchildren[j]
                    if selfchild.eq(otherchild):
                        selfchildren.pop(i)
                        otherchildren.pop(j)
                        i -= 1
                        j -= 1
                        unbroken = False
                        break
                    j += 1
                if unbroken:
                    return False
                i += 1
            if len(selfchildren) == 0 and len(otherchildren) == 0:
                return True
            else:
                return False
        else:
            return all([selfchild.eq(otherchild) for selfchild, otherchild in zip(self, other)])

    @classmethod
    def convert(cls, tree):
        """
        Convert a tree between different subtypes of Tree.  ``cls`` determines
        which class will be used to encode the new tree.

        :type tree: Tree
        :param tree: The tree that should be converted.
        :return: The new Tree.
        """
        if isinstance(tree, Tree):
            children = [cls.convert(child) for child in tree]
            ret = cls(tree._label, children)
            ret._action = tree._action
            ret._align = tree._align
            return ret
        else:
            return tree


def are_equal_trees(self, other, orderless={"and", "or"}, unktoken="@UNK@", use_terminator=False):
    if self is None or other is None:
        return False
    assert(isinstance(other, Tree) and isinstance(self, Tree))
    if self._label != other._label or self._label == unktoken or other._label == unktoken:
        return False
    if self._label in orderless or orderless == "__ALL__":
        # check if every child is in other and other contains no more
        if len(self) != len(other):
            return False
        selfchildren = [selfchild for selfchild in self]
        otherchildren = [otherchild for otherchild in other]
        if use_terminator:
            if not are_equal_trees(selfchildren[-1], otherchildren[-1], orderless=orderless, use_terminator=use_terminator):
                return False    # terminator must be same and in the end
            else:
                selfchildren = selfchildren[:-1]
                otherchildren = otherchildren[:-1]
        i = 0
        while i < len(selfchildren):
            selfchild = selfchildren[i]
            j = 0
            unbroken = True
            while j < len(otherchildren):
                otherchild = otherchildren[j]
                if are_equal_trees(selfchild, otherchild, orderless=orderless, use_terminator=use_terminator):
                    selfchildren.pop(i)
                    otherchildren.pop(j)
                    i -= 1
                    j -= 1
                    unbroken = False
                    break
                j += 1
            if unbroken:
                return False
            i += 1
        if len(selfchildren) == 0 and len(otherchildren) == 0:
            return True
        else:
            return False
    else:
        if len(self) != len(other):
            return False
        return all([are_equal_trees(selfchild, otherchild, orderless=orderless, use_terminator=use_terminator)
                    for selfchild, otherchild in zip(self, other)])


def action_seq_from_tree():
    # depth first action sequence from action tree with actions attached
    pass    # TODO


class FuncGrammar(object):
    typere = re.compile("<([^>]+)>([\*\+]?)")
    def __init__(self, start_type:str, **kw):
        super(FuncGrammar, self).__init__(**kw)
        self.rules_by_type = {}
        self.rules_by_arg = {}
        self.constants = {}
        self.symbols = set()
        self.all_rules = set()
        self.start_type = start_type
        self.start_types = set([start_type])

    def add_rule(self, rule:str):
        if rule in self.all_rules:
            return
        splits = rule.split(" -> ")
        assert(len(splits) == 2)
        t, body = splits
        func_splits = body.split(" :: ")
        sibl_splits = body.split(" -- ")
        assert(int(len(func_splits) > 1) + int(len(sibl_splits) > 1) <= 1)
        if len(func_splits) > 1:
            # function rule --> add children
            arg_name = func_splits[0]
            argchildren = func_splits[1].split(" ")
        elif len(sibl_splits) > 1:
            # sibling rule --> add siblings
            arg_name = sibl_splits[0]
            argchildren = sibl_splits[1].split(" ")
            assert(len(argchildren) == 1)
        else:
            assert(len(body.split(" ")) == 1)
            arg_name = body
            argchildren = []

        if t not in self.rules_by_type:
            self.rules_by_type[t] = set()
        if arg_name not in self.rules_by_arg:
            self.rules_by_arg[arg_name] = set()

        self.rules_by_type[t].add(rule)
        self.rules_by_arg[arg_name].add(rule)
        self.all_rules.add(rule)

        self.symbols.add(arg_name)
        for argchild in argchildren:
            if not self.typere.match(argchild):
                self.symbols.add(argchild)

    def actions_for(self, x:str, format="lisp"):
        if format == "lisp":
            pas = lisp_to_pas(x)
        elif format == "pred" or format == "prolog":
            pas = prolog_to_pas(x)
        else:
            raise Exception(f"unknown format {format}")
        ret = self._actions_for_rec_bottom_up(pas)
        return ret

    def _actions_for_rec_bottom_up(self, pas):
        if isinstance(pas, tuple):      # has children
            arg_name, children = pas
            children_rules, children_types = [], []
            children_ruleses = [self._actions_for_rec_bottom_up(child) for child in children]
            # get child types
            for _child_rules in children_ruleses:
                _child_rule = _child_rules[0]
                child_type, child_body = _child_rule.split(" -> ")
                children_types.append(child_type)
                children_rules += _child_rules

            # merge siblings into single child type
            if len(children_types) > 0 and children_types[-1][-1] in "*+":
                # variable number of children rules
                exp_child_type = children_types[-1][:-1]
                for child_type in children_types[:-1]:
                    assert(child_type == exp_child_type)
                children_types = [children_types[-1]]
        else:
            arg_name, children = pas, []
            children_types = []
            children_rules = []

        # find applicable rules
        rules = self.rules_by_arg[arg_name]
        valid_rules = set()
        # has_sibl_rules = False
        # has_func_rules = False
        for rule in rules:
            rule_type, rule_body = rule.split(" -> ")
            func_splits = rule_body.split(" :: ")
            sibl_splits = rule_body.split(" -- ")
            is_func_rule = len(func_splits) > 1
            is_sibl_rule = len(sibl_splits) > 1
            if not is_sibl_rule and not is_func_rule:   # terminal
                valid_rules.add(rule)
            elif not is_sibl_rule:  # func nonterminal
                rule_arg, rule_inptypes = func_splits[0], func_splits[1].split(" ")
                addit = True
                if len(children_types) != len(rule_inptypes):   # must have same number of children
                    addit = False
                    continue
                # children must match types
                for rule_inptype, child_type in zip(rule_inptypes, children_types):
                    if rule_inptype != child_type:
                        addit = False
                        break
                if not addit:
                    continue
                if addit:
                    valid_rules.add(rule)
            else:
                raise Exception("sibling rule syntax no longer supported")
                valid_rules.add(rule)

        if len(valid_rules) == 0:
            raise Exception(f"can not parse, valid rules for arg '{arg_name}' not found")
        elif len(valid_rules) > 1:
            raise Exception(f"can not parse, multiple valid rules for arg '{arg_name}' found")
        else:
            rule = list(valid_rules)[0]
            return [rule] + children_rules

    def _actions_for_rec_top_down(self, pas, out_type:str=None):
        out_types = self.start_types if out_type is None else set([out_type])
        ret = []
        if isinstance(pas, tuple):
            arg_name, children = pas
        else:
            arg_name, children = pas, []
        # what's this doing??
        if out_type is not None and out_type not in self.rules_by_type:
            assert (pas == arg_name and len(children) == 0 and arg_name == out_type)
            assert (arg_name in self.symbols)
            return []

        # find sibling child rules
        # resolve sibling children before parenting
        prev_sibl_type = None
        new_children = []
        new_children_types = []
        new_children_rules = []
        for child in children:
            child_arg_name = child[0] if isinstance(child, tuple) else child
            if child_arg_name not in self.rules_by_arg:
                new_children.append(child_arg_name)
                new_children_types.append(child_arg_name)
                new_children_rules.append(None)
                continue
            possible_child_rules = self.rules_by_arg[child_arg_name]
            is_sibl = False
            child_type = None
            for pcr in possible_child_rules:
                rule_type, body = pcr.split(" -> ")
                if rule_type[-1] in "*+":
                    is_sibl = True
                    assert(len(possible_child_rules) == 1)  # can't have more than one rule if has sibl rule
                    # do sibl rule stuff
                    if prev_sibl_type is None:
                        prev_sibl_type = rule_type
                        new_children_rules.append([])
                    assert(rule_type == prev_sibl_type)
                    new_children_rules[-1].append(pcr)
                    if " -- " not in body:  # seq terminator
                        prev_sibl_type = None   # done doing siblings
                        new_children.append(None)
                        new_children_types.append(rule_type)
                else:
                    rule_type, body = pcr.split(" -> ")
                    assert(child_type is None or child_type == rule_type)   # arg can have only one return type
                    child_type = rule_type

            if not is_sibl: # normal child
                assert(prev_sibl_type is None)
                new_children.append(child)
                new_children_types.append(child_type)
                new_children_rules.append(None)


        rules = self.rules_by_arg[arg_name]
        valid_rules = set()

        for rule in rules:
            rule_type, rule_body = rule.split(" -> ")
            assert(rule_type[-1] not in "+*")   # no sibling rules here
            func_splits = rule_body.split(" :: ")
            is_func_rule = len(func_splits) > 1
            addit = True
            if is_func_rule:
                func_arg, func_inptypes = func_splits
                func_inptypes = func_inptypes.split(" ")
            else:
                func_arg, func_inptypes = rule_body, []
            # filter by output type
            if rule_type not in out_types:
                addit = False
                continue
            # filter by number of children
            if len(func_inptypes) != len(new_children_types):
                addit = False
                continue
            # filter by children signature
            for new_child_type, func_inptype in zip(new_children_types, func_inptypes):
                if new_child_type is None:
                    if func_inptype[-1] in "*+":
                        addit = False
                        break
                else:
                    if new_child_type != func_inptype:
                        addit = False
                        break
            if not addit:
                continue
            if addit:
                valid_rules.add(rule)

        if len(valid_rules) == 0:
            raise Exception(f"can not parse, valid rules for arg '{arg_name}' not found")
        elif len(valid_rules) > 1:
            raise Exception(f"can not parse, multiple valid rules for arg '{arg_name}' found")
        else:
            rule = list(valid_rules)[0]
            ret.append(rule)
            rule_type, rule_body = rule.split(" -> ")
            assert(rule_type[-1] not in "+*")
            func_splits = rule_body.split(" :: ")
            is_func_rule = len(func_splits) > 1
            addit = True
            if is_func_rule:
                func_arg, func_inptypes = func_splits
                func_inptypes = func_inptypes.split(" ")
            else:
                func_inptypes = []

            assert (len(func_inptypes) == len(new_children_types))
            child_rules = []
            for new_child, new_child_rules, func_inptype in zip(new_children, new_children_rules, func_inptypes):
                if new_child is not None:
                    r = self._actions_for_rec_top_down(new_child, func_inptype)
                    child_rules += r
                else:
                    child_rules += new_child_rules
            # child_rules = [x for x in child_rules if x is not None]
            ret = ret + child_rules  #[rule for child_rule in child_rules for rule in child_rule]
            return ret


    def _actions_for_rec_old(self, pas, out_type:str=None):
        out_type = self.start_type if out_type is None else out_type
        ret = []
        # region
        numchildren = None
        if isinstance(pas, tuple):  # function
            arg_name, body = pas
            numchildren = len(body)
        elif isinstance(pas, list):
            assert(out_type[-1] in ["*", "+"])
            arg_name, body = pas[0], pas[1:]
        else:       # not a function
            arg_name, body = pas, []
            numchildren = 0

        if out_type not in self.rules_by_type:
            assert(pas == arg_name and len(body) == 0 and arg_name == out_type)
            assert(arg_name in self.symbols)
            return [], []

        func_rules = self.rules_by_arg[arg_name]
        valid_rules = set()
        # endregion

        # print(arg_name, body)
        if body is not None:
            body_terminal_signature = [None if isinstance(x, (tuple, list)) else x for x in body]
        if arg_name == "_":
            print(arg_name)
        for rule in func_rules:
            rule_type, rule_body = rule.split(" -> ")
            func_splits = rule_body.split(" :: ")
            sibl_splits = rule_body.split(" -- ")
            addit = True
            if len(func_splits) > 1:
                rule_numchildren = len(func_splits[1].split(" "))
                rule_argchildren = func_splits[1].split(" ")
            elif len(sibl_splits) > 1:
                rule_numchildren = len(sibl_splits[1].split(" "))
                rule_argchildren = sibl_splits[1].split(" ")
                assert(rule_numchildren == 1)
            else:
                rule_numchildren = 0
            if rule_type != out_type:
                addit = False
                continue
            if len(sibl_splits) == 1:
                if numchildren != rule_numchildren:
                    addit = False
                    continue
            if len(func_splits) > 1 and rule_argchildren[-1][-1] not in ["+", "*"]:
                for body_terminal_signature_e, rule_arg_child in zip(body_terminal_signature, rule_argchildren):
                    if rule_arg_child in self.rules_by_type:
                        continue
                    elif body_terminal_signature_e is not None \
                            and body_terminal_signature_e != rule_arg_child:
                        addit = False
                        break
            if addit:
                valid_rules.add(rule)
        if len(valid_rules) == 0:
            raise Exception(f"can not parse, valid rules for arg '{arg_name}' not found")
        elif len(valid_rules) > 1:
            raise Exception(f"can not parse, multiple valid rules for arg '{arg_name}' found")
        else:
            rule = list(valid_rules)[0]
            ret.append(rule)
            rule_type, rule_body = rule.split(" -> ")

            rule_func_splits = rule_body.split(" :: ")
            rule_sibl_splits = rule_body.split(" -- ")
            assert (int(len(rule_func_splits) > 1) + int(len(rule_sibl_splits) > 1) <= 1)

            if len(rule_func_splits) > 1:
                rule_arg, rule_inptypes = rule_func_splits
                rule_inptypes = rule_inptypes.split(" ")
                assert(rule_arg == arg_name)
                for rule_inptype in rule_inptypes:
                    if rule_inptype[-1] in ["+", "*"]:
                        child_rules, body = self._actions_for_rec(body, rule_inptype)
                        ret = ret + child_rules
                    else:
                        child_rules, _body = self._actions_for_rec(body[0], rule_inptype)
                        body = body[1:]
                        ret = ret + child_rules
                return ret, body
                # if len(rule_inptypes) == 1 and rule_inptypes[0][-1] in ["+", "*"]:
                #     child_rules = self._actions_for_rec(body, rule_inptypes[0])
                #     ret = ret + child_rules
                # else:
                #     if len(rule_inptypes) != len(body):
                #         print(rule_inptypes)
                #     assert(len(rule_inptypes) == len(body))
                #     child_rules = [self._actions_for_rec(body_e, rule_inptype) for body_e, rule_inptype in zip(body, rule_inptypes)]
                #     child_rules = [x for x in child_rules if x is not None]
                #     ret = ret + [rule for child_rule in child_rules for rule in child_rule]
                # return ret
            elif len(rule_sibl_splits) > 1:
                rule_arg, rule_sibltypes = rule_sibl_splits
                rule_sibltypes = rule_sibltypes.split(" ")
                assert(len(rule_sibltypes) == 1)
                assert(rule_arg == arg_name)
                if rule_sibltypes[0][-1] in ["*", "+"]:
                    sibl_rules, body = self._actions_for_rec(body, rule_sibltypes[0])
                else:
                    raise NotImplemented()
                ret = ret + sibl_rules
                return ret, body
            else:
                assert(len(rule_body.split(" ")) == 1)
                return ret, body

    def actions_to_tree(self, remaining_actions:List[str]):
        tree, remaining_actions = self._actions_to_tree_rec(remaining_actions)
        assert(len(remaining_actions) == 0)
        return tree

    def _actions_to_tree_rec(self, actions:List[str], out_type:str=None):
        out_type = self.start_type if out_type is None else out_type
        ret = ActionTree(out_type, [])
        remaining_actions = actions
        if out_type not in self.rules_by_type:
            assert(out_type in self.symbols)
            return ret, remaining_actions
        while len(remaining_actions) > 0:
            action = remaining_actions[0]
            ret.set_action(action)
            rule_type, rule_body = action.split(" -> ")

            rule_func_splits = rule_body.split(" :: ")
            rule_sibl_splits = rule_body.split(" -- ")
            assert (int(len(rule_func_splits) > 1) + int(len(rule_sibl_splits) > 1) <= 1)

            if len(rule_func_splits) > 1:
                rule_arg, rule_inptypes = rule_func_splits
                rule_inptypes = rule_inptypes.split(" ")
                ret.set_label(rule_arg)
                remaining_actions = remaining_actions[1:]
                if len(rule_inptypes) == 1 and rule_inptypes[-1][-1] in "*+":
                    rule_inptype = rule_inptypes[-1][:-1]
                    terminated = False
                    while not terminated:
                        subtree, remaining_actions = self._actions_to_tree_rec(remaining_actions, out_type=rule_inptype)
                        ret.append(subtree)
                        if subtree.label() == f"{rule_inptypes[-1]}:END@":
                            terminated = True
                else:
                    for rule_inptype in rule_inptypes:
                        subtree, remaining_actions = self._actions_to_tree_rec(remaining_actions, out_type=rule_inptype)
                        if isinstance(subtree, Tree):
                            subtree = [subtree]
                        for subtree_e in subtree:
                            ret.append(subtree_e)
                return ret, remaining_actions
            elif len(rule_sibl_splits) > 1:
                raise Exception("sibling rules no longer supported")
                rule_arg, rule_sibl_types = rule_sibl_splits
                rule_sibl_types = rule_sibl_types.split(" ")
                assert(len(rule_sibl_types) == 1)
                ret.set_label(rule_arg)
                remaining_actions = remaining_actions[1:]
                siblings, remaining_actions = self._actions_to_tree_rec(remaining_actions, out_type=rule_sibl_types[-1])
                if isinstance(siblings, Tree):
                    siblings = [siblings]
                ret = [ret] + siblings
                return ret, remaining_actions
            else:
                assert(len(rule_body.split(" ")) == 1)
                ret.set_label(rule_body)
                remaining_actions = remaining_actions[1:]
                return ret, remaining_actions


if __name__ == '__main__':
    # try_str_to_pas()
    try_lisp_to_tree()