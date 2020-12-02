from unittest import TestCase

from parseq.grammar import lisp_to_pas, pas_to_tree, LispToPas, lisp_to_tree, are_equal_trees


class Test_lisp_to_pas(TestCase):
    def print_(self, x, y):
        print(x)
        print(y)
        print(pas_to_tree(y))

    def test_normal(self):
        x = "(call me maybe)"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("call", ["me", "maybe"]))

        x = "(threw I wish (in the well))"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("threw", ["I", "wish", ("in", ["the", "well"])]))

    def test_with_strings_single_quotes(self):
        x = "(call me by \'your name\')"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("call", ["me", "by", "'your name'"]))

        x = "(director (call me by 'your name' man))"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("director", [("call", ["me", "by", "'your name'", "man"])]))

    def test_with_strings_single_quotes_with_parentheses(self):
        x = "(who wrote 'bohemian rhapsody (queen song)')"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("who", ["wrote", "'bohemian rhapsody (queen song)'"]))

    def test_with_strings_double_quotes(self):
        x = "(call me by \"your name\")"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("call", ["me", "by", '"your name"']))

        x = "(director (call me by \"your name\" man))"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("director", [("call", ["me", "by", '"your name"', "man"])]))

    def test_with_strings_double_quotes_with_parentheses(self):
        x = "(who wrote \"bohemian rhapsody (queen song)\")"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("who", ["wrote", "\"bohemian rhapsody (queen song)\""]))

    def test_with_strings_single_and_double_quotes_and_parentheses(self):
        x = "(spouse (wife (director 'federal \"bureau\" of (investigations of )')))"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("spouse", [("wife", [("director", ["'federal \"bureau\" of (investigations of )'"])])]))

    def test_string_escape(self):
        x = "(spouse '\\' qsdf \\'')"
        y = lisp_to_pas(x)
        self.print_(x, y)
        self.assertEqual(y, ("spouse", ["'\\' qsdf \\''"]))

    def test_lisp_to_pass_feed(self):
        x = "(spouse (wife (director 'federal \"bureau\" of (investigations of )')))"
        y = lisp_to_pas(x)
        ltp = LispToPas()
        for xe in x.split():
            z = ltp.feed(xe)
            z = ltp.feed(" ")
            if z is not None:
                break
        self.assertEqual(y, z)
        self.print_(x, z)
        self.assertEqual(z, ("spouse", [("wife", [("director", ["'federal \"bureau\" of (investigations of )'"])])]))

    def test_lisp_to_pass_feed_func(self):
        x = "(spouse (wife (director 'federal \"bureau\" of (investigations of )')))"
        y = lisp_to_pas(x)
        ltp = None
        for xe in x.split():
            z, ltp = lisp_to_pas(xe, ltp)
            z, ltp = lisp_to_pas(" ", ltp)
            if z is not None:
                break
        self.assertEqual(y, z)
        self.print_(x, z)
        self.assertEqual(z, ("spouse", [("wife", [("director", ["'federal \"bureau\" of (investigations of )'"])])]))


class TestEqualTrees(TestCase):
    def test_it(self):
        a = lisp_to_tree("( and ( wife BO ) ( spouse BO ) )")
        b = lisp_to_tree("(and (spouse BO) (wife BO))")
        c = lisp_to_tree("(and (wife BO) (wife BO))")

        print(are_equal_trees(a, a))  # should be True
        print(are_equal_trees(a, b))  # should be True
        print(are_equal_trees(a, c))  # should be False
        self.assertTrue(are_equal_trees(a, a))
        self.assertTrue(are_equal_trees(a, b))
        self.assertFalse(are_equal_trees(a, c))

    def test_it_unk(self):
        a = lisp_to_tree("(and (wife BO) (spouse BO))")
        b = lisp_to_tree("(and (wife BO) (spouse @UNK@))")
        print(are_equal_trees(a, b))
        print(are_equal_trees(b, b))
        self.assertFalse(are_equal_trees(a, b))
        self.assertFalse(are_equal_trees(b, b))

    def test_ordered(self):
        a = lisp_to_tree("(nand (wife BO) (spouse BO))")
        b = lisp_to_tree("(nand (wife BO) (spouse BO) (child BO))")
        print(are_equal_trees(a, b))
        self.assertFalse(are_equal_trees(a, b))

    def test_unordered_twolevel(self):
        a = lisp_to_tree("(and  (or (wife BO) (spouse BO))  (or (wife BR) (spouse BR))  ) ")
        b = lisp_to_tree("(and (or (spouse BR) (wife BR))  (or (spouse BO) (wife BO))  )")
        print(are_equal_trees(a, b))
        self.assertTrue(are_equal_trees(a, b))

    def test_unordered_duplicate(self):
        a = lisp_to_tree("(and (wife BO) (wife BR) (wife BO))")
        b = lisp_to_tree("(and (wife BO) (wife BR) (wife BR))")
        print(are_equal_trees(a, b))
        self.assertFalse(are_equal_trees(a, b))


