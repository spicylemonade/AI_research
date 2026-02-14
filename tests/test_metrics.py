#!/usr/bin/env python3
"""Unit tests for the evaluation metrics suite."""

import os
import sys
import math
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import (
    exact_match, symbolic_equivalence, numerical_r2,
    tree_edit_distance, complexity_penalty, composite_score,
    prefix_to_sympy, _tree_depth
)


class TestPrefixToSympy(unittest.TestCase):
    def test_simple(self):
        expr = prefix_to_sympy("mul m a")
        self.assertIsNotNone(expr)
        self.assertEqual(str(expr), "a*m")

    def test_nested(self):
        expr = prefix_to_sympy("add v0 mul a t")
        self.assertIsNotNone(expr)

    def test_integer(self):
        expr = prefix_to_sympy("INT_5")
        self.assertEqual(float(expr), 5.0)

    def test_named_constant(self):
        expr = prefix_to_sympy("pi")
        self.assertAlmostEqual(float(expr), math.pi, places=5)

    def test_invalid_returns_none(self):
        expr = prefix_to_sympy("")
        self.assertIsNone(expr)


class TestExactMatch(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(exact_match("mul m a", "mul m a"), 1.0)

    def test_commutative(self):
        # mul m a == mul a m after simplification
        self.assertEqual(exact_match("mul m a", "mul a m"), 1.0)

    def test_different(self):
        self.assertEqual(exact_match("mul m a", "add m a"), 0.0)

    def test_simplified_equal(self):
        # x + x should equal 2*x
        self.assertEqual(exact_match("add x x", "mul INT_2 x"), 1.0)

    def test_invalid_pred(self):
        self.assertEqual(exact_match("", "mul m a"), 0.0)


class TestSymbolicEquivalence(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(symbolic_equivalence("mul m a", "mul m a"), 1.0)

    def test_commutative(self):
        self.assertEqual(symbolic_equivalence("mul m a", "mul a m"), 1.0)

    def test_algebraic_identity(self):
        # (a+b)^2 == a^2 + 2ab + b^2
        self.assertEqual(
            symbolic_equivalence(
                "pow add a b INT_2",
                "add add pow a INT_2 mul mul INT_2 a b pow b INT_2"
            ), 1.0
        )

    def test_different(self):
        self.assertEqual(symbolic_equivalence("mul m a", "div m a"), 0.0)

    def test_numerical_fallback(self):
        # sin(x)^2 + cos(x)^2 == 1
        self.assertEqual(
            symbolic_equivalence(
                "add pow sin x INT_2 pow cos x INT_2",
                "INT_1"
            ), 1.0
        )


class TestNumericalR2(unittest.TestCase):
    def test_perfect_match(self):
        r2 = numerical_r2("mul m a", "mul m a")
        self.assertAlmostEqual(r2, 1.0, places=3)

    def test_zero_match(self):
        r2 = numerical_r2("INT_0", "mul m a")
        self.assertLess(r2, 0.5)

    def test_with_data(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        Y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        r2 = numerical_r2("pow x INT_2", "pow x INT_2", X, Y, ["x"])
        self.assertAlmostEqual(r2, 1.0, places=3)

    def test_partial_match(self):
        # x^2 vs x^2 + 1 should have high but not perfect R²
        r2 = numerical_r2("pow x INT_2", "add pow x INT_2 INT_1")
        self.assertGreater(r2, 0.5)
        self.assertLess(r2, 1.0)


class TestTreeEditDistance(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(tree_edit_distance("mul m a", "mul m a"), 0.0)

    def test_completely_different(self):
        ted = tree_edit_distance("mul m a", "sin x")
        self.assertGreater(ted, 0.0)

    def test_one_token_diff(self):
        ted = tree_edit_distance("mul m a", "mul m b")
        # Only 1 token different out of 3
        self.assertAlmostEqual(ted, 1.0/3.0, places=3)

    def test_empty(self):
        self.assertEqual(tree_edit_distance("", ""), 0.0)

    def test_one_empty(self):
        self.assertEqual(tree_edit_distance("mul m a", ""), 1.0)


class TestComplexityPenalty(unittest.TestCase):
    def test_same_depth(self):
        self.assertEqual(complexity_penalty("mul m a", "mul x y"), 0.0)

    def test_different_depth(self):
        cp = complexity_penalty("mul m a", "add mul m a mul m b")
        self.assertGreater(cp, 0.0)

    def test_much_deeper(self):
        cp = complexity_penalty(
            "add mul mul a b c d",
            "mul m a"
        )
        self.assertGreater(cp, 0.0)
        self.assertLessEqual(cp, 1.0)


class TestCompositeScore(unittest.TestCase):
    def test_perfect_score(self):
        result = composite_score("mul m a", "mul m a")
        self.assertEqual(result['exact_match'], 1.0)
        self.assertEqual(result['symbolic_equivalence'], 1.0)
        self.assertAlmostEqual(result['numerical_r2'], 1.0, places=2)
        self.assertEqual(result['tree_edit_distance'], 0.0)
        self.assertEqual(result['complexity_penalty'], 0.0)
        self.assertAlmostEqual(result['composite_score'], 100.0, places=0)

    def test_zero_score(self):
        result = composite_score("INT_0", "sin mul x pow y INT_3")
        self.assertEqual(result['exact_match'], 0.0)
        self.assertLess(result['composite_score'], 50.0)

    def test_partial_score(self):
        # Correct structure but wrong constant
        result = composite_score("mul INT_3 x", "mul INT_2 x")
        self.assertEqual(result['exact_match'], 0.0)
        self.assertGreater(result['composite_score'], 0.0)

    def test_score_in_range(self):
        result = composite_score("mul m a", "add m a")
        self.assertGreaterEqual(result['composite_score'], 0.0)
        self.assertLessEqual(result['composite_score'], 100.0)


class TestTreeDepth(unittest.TestCase):
    def test_leaf(self):
        self.assertEqual(_tree_depth("x"), 0)

    def test_simple_op(self):
        self.assertEqual(_tree_depth("mul m a"), 1)

    def test_nested(self):
        self.assertEqual(_tree_depth("add v0 mul a t"), 2)


if __name__ == '__main__':
    unittest.main()
