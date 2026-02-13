"""Unit tests for the evaluation metrics suite."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.metrics import (
    exact_match, symbolic_equivalence, numerical_r2,
    tree_edit_distance, complexity_penalty, composite_score,
    evaluate_batch,
)


class TestExactMatch:
    def test_identical(self):
        assert exact_match("x + 1", "x + 1") == 1.0

    def test_different(self):
        assert exact_match("x + 1", "x + 2") == 0.0

    def test_simplified_match(self):
        assert exact_match("x + x", "2*x") == 1.0

    def test_invalid_input(self):
        assert exact_match("???", "x + 1") == 0.0


class TestSymbolicEquivalence:
    def test_same_expression(self):
        assert symbolic_equivalence("x**2 + 2*x + 1", "(x+1)**2") == 1.0

    def test_different_form_same_value(self):
        assert symbolic_equivalence("sin(x)**2 + cos(x)**2", "1") == 1.0

    def test_not_equivalent(self):
        assert symbolic_equivalence("x**2", "x**3") == 0.0

    def test_constants(self):
        assert symbolic_equivalence("3.14159", "3.14159") == 1.0


class TestNumericalR2:
    def test_perfect_prediction(self):
        r2 = numerical_r2("x**2", "x**2")
        assert r2 > 0.99

    def test_wrong_prediction(self):
        r2 = numerical_r2("x", "x**3")
        assert r2 < 0.5

    def test_constant(self):
        r2 = numerical_r2("5", "5")
        assert r2 == 1.0


class TestTreeEditDistance:
    def test_identical(self):
        assert tree_edit_distance("x + y", "x + y") == 0.0

    def test_one_op_different(self):
        dist = tree_edit_distance("x + y", "x * y")
        assert 0 < dist <= 1.0

    def test_very_different(self):
        dist = tree_edit_distance("x", "sin(x**2 + y**2)")
        assert dist > 0.5

    def test_invalid(self):
        assert tree_edit_distance("???", "x") == 1.0


class TestComplexityPenalty:
    def test_same_complexity(self):
        cp = complexity_penalty("x + y", "a + b")
        assert cp == 0.0

    def test_more_complex_prediction(self):
        cp = complexity_penalty("sin(x**2 + y**2)", "x")
        assert cp > 0.0

    def test_invalid(self):
        assert complexity_penalty("???", "x") == 1.0


class TestCompositeScore:
    def test_perfect_score(self):
        score = composite_score("x**2", "x**2")
        assert score > 0.9

    def test_bad_score(self):
        score = composite_score("1", "sin(x**2)")
        assert score < 0.3

    def test_score_range(self):
        score = composite_score("x + 1", "x + 1")
        assert 0 <= score <= 1.0


class TestBatchEvaluation:
    def test_batch(self):
        preds = ["x**2", "x + 1", "sin(x)"]
        truths = ["x**2", "x + 1", "cos(x)"]
        results = evaluate_batch(preds, truths)
        assert "composite" in results
        assert "exact_match" in results
        assert 0 <= results["composite"] <= 1.0
        assert results["exact_match"] >= 0.5  # 2/3 match


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
