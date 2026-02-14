"""Tests for the evaluation metrics suite."""

import sys
import numpy as np
import pytest

sys.path.insert(0, '.')
from src.evaluation.metrics import (
    solution_rate, r_squared, rmse, normalized_edit_distance,
    symbolic_accuracy, compute_all_metrics, compute_r2_from_expr,
)


class TestSolutionRate:
    def test_exact_match(self):
        assert solution_rate("x1*x2", "x1*x2") == 1.0

    def test_equivalent_commutative(self):
        assert solution_rate("x2*x1", "x1*x2") == 1.0

    def test_equivalent_expanded(self):
        assert solution_rate("x1**2 + 2*x1*x2 + x2**2", "(x1+x2)**2") == 1.0

    def test_constant_folding(self):
        assert solution_rate("6*x1", "2*3*x1") == 1.0

    def test_not_equivalent(self):
        assert solution_rate("x1+x2", "x1*x2") == 0.0

    def test_partial_match(self):
        assert solution_rate("x1+x2+1", "x1+x2") == 0.0

    def test_invalid_prediction(self):
        assert solution_rate("invalid_expr!!!", "x1+x2") == 0.0

    def test_trig_identity(self):
        # sin²(x) + cos²(x) = 1
        assert solution_rate("sin(x1)**2 + cos(x1)**2", "1") == 1.0


class TestRSquared:
    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r_squared(y, y) == 1.0

    def test_constant_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, 3.0)
        assert r_squared(y_true, y_pred) == 0.0

    def test_negative_r2(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 1.0, 2.0])
        r2 = r_squared(y_true, y_pred)
        assert r2 < 0.5  # Poor fit


class TestRMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert abs(rmse(y_true, y_pred) - 1.0) < 1e-10


class TestNED:
    def test_exact_match(self):
        ned = normalized_edit_distance("x1*x2", "x1*x2")
        assert ned == 0.0

    def test_completely_different(self):
        ned = normalized_edit_distance("x1", "sin(x2*x3)**2 + cos(x4)")
        assert ned > 0.0

    def test_partial_match(self):
        ned = normalized_edit_distance("x1+x2", "x1+x3")
        assert 0.0 < ned < 1.0

    def test_invalid_pred(self):
        ned = normalized_edit_distance("!!!invalid!!!", "x1+x2")
        assert ned == 1.0

    def test_equivalent_forms(self):
        ned = normalized_edit_distance("x2*x1", "x1*x2")
        # Commutative equivalent - should be 0 or very small
        assert ned <= 0.5


class TestSymbolicAccuracy:
    def test_perfect_match(self):
        tokens = [1, 10, 20, 4, 2]
        assert symbolic_accuracy(tokens, tokens) == 1.0

    def test_no_match(self):
        pred = [1, 10, 20, 4, 2]
        true = [1, 30, 40, 5, 2]
        acc = symbolic_accuracy(pred, true)
        assert acc < 1.0

    def test_partial_match(self):
        pred = [1, 10, 20, 2]
        true = [1, 10, 30, 2]
        acc = symbolic_accuracy(pred, true)
        assert 0.0 < acc < 1.0

    def test_empty(self):
        assert symbolic_accuracy([], []) == 1.0

    def test_with_padding(self):
        pred = [1, 10, 2, 0, 0]
        true = [1, 10, 2, 0, 0]
        assert symbolic_accuracy(pred, true) == 1.0


class TestComputeR2FromExpr:
    def test_linear(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_true = np.array([2.0, 4.0, 6.0, 8.0])
        r2 = compute_r2_from_expr("2*x1", "2*x1", X, y_true)
        assert r2 > 0.99

    def test_quadratic(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_true = X[:, 0] ** 2
        r2 = compute_r2_from_expr("x1**2", "x1**2", X, y_true)
        assert r2 > 0.99


class TestComputeAllMetrics:
    def test_all_metrics_returned(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_true = np.array([2.0, 12.0, 30.0])
        pred_tokens = [1, 16, 17, 6, 2]
        true_tokens = [1, 16, 17, 6, 2]
        metrics = compute_all_metrics(
            "x1*x2", "x1*x2",
            X=X, y_true=y_true,
            pred_tokens=pred_tokens, true_tokens=true_tokens,
            inference_time=0.5,
        )
        assert 'solution_rate' in metrics
        assert 'r_squared' in metrics
        assert 'rmse' in metrics
        assert 'ned' in metrics
        assert 'symbolic_accuracy' in metrics
        assert 'inference_time' in metrics
        assert metrics['solution_rate'] == 1.0
        assert metrics['r_squared'] > 0.99
        assert metrics['rmse'] < 0.01
        assert metrics['ned'] == 0.0
        assert metrics['symbolic_accuracy'] == 1.0
        assert metrics['inference_time'] == 0.5

    def test_without_data(self):
        metrics = compute_all_metrics("x1+x2", "x1+x2")
        assert metrics['solution_rate'] == 1.0
        assert metrics['ned'] == 0.0
        assert metrics['r_squared'] is None
        assert metrics['rmse'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
