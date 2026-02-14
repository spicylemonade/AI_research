"""Unit tests for evaluation metrics."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from evaluation.metrics import (
    exact_symbolic_match, normalized_edit_distance, numerical_r2,
    symbolic_complexity_ratio, dimensional_consistency_check,
    compute_all_metrics, tier_stratified_report, paired_bootstrap_test
)


def test_exact_match():
    """Test exact symbolic match after simplification."""
    # Identical expressions
    assert exact_symbolic_match(['mul', 'x_0', 'x_1'], ['mul', 'x_0', 'x_1']) == True

    # Commutative equivalence: x*y == y*x
    assert exact_symbolic_match(['mul', 'x_0', 'x_1'], ['mul', 'x_1', 'x_0']) == True

    # Different expressions
    assert exact_symbolic_match(['mul', 'x_0', 'x_1'], ['add', 'x_0', 'x_1']) == False

    # Algebraic equivalence: 2*x == x + x
    assert exact_symbolic_match(
        ['mul', 'int_2', 'x_0'],
        ['add', 'x_0', 'x_0']
    ) == True

    print("  [PASS] Exact symbolic match test")


def test_ned():
    """Test Normalized Edit Distance."""
    # Identical sequences
    assert normalized_edit_distance(['mul', 'x_0', 'x_1'], ['mul', 'x_0', 'x_1']) == 0.0

    # Completely different
    ned = normalized_edit_distance(['mul', 'x_0', 'x_1'], ['add', 'x_2', 'x_3'])
    assert ned > 0.5

    # One edit
    ned = normalized_edit_distance(['mul', 'x_0', 'x_1'], ['mul', 'x_0', 'x_2'])
    assert 0 < ned < 0.5

    # Empty
    assert normalized_edit_distance([], []) == 0.0
    assert normalized_edit_distance(['x_0'], []) == 1.0

    print("  [PASS] Normalized Edit Distance test")


def test_r2():
    """Test numerical R^2 score."""
    # Perfect prediction: same expression
    rng = np.random.default_rng(42)
    table = np.column_stack([rng.uniform(0.1, 10, 200), rng.uniform(0.1, 10, 200)])
    table = np.column_stack([table, table[:, 0] * table[:, 1]])

    r2 = numerical_r2(['mul', 'x_0', 'x_1'], ['mul', 'x_0', 'x_1'], table)
    assert r2 > 0.99, f"Expected R² > 0.99, got {r2}"

    # Bad prediction
    r2_bad = numerical_r2(['add', 'x_0', 'x_1'], ['mul', 'x_0', 'x_1'], table)
    assert r2_bad < 0.9

    print("  [PASS] Numerical R² test")


def test_complexity_ratio():
    """Test symbolic complexity ratio."""
    # Same complexity
    ratio = symbolic_complexity_ratio(['mul', 'x_0', 'x_1'], ['mul', 'x_0', 'x_1'])
    assert ratio == 1.0

    # More complex prediction
    ratio = symbolic_complexity_ratio(
        ['mul', 'mul', 'x_0', 'x_1', 'x_2'],  # 5 nodes
        ['mul', 'x_0', 'x_1']  # 3 nodes
    )
    assert ratio > 1.0

    print("  [PASS] Symbolic complexity ratio test")


def test_dimensional_consistency():
    """Test dimensional consistency check."""
    # F = m * a  (consistent: [1,0,0] * [0,1,-2] = [1,1,-2])
    var_units = {'x_0': [1, 0, 0], 'x_1': [0, 1, -2]}
    assert dimensional_consistency_check(['mul', 'x_0', 'x_1'], var_units) == True

    # F = m + a  (inconsistent: can't add [1,0,0] + [0,1,-2])
    assert dimensional_consistency_check(['add', 'x_0', 'x_1'], var_units) == False

    # E = 0.5 * m * v^2
    var_units2 = {'x_0': [1, 0, 0], 'x_1': [0, 1, -1]}
    assert dimensional_consistency_check(
        ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'], var_units2) == True

    # sin(angle) - dimensionless angle OK
    var_units3 = {'x_0': [0, 0, 0]}
    assert dimensional_consistency_check(['sin', 'x_0'], var_units3) == True

    # sin(mass) - dimensioned argument NOT OK
    var_units4 = {'x_0': [1, 0, 0]}
    assert dimensional_consistency_check(['sin', 'x_0'], var_units4) == False

    print("  [PASS] Dimensional consistency test")


def test_all_metrics():
    """Test computing all metrics together."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 10, 200)
    y = rng.uniform(0.1, 10, 200)
    table = np.column_stack([x, y, x * y])

    metrics = compute_all_metrics(
        ['mul', 'x_0', 'x_1'],
        ['mul', 'x_0', 'x_1'],
        table,
        variable_units={'x_0': [1, 0, 0], 'x_1': [0, 1, -1]},
    )

    assert metrics['exact_match'] == True
    assert metrics['ned'] == 0.0
    assert metrics['r2'] > 0.99
    assert metrics['complexity_ratio'] == 1.0
    assert metrics['dim_consistent'] == True

    print("  [PASS] All metrics test")


def test_tier_report():
    """Test tier-stratified reporting."""
    results = [
        {'exact_match': True, 'ned': 0.0, 'r2': 1.0, 'complexity_ratio': 1.0, 'dim_consistent': True},
        {'exact_match': False, 'ned': 0.3, 'r2': 0.8, 'complexity_ratio': 1.2, 'dim_consistent': True},
        {'exact_match': True, 'ned': 0.0, 'r2': 0.99, 'complexity_ratio': 1.0, 'dim_consistent': True},
        {'exact_match': False, 'ned': 0.5, 'r2': 0.5, 'complexity_ratio': 1.5, 'dim_consistent': False},
    ]
    tiers = [1, 1, 2, 2]

    report = tier_stratified_report(results, tiers)
    assert 'tier_1' in report
    assert 'tier_2' in report
    assert 'overall' in report
    assert report['tier_1']['exact_match_rate'] == 0.5
    assert report['overall']['count'] == 4

    print("  [PASS] Tier-stratified report test")


def test_bootstrap():
    """Test paired bootstrap test."""
    rng = np.random.default_rng(42)
    a = rng.uniform(0.7, 1.0, 30).tolist()
    b = rng.uniform(0.3, 0.6, 30).tolist()

    p = paired_bootstrap_test(a, b)
    assert p < 0.05, f"Expected p < 0.05, got {p}"

    # Same distribution should not be significant
    p2 = paired_bootstrap_test(a, a)
    assert p2 > 0.05

    print("  [PASS] Bootstrap test")


if __name__ == '__main__':
    print("Running evaluation metrics unit tests...")
    test_exact_match()
    test_ned()
    test_r2()
    test_complexity_ratio()
    test_dimensional_consistency()
    test_all_metrics()
    test_tier_report()
    test_bootstrap()
    print("\nAll metric tests passed!")
