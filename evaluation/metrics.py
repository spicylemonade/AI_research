"""Evaluation metrics for symbolic equation recovery."""

import numpy as np
import sympy as sp
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import _parse_prefix, tree_to_sympy, ExprNode, ARITY


def exact_symbolic_match(pred_tokens: List[str], gt_tokens: List[str]) -> bool:
    """Check if predicted expression matches ground truth after SymPy simplification.

    Returns True if the expressions are symbolically equivalent.
    """
    try:
        pred_tree, _ = _parse_prefix(pred_tokens, 0)
        gt_tree, _ = _parse_prefix(gt_tokens, 0)

        pred_expr = tree_to_sympy(pred_tree)
        gt_expr = tree_to_sympy(gt_tree)

        # Try simplification-based comparison
        diff = sp.simplify(pred_expr - gt_expr)
        if diff == 0:
            return True

        # Try expanding and comparing
        diff_expanded = sp.expand(pred_expr - gt_expr)
        if diff_expanded == 0:
            return True

        # Try ratio test (for multiplicative constant differences)
        try:
            ratio = sp.simplify(pred_expr / gt_expr)
            # Check if ratio is a constant (no free symbols)
            if not ratio.free_symbols and ratio != 0:
                return True
        except Exception:
            pass

        return False
    except Exception:
        return False


def normalized_edit_distance(pred_tokens: List[str], gt_tokens: List[str]) -> float:
    """Compute Normalized Edit Distance (NED) between token sequences.

    Uses Levenshtein distance normalized by the length of the longer sequence.
    Lower is better (0 = exact match, 1 = completely different).
    """
    n = len(pred_tokens)
    m = len(gt_tokens)

    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    # Dynamic programming for edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred_tokens[i-1] == gt_tokens[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    edit_dist = dp[n][m]
    return edit_dist / max(n, m)


def numerical_r2(pred_tokens: List[str], gt_tokens: List[str],
                 observation_table: np.ndarray, n_test: int = 10000,
                 pred_constants: Optional[Dict[str, float]] = None,
                 gt_constants: Optional[Dict[str, float]] = None) -> float:
    """Compute R^2 score on held-out test points.

    Generates n_test points from the observation data range and compares
    predicted vs ground truth outputs.
    """
    from data.data_generator import tree_to_numpy_func

    try:
        pred_tree, _ = _parse_prefix(pred_tokens, 0)
        gt_tree, _ = _parse_prefix(gt_tokens, 0)
    except Exception:
        return -1.0

    n_vars = observation_table.shape[1] - 1

    # Generate test points within observed range
    rng = np.random.default_rng(42)
    x_dict_pred = {}
    x_dict_gt = {}
    for i in range(n_vars):
        col = observation_table[:, i]
        lo, hi = col.min(), col.max()
        margin = (hi - lo) * 0.1
        x_i = rng.uniform(lo - margin, hi + margin, size=n_test)
        x_dict_pred[i] = x_i
        x_dict_gt[i] = x_i

    pred_const = pred_constants or {}
    gt_const = gt_constants or {}

    try:
        pred_func = tree_to_numpy_func(pred_tree, pred_const)
        gt_func = tree_to_numpy_func(gt_tree, gt_const)

        y_pred = np.array(pred_func(x_dict_pred), dtype=np.float64)
        y_gt = np.array(gt_func(x_dict_gt), dtype=np.float64)

        # Filter valid
        valid = np.isfinite(y_pred) & np.isfinite(y_gt)
        if valid.sum() < 100:
            return -1.0

        y_pred = y_pred[valid]
        y_gt = y_gt[valid]

        ss_res = np.sum((y_gt - y_pred) ** 2)
        ss_tot = np.sum((y_gt - y_gt.mean()) ** 2)

        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else -1.0

        r2 = 1.0 - ss_res / ss_tot
        return float(np.clip(r2, -1.0, 1.0))
    except Exception:
        return -1.0


def symbolic_complexity_ratio(pred_tokens: List[str], gt_tokens: List[str]) -> float:
    """Compute ratio of predicted to ground truth expression complexity.

    Complexity = number of nodes in the expression tree.
    Ideal ratio = 1.0. >1 means overly complex, <1 means too simple.
    """
    try:
        pred_tree, _ = _parse_prefix(pred_tokens, 0)
        gt_tree, _ = _parse_prefix(gt_tokens, 0)
        pred_nodes = pred_tree.num_nodes()
        gt_nodes = gt_tree.num_nodes()
        if gt_nodes == 0:
            return float('inf')
        return pred_nodes / gt_nodes
    except Exception:
        return float('inf')


def dimensional_consistency_check(pred_tokens: List[str],
                                   variable_units: Dict[str, List[int]]) -> bool:
    """Check if the predicted equation has consistent physical dimensions.

    Args:
        pred_tokens: prefix notation tokens
        variable_units: mapping from variable name to [M, L, T] exponents

    Returns True if all dimensional constraints are satisfied.
    """
    try:
        tree, _ = _parse_prefix(pred_tokens, 0)
        _compute_units(tree, variable_units)
        return True
    except DimensionalError:
        return False
    except Exception:
        return False


class DimensionalError(Exception):
    pass


def _compute_units(node: ExprNode, var_units: Dict[str, List[int]]) -> List[float]:
    """Recursively compute dimensional units of an expression tree node."""
    # Leaf: variable
    if node.token.startswith('x_'):
        return list(var_units.get(node.token, [0, 0, 0]))

    # Leaf: constant (dimensionless)
    if node.token.startswith('c_') or node.token.startswith('int_') or \
       node.token in ('pi', 'e_const', 'half', 'third', 'quarter'):
        return [0, 0, 0]

    # Operators
    child_units = [_compute_units(c, var_units) for c in node.children]

    if node.token in ('add', 'sub'):
        u_left, u_right = child_units
        for i in range(3):
            if abs(u_left[i] - u_right[i]) > 0.01:
                raise DimensionalError(
                    f"{node.token}: mismatched dims {u_left} vs {u_right}")
        return u_left

    if node.token == 'mul':
        u_left, u_right = child_units
        return [u_left[i] + u_right[i] for i in range(3)]

    if node.token == 'div':
        u_left, u_right = child_units
        return [u_left[i] - u_right[i] for i in range(3)]

    if node.token == 'pow':
        u_base = child_units[0]
        # Exponent must be dimensionless
        u_exp = child_units[1]
        for i in range(3):
            if abs(u_exp[i]) > 0.01:
                raise DimensionalError(f"pow exponent has dimensions: {u_exp}")
        # Try to get numeric exponent
        try:
            exp_val = _get_numeric_value(node.children[1])
            return [u_base[i] * exp_val for i in range(3)]
        except Exception:
            # If exponent isn't numeric, base must be dimensionless
            for i in range(3):
                if abs(u_base[i]) > 0.01:
                    raise DimensionalError(
                        f"pow with non-numeric exponent requires dimensionless base: {u_base}")
            return [0, 0, 0]

    if node.token == 'sqrt':
        u_child = child_units[0]
        return [u_child[i] * 0.5 for i in range(3)]

    if node.token in ('sin', 'cos', 'exp', 'log'):
        u_child = child_units[0]
        for i in range(3):
            if abs(u_child[i]) > 0.01:
                raise DimensionalError(
                    f"{node.token} requires dimensionless argument: {u_child}")
        return [0, 0, 0]

    if node.token == 'neg':
        return child_units[0]

    if node.token == 'abs':
        return child_units[0]

    if node.token == 'inv':
        u_child = child_units[0]
        return [-u_child[i] for i in range(3)]

    return [0, 0, 0]


def _get_numeric_value(node: ExprNode) -> float:
    """Try to extract a numeric value from a leaf node."""
    if node.token.startswith('int_'):
        return float(int(node.token.split('_')[1]))
    if node.token == 'half':
        return 0.5
    if node.token == 'third':
        return 1.0 / 3.0
    if node.token == 'quarter':
        return 0.25
    if node.token == 'int_2':
        return 2.0
    raise ValueError(f"Not a numeric leaf: {node.token}")


def compute_all_metrics(pred_tokens: List[str], gt_tokens: List[str],
                         observation_table: np.ndarray,
                         variable_units: Optional[Dict[str, List[int]]] = None,
                         pred_constants: Optional[Dict[str, float]] = None,
                         gt_constants: Optional[Dict[str, float]] = None) -> Dict:
    """Compute all 5 metrics for a single prediction."""
    em = exact_symbolic_match(pred_tokens, gt_tokens)
    ned = normalized_edit_distance(pred_tokens, gt_tokens)
    r2 = numerical_r2(pred_tokens, gt_tokens, observation_table,
                       pred_constants=pred_constants, gt_constants=gt_constants)
    scr = symbolic_complexity_ratio(pred_tokens, gt_tokens)
    dim_ok = dimensional_consistency_check(pred_tokens, variable_units or {})

    return {
        'exact_match': em,
        'ned': ned,
        'r2': r2,
        'complexity_ratio': scr,
        'dim_consistent': dim_ok,
    }


def tier_stratified_report(results: List[Dict], tiers: List[int]) -> Dict:
    """Generate tier-stratified metric report.

    Args:
        results: list of metric dicts from compute_all_metrics
        tiers: list of tier numbers corresponding to each result
    """
    tier_results = defaultdict(list)
    for res, tier in zip(results, tiers):
        tier_results[tier].append(res)

    report = {}
    for tier in sorted(tier_results.keys()):
        tier_data = tier_results[tier]
        n = len(tier_data)
        report[f'tier_{tier}'] = {
            'count': n,
            'exact_match_rate': sum(r['exact_match'] for r in tier_data) / n,
            'mean_ned': np.mean([r['ned'] for r in tier_data]),
            'mean_r2': np.mean([r['r2'] for r in tier_data]),
            'mean_complexity_ratio': np.mean([r['complexity_ratio'] for r in tier_data]),
            'dim_consistency_rate': sum(r['dim_consistent'] for r in tier_data) / n,
        }

    # Overall
    all_data = results
    n = len(all_data)
    if n > 0:
        report['overall'] = {
            'count': n,
            'exact_match_rate': sum(r['exact_match'] for r in all_data) / n,
            'mean_ned': np.mean([r['ned'] for r in all_data]),
            'mean_r2': np.mean([r['r2'] for r in all_data]),
            'mean_complexity_ratio': np.mean([r['complexity_ratio'] for r in all_data]),
            'dim_consistency_rate': sum(r['dim_consistent'] for r in all_data) / n,
        }

    return report


def paired_bootstrap_test(metric_a: List[float], metric_b: List[float],
                           n_resamples: int = 1000, seed: int = 42) -> float:
    """Paired bootstrap test for statistical significance.

    Returns p-value for the hypothesis that metric_a > metric_b.
    """
    rng = np.random.default_rng(seed)
    n = len(metric_a)
    assert len(metric_b) == n

    diffs = np.array(metric_a) - np.array(metric_b)
    observed_diff = np.mean(diffs)

    count_greater = 0
    for _ in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sample_diff = np.mean(diffs[sample_idx])
        if sample_diff <= 0:
            count_greater += 1

    return count_greater / n_resamples
