"""Evaluation metrics suite for symbolic regression.

Computes: solution rate, R², RMSE, NED, symbolic accuracy, inference time.
"""

import time
import numpy as np
import sympy
from sympy import Symbol, simplify, N
from typing import Dict, List, Optional, Tuple


SYMBOLS = {f'x{i}': Symbol(f'x{i}') for i in range(1, 10)}


def _parse_expr(expr_str: str):
    """Safely parse an expression string to SymPy."""
    local_dict = dict(SYMBOLS)
    local_dict['pi'] = sympy.pi
    local_dict['e'] = sympy.E
    local_dict['E'] = sympy.E
    try:
        return sympy.sympify(expr_str, locals=local_dict)
    except Exception:
        try:
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            return parse_expr(expr_str, local_dict=local_dict,
                              transformations=standard_transformations + (implicit_multiplication_application,))
        except Exception:
            return None


def _expr_to_tree(expr) -> list:
    """Convert SymPy expression to a nested list (tree) representation."""
    if expr is None:
        return ['NULL']
    if isinstance(expr, Symbol):
        return [str(expr)]
    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        return [str(float(expr))]
    if expr.is_number:
        return [str(float(N(expr)))]
    func_name = type(expr).__name__
    children = [_expr_to_tree(arg) for arg in expr.args]
    return [func_name] + children


def _tree_edit_distance(tree1: list, tree2: list) -> int:
    """Compute tree edit distance between two tree representations.

    Simple recursive implementation: cost is sum of mismatches and
    size differences.
    """
    if not tree1 and not tree2:
        return 0
    if not tree1:
        return _tree_size(tree2)
    if not tree2:
        return _tree_size(tree1)

    # Compare root labels
    cost = 0 if tree1[0] == tree2[0] else 1

    # Compare children
    children1 = tree1[1:] if len(tree1) > 1 else []
    children2 = tree2[1:] if len(tree2) > 1 else []

    # Pad shorter list
    max_children = max(len(children1), len(children2))
    while len(children1) < max_children:
        children1.append([])
    while len(children2) < max_children:
        children2.append([])

    for c1, c2 in zip(children1, children2):
        cost += _tree_edit_distance(c1, c2)

    return cost


def _tree_size(tree: list) -> int:
    """Count number of nodes in a tree."""
    if not tree:
        return 0
    size = 1
    for child in tree[1:]:
        if isinstance(child, list):
            size += _tree_size(child)
    return size


def solution_rate(pred_str: str, true_str: str) -> float:
    """Check if predicted expression is symbolically equivalent to ground truth.

    Uses SymPy simplification and equivalence checking.

    Returns:
        1.0 if equivalent, 0.0 otherwise.
    """
    pred_expr = _parse_expr(pred_str)
    true_expr = _parse_expr(true_str)

    if pred_expr is None or true_expr is None:
        return 0.0

    try:
        diff = simplify(pred_expr - true_expr)
        if diff == 0:
            return 1.0
    except Exception:
        pass

    # Numerical check as fallback
    try:
        test_points = [
            {Symbol(f'x{i}'): float(v) for i, v in enumerate([1.1, 2.3, 0.7, 1.5, 3.2, 0.9, 2.1, 1.8, 0.5], 1)}
        ]
        for vals in test_points:
            v_pred = complex(pred_expr.subs(vals))
            v_true = complex(true_expr.subs(vals))
            if not (np.isfinite(v_pred.real) and np.isfinite(v_true.real)):
                return 0.0
            if abs(v_pred - v_true) > 1e-6 * max(abs(v_true), 1):
                return 0.0
        return 1.0
    except Exception:
        return 0.0


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination).

    Returns:
        R² value. 1.0 = perfect fit, <0 = worse than mean prediction.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return float(1 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def normalized_edit_distance(pred_str: str, true_str: str) -> float:
    """Compute Normalized Edit Distance between expression trees.

    NED = tree_edit_distance(pred, true) / tree_size(true)

    Returns:
        NED value. 0.0 = exact match, higher = more different.
    """
    pred_expr = _parse_expr(pred_str)
    true_expr = _parse_expr(true_str)

    if pred_expr is None or true_expr is None:
        return 1.0

    pred_tree = _expr_to_tree(pred_expr)
    true_tree = _expr_to_tree(true_expr)

    dist = _tree_edit_distance(pred_tree, true_tree)
    size = _tree_size(true_tree)

    if size == 0:
        return 0.0 if dist == 0 else 1.0

    return min(float(dist) / size, 1.0)


def symbolic_accuracy(pred_tokens: List[int], true_tokens: List[int]) -> float:
    """Compute token-level symbolic accuracy.

    Fraction of correctly predicted tokens (ignoring PAD tokens).

    Args:
        pred_tokens: Predicted token ID sequence.
        true_tokens: Ground-truth token ID sequence.

    Returns:
        Accuracy between 0.0 and 1.0.
    """
    # Remove padding (token 0)
    pred = [t for t in pred_tokens if t != 0]
    true = [t for t in true_tokens if t != 0]

    if not true:
        return 1.0 if not pred else 0.0

    # Compare up to shorter length
    max_len = max(len(pred), len(true))
    correct = 0
    for i in range(min(len(pred), len(true))):
        if pred[i] == true[i]:
            correct += 1

    return correct / max_len


def compute_r2_from_expr(pred_str: str, true_str: str,
                         X: np.ndarray, y_true: np.ndarray) -> float:
    """Compute R² by evaluating predicted expression on test data.

    Args:
        pred_str: Predicted expression string.
        true_str: Ground-truth expression string (unused, for reference).
        X: (n_points, n_vars) input data matrix.
        y_true: (n_points,) ground-truth output values.

    Returns:
        R² value.
    """
    pred_expr = _parse_expr(pred_str)
    if pred_expr is None:
        return 0.0

    n_vars = X.shape[1]
    y_pred = []
    for i in range(len(X)):
        vals = {Symbol(f'x{j+1}'): float(X[i, j]) for j in range(n_vars)}
        try:
            val = complex(pred_expr.subs(vals))
            if np.isfinite(val.real) and abs(val.imag) < 1e-10:
                y_pred.append(val.real)
            else:
                y_pred.append(np.nan)
        except Exception:
            y_pred.append(np.nan)

    y_pred = np.array(y_pred)
    valid = np.isfinite(y_pred)
    if np.sum(valid) < 2:
        return 0.0

    return r_squared(y_true[valid], y_pred[valid])


def compute_all_metrics(pred_str: str, true_str: str,
                        X: Optional[np.ndarray] = None,
                        y_true: Optional[np.ndarray] = None,
                        pred_tokens: Optional[List[int]] = None,
                        true_tokens: Optional[List[int]] = None,
                        inference_time: float = 0.0) -> Dict[str, float]:
    """Compute all 6 evaluation metrics.

    Args:
        pred_str: Predicted equation string.
        true_str: Ground-truth equation string.
        X: Optional input data matrix for R² computation.
        y_true: Optional ground-truth outputs for R² computation.
        pred_tokens: Optional predicted token IDs for symbolic accuracy.
        true_tokens: Optional ground-truth token IDs for symbolic accuracy.
        inference_time: Wall-clock inference time in seconds.

    Returns:
        Dictionary with all 6 metrics.
    """
    metrics = {}

    # 1. Solution rate
    metrics['solution_rate'] = solution_rate(pred_str, true_str)

    # 2. R² (if data provided)
    if X is not None and y_true is not None:
        metrics['r_squared'] = compute_r2_from_expr(pred_str, true_str, X, y_true)
    else:
        metrics['r_squared'] = None

    # 3. RMSE (if data provided)
    if X is not None and y_true is not None:
        pred_expr = _parse_expr(pred_str)
        if pred_expr is not None:
            n_vars = X.shape[1]
            y_pred = []
            for i in range(len(X)):
                vals = {Symbol(f'x{j+1}'): float(X[i, j]) for j in range(n_vars)}
                try:
                    val = complex(pred_expr.subs(vals))
                    if np.isfinite(val.real):
                        y_pred.append(val.real)
                    else:
                        y_pred.append(0.0)
                except Exception:
                    y_pred.append(0.0)
            metrics['rmse'] = rmse(y_true, np.array(y_pred))
        else:
            metrics['rmse'] = float('inf')
    else:
        metrics['rmse'] = None

    # 4. Normalized Edit Distance
    metrics['ned'] = normalized_edit_distance(pred_str, true_str)

    # 5. Symbolic accuracy (if tokens provided)
    if pred_tokens is not None and true_tokens is not None:
        metrics['symbolic_accuracy'] = symbolic_accuracy(pred_tokens, true_tokens)
    else:
        metrics['symbolic_accuracy'] = None

    # 6. Inference time
    metrics['inference_time'] = inference_time

    return metrics
