"""
Evaluation metrics for symbolic equation derivation.

Implements 5 metrics + composite score for comparing predicted equations
against ground truth: exact_match, symbolic_equivalence, numerical_r2,
tree_edit_distance, and complexity_penalty.

References:
    - lample2020deep: evaluation methodology for symbolic math
    - udrescu2020ai: AI Feynman evaluation criteria
"""

import math
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from typing import Dict, List, Optional, Tuple, Any


def _safe_parse(expr_str: str) -> Optional[sp.Expr]:
    """Safely parse an expression string into a sympy expression."""
    try:
        return parse_expr(
            expr_str,
            transformations=standard_transformations + (implicit_multiplication_application,)
        )
    except Exception:
        return None


def _tree_depth(expr: sp.Expr) -> int:
    """Compute the depth of a sympy expression tree."""
    if expr.args:
        return 1 + max(_tree_depth(a) for a in expr.args)
    return 0


def _tree_size(expr: sp.Expr) -> int:
    """Compute the number of nodes in a sympy expression tree."""
    return 1 + sum(_tree_size(a) for a in expr.args)


def _tree_edit_distance_recursive(t1: sp.Expr, t2: sp.Expr) -> int:
    """Simple recursive tree edit distance (Zhang-Shasha simplified)."""
    if t1 == t2:
        return 0

    # If both are leaves
    if not t1.args and not t2.args:
        return 0 if t1 == t2 else 1

    # If one is leaf, other is not
    if not t1.args:
        return _tree_size(t2)
    if not t2.args:
        return _tree_size(t1)

    # Both have children
    # Cost of substituting root
    root_cost = 0 if type(t1).__name__ == type(t2).__name__ else 1

    # Align children
    args1 = list(t1.args)
    args2 = list(t2.args)

    n1, n2 = len(args1), len(args2)

    if n1 == 0 and n2 == 0:
        return root_cost

    # Simple alignment: pad shorter list
    child_cost = 0
    for i in range(max(n1, n2)):
        if i < n1 and i < n2:
            child_cost += _tree_edit_distance_recursive(args1[i], args2[i])
        elif i < n1:
            child_cost += _tree_size(args1[i])
        else:
            child_cost += _tree_size(args2[i])

    return root_cost + child_cost


# ─── Metric Functions ────────────────────────────────────────────────────────

def exact_match(pred_str: str, true_str: str) -> float:
    """Check if predicted equation is identical to ground truth after simplification.

    Returns 1.0 if match, 0.0 otherwise.
    """
    pred = _safe_parse(pred_str)
    true = _safe_parse(true_str)
    if pred is None or true is None:
        return 0.0

    try:
        pred_simp = sp.simplify(pred)
        true_simp = sp.simplify(true)
        return 1.0 if pred_simp == true_simp else 0.0
    except Exception:
        return 0.0


def symbolic_equivalence(pred_str: str, true_str: str) -> float:
    """Check if predicted equation is symbolically equivalent to ground truth.

    Uses sympy.simplify(pred - true) == 0 and numerical random testing.
    Returns 1.0 if equivalent, 0.0 otherwise.
    """
    pred = _safe_parse(pred_str)
    true = _safe_parse(true_str)
    if pred is None or true is None:
        return 0.0

    try:
        diff = sp.simplify(pred - true)
        if diff == 0:
            return 1.0
    except Exception:
        pass

    # Fallback: numerical equivalence check
    try:
        free_syms = list(pred.free_symbols | true.free_symbols)
        if not free_syms:
            pred_val = float(pred.evalf())
            true_val = float(true.evalf())
            return 1.0 if abs(pred_val - true_val) < 1e-6 else 0.0

        rng = np.random.RandomState(42)
        n_tests = 20
        all_close = True
        for _ in range(n_tests):
            subs = {s: rng.uniform(0.1, 5.0) for s in free_syms}
            try:
                pv = float(pred.subs(subs).evalf())
                tv = float(true.subs(subs).evalf())
                if not math.isfinite(pv) or not math.isfinite(tv):
                    continue
                if abs(pv - tv) > 1e-4 * max(1, abs(tv)):
                    all_close = False
                    break
            except Exception:
                continue

        return 1.0 if all_close else 0.0
    except Exception:
        return 0.0


def numerical_r2(pred_str: str, true_str: str,
                 x_data: Optional[List[Dict]] = None,
                 n_points: int = 100) -> float:
    """Compute R² score between predicted and true equations on data points.

    If x_data is not provided, generates random test points.
    Returns R² score clipped to [0, 1].
    """
    pred = _safe_parse(pred_str)
    true = _safe_parse(true_str)
    if pred is None or true is None:
        return 0.0

    free_syms = list(pred.free_symbols | true.free_symbols)
    rng = np.random.RandomState(42)

    y_pred = []
    y_true = []

    if x_data:
        for point in x_data:
            subs = {sp.Symbol(k): v for k, v in point.items() if sp.Symbol(k) in set(free_syms)}
            try:
                pv = float(pred.subs(subs).evalf())
                tv = float(true.subs(subs).evalf())
                if math.isfinite(pv) and math.isfinite(tv):
                    y_pred.append(pv)
                    y_true.append(tv)
            except Exception:
                continue
    else:
        for _ in range(n_points):
            subs = {s: rng.uniform(0.1, 5.0) for s in free_syms}
            try:
                pv = float(pred.subs(subs).evalf())
                tv = float(true.subs(subs).evalf())
                if math.isfinite(pv) and math.isfinite(tv):
                    y_pred.append(pv)
                    y_true.append(tv)
            except Exception:
                continue

    if len(y_true) < 5:
        return 0.0

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0

    r2 = 1.0 - ss_res / ss_tot
    return max(0.0, min(1.0, r2))


def tree_edit_distance(pred_str: str, true_str: str) -> float:
    """Compute normalized tree edit distance between predicted and true expression trees.

    Returns value in [0, 1] where 0 means identical trees and 1 means maximum difference.
    """
    pred = _safe_parse(pred_str)
    true = _safe_parse(true_str)
    if pred is None or true is None:
        return 1.0

    try:
        raw_dist = _tree_edit_distance_recursive(pred, true)
        max_size = max(_tree_size(pred), _tree_size(true))
        if max_size == 0:
            return 0.0
        return min(1.0, raw_dist / max_size)
    except Exception:
        return 1.0


def complexity_penalty(pred_str: str, true_str: str) -> float:
    """Compute complexity penalty: ratio of predicted to true expression tree depth.

    Returns value in [0, inf) where 1.0 means same complexity,
    >1.0 means over-complex, <1.0 means under-complex.
    Clipped to [0, 3] and normalized to [0, 1] as penalty/3.
    """
    pred = _safe_parse(pred_str)
    true = _safe_parse(true_str)
    if pred is None or true is None:
        return 1.0

    try:
        pred_depth = _tree_depth(pred)
        true_depth = _tree_depth(true)
        if true_depth == 0:
            return 0.0 if pred_depth == 0 else 1.0
        ratio = abs(pred_depth - true_depth) / true_depth
        return min(1.0, ratio)
    except Exception:
        return 1.0


def composite_score(pred_str: str, true_str: str,
                    x_data: Optional[List[Dict]] = None) -> float:
    """Compute the composite evaluation score.

    S = 0.3*exact_match + 0.3*symbolic_equivalence + 0.25*numerical_r2
        + 0.1*(1 - tree_edit_distance) + 0.05*(1 - complexity_penalty)
    """
    em = exact_match(pred_str, true_str)
    se = symbolic_equivalence(pred_str, true_str)
    r2 = numerical_r2(pred_str, true_str, x_data)
    ted = tree_edit_distance(pred_str, true_str)
    cp = complexity_penalty(pred_str, true_str)

    score = (0.3 * em + 0.3 * se + 0.25 * r2 +
             0.1 * (1 - ted) + 0.05 * (1 - cp))
    return score


def evaluate_batch(predictions: List[str], ground_truths: List[str],
                   x_data_list: Optional[List[Optional[List[Dict]]]] = None
                   ) -> Dict[str, Any]:
    """Evaluate a batch of predictions against ground truths.

    Returns dict with all metrics averaged over the batch.
    """
    n = len(predictions)
    assert n == len(ground_truths)

    metrics = {
        "exact_match": [],
        "symbolic_equivalence": [],
        "numerical_r2": [],
        "tree_edit_distance": [],
        "complexity_penalty": [],
        "composite": [],
    }

    for i in range(n):
        x_data = x_data_list[i] if x_data_list else None
        em = exact_match(predictions[i], ground_truths[i])
        se = symbolic_equivalence(predictions[i], ground_truths[i])
        r2 = numerical_r2(predictions[i], ground_truths[i], x_data)
        ted = tree_edit_distance(predictions[i], ground_truths[i])
        cp = complexity_penalty(predictions[i], ground_truths[i])
        comp = 0.3 * em + 0.3 * se + 0.25 * r2 + 0.1 * (1 - ted) + 0.05 * (1 - cp)

        metrics["exact_match"].append(em)
        metrics["symbolic_equivalence"].append(se)
        metrics["numerical_r2"].append(r2)
        metrics["tree_edit_distance"].append(ted)
        metrics["complexity_penalty"].append(cp)
        metrics["composite"].append(comp)

    return {k: float(np.mean(v)) for k, v in metrics.items()}
