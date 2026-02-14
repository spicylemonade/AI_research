#!/usr/bin/env python3
"""Evaluation metrics suite for symbolic regression.

Implements 5 metrics + composite scoring:
1. exact_match: via sympy.simplify canonicalization
2. symbolic_equivalence: via sympy.equals with numerical fallback
3. numerical_r2: R² on held-out test points
4. tree_edit_distance: normalized tree edit distance between expression trees
5. complexity_penalty: predicted vs ground-truth tree depth ratio

Composite: S = 0.3*EM + 0.3*SE + 0.25*R2 + 0.1*(1-TED) + 0.05*(1-CP)
"""

import math
import numpy as np
import sympy
from sympy import sympify, simplify, Symbol, N
from typing import Dict, Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Prefix notation -> sympy expression conversion
# ---------------------------------------------------------------------------

OPERATOR_ARITIES = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2,
    'neg': 1,
    'sin': 1, 'cos': 1, 'tan': 1, 'exp': 1, 'log': 1, 'sqrt': 1, 'abs': 1,
    'asin': 1, 'acos': 1, 'atan': 1, 'sinh': 1, 'cosh': 1,
}

SYMPY_OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b,
    'pow': lambda a, b: a ** b,
    'neg': lambda a: -a,
    'sin': lambda a: sympy.sin(a),
    'cos': lambda a: sympy.cos(a),
    'tan': lambda a: sympy.tan(a),
    'exp': lambda a: sympy.exp(a),
    'log': lambda a: sympy.log(a),
    'sqrt': lambda a: sympy.sqrt(a),
    'abs': lambda a: sympy.Abs(a),
    'asin': lambda a: sympy.asin(a),
    'acos': lambda a: sympy.acos(a),
    'atan': lambda a: sympy.atan(a),
    'sinh': lambda a: sympy.sinh(a),
    'cosh': lambda a: sympy.cosh(a),
}

# Map token names to sympy symbols/constants
NAMED_CONSTANTS = {
    'pi': sympy.pi,
    'euler': sympy.E,
    'g_accel': sympy.Float(9.81),
    'G_const': sympy.Float(6.674e-11),
    'c_light': sympy.Float(3e8),
    'k_boltz': sympy.Float(1.38e-23),
    'h_planck': sympy.Float(6.626e-34),
    'epsilon0': sympy.Float(8.854e-12),
}


def prefix_to_sympy(prefix_str: str) -> Optional[sympy.Expr]:
    """Convert prefix notation string to a sympy expression."""
    tokens = prefix_str.strip().split()
    try:
        expr, remaining = _parse_prefix(tokens, 0)
        return expr
    except (IndexError, ValueError, TypeError):
        return None


def _parse_prefix(tokens: list, idx: int) -> Tuple[sympy.Expr, int]:
    """Recursively parse prefix notation tokens into sympy."""
    if idx >= len(tokens):
        raise IndexError("Unexpected end of tokens")

    tok = tokens[idx]

    if tok in OPERATOR_ARITIES:
        arity = OPERATOR_ARITIES[tok]
        op = SYMPY_OPS[tok]
        if arity == 1:
            arg, next_idx = _parse_prefix(tokens, idx + 1)
            return op(arg), next_idx
        elif arity == 2:
            left, next_idx = _parse_prefix(tokens, idx + 1)
            right, next_idx = _parse_prefix(tokens, next_idx)
            return op(left, right), next_idx
    elif tok.startswith('INT_'):
        val = int(tok.split('_')[1])
        return sympy.Integer(val), idx + 1
    elif tok in NAMED_CONSTANTS:
        return NAMED_CONSTANTS[tok], idx + 1
    elif tok == 'CONST_START':
        # Parse float constant: CONST_START sign mantissa_digits exp_sign exp_digits CONST_END
        const_tokens = []
        idx += 1
        while idx < len(tokens) and tokens[idx] != 'CONST_END':
            const_tokens.append(tokens[idx])
            idx += 1
        val = _decode_float_constant(const_tokens)
        return sympy.Float(val), idx + 1
    else:
        # Must be a variable name
        return Symbol(tok), idx + 1


def _decode_float_constant(tokens: list) -> float:
    """Decode a float constant from its token decomposition."""
    if not tokens:
        return 0.0

    sign = 1.0 if tokens[0] == 'C_+' else -1.0
    mantissa_str = ''
    exp_str = ''
    exp_sign = 1

    in_exp = False
    for tok in tokens[1:]:
        if tok.startswith('D_'):
            if not in_exp:
                mantissa_str += tok[2:]
        elif tok == 'DOT':
            mantissa_str += '.'
        elif tok in ('E_+', 'E_-'):
            in_exp = True
            exp_sign = 1 if tok == 'E_+' else -1
        elif tok.startswith('E_') and in_exp:
            exp_str += tok[2:]

    mantissa = float(mantissa_str) if mantissa_str else 0.0
    exponent = int(exp_str) * exp_sign if exp_str else 0
    return sign * mantissa * (10.0 ** exponent)


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def _tree_nodes(prefix_str: str) -> List[str]:
    """Get the list of tokens (nodes) in a prefix expression."""
    return prefix_str.strip().split()


def _tree_depth(prefix_str: str) -> int:
    """Compute the maximum depth of a prefix expression tree."""
    tokens = prefix_str.strip().split()
    if not tokens:
        return 0
    max_depth = 0
    stack = []
    for tok in tokens:
        if stack:
            current_depth = stack[-1][1]
        else:
            current_depth = 0
        max_depth = max(max_depth, current_depth)
        if tok in OPERATOR_ARITIES:
            arity = OPERATOR_ARITIES[tok]
            stack.append([arity, current_depth + 1])
        else:
            while stack:
                stack[-1][0] -= 1
                if stack[-1][0] == 0:
                    stack.pop()
                else:
                    break
    return max_depth


# ---------------------------------------------------------------------------
# Metric 1: Exact Match
# ---------------------------------------------------------------------------

def exact_match(pred_prefix: str, gt_prefix: str) -> float:
    """Check if predicted and ground-truth expressions are exactly equal
    after sympy simplification.

    Returns 1.0 if match, 0.0 otherwise.
    """
    try:
        pred_expr = prefix_to_sympy(pred_prefix)
        gt_expr = prefix_to_sympy(gt_prefix)
        if pred_expr is None or gt_expr is None:
            return 0.0
        diff = simplify(pred_expr - gt_expr)
        return 1.0 if diff == 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Metric 2: Symbolic Equivalence
# ---------------------------------------------------------------------------

def symbolic_equivalence(pred_prefix: str, gt_prefix: str,
                         n_test_points: int = 100, tol: float = 1e-6) -> float:
    """Check symbolic equivalence via sympy.equals, with numerical fallback.

    Returns 1.0 if equivalent, 0.0 otherwise.
    """
    try:
        pred_expr = prefix_to_sympy(pred_prefix)
        gt_expr = prefix_to_sympy(gt_prefix)
        if pred_expr is None or gt_expr is None:
            return 0.0

        # Try sympy symbolic equality
        diff = simplify(pred_expr - gt_expr)
        if diff == 0:
            return 1.0

        # Try sympy .equals()
        try:
            if pred_expr.equals(gt_expr):
                return 1.0
        except Exception:
            pass

        # Numerical fallback: evaluate at random points
        symbols = list(pred_expr.free_symbols | gt_expr.free_symbols)
        if not symbols:
            # Both are constants
            try:
                return 1.0 if abs(float(N(pred_expr)) - float(N(gt_expr))) < tol else 0.0
            except Exception:
                return 0.0

        rng = np.random.RandomState(42)
        matches = 0
        valid = 0
        for _ in range(n_test_points):
            subs = {s: rng.uniform(0.1, 5.0) for s in symbols}
            try:
                pred_val = complex(N(pred_expr.subs(subs)))
                gt_val = complex(N(gt_expr.subs(subs)))
                if math.isfinite(pred_val.real) and math.isfinite(gt_val.real):
                    valid += 1
                    if abs(pred_val - gt_val) < tol * max(1, abs(gt_val)):
                        matches += 1
            except Exception:
                continue

        if valid < 10:
            return 0.0
        return 1.0 if matches / valid > 0.95 else 0.0

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Metric 3: Numerical R²
# ---------------------------------------------------------------------------

def numerical_r2(pred_prefix: str, gt_prefix: str,
                 X: Optional[np.ndarray] = None,
                 Y_gt: Optional[np.ndarray] = None,
                 variable_names: Optional[List[str]] = None) -> float:
    """Compute R² between predicted and ground-truth expressions.

    If X and Y_gt are provided, evaluate pred on X and compare to Y_gt.
    Otherwise, generate random test points.

    Returns R² clipped to [0, 1].
    """
    try:
        pred_expr = prefix_to_sympy(pred_prefix)
        if pred_expr is None:
            return 0.0

        symbols = sorted(pred_expr.free_symbols, key=str)

        if X is not None and Y_gt is not None:
            # Use provided data
            if variable_names is None:
                variable_names = [str(s) for s in symbols]
            sym_map = {Symbol(name): symbols[i] if i < len(symbols) else Symbol(name)
                       for i, name in enumerate(variable_names)}

            Y_pred = []
            for i in range(len(X)):
                subs = {}
                for j, name in enumerate(variable_names):
                    sym = Symbol(name)
                    subs[sym] = float(X[i, j]) if j < X.shape[1] else 1.0
                try:
                    val = float(N(pred_expr.subs(subs)))
                    Y_pred.append(val if math.isfinite(val) else 0.0)
                except Exception:
                    Y_pred.append(0.0)

            Y_pred = np.array(Y_pred)
            ss_res = np.sum((Y_gt - Y_pred) ** 2)
            ss_tot = np.sum((Y_gt - np.mean(Y_gt)) ** 2)
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            r2 = 1.0 - ss_res / ss_tot
            return float(np.clip(r2, 0.0, 1.0))
        else:
            # Generate random test points and compare with gt
            gt_expr = prefix_to_sympy(gt_prefix)
            if gt_expr is None:
                return 0.0
            all_symbols = sorted(pred_expr.free_symbols | gt_expr.free_symbols, key=str)
            rng = np.random.RandomState(42)
            Y_pred_list, Y_gt_list = [], []
            for _ in range(200):
                subs = {s: rng.uniform(0.1, 5.0) for s in all_symbols}
                try:
                    pv = float(N(pred_expr.subs(subs)))
                    gv = float(N(gt_expr.subs(subs)))
                    if math.isfinite(pv) and math.isfinite(gv):
                        Y_pred_list.append(pv)
                        Y_gt_list.append(gv)
                except Exception:
                    continue

            if len(Y_gt_list) < 10:
                return 0.0

            Y_pred_arr = np.array(Y_pred_list)
            Y_gt_arr = np.array(Y_gt_list)
            ss_res = np.sum((Y_gt_arr - Y_pred_arr) ** 2)
            ss_tot = np.sum((Y_gt_arr - np.mean(Y_gt_arr)) ** 2)
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            r2 = 1.0 - ss_res / ss_tot
            return float(np.clip(r2, 0.0, 1.0))

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Metric 4: Tree Edit Distance (normalized)
# ---------------------------------------------------------------------------

def tree_edit_distance(pred_prefix: str, gt_prefix: str) -> float:
    """Compute normalized tree edit distance between expression trees.

    Uses a simple token-level edit distance as proxy for tree edit distance.
    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    pred_tokens = _tree_nodes(pred_prefix)
    gt_tokens = _tree_nodes(gt_prefix)

    if not pred_tokens and not gt_tokens:
        return 0.0
    if not pred_tokens or not gt_tokens:
        return 1.0

    # Levenshtein distance
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_tokens[i-1] == gt_tokens[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    max_len = max(m, n)
    return dp[m][n] / max_len if max_len > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric 5: Complexity Penalty
# ---------------------------------------------------------------------------

def complexity_penalty(pred_prefix: str, gt_prefix: str) -> float:
    """Compute complexity penalty as ratio of predicted to ground-truth tree depth.

    Returns value clipped to [0, 1] where 0 = same complexity, 1 = very different.
    """
    pred_depth = _tree_depth(pred_prefix)
    gt_depth = _tree_depth(gt_prefix)

    if gt_depth == 0:
        return 0.0 if pred_depth == 0 else 1.0

    ratio = abs(pred_depth - gt_depth) / gt_depth
    return float(min(ratio, 1.0))


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

def composite_score(pred_prefix: str, gt_prefix: str,
                    X: Optional[np.ndarray] = None,
                    Y_gt: Optional[np.ndarray] = None,
                    variable_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute all 5 metrics and the composite score.

    S = 0.3*EM + 0.3*SE + 0.25*R2 + 0.1*(1-TED) + 0.05*(1-CP)

    Returns dict with all metrics and composite score (scale 0-100).
    """
    em = exact_match(pred_prefix, gt_prefix)
    se = symbolic_equivalence(pred_prefix, gt_prefix)
    r2 = numerical_r2(pred_prefix, gt_prefix, X, Y_gt, variable_names)
    ted = tree_edit_distance(pred_prefix, gt_prefix)
    cp = complexity_penalty(pred_prefix, gt_prefix)

    score = (0.3 * em + 0.3 * se + 0.25 * r2 + 0.1 * (1.0 - ted) + 0.05 * (1.0 - cp)) * 100

    return {
        'exact_match': em,
        'symbolic_equivalence': se,
        'numerical_r2': r2,
        'tree_edit_distance': ted,
        'complexity_penalty': cp,
        'composite_score': score,
    }
