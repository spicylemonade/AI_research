"""
Evaluation metrics for symbolic regression.
Implements symbolic equivalence checking, R² scoring, tree-edit distance,
and complexity metrics.
"""

import numpy as np
import signal
from typing import List, Dict, Optional, Tuple, Any
from functools import wraps

try:
    import sympy
    from sympy import symbols, simplify, sympify, sin, cos, tan, exp, log, sqrt, pi, E
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class TimeoutError(Exception):
    pass


def timeout(seconds=5):
    """Decorator to add timeout to a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds}s")

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                result = None
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


# ============================================================
# Prefix notation parsing utilities
# ============================================================

BINARY_OPS = {'add', 'sub', 'mul', 'div', 'pow'}
UNARY_OPS = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'neg', 'abs', 'asin', 'acos', 'atan'}

def prefix_to_sympy(tokens: List[str]) -> Optional[Any]:
    """Convert prefix notation tokens to a SymPy expression."""
    if not SYMPY_AVAILABLE:
        return None

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9')
    var_map = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
               'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9}

    op_map = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b,
        'pow': lambda a, b: a ** b,
        'sin': lambda a: sympy.sin(a),
        'cos': lambda a: sympy.cos(a),
        'tan': lambda a: sympy.tan(a),
        'exp': lambda a: sympy.exp(a),
        'log': lambda a: sympy.log(a),
        'sqrt': lambda a: sympy.sqrt(a),
        'neg': lambda a: -a,
        'abs': lambda a: sympy.Abs(a),
        'asin': lambda a: sympy.asin(a),
        'acos': lambda a: sympy.acos(a),
        'atan': lambda a: sympy.atan(a),
    }

    pos = [0]

    def parse():
        if pos[0] >= len(tokens):
            return None
        token = tokens[pos[0]]
        pos[0] += 1

        if token in BINARY_OPS:
            left = parse()
            right = parse()
            if left is None or right is None:
                return None
            return op_map[token](left, right)
        elif token in UNARY_OPS:
            child = parse()
            if child is None:
                return None
            return op_map[token](child)
        elif token in var_map:
            return var_map[token]
        elif token == 'pi':
            return sympy.pi
        elif token == 'e':
            return sympy.E
        elif token == 'C':
            return sympy.Symbol('C')
        else:
            try:
                return sympy.Rational(token) if '.' not in token else sympy.Float(token)
            except (ValueError, TypeError):
                return None

    try:
        result = parse()
        return result
    except Exception:
        return None


def prefix_to_callable(tokens: List[str], num_vars: int = 9):
    """Convert prefix tokens to a callable numpy function."""
    import numpy as np_eval

    op_map = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: np_eval.where(np_eval.abs(b) < 1e-10, 1e10, a / (b + 1e-30)),
        'pow': lambda a, b: np_eval.power(np_eval.abs(a) + 1e-10, b),
        'sin': np_eval.sin,
        'cos': np_eval.cos,
        'tan': np_eval.tan,
        'exp': lambda a: np_eval.exp(np_eval.clip(a, -10, 10)),
        'log': lambda a: np_eval.log(np_eval.abs(a) + 1e-10),
        'sqrt': lambda a: np_eval.sqrt(np_eval.abs(a)),
        'neg': lambda a: -a,
        'abs': np_eval.abs,
        'asin': lambda a: np_eval.arcsin(np_eval.clip(a, -1, 1)),
        'acos': lambda a: np_eval.arccos(np_eval.clip(a, -1, 1)),
        'atan': np_eval.arctan,
    }

    pos = [0]

    def _make_binary(op, left_fn, right_fn):
        def fn(**kw):
            return op(left_fn(**kw), right_fn(**kw))
        return fn

    def _make_unary(op, child_fn):
        def fn(**kw):
            return op(child_fn(**kw))
        return fn

    def _make_var(var_name):
        def fn(**kw):
            return kw.get(var_name, np_eval.zeros(1))
        return fn

    def _make_const(val):
        def fn(**kw):
            return np_eval.full_like(list(kw.values())[0], val)
        return fn

    def parse():
        if pos[0] >= len(tokens):
            return lambda **kw: np_eval.zeros(1)
        token = tokens[pos[0]]
        pos[0] += 1

        if token in BINARY_OPS:
            left_fn = parse()
            right_fn = parse()
            op = op_map[token]
            return _make_binary(op, left_fn, right_fn)
        elif token in UNARY_OPS:
            child_fn = parse()
            op = op_map[token]
            return _make_unary(op, child_fn)
        elif token.startswith('x') and token[1:].isdigit():
            return _make_var(token)
        elif token == 'pi':
            return _make_const(np_eval.pi)
        elif token == 'e':
            return _make_const(np_eval.e)
        elif token == 'C':
            return _make_const(1.0)
        else:
            try:
                val = float(token)
                return _make_const(val)
            except ValueError:
                return _make_const(0.0)

    try:
        fn = parse()
        return fn
    except Exception:
        return None


# ============================================================
# Core metrics
# ============================================================

@timeout(5)
def symbolic_equivalence(pred_tokens: List[str], true_tokens: List[str]) -> Optional[bool]:
    """Check symbolic equivalence via SymPy simplification.

    Returns True if expressions are algebraically equivalent, False otherwise.
    Returns None on timeout or parsing error.
    """
    if not SYMPY_AVAILABLE:
        return None

    pred_expr = prefix_to_sympy(pred_tokens)
    true_expr = prefix_to_sympy(true_tokens)

    if pred_expr is None or true_expr is None:
        return False

    try:
        diff = simplify(pred_expr - true_expr)
        return diff == 0
    except Exception:
        return None


def r_squared(pred_tokens: List[str], X_test: np.ndarray, y_test: np.ndarray,
              num_vars: int = None) -> float:
    """Compute R² score on test points.

    Args:
        pred_tokens: Predicted equation in prefix notation
        X_test: Test input points, shape [N_test, num_vars]
        y_test: True output values, shape [N_test]
        num_vars: Number of variables (inferred from X_test if None)

    Returns:
        R² score (can be negative if prediction is worse than mean)
    """
    if num_vars is None:
        num_vars = X_test.shape[1] if X_test.ndim > 1 else 1

    fn = prefix_to_callable(pred_tokens, num_vars)
    if fn is None:
        return -1.0

    try:
        kw = {}
        for i in range(num_vars):
            if X_test.ndim > 1:
                kw[f'x{i+1}'] = X_test[:, i]
            else:
                kw[f'x{i+1}'] = X_test

        y_pred = fn(**kw)

        # Handle NaN/Inf
        valid = np.isfinite(y_pred) & np.isfinite(y_test)
        if valid.sum() < 10:
            return -1.0

        y_pred = y_pred[valid]
        y_true = y_test[valid]

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else -1.0

        return 1.0 - ss_res / ss_tot
    except Exception:
        return -1.0


def tree_edit_distance(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute normalized tree-edit distance between two prefix expressions.

    Uses a simplified token-level edit distance as approximation.
    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    n, m = len(tokens1), len(tokens2)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    # Dynamic programming edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if tokens1[i-1] == tokens2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost, # substitution
            )

    max_len = max(n, m)
    return dp[n][m] / max_len


def equation_complexity(tokens: List[str]) -> int:
    """Compute equation complexity as total node count in expression tree."""
    return len(tokens)


# ============================================================
# Aggregation functions
# ============================================================

def evaluate_equation(
    pred_tokens: List[str],
    true_tokens: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_vars: int = None,
) -> Dict[str, Any]:
    """Evaluate a single predicted equation against ground truth.

    Returns dict with all metrics.
    """
    result = {
        'exact_match': False,
        'r_squared': -1.0,
        'tree_edit_distance': 1.0,
        'pred_complexity': equation_complexity(pred_tokens),
        'true_complexity': equation_complexity(true_tokens),
    }

    # Symbolic equivalence
    equiv = symbolic_equivalence(pred_tokens, true_tokens)
    if equiv is True:
        result['exact_match'] = True

    # R² score
    result['r_squared'] = r_squared(pred_tokens, X_test, y_test, num_vars)

    # Tree edit distance
    result['tree_edit_distance'] = tree_edit_distance(pred_tokens, true_tokens)

    return result


def aggregate_results(
    results: List[Dict[str, Any]],
    tier_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregate evaluation results.

    Args:
        results: List of per-equation result dicts
        tier_labels: Optional tier label for each equation

    Returns:
        Aggregated metrics overall and per-tier
    """
    n = len(results)
    if n == 0:
        return {}

    # Overall metrics
    exact_matches = sum(1 for r in results if r['exact_match'])
    r2_scores = [r['r_squared'] for r in results]
    r2_above_09 = sum(1 for r2 in r2_scores if r2 > 0.9)
    r2_above_095 = sum(1 for r2 in r2_scores if r2 > 0.95)
    r2_above_099 = sum(1 for r2 in r2_scores if r2 > 0.99)
    ted_scores = [r['tree_edit_distance'] for r in results]

    agg = {
        'overall': {
            'n': n,
            'exact_match_rate': exact_matches / n,
            'exact_match_count': exact_matches,
            'mean_r_squared': float(np.mean(r2_scores)),
            'median_r_squared': float(np.median(r2_scores)),
            'r2_above_0.9': r2_above_09 / n,
            'r2_above_0.95': r2_above_095 / n,
            'r2_above_0.99': r2_above_099 / n,
            'mean_tree_edit_distance': float(np.mean(ted_scores)),
        }
    }

    # Per-tier metrics
    if tier_labels is not None:
        tiers = sorted(set(tier_labels))
        agg['per_tier'] = {}
        for tier in tiers:
            tier_results = [r for r, t in zip(results, tier_labels) if t == tier]
            tier_n = len(tier_results)
            if tier_n == 0:
                continue
            tier_exact = sum(1 for r in tier_results if r['exact_match'])
            tier_r2 = [r['r_squared'] for r in tier_results]
            tier_ted = [r['tree_edit_distance'] for r in tier_results]
            agg['per_tier'][tier] = {
                'n': tier_n,
                'exact_match_rate': tier_exact / tier_n,
                'exact_match_count': tier_exact,
                'mean_r_squared': float(np.mean(tier_r2)),
                'mean_tree_edit_distance': float(np.mean(tier_ted)),
            }

    return agg


# ============================================================
# Unit tests
# ============================================================

def run_tests():
    """Run unit tests for metrics."""
    print("Running metrics tests...")

    # Test 1: Exact match - sin(x1) == sin(x1)
    assert symbolic_equivalence(['sin', 'x1'], ['sin', 'x1']) == True, "Test 1 failed: sin(x1) == sin(x1)"
    print("  [PASS] sin(x1) == sin(x1)")

    # Test 2: Algebraic equivalence - x1+x1 == 2*x1
    result = symbolic_equivalence(['add', 'x1', 'x1'], ['mul', '2', 'x1'])
    assert result == True, f"Test 2 failed: x1+x1 == 2*x1, got {result}"
    print("  [PASS] x1+x1 == 2*x1")

    # Test 3: Non-equivalence
    result = symbolic_equivalence(['sin', 'x1'], ['cos', 'x1'])
    assert result == False, "Test 3 failed: sin(x1) != cos(x1)"
    print("  [PASS] sin(x1) != cos(x1)")

    # Test 4: R² score for perfect prediction
    X = np.linspace(-5, 5, 1000).reshape(-1, 1)
    y = np.sin(X[:, 0])
    r2 = r_squared(['sin', 'x1'], X, y, num_vars=1)
    assert r2 > 0.999, f"Test 4 failed: R² for sin(x1) = {r2}"
    print(f"  [PASS] R² for perfect sin(x1) = {r2:.6f}")

    # Test 5: Tree edit distance - identical
    ted = tree_edit_distance(['sin', 'x1'], ['sin', 'x1'])
    assert ted == 0.0, f"Test 5 failed: TED for identical = {ted}"
    print("  [PASS] TED for identical expressions = 0.0")

    # Test 6: Tree edit distance - different
    ted = tree_edit_distance(['sin', 'x1'], ['cos', 'x2'])
    assert 0 < ted <= 1.0, f"Test 6 failed: TED = {ted}"
    print(f"  [PASS] TED for different expressions = {ted:.4f}")

    # Test 7: Complexity
    assert equation_complexity(['add', 'mul', 'x1', 'x2', 'sin', 'x3']) == 6
    print("  [PASS] Complexity count correct")

    # Test 8: Aggregation
    results = [
        {'exact_match': True, 'r_squared': 0.99, 'tree_edit_distance': 0.0,
         'pred_complexity': 3, 'true_complexity': 3},
        {'exact_match': False, 'r_squared': 0.85, 'tree_edit_distance': 0.3,
         'pred_complexity': 5, 'true_complexity': 4},
    ]
    agg = aggregate_results(results, tier_labels=['trivial', 'simple'])
    assert agg['overall']['exact_match_rate'] == 0.5
    print("  [PASS] Aggregation correct")

    # Test 9: R² for near-miss (partial credit)
    X = np.linspace(-5, 5, 1000).reshape(-1, 1)
    y = X[:, 0] ** 2
    r2 = r_squared(['mul', 'x1', 'x1'], X, y, num_vars=1)
    assert r2 > 0.999, f"Test 9 failed: R² for x1*x1 vs x1^2 = {r2}"
    print(f"  [PASS] R² for x1*x1 vs x1^2 = {r2:.6f}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    run_tests()
