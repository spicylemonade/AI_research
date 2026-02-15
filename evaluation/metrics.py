"""Comprehensive evaluation metrics suite for symbolic regression.

Implements:
    1. Symbolic Equivalence Accuracy  -- SymPy-based algebraic equivalence
    2. Numeric R^2 Score              -- held-out point evaluation
    3. Complexity-Weighted Score      -- accuracy broken down by equation tier
    4. Token Edit Distance            -- normalised Levenshtein distance
    5. Novel Discovery Rate           -- fraction of held-out equations recovered
    6. Aggregate evaluate_predictions -- JSON-serialisable results dict

Uses sympy, numpy, and the project's data.tokenizer / data.equations modules.
"""

from __future__ import annotations

import json
import signal
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy
from sympy import Symbol

# ---- project imports -------------------------------------------------------
sys.path.insert(0, ".")  # ensure project root is importable
from data.tokenizer import ExprTokenizer, SPECIAL_TOKENS
from data.equations import Equation, get_all_equations, get_held_out_equations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_tokenizer = ExprTokenizer()


class _Timeout:
    """Context-manager that raises TimeoutError after *seconds* on POSIX.

    On platforms without SIGALRM (Windows) the timeout is silently ignored.
    """

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._has_alarm = hasattr(signal, "SIGALRM")

    def _handler(self, signum, frame):
        raise TimeoutError(f"Timed out after {self.seconds}s")

    def __enter__(self):
        if self._has_alarm:
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        # Suppress TimeoutError so caller can handle it
        return exc_type is TimeoutError


def _to_sympy(expr_or_tokens) -> Optional[sympy.Expr]:
    """Convert *expr_or_tokens* to a SymPy expression.

    Accepts:
      - A SymPy expression (returned as-is)
      - A list of **int** token indices  -> decoded via ExprTokenizer
      - A list of **str** token strings  -> decoded via ExprTokenizer
      - A **str** that can be sympified
    Returns None on failure.
    """
    if isinstance(expr_or_tokens, sympy.Basic):
        return expr_or_tokens
    if isinstance(expr_or_tokens, str):
        try:
            return sympy.sympify(expr_or_tokens)
        except Exception:
            return None
    if isinstance(expr_or_tokens, (list, tuple)) and len(expr_or_tokens) > 0:
        if isinstance(expr_or_tokens[0], int):
            try:
                return _tokenizer.decode(list(expr_or_tokens))
            except Exception:
                return None
        if isinstance(expr_or_tokens[0], str):
            try:
                return _tokenizer.decode_from_tokens(list(expr_or_tokens))
            except Exception:
                return None
    return None


def _to_tokens(expr_or_tokens) -> Optional[List[str]]:
    """Return prefix token-string list.  Accepts same types as _to_sympy."""
    if isinstance(expr_or_tokens, (list, tuple)) and len(expr_or_tokens) > 0:
        if isinstance(expr_or_tokens[0], str):
            return [t for t in expr_or_tokens if t not in SPECIAL_TOKENS]
        if isinstance(expr_or_tokens[0], int):
            tok_strs = [_tokenizer.get_token_str(i) for i in expr_or_tokens]
            return [t for t in tok_strs if t not in SPECIAL_TOKENS]
    if isinstance(expr_or_tokens, sympy.Basic):
        try:
            return _tokenizer.encode_to_tokens(expr_or_tokens)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# 1. Symbolic Equivalence Accuracy
# ---------------------------------------------------------------------------

def symbolic_equivalence(pred_expr, true_expr, timeout: int = 5) -> bool:
    """Check whether *pred_expr* and *true_expr* are algebraically equivalent.

    Uses ``sympy.simplify(pred - true) == 0`` with a configurable timeout
    (default 5 s).  Falls back to a numerical spot-check if simplification
    is inconclusive or times out.

    Parameters
    ----------
    pred_expr : sympy.Expr | list[int] | list[str] | str
        Predicted expression (any representation accepted by ``_to_sympy``).
    true_expr : sympy.Expr | list[int] | list[str] | str
        Ground-truth expression.
    timeout : int
        Maximum wall-clock seconds for SymPy simplification.

    Returns
    -------
    bool
        True if the two expressions are considered equivalent.
    """
    pred = _to_sympy(pred_expr)
    true = _to_sympy(true_expr)
    if pred is None or true is None:
        return False

    # ---- Fast structural check -------------------------------------------
    if pred.equals(true):
        return True

    # ---- Simplification with timeout -------------------------------------
    equivalent = False
    timed_out = False
    with _Timeout(timeout):
        try:
            diff = sympy.simplify(pred - true)
            if diff == 0 or diff == sympy.S.Zero:
                equivalent = True
        except TimeoutError:
            timed_out = True
        except Exception:
            pass

    if equivalent:
        return True

    # ---- Numerical fall-back (useful when simplify times out) ------------
    try:
        all_syms = pred.free_symbols | true.free_symbols
        if not all_syms:
            # Both are constants -- compare their float values
            return abs(complex(pred) - complex(true)) < 1e-9
        rng = np.random.RandomState(12345)
        for _ in range(5):
            subs = {s: float(rng.uniform(0.5, 5.0)) for s in all_syms}
            v_pred = complex(pred.subs(subs))
            v_true = complex(true.subs(subs))
            if not np.isfinite(v_pred.real) or not np.isfinite(v_true.real):
                continue
            if abs(v_pred - v_true) > 1e-6 * (abs(v_true) + 1):
                return False
        return True  # survived all numerical checks
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. Numeric R^2 Score
# ---------------------------------------------------------------------------

def numeric_r2(
    pred_expr,
    true_expr,
    variables: Optional[List[sympy.Symbol]] = None,
    n_points: int = 100,
    seed: int = 42,
) -> float:
    """Compute R^2 between *pred_expr* and *true_expr* on random test points.

    Parameters
    ----------
    pred_expr : sympy.Expr | list | str
        Predicted expression.
    true_expr : sympy.Expr | list | str
        Ground-truth expression.
    variables : list[sympy.Symbol] | None
        Variables to sample.  If None, derived from the union of free symbols
        in both expressions.
    n_points : int
        Number of random evaluation points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        R^2 score in (-inf, 1].  Returns -1.0 for invalid / non-evaluable
        expressions.
    """
    pred = _to_sympy(pred_expr)
    true = _to_sympy(true_expr)
    if pred is None or true is None:
        return -1.0

    if variables is None:
        variables = sorted(pred.free_symbols | true.free_symbols, key=str)
    if not variables:
        # Both are constants
        try:
            return 1.0 if abs(complex(pred) - complex(true)) < 1e-9 else -1.0
        except Exception:
            return -1.0

    rng = np.random.RandomState(seed)
    points = rng.uniform(0.5, 5.0, size=(n_points, len(variables)))

    y_true = np.empty(n_points, dtype=np.float64)
    y_pred = np.empty(n_points, dtype=np.float64)
    valid = 0

    for i in range(n_points):
        subs = {v: float(points[i, j]) for j, v in enumerate(variables)}
        try:
            vt = complex(true.subs(subs))
            vp = complex(pred.subs(subs))
            if not (np.isfinite(vt.real) and np.isfinite(vp.real)):
                continue
            if abs(vt.imag) > 1e-8 or abs(vp.imag) > 1e-8:
                continue
            y_true[valid] = vt.real
            y_pred[valid] = vp.real
            valid += 1
        except Exception:
            continue

    if valid < 2:
        return -1.0

    y_true = y_true[:valid]
    y_pred = y_pred[:valid]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        # All true values identical -- R^2 is 1 if residuals are also ~0
        return 1.0 if ss_res < 1e-12 else -1.0

    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


# ---------------------------------------------------------------------------
# 3. Token Edit Distance (normalised Levenshtein)
# ---------------------------------------------------------------------------

def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def token_edit_distance(pred_tokens, true_tokens) -> float:
    """Normalised Levenshtein distance between two token sequences.

    Accepts token-index lists (``list[int]``), token-string lists
    (``list[str]``), or SymPy expressions (which are tokenised first).
    Special tokens (<SOS>, <EOS>, <PAD>, ...) are stripped before comparison.

    Returns
    -------
    float
        A value in [0, 1] where 0 means identical sequences and 1 means
        maximally different.  Returns 1.0 when both sequences are empty.
    """
    a = _to_tokens(pred_tokens)
    b = _to_tokens(true_tokens)
    if a is None:
        a = []
    if b is None:
        b = []
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return _levenshtein(a, b) / max_len


# ---------------------------------------------------------------------------
# 4. Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: List,
    ground_truths: List,
    equations: List[Equation],
    n_points: int = 100,
) -> Dict[str, Any]:
    """Evaluate a batch of predictions against ground truths.

    Parameters
    ----------
    predictions : list
        Predicted expressions (SymPy, token indices, or token strings).
    ground_truths : list
        Ground-truth expressions (same formats accepted).
    equations : list[Equation]
        Corresponding ``Equation`` objects from the corpus (used for tier
        information and held-out flag).
    n_points : int
        Number of random points for numeric R^2.

    Returns
    -------
    dict
        {
            'symbolic_accuracy': float,
            'mean_r2': float,
            'tier_accuracy': {1: float, ..., 5: float},
            'tier_r2': {1: float, ..., 5: float},
            'mean_edit_distance': float,
            'novel_discovery_rate': float,
            'per_equation': [...]
        }
    """
    assert len(predictions) == len(ground_truths) == len(equations), (
        "predictions, ground_truths, and equations must have the same length"
    )

    per_eq: List[Dict[str, Any]] = []
    tier_correct: Dict[int, List[bool]] = {t: [] for t in range(1, 6)}
    tier_r2s: Dict[int, List[float]] = {t: [] for t in range(1, 6)}
    held_out_correct: List[bool] = []

    sym_acc_total = 0
    r2_total = 0.0
    edit_total = 0.0

    for pred, true, eq in zip(predictions, ground_truths, equations):
        equiv = symbolic_equivalence(pred, true)
        r2 = numeric_r2(pred, true, variables=eq.variables, n_points=n_points)
        ed = token_edit_distance(pred, true)

        sym_acc_total += int(equiv)
        r2_total += r2
        edit_total += ed

        tier_correct[eq.tier].append(equiv)
        tier_r2s[eq.tier].append(r2)

        if eq.held_out:
            held_out_correct.append(equiv)

        per_eq.append({
            'id': eq.id,
            'name': eq.name,
            'tier': eq.tier,
            'held_out': eq.held_out,
            'symbolic_equivalent': equiv,
            'r2': r2,
            'edit_distance': ed,
        })

    n = len(predictions)
    symbolic_accuracy = sym_acc_total / n if n > 0 else 0.0
    mean_r2 = r2_total / n if n > 0 else 0.0
    mean_edit = edit_total / n if n > 0 else 0.0

    tier_accuracy: Dict[int, float] = {}
    tier_r2: Dict[int, float] = {}
    for t in range(1, 6):
        tc = tier_correct[t]
        tier_accuracy[t] = sum(tc) / len(tc) if tc else 0.0
        tr = tier_r2s[t]
        tier_r2[t] = sum(tr) / len(tr) if tr else 0.0

    novel_discovery_rate = (
        sum(held_out_correct) / len(held_out_correct)
        if held_out_correct
        else 0.0
    )

    return {
        'symbolic_accuracy': symbolic_accuracy,
        'mean_r2': mean_r2,
        'tier_accuracy': tier_accuracy,
        'tier_r2': tier_r2,
        'mean_edit_distance': mean_edit,
        'novel_discovery_rate': novel_discovery_rate,
        'per_equation': per_eq,
    }


# ---------------------------------------------------------------------------
# 5. JSON serialisation helper
# ---------------------------------------------------------------------------

def results_to_json(results: Dict[str, Any], path: Optional[str] = None) -> str:
    """Serialise evaluation results to a JSON string (and optionally to file).

    Handles numpy types and SymPy objects that are not natively JSON
    serialisable.
    """

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, sympy.Basic):
                return str(o)
            return super().default(o)

    txt = json.dumps(results, indent=2, cls=_Encoder)
    if path is not None:
        with open(path, 'w') as f:
            f.write(txt)
    return txt


# ===================================================================
# Unit Tests
# ===================================================================

def _run_tests() -> None:
    """Run self-contained unit tests.  Exit with non-zero on failure."""
    import traceback

    passed = 0
    failed = 0
    total = 0

    def _assert(condition: bool, msg: str):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  PASS: {msg}")
        else:
            failed += 1
            print(f"  FAIL: {msg}")

    x0 = Symbol('x0', positive=True)
    x1 = Symbol('x1', positive=True)
    x2 = Symbol('x2', positive=True)

    # ------------------------------------------------------------------
    print("\n=== Symbolic Equivalence Tests ===")
    # ------------------------------------------------------------------

    # 1. x0*x1 == x1*x0  (commutativity of multiplication)
    _assert(symbolic_equivalence(x0 * x1, x1 * x0),
            "x0*x1 is equivalent to x1*x0 (commutative multiplication)")

    # 2. x0+x1 == x1+x0  (commutativity of addition)
    _assert(symbolic_equivalence(x0 + x1, x1 + x0),
            "x0+x1 is equivalent to x1+x0 (commutative addition)")

    # 3. Associativity: (x0+x1)+x2 == x0+(x1+x2)
    _assert(symbolic_equivalence((x0 + x1) + x2, x0 + (x1 + x2)),
            "(x0+x1)+x2 is equivalent to x0+(x1+x2) (associativity)")

    # 4. Distribution: x0*(x1+x2) == x0*x1 + x0*x2
    _assert(symbolic_equivalence(x0 * (x1 + x2), x0 * x1 + x0 * x2),
            "x0*(x1+x2) is equivalent to x0*x1+x0*x2 (distribution)")

    # 5. Non-equivalent: x0+x1 != x0*x1
    _assert(not symbolic_equivalence(x0 + x1, x0 * x1),
            "x0+x1 is NOT equivalent to x0*x1")

    # 6. Equivalent via simplification: (x0^2 - x1^2) / (x0 - x1) == x0 + x1
    _assert(symbolic_equivalence((x0**2 - x1**2) / (x0 - x1), x0 + x1),
            "(x0^2 - x1^2)/(x0 - x1) is equivalent to x0+x1")

    # 7. Constant expressions
    _assert(symbolic_equivalence(sympy.Integer(6), sympy.Integer(2) * 3),
            "6 is equivalent to 2*3")

    # 8. Token-index input (round-trip via tokenizer)
    indices_pred = _tokenizer.encode(x0 * x1)
    indices_true = _tokenizer.encode(x1 * x0)
    _assert(symbolic_equivalence(indices_pred, indices_true),
            "Token-index inputs x0*x1 vs x1*x0 are equivalent")

    # ------------------------------------------------------------------
    print("\n=== Numeric R^2 Tests ===")
    # ------------------------------------------------------------------

    # 9. Perfect prediction -> R^2 == 1.0
    r2_perfect = numeric_r2(x0 * x1, x0 * x1, variables=[x0, x1])
    _assert(abs(r2_perfect - 1.0) < 1e-9,
            f"Perfect prediction R^2 = {r2_perfect} (expected 1.0)")

    # 10. Equivalent forms -> R^2 == 1.0
    r2_equiv = numeric_r2(x0 + x1, x1 + x0, variables=[x0, x1])
    _assert(abs(r2_equiv - 1.0) < 1e-9,
            f"Equivalent form R^2 = {r2_equiv} (expected 1.0)")

    # 11. Completely wrong prediction -> R^2 should be low
    r2_wrong = numeric_r2(x0 + x1, x0 * x1 * x2, variables=[x0, x1, x2])
    _assert(r2_wrong < 0.5,
            f"Wrong prediction R^2 = {r2_wrong} (expected < 0.5)")

    # 12. Invalid expression -> R^2 == -1
    r2_invalid = numeric_r2(None, x0 + x1, variables=[x0, x1])
    _assert(r2_invalid == -1.0,
            f"Invalid expression R^2 = {r2_invalid} (expected -1.0)")

    # 13. Constant expression (no variables)
    r2_const = numeric_r2(sympy.Integer(7), sympy.Integer(7))
    _assert(r2_const == 1.0,
            f"Matching constants R^2 = {r2_const} (expected 1.0)")

    # 14. Expressions with different free symbols -- auto-detected
    r2_diff_vars = numeric_r2(x0, x0, variables=[x0])
    _assert(abs(r2_diff_vars - 1.0) < 1e-9,
            f"Same expr, explicit vars R^2 = {r2_diff_vars} (expected 1.0)")

    # ------------------------------------------------------------------
    print("\n=== Token Edit Distance Tests ===")
    # ------------------------------------------------------------------

    # 15. Identical sequences -> 0.0
    tokens_a = ['*', 'x0', 'x1']
    _assert(token_edit_distance(tokens_a, tokens_a) == 0.0,
            "Identical token sequences have edit distance 0.0")

    # 16. Completely different sequences
    tokens_b = ['sin', 'x2']
    ed_diff = token_edit_distance(tokens_a, tokens_b)
    _assert(0.5 < ed_diff <= 1.0,
            f"Completely different sequences have edit distance {ed_diff} (expected close to 1.0)")

    # 17. One insertion difference
    tokens_c = ['*', 'x0', 'x1']
    tokens_d = ['+', '*', 'x0', 'x1', 'x2']
    ed_ins = token_edit_distance(tokens_c, tokens_d)
    _assert(0.0 < ed_ins < 1.0,
            f"One-insertion edit distance = {ed_ins}")

    # 18. SymPy expression input
    ed_sympy = token_edit_distance(x0 * x1, x0 * x1)
    _assert(ed_sympy == 0.0,
            "SymPy expressions x0*x1 vs x0*x1 have edit distance 0.0")

    # 19. Empty vs non-empty
    ed_empty = token_edit_distance([], tokens_a)
    _assert(ed_empty == 1.0,
            f"Empty vs non-empty edit distance = {ed_empty} (expected 1.0)")

    # 20. Both empty
    ed_both_empty = token_edit_distance([], [])
    _assert(ed_both_empty == 0.0,
            f"Both-empty edit distance = {ed_both_empty} (expected 0.0)")

    # ------------------------------------------------------------------
    print("\n=== evaluate_predictions Integration Test ===")
    # ------------------------------------------------------------------

    # Build small synthetic dataset
    eqs = get_all_equations()[:6]  # first 6 equations
    preds = [eq.symbolic_expr for eq in eqs]            # perfect preds
    truths = [eq.symbolic_expr for eq in eqs]

    results = evaluate_predictions(preds, truths, eqs)

    _assert(abs(results['symbolic_accuracy'] - 1.0) < 1e-9,
            f"All-correct symbolic accuracy = {results['symbolic_accuracy']}")
    _assert(abs(results['mean_r2'] - 1.0) < 1e-9,
            f"All-correct mean R^2 = {results['mean_r2']}")
    _assert(abs(results['mean_edit_distance']) < 1e-9,
            f"All-correct mean edit distance = {results['mean_edit_distance']}")

    # Check tier_accuracy keys
    for t in range(1, 6):
        _assert(t in results['tier_accuracy'],
                f"tier_accuracy contains tier {t}")

    # Check per_equation length
    _assert(len(results['per_equation']) == len(eqs),
            f"per_equation has {len(results['per_equation'])} entries (expected {len(eqs)})")

    # ------------------------------------------------------------------
    print("\n=== Novel Discovery Rate Test ===")
    # ------------------------------------------------------------------

    held_out = get_held_out_equations()[:3]
    # Simulate: 2 out of 3 correctly recovered
    ho_preds = [
        held_out[0].symbolic_expr,      # correct
        held_out[1].symbolic_expr,      # correct
        x0 + x1,                        # wrong
    ]
    ho_truths = [eq.symbolic_expr for eq in held_out]
    ho_results = evaluate_predictions(ho_preds, ho_truths, held_out)

    # 2 out of 3 held-out => 0.6667
    expected_ndr = 2.0 / 3.0
    _assert(abs(ho_results['novel_discovery_rate'] - expected_ndr) < 1e-4,
            f"Novel discovery rate = {ho_results['novel_discovery_rate']:.4f} "
            f"(expected {expected_ndr:.4f})")

    # ------------------------------------------------------------------
    print("\n=== JSON Serialisation Test ===")
    # ------------------------------------------------------------------

    try:
        json_str = results_to_json(results)
        parsed = json.loads(json_str)
        _assert('symbolic_accuracy' in parsed,
                "JSON round-trip preserves symbolic_accuracy key")
    except Exception as e:
        _assert(False, f"JSON serialisation raised {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Tests: {passed} passed, {failed} failed, {total} total")
    print(f"{'='*60}")
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    _run_tests()
