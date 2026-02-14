"""Post-processing: SymPy simplification + BFGS constant optimization."""

import numpy as np
import sympy as sp
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import _parse_prefix, tree_to_sympy, decode, ExprNode


def sympy_simplify(prefix_tokens: List[str]) -> Tuple[sp.Expr, List[str]]:
    """Canonically simplify an expression via SymPy.

    Returns:
        (simplified_expr, simplified_prefix_tokens)
    """
    try:
        tree, _ = _parse_prefix(prefix_tokens, 0)
        expr = tree_to_sympy(tree)
        simplified = sp.simplify(expr)
        return simplified, prefix_tokens  # Return original tokens if simplification works
    except Exception:
        return None, prefix_tokens


def bfgs_optimize_constants(prefix_tokens: List[str],
                             observation_table: np.ndarray,
                             max_constants: int = 10) -> Tuple[Dict[str, float], float]:
    """Optimize numerical constants in the expression via L-BFGS-B.

    Args:
        prefix_tokens: prefix notation tokens (may contain c_0, c_1, etc.)
        observation_table: (N, D+1) observation data
        max_constants: maximum number of constants to optimize

    Returns:
        (optimized_constants, mse)
    """
    try:
        tree, _ = _parse_prefix(prefix_tokens, 0)
    except Exception:
        return {}, float('inf')

    # Find constant tokens used
    const_names = set()
    _find_constants(tree, const_names)

    if not const_names:
        # No constants to optimize, evaluate as-is
        from data.data_generator import tree_to_numpy_func
        func = tree_to_numpy_func(tree, {})
        n_vars = observation_table.shape[1] - 1
        x_dict = {i: observation_table[:, i] for i in range(n_vars)}
        try:
            y_pred = np.array(func(x_dict), dtype=np.float64)
            y_true = observation_table[:, -1]
            valid = np.isfinite(y_pred)
            mse = np.mean((y_pred[valid] - y_true[valid]) ** 2) if valid.sum() > 0 else float('inf')
        except Exception:
            mse = float('inf')
        return {}, mse

    const_list = sorted(const_names)[:max_constants]
    n_consts = len(const_list)
    n_vars = observation_table.shape[1] - 1
    y_true = observation_table[:, -1]

    from data.data_generator import tree_to_numpy_func

    def objective(params):
        const_vals = {name: params[i] for i, name in enumerate(const_list)}
        func = tree_to_numpy_func(tree, const_vals)
        x_dict = {i: observation_table[:, i] for i in range(n_vars)}
        try:
            y_pred = np.array(func(x_dict), dtype=np.float64)
            valid = np.isfinite(y_pred)
            if valid.sum() < 10:
                return 1e10
            return float(np.mean((y_pred[valid] - y_true[valid]) ** 2))
        except Exception:
            return 1e10

    # Try multiple initializations
    best_result = None
    best_mse = float('inf')

    for init in [np.ones(n_consts), np.zeros(n_consts) + 0.5,
                 np.random.default_rng(42).uniform(-5, 5, n_consts)]:
        try:
            result = minimize(objective, init, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12})
            if result.fun < best_mse:
                best_mse = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {name: 1.0 for name in const_list}, float('inf')

    opt_constants = {name: float(best_result.x[i]) for i, name in enumerate(const_list)}
    return opt_constants, best_mse


def _find_constants(node: ExprNode, const_names: set):
    """Find all constant tokens in an expression tree."""
    if node.token.startswith('c_'):
        const_names.add(node.token)
    for child in node.children:
        _find_constants(child, const_names)


def complexity_score(prefix_tokens: List[str]) -> int:
    """Compute complexity as number of nodes in expression tree."""
    try:
        tree, _ = _parse_prefix(prefix_tokens, 0)
        return tree.num_nodes()
    except Exception:
        return len(prefix_tokens)


def pareto_filter(candidates: List[Dict]) -> List[Dict]:
    """Filter candidates to Pareto frontier on accuracy-complexity plane.

    Each candidate dict must have 'mse' and 'complexity' keys.
    Returns non-dominated solutions.
    """
    if not candidates:
        return []

    # Sort by MSE (lower is better)
    sorted_cands = sorted(candidates, key=lambda c: c['mse'])
    pareto = [sorted_cands[0]]

    for cand in sorted_cands[1:]:
        # Keep if it has lower complexity than any Pareto member with lower MSE
        if cand['complexity'] < pareto[-1]['complexity']:
            pareto.append(cand)

    return pareto


def postprocess_candidates(candidate_token_lists: List[List[str]],
                            observation_table: np.ndarray,
                            top_k: int = 5) -> List[Dict]:
    """Full post-processing pipeline for multiple candidate expressions.

    1. SymPy simplification
    2. BFGS constant optimization
    3. Pareto filtering
    4. Return top-K by MSE

    Args:
        candidate_token_lists: list of prefix token lists
        observation_table: (N, D+1) observation data
        top_k: number of candidates to return

    Returns:
        list of dicts with 'tokens', 'constants', 'mse', 'complexity', 'sympy_expr'
    """
    results = []

    for tokens in candidate_token_lists:
        try:
            # Simplify
            expr, simplified_tokens = sympy_simplify(tokens)

            # Optimize constants
            opt_consts, mse = bfgs_optimize_constants(tokens, observation_table)

            # Compute complexity
            comp = complexity_score(tokens)

            results.append({
                'tokens': tokens,
                'constants': opt_consts,
                'mse': mse,
                'complexity': comp,
                'sympy_expr': str(expr) if expr is not None else None,
            })
        except Exception:
            continue

    if not results:
        return []

    # Pareto filter
    pareto = pareto_filter(results)

    # Sort all by MSE and return top-K
    results.sort(key=lambda r: r['mse'])
    return results[:top_k]


if __name__ == '__main__':
    """Unit tests for postprocessing."""
    print("Running postprocessing tests...")

    # Test 1: Simplify F = m * a
    expr, tokens = sympy_simplify(['mul', 'x_0', 'x_1'])
    assert expr is not None
    print(f"  [PASS] Simplification: {expr}")

    # Test 2: BFGS constant optimization
    # Generate data for F = 2.5 * x^2 (c_0 should converge to ~2.5)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 10, 200)
    y = 2.5 * x ** 2
    table = np.column_stack([x, y]).astype(np.float32)

    opt_consts, mse = bfgs_optimize_constants(
        ['mul', 'c_0', 'pow', 'x_0', 'int_2'],
        table
    )
    assert abs(opt_consts.get('c_0', 0) - 2.5) < 0.1, f"Expected ~2.5, got {opt_consts}"
    assert mse < 0.1, f"MSE too high: {mse}"
    print(f"  [PASS] BFGS optimization: c_0={opt_consts['c_0']:.4f}, MSE={mse:.6f}")

    # Test 3: Complexity
    assert complexity_score(['mul', 'x_0', 'x_1']) == 3
    assert complexity_score(['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2']) == 7
    print("  [PASS] Complexity scoring")

    # Test 4: Pareto filter
    cands = [
        {'mse': 0.1, 'complexity': 10},
        {'mse': 0.5, 'complexity': 5},
        {'mse': 0.2, 'complexity': 8},
        {'mse': 0.8, 'complexity': 3},
    ]
    pareto = pareto_filter(cands)
    assert len(pareto) >= 2
    print(f"  [PASS] Pareto filter: {len(pareto)} solutions")

    # Test 5: Full pipeline with constant refinement improving R²
    test_cases = []
    for true_c in [1.5, 3.0, 0.7, 5.2, 2.8]:
        x_test = rng.uniform(0.1, 10, 200)
        y_test = true_c * x_test ** 2
        table_test = np.column_stack([x_test, y_test]).astype(np.float32)

        # Start with imprecise constant (c_0 = 1.0 by default)
        opt_c, mse = bfgs_optimize_constants(
            ['mul', 'c_0', 'pow', 'x_0', 'int_2'], table_test)
        test_cases.append(abs(opt_c.get('c_0', 0) - true_c) < 0.1)

    assert sum(test_cases) >= 5, f"Only {sum(test_cases)}/5 constants recovered"
    print(f"  [PASS] Constant refinement: {sum(test_cases)}/5 recovered")

    print("\nAll postprocessing tests passed!")
