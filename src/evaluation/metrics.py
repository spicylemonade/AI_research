"""
Comprehensive evaluation metrics for physics equation derivation.
Implements exact symbolic match, R² score, tree edit distance,
complexity-adjusted accuracy, and statistical testing.
"""
import json
import numpy as np
import sympy as sp
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.data.equation_templates import (
    INV_EQUATION_VOCAB, EQUATION_VOCAB, VOCAB_SIZE,
    bin_to_constant, CONST_BIN_START, NUM_CONST_BINS
)


def tokens_to_prefix_str(token_ids: List[int]) -> List[str]:
    """Convert token IDs back to prefix notation strings."""
    result = []
    for tid in token_ids:
        if tid == EQUATION_VOCAB["[EQ_START]"]:
            continue
        if tid == EQUATION_VOCAB["[EQ_END]"]:
            break
        if tid == EQUATION_VOCAB["[PAD]"]:
            continue
        if tid == EQUATION_VOCAB["[MASK]"]:
            continue
        if tid in INV_EQUATION_VOCAB:
            result.append(INV_EQUATION_VOCAB[tid])
        else:
            result.append(f"UNK{tid}")
    return result


def prefix_to_sympy(tokens: List[str], variables: Dict[str, sp.Symbol] = None) -> Optional[sp.Expr]:
    """Convert prefix notation token list to SymPy expression."""
    if variables is None:
        variables = {f"x{i}": sp.Symbol(f"x{i}") for i in range(1, 7)}

    idx = [0]  # Mutable index for recursive parsing

    def parse():
        if idx[0] >= len(tokens):
            return None
        tok = tokens[idx[0]]
        idx[0] += 1

        # Operators
        if tok == "add":
            a = parse()
            b = parse()
            return a + b if a is not None and b is not None else None
        elif tok == "sub":
            a = parse()
            b = parse()
            return a - b if a is not None and b is not None else None
        elif tok == "mul":
            a = parse()
            b = parse()
            return a * b if a is not None and b is not None else None
        elif tok == "div":
            a = parse()
            b = parse()
            return a / b if a is not None and b is not None else None
        elif tok == "pow":
            a = parse()
            b = parse()
            return a ** b if a is not None and b is not None else None
        elif tok == "sqrt":
            a = parse()
            return sp.sqrt(a) if a is not None else None
        elif tok == "sin":
            a = parse()
            return sp.sin(a) if a is not None else None
        elif tok == "cos":
            a = parse()
            return sp.cos(a) if a is not None else None
        elif tok == "exp":
            a = parse()
            return sp.exp(a) if a is not None else None
        elif tok == "log":
            a = parse()
            return sp.log(a) if a is not None else None
        elif tok == "neg":
            a = parse()
            return -a if a is not None else None
        elif tok == "abs":
            a = parse()
            return sp.Abs(a) if a is not None else None
        elif tok == "sgn":
            a = parse()
            return sp.sign(a) if a is not None else None

        # Variables
        elif tok in variables:
            return variables[tok]

        # Constants
        elif tok == "C0.5":
            return sp.Rational(1, 2)
        elif tok == "C1":
            return sp.Integer(1)
        elif tok == "C2":
            return sp.Integer(2)
        elif tok == "C3":
            return sp.Integer(3)
        elif tok == "Cpi":
            return sp.pi

        # Discretized constant bins
        elif tok.startswith("CBIN"):
            try:
                bin_idx = int(tok[4:])
                val = bin_to_constant(bin_idx)
                return sp.Float(val, 4)
            except (ValueError, IndexError):
                return sp.Float(1.0)

        # Direct constant values (e.g., "C9.8100")
        elif tok.startswith("C"):
            try:
                val = float(tok[1:])
                return sp.Float(val, 4)
            except ValueError:
                return None

        else:
            return None

    try:
        result = parse()
        return result
    except (RecursionError, TypeError):
        return None


def expression_tree_size(expr: sp.Expr) -> int:
    """Count nodes in a SymPy expression tree."""
    if expr is None:
        return 0
    return 1 + sum(expression_tree_size(arg) for arg in expr.args)


def exact_symbolic_match(pred_tokens: List[int], gt_tokens: List[int]) -> bool:
    """Check if predicted equation is symbolically equivalent to ground truth."""
    pred_prefix = tokens_to_prefix_str(pred_tokens)
    gt_prefix = tokens_to_prefix_str(gt_tokens)

    pred_expr = prefix_to_sympy(pred_prefix)
    gt_expr = prefix_to_sympy(gt_prefix)

    if pred_expr is None or gt_expr is None:
        return False

    try:
        # Try simplification-based comparison
        diff = sp.simplify(sp.expand(pred_expr - gt_expr))
        if diff == 0:
            return True

        # Try numerical evaluation at random points
        variables = list(pred_expr.free_symbols | gt_expr.free_symbols)
        if not variables:
            return bool(sp.simplify(pred_expr - gt_expr) == 0)

        rng = np.random.RandomState(42)
        for _ in range(10):
            vals = {v: sp.Float(rng.uniform(0.1, 5.0)) for v in variables}
            try:
                pred_val = float(pred_expr.subs(vals))
                gt_val = float(gt_expr.subs(vals))
                if not np.isfinite(pred_val) or not np.isfinite(gt_val):
                    continue
                if abs(pred_val - gt_val) > 1e-4 * max(abs(gt_val), 1):
                    return False
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        return True
    except (sp.SympifyError, TypeError, RecursionError, TimeoutError):
        return False


def r2_score(pred_tokens: List[int], observations: np.ndarray,
             n_input_vars: int, n_obs: int) -> float:
    """Compute R² score of predicted equation against observations."""
    pred_prefix = tokens_to_prefix_str(pred_tokens)
    pred_expr = prefix_to_sympy(pred_prefix)

    if pred_expr is None:
        return -1.0

    variables = {f"x{i}": sp.Symbol(f"x{i}") for i in range(1, 7)}

    try:
        # Compile to numpy function
        var_symbols = [variables[f"x{i}"] for i in range(1, n_input_vars + 1)]
        func = sp.lambdify(var_symbols, pred_expr, modules=['numpy'])

        # Get actual data
        x = observations[:n_obs, :n_input_vars]
        y_true = observations[:n_obs, 6]  # y is last column (index 6)

        # Evaluate prediction
        args = [x[:, i] for i in range(n_input_vars)]
        y_pred = np.array(func(*args), dtype=np.float64)

        if not np.all(np.isfinite(y_pred)):
            return -1.0

        # Compute R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else -1.0

        r2 = 1 - ss_res / ss_tot
        return float(np.clip(r2, -1.0, 1.0))
    except (TypeError, ValueError, ZeroDivisionError, RuntimeError):
        return -1.0


def tree_edit_distance(pred_tokens: List[int], gt_tokens: List[int]) -> float:
    """Compute normalized tree edit distance between predicted and ground truth."""
    pred_prefix = tokens_to_prefix_str(pred_tokens)
    gt_prefix = tokens_to_prefix_str(gt_tokens)

    # Simple Levenshtein distance on token sequences as proxy for tree edit distance
    m, n = len(pred_prefix), len(gt_prefix)
    if m == 0:
        return 1.0
    if n == 0:
        return 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_prefix[i - 1] == gt_prefix[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Normalize by max length
    return dp[m][n] / max(m, n)


def complexity_adjusted_accuracy(
    pred_tokens: List[int], gt_tokens: List[int], alpha: float = 0.3
) -> float:
    """ESM × (1 - α × max(0, complexity_ratio - 1))."""
    esm = 1.0 if exact_symbolic_match(pred_tokens, gt_tokens) else 0.0
    if esm == 0:
        return 0.0

    pred_prefix = tokens_to_prefix_str(pred_tokens)
    gt_prefix = tokens_to_prefix_str(gt_tokens)

    pred_expr = prefix_to_sympy(pred_prefix)
    gt_expr = prefix_to_sympy(gt_prefix)

    if pred_expr is None or gt_expr is None:
        return esm

    pred_size = expression_tree_size(pred_expr)
    gt_size = expression_tree_size(gt_expr)

    if gt_size == 0:
        return esm

    complexity_ratio = pred_size / gt_size
    penalty = alpha * max(0, complexity_ratio - 1)
    return esm * (1 - penalty)


def bootstrap_confidence_interval(
    values: List[float], n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, low, high)."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.sort(bootstrap_means)
    alpha = (1 - confidence) / 2
    low = bootstrap_means[int(alpha * n_bootstrap)]
    high = bootstrap_means[int((1 - alpha) * n_bootstrap)]
    return float(np.mean(values)), float(low), float(high)


def evaluate_model_predictions(
    predictions: List[List[int]],
    ground_truths: List[List[int]],
    observations: np.ndarray,
    metadata: List[Dict],
    max_eval: int = 5000,
) -> Dict:
    """Full evaluation of model predictions.

    Args:
        predictions: List of predicted token sequences
        ground_truths: List of ground truth token sequences
        observations: Observation arrays (N, MAX_OBS, MAX_VARS+1)
        metadata: List of metadata dicts with tier, n_input_vars, n_obs
        max_eval: Maximum number of samples to evaluate (for speed)

    Returns:
        Dict with per-tier and overall metrics
    """
    n = min(len(predictions), len(ground_truths), max_eval)

    tier_results = defaultdict(lambda: {
        "esm": [], "r2": [], "nted": [], "caa": []
    })

    for i in range(n):
        tier = metadata[i]["tier"]
        n_vars = metadata[i]["n_input_vars"]
        n_obs = metadata[i]["n_obs"]

        pred = predictions[i]
        gt = ground_truths[i]
        obs = observations[i]

        # Exact symbolic match
        esm = exact_symbolic_match(pred, gt)
        tier_results[tier]["esm"].append(float(esm))

        # R² score
        r2 = r2_score(pred, obs, n_vars, n_obs)
        tier_results[tier]["r2"].append(r2)

        # Normalized tree edit distance
        nted = tree_edit_distance(pred, gt)
        tier_results[tier]["nted"].append(nted)

        # Complexity-adjusted accuracy
        caa = complexity_adjusted_accuracy(pred, gt)
        tier_results[tier]["caa"].append(caa)

    # Aggregate results
    results = {"per_tier": {}, "overall": {}}

    all_esm, all_r2, all_nted, all_caa = [], [], [], []

    for tier in sorted(tier_results.keys()):
        tr = tier_results[tier]
        esm_mean, esm_low, esm_high = bootstrap_confidence_interval(tr["esm"])
        r2_vals = [v for v in tr["r2"] if v > -1]  # Filter invalid
        r2_mean = np.mean(r2_vals) if r2_vals else 0.0
        nted_mean = np.mean(tr["nted"])
        caa_mean = np.mean(tr["caa"])

        results["per_tier"][str(tier)] = {
            "exact_match_rate": esm_mean,
            "exact_match_ci": [esm_low, esm_high],
            "r2_score": float(r2_mean),
            "tree_edit_distance": float(nted_mean),
            "complexity_adjusted_accuracy": float(caa_mean),
            "n_samples": len(tr["esm"]),
        }

        all_esm.extend(tr["esm"])
        all_r2.extend(r2_vals)
        all_nted.extend(tr["nted"])
        all_caa.extend(tr["caa"])

    # Overall
    results["overall"] = {
        "exact_match_rate": float(np.mean(all_esm)) if all_esm else 0.0,
        "r2_score": float(np.mean(all_r2)) if all_r2 else 0.0,
        "tree_edit_distance": float(np.mean(all_nted)) if all_nted else 1.0,
        "complexity_adjusted_accuracy": float(np.mean(all_caa)) if all_caa else 0.0,
        "n_samples": len(all_esm),
    }

    return results


def format_results_latex(results: Dict) -> str:
    """Format results as a LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-tier evaluation results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Tier & ESM (\%) & R² & NTED & CAA \\",
        r"\midrule",
    ]
    for tier in sorted(results["per_tier"].keys()):
        tr = results["per_tier"][tier]
        lines.append(
            f"Tier {tier} & {tr['exact_match_rate']*100:.1f} & "
            f"{tr['r2_score']:.3f} & {tr['tree_edit_distance']:.3f} & "
            f"{tr['complexity_adjusted_accuracy']:.3f} \\\\"
        )
    lines.append(r"\midrule")
    ov = results["overall"]
    lines.append(
        f"Overall & {ov['exact_match_rate']*100:.1f} & "
        f"{ov['r2_score']:.3f} & {ov['tree_edit_distance']:.3f} & "
        f"{ov['complexity_adjusted_accuracy']:.3f} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)
