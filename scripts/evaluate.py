"""
Evaluate a trained PhysDiffuser+ checkpoint on the Feynman or OOD benchmark.

Usage
-----
    # Evaluate on all 120 Feynman equations
    python scripts/evaluate.py --benchmark feynman --checkpoint models/physdiffuser_plus_checkpoint.pt

    # Evaluate on 20 out-of-distribution equations
    python scripts/evaluate.py --benchmark ood --checkpoint models/physdiffuser_plus_checkpoint.pt

See ``python scripts/evaluate.py --help`` for all options.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.model.phys_diffuser_plus import PhysDiffuserPlus, PhysDiffuserPlusConfig
from src.eval.metrics import (
    r_squared,
    symbolic_equivalence,
    tree_edit_distance,
    equation_complexity,
    evaluate_equation,
    aggregate_results,
)


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------
BENCHMARK_PATHS = {
    "feynman": os.path.join(_REPO_ROOT, "benchmarks", "feynman_equations.json"),
    "ood": os.path.join(_REPO_ROOT, "benchmarks", "ood_equations.json"),
}


def load_benchmark(name_or_path):
    """Return (equations_list, source_name)."""
    path = BENCHMARK_PATHS.get(name_or_path, name_or_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path) as fh:
        data = json.load(fh)
    equations = data.get("equations", data) if isinstance(data, dict) else data
    return equations, os.path.basename(path)


# ---------------------------------------------------------------------------
# Parse symbolic formula to prefix tokens (reused from train_baseline.py)
# ---------------------------------------------------------------------------
def parse_symbolic_to_prefix(symbolic):
    """Parse symbolic expression like ``mul(x1, pow(x2, 2))`` into prefix token list."""

    def parse_expr(s, pos):
        while pos < len(s) and s[pos] in " ,":
            pos += 1
        if pos >= len(s):
            return [], pos
        start = pos
        while pos < len(s) and (s[pos].isalnum() or s[pos] in "_.-"):
            pos += 1
        name = s[start:pos]
        while pos < len(s) and s[pos] == " ":
            pos += 1
        if pos < len(s) and s[pos] == "(":
            pos += 1
            tokens_list = [name]
            while pos < len(s) and s[pos] != ")":
                while pos < len(s) and s[pos] in " ,":
                    pos += 1
                if pos < len(s) and s[pos] == ")":
                    break
                child_tokens, pos = parse_expr(s, pos)
                tokens_list.extend(child_tokens)
            if pos < len(s) and s[pos] == ")":
                pos += 1
            return tokens_list, pos
        else:
            return [name], pos

    result, _ = parse_expr(symbolic, 0)
    return result


# ---------------------------------------------------------------------------
# Single-equation evaluation
# ---------------------------------------------------------------------------
def evaluate_single(model, eq, seed=42, idx=0, num_test_points=200):
    """Evaluate the model on a single benchmark equation.

    Returns a dict with per-equation metrics, or ``None`` on failure.
    """
    try:
        formula_python = eq["formula_python"]
        variables = eq["variables"]
        num_vars = eq["num_variables"]

        rng = np.random.RandomState(seed + idx)
        X = rng.uniform(0.1, 5.0, size=(num_test_points, num_vars))

        safe_globals = {
            "__builtins__": {},
            "pi": np.pi, "e": np.e,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            "abs": np.abs,
        }
        for vi, var in enumerate(variables):
            safe_globals[var["name"]] = X[:, vi]

        y = eval(formula_python, safe_globals)  # noqa: S307

        if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.any(np.abs(y) > 1e6):
            return {
                "id": eq.get("id", idx),
                "name": eq.get("name", ""),
                "tier": eq.get("difficulty_tier", "unknown"),
                "exact_match": False,
                "r_squared": -1.0,
                "tree_edit_distance": 1.0,
                "predicted": [],
                "error": "invalid_data",
            }

        encoded = model.encoder.encode_observations(X, y, num_vars)
        obs = encoded.unsqueeze(0)

        with torch.no_grad():
            result = model.predict(obs, X, y)

        pred = result["prediction"]

        # Ground-truth prefix tokens
        true_tokens = parse_symbolic_to_prefix(eq.get("formula_symbolic", ""))

        # R-squared on fresh test points
        X_test = rng.uniform(0.1, 5.0, size=(1000, num_vars))
        for vi, var in enumerate(variables):
            safe_globals[var["name"]] = X_test[:, vi]
        y_test = eval(formula_python, safe_globals)  # noqa: S307
        if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
            r2 = -1.0
        else:
            r2 = r_squared(pred, X_test, y_test, num_vars)

        # Symbolic equivalence
        exact = symbolic_equivalence(pred, true_tokens) if (pred and true_tokens) else False
        if exact is None:
            exact = False

        ted = tree_edit_distance(pred, true_tokens) if (pred and true_tokens) else 1.0

        return {
            "id": eq.get("id", idx),
            "name": eq.get("name", ""),
            "tier": eq.get("difficulty_tier", "unknown"),
            "exact_match": bool(exact),
            "r_squared": float(r2),
            "tree_edit_distance": float(ted),
            "predicted": pred,
            "true_tokens": true_tokens,
            "pred_complexity": len(pred),
            "true_complexity": len(true_tokens),
            "timings": result.get("timings", {}),
        }
    except Exception as exc:
        return {
            "id": eq.get("id", idx),
            "name": eq.get("name", ""),
            "tier": eq.get("difficulty_tier", "unknown"),
            "exact_match": False,
            "r_squared": -1.0,
            "tree_edit_distance": 1.0,
            "predicted": [],
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a PhysDiffuser+ checkpoint on a symbolic regression benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--benchmark",
        type=str,
        default="feynman",
        choices=["feynman", "ood"],
        help="Benchmark to evaluate on: 'feynman' (120 equations) or 'ood' (20 equations).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_REPO_ROOT, "models", "physdiffuser_plus_checkpoint.pt"),
        help="Path to model checkpoint.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "results"),
        help="Directory to save evaluation results.",
    )
    p.add_argument("--num_threads", type=int, default=4, help="PyTorch CPU threads.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.num_threads)

    print("=" * 60)
    print(f"PhysDiffuser+ Evaluation  --  benchmark={args.benchmark}")
    print("=" * 60)

    # ---- Load checkpoint ----
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        print("Train a model first with: python scripts/train.py")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Reconstruct config
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        config = PhysDiffuserPlusConfig(**{
            k: v for k, v in ckpt["config"].items()
            if k in PhysDiffuserPlusConfig.__dataclass_fields__
        })
    else:
        config = PhysDiffuserPlusConfig()

    model = PhysDiffuserPlus(config)
    state_key = "model_state" if "model_state" in ckpt else "state_dict"
    if state_key in ckpt:
        model.load_state_dict(ckpt[state_key], strict=False)
    print(f"Model loaded ({model.count_parameters():,} parameters)")

    # ---- Load benchmark ----
    equations, bench_name = load_benchmark(args.benchmark)
    print(f"Benchmark: {bench_name} ({len(equations)} equations)\n")

    # ---- Evaluate ----
    model.eval()
    all_results = []
    t0 = time.time()

    for idx, eq in enumerate(equations):
        res = evaluate_single(model, eq, seed=args.seed, idx=idx)
        all_results.append(res)

        if (idx + 1) % 20 == 0:
            n_exact = sum(1 for r in all_results if r["exact_match"])
            n_good = sum(1 for r in all_results if r["r_squared"] > 0.9)
            print(
                f"  Evaluated {idx + 1}/{len(equations)}: "
                f"exact={n_exact}, R^2>0.9={n_good}"
            )

    eval_time = time.time() - t0

    # ---- Aggregate ----
    n_total = len(all_results)
    n_exact = sum(1 for r in all_results if r["exact_match"])
    n_r2_good = sum(1 for r in all_results if r["r_squared"] > 0.9)
    r2_scores = [r["r_squared"] for r in all_results]

    print(f"\n{'=' * 60}")
    print(f"RESULTS  ({args.benchmark})")
    print(f"{'=' * 60}")
    print(f"Total equations evaluated: {n_total}")
    if n_total > 0:
        print(f"Exact match rate: {n_exact}/{n_total} ({n_exact / n_total * 100:.1f}%)")
        print(f"R^2 > 0.9: {n_r2_good}/{n_total} ({n_r2_good / n_total * 100:.1f}%)")
        print(f"Mean R^2: {np.mean(r2_scores):.4f}")
        print(f"Evaluation time: {eval_time:.1f}s ({eval_time / n_total:.2f}s per eq)")

    # ---- Per-tier breakdown ----
    tiers = sorted(set(r["tier"] for r in all_results))
    tier_summary = {}
    print(f"\nPer-tier breakdown:")
    for tier in tiers:
        tier_results = [r for r in all_results if r["tier"] == tier]
        tn = len(tier_results)
        if tn == 0:
            continue
        te = sum(1 for r in tier_results if r["exact_match"])
        tr2 = [r["r_squared"] for r in tier_results]
        tg = sum(1 for v in tr2 if v > 0.9)
        print(
            f"  {tier:12s}: {te}/{tn} exact ({te / tn * 100:.0f}%), "
            f"R^2>0.9: {tg}/{tn}, mean R^2: {np.mean(tr2):.3f}"
        )
        tier_summary[tier] = {
            "n": tn,
            "exact_match": te,
            "exact_match_rate": te / tn,
            "r2_above_0.9": tg,
            "mean_r2": float(np.mean(tr2)),
        }

    # ---- Save results ----
    os.makedirs(args.results_dir, exist_ok=True)
    output_name = f"eval_{args.benchmark}.json"
    output_path = os.path.join(args.results_dir, output_name)

    # Make results JSON-serialisable
    serialisable = []
    for r in all_results:
        sr = {k: v for k, v in r.items()}
        serialisable.append(sr)

    payload = {
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "overall": {
            "n": n_total,
            "exact_match_rate": n_exact / max(1, n_total),
            "exact_match_count": n_exact,
            "mean_r_squared": float(np.mean(r2_scores)) if r2_scores else 0.0,
            "r2_above_0.9": n_r2_good,
            "eval_time_seconds": eval_time,
        },
        "per_tier": tier_summary,
        "per_equation": serialisable,
    }
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
