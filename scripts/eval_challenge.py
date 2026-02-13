#!/usr/bin/env python3
"""
Challenge set evaluation for item_020 of the research rubric.

Loads the 50 complex Newtonian physics equations from data/challenge_set.json,
generates numerical observations, runs PhysMDT and the AR baseline, computes
all metrics, and writes the results to results/challenge/eval_results.json.

Usage:
    python scripts/eval_challenge.py [--n_obs 20] [--mdt_steps 20] [--device cpu]
"""

import os
import sys
import json
import math
import time
import random
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.phys_mdt import build_phys_mdt
from src.baseline_ar import build_baseline_model
from src.tokenizer import (
    encode,
    decode,
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    MASK_IDX,
    MAX_SEQ_LEN,
)
from src.metrics import evaluate_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
CHALLENGE_SET_PATH = os.path.join(REPO_ROOT, "data", "challenge_set.json")
MDT_MODEL_PATH = os.path.join(REPO_ROOT, "results", "phys_mdt", "model.pt")
AR_MODEL_PATH = os.path.join(REPO_ROOT, "results", "baseline_ar", "model.pt")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "challenge")


# ---------------------------------------------------------------------------
# Numerical observation generator
# ---------------------------------------------------------------------------

def _safe_eval(infix: str, variables: Dict[str, float]) -> Optional[float]:
    """Evaluate an infix equation string safely given a variable mapping.

    Replaces common math functions with their ``math`` module equivalents and
    uses a restricted eval environment.
    """
    expr = infix
    expr = expr.replace("sin(", "math.sin(")
    expr = expr.replace("cos(", "math.cos(")
    expr = expr.replace("tan(", "math.tan(")
    expr = expr.replace("exp(", "math.exp(")
    expr = expr.replace("log(", "math.log(")
    expr = expr.replace("sqrt(", "math.sqrt(")
    expr = expr.replace("abs(", "abs(")
    expr = expr.replace("atan(", "math.atan(")
    expr = expr.replace("asin(", "math.asin(")
    expr = expr.replace("acos(", "math.acos(")
    expr = expr.replace("pi", str(math.pi))

    # Build the local namespace: coefficients merged with sampled variables
    ns: Dict[str, Any] = {"__builtins__": {}, "math": math}
    ns.update(variables)

    try:
        val = eval(compile(expr, "<challenge_eq>", "eval"), ns)  # noqa: S307
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
    except Exception:
        pass
    return None


def generate_observations(
    entry: Dict[str, Any],
    n_obs: int = 20,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Any]]:
    """Generate observation pairs (x, y) for a single challenge equation.

    The *infix* string may contain free variables (with declared ranges) and
    fixed coefficients.  We substitute the coefficients into the expression
    first, then sample random values for the remaining free variables.

    Returns ``None`` if fewer than 5 valid data points can be generated.
    """
    if rng is None:
        rng = random.Random(SEED)

    infix_template = entry["infix"]
    coefficients = entry.get("coefficients", {})
    variable_ranges = entry.get("variables", {})

    # Substitute coefficients into the infix string so that only free
    # variables remain.  We sort by name length (descending) so that e.g.
    # "omega_d" is replaced before "omega".
    infix = infix_template
    for name in sorted(coefficients, key=len, reverse=True):
        val = coefficients[name]
        infix = infix.replace(name, str(val))

    x_data: List[Dict[str, float]] = []
    y_data: List[float] = []

    for _ in range(n_obs * 3):  # over-sample to account for failures
        point: Dict[str, float] = {}
        for var, var_range in variable_ranges.items():
            lo, hi = var_range
            point[var] = rng.uniform(lo, hi)

        y = _safe_eval(infix, point)
        if y is not None:
            x_data.append(point)
            y_data.append(round(y, 8))

        if len(x_data) >= n_obs:
            break

    if len(x_data) < 5:
        return None

    return {"x": x_data[:n_obs], "y": y_data[:n_obs]}


# ---------------------------------------------------------------------------
# Observation tensor builder
# ---------------------------------------------------------------------------

def build_obs_tensor(
    obs_data: Optional[Dict[str, Any]],
    max_vars: int = 6,
    n_obs: int = 20,
) -> torch.Tensor:
    """Convert observation dict to a (1, n_obs, max_vars+1) tensor.

    Each row contains the variable values followed by the target y.
    """
    tensor = torch.zeros(1, n_obs, max_vars + 1)
    if obs_data is None:
        return tensor

    for i in range(min(len(obs_data["x"]), n_obs)):
        vals = list(obs_data["x"][i].values())
        for j in range(min(len(vals), max_vars)):
            tensor[0, i, j] = vals[j]
        tensor[0, i, max_vars] = obs_data["y"][i]
    return tensor


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_phys_mdt(device: torch.device) -> torch.nn.Module:
    """Instantiate and optionally load PhysMDT weights."""
    model = build_phys_mdt(d_model=128, n_layers=3, n_heads=4).to(device)
    if os.path.isfile(MDT_MODEL_PATH):
        state = torch.load(MDT_MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"  [PhysMDT] loaded weights from {MDT_MODEL_PATH}")
    else:
        print(f"  [PhysMDT] checkpoint not found at {MDT_MODEL_PATH}; "
              "using randomly initialised model")
    model.eval()
    return model


def load_ar_baseline(device: torch.device) -> torch.nn.Module:
    """Instantiate and optionally load AR baseline weights."""
    model = build_baseline_model(
        d_model=128, n_heads=4, n_enc_layers=3, n_dec_layers=3, d_ff=512,
    ).to(device)
    if os.path.isfile(AR_MODEL_PATH):
        state = torch.load(AR_MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"  [AR]      loaded weights from {AR_MODEL_PATH}")
    else:
        print(f"  [AR]      checkpoint not found at {AR_MODEL_PATH}; "
              "using randomly initialised model")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_phys_mdt(
    model: torch.nn.Module,
    obs: torch.Tensor,
    max_len: int = MAX_SEQ_LEN,
    n_steps: int = 20,
) -> str:
    """Run PhysMDT masked-diffusion generation and decode to infix string."""
    gen_tokens = model.generate(obs, max_len=max_len, n_steps=n_steps)
    return decode(gen_tokens[0].cpu().tolist())


@torch.no_grad()
def infer_ar_baseline(
    model: torch.nn.Module,
    obs: torch.Tensor,
    max_len: int = MAX_SEQ_LEN,
) -> str:
    """Run AR baseline autoregressive generation and decode to infix string."""
    gen_tokens = model.generate(obs, max_len=max_len, temperature=1.0)
    return decode(gen_tokens[0].cpu().tolist())


# ---------------------------------------------------------------------------
# Category breakdown helper
# ---------------------------------------------------------------------------

def metrics_by_group(
    predictions: List[str],
    ground_truths: List[str],
    groups: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-group metrics."""
    group_preds: Dict[str, List[str]] = defaultdict(list)
    group_truths: Dict[str, List[str]] = defaultdict(list)
    for pred, gt, grp in zip(predictions, ground_truths, groups):
        group_preds[grp].append(pred)
        group_truths[grp].append(gt)

    result: Dict[str, Dict[str, float]] = {}
    for grp in sorted(group_preds):
        result[grp] = evaluate_batch(group_preds[grp], group_truths[grp])
    return result


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PhysMDT and AR baseline on the challenge set"
    )
    parser.add_argument("--n_obs", type=int, default=20,
                        help="Number of observation points per equation")
    parser.add_argument("--mdt_steps", type=int, default=20,
                        help="Number of iterative unmasking steps for PhysMDT")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for observation generation")
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN,
                        help="Maximum token sequence length for generation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Load challenge set
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Challenge Set Evaluation (item_020)")
    print("=" * 60)

    with open(CHALLENGE_SET_PATH, "r") as f:
        challenge_set: List[Dict[str, Any]] = json.load(f)
    print(f"\nLoaded {len(challenge_set)} challenge equations from {CHALLENGE_SET_PATH}")

    # ------------------------------------------------------------------
    # 2. Generate numerical observations for each equation
    # ------------------------------------------------------------------
    print("\nGenerating observations ...")
    obs_rng = random.Random(args.seed)
    observations: List[Optional[Dict[str, Any]]] = []
    for entry in challenge_set:
        obs_data = generate_observations(entry, n_obs=args.n_obs, rng=obs_rng)
        observations.append(obs_data)
    n_with_obs = sum(1 for o in observations if o is not None)
    print(f"  {n_with_obs}/{len(challenge_set)} equations have valid observations")

    # ------------------------------------------------------------------
    # 3. Load models
    # ------------------------------------------------------------------
    print("\nLoading models ...")
    mdt_model = load_phys_mdt(device)
    ar_model = load_ar_baseline(device)
    print(f"  PhysMDT  params: {sum(p.numel() for p in mdt_model.parameters()):,}")
    print(f"  AR       params: {sum(p.numel() for p in ar_model.parameters()):,}")

    # ------------------------------------------------------------------
    # 4. Run inference on each challenge equation
    # ------------------------------------------------------------------
    mdt_predictions: List[str] = []
    ar_predictions: List[str] = []
    ground_truths: List[str] = []
    categories: List[str] = []
    families: List[str] = []
    names: List[str] = []

    mdt_times: List[float] = []
    ar_times: List[float] = []

    print("\nRunning inference ...")
    for idx, entry in enumerate(challenge_set):
        gt_infix = entry["infix"]

        # Substitute coefficients into ground truth to get the concrete equation
        concrete_gt = gt_infix
        for coeff_name in sorted(entry.get("coefficients", {}), key=len, reverse=True):
            concrete_gt = concrete_gt.replace(
                coeff_name, str(entry["coefficients"][coeff_name])
            )

        ground_truths.append(concrete_gt)
        categories.append(entry.get("category", "unknown"))
        families.append(entry.get("family", "unknown"))
        names.append(entry.get("name", f"eq_{idx}"))

        obs_tensor = build_obs_tensor(observations[idx], n_obs=args.n_obs).to(device)

        # --- PhysMDT ---
        t0 = time.time()
        mdt_pred = infer_phys_mdt(mdt_model, obs_tensor,
                                   max_len=args.max_len, n_steps=args.mdt_steps)
        mdt_times.append(time.time() - t0)
        mdt_predictions.append(mdt_pred)

        # --- AR Baseline ---
        t0 = time.time()
        ar_pred = infer_ar_baseline(ar_model, obs_tensor, max_len=args.max_len)
        ar_times.append(time.time() - t0)
        ar_predictions.append(ar_pred)

        if (idx + 1) % 10 == 0 or idx == len(challenge_set) - 1:
            print(f"  [{idx + 1}/{len(challenge_set)}] processed")

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    print("\nComputing metrics ...")

    mdt_overall = evaluate_batch(mdt_predictions, ground_truths)
    ar_overall = evaluate_batch(ar_predictions, ground_truths)

    mdt_by_category = metrics_by_group(mdt_predictions, ground_truths, categories)
    ar_by_category = metrics_by_group(ar_predictions, ground_truths, categories)

    mdt_by_family = metrics_by_group(mdt_predictions, ground_truths, families)
    ar_by_family = metrics_by_group(ar_predictions, ground_truths, families)

    # ------------------------------------------------------------------
    # 6. Build per-equation detail list
    # ------------------------------------------------------------------
    per_equation: List[Dict[str, Any]] = []
    for i in range(len(challenge_set)):
        per_equation.append({
            "name": names[i],
            "category": categories[i],
            "family": families[i],
            "ground_truth": ground_truths[i],
            "mdt_prediction": mdt_predictions[i],
            "ar_prediction": ar_predictions[i],
            "mdt_time_s": round(mdt_times[i], 4),
            "ar_time_s": round(ar_times[i], 4),
        })

    # ------------------------------------------------------------------
    # 7. Assemble and save results
    # ------------------------------------------------------------------
    results = {
        "phys_mdt": {
            "overall": mdt_overall,
            "by_category": mdt_by_category,
            "by_family": mdt_by_family,
            "avg_inference_time_s": round(float(np.mean(mdt_times)), 4),
        },
        "ar_baseline": {
            "overall": ar_overall,
            "by_category": ar_by_category,
            "by_family": ar_by_family,
            "avg_inference_time_s": round(float(np.mean(ar_times)), 4),
        },
        "n_equations": len(challenge_set),
        "n_with_observations": n_with_obs,
        "categories": sorted(set(categories)),
        "per_equation": per_equation,
        "config": {
            "n_obs": args.n_obs,
            "mdt_steps": args.mdt_steps,
            "max_len": args.max_len,
            "seed": args.seed,
            "device": args.device,
            "mdt_model_path": MDT_MODEL_PATH,
            "ar_model_path": AR_MODEL_PATH,
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    def _fmt(metrics: Dict[str, float]) -> str:
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        return "  ".join(parts)

    print(f"\nPhysMDT overall:")
    print(f"  {_fmt(mdt_overall)}")
    print(f"  avg inference time: {np.mean(mdt_times):.3f}s")

    print(f"\nAR Baseline overall:")
    print(f"  {_fmt(ar_overall)}")
    print(f"  avg inference time: {np.mean(ar_times):.3f}s")

    print("\n--- PhysMDT by category ---")
    for cat, metrics in sorted(mdt_by_category.items()):
        print(f"  {cat:25s}  composite={metrics['composite']:.4f}  "
              f"sym_equiv={metrics['symbolic_equivalence']:.4f}")

    print("\n--- AR Baseline by category ---")
    for cat, metrics in sorted(ar_by_category.items()):
        print(f"  {cat:25s}  composite={metrics['composite']:.4f}  "
              f"sym_equiv={metrics['symbolic_equivalence']:.4f}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
