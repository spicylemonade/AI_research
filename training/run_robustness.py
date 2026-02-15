"""Robustness evaluation of PhysMDT under noise, limited data, and variable scaling.

Implements item_020 of the research rubric:
  1. Noise robustness   -- accuracy at noise levels [0.0, 0.01, 0.05, 0.1, 0.2]
                           on Tier 3 training equations
  2. Data efficiency    -- accuracy with [5, 10, 20, 50, 100] observation points
                           per equation (Tier 3)
  3. Variable count     -- accuracy on equations with 2, 3, 4, 5 variables
                           (training equations filtered by variable count)

Produces:
    results/robustness_results.json
    figures/noise_robustness.png
    figures/data_efficiency.png

Usage:
    python training/run_robustness.py
    python training/run_robustness.py --checkpoint checkpoints/physmdt_best.pt
    python training/run_robustness.py --quick          # smoke test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from data.equations import get_training_equations, get_equations_by_tier
from data.physics_generator import PhysicsDataset
from data.tokenizer import ExprTokenizer, PAD_IDX, SOS_IDX, EOS_IDX, MASK_IDX
from evaluation.metrics import symbolic_equivalence, numeric_r2, token_edit_distance
from models.physmdt import PhysMDT, PhysMDTConfig
from models.refinement import generate_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness evaluation: noise, data efficiency, variable scaling."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/physmdt_best.pt",
        help="Path to the trained PhysMDT checkpoint (default: checkpoints/physmdt_best.pt).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: fewer samples, fewer candidates, fewer refinement steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a PhysMDT model from a checkpoint file.

    Returns (model, config) on success, or (None, None) if the checkpoint
    does not exist or cannot be loaded.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("  Train the model first with: python training/train_physmdt.py")
        return None, None

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"[ERROR] Failed to load checkpoint: {exc}")
        return None, None

    # Reconstruct config
    raw_config = ckpt.get("config", None)
    if raw_config is None:
        print("[WARN] Checkpoint has no 'config' key; using default PhysMDTConfig.")
        config = PhysMDTConfig()
    elif isinstance(raw_config, PhysMDTConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = PhysMDTConfig(**raw_config)
    else:
        config = raw_config

    model = PhysMDT(config).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        print("[WARN] No model_state_dict found in checkpoint; using freshly-initialised weights.")

    model.eval()
    return model, config


def _decode_tokens_to_expr(token_ids, tokenizer: ExprTokenizer):
    """Decode a token-id sequence to a SymPy expression.

    Strips everything after the first EOS (inclusive) and removes PAD/MASK
    tokens before decoding.  Returns the decoded SymPy expression or None.
    """
    ids = list(token_ids)

    if EOS_IDX in ids:
        ids = ids[: ids.index(EOS_IDX) + 1]

    ids = [i for i in ids if i not in (PAD_IDX, MASK_IDX)]

    if not ids:
        return None

    try:
        expr = tokenizer.decode(ids, strip_special=True)
        return expr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-equation evaluation
# ---------------------------------------------------------------------------

def evaluate_equation(
    model: PhysMDT,
    eq,
    device: torch.device,
    tokenizer: ExprTokenizer,
    n_test_samples: int = 5,
    n_points: int = 100,
    noise_level: float = 0.0,
    n_candidates: int = 8,
    n_steps: int = 64,
    seed: int = 42,
) -> Dict:
    """Evaluate the model on a single equation under the given conditions.

    Returns a dict with symbolic_accuracy, mean_r2, and per-sample details.
    """
    ds = PhysicsDataset(
        equations=[eq],
        n_samples=n_test_samples,
        n_points=n_points,
        noise_level=noise_level,
        seed=seed,
    )

    if len(ds) == 0:
        return {
            "equation_id": eq.id,
            "equation_name": eq.name,
            "tier": eq.tier,
            "n_vars": len(eq.variables),
            "n_samples": 0,
            "symbolic_accuracy": 0.0,
            "mean_r2": -1.0,
        }

    sym_correct = 0
    r2_scores: List[float] = []
    seq_len = model.config.max_expr_len

    for i in range(min(len(ds), n_test_samples)):
        sample = ds[i]
        obs = sample["observations"].unsqueeze(0).to(device)
        obs_mask = sample["obs_mask"].unsqueeze(0).to(device)
        true_token_ids = sample["tokens"][: sample["token_len"]].tolist()

        with torch.no_grad():
            pred_tokens, confidence = generate_candidates(
                model=model,
                observations=obs,
                obs_mask=obs_mask,
                seq_len=seq_len,
                n_steps=n_steps,
                n_candidates=n_candidates,
                temperature=1.0,
                device=device,
            )

        gen_ids = pred_tokens[0].cpu().tolist()
        pred_expr = _decode_tokens_to_expr(gen_ids, tokenizer)
        true_expr = _decode_tokens_to_expr(true_token_ids, tokenizer)

        is_equiv = False
        r2 = -1.0
        if pred_expr is not None and true_expr is not None:
            try:
                is_equiv = symbolic_equivalence(pred_expr, true_expr)
            except Exception:
                is_equiv = False
            try:
                r2 = numeric_r2(pred_expr, true_expr, n_points=200)
            except Exception:
                r2 = -1.0

        if is_equiv:
            sym_correct += 1
        r2_scores.append(r2)

    n_evaluated = min(len(ds), n_test_samples)
    valid_r2 = [r for r in r2_scores if r >= 0]

    return {
        "equation_id": eq.id,
        "equation_name": eq.name,
        "tier": eq.tier,
        "n_vars": len(eq.variables),
        "n_samples": n_evaluated,
        "symbolic_accuracy": sym_correct / max(n_evaluated, 1),
        "mean_r2": float(np.mean(valid_r2)) if valid_r2 else -1.0,
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_noise_robustness(
    model: PhysMDT,
    device: torch.device,
    tokenizer: ExprTokenizer,
    noise_levels: List[float],
    n_test_samples: int,
    n_points: int,
    n_candidates: int,
    n_steps: int,
    seed: int,
) -> Dict:
    """Evaluate accuracy at varying noise levels on Tier 3 training equations."""
    tier3_train = [eq for eq in get_equations_by_tier(3) if not eq.held_out]
    print(f"  Tier 3 training equations: {len(tier3_train)}")

    results: Dict = {"noise_levels": noise_levels, "per_level": []}

    for nl in noise_levels:
        print(f"  Noise level: {nl} ...", flush=True)
        level_results: List[Dict] = []
        for eq in tier3_train:
            r = evaluate_equation(
                model=model,
                eq=eq,
                device=device,
                tokenizer=tokenizer,
                n_test_samples=n_test_samples,
                n_points=n_points,
                noise_level=nl,
                n_candidates=n_candidates,
                n_steps=n_steps,
                seed=seed,
            )
            level_results.append(r)

        accs = [r["symbolic_accuracy"] for r in level_results]
        r2s = [r["mean_r2"] for r in level_results if r["mean_r2"] >= 0]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_r2 = float(np.mean(r2s)) if r2s else -1.0
        std_acc = float(np.std(accs)) if len(accs) > 1 else 0.0
        std_r2 = float(np.std(r2s)) if len(r2s) > 1 else 0.0

        level_summary = {
            "noise_level": nl,
            "mean_symbolic_accuracy": mean_acc,
            "std_symbolic_accuracy": std_acc,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "n_equations": len(tier3_train),
            "per_equation": level_results,
        }
        results["per_level"].append(level_summary)
        print(
            f"    Acc={mean_acc*100:.1f}% (+/-{std_acc*100:.1f}%)  "
            f"R2={mean_r2:.4f} (+/-{std_r2:.4f})"
        )

    return results


def run_data_efficiency(
    model: PhysMDT,
    device: torch.device,
    tokenizer: ExprTokenizer,
    n_points_list: List[int],
    n_test_samples: int,
    noise_level: float,
    n_candidates: int,
    n_steps: int,
    seed: int,
) -> Dict:
    """Evaluate accuracy with varying numbers of observation points on Tier 3."""
    tier3_train = [eq for eq in get_equations_by_tier(3) if not eq.held_out]
    print(f"  Tier 3 training equations: {len(tier3_train)}")

    results: Dict = {"n_points_list": n_points_list, "per_n_points": []}

    for np_ in n_points_list:
        print(f"  n_points: {np_} ...", flush=True)
        np_results: List[Dict] = []
        for eq in tier3_train:
            r = evaluate_equation(
                model=model,
                eq=eq,
                device=device,
                tokenizer=tokenizer,
                n_test_samples=n_test_samples,
                n_points=np_,
                noise_level=noise_level,
                n_candidates=n_candidates,
                n_steps=n_steps,
                seed=seed,
            )
            np_results.append(r)

        accs = [r["symbolic_accuracy"] for r in np_results]
        r2s = [r["mean_r2"] for r in np_results if r["mean_r2"] >= 0]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_r2 = float(np.mean(r2s)) if r2s else -1.0
        std_acc = float(np.std(accs)) if len(accs) > 1 else 0.0
        std_r2 = float(np.std(r2s)) if len(r2s) > 1 else 0.0

        np_summary = {
            "n_points": np_,
            "mean_symbolic_accuracy": mean_acc,
            "std_symbolic_accuracy": std_acc,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "n_equations": len(tier3_train),
            "per_equation": np_results,
        }
        results["per_n_points"].append(np_summary)
        print(
            f"    Acc={mean_acc*100:.1f}% (+/-{std_acc*100:.1f}%)  "
            f"R2={mean_r2:.4f} (+/-{std_r2:.4f})"
        )

    return results


def run_variable_count_scaling(
    model: PhysMDT,
    device: torch.device,
    tokenizer: ExprTokenizer,
    var_counts: List[int],
    n_test_samples: int,
    n_points: int,
    noise_level: float,
    n_candidates: int,
    n_steps: int,
    seed: int,
) -> Dict:
    """Evaluate accuracy on training equations grouped by variable count."""
    all_train = get_training_equations()
    print(f"  Total training equations: {len(all_train)}")

    results: Dict = {"var_counts": var_counts, "per_var_count": []}

    for nv in var_counts:
        eqs_nv = [eq for eq in all_train if len(eq.variables) == nv]
        if not eqs_nv:
            print(f"  n_vars={nv}: no training equations found, skipping")
            results["per_var_count"].append({
                "n_vars": nv,
                "mean_symbolic_accuracy": 0.0,
                "std_symbolic_accuracy": 0.0,
                "mean_r2": -1.0,
                "std_r2": 0.0,
                "n_equations": 0,
                "per_equation": [],
            })
            continue

        print(f"  n_vars={nv} ({len(eqs_nv)} equations) ...", flush=True)
        nv_results: List[Dict] = []
        for eq in eqs_nv:
            r = evaluate_equation(
                model=model,
                eq=eq,
                device=device,
                tokenizer=tokenizer,
                n_test_samples=n_test_samples,
                n_points=n_points,
                noise_level=noise_level,
                n_candidates=n_candidates,
                n_steps=n_steps,
                seed=seed,
            )
            nv_results.append(r)

        accs = [r["symbolic_accuracy"] for r in nv_results]
        r2s = [r["mean_r2"] for r in nv_results if r["mean_r2"] >= 0]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_r2 = float(np.mean(r2s)) if r2s else -1.0
        std_acc = float(np.std(accs)) if len(accs) > 1 else 0.0
        std_r2 = float(np.std(r2s)) if len(r2s) > 1 else 0.0

        vc_summary = {
            "n_vars": nv,
            "mean_symbolic_accuracy": mean_acc,
            "std_symbolic_accuracy": std_acc,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "n_equations": len(eqs_nv),
            "per_equation": nv_results,
        }
        results["per_var_count"].append(vc_summary)
        print(
            f"    Acc={mean_acc*100:.1f}% (+/-{std_acc*100:.1f}%)  "
            f"R2={mean_r2:.4f} (+/-{std_r2:.4f})"
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_noise_robustness(noise_results: Dict, output_path: str) -> None:
    """Generate publication-quality noise robustness figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    levels = [entry["noise_level"] for entry in noise_results["per_level"]]
    accs = [entry["mean_symbolic_accuracy"] for entry in noise_results["per_level"]]
    acc_stds = [entry["std_symbolic_accuracy"] for entry in noise_results["per_level"]]
    r2s = [entry["mean_r2"] for entry in noise_results["per_level"]]
    r2_stds = [entry["std_r2"] for entry in noise_results["per_level"]]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))

    color_acc = "#2166ac"
    color_r2 = "#b2182b"

    # Symbolic accuracy (left y-axis)
    ax1.set_xlabel("Noise Level (fraction of |y|)")
    ax1.set_ylabel("Symbolic Accuracy", color=color_acc)
    line1 = ax1.errorbar(
        levels, accs, yerr=acc_stds,
        color=color_acc, marker="o", linewidth=2, markersize=7,
        capsize=4, capthick=1.5, label="Symbolic Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(-0.05, 1.05)

    # R^2 (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean R$^2$", color=color_r2)
    line2 = ax2.errorbar(
        levels, r2s, yerr=r2_stds,
        color=color_r2, marker="s", linewidth=2, markersize=7,
        capsize=4, capthick=1.5, linestyle="--", label="Mean R$^2$",
    )
    ax2.tick_params(axis="y", labelcolor=color_r2)
    ax2.set_ylim(-0.05, 1.05)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9)

    ax1.set_title("Noise Robustness (Tier 3 Equations)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def plot_data_efficiency(de_results: Dict, output_path: str) -> None:
    """Generate publication-quality data efficiency figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    n_points = [entry["n_points"] for entry in de_results["per_n_points"]]
    accs = [entry["mean_symbolic_accuracy"] for entry in de_results["per_n_points"]]
    acc_stds = [entry["std_symbolic_accuracy"] for entry in de_results["per_n_points"]]
    r2s = [entry["mean_r2"] for entry in de_results["per_n_points"]]
    r2_stds = [entry["std_r2"] for entry in de_results["per_n_points"]]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))

    color_acc = "#2166ac"
    color_r2 = "#b2182b"

    # Symbolic accuracy (left y-axis)
    ax1.set_xlabel("Number of Observation Points")
    ax1.set_ylabel("Symbolic Accuracy", color=color_acc)
    line1 = ax1.errorbar(
        n_points, accs, yerr=acc_stds,
        color=color_acc, marker="o", linewidth=2, markersize=7,
        capsize=4, capthick=1.5, label="Symbolic Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xscale("log")
    ax1.set_xticks(n_points)
    ax1.set_xticklabels([str(n) for n in n_points])

    # R^2 (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean R$^2$", color=color_r2)
    line2 = ax2.errorbar(
        n_points, r2s, yerr=r2_stds,
        color=color_r2, marker="s", linewidth=2, markersize=7,
        capsize=4, capthick=1.5, linestyle="--", label="Mean R$^2$",
    )
    ax2.tick_params(axis="y", labelcolor=color_r2)
    ax2.set_ylim(-0.05, 1.05)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9)

    ax1.set_title("Data Efficiency (Tier 3 Equations)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Quick mode: {args.quick}")
    print(f"Seed: {args.seed}")
    print()

    # ---- Load model -------------------------------------------------------
    model, config = _load_checkpoint(args.checkpoint, device)
    if model is None:
        sys.exit(1)

    n_params = model.count_parameters()
    print(f"Model loaded: {n_params:,} parameters ({n_params / 1e6:.1f}M)")
    print()

    tokenizer = ExprTokenizer()

    # ---- Configure parameters (full vs quick) -----------------------------
    if args.quick:
        n_test_samples = 2
        n_candidates = 2
        n_steps = 8
        default_n_points = 30
        noise_levels = [0.0, 0.05, 0.2]
        n_points_list = [5, 20, 50]
        var_counts = [2, 3, 4]
        print("[Quick mode] Using reduced parameters for smoke testing.\n")
    else:
        n_test_samples = 5
        n_candidates = 8
        n_steps = 64
        default_n_points = 100
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        n_points_list = [5, 10, 20, 50, 100]
        var_counts = [2, 3, 4, 5]

    default_noise = 0.01

    overall_start = time.time()

    # ====================================================================
    # Experiment 1: Noise Robustness
    # ====================================================================
    print("=" * 70)
    print("EXPERIMENT 1: Noise Robustness (Tier 3 Equations)")
    print("=" * 70)

    noise_results = run_noise_robustness(
        model=model,
        device=device,
        tokenizer=tokenizer,
        noise_levels=noise_levels,
        n_test_samples=n_test_samples,
        n_points=default_n_points,
        n_candidates=n_candidates,
        n_steps=n_steps,
        seed=args.seed,
    )
    print()

    # ====================================================================
    # Experiment 2: Data Efficiency
    # ====================================================================
    print("=" * 70)
    print("EXPERIMENT 2: Data Efficiency (Tier 3 Equations)")
    print("=" * 70)

    data_efficiency_results = run_data_efficiency(
        model=model,
        device=device,
        tokenizer=tokenizer,
        n_points_list=n_points_list,
        n_test_samples=n_test_samples,
        noise_level=default_noise,
        n_candidates=n_candidates,
        n_steps=n_steps,
        seed=args.seed,
    )
    print()

    # ====================================================================
    # Experiment 3: Variable Count Scaling
    # ====================================================================
    print("=" * 70)
    print("EXPERIMENT 3: Variable Count Scaling")
    print("=" * 70)

    var_scaling_results = run_variable_count_scaling(
        model=model,
        device=device,
        tokenizer=tokenizer,
        var_counts=var_counts,
        n_test_samples=n_test_samples,
        n_points=default_n_points,
        noise_level=default_noise,
        n_candidates=n_candidates,
        n_steps=n_steps,
        seed=args.seed,
    )
    print()

    total_time = time.time() - overall_start

    # ====================================================================
    # Save results
    # ====================================================================
    gpu_mem_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )

    combined_results = {
        "model": "PhysMDT",
        "checkpoint": args.checkpoint,
        "total_params": n_params,
        "device": str(device),
        "seed": args.seed,
        "quick_mode": args.quick,
        "total_time_seconds": float(total_time),
        "peak_gpu_memory_gb": float(gpu_mem_gb),
        "evaluation_config": {
            "n_test_samples": n_test_samples,
            "n_candidates": n_candidates,
            "n_steps": n_steps,
            "default_n_points": default_n_points,
            "default_noise_level": default_noise,
        },
        "noise_robustness": noise_results,
        "data_efficiency": data_efficiency_results,
        "variable_count_scaling": var_scaling_results,
    }

    os.makedirs("results", exist_ok=True)
    results_path = "results/robustness_results.json"
    with open(results_path, "w") as f:
        json.dump(combined_results, f, indent=2, default=str)
    print(f"Saved results to {results_path}")

    # ====================================================================
    # Generate figures
    # ====================================================================
    print()
    print("Generating figures...")

    os.makedirs("figures", exist_ok=True)

    plot_noise_robustness(
        noise_results,
        output_path="figures/noise_robustness.png",
    )

    plot_data_efficiency(
        data_efficiency_results,
        output_path="figures/data_efficiency.png",
    )

    # ====================================================================
    # Final summary
    # ====================================================================
    print()
    print("=" * 70)
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Model:          PhysMDT ({n_params:,} params)")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Quick mode:     {args.quick}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Peak GPU mem:   {gpu_mem_gb:.2f} GB")
    print()

    # Noise summary
    print("  Noise Robustness (Tier 3):")
    for entry in noise_results["per_level"]:
        print(
            f"    noise={entry['noise_level']:<6}  "
            f"Acc={entry['mean_symbolic_accuracy']*100:5.1f}%  "
            f"R2={entry['mean_r2']:.4f}"
        )

    # Data efficiency summary
    print()
    print("  Data Efficiency (Tier 3):")
    for entry in data_efficiency_results["per_n_points"]:
        print(
            f"    n_points={entry['n_points']:<4}  "
            f"Acc={entry['mean_symbolic_accuracy']*100:5.1f}%  "
            f"R2={entry['mean_r2']:.4f}"
        )

    # Variable count summary
    print()
    print("  Variable Count Scaling:")
    for entry in var_scaling_results["per_var_count"]:
        print(
            f"    n_vars={entry['n_vars']}  "
            f"({entry['n_equations']:2d} eqs)  "
            f"Acc={entry['mean_symbolic_accuracy']*100:5.1f}%  "
            f"R2={entry['mean_r2']:.4f}"
        )

    print()
    print(f"  Results:   {results_path}")
    print(f"  Figures:   figures/noise_robustness.png")
    print(f"             figures/data_efficiency.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
