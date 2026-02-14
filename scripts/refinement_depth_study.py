#!/usr/bin/env python3
"""Refinement depth study: sweep refinement steps K and measure composite score.

Sweeps K in {0 (single-pass), 1, 5, 10, 25, 50} on the PhysMDT model,
running 3 seeds (42, 43, 44) per K value.  Saves numerical results to
results/refinement_depth/results.json and a publication-quality figure to
figures/refinement_depth.png.

Timeout: the entire script is budgeted for 5 minutes.  Individual K values
that exceed their per-K budget are skipped gracefully.
"""

import json
import os
import signal
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from data.generator import PhysicsDatasetGenerator
from src.tokenizer import PhysicsTokenizer
from src.phys_mdt import PhysMDT
from src.refinement import SoftMaskRefinement, RefinementConfig
from src.metrics import composite_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(REPO_ROOT, "results", "phys_mdt", "model.pt")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "refinement_depth")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

N_POINTS = 10
MAX_VARS = 5
MAX_SEQ_LEN = 48
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
D_FF = 256

K_VALUES = [0, 1, 5, 10, 25, 50]
SEEDS = [42, 43, 44]
N_TEST = 50  # first 50 of 500 test samples

TOTAL_BUDGET_SEC = 290  # ~4 min 50 sec, leave margin for plotting


# ---------------------------------------------------------------------------
# Timeout helper (SIGALRM based)
# ---------------------------------------------------------------------------
class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("Per-K timeout exceeded")


# ---------------------------------------------------------------------------
# Data helpers (same pattern as eval_quick.py)
# ---------------------------------------------------------------------------
def build_test_tensors(test_data, tokenizer, max_seq_len=MAX_SEQ_LEN,
                       n_points=N_POINTS, max_vars=MAX_VARS):
    X_all, Y_all, tgt_all = [], [], []
    for sample in test_data:
        ox = np.array(sample["observations_x"])[:n_points]
        oy = np.array(sample["observations_y"])[:n_points]
        if ox.shape[0] < n_points:
            p = n_points - ox.shape[0]
            ox = np.vstack([ox, np.zeros((p, ox.shape[1]))])
            oy = np.concatenate([oy, np.zeros(p)])
        if ox.shape[1] < max_vars:
            ox = np.hstack([ox, np.zeros((ox.shape[0], max_vars - ox.shape[1]))])
        elif ox.shape[1] > max_vars:
            ox = ox[:, :max_vars]

        tids = tokenizer.encode(sample["prefix_notation"], max_length=max_seq_len)
        ox = np.clip(np.nan_to_num(ox, 0, 1e6, -1e6), -1e6, 1e6)
        oy = np.clip(np.nan_to_num(oy, 0, 1e6, -1e6), -1e6, 1e6)
        ox /= (np.std(ox) + 1e-8)
        oy /= (np.std(oy) + 1e-8)
        X_all.append(ox)
        Y_all.append(oy)
        tgt_all.append(tids)

    X = torch.tensor(np.clip(np.array(X_all, np.float32), -100, 100))
    Y = torch.tensor(np.clip(np.array(Y_all, np.float32), -100, 100))
    T = torch.tensor(np.array(tgt_all)).long()
    return X, Y, T


def evaluate_with_k(model, X, Y, tokenizer, test_data, K):
    """Evaluate model with refinement depth K.  K=0 means single-pass."""
    model.eval()
    n = X.shape[0]

    with torch.no_grad():
        if K == 0:
            pred = model.generate_single_pass(X, Y, seq_len=MAX_SEQ_LEN)
        else:
            cfg = RefinementConfig(
                total_steps=K,
                cold_restart=False,
                convergence_detection=True,
                confidence_threshold=0.9,
                convergence_patience=2,
                soft_masking=False,
                candidate_tracking=True,
            )
            refiner = SoftMaskRefinement(model, cfg)
            pred = refiner.refine(X, Y, seq_len=MAX_SEQ_LEN)

    scores = []
    for i in range(n):
        pred_prefix = tokenizer.decode(pred[i].cpu().tolist())
        gt_prefix = test_data[i]["prefix_notation"]
        m = composite_score(pred_prefix, gt_prefix)
        scores.append(m["composite_score"])
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global_start = time.time()

    # ------------------------------------------------------------------
    # 1. Build test set (same splits as other scripts)
    # ------------------------------------------------------------------
    tokenizer = PhysicsTokenizer()

    gen = PhysicsDatasetGenerator(seed=42)
    _ = gen.generate_dataset(4000, N_POINTS)   # train — skip
    _ = gen.generate_dataset(500, N_POINTS)    # val   — skip
    test_data = gen.generate_dataset(500, N_POINTS)
    test_data = test_data[:N_TEST]

    X, Y, T = build_test_tensors(test_data, tokenizer)

    # ------------------------------------------------------------------
    # 2. Load model template
    # ------------------------------------------------------------------
    model = PhysMDT(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        max_vars=MAX_VARS,
        n_points=N_POINTS,
        lora_rank=0,
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"WARNING: model not found at {MODEL_PATH}, using random init")

    # ------------------------------------------------------------------
    # 3. Sweep K values x seeds
    # ------------------------------------------------------------------
    # Budget: distribute remaining time across K values
    elapsed_setup = time.time() - global_start
    remaining = TOTAL_BUDGET_SEC - elapsed_setup
    per_k_budget = max(remaining / len(K_VALUES), 10)

    results = {}  # K -> list of composite scores across seeds

    for K in K_VALUES:
        k_label = str(K)
        results[k_label] = []
        print(f"\n--- K={K} (budget {per_k_budget:.0f}s) ---")

        # Check global budget
        if time.time() - global_start > TOTAL_BUDGET_SEC:
            print(f"  SKIPPED (global timeout)")
            continue

        for seed in SEEDS:
            if time.time() - global_start > TOTAL_BUDGET_SEC:
                print(f"  Seed {seed} SKIPPED (global timeout)")
                continue

            # Re-seed to get different internal randomness per seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Re-initialise model weights from checkpoint to ensure
            # identical starting point, then perturb internal state via
            # the random seed (affects any stochastic components).
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(
                    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
                )

            t0 = time.time()
            timed_out = False

            # Set per-K timeout via SIGALRM
            try:
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(int(per_k_budget))
                score = evaluate_with_k(model, X, Y, tokenizer, test_data, K)
                signal.alarm(0)  # cancel alarm
            except TimeoutError:
                print(f"  Seed {seed}: TIMEOUT after {time.time() - t0:.1f}s")
                timed_out = True
                signal.alarm(0)
            except Exception as exc:
                print(f"  Seed {seed}: ERROR — {exc}")
                timed_out = True
                signal.alarm(0)

            if not timed_out:
                results[k_label].append(score)
                print(f"  Seed {seed}: composite={score:.2f} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Aggregate results
    # ------------------------------------------------------------------
    summary = {}
    for k_label, scores in results.items():
        if scores:
            summary[k_label] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores,
                "n_seeds": len(scores),
            }
        else:
            summary[k_label] = {
                "mean": None,
                "std": None,
                "scores": [],
                "n_seeds": 0,
                "note": "skipped due to timeout",
            }

    output = {
        "experiment": "refinement_depth_study",
        "model_path": MODEL_PATH,
        "K_values": K_VALUES,
        "seeds": SEEDS,
        "n_test_samples": N_TEST,
        "config": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "max_seq_len": MAX_SEQ_LEN,
            "max_vars": MAX_VARS,
            "n_points": N_POINTS,
        },
        "refinement_config": {
            "cold_restart": False,
            "soft_masking": False,
            "candidate_tracking": True,
            "convergence_detection": True,
            "confidence_threshold": 0.9,
            "convergence_patience": 2,
        },
        "results": summary,
        "elapsed_seconds": time.time() - global_start,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"{'K':>5s}  {'Mean CS':>8s}  {'Std':>8s}  {'Seeds':>5s}")
    print("-" * 50)
    for k_label in [str(k) for k in K_VALUES]:
        s = summary.get(k_label, {})
        mean = s.get("mean")
        std = s.get("std")
        ns = s.get("n_seeds", 0)
        if mean is not None:
            print(f"{k_label:>5s}  {mean:>8.2f}  {std:>8.2f}  {ns:>5d}")
        else:
            print(f"{k_label:>5s}  {'N/A':>8s}  {'N/A':>8s}  {ns:>5d}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 6. Generate figure
    # ------------------------------------------------------------------
    _generate_figure(summary)

    print(f"\nTotal elapsed: {time.time() - global_start:.1f}s")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _generate_figure(summary):
    """Create a publication-quality figure: composite score vs K with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Colorblind-friendly palette (IBM Design Library / Wong 2011)
    CB_BLUE = "#0072B2"
    CB_ORANGE = "#E69F00"
    CB_FILL = "#56B4E9"

    # Collect plottable data (skip K values with no results)
    k_plot, mean_plot, std_plot = [], [], []
    for K in K_VALUES:
        entry = summary.get(str(K), {})
        if entry.get("mean") is not None:
            k_plot.append(K)
            mean_plot.append(entry["mean"])
            std_plot.append(entry["std"])

    if not k_plot:
        print("WARNING: no data to plot, skipping figure generation")
        return

    k_plot = np.array(k_plot)
    mean_plot = np.array(mean_plot)
    std_plot = np.array(std_plot)

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    # Error band (shaded region for +/- 1 std)
    ax.fill_between(
        k_plot,
        mean_plot - std_plot,
        mean_plot + std_plot,
        alpha=0.25,
        color=CB_FILL,
        label=r"$\pm 1$ std",
    )

    # Main line with markers
    ax.plot(
        k_plot,
        mean_plot,
        marker="o",
        markersize=7,
        linewidth=2,
        color=CB_BLUE,
        label="Mean composite score",
        zorder=5,
    )

    # Error bars (caps)
    ax.errorbar(
        k_plot,
        mean_plot,
        yerr=std_plot,
        fmt="none",
        ecolor=CB_BLUE,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
    )

    # Highlight the baseline (K=0) with a horizontal dashed line if present
    if 0 in k_plot:
        baseline_idx = list(k_plot).index(0)
        ax.axhline(
            mean_plot[baseline_idx],
            color=CB_ORANGE,
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            label=f"Baseline (K=0): {mean_plot[baseline_idx]:.1f}",
        )

    # Identify and annotate optimal K
    best_idx = int(np.argmax(mean_plot))
    ax.annotate(
        f"Best: K={k_plot[best_idx]} ({mean_plot[best_idx]:.2f})",
        xy=(k_plot[best_idx], mean_plot[best_idx]),
        xytext=(30, 25),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )

    # Set y-axis limits with some padding around the data range
    y_min = min(mean_plot - std_plot)
    y_max = max(mean_plot + std_plot)
    y_range = y_max - y_min if y_max > y_min else 0.1
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.25 * y_range)

    # Axes labels and formatting
    ax.set_xlabel("Refinement Steps (K)", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("Refinement Depth Study: Composite Score vs K", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.set_xticks(k_plot)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "refinement_depth.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
