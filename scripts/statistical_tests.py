#!/usr/bin/env python3
"""
Statistical significance tests and error analysis for PhysMDT vs AR baseline.

Implements item 023 of the research rubric:
1. Run 5 independent training runs with different seeds (42-46)
2. Record all 5 metrics + composite score per run
3. Compute mean +/- std for both models
4. Run paired t-test and Wilcoxon signed-rank test
5. Save results to results/significance/stats.json
6. Write results/significance/error_analysis.md

Usage:
    python scripts/statistical_tests.py
"""

import os
import sys
import json
import time
import math
import random
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_ROOT)

from src.tokenizer import (
    encode, decode, VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX, MASK_IDX, MAX_SEQ_LEN,
)
from src.baseline_ar import build_baseline_model
from src.phys_mdt import build_phys_mdt
from src.metrics import evaluate_batch
from data.generator import generate_dataset, split_dataset

# ---------------------------------------------------------------------------
# Optional scipy
# ---------------------------------------------------------------------------
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]
N_SAMPLES = 200          # small dataset for speed
N_EPOCHS = 3             # max training epochs
TIMEOUT_PER_RUN = 30     # seconds per training run
N_EVAL = 30              # test samples per run
BATCH_SIZE = 32
LR = 5e-4

# Small model hyper-parameters
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 2

DEVICE = torch.device("cpu")

METRIC_NAMES = [
    "exact_match",
    "symbolic_equivalence",
    "numerical_r2",
    "tree_edit_distance",
    "complexity_penalty",
    "composite",
]

OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "significance")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class EquationDataset(Dataset):
    """Lightweight wrapper identical to the one in run_experiments.py."""

    def __init__(self, samples, max_vars=6, n_obs=20, max_seq_len=MAX_SEQ_LEN):
        self.data = []
        for s in samples:
            if s.get("token_ids") is None:
                continue
            obs = np.zeros((n_obs, max_vars + 1), dtype=np.float32)
            if s.get("numerical_data") and s["numerical_data"]:
                nd = s["numerical_data"]
                for i in range(min(len(nd["x"]), n_obs)):
                    vals = list(nd["x"][i].values())
                    for j in range(min(len(vals), max_vars)):
                        obs[i, j] = vals[j]
                    obs[i, max_vars] = nd["y"][i]
            self.data.append({
                "obs": torch.tensor(obs),
                "tokens": torch.tensor(s["token_ids"][:max_seq_len], dtype=torch.long),
                "infix": s["infix"],
                "family": s["family"],
                "difficulty": s["difficulty"],
                "name": s["name"],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["obs"], d["tokens"]


# ═══════════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int):
    """Deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_ar(train_ds, val_ds, seed: int):
    """Train a small AR baseline model and return it."""
    _set_seed(seed)
    model = build_baseline_model(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_enc_layers=N_LAYERS,
        n_dec_layers=N_LAYERS,
        d_ff=D_MODEL * 4,
    ).to(DEVICE)

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    start = time.time()
    for epoch in range(N_EPOCHS):
        if time.time() - start > TIMEOUT_PER_RUN:
            break
        model.train()
        for obs, tgt in loader:
            if time.time() - start > TIMEOUT_PER_RUN:
                break
            obs, tgt = obs.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(obs, tgt_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def train_mdt(train_ds, val_ds, seed: int):
    """Train a small PhysMDT model and return it."""
    _set_seed(seed)
    model = build_phys_mdt(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
    ).to(DEVICE)

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    start = time.time()
    for epoch in range(N_EPOCHS):
        if time.time() - start > TIMEOUT_PER_RUN:
            break
        model.train()
        for obs, tgt in loader:
            if time.time() - start > TIMEOUT_PER_RUN:
                break
            obs, tgt = obs.to(DEVICE), tgt.to(DEVICE)
            loss, _ = model.compute_loss(obs, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_samples, model_type="ar"):
    """Evaluate a model on *test_samples* and return per-sample + aggregate metrics."""
    model.eval()
    predictions = []
    ground_truths = []
    families = []
    difficulties = []
    names = []

    for sample in test_samples[:N_EVAL]:
        if sample.get("token_ids") is None:
            continue

        obs = torch.zeros(1, 20, 7)
        if sample.get("numerical_data") and sample["numerical_data"]:
            nd = sample["numerical_data"]
            for j in range(min(len(nd["x"]), 20)):
                vals = list(nd["x"][j].values())
                for k in range(min(len(vals), 6)):
                    obs[0, j, k] = vals[k]
                obs[0, j, 6] = nd["y"][j]
        obs = obs.to(DEVICE)

        with torch.no_grad():
            if model_type == "ar":
                gen = model.generate(obs, max_len=MAX_SEQ_LEN, temperature=1.0)
            else:
                gen = model.generate(obs, max_len=MAX_SEQ_LEN, n_steps=15)

        pred_str = decode(gen[0].cpu().tolist())
        true_str = sample["infix"]
        predictions.append(pred_str)
        ground_truths.append(true_str)
        families.append(sample["family"])
        difficulties.append(sample["difficulty"])
        names.append(sample.get("name", ""))

    overall = evaluate_batch(predictions, ground_truths)
    return {
        "overall": overall,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "families": families,
        "difficulties": difficulties,
        "names": names,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════

def _manual_paired_ttest(a, b):
    """Compute a paired t-test without scipy.  Returns (t_stat, p_value)."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d < 1e-15:
        return float("inf") if mean_d > 0 else float("-inf"), 0.0
    se = math.sqrt(var_d / n)
    t_stat = mean_d / se
    # Two-tailed p-value approximation using the normal distribution
    # (reasonable for descriptive purposes when scipy is absent)
    z = abs(t_stat)
    # Abramowitz & Stegun approximation of the normal CDF tail
    p_approx = math.erfc(z / math.sqrt(2))
    return t_stat, p_approx


def paired_ttest(a, b):
    """Paired t-test.  Uses scipy if available, else manual fallback."""
    if HAS_SCIPY:
        res = scipy_stats.ttest_rel(a, b)
        return float(res.statistic), float(res.pvalue)
    return _manual_paired_ttest(a, b)


def wilcoxon_test(a, b):
    """Wilcoxon signed-rank test.  Returns (statistic, p_value)."""
    if HAS_SCIPY:
        try:
            res = scipy_stats.wilcoxon(a, b)
            return float(res.statistic), float(res.pvalue)
        except ValueError:
            # All differences are zero or too few samples
            return float("nan"), float("nan")
    return float("nan"), float("nan")


# ═══════════════════════════════════════════════════════════════════════════
# Error / failure classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_failure(pred: str, truth: str):
    """Heuristically classify a failure into a category."""
    if not pred or pred.strip() == "":
        return "empty_prediction"
    # Very short prediction compared to ground truth
    if len(pred) < len(truth) * 0.3:
        return "truncation"

    # Check for numeric-only differences (coefficient errors)
    pred_tokens = set(pred.replace(" ", ""))
    truth_tokens = set(truth.replace(" ", ""))

    # Remove digits to see if structure is similar
    import re
    pred_no_num = re.sub(r"[0-9\.\-]+", "#", pred)
    truth_no_num = re.sub(r"[0-9\.\-]+", "#", truth)
    if pred_no_num == truth_no_num:
        return "coefficient_error"

    # Check for missing terms (pred is substring of truth or much shorter)
    if len(pred) < len(truth) * 0.6:
        return "missing_terms"

    # Check for wrong operators
    ops_pred = set(re.findall(r"[+\-\*/\^]", pred))
    ops_truth = set(re.findall(r"[+\-\*/\^]", truth))
    if ops_pred != ops_truth:
        return "wrong_operator"

    # Check for variable confusion
    vars_pred = set(re.findall(r"[a-zA-Z_]\w*", pred))
    vars_truth = set(re.findall(r"[a-zA-Z_]\w*", truth))
    if vars_pred != vars_truth:
        return "variable_confusion"

    return "other"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Statistical Significance Tests  (item 023)")
    print("=" * 65)
    total_start = time.time()

    # ------------------------------------------------------------------
    # 1. Generate a small shared dataset
    # ------------------------------------------------------------------
    print(f"\nGenerating {N_SAMPLES} samples (shared across runs) ...")
    all_samples = generate_dataset(N_SAMPLES, seed=42, include_numerical=True, n_points=10)
    train_samples, val_samples, test_samples = split_dataset(all_samples, seed=42)
    train_ds = EquationDataset(train_samples)
    val_ds = EquationDataset(val_samples)
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_samples)}")

    # ------------------------------------------------------------------
    # 2. Run 5 independent training + evaluation runs per model
    # ------------------------------------------------------------------
    ar_run_metrics = []     # list of dicts (one per seed)
    mdt_run_metrics = []

    # Collect detailed results from the last run for qualitative examples
    last_ar_eval = None
    last_mdt_eval = None

    for idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({idx+1}/{len(SEEDS)}) ---")

        # AR baseline
        print(f"  Training AR baseline (seed={seed}) ...")
        t0 = time.time()
        try:
            ar_model = train_ar(train_ds, val_ds, seed)
            print(f"    trained in {time.time()-t0:.1f}s")
            ar_eval = evaluate_model(ar_model, test_samples, model_type="ar")
            ar_run_metrics.append(ar_eval["overall"])
            last_ar_eval = ar_eval
            print(f"    composite={ar_eval['overall']['composite']:.4f}")
        except Exception:
            traceback.print_exc()
            ar_run_metrics.append({m: 0.0 for m in METRIC_NAMES})

        # PhysMDT
        print(f"  Training PhysMDT   (seed={seed}) ...")
        t0 = time.time()
        try:
            mdt_model = train_mdt(train_ds, val_ds, seed)
            print(f"    trained in {time.time()-t0:.1f}s")
            mdt_eval = evaluate_model(mdt_model, test_samples, model_type="mdt")
            mdt_run_metrics.append(mdt_eval["overall"])
            last_mdt_eval = mdt_eval
            print(f"    composite={mdt_eval['overall']['composite']:.4f}")
        except Exception:
            traceback.print_exc()
            mdt_run_metrics.append({m: 0.0 for m in METRIC_NAMES})

    # ------------------------------------------------------------------
    # 3. Aggregate statistics: mean +/- std per metric
    # ------------------------------------------------------------------
    def _aggregate(run_list):
        agg = {}
        for m in METRIC_NAMES:
            vals = [r[m] for r in run_list]
            agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        return agg

    ar_stats = _aggregate(ar_run_metrics)
    mdt_stats = _aggregate(mdt_run_metrics)

    # ------------------------------------------------------------------
    # 4. Paired statistical tests (PhysMDT vs AR) for every metric
    # ------------------------------------------------------------------
    test_results = {}
    for m in METRIC_NAMES:
        ar_vals = [r[m] for r in ar_run_metrics]
        mdt_vals = [r[m] for r in mdt_run_metrics]

        t_stat, t_pval = paired_ttest(mdt_vals, ar_vals)
        w_stat, w_pval = wilcoxon_test(mdt_vals, ar_vals)

        test_results[m] = {
            "ttest": {"statistic": t_stat, "p_value": t_pval},
            "wilcoxon": {"statistic": w_stat, "p_value": w_pval},
        }

    # ------------------------------------------------------------------
    # 5. Save results/significance/stats.json
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats_payload = {
        "seeds": SEEDS,
        "config": {
            "n_samples": N_SAMPLES,
            "n_epochs": N_EPOCHS,
            "timeout_per_run_s": TIMEOUT_PER_RUN,
            "n_eval": N_EVAL,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        },
        "per_run": {
            "ar_baseline": [
                {m: round(r[m], 6) for m in METRIC_NAMES}
                for r in ar_run_metrics
            ],
            "phys_mdt": [
                {m: round(r[m], 6) for m in METRIC_NAMES}
                for r in mdt_run_metrics
            ],
        },
        "summary": {
            "ar_baseline": {
                m: {"mean": round(ar_stats[m]["mean"], 6),
                    "std": round(ar_stats[m]["std"], 6)}
                for m in METRIC_NAMES
            },
            "phys_mdt": {
                m: {"mean": round(mdt_stats[m]["mean"], 6),
                    "std": round(mdt_stats[m]["std"], 6)}
                for m in METRIC_NAMES
            },
        },
        "tests": {
            m: {
                "paired_ttest": {
                    "statistic": round(test_results[m]["ttest"]["statistic"], 6)
                        if not math.isnan(test_results[m]["ttest"]["statistic"]) else None,
                    "p_value": round(test_results[m]["ttest"]["p_value"], 6)
                        if not math.isnan(test_results[m]["ttest"]["p_value"]) else None,
                },
                "wilcoxon_signed_rank": {
                    "statistic": round(test_results[m]["wilcoxon"]["statistic"], 6)
                        if not math.isnan(test_results[m]["wilcoxon"]["statistic"]) else None,
                    "p_value": round(test_results[m]["wilcoxon"]["p_value"], 6)
                        if not math.isnan(test_results[m]["wilcoxon"]["p_value"]) else None,
                },
            }
            for m in METRIC_NAMES
        },
        "scipy_available": HAS_SCIPY,
        "total_runtime_s": round(time.time() - total_start, 1),
    }

    stats_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_payload, f, indent=2)
    print(f"\nSaved {stats_path}")

    # ------------------------------------------------------------------
    # 6. Write results/significance/error_analysis.md
    # ------------------------------------------------------------------

    # Collect qualitative examples from last seed's evaluation
    correct_examples = []
    failure_examples = []
    failure_category_counts = {}

    # Use whichever eval produced predictions (prefer mdt, fall back to ar)
    eval_to_analyse = last_mdt_eval if last_mdt_eval else last_ar_eval
    if eval_to_analyse:
        preds = eval_to_analyse["predictions"]
        truths = eval_to_analyse["ground_truths"]
        fams = eval_to_analyse["families"]
        diffs = eval_to_analyse["difficulties"]
        names = eval_to_analyse["names"]

        for i in range(len(preds)):
            from src.metrics import composite_score as cs_fn
            try:
                score = cs_fn(preds[i], truths[i])
            except Exception:
                score = 0.0
            entry = {
                "prediction": preds[i],
                "ground_truth": truths[i],
                "family": fams[i],
                "difficulty": diffs[i],
                "name": names[i],
                "composite": round(score, 4),
            }
            if score >= 0.5:
                correct_examples.append(entry)
            else:
                cat = classify_failure(preds[i], truths[i])
                entry["failure_category"] = cat
                failure_examples.append(entry)
                failure_category_counts[cat] = failure_category_counts.get(cat, 0) + 1

    # Sort so we get best correct and most interesting failures
    correct_examples.sort(key=lambda e: e["composite"], reverse=True)
    failure_examples.sort(key=lambda e: e["composite"], reverse=True)

    top_correct = correct_examples[:5]
    top_failures = failure_examples[:5]

    md_lines = []
    md_lines.append("# Statistical Significance Tests & Error Analysis")
    md_lines.append("")
    md_lines.append("Item 023 of the research rubric.")
    md_lines.append("")

    # --- Comparison table ---
    md_lines.append("## 1. Statistical Comparison: PhysMDT vs AR Baseline")
    md_lines.append("")
    md_lines.append(f"- **Seeds**: {SEEDS}")
    md_lines.append(f"- **Dataset**: {N_SAMPLES} samples, {N_EVAL} test samples per run")
    md_lines.append(f"- **Model size**: d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
    md_lines.append(f"- **Training**: {N_EPOCHS} epochs, {TIMEOUT_PER_RUN}s timeout, batch_size={BATCH_SIZE}")
    md_lines.append(f"- **scipy available**: {HAS_SCIPY}")
    md_lines.append("")
    md_lines.append("| Metric | AR Baseline (mean +/- std) | PhysMDT (mean +/- std) | t-test p | Wilcoxon p |")
    md_lines.append("|--------|---------------------------|------------------------|----------|------------|")
    for m in METRIC_NAMES:
        ar_m = ar_stats[m]
        mdt_m = mdt_stats[m]
        tp = test_results[m]["ttest"]["p_value"]
        wp = test_results[m]["wilcoxon"]["p_value"]
        tp_str = f"{tp:.4f}" if not math.isnan(tp) else "N/A"
        wp_str = f"{wp:.4f}" if not math.isnan(wp) else "N/A"
        md_lines.append(
            f"| {m} | {ar_m['mean']:.4f} +/- {ar_m['std']:.4f} "
            f"| {mdt_m['mean']:.4f} +/- {mdt_m['std']:.4f} "
            f"| {tp_str} | {wp_str} |"
        )
    md_lines.append("")

    # --- Per-run breakdown ---
    md_lines.append("### Per-Run Composite Scores")
    md_lines.append("")
    md_lines.append("| Seed | AR Baseline | PhysMDT |")
    md_lines.append("|------|-------------|---------|")
    for i, seed in enumerate(SEEDS):
        ar_c = ar_run_metrics[i].get("composite", 0)
        mdt_c = mdt_run_metrics[i].get("composite", 0)
        md_lines.append(f"| {seed} | {ar_c:.4f} | {mdt_c:.4f} |")
    md_lines.append("")

    # --- Failure categories ---
    md_lines.append("## 2. Failure Categories")
    md_lines.append("")
    md_lines.append("Failure categories are determined heuristically by comparing predicted")
    md_lines.append("and ground-truth equation strings. A sample is classified as a failure")
    md_lines.append("when its composite score is below 0.5.")
    md_lines.append("")
    md_lines.append("| Category | Count | Description |")
    md_lines.append("|----------|-------|-------------|")
    category_descriptions = {
        "coefficient_error": "Structural skeleton matches but numeric coefficients differ.",
        "missing_terms": "Prediction is substantially shorter; likely dropped terms.",
        "wrong_operator": "Different arithmetic or function operators used.",
        "variable_confusion": "Correct structure but different variable names.",
        "truncation": "Prediction is very short (< 30% of ground truth length).",
        "empty_prediction": "Model produced an empty or unparseable output.",
        "other": "Failure does not fit the above categories.",
    }
    for cat in sorted(failure_category_counts.keys(), key=lambda c: -failure_category_counts[c]):
        cnt = failure_category_counts[cat]
        desc = category_descriptions.get(cat, "")
        md_lines.append(f"| {cat} | {cnt} | {desc} |")
    if not failure_category_counts:
        md_lines.append("| (none detected) | 0 | All samples scored >= 0.5 |")
    md_lines.append("")

    # --- Correct derivations ---
    md_lines.append("## 3. Examples of Correct Derivations (top 5)")
    md_lines.append("")
    for i, ex in enumerate(top_correct, 1):
        md_lines.append(f"### Correct Example {i}")
        md_lines.append(f"- **Equation name**: {ex['name']}")
        md_lines.append(f"- **Family / difficulty**: {ex['family']} / {ex['difficulty']}")
        md_lines.append(f"- **Ground truth**: `{ex['ground_truth']}`")
        md_lines.append(f"- **Prediction**:   `{ex['prediction']}`")
        md_lines.append(f"- **Composite score**: {ex['composite']}")
        md_lines.append("")
    if not top_correct:
        md_lines.append("No predictions achieved composite >= 0.5 in the sampled run.")
        md_lines.append("")

    # --- Instructive failures ---
    md_lines.append("## 4. Instructive Failures (top 5)")
    md_lines.append("")
    for i, ex in enumerate(top_failures, 1):
        md_lines.append(f"### Failure Example {i}")
        md_lines.append(f"- **Equation name**: {ex['name']}")
        md_lines.append(f"- **Family / difficulty**: {ex['family']} / {ex['difficulty']}")
        md_lines.append(f"- **Ground truth**: `{ex['ground_truth']}`")
        md_lines.append(f"- **Prediction**:   `{ex['prediction']}`")
        md_lines.append(f"- **Composite score**: {ex['composite']}")
        md_lines.append(f"- **Failure category**: {ex.get('failure_category', 'unknown')}")
        md_lines.append("")
    if not top_failures:
        md_lines.append("No failures detected (all composite >= 0.5).")
        md_lines.append("")

    # --- Summary ---
    md_lines.append("## 5. Summary")
    md_lines.append("")
    ar_comp_mean = ar_stats["composite"]["mean"]
    mdt_comp_mean = mdt_stats["composite"]["mean"]
    delta = mdt_comp_mean - ar_comp_mean
    md_lines.append(f"- AR baseline mean composite: **{ar_comp_mean:.4f}**")
    md_lines.append(f"- PhysMDT mean composite:     **{mdt_comp_mean:.4f}**")
    md_lines.append(f"- Difference (PhysMDT - AR):  **{delta:+.4f}**")
    tp_comp = test_results["composite"]["ttest"]["p_value"]
    tp_comp_str = f"{tp_comp:.4f}" if not math.isnan(tp_comp) else "N/A"
    md_lines.append(f"- Paired t-test p-value (composite): **{tp_comp_str}**")
    md_lines.append("")
    md_lines.append(f"Total runtime: {time.time()-total_start:.1f}s")
    md_lines.append("")

    md_path = os.path.join(OUTPUT_DIR, "error_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved {md_path}")

    # ------------------------------------------------------------------
    # Print summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    for m in METRIC_NAMES:
        ar_m = ar_stats[m]
        mdt_m = mdt_stats[m]
        print(f"  {m:25s}  AR {ar_m['mean']:.4f}+/-{ar_m['std']:.4f}"
              f"  MDT {mdt_m['mean']:.4f}+/-{mdt_m['std']:.4f}")
    print(f"\n  Total time: {time.time()-total_start:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
