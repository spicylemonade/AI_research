#!/usr/bin/env python3
"""Benchmark evaluation of PhysMDT on AI Feynman, Nguyen, and Strogatz-style datasets.

Evaluates the model on three benchmark-style subsets constructed from
PhysicsDatasetGenerator, grouped by physics family to simulate standard
symbolic regression benchmarks:

  - AI Feynman-style : 50+ equations from energy, gravitation, dynamics families
  - Nguyen-style     : 12 simple single-variable equations from kinematics
  - Strogatz-style   : oscillations + rotational (ODE-related families)

Per-family and per-benchmark metrics are computed using src.metrics.composite_score.
Results are compared against AR baseline and classical SR baseline.
"""

import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import PhysicsDatasetGenerator
from src.tokenizer import PhysicsTokenizer
from src.phys_mdt import PhysMDT
from src.refinement import SoftMaskRefinement, RefinementConfig
from src.metrics import composite_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_POINTS = 10
MAX_VARS = 5
MAX_SEQ_LEN = 48
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
D_FF = 256
MAX_EVAL = 100  # hard cap on total evaluations

ALL_FAMILIES = [
    "kinematics", "dynamics", "energy",
    "rotational", "gravitation", "oscillations", "fluid_statics",
]

# Benchmark definitions: which families contribute to each benchmark
BENCHMARK_DEFS = {
    "ai_feynman": {
        "families": ["energy", "gravitation", "dynamics"],
        "min_count": 50,
        "description": "AI Feynman-style: energy, gravitation, dynamics equations",
    },
    "nguyen": {
        "families": ["kinematics"],
        "target_count": 12,
        "difficulty": "simple",
        "description": "Nguyen-style: 12 simple single-variable kinematics equations",
    },
    "strogatz": {
        "families": ["oscillations", "rotational"],
        "description": "Strogatz-style: ODE-related (oscillations + rotational)",
    },
}

# Literature SOTA for comparison context
LITERATURE_CONTEXT = {
    "ai_feynman": {
        "QDSR (Bruneton 2025)": {"exact_recovery_pct": 91.6, "dataset": "AI Feynman noiseless"},
        "AI Feynman 2.0 (Udrescu 2020)": {"exact_recovery_pct": 72.0, "dataset": "AI Feynman noiseless"},
        "TPSR (Shojaee 2023)": {"exact_recovery_pct": 45.0, "dataset": "AI Feynman noiseless"},
        "E2E Transformer (Kamienny 2022)": {"exact_recovery_pct": 38.0, "dataset": "AI Feynman noiseless"},
        "NeSymReS (Biggio 2021)": {"exact_recovery_pct": 30.0, "dataset": "AI Feynman noiseless"},
        "DiffuSR (2025)": {"exact_recovery_pct": 32.0, "dataset": "AI Feynman noiseless"},
        "PySR (Cranmer 2023)": {"exact_recovery_pct": 35.0, "dataset": "AI Feynman noiseless"},
    },
    "nguyen": {
        "PySR (Cranmer 2023)": {"exact_recovery_pct": 100.0, "equations_solved": "12/12"},
        "QDSR (Bruneton 2025)": {"exact_recovery_pct": 100.0, "equations_solved": "12/12"},
        "TPSR (Shojaee 2023)": {"exact_recovery_pct": 91.7, "equations_solved": "11/12"},
        "E2E Transformer (Kamienny 2022)": {"exact_recovery_pct": 83.3, "equations_solved": "10/12"},
        "NeSymReS (Biggio 2021)": {"exact_recovery_pct": 75.0, "equations_solved": "9/12"},
    },
    "strogatz": {
        "AI Feynman (Udrescu 2020)": {"solution_rate_pct": 40.0, "avg_r2": 0.92},
        "GP-GOMEA (La Cava 2021)": {"solution_rate_pct": 35.0, "avg_r2": 0.94},
        "PySR (Cranmer 2023)": {"solution_rate_pct": 35.0, "avg_r2": 0.93},
        "DSR (Petersen 2021)": {"solution_rate_pct": 30.0, "avg_r2": 0.90},
    },
}


# ---------------------------------------------------------------------------
# Tensor building (same pattern as eval_quick.py)
# ---------------------------------------------------------------------------

def build_test_tensors(test_data, tokenizer, max_seq_len=MAX_SEQ_LEN,
                       n_points=N_POINTS, max_vars=MAX_VARS):
    """Convert list-of-dicts test data into padded tensors for the model."""
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(model, X, Y, seq_len, tokenizer, test_data,
                   use_refinement=True, refine_steps=4):
    """Run model inference and compute per-sample metrics."""
    model.eval()
    n = X.shape[0]
    if n == 0:
        return []

    with torch.no_grad():
        if use_refinement:
            cfg = RefinementConfig(
                total_steps=refine_steps,
                cold_restart=False,
                convergence_detection=True,
                confidence_threshold=0.9,
                convergence_patience=2,
                soft_masking=False,
                candidate_tracking=True,
            )
            refiner = SoftMaskRefinement(model, cfg)
            pred = refiner.refine(X, Y, seq_len=seq_len)
        else:
            pred = model.generate_single_pass(X, Y, seq_len=seq_len)

    results = []
    for i in range(n):
        pred_prefix = tokenizer.decode(pred[i].cpu().tolist())
        gt_prefix = test_data[i]["prefix_notation"]
        m = composite_score(pred_prefix, gt_prefix)
        m["template_name"] = test_data[i]["template_name"]
        m["family"] = test_data[i]["family"]
        m["difficulty"] = test_data[i]["difficulty"]
        m["pred_prefix"] = pred_prefix
        m["gt_prefix"] = gt_prefix
        results.append(m)
    return results


def agg(results):
    """Aggregate a list of per-sample metric dicts into mean metrics."""
    metric_keys = [
        "exact_match", "symbolic_equivalence", "numerical_r2",
        "tree_edit_distance", "complexity_penalty", "composite_score",
    ]
    if not results:
        return {k: 0.0 for k in metric_keys}
    return {k: float(np.mean([r[k] for r in results])) for k in metric_keys}


# ---------------------------------------------------------------------------
# Dataset construction helpers
# ---------------------------------------------------------------------------

def generate_family_pool(gen, n_per_family=30, n_points=N_POINTS):
    """Generate a large pool of equations organised by family.

    Returns a dict mapping family -> list of sample dicts.
    """
    pool = {f: [] for f in ALL_FAMILIES}
    # Generate enough samples per family
    for family in ALL_FAMILIES:
        samples = gen.generate_dataset(n_per_family, n_points, family=family)
        pool[family].extend(samples)
    return pool


def build_benchmark_subset(pool, families, max_count=None, difficulty=None):
    """Select samples from the pool for a given benchmark definition."""
    subset = []
    for fam in families:
        for sample in pool.get(fam, []):
            if difficulty is not None and sample["difficulty"] != difficulty:
                continue
            subset.append(sample)
    if max_count is not None:
        subset = subset[:max_count]
    return subset


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def load_baseline_metrics(path):
    """Load baseline metrics from a JSON file if it exists."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get("test_metrics", {})
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = PhysicsTokenizer()

    # ------------------------------------------------------------------
    # 1. Generate equation pool by family
    # ------------------------------------------------------------------
    print("=" * 65)
    print("PhysMDT Benchmark Evaluation")
    print("=" * 65)

    gen = PhysicsDatasetGenerator(seed=SEED)
    # Skip train and val splits to align RNG with eval_quick.py
    _ = gen.generate_dataset(4000, N_POINTS)
    _ = gen.generate_dataset(500, N_POINTS)

    # Generate per-family pool (enough for all benchmarks)
    # We generate more for the AI Feynman families to ensure >=50 equations
    pool = {f: [] for f in ALL_FAMILIES}
    for family in ALL_FAMILIES:
        need = 20 if family not in ["energy", "gravitation", "dynamics"] else 25
        samples = gen.generate_dataset(need, N_POINTS, family=family)
        pool[family] = samples

    print("\nEquation pool by family:")
    for fam in ALL_FAMILIES:
        print(f"  {fam:20s}: {len(pool[fam]):3d} equations")

    # ------------------------------------------------------------------
    # 2. Build benchmark subsets
    # ------------------------------------------------------------------

    # AI Feynman-style: energy + gravitation + dynamics (>=50 equations)
    ai_feynman_data = build_benchmark_subset(
        pool, ["energy", "gravitation", "dynamics"]
    )
    # Ensure at least 50
    assert len(ai_feynman_data) >= 50, (
        f"AI Feynman subset has only {len(ai_feynman_data)} equations, need >=50"
    )

    # Nguyen-style: 12 simple kinematics equations
    nguyen_data = build_benchmark_subset(
        pool, ["kinematics"], max_count=12, difficulty="simple"
    )
    # Pad with non-simple kinematics if not enough simple
    if len(nguyen_data) < 12:
        extra = build_benchmark_subset(pool, ["kinematics"], max_count=12)
        for s in extra:
            if len(nguyen_data) >= 12:
                break
            if s not in nguyen_data:
                nguyen_data.append(s)

    # Strogatz-style: oscillations + rotational
    strogatz_data = build_benchmark_subset(pool, ["oscillations", "rotational"])

    # Enforce total evaluation cap
    total_needed = len(ai_feynman_data) + len(nguyen_data) + len(strogatz_data)
    if total_needed > MAX_EVAL:
        # Scale down proportionally
        ratio = MAX_EVAL / total_needed
        ai_feynman_data = ai_feynman_data[:max(50, int(len(ai_feynman_data) * ratio))]
        nguyen_data = nguyen_data[:max(12, int(len(nguyen_data) * ratio))]
        remaining = MAX_EVAL - len(ai_feynman_data) - len(nguyen_data)
        strogatz_data = strogatz_data[:max(1, remaining)]

    print(f"\nBenchmark subsets:")
    print(f"  AI Feynman-style : {len(ai_feynman_data):3d} equations "
          f"(energy, gravitation, dynamics)")
    print(f"  Nguyen-style     : {len(nguyen_data):3d} equations "
          f"(simple kinematics)")
    print(f"  Strogatz-style   : {len(strogatz_data):3d} equations "
          f"(oscillations, rotational)")
    print(f"  Total            : {len(ai_feynman_data) + len(nguyen_data) + len(strogatz_data):3d} "
          f"equations (cap={MAX_EVAL})")

    # ------------------------------------------------------------------
    # 3. Load model
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

    model_path = "results/phys_mdt/model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        print(f"\nLoaded model from {model_path}")
    else:
        print(f"\nWARNING: No saved model at {model_path}, using random weights")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 4. Evaluate per family (across all data)
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Per-Family Evaluation")
    print("-" * 65)

    family_results = {}
    for family in ALL_FAMILIES:
        fam_data = pool[family]
        if not fam_data:
            family_results[family] = {"metrics": agg([]), "n_evaluated": 0}
            continue
        X, Y, T = build_test_tensors(fam_data, tokenizer)
        results = evaluate_batch(
            model, X, Y, MAX_SEQ_LEN, tokenizer, fam_data,
            use_refinement=True, refine_steps=4,
        )
        fam_metrics = agg(results)
        family_results[family] = {
            "metrics": fam_metrics,
            "n_evaluated": len(results),
            "per_sample": results[:5],  # keep first 5 for reference
        }
        print(f"  {family:20s}  CS={fam_metrics['composite_score']:6.2f}  "
              f"EM={fam_metrics['exact_match']:.3f}  "
              f"SE={fam_metrics['symbolic_equivalence']:.3f}  "
              f"R2={fam_metrics['numerical_r2']:.3f}  "
              f"(n={len(results)})")

    # ------------------------------------------------------------------
    # 5. Evaluate each benchmark subset
    # ------------------------------------------------------------------
    benchmark_results = {}

    for bench_name, bench_data in [
        ("ai_feynman", ai_feynman_data),
        ("nguyen", nguyen_data),
        ("strogatz", strogatz_data),
    ]:
        print(f"\n{'-' * 65}")
        print(f"Benchmark: {bench_name} ({len(bench_data)} equations)")
        print(f"{'-' * 65}")

        t0 = time.time()
        X, Y, T = build_test_tensors(bench_data, tokenizer)
        results = evaluate_batch(
            model, X, Y, MAX_SEQ_LEN, tokenizer, bench_data,
            use_refinement=True, refine_steps=4,
        )
        elapsed = time.time() - t0
        bench_metrics = agg(results)

        # Per-difficulty breakdown
        diff_buckets = {}
        for r in results:
            d = r["difficulty"]
            diff_buckets.setdefault(d, []).append(r)
        per_difficulty = {d: agg(rs) for d, rs in diff_buckets.items()}

        # Per-family breakdown within this benchmark
        fam_buckets = {}
        for r in results:
            f = r["family"]
            fam_buckets.setdefault(f, []).append(r)
        per_family_in_bench = {f: agg(rs) for f, rs in fam_buckets.items()}

        benchmark_results[bench_name] = {
            "metrics": bench_metrics,
            "n_evaluated": len(results),
            "elapsed_sec": round(elapsed, 2),
            "per_difficulty": per_difficulty,
            "per_family": per_family_in_bench,
            "per_sample_results": results[:10],
            "description": BENCHMARK_DEFS[bench_name]["description"],
        }

        print(f"  Composite Score : {bench_metrics['composite_score']:.2f}")
        print(f"  Exact Match     : {bench_metrics['exact_match']:.3f} "
              f"({bench_metrics['exact_match'] * 100:.1f}%)")
        print(f"  Symbolic Equiv  : {bench_metrics['symbolic_equivalence']:.3f}")
        print(f"  Numerical R2    : {bench_metrics['numerical_r2']:.3f}")
        print(f"  Tree Edit Dist  : {bench_metrics['tree_edit_distance']:.3f}")
        print(f"  Complexity Pen  : {bench_metrics['complexity_penalty']:.3f}")
        print(f"  Time            : {elapsed:.1f}s")

        if per_difficulty:
            print("  Per-difficulty:")
            for d in ["simple", "medium", "complex"]:
                if d in per_difficulty:
                    dm = per_difficulty[d]
                    n_d = len(diff_buckets[d])
                    print(f"    {d:10s}: CS={dm['composite_score']:6.2f}  "
                          f"EM={dm['exact_match']:.3f}  n={n_d}")

    # ------------------------------------------------------------------
    # 6. Load baselines for comparison
    # ------------------------------------------------------------------
    ar_metrics = load_baseline_metrics("results/baseline_ar/metrics.json")
    sr_metrics = load_baseline_metrics("results/sr_baseline/metrics.json")

    # ------------------------------------------------------------------
    # 7. Save per-benchmark results
    # ------------------------------------------------------------------
    for bench_name in ["ai_feynman", "nguyen", "strogatz"]:
        out_dir = f"results/{bench_name}"
        os.makedirs(out_dir, exist_ok=True)
        bench_output = {
            "model": "PhysMDT",
            "benchmark": bench_name,
            "config": {
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "max_seq_len": MAX_SEQ_LEN,
                "max_vars": MAX_VARS,
                "n_points": N_POINTS,
                "total_params": total_params,
                "seed": SEED,
                "refinement_steps": 4,
            },
            "description": BENCHMARK_DEFS[bench_name]["description"],
            "test_metrics": benchmark_results[bench_name]["metrics"],
            "n_evaluated": benchmark_results[bench_name]["n_evaluated"],
            "elapsed_sec": benchmark_results[bench_name]["elapsed_sec"],
            "per_difficulty": benchmark_results[bench_name]["per_difficulty"],
            "per_family": benchmark_results[bench_name].get("per_family", {}),
            "per_sample_results": benchmark_results[bench_name]["per_sample_results"],
            "literature_context": LITERATURE_CONTEXT.get(bench_name, {}),
            "baselines": {
                "ar_baseline": ar_metrics,
                "sr_baseline": sr_metrics,
            },
        }
        out_path = os.path.join(out_dir, "metrics.json")
        with open(out_path, "w") as f:
            json.dump(bench_output, f, indent=2)
        print(f"\nSaved {bench_name} results -> {out_path}")

    # ------------------------------------------------------------------
    # 8. Build cross-benchmark comparison
    # ------------------------------------------------------------------
    comparison = {
        "model": "PhysMDT",
        "config": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "total_params": total_params,
            "seed": SEED,
        },
        "benchmarks": {},
        "per_family": {},
        "baselines": {
            "ar_baseline": ar_metrics,
            "sr_baseline": sr_metrics,
        },
        "literature_context": LITERATURE_CONTEXT,
        "performance_targets": {
            "ai_feynman_exact_match_target": ">=15% (comparable to NeSymReS on harder subset)",
            "nguyen_exact_match_target": ">=25% (3/12, competitive with NeSymReS)",
            "composite_vs_ar_baseline": ">=2x AR baseline composite score",
        },
    }

    for bench_name in ["ai_feynman", "nguyen", "strogatz"]:
        bm = benchmark_results[bench_name]["metrics"]
        comparison["benchmarks"][bench_name] = {
            "metrics": bm,
            "n_evaluated": benchmark_results[bench_name]["n_evaluated"],
            "description": BENCHMARK_DEFS[bench_name]["description"],
        }

    for family in ALL_FAMILIES:
        comparison["per_family"][family] = family_results[family]["metrics"]

    # Compute target achievement
    ar_cs = ar_metrics.get("composite_score", 0)
    targets_met = {}

    ai_em = benchmark_results["ai_feynman"]["metrics"]["exact_match"]
    targets_met["ai_feynman_em_ge_15pct"] = bool(ai_em >= 0.15)
    targets_met["ai_feynman_em_actual"] = f"{ai_em * 100:.1f}%"

    nguyen_em = benchmark_results["nguyen"]["metrics"]["exact_match"]
    targets_met["nguyen_em_ge_25pct"] = bool(nguyen_em >= 0.25)
    targets_met["nguyen_em_actual"] = f"{nguyen_em * 100:.1f}%"

    # Average composite across benchmarks
    avg_cs = np.mean([
        benchmark_results[b]["metrics"]["composite_score"]
        for b in ["ai_feynman", "nguyen", "strogatz"]
    ])
    targets_met["composite_ge_2x_ar"] = bool(avg_cs >= 2.0 * ar_cs) if ar_cs > 0 else False
    targets_met["avg_composite"] = round(float(avg_cs), 2)
    targets_met["ar_baseline_composite"] = round(float(ar_cs), 2)

    comparison["target_achievement"] = targets_met

    os.makedirs("results", exist_ok=True)
    comp_path = "results/benchmark_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved benchmark comparison -> {comp_path}")

    # ------------------------------------------------------------------
    # 9. Print summary table
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 65)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Benchmark':<18s} {'N':>4s} {'Composite':>10s} {'EM':>8s} "
          f"{'SE':>8s} {'R2':>8s}")
    print("-" * 65)
    for bench_name in ["ai_feynman", "nguyen", "strogatz"]:
        bm = benchmark_results[bench_name]["metrics"]
        n = benchmark_results[bench_name]["n_evaluated"]
        print(f"{bench_name:<18s} {n:>4d} {bm['composite_score']:>10.2f} "
              f"{bm['exact_match']:>8.3f} "
              f"{bm['symbolic_equivalence']:>8.3f} "
              f"{bm['numerical_r2']:>8.3f}")

    print(f"\n{'--- Baselines ---':^65s}")
    if ar_metrics:
        print(f"{'AR Baseline':<18s} {'':>4s} "
              f"{ar_metrics.get('composite_score', 0):>10.2f} "
              f"{ar_metrics.get('exact_match', 0):>8.3f} "
              f"{ar_metrics.get('symbolic_equivalence', 0):>8.3f} "
              f"{ar_metrics.get('numerical_r2', 0):>8.3f}")
    if sr_metrics:
        print(f"{'SR Baseline':<18s} {'':>4s} "
              f"{sr_metrics.get('composite_score', 0):>10.2f} "
              f"{sr_metrics.get('exact_match', 0):>8.3f} "
              f"{sr_metrics.get('symbolic_equivalence', 0):>8.3f} "
              f"{sr_metrics.get('numerical_r2', 0):>8.3f}")

    print(f"\n{'--- Per Family ---':^65s}")
    print(f"{'Family':<18s} {'N':>4s} {'Composite':>10s} {'EM':>8s} "
          f"{'SE':>8s} {'R2':>8s}")
    print("-" * 65)
    for fam in ALL_FAMILIES:
        fm = family_results[fam]["metrics"]
        n = family_results[fam]["n_evaluated"]
        print(f"{fam:<18s} {n:>4d} {fm['composite_score']:>10.2f} "
              f"{fm['exact_match']:>8.3f} "
              f"{fm['symbolic_equivalence']:>8.3f} "
              f"{fm['numerical_r2']:>8.3f}")

    print(f"\n{'--- Literature Context ---':^65s}")
    print("  AI Feynman SOTA: QDSR 91.6%, TPSR 45%, E2E 38%, NeSymReS 30%")
    print("  Nguyen SOTA:     PySR 100%, TPSR 92%, E2E 83%, NeSymReS 75%")
    print("  Strogatz SOTA:   AI Feynman 40%, GP-GOMEA 35%, PySR 35%")

    print(f"\n{'--- Target Achievement ---':^65s}")
    for key, val in targets_met.items():
        print(f"  {key}: {val}")

    print(f"\nTotal evaluation time: {elapsed_total:.1f}s")
    print(f"Results saved to: results/ai_feynman/, results/nguyen/, "
          f"results/strogatz/, results/benchmark_comparison.json")


if __name__ == "__main__":
    main()
