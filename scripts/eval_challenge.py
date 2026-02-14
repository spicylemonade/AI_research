#!/usr/bin/env python3
"""Evaluate PhysMDT on a challenge set of complex equations.

Evaluates on at least 15 complex equations not seen during standard training,
including generator-sourced complex-difficulty equations and 5 custom
Kepler/Lagrangian-style challenge equations.  Uses SoftMaskRefinement with
K=10 for generation.

Outputs:
    results/challenge/metrics.json          -- per-equation and aggregate metrics
    results/challenge/qualitative_examples.md -- 5 best and 5 worst predictions
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import PhysicsDatasetGenerator, EquationTemplate, VARIABLE_RANGES
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
MODEL_PATH = "results/phys_mdt/model.pt"
OUTPUT_DIR = "results/challenge"


# ---------------------------------------------------------------------------
# Custom Kepler / Lagrangian challenge equations
# ---------------------------------------------------------------------------

def build_custom_challenge_templates():
    """Return 5 extra Kepler/Lagrangian-style challenge equations."""
    return [
        # 1. Kepler's Third Law: T^2 = (4 pi^2 / (G M)) r^3
        EquationTemplate(
            name="kepler_third_law_full",
            family="gravitation",
            difficulty="complex",
            prefix_notation=(
                "mul div mul INT_4 pow pi INT_2 mul G_const m pow r INT_3"
            ),
            infix_readable="T^2 = (4*pi^2/(G*M)) * r^3",
            variables=["m", "r"],
            coeff_ranges={"c_m": (1e25, 1e30)},
            eval_func="4 * math.pi**2 / (6.674e-11 * c_m) * var_r**3",
        ),
        # 2. Vis-viva equation (orbital mechanics):
        #    v^2 = G M (2/r - 1/a)  -->  v^2 = G*M*sub(div(2,r), div(1,a))
        EquationTemplate(
            name="vis_viva_equation",
            family="gravitation",
            difficulty="complex",
            prefix_notation=(
                "mul G_const mul m sub div INT_2 r div INT_1 a"
            ),
            infix_readable="v^2 = G*M*(2/r - 1/a)",
            variables=["m", "r", "a"],
            coeff_ranges={"c_m": (1e25, 1e30)},
            eval_func="6.674e-11 * c_m * (2.0 / var_r - 1.0 / var_a)",
        ),
        # 3. Lagrangian of simple pendulum (small angle):
        #    L = 0.5*m*l^2*omega^2 - m*g*l*(1-cos(theta))
        EquationTemplate(
            name="lagrangian_pendulum",
            family="oscillations",
            difficulty="complex",
            prefix_notation=(
                "sub mul div INT_1 INT_2 mul m mul pow l_length INT_2 pow omega INT_2 "
                "mul m mul g_accel mul l_length sub INT_1 cos theta"
            ),
            infix_readable="L = 0.5*m*l^2*omega^2 - m*g*l*(1-cos(theta))",
            variables=["m", "l_length", "omega", "theta"],
            coeff_ranges={"c_m": (0.5, 10.0)},
            eval_func=(
                "0.5 * c_m * var_l_length**2 * var_omega**2 "
                "- c_m * 9.81 * var_l_length * (1 - math.cos(var_theta))"
            ),
        ),
        # 4. Reduced Kepler orbit equation  r = a(1-e^2)/(1+e*cos(theta))
        EquationTemplate(
            name="kepler_orbit_equation",
            family="gravitation",
            difficulty="complex",
            prefix_notation=(
                "div mul a sub INT_1 pow mu INT_2 add INT_1 mul mu cos theta"
            ),
            infix_readable="r = a*(1-e^2)/(1+e*cos(theta))",
            variables=["a", "mu", "theta"],
            coeff_ranges={"c_a": (1.0, 10.0), "c_mu": (0.1, 0.9)},
            eval_func=(
                "c_a * (1 - c_mu**2) / (1 + c_mu * math.cos(var_theta))"
            ),
        ),
        # 5. Coupled oscillator normal mode frequency:
        #    omega^2 = (k/m) + (2*k_c/m) = k/m + 2*k_c/m
        #    Using prefix: add div k_spring m div mul INT_2 mu m
        EquationTemplate(
            name="coupled_oscillator_freq",
            family="oscillations",
            difficulty="complex",
            prefix_notation=(
                "add div k_spring m div mul INT_2 mu m"
            ),
            infix_readable="omega^2 = k/m + 2*k_c/m",
            variables=["k_spring", "m", "mu"],
            coeff_ranges={
                "c_k": (1.0, 50.0),
                "c_m": (0.5, 10.0),
                "c_mu": (0.1, 5.0),
            },
            eval_func="c_k / c_m + 2 * c_mu / c_m",
        ),
    ]


# ---------------------------------------------------------------------------
# Data helpers  (pattern from eval_quick.py / eval_ablations.py)
# ---------------------------------------------------------------------------

def build_test_tensors(test_data, tokenizer, max_seq_len=MAX_SEQ_LEN,
                       n_points=N_POINTS, max_vars=MAX_VARS):
    """Convert a list of sample dicts to padded/clipped tensors."""
    X_all, Y_all, tgt_all = [], [], []
    for sample in test_data:
        ox = np.array(sample["observations_x"])[:n_points]
        oy = np.array(sample["observations_y"])[:n_points]

        # Pad rows if fewer than n_points
        if ox.shape[0] < n_points:
            pad = n_points - ox.shape[0]
            ox = np.vstack([ox, np.zeros((pad, ox.shape[1]))])
            oy = np.concatenate([oy, np.zeros(pad)])

        # Pad / truncate variable columns
        if ox.shape[1] < max_vars:
            ox = np.hstack([ox, np.zeros((ox.shape[0], max_vars - ox.shape[1]))])
        elif ox.shape[1] > max_vars:
            ox = ox[:, :max_vars]

        tids = tokenizer.encode(sample["prefix_notation"], max_length=max_seq_len)

        ox = np.clip(np.nan_to_num(ox, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
        oy = np.clip(np.nan_to_num(oy, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
        ox = ox / (np.std(ox) + 1e-8)
        oy = oy / (np.std(oy) + 1e-8)

        X_all.append(ox)
        Y_all.append(oy)
        tgt_all.append(tids)

    X = torch.tensor(np.clip(np.array(X_all, dtype=np.float32), -100, 100))
    Y = torch.tensor(np.clip(np.array(Y_all, dtype=np.float32), -100, 100))
    T = torch.tensor(np.array(tgt_all)).long()
    return X, Y, T


def generate_custom_samples(templates, rng_seed=SEED, n_points=N_POINTS):
    """Generate observation data for the custom challenge templates.

    Returns a list of sample dicts compatible with the generator output format.
    """
    rng = np.random.RandomState(rng_seed)
    py_rng = __import__("random").Random(rng_seed)
    samples = []
    for tmpl in templates:
        coeffs = tmpl.generate_coefficients(py_rng)
        prefix = tmpl.instantiate_prefix(coeffs)

        # Sample observation points
        n_vars = len(tmpl.variables)
        X = np.zeros((n_points, n_vars))
        Y = np.zeros(n_points)
        for i in range(n_points):
            var_values = {}
            for j, var in enumerate(tmpl.variables):
                lo, hi = VARIABLE_RANGES.get(var, (0.1, 10.0))
                val = rng.uniform(lo, hi)
                var_values[f"var_{var}"] = val
                X[i, j] = val

            namespace = {
                **coeffs, **var_values,
                "math": math, "np": np,
                "pi": math.pi, "g": 9.81, "G": 6.674e-11,
            }
            try:
                y_val = float(eval(tmpl.eval_func, {"__builtins__": {}}, namespace))
                Y[i] = y_val if math.isfinite(y_val) else 0.0
            except Exception:
                Y[i] = 0.0

        samples.append({
            "template_name": tmpl.name,
            "family": tmpl.family,
            "difficulty": tmpl.difficulty,
            "prefix_notation": prefix,
            "infix_readable": tmpl.infix_readable,
            "variables": tmpl.variables,
            "observations_x": X.tolist(),
            "observations_y": Y.tolist(),
            "n_points": n_points,
        })
    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(model, X, Y, seq_len, tokenizer, test_data,
                   refine_steps=10):
    """Evaluate the model on all samples using SoftMaskRefinement with K steps."""
    model.eval()
    n = X.shape[0]

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

    with torch.no_grad():
        pred = refiner.refine(X, Y, seq_len=seq_len)

    results = []
    for i in range(n):
        pred_prefix = tokenizer.decode(pred[i].cpu().tolist())
        gt_prefix = test_data[i]["prefix_notation"]
        m = composite_score(pred_prefix, gt_prefix)
        m["predicted_prefix"] = pred_prefix
        m["gt_prefix"] = gt_prefix
        m["template_name"] = test_data[i]["template_name"]
        m["family"] = test_data[i]["family"]
        m["difficulty"] = test_data[i]["difficulty"]
        m["infix_readable"] = test_data[i].get("infix_readable", "")
        results.append(m)
    return results


def aggregate(results):
    """Compute mean metrics across a list of per-sample result dicts."""
    metric_keys = [
        "exact_match", "symbolic_equivalence", "numerical_r2",
        "tree_edit_distance", "complexity_penalty", "composite_score",
    ]
    if not results:
        return {k: 0.0 for k in metric_keys}
    return {k: float(np.mean([r[k] for r in results])) for k in metric_keys}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_qualitative_report(results, path):
    """Write a markdown file showing the 5 best and 5 worst predictions."""
    sorted_by_score = sorted(results, key=lambda r: r["composite_score"], reverse=True)
    best_5 = sorted_by_score[:5]
    worst_5 = sorted_by_score[-5:]

    lines = [
        "# Challenge Set: Qualitative Examples",
        "",
        "## 5 Best Predictions",
        "",
        "| # | Template | Family | Composite | EM | SE | Ground Truth | Prediction |",
        "|---|----------|--------|-----------|----|----|--------------|------------|",
    ]
    for i, r in enumerate(best_5, 1):
        lines.append(
            f"| {i} | {r['template_name']} | {r['family']} | "
            f"{r['composite_score']:.2f} | {r['exact_match']:.0f} | "
            f"{r['symbolic_equivalence']:.0f} | "
            f"`{r['gt_prefix'][:60]}` | `{r['predicted_prefix'][:60]}` |"
        )

    lines += [
        "",
        "## 5 Worst Predictions",
        "",
        "| # | Template | Family | Composite | EM | SE | Ground Truth | Prediction |",
        "|---|----------|--------|-----------|----|----|--------------|------------|",
    ]
    for i, r in enumerate(worst_5, 1):
        lines.append(
            f"| {i} | {r['template_name']} | {r['family']} | "
            f"{r['composite_score']:.2f} | {r['exact_match']:.0f} | "
            f"{r['symbolic_equivalence']:.0f} | "
            f"`{r['gt_prefix'][:60]}` | `{r['predicted_prefix'][:60]}` |"
        )

    lines += [
        "",
        "---",
        f"Total equations evaluated: {len(results)}",
        f"Symbolic equivalence rate: "
        f"{np.mean([r['symbolic_equivalence'] for r in results]) * 100:.1f}%",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    tokenizer = PhysicsTokenizer()

    # ----- 1. Collect complex equations from the generator test set ---------
    gen = PhysicsDatasetGenerator(seed=SEED)
    # Advance the RNG state the same way as training/eval scripts so the test
    # split is consistent.
    _ = gen.generate_dataset(4000, N_POINTS)   # train
    _ = gen.generate_dataset(500, N_POINTS)    # val
    full_test = gen.generate_dataset(500, N_POINTS)

    complex_from_gen = [s for s in full_test if s["difficulty"] == "complex"]
    # Ensure we have at least 10 from the generator
    if len(complex_from_gen) < 10:
        # Fall back: generate more complex-only samples
        extra_complex = gen.generate_dataset(50, N_POINTS, difficulty="complex")
        complex_from_gen.extend(extra_complex)
    complex_from_gen = complex_from_gen[:15]

    # ----- 2. Custom Kepler / Lagrangian challenge equations ----------------
    custom_templates = build_custom_challenge_templates()
    custom_samples = generate_custom_samples(custom_templates)

    # Merge into a single challenge set (>= 15 equations)
    challenge_data = complex_from_gen + custom_samples
    print(f"Challenge set: {len(challenge_data)} equations "
          f"({len(complex_from_gen)} from generator, {len(custom_samples)} custom)")

    # ----- 3. Build tensors -------------------------------------------------
    X, Y, T = build_test_tensors(challenge_data, tokenizer)

    # ----- 4. Load model ----------------------------------------------------
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
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"WARNING: No saved model at {MODEL_PATH}, using random weights")

    # ----- 5. Evaluate with SoftMaskRefinement K=10 -------------------------
    print(f"\nEvaluating {len(challenge_data)} challenge equations "
          f"with SoftMaskRefinement K=10 ...")
    t0 = time.time()
    results = evaluate_batch(model, X, Y, MAX_SEQ_LEN, tokenizer,
                             challenge_data, refine_steps=10)
    elapsed = time.time() - t0
    agg = aggregate(results)

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Composite:  {agg['composite_score']:.2f}")
    print(f"  Exact Match:          {agg['exact_match'] * 100:.1f}%")
    print(f"  Symbolic Equivalence: {agg['symbolic_equivalence'] * 100:.1f}%")
    print(f"  Numerical R2:         {agg['numerical_r2']:.4f}")
    print(f"  Tree Edit Distance:   {agg['tree_edit_distance']:.4f}")
    print(f"  Complexity Penalty:   {agg['complexity_penalty']:.4f}")

    # Per-family breakdown
    families = {}
    for r in results:
        families.setdefault(r["family"], []).append(r)
    print("\n--- Per-family breakdown ---")
    for fam, fam_results in sorted(families.items()):
        fa = aggregate(fam_results)
        print(f"  {fam:<20s} Composite={fa['composite_score']:.2f}  "
              f"SE={fa['symbolic_equivalence'] * 100:.1f}%  "
              f"({len(fam_results)} eqs)")

    # ----- 6. Save results/challenge/metrics.json ---------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Strip long observation data for cleaner JSON
    per_eq_results = []
    for r in results:
        per_eq_results.append({
            "template_name": r["template_name"],
            "family": r["family"],
            "difficulty": r["difficulty"],
            "infix_readable": r["infix_readable"],
            "gt_prefix": r["gt_prefix"],
            "predicted_prefix": r["predicted_prefix"],
            "exact_match": r["exact_match"],
            "symbolic_equivalence": r["symbolic_equivalence"],
            "numerical_r2": r["numerical_r2"],
            "tree_edit_distance": r["tree_edit_distance"],
            "complexity_penalty": r["complexity_penalty"],
            "composite_score": r["composite_score"],
        })

    output = {
        "model": "PhysMDT",
        "config": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "max_seq_len": MAX_SEQ_LEN,
            "refinement_K": 10,
            "seed": SEED,
        },
        "n_challenge_equations": len(challenge_data),
        "n_from_generator": len(complex_from_gen),
        "n_custom_kepler_lagrangian": len(custom_samples),
        "aggregate_metrics": agg,
        "symbolic_equivalence_rate": float(
            np.mean([r["symbolic_equivalence"] for r in results])
        ),
        "per_family": {
            fam: aggregate(fam_results)
            for fam, fam_results in sorted(families.items())
        },
        "per_equation_results": per_eq_results,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # ----- 7. Qualitative examples markdown ---------------------------------
    qual_path = os.path.join(OUTPUT_DIR, "qualitative_examples.md")
    write_qualitative_report(results, qual_path)
    print(f"Saved qualitative examples to {qual_path}")

    # ----- 8. Report symbolic equivalence rate ------------------------------
    se_rate = np.mean([r["symbolic_equivalence"] for r in results]) * 100
    print(f"\n{'='*60}")
    print(f"CHALLENGE SET SYMBOLIC EQUIVALENCE RATE: {se_rate:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
