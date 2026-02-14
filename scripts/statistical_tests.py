#!/usr/bin/env python3
"""Statistical significance tests comparing PhysMDT vs AR baseline.

For each metric (exact_match, symbolic_equivalence, numerical_r2,
tree_edit_distance, complexity_penalty, composite_score) this script:
  1. Computes a paired bootstrap 95% CI for the difference (PhysMDT - AR)
  2. Runs a Wilcoxon signed-rank test
  3. Computes Cohen's d effect size

Results are saved to results/statistical_tests.json and a summary table
is printed to stdout.
"""

import json
import math
import os
import sys

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS_MDT_PATH = os.path.join(REPO_ROOT, "results", "phys_mdt", "metrics.json")
BASELINE_AR_PATH = os.path.join(REPO_ROOT, "results", "baseline_ar", "metrics.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "statistical_tests.json")

# ---------------------------------------------------------------------------
# Metrics of interest
# ---------------------------------------------------------------------------

METRICS = [
    "exact_match",
    "symbolic_equivalence",
    "numerical_r2",
    "tree_edit_distance",
    "complexity_penalty",
    "composite_score",
]

# ---------------------------------------------------------------------------
# Bootstrap parameters
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_per_sample(path: str) -> list[dict]:
    """Load the per_sample_results array from a metrics JSON file."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("per_sample_results")
    if results is None:
        raise ValueError(f"No 'per_sample_results' key found in {path}")
    return results


def align_samples(
    phys_results: list[dict],
    ar_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Align PhysMDT and AR baseline per-sample results.

    First tries to align on common ``template_name`` keys.  If PhysMDT
    samples lack that field, falls back to pairing by index up to the
    minimum length of the two lists.
    """
    # Check whether both sides carry template_name
    phys_has_name = all("template_name" in r for r in phys_results)
    ar_has_name = all("template_name" in r for r in ar_results)

    if phys_has_name and ar_has_name:
        # Build lookup from template_name -> sample for the AR baseline.
        # If there are duplicates we keep the first occurrence.
        ar_by_name: dict[str, dict] = {}
        for r in ar_results:
            name = r["template_name"]
            if name not in ar_by_name:
                ar_by_name[name] = r

        aligned_phys = []
        aligned_ar = []
        for r in phys_results:
            name = r["template_name"]
            if name in ar_by_name:
                aligned_phys.append(r)
                aligned_ar.append(ar_by_name[name])

        if aligned_phys:
            return aligned_phys, aligned_ar

    # Fallback: pair by index up to min length
    n = min(len(phys_results), len(ar_results))
    return phys_results[:n], ar_results[:n]


def extract_metric(samples: list[dict], metric: str) -> np.ndarray:
    """Return a 1-D numpy array of the given metric from a sample list."""
    return np.array([s[metric] for s in samples], dtype=np.float64)


# ---------------------------------------------------------------------------
# Statistical routines
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    ci: float = BOOTSTRAP_CI,
    seed: int = SEED,
) -> dict:
    """Paired bootstrap 95% CI for mean(a - b).

    Parameters
    ----------
    a, b : array-like, same length
        Paired observations (a = PhysMDT, b = AR baseline).
    n_resamples : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: mean_diff, ci_lower, ci_upper, ci_level
    """
    rng = np.random.RandomState(seed)
    diff = a - b
    n = len(diff)

    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diff[idx].mean()

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean_diff": float(diff.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_level": ci,
        "n_resamples": n_resamples,
    }


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired samples.

    Returns statistic, p-value, and whether the result is significant
    at alpha = 0.05.  If all differences are zero the test is undefined;
    we return p = 1.0.
    """
    diff = a - b
    # scipy raises an error when all differences are zero
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "note": "All paired differences are zero; test undefined.",
        }
    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        # Can happen if there are too few non-zero differences
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant_at_0.05": False,
            "note": "Wilcoxon test could not be computed (too few non-zero differences).",
        }
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> dict:
    """Cohen's d for paired samples (a - b).

    Uses the standard deviation of the difference scores as the
    denominator (Cohen's d_z for paired designs).

    Interpretation thresholds (Cohen 1988):
      |d| < 0.2  -> negligible
      0.2 <= |d| < 0.5 -> small
      0.5 <= |d| < 0.8 -> medium
      |d| >= 0.8 -> large
    """
    diff = a - b
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1) if len(diff) > 1 else 0.0

    if sd_diff == 0:
        d = 0.0 if mean_diff == 0 else float("inf") * np.sign(mean_diff)
    else:
        d = mean_diff / sd_diff

    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "d": float(d),
        "magnitude": magnitude,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not os.path.exists(PHYS_MDT_PATH):
        print(f"ERROR: PhysMDT results not found at {PHYS_MDT_PATH}")
        sys.exit(1)
    if not os.path.exists(BASELINE_AR_PATH):
        print(f"ERROR: AR baseline results not found at {BASELINE_AR_PATH}")
        sys.exit(1)

    phys_raw = load_per_sample(PHYS_MDT_PATH)
    ar_raw = load_per_sample(BASELINE_AR_PATH)

    print(f"Loaded {len(phys_raw)} PhysMDT samples from {PHYS_MDT_PATH}")
    print(f"Loaded {len(ar_raw)} AR baseline samples from {BASELINE_AR_PATH}")

    phys_aligned, ar_aligned = align_samples(phys_raw, ar_raw)
    n_paired = len(phys_aligned)
    print(f"Paired samples for testing: {n_paired}")

    if n_paired == 0:
        print("ERROR: No paired samples available. Cannot run tests.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Run tests for each metric
    # ------------------------------------------------------------------
    output: dict = {
        "description": (
            "Statistical significance tests comparing PhysMDT vs AR baseline. "
            "Positive mean_diff indicates PhysMDT > AR baseline."
        ),
        "n_paired_samples": n_paired,
        "bootstrap_config": {
            "n_resamples": N_BOOTSTRAP,
            "ci_level": BOOTSTRAP_CI,
            "seed": SEED,
        },
        "per_metric": {},
    }

    for metric in METRICS:
        phys_vals = extract_metric(phys_aligned, metric)
        ar_vals = extract_metric(ar_aligned, metric)

        boot = paired_bootstrap_ci(phys_vals, ar_vals)
        wilcox = wilcoxon_test(phys_vals, ar_vals)
        effect = cohens_d(phys_vals, ar_vals)

        output["per_metric"][metric] = {
            "phys_mdt_mean": float(phys_vals.mean()),
            "ar_baseline_mean": float(ar_vals.mean()),
            "bootstrap_ci": boot,
            "wilcoxon": wilcox,
            "cohens_d": effect,
        }

    # ------------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 4. Print summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("STATISTICAL SIGNIFICANCE TESTS: PhysMDT vs AR Baseline")
    print(f"Paired samples: {n_paired}")
    print("=" * 90)
    header = (
        f"{'Metric':<24s} {'PhysMDT':>8s} {'AR':>8s} "
        f"{'Diff':>8s} {'95% CI':>20s} "
        f"{'p-value':>9s} {'Cohen d':>8s} {'Effect':>10s}"
    )
    print(header)
    print("-" * 90)

    for metric in METRICS:
        m = output["per_metric"][metric]
        boot = m["bootstrap_ci"]
        wilcox = m["wilcoxon"]
        effect = m["cohens_d"]

        ci_str = f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]"
        p_str = f"{wilcox['p_value']:.4f}" if not math.isnan(wilcox.get("statistic", 0)) else "  N/A"
        sig = "*" if wilcox.get("significant_at_0.05", False) else ""

        print(
            f"{metric:<24s} "
            f"{m['phys_mdt_mean']:>8.4f} "
            f"{m['ar_baseline_mean']:>8.4f} "
            f"{boot['mean_diff']:>+8.4f} "
            f"{ci_str:>20s} "
            f"{p_str:>8s}{sig:<1s} "
            f"{effect['d']:>+7.3f} "
            f"{effect['magnitude']:>10s}"
        )

    print("-" * 90)
    print("* significant at alpha = 0.05")
    print()


if __name__ == "__main__":
    main()
