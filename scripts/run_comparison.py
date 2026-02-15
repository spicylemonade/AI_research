"""
Head-to-head comparison between PARR and baseline transformer models.
Evaluates both models on the same test samples and produces a comparison table.
"""
import os
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import PhysicsDataset
from src.data.equation_templates import EQUATION_VOCAB
from src.models.parr_transformer import create_parr_model
from src.models.baseline_transformer import create_baseline_model
from src.evaluation.metrics import evaluate_model_predictions


def collate_fn(batch):
    return {
        "observations": torch.stack([b["observations"] for b in batch]),
        "equations": torch.stack([b["equation"] for b in batch]),
        "tiers": [b["tier"] for b in batch],
        "n_vars": [b["n_input_vars"] for b in batch],
        "n_obs": [b["n_obs"] for b in batch],
    }


def collect_test_data(test_loader, n_samples=5000):
    """Collect test data batches up to n_samples."""
    all_obs = []
    all_gts = []
    all_tiers = []
    all_n_vars = []
    all_n_obs = []
    n_collected = 0

    for batch in test_loader:
        if n_collected >= n_samples:
            break
        all_obs.append(batch["observations"])
        all_gts.extend(batch["equations"].numpy().tolist())
        all_tiers.extend(batch["tiers"])
        all_n_vars.extend(batch["n_vars"])
        all_n_obs.extend(batch["n_obs"])
        n_collected += len(batch["observations"])

    # Trim to exact count
    all_obs_tensor = torch.cat(all_obs, dim=0)[:n_samples]
    all_gts = all_gts[:n_samples]
    all_tiers = all_tiers[:n_samples]
    all_n_vars = all_n_vars[:n_samples]
    all_n_obs = all_n_obs[:n_samples]

    return all_obs_tensor, all_gts, all_tiers, all_n_vars, all_n_obs


def generate_predictions(model, obs_tensor, device, batch_size=32, K=None):
    """Generate predictions for all observations in batches.

    Args:
        model: The model to generate with.
        obs_tensor: (N, max_obs, max_vars+1) tensor of observations.
        device: torch device.
        batch_size: Batch size for generation.
        K: Refinement steps (only used for PARR model, pass None for baseline).

    Returns:
        List of predicted token sequences.
    """
    model.eval()
    all_preds = []
    n = obs_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            obs_batch = obs_tensor[start:end].to(device)
            if K is not None:
                pred_tokens = model.generate(obs_batch, K=K)
            else:
                pred_tokens = model.generate(obs_batch)
            all_preds.extend(pred_tokens.cpu().numpy().tolist())

    return all_preds


def print_comparison_table(baseline_results, parr_results):
    """Print a formatted comparison table."""
    # Collect all tiers from both results
    all_tiers = sorted(
        set(list(baseline_results.get("per_tier", {}).keys()) +
            list(parr_results.get("per_tier", {}).keys()))
    )

    header = f"{'':>12} | {'Baseline ESM':>13} {'R2':>7} {'NTED':>7} {'CAA':>7} | {'PARR ESM':>10} {'R2':>7} {'NTED':>7} {'CAA':>7} | {'Delta ESM':>10}"
    sep = "-" * len(header)

    print("\n" + sep)
    print("                        HEAD-TO-HEAD COMPARISON")
    print(sep)
    print(header)
    print(sep)

    for tier in all_tiers:
        bt = baseline_results["per_tier"].get(tier, {})
        pt = parr_results["per_tier"].get(tier, {})

        b_esm = bt.get("exact_match_rate", 0.0)
        b_r2 = bt.get("r2_score", 0.0)
        b_nted = bt.get("tree_edit_distance", 1.0)
        b_caa = bt.get("complexity_adjusted_accuracy", 0.0)

        p_esm = pt.get("exact_match_rate", 0.0)
        p_r2 = pt.get("r2_score", 0.0)
        p_nted = pt.get("tree_edit_distance", 1.0)
        p_caa = pt.get("complexity_adjusted_accuracy", 0.0)

        delta_esm = p_esm - b_esm
        sign = "+" if delta_esm >= 0 else ""

        print(f"  Tier {tier:>5} | {b_esm:>12.3f} {b_r2:>7.3f} {b_nted:>7.3f} {b_caa:>7.3f} | {p_esm:>9.3f} {p_r2:>7.3f} {p_nted:>7.3f} {p_caa:>7.3f} | {sign}{delta_esm:>9.3f}")

    print(sep)

    # Overall row
    bo = baseline_results["overall"]
    po = parr_results["overall"]
    delta = po["exact_match_rate"] - bo["exact_match_rate"]
    sign = "+" if delta >= 0 else ""

    print(f"  {'Overall':>9} | {bo['exact_match_rate']:>12.3f} {bo['r2_score']:>7.3f} {bo['tree_edit_distance']:>7.3f} {bo['complexity_adjusted_accuracy']:>7.3f} | {po['exact_match_rate']:>9.3f} {po['r2_score']:>7.3f} {po['tree_edit_distance']:>7.3f} {po['complexity_adjusted_accuracy']:>7.3f} | {sign}{delta:>9.3f}")
    print(sep)


def run_comparison(
    baseline_checkpoint="checkpoints/baseline_best.pt",
    parr_checkpoint="checkpoints/parr_best.pt",
    data_dir="data",
    results_dir="results",
    device="cuda",
    n_samples=5000,
    batch_size=32,
):
    # Check that checkpoints exist
    if not os.path.exists(baseline_checkpoint):
        print(f"ERROR: Baseline checkpoint not found at {baseline_checkpoint}")
        return None
    if not os.path.exists(parr_checkpoint):
        print(f"ERROR: PARR checkpoint not found at {parr_checkpoint}")
        return None

    os.makedirs(results_dir, exist_ok=True)

    # Load test data
    print("Loading test dataset...")
    test_ds = PhysicsDataset(data_dir, "test")
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    print("Collecting test samples...")
    obs_tensor, all_gts, all_tiers, all_n_vars, all_n_obs = collect_test_data(
        test_loader, n_samples=n_samples
    )
    actual_n = obs_tensor.shape[0]
    print(f"Collected {actual_n} test samples")

    metadata = [
        {"tier": all_tiers[i], "n_input_vars": all_n_vars[i], "n_obs": all_n_obs[i]}
        for i in range(actual_n)
    ]
    obs_array = obs_tensor.numpy()

    # --- Evaluate Baseline ---
    print("\nLoading baseline model...")
    baseline_model = create_baseline_model(device=device)
    baseline_model.load_state_dict(
        torch.load(baseline_checkpoint, map_location=device)
    )
    baseline_model.eval()

    print("Generating baseline predictions...")
    t0 = time.time()
    baseline_preds = generate_predictions(
        baseline_model, obs_tensor, device, batch_size=batch_size, K=None
    )
    baseline_gen_time = time.time() - t0
    print(f"Baseline generation: {baseline_gen_time:.1f}s ({actual_n / baseline_gen_time:.1f} eq/s)")

    # Free baseline model memory
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Evaluating baseline predictions...")
    baseline_results = evaluate_model_predictions(
        baseline_preds, all_gts, obs_array, metadata, max_eval=n_samples
    )
    baseline_results["model"] = "Baseline Transformer"
    baseline_results["checkpoint"] = baseline_checkpoint
    baseline_results["n_samples"] = actual_n
    baseline_results["generation_time_s"] = baseline_gen_time

    # --- Evaluate PARR ---
    print("\nLoading PARR model...")
    parr_model = create_parr_model(d_model=512, K=8, device=device)
    parr_model.load_state_dict(
        torch.load(parr_checkpoint, map_location=device)
    )
    parr_model.eval()

    print("Generating PARR predictions...")
    t0 = time.time()
    parr_preds = generate_predictions(
        parr_model, obs_tensor, device, batch_size=batch_size, K=8
    )
    parr_gen_time = time.time() - t0
    print(f"PARR generation: {parr_gen_time:.1f}s ({actual_n / parr_gen_time:.1f} eq/s)")

    del parr_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Evaluating PARR predictions...")
    parr_results = evaluate_model_predictions(
        parr_preds, all_gts, obs_array, metadata, max_eval=n_samples
    )
    parr_results["model"] = "PARR Transformer"
    parr_results["checkpoint"] = parr_checkpoint
    parr_results["n_samples"] = actual_n
    parr_results["generation_time_s"] = parr_gen_time

    # --- Build comparison output ---
    comparison = {
        "baseline": baseline_results,
        "parr": parr_results,
        "n_samples": actual_n,
        "speedup": baseline_gen_time / parr_gen_time if parr_gen_time > 0 else 0.0,
        "delta_overall": {
            "exact_match_rate": parr_results["overall"]["exact_match_rate"] - baseline_results["overall"]["exact_match_rate"],
            "r2_score": parr_results["overall"]["r2_score"] - baseline_results["overall"]["r2_score"],
            "tree_edit_distance": parr_results["overall"]["tree_edit_distance"] - baseline_results["overall"]["tree_edit_distance"],
            "complexity_adjusted_accuracy": parr_results["overall"]["complexity_adjusted_accuracy"] - baseline_results["overall"]["complexity_adjusted_accuracy"],
        },
    }

    # Save results
    results_path = os.path.join(results_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison results saved to {results_path}")

    # Print comparison table
    print_comparison_table(baseline_results, parr_results)

    return comparison


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_comparison(device=device)
