"""
Robustness evaluation for the PARR model.
Tests performance under varying levels of Gaussian noise added to observations.
Uses quick token-level matching for speed instead of full SymPy evaluation.
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


NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]
SEED = 42


def collate_fn(batch):
    return {
        "observations": torch.stack([b["observations"] for b in batch]),
        "equations": torch.stack([b["equation"] for b in batch]),
        "tiers": [b["tier"] for b in batch],
        "n_vars": [b["n_input_vars"] for b in batch],
        "n_obs": [b["n_obs"] for b in batch],
    }


def quick_token_match(pred, gt):
    """Check if predicted tokens match ground truth (ignoring PAD tokens)."""
    pad = EQUATION_VOCAB["[PAD]"]
    p = [t for t in pred if t != pad]
    g = [t for t in gt if t != pad]
    return p == g


def add_observation_noise(obs_tensor, noise_level, rng):
    """Add Gaussian noise to observations.

    Args:
        obs_tensor: (N, max_obs, max_vars+1) tensor.
        noise_level: Standard deviation of Gaussian noise.
        rng: numpy RandomState for reproducibility.

    Returns:
        Noisy observation tensor (same shape).
    """
    if noise_level <= 0.0:
        return obs_tensor.clone()

    noise = torch.from_numpy(
        rng.randn(*obs_tensor.shape).astype(np.float32)
    ) * noise_level

    noisy_obs = obs_tensor + noise

    # Preserve zero-padded rows: if original row is all zeros, keep it zero
    row_is_padding = (obs_tensor.abs().sum(dim=-1) == 0)  # (N, max_obs)
    noisy_obs[row_is_padding] = 0.0

    return noisy_obs


def evaluate_robustness_at_noise(model, obs_tensor, all_gts, all_tiers,
                                  noise_level, rng, device, batch_size=32):
    """Evaluate model at a specific noise level using quick token match.

    Returns:
        Dict with overall and per-tier token match accuracy.
    """
    noisy_obs = add_observation_noise(obs_tensor, noise_level, rng)
    n = noisy_obs.shape[0]

    model.eval()
    all_preds = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            obs_batch = noisy_obs[start:end].to(device)
            pred_tokens = model.generate(obs_batch, K=8)
            all_preds.extend(pred_tokens.cpu().numpy().tolist())

    # Compute per-tier and overall token match accuracy
    from collections import defaultdict
    tier_correct = defaultdict(int)
    tier_total = defaultdict(int)
    total_correct = 0

    for i in range(n):
        tier = all_tiers[i]
        matched = quick_token_match(all_preds[i], all_gts[i])
        tier_total[tier] += 1
        if matched:
            tier_correct[tier] += 1
            total_correct += 1

    per_tier = {}
    for tier in sorted(tier_total.keys()):
        acc = tier_correct[tier] / tier_total[tier] if tier_total[tier] > 0 else 0.0
        per_tier[str(tier)] = {
            "token_match_accuracy": acc,
            "n_correct": tier_correct[tier],
            "n_total": tier_total[tier],
        }

    overall_acc = total_correct / n if n > 0 else 0.0

    return {
        "per_tier": per_tier,
        "overall": {
            "token_match_accuracy": overall_acc,
            "n_correct": total_correct,
            "n_total": n,
        },
    }


def print_robustness_table(all_results):
    """Print a formatted robustness results table."""
    # Gather all tiers across all noise levels
    all_tiers = set()
    for noise_level, result in all_results.items():
        all_tiers.update(result["per_tier"].keys())
    all_tiers = sorted(all_tiers)

    # Header
    noise_cols = "".join(f"  {nl:>8}" for nl in sorted(all_results.keys(), key=float))
    header = f"{'':>12}{noise_cols}"
    sep = "-" * len(header)

    print("\n" + sep)
    print("         ROBUSTNESS: Token Match Accuracy vs. Observation Noise")
    print(sep)
    print(f"{'Noise std ->':>12}{noise_cols}")
    print(sep)

    # Per-tier rows
    sorted_noise = sorted(all_results.keys(), key=float)
    for tier in all_tiers:
        vals = ""
        for nl in sorted_noise:
            acc = all_results[nl]["per_tier"].get(tier, {}).get("token_match_accuracy", 0.0)
            vals += f"  {acc:>8.3f}"
        print(f"  Tier {tier:>5}{vals}")

    print(sep)

    # Overall row
    vals = ""
    for nl in sorted_noise:
        acc = all_results[nl]["overall"]["token_match_accuracy"]
        vals += f"  {acc:>8.3f}"
    print(f"  {'Overall':>9}{vals}")
    print(sep)


def run_robustness(
    checkpoint_path="checkpoints/parr_best.pt",
    data_dir="data",
    results_dir="results",
    device="cuda",
    n_samples=5000,
    batch_size=32,
):
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: PARR checkpoint not found at {checkpoint_path}")
        return None

    os.makedirs(results_dir, exist_ok=True)

    # Load test data
    print("Loading test dataset...")
    test_ds = PhysicsDataset(data_dir, "test")
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    # Collect test samples once
    print("Collecting test samples...")
    all_obs = []
    all_gts = []
    all_tiers = []
    n_collected = 0

    for batch in test_loader:
        if n_collected >= n_samples:
            break
        all_obs.append(batch["observations"])
        all_gts.extend(batch["equations"].numpy().tolist())
        all_tiers.extend(batch["tiers"])
        n_collected += len(batch["observations"])

    obs_tensor = torch.cat(all_obs, dim=0)[:n_samples]
    all_gts = all_gts[:n_samples]
    all_tiers = all_tiers[:n_samples]
    actual_n = obs_tensor.shape[0]
    print(f"Collected {actual_n} test samples")

    # Load model
    print("Loading PARR model...")
    model = create_parr_model(d_model=512, K=8, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Evaluate at each noise level
    all_results = {}

    for noise_level in NOISE_LEVELS:
        rng = np.random.RandomState(SEED)
        print(f"\nEvaluating with noise_level={noise_level:.2f} ...")
        t0 = time.time()

        result = evaluate_robustness_at_noise(
            model, obs_tensor, all_gts, all_tiers,
            noise_level, rng, device, batch_size=batch_size,
        )
        elapsed = time.time() - t0

        result["noise_level"] = noise_level
        result["eval_time_s"] = elapsed
        all_results[str(noise_level)] = result

        acc = result["overall"]["token_match_accuracy"]
        print(f"  Overall token match accuracy: {acc:.3f} ({elapsed:.1f}s)")

    # Build output
    output = {
        "model": "PARR Transformer",
        "checkpoint": checkpoint_path,
        "n_samples": actual_n,
        "noise_levels": NOISE_LEVELS,
        "seed": SEED,
        "results": all_results,
    }

    results_path = os.path.join(results_dir, "robustness_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nRobustness results saved to {results_path}")

    # Print table
    print_robustness_table(all_results)

    return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_robustness(device=device)
