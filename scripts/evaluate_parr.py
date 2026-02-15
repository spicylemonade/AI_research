"""
Full evaluation script for PARR model on test set.
Produces results/parr_results.json and LaTeX table.
"""
import os
import sys
import json
import time
import numpy as np
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import PhysicsDataset
from src.data.equation_templates import EQUATION_VOCAB
from src.models.parr_transformer import create_parr_model
from src.evaluation.metrics import evaluate_model_predictions, format_results_latex


def collate_fn(batch):
    observations = torch.stack([b["observations"] for b in batch])
    equations = torch.stack([b["equation"] for b in batch])
    tiers = [b["tier"] for b in batch]
    n_vars = [b["n_input_vars"] for b in batch]
    n_obs = [b["n_obs"] for b in batch]
    return {
        "observations": observations,
        "equations": equations,
        "tiers": tiers,
        "n_vars": n_vars,
        "n_obs": n_obs,
    }


def evaluate_parr(
    checkpoint_path="checkpoints/parr_best.pt",
    data_dir="data",
    results_dir="results",
    d_model=512,
    K=8,
    device="cuda",
    n_samples=5000,
):
    os.makedirs(results_dir, exist_ok=True)

    print("Loading test dataset...")
    test_ds = PhysicsDataset(data_dir, "test")
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    print("Loading PARR model...")
    model = create_parr_model(d_model=d_model, K=K, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Generating predictions...")
    all_preds = []
    all_gts = []
    all_obs = []
    all_tiers = []
    all_n_vars = []
    all_n_obs = []

    n_collected = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            if n_collected >= n_samples:
                break

            obs = batch["observations"].to(device)
            pred_tokens = model.generate(obs, K=K)

            all_preds.extend(pred_tokens.cpu().numpy().tolist())
            all_gts.extend(batch["equations"].numpy().tolist())
            all_obs.extend(batch["observations"].numpy().tolist())
            all_tiers.extend(batch["tiers"])
            all_n_vars.extend(batch["n_vars"])
            all_n_obs.extend(batch["n_obs"])
            n_collected += len(obs)

    gen_time = time.time() - start_time
    print(f"Generated {n_collected} predictions in {gen_time:.1f}s")

    # Trim to exact count
    all_preds = all_preds[:n_samples]
    all_gts = all_gts[:n_samples]
    all_obs = all_obs[:n_samples]
    all_tiers = all_tiers[:n_samples]
    all_n_vars = all_n_vars[:n_samples]
    all_n_obs = all_n_obs[:n_samples]

    print(f"Evaluating {n_samples} predictions...")
    metadata = [
        {"tier": all_tiers[i], "n_input_vars": all_n_vars[i], "n_obs": all_n_obs[i]}
        for i in range(len(all_preds))
    ]
    obs_array = np.array(all_obs)
    results = evaluate_model_predictions(
        all_preds, all_gts, obs_array, metadata, max_eval=n_samples
    )

    # Add metadata
    results["model"] = "PARR Transformer"
    results["checkpoint"] = checkpoint_path
    results["n_samples"] = n_samples
    results["generation_time_s"] = gen_time
    results["throughput_eq_per_s"] = n_samples / gen_time

    # Save results
    results_path = f"{results_dir}/parr_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # Print results
    print("\n=== PARR Results ===")
    for tier in sorted(results.get("per_tier", {}).keys()):
        t = results["per_tier"][tier]
        print(f"Tier {tier}: ESM={t['exact_match_rate']:.3f}, R²={t['r2_score']:.3f}, "
              f"NTED={t['tree_edit_distance']:.3f}, CAA={t['complexity_adjusted_accuracy']:.3f}")
    ov = results["overall"]
    print(f"Overall: ESM={ov['exact_match_rate']:.3f}, R²={ov['r2_score']:.3f}, "
          f"NTED={ov['tree_edit_distance']:.3f}, CAA={ov['complexity_adjusted_accuracy']:.3f}")

    # LaTeX table
    latex = format_results_latex(results)
    print(f"\n{latex}")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_parr(device=device)
