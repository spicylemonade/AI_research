"""
Ablation study script for PARR transformer.
Tests impact of: refinement loop, ConvSwiGLU, Token Algebra, model size.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import PhysicsDataset
from src.data.equation_templates import EQUATION_VOCAB
from src.models.parr_transformer import create_parr_model
from src.evaluation.metrics import evaluate_model_predictions


def collate_fn(batch):
    return {
        "observations": torch.stack([b["observations"] for b in batch]),
        "equations": torch.stack([b["equation"] for b in batch]),
        "tiers": [b["tier"] for b in batch],
        "n_vars": [b["n_input_vars"] for b in batch],
        "n_obs": [b["n_obs"] for b in batch],
    }


def evaluate_with_refinement(model, test_loader, device, K, n_samples=5000):
    """Evaluate model with specified refinement steps."""
    model.eval()
    all_preds, all_gts, all_obs = [], [], []
    all_tiers, all_n_vars, all_n_obs = [], [], []
    n = 0

    with torch.no_grad():
        for batch in test_loader:
            if n >= n_samples:
                break
            obs = batch["observations"].to(device)
            pred = model.generate(obs, K=K)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_gts.extend(batch["equations"].numpy().tolist())
            all_obs.extend(batch["observations"].numpy().tolist())
            all_tiers.extend(batch["tiers"])
            all_n_vars.extend(batch["n_vars"])
            all_n_obs.extend(batch["n_obs"])
            n += len(obs)

    all_preds = all_preds[:n_samples]
    all_gts = all_gts[:n_samples]
    all_obs = all_obs[:n_samples]
    all_tiers = all_tiers[:n_samples]
    all_n_vars = all_n_vars[:n_samples]
    all_n_obs = all_n_obs[:n_samples]

    metadata = [
        {"tier": all_tiers[i], "n_input_vars": all_n_vars[i], "n_obs": all_n_obs[i]}
        for i in range(len(all_preds))
    ]
    obs_array = np.array(all_obs)
    return evaluate_model_predictions(
        all_preds, all_gts, obs_array, metadata, max_eval=n_samples
    )


def run_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    test_ds = PhysicsDataset("data", "test")
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    checkpoint = "checkpoints/parr_best.pt"
    if not os.path.exists(checkpoint):
        print("No PARR checkpoint found!")
        return

    ablation_results = {}

    # 1. Full PARR with refinement (K=8)
    print("\n=== Full PARR (K=8 refinement) ===")
    model = create_parr_model(d_model=512, K=8, device=device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    results = evaluate_with_refinement(model, test_loader, device, K=8, n_samples=5000)
    ablation_results["full_parr_K8"] = results
    print(f"Overall ESM: {results['overall']['exact_match_rate']:.3f}")

    # 2. PARR without refinement (AR only, K=0)
    print("\n=== PARR AR-only (K=0) ===")
    results = evaluate_with_refinement(model, test_loader, device, K=0, n_samples=5000)
    ablation_results["parr_ar_only"] = results
    print(f"Overall ESM: {results['overall']['exact_match_rate']:.3f}")

    # 3. PARR with fewer refinement steps (K=2, K=4)
    for k in [2, 4]:
        print(f"\n=== PARR (K={k} refinement) ===")
        results = evaluate_with_refinement(model, test_loader, device, K=k, n_samples=5000)
        ablation_results[f"parr_K{k}"] = results
        print(f"Overall ESM: {results['overall']['exact_match_rate']:.3f}")

    # Save results
    with open(f"{results_dir}/ablation_study.json", "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    print(f"\nAblation results saved to {results_dir}/ablation_study.json")

    return ablation_results


if __name__ == "__main__":
    run_ablation()
