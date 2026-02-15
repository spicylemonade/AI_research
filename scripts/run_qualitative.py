"""
Qualitative analysis: show how PARR refinement improves predictions.
For 20 test examples (5 from each tier), compare AR-only (K=0) vs refined (K=4).
Saves results to results/qualitative_analysis.json.
"""
import os
import sys
import json
import random
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import PhysicsDataset
from src.data.equation_templates import EQUATION_VOCAB
from src.models.parr_transformer import create_parr_model
from src.evaluation.metrics import tokens_to_prefix_str, exact_symbolic_match


PAD_TOKEN = EQUATION_VOCAB["[PAD]"]


def collate_fn(batch):
    return {
        "observations": torch.stack([b["observations"] for b in batch]),
        "equations": torch.stack([b["equation"] for b in batch]),
        "tiers": [b["tier"] for b in batch],
        "n_vars": [b["n_input_vars"] for b in batch],
        "n_obs": [b["n_obs"] for b in batch],
    }


def format_tokens(token_ids):
    """Convert token IDs to a readable prefix string."""
    prefix = tokens_to_prefix_str(token_ids)
    return " ".join(prefix) if prefix else "<empty>"


def run_qualitative(
    checkpoint_path="checkpoints/parr_best.pt",
    data_dir="data",
    results_dir="results",
    d_model=512,
    K_trained=8,
    K_refine=4,
    device="cuda",
    seed=42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading test dataset...")
    test_ds = PhysicsDataset(data_dir, "test")

    # Select 5 examples per tier (20 total)
    selected_indices = []
    for tier in [1, 2, 3, 4]:
        tier_indices = test_ds.get_tier_indices(tier)
        if len(tier_indices) < 5:
            print(f"Warning: only {len(tier_indices)} samples for tier {tier}")
            chosen = tier_indices
        else:
            chosen = random.sample(tier_indices, 5)
        selected_indices.extend(sorted(chosen))

    subset = Subset(test_ds, selected_indices)
    loader = DataLoader(
        subset, batch_size=len(selected_indices), shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. "
              "Creating model with random weights for demonstration.")
        model = create_parr_model(d_model=d_model, K=K_trained, device=device)
    else:
        print(f"Loading PARR model from {checkpoint_path}...")
        model = create_parr_model(d_model=d_model, K=K_trained, device=device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ------------------------------------------------------------------
    # Generate predictions
    # ------------------------------------------------------------------
    print("Generating AR-only (K=0) and refined (K=4) predictions...")
    batch = next(iter(loader))
    obs = batch["observations"].to(device)
    gt_tokens = batch["equations"]  # stays on CPU

    with torch.no_grad():
        ar_only_tokens = model.generate(obs, K=0)
        refined_tokens = model.generate(obs, K=K_refine)

    ar_only_list = ar_only_tokens.cpu().tolist()
    refined_list = refined_tokens.cpu().tolist()
    gt_list = gt_tokens.tolist()
    tiers = batch["tiers"]

    # ------------------------------------------------------------------
    # Build analysis records
    # ------------------------------------------------------------------
    examples = []
    for i in range(len(gt_list)):
        gt = gt_list[i]
        ar = ar_only_list[i]
        ref = refined_list[i]

        gt_str = format_tokens(gt)
        ar_str = format_tokens(ar)
        ref_str = format_tokens(ref)

        ar_match = exact_symbolic_match(ar, gt)
        ref_match = exact_symbolic_match(ref, gt)

        example = {
            "index": selected_indices[i],
            "tier": tiers[i],
            "ground_truth": gt_str,
            "ar_only_K0": ar_str,
            "refined_K4": ref_str,
            "ar_only_correct": bool(ar_match),
            "refined_correct": bool(ref_match),
            "refinement_helped": bool(ref_match and not ar_match),
        }
        examples.append(example)

    # ------------------------------------------------------------------
    # Print formatted output
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("QUALITATIVE ANALYSIS: AR-only (K=0) vs Refined (K=4)")
    print("=" * 80)

    current_tier = None
    ar_correct_total = 0
    ref_correct_total = 0
    helped_total = 0

    for ex in examples:
        if ex["tier"] != current_tier:
            current_tier = ex["tier"]
            print(f"\n--- Tier {current_tier} ---")

        status_ar = "CORRECT" if ex["ar_only_correct"] else "WRONG"
        status_ref = "CORRECT" if ex["refined_correct"] else "WRONG"
        helped_tag = " ** REFINEMENT HELPED **" if ex["refinement_helped"] else ""

        print(f"\n  Example (idx={ex['index']}):")
        print(f"    Ground Truth : {ex['ground_truth']}")
        print(f"    AR-only (K=0): {ex['ar_only_K0']}  [{status_ar}]")
        print(f"    Refined (K=4): {ex['refined_K4']}  [{status_ref}]{helped_tag}")

        ar_correct_total += int(ex["ar_only_correct"])
        ref_correct_total += int(ex["refined_correct"])
        helped_total += int(ex["refinement_helped"])

    n = len(examples)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Total examples        : {n}")
    print(f"  AR-only correct       : {ar_correct_total}/{n} ({100*ar_correct_total/n:.1f}%)")
    print(f"  Refined correct       : {ref_correct_total}/{n} ({100*ref_correct_total/n:.1f}%)")
    print(f"  Refinement helped     : {helped_total}/{n}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "description": "Qualitative analysis: AR-only (K=0) vs Refined (K=4)",
        "checkpoint": checkpoint_path,
        "n_examples": n,
        "summary": {
            "ar_only_correct": ar_correct_total,
            "refined_correct": ref_correct_total,
            "refinement_helped": helped_total,
        },
        "examples": examples,
    }

    out_path = os.path.join(results_dir, "qualitative_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_qualitative(device=device)
