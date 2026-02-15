"""Evaluate the trained PhysMDT model on in-distribution and held-out equations.

Produces:
    results/in_distribution_comparison.json   -- per-tier accuracy and R^2 stats
    results/zero_shot_discovery.json          -- per-equation results for held-out set

Usage:
    python training/evaluate_physmdt.py
    python training/evaluate_physmdt.py --checkpoint checkpoints/physmdt_best.pt
    python training/evaluate_physmdt.py --quick          # smoke test with fewer samples
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from data.equations import (
    get_training_equations,
    get_held_out_equations,
    get_equations_by_tier,
)
from data.physics_generator import PhysicsDataset
from data.tokenizer import ExprTokenizer, PAD_IDX, SOS_IDX, EOS_IDX, MASK_IDX
from evaluation.metrics import symbolic_equivalence, numeric_r2, token_edit_distance
from models.physmdt import PhysMDT, PhysMDTConfig
from models.refinement import generate_candidates, refine
from models.ttft import test_time_finetune, remove_lora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PhysMDT on in-distribution and held-out equations."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/physmdt_best.pt",
        help="Path to the trained PhysMDT checkpoint (default: checkpoints/physmdt_best.pt).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: fewer samples per equation and fewer refinement steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a PhysMDT model from a checkpoint file.

    Returns (model, config) on success, or (None, None) if the checkpoint
    does not exist or cannot be loaded.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("  Train the model first with: python training/train_physmdt.py")
        return None, None

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"[ERROR] Failed to load checkpoint: {exc}")
        return None, None

    # Reconstruct config -- the checkpoint may store a PhysMDTConfig directly
    # or as a dict.
    raw_config = ckpt.get("config", None)
    if raw_config is None:
        print("[WARN] Checkpoint has no 'config' key; using default PhysMDTConfig.")
        config = PhysMDTConfig()
    elif isinstance(raw_config, PhysMDTConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = PhysMDTConfig(**raw_config)
    else:
        config = raw_config  # assume it's already a dataclass

    model = PhysMDT(config).to(device)

    # Load state dict -- handle both top-level keys
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        print("[WARN] No model_state_dict found in checkpoint; using freshly-initialised weights.")

    model.eval()
    return model, config


def _decode_tokens_to_expr(token_ids, tokenizer: ExprTokenizer):
    """Attempt to decode a token-id sequence to a SymPy expression.

    Strips everything after the first EOS (inclusive) and removes PAD/MASK
    tokens before decoding.

    Returns the decoded SymPy expression, or None on failure.
    """
    ids = list(token_ids)

    # Truncate at first EOS (inclusive)
    if EOS_IDX in ids:
        ids = ids[: ids.index(EOS_IDX) + 1]

    # Remove stray MASK / PAD tokens before decoding
    ids = [i for i in ids if i not in (PAD_IDX, MASK_IDX)]

    if not ids:
        return None

    try:
        expr = tokenizer.decode(ids, strip_special=True)
        return expr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core evaluation routines
# ---------------------------------------------------------------------------

def evaluate_equation(
    model: PhysMDT,
    eq,
    device: torch.device,
    tokenizer: ExprTokenizer,
    n_test_samples: int = 5,
    n_points: int = 100,
    noise_level: float = 0.01,
    n_candidates: int = 8,
    n_steps: int = 64,
    seed: int = 42,
):
    """Evaluate the model on a single equation.

    Generates test observations via PhysicsDataset, runs PhysMDT with
    refinement (generate_candidates), decodes predicted tokens, and
    computes symbolic equivalence and R^2.

    Returns a dict of per-equation results.
    """
    # Generate test observations for this equation
    ds = PhysicsDataset(
        equations=[eq],
        n_samples=n_test_samples,
        n_points=n_points,
        noise_level=noise_level,
        seed=seed,
    )

    if len(ds) == 0:
        return {
            "equation_id": eq.id,
            "equation_name": eq.name,
            "tier": eq.tier,
            "held_out": eq.held_out,
            "n_samples": 0,
            "symbolic_accuracy": 0.0,
            "mean_r2": -1.0,
            "mean_edit_distance": 1.0,
            "mean_latency_ms": 0.0,
            "candidates": [],
        }

    sym_correct = 0
    r2_scores = []
    edit_distances = []
    latencies = []
    candidate_details = []

    seq_len = model.config.max_expr_len

    for i in range(min(len(ds), n_test_samples)):
        sample = ds[i]
        obs = sample["observations"].unsqueeze(0).to(device)
        obs_mask = sample["obs_mask"].unsqueeze(0).to(device)
        true_token_ids = sample["tokens"][: sample["token_len"]].tolist()

        start_time = time.time()
        with torch.no_grad():
            pred_tokens, confidence = generate_candidates(
                model=model,
                observations=obs,
                obs_mask=obs_mask,
                seq_len=seq_len,
                n_steps=n_steps,
                n_candidates=n_candidates,
                temperature=1.0,
                device=device,
            )
        latency_ms = (time.time() - start_time) * 1000.0
        latencies.append(latency_ms)

        # Decode predicted tokens
        gen_ids = pred_tokens[0].cpu().tolist()
        pred_expr = _decode_tokens_to_expr(gen_ids, tokenizer)
        true_expr = _decode_tokens_to_expr(true_token_ids, tokenizer)

        # Metrics
        is_equiv = False
        r2 = -1.0
        ed = 1.0
        if pred_expr is not None and true_expr is not None:
            try:
                is_equiv = symbolic_equivalence(pred_expr, true_expr)
            except Exception:
                is_equiv = False
            try:
                r2 = numeric_r2(pred_expr, true_expr, n_points=200)
            except Exception:
                r2 = -1.0
            try:
                ed = token_edit_distance(gen_ids, true_token_ids)
            except Exception:
                ed = 1.0

        if is_equiv:
            sym_correct += 1
        r2_scores.append(r2)
        edit_distances.append(ed)

        # Record mean confidence for this sample
        mean_conf = float(confidence[0].mean().cpu().item())
        candidate_details.append({
            "sample_idx": i,
            "symbolic_equivalent": is_equiv,
            "r2": float(r2),
            "edit_distance": float(ed),
            "mean_confidence": mean_conf,
            "predicted_str": str(pred_expr) if pred_expr is not None else None,
            "true_str": str(true_expr) if true_expr is not None else None,
        })

    n_evaluated = min(len(ds), n_test_samples)
    valid_r2 = [r for r in r2_scores if r >= 0]

    return {
        "equation_id": eq.id,
        "equation_name": eq.name,
        "tier": eq.tier,
        "held_out": eq.held_out,
        "n_samples": n_evaluated,
        "symbolic_accuracy": sym_correct / max(n_evaluated, 1),
        "mean_r2": float(np.mean(valid_r2)) if valid_r2 else -1.0,
        "std_r2": float(np.std(valid_r2)) if len(valid_r2) > 1 else 0.0,
        "mean_edit_distance": float(np.mean(edit_distances)),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "candidates": candidate_details,
    }


def evaluate_held_out_with_ttft(
    model: PhysMDT,
    eq,
    device: torch.device,
    tokenizer: ExprTokenizer,
    n_test_samples: int = 5,
    n_points: int = 100,
    noise_level: float = 0.01,
    n_candidates: int = 8,
    n_steps: int = 64,
    ttft_steps: int = 128,
    ttft_lr: float = 1e-3,
    ttft_rank: int = 16,
    seed: int = 42,
):
    """Evaluate a held-out equation both zero-shot and with TTFT.

    For each test sample:
      1. Run zero-shot prediction via generate_candidates.
      2. Deep-copy the model, apply TTFT, then re-run generate_candidates.
      3. Report top-3 candidates with confidence (from the voting procedure,
         the top-3 distinct predictions across samples).

    Returns a dict of per-equation results including both zero-shot and TTFT
    outcomes.
    """
    n_vars = len(eq.variables)

    # Generate test observations
    ds = PhysicsDataset(
        equations=[eq],
        n_samples=n_test_samples,
        n_points=n_points,
        noise_level=noise_level,
        seed=seed,
    )

    if len(ds) == 0:
        return {
            "equation_id": eq.id,
            "equation_name": eq.name,
            "tier": eq.tier,
            "held_out": eq.held_out,
            "n_samples": 0,
            "zero_shot": {
                "symbolic_accuracy": 0.0,
                "mean_r2": -1.0,
                "mean_edit_distance": 1.0,
            },
            "ttft": {
                "symbolic_accuracy": 0.0,
                "mean_r2": -1.0,
                "mean_edit_distance": 1.0,
            },
            "top3_candidates": [],
        }

    seq_len = model.config.max_expr_len

    # --- Zero-shot evaluation ---
    zs_sym_correct = 0
    zs_r2_scores = []
    zs_edit_distances = []
    zs_latencies = []
    zs_details = []

    for i in range(min(len(ds), n_test_samples)):
        sample = ds[i]
        obs = sample["observations"].unsqueeze(0).to(device)
        obs_mask = sample["obs_mask"].unsqueeze(0).to(device)
        true_token_ids = sample["tokens"][: sample["token_len"]].tolist()

        start_time = time.time()
        with torch.no_grad():
            pred_tokens, confidence = generate_candidates(
                model=model,
                observations=obs,
                obs_mask=obs_mask,
                seq_len=seq_len,
                n_steps=n_steps,
                n_candidates=n_candidates,
                temperature=1.0,
                device=device,
            )
        latency_ms = (time.time() - start_time) * 1000.0
        zs_latencies.append(latency_ms)

        gen_ids = pred_tokens[0].cpu().tolist()
        pred_expr = _decode_tokens_to_expr(gen_ids, tokenizer)
        true_expr = _decode_tokens_to_expr(true_token_ids, tokenizer)

        is_equiv = False
        r2 = -1.0
        ed = 1.0
        if pred_expr is not None and true_expr is not None:
            try:
                is_equiv = symbolic_equivalence(pred_expr, true_expr)
            except Exception:
                is_equiv = False
            try:
                r2 = numeric_r2(pred_expr, true_expr, n_points=200)
            except Exception:
                r2 = -1.0
            try:
                ed = token_edit_distance(gen_ids, true_token_ids)
            except Exception:
                ed = 1.0

        if is_equiv:
            zs_sym_correct += 1
        zs_r2_scores.append(r2)
        zs_edit_distances.append(ed)
        zs_details.append({
            "sample_idx": i,
            "symbolic_equivalent": is_equiv,
            "r2": float(r2),
            "edit_distance": float(ed),
            "mean_confidence": float(confidence[0].mean().cpu().item()),
            "predicted_str": str(pred_expr) if pred_expr is not None else None,
            "true_str": str(true_expr) if true_expr is not None else None,
        })

    # --- TTFT evaluation ---
    # Use the first sample's observations to fine-tune (the adaptation data).
    # Then re-evaluate on all samples.
    ttft_sym_correct = 0
    ttft_r2_scores = []
    ttft_edit_distances = []
    ttft_latencies = []
    ttft_details = []

    # Prepare adaptation data from the first sample
    adapt_sample = ds[0]
    adapt_obs = adapt_sample["observations"].unsqueeze(0).to(device)
    adapt_obs_mask = adapt_sample["obs_mask"].unsqueeze(0).to(device)

    # Deep-copy model so TTFT does not pollute the base model
    ttft_model = copy.deepcopy(model)
    try:
        ttft_model = test_time_finetune(
            model=ttft_model,
            observations=adapt_obs,
            obs_mask=adapt_obs_mask,
            n_steps=ttft_steps,
            lr=ttft_lr,
            rank=ttft_rank,
            n_vars=n_vars,
            verbose=False,
        )
    except Exception as exc:
        print(f"    [WARN] TTFT failed for {eq.name}: {exc}")
        # Fall back to zero-shot results for TTFT
        ttft_model = model

    for i in range(min(len(ds), n_test_samples)):
        sample = ds[i]
        obs = sample["observations"].unsqueeze(0).to(device)
        obs_mask = sample["obs_mask"].unsqueeze(0).to(device)
        true_token_ids = sample["tokens"][: sample["token_len"]].tolist()

        start_time = time.time()
        with torch.no_grad():
            pred_tokens, confidence = generate_candidates(
                model=ttft_model,
                observations=obs,
                obs_mask=obs_mask,
                seq_len=seq_len,
                n_steps=n_steps,
                n_candidates=n_candidates,
                temperature=1.0,
                device=device,
            )
        latency_ms = (time.time() - start_time) * 1000.0
        ttft_latencies.append(latency_ms)

        gen_ids = pred_tokens[0].cpu().tolist()
        pred_expr = _decode_tokens_to_expr(gen_ids, tokenizer)
        true_expr = _decode_tokens_to_expr(true_token_ids, tokenizer)

        is_equiv = False
        r2 = -1.0
        ed = 1.0
        if pred_expr is not None and true_expr is not None:
            try:
                is_equiv = symbolic_equivalence(pred_expr, true_expr)
            except Exception:
                is_equiv = False
            try:
                r2 = numeric_r2(pred_expr, true_expr, n_points=200)
            except Exception:
                r2 = -1.0
            try:
                ed = token_edit_distance(gen_ids, true_token_ids)
            except Exception:
                ed = 1.0

        if is_equiv:
            ttft_sym_correct += 1
        ttft_r2_scores.append(r2)
        ttft_edit_distances.append(ed)
        ttft_details.append({
            "sample_idx": i,
            "symbolic_equivalent": is_equiv,
            "r2": float(r2),
            "edit_distance": float(ed),
            "mean_confidence": float(confidence[0].mean().cpu().item()),
            "predicted_str": str(pred_expr) if pred_expr is not None else None,
            "true_str": str(true_expr) if true_expr is not None else None,
        })

    # Clean up TTFT model to free memory
    try:
        remove_lora(ttft_model)
    except Exception:
        pass
    del ttft_model

    n_evaluated = min(len(ds), n_test_samples)
    zs_valid_r2 = [r for r in zs_r2_scores if r >= 0]
    ttft_valid_r2 = [r for r in ttft_r2_scores if r >= 0]

    # --- Build top-3 candidates ---
    # Collect all distinct predicted expressions from both zero-shot and TTFT,
    # sorted by R^2 descending.
    all_preds = []
    for detail_list, method in [(zs_details, "zero_shot"), (ttft_details, "ttft")]:
        for d in detail_list:
            if d["predicted_str"] is not None:
                all_preds.append({
                    "expression": d["predicted_str"],
                    "r2": d["r2"],
                    "confidence": d["mean_confidence"],
                    "method": method,
                    "symbolic_equivalent": d["symbolic_equivalent"],
                })

    # De-duplicate by expression string, keeping the best R^2
    seen = {}
    for p in all_preds:
        key = p["expression"]
        if key not in seen or p["r2"] > seen[key]["r2"]:
            seen[key] = p
    top3 = sorted(seen.values(), key=lambda x: x["r2"], reverse=True)[:3]

    return {
        "equation_id": eq.id,
        "equation_name": eq.name,
        "tier": eq.tier,
        "held_out": eq.held_out,
        "description": eq.description,
        "true_expression": str(eq.symbolic_expr),
        "n_samples": n_evaluated,
        "zero_shot": {
            "symbolic_accuracy": zs_sym_correct / max(n_evaluated, 1),
            "mean_r2": float(np.mean(zs_valid_r2)) if zs_valid_r2 else -1.0,
            "std_r2": float(np.std(zs_valid_r2)) if len(zs_valid_r2) > 1 else 0.0,
            "mean_edit_distance": float(np.mean(zs_edit_distances)),
            "mean_latency_ms": float(np.mean(zs_latencies)) if zs_latencies else 0.0,
            "details": zs_details,
        },
        "ttft": {
            "symbolic_accuracy": ttft_sym_correct / max(n_evaluated, 1),
            "mean_r2": float(np.mean(ttft_valid_r2)) if ttft_valid_r2 else -1.0,
            "std_r2": float(np.std(ttft_valid_r2)) if len(ttft_valid_r2) > 1 else 0.0,
            "mean_edit_distance": float(np.mean(ttft_edit_distances)),
            "mean_latency_ms": float(np.mean(ttft_latencies)) if ttft_latencies else 0.0,
            "details": ttft_details,
        },
        "top3_candidates": top3,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_tier_results(per_equation_results):
    """Aggregate per-equation results into per-tier summary statistics."""
    tier_data = {}
    for r in per_equation_results:
        t = r["tier"]
        if t not in tier_data:
            tier_data[t] = {
                "sym_accs": [],
                "r2s": [],
                "edit_dists": [],
                "latencies": [],
                "n_equations": 0,
            }
        tier_data[t]["sym_accs"].append(r["symbolic_accuracy"])
        tier_data[t]["r2s"].append(r["mean_r2"])
        tier_data[t]["edit_dists"].append(r["mean_edit_distance"])
        tier_data[t]["latencies"].append(r["mean_latency_ms"])
        tier_data[t]["n_equations"] += 1

    tier_summary = {}
    for t in sorted(tier_data.keys()):
        d = tier_data[t]
        valid_r2 = [x for x in d["r2s"] if x >= 0]
        tier_summary[str(t)] = {
            "n_equations": d["n_equations"],
            "symbolic_accuracy": float(np.mean(d["sym_accs"])),
            "mean_r2": float(np.mean(valid_r2)) if valid_r2 else -1.0,
            "std_r2": float(np.std(valid_r2)) if len(valid_r2) > 1 else 0.0,
            "min_r2": float(np.min(valid_r2)) if valid_r2 else -1.0,
            "max_r2": float(np.max(valid_r2)) if valid_r2 else -1.0,
            "mean_edit_distance": float(np.mean(d["edit_dists"])),
            "mean_latency_ms": float(np.mean(d["latencies"])),
        }

    return tier_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Quick mode: {args.quick}")
    print(f"Seed: {args.seed}")
    print()

    # ---- Load model -------------------------------------------------------
    model, config = _load_checkpoint(args.checkpoint, device)
    if model is None:
        sys.exit(1)

    n_params = model.count_parameters()
    print(f"Model loaded: {n_params:,} parameters ({n_params / 1e6:.1f}M)")
    print()

    tokenizer = ExprTokenizer()

    # ---- Quick-mode parameter overrides -----------------------------------
    if args.quick:
        n_test_samples = 2
        n_points = 30
        n_candidates = 2
        n_steps = 8
        ttft_steps = 8
        print("[Quick mode] Using reduced parameters for smoke testing.")
    else:
        n_test_samples = 5
        n_points = 100
        n_candidates = 8
        n_steps = 64
        ttft_steps = 128

    noise_level = 0.01

    # ====================================================================
    # Part 1: In-distribution evaluation (training equations, by tier)
    # ====================================================================
    print("=" * 70)
    print("PART 1: In-Distribution Evaluation (Training Equations)")
    print("=" * 70)

    training_equations = get_training_equations()
    print(f"Total training equations: {len(training_equations)}")
    print()

    in_dist_results = []
    for tier in range(1, 6):
        tier_eqs = [eq for eq in training_equations if eq.tier == tier]
        if not tier_eqs:
            continue
        print(f"--- Tier {tier} ({len(tier_eqs)} equations) ---")
        for eq in tier_eqs:
            print(f"  Evaluating: {eq.id} - {eq.name} ...", end=" ", flush=True)
            result = evaluate_equation(
                model=model,
                eq=eq,
                device=device,
                tokenizer=tokenizer,
                n_test_samples=n_test_samples,
                n_points=n_points,
                noise_level=noise_level,
                n_candidates=n_candidates,
                n_steps=n_steps,
                seed=args.seed,
            )
            in_dist_results.append(result)
            acc_pct = result["symbolic_accuracy"] * 100
            r2_val = result["mean_r2"]
            lat = result["mean_latency_ms"]
            print(f"Acc={acc_pct:.0f}%  R2={r2_val:.4f}  Latency={lat:.0f}ms")
        print()

    # Aggregate tier results
    tier_summary = _aggregate_tier_results(in_dist_results)

    # Overall statistics
    all_sym_accs = [r["symbolic_accuracy"] for r in in_dist_results]
    all_r2s = [r["mean_r2"] for r in in_dist_results if r["mean_r2"] >= 0]
    all_edit = [r["mean_edit_distance"] for r in in_dist_results]
    all_lat = [r["mean_latency_ms"] for r in in_dist_results if r["mean_latency_ms"] > 0]

    overall_sym_acc = float(np.mean(all_sym_accs)) if all_sym_accs else 0.0
    overall_r2 = float(np.mean(all_r2s)) if all_r2s else -1.0
    overall_edit = float(np.mean(all_edit)) if all_edit else 1.0
    overall_lat = float(np.mean(all_lat)) if all_lat else 0.0
    gpu_mem_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )

    print("Per-Tier Summary (PhysMDT):")
    for t in sorted(tier_summary.keys()):
        ts = tier_summary[t]
        print(
            f"  Tier {t}: Acc={ts['symbolic_accuracy']*100:.1f}%  "
            f"R2={ts['mean_r2']:.4f} (+/- {ts['std_r2']:.4f})  "
            f"EditDist={ts['mean_edit_distance']:.3f}  "
            f"Latency={ts['mean_latency_ms']:.0f}ms"
        )
    print(
        f"\nOverall: Acc={overall_sym_acc*100:.1f}%  R2={overall_r2:.4f}  "
        f"EditDist={overall_edit:.3f}  Latency={overall_lat:.0f}ms"
    )
    print(f"Peak GPU memory: {gpu_mem_gb:.2f} GB")
    print()

    # Save in-distribution results
    in_dist_output = {
        "model": "PhysMDT",
        "checkpoint": args.checkpoint,
        "total_params": n_params,
        "device": str(device),
        "seed": args.seed,
        "quick_mode": args.quick,
        "evaluation_config": {
            "n_test_samples": n_test_samples,
            "n_points": n_points,
            "noise_level": noise_level,
            "n_candidates": n_candidates,
            "n_steps": n_steps,
        },
        "overall": {
            "symbolic_accuracy": overall_sym_acc,
            "mean_r2": overall_r2,
            "mean_edit_distance": overall_edit,
            "mean_latency_ms": overall_lat,
            "peak_gpu_memory_gb": float(gpu_mem_gb),
        },
        "tier_results": tier_summary,
        "per_equation": in_dist_results,
    }

    os.makedirs("results", exist_ok=True)
    in_dist_path = "results/in_distribution_comparison.json"
    with open(in_dist_path, "w") as f:
        json.dump(in_dist_output, f, indent=2, default=str)
    print(f"Saved in-distribution results to {in_dist_path}")

    # ====================================================================
    # Part 2: Held-out (zero-shot discovery) evaluation
    # ====================================================================
    print()
    print("=" * 70)
    print("PART 2: Zero-Shot Discovery Evaluation (Held-Out Equations)")
    print("=" * 70)

    held_out_equations = get_held_out_equations()
    print(f"Total held-out equations: {len(held_out_equations)}")
    print()

    held_out_results = []
    for eq in held_out_equations:
        print(f"  Evaluating: {eq.id} - {eq.name} ...", flush=True)
        result = evaluate_held_out_with_ttft(
            model=model,
            eq=eq,
            device=device,
            tokenizer=tokenizer,
            n_test_samples=n_test_samples,
            n_points=n_points,
            noise_level=noise_level,
            n_candidates=n_candidates,
            n_steps=n_steps,
            ttft_steps=ttft_steps,
            ttft_lr=1e-3,
            ttft_rank=16,
            seed=args.seed,
        )
        held_out_results.append(result)

        zs = result["zero_shot"]
        tt = result["ttft"]
        print(
            f"    Zero-shot: Acc={zs['symbolic_accuracy']*100:.0f}%  "
            f"R2={zs['mean_r2']:.4f}"
        )
        print(
            f"    TTFT:      Acc={tt['symbolic_accuracy']*100:.0f}%  "
            f"R2={tt['mean_r2']:.4f}"
        )
        if result["top3_candidates"]:
            print("    Top-3 candidates:")
            for rank, cand in enumerate(result["top3_candidates"], 1):
                print(
                    f"      #{rank}: {cand['expression']}  "
                    f"(R2={cand['r2']:.4f}, conf={cand['confidence']:.3f}, "
                    f"method={cand['method']}, equiv={cand['symbolic_equivalent']})"
                )
        print()

    # Aggregate held-out statistics
    zs_accs = [r["zero_shot"]["symbolic_accuracy"] for r in held_out_results]
    tt_accs = [r["ttft"]["symbolic_accuracy"] for r in held_out_results]
    zs_r2s = [r["zero_shot"]["mean_r2"] for r in held_out_results if r["zero_shot"]["mean_r2"] >= 0]
    tt_r2s = [r["ttft"]["mean_r2"] for r in held_out_results if r["ttft"]["mean_r2"] >= 0]

    zs_overall_acc = float(np.mean(zs_accs)) if zs_accs else 0.0
    tt_overall_acc = float(np.mean(tt_accs)) if tt_accs else 0.0
    zs_overall_r2 = float(np.mean(zs_r2s)) if zs_r2s else -1.0
    tt_overall_r2 = float(np.mean(tt_r2s)) if tt_r2s else -1.0

    # Novel discovery rate: fraction of held-out equations with at least one
    # symbolically correct prediction (across either method)
    n_discovered_zs = sum(
        1 for r in held_out_results if r["zero_shot"]["symbolic_accuracy"] > 0
    )
    n_discovered_ttft = sum(
        1 for r in held_out_results if r["ttft"]["symbolic_accuracy"] > 0
    )
    n_discovered_any = sum(
        1
        for r in held_out_results
        if r["zero_shot"]["symbolic_accuracy"] > 0
        or r["ttft"]["symbolic_accuracy"] > 0
    )
    n_held_out = len(held_out_results)

    print("Held-Out Summary:")
    print(
        f"  Zero-shot: Acc={zs_overall_acc*100:.1f}%  R2={zs_overall_r2:.4f}  "
        f"Discovered={n_discovered_zs}/{n_held_out}"
    )
    print(
        f"  TTFT:      Acc={tt_overall_acc*100:.1f}%  R2={tt_overall_r2:.4f}  "
        f"Discovered={n_discovered_ttft}/{n_held_out}"
    )
    print(
        f"  Either:    Discovered={n_discovered_any}/{n_held_out}  "
        f"({n_discovered_any / max(n_held_out, 1) * 100:.1f}%)"
    )
    print()

    # Save held-out results
    held_out_output = {
        "model": "PhysMDT",
        "checkpoint": args.checkpoint,
        "total_params": n_params,
        "device": str(device),
        "seed": args.seed,
        "quick_mode": args.quick,
        "evaluation_config": {
            "n_test_samples": n_test_samples,
            "n_points": n_points,
            "noise_level": noise_level,
            "n_candidates": n_candidates,
            "n_steps": n_steps,
            "ttft_steps": ttft_steps,
            "ttft_lr": 1e-3,
            "ttft_rank": 16,
        },
        "summary": {
            "n_held_out_equations": n_held_out,
            "zero_shot": {
                "symbolic_accuracy": zs_overall_acc,
                "mean_r2": zs_overall_r2,
                "n_discovered": n_discovered_zs,
                "discovery_rate": n_discovered_zs / max(n_held_out, 1),
            },
            "ttft": {
                "symbolic_accuracy": tt_overall_acc,
                "mean_r2": tt_overall_r2,
                "n_discovered": n_discovered_ttft,
                "discovery_rate": n_discovered_ttft / max(n_held_out, 1),
            },
            "either_method": {
                "n_discovered": n_discovered_any,
                "discovery_rate": n_discovered_any / max(n_held_out, 1),
            },
        },
        "per_equation": held_out_results,
    }

    held_out_path = "results/zero_shot_discovery.json"
    with open(held_out_path, "w") as f:
        json.dump(held_out_output, f, indent=2, default=str)
    print(f"Saved held-out results to {held_out_path}")

    # ---- Final summary ----------------------------------------------------
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Model:         PhysMDT ({n_params:,} params)")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  In-dist Acc:   {overall_sym_acc*100:.1f}%")
    print(f"  In-dist R2:    {overall_r2:.4f}")
    print(f"  Held-out (ZS): {zs_overall_acc*100:.1f}% acc, "
          f"{n_discovered_zs}/{n_held_out} discovered")
    print(f"  Held-out (TT): {tt_overall_acc*100:.1f}% acc, "
          f"{n_discovered_ttft}/{n_held_out} discovered")
    print(f"  Results:       {in_dist_path}")
    print(f"                 {held_out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
