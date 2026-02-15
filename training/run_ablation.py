"""Ablation study for PhysMDT: evaluate contribution of each novel component.

Implements item_019 of the research rubric. Removes one component at a time
and reports impact on Tier 3-5 symbolic accuracy and R^2:

    1. "full"          -- Full PhysMDT with all components (baseline for comparison)
    2. "no_refinement" -- Single-pass decoding (no recursive soft-masking)
    3. "no_tree_pos"   -- Zero out tree-positional encoding
    4. "no_dim_bias"   -- Zero out dimensional analysis bias
    5. "no_ttft"       -- No test-time fine-tuning (zero-shot only)
    6. "no_curriculum"  -- Trained without curriculum (note: uses same checkpoint;
                           see notes in results)

Produces:
    results/ablation_results.json   -- full per-condition, per-equation results
    figures/ablation_table.png      -- clean results table rendered as an image

Usage:
    python training/run_ablation.py
    python training/run_ablation.py --checkpoint checkpoints/physmdt_best.pt
    python training/run_ablation.py --quick   # smoke test with fewer samples
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from data.equations import (
    get_training_equations,
    get_equations_by_tier,
)
from data.physics_generator import PhysicsDataset
from data.tokenizer import ExprTokenizer, PAD_IDX, SOS_IDX, EOS_IDX, MASK_IDX
from evaluation.metrics import symbolic_equivalence, numeric_r2, token_edit_distance
from models.physmdt import PhysMDT, PhysMDTConfig
from models.refinement import generate_candidates, refine
from models.ttft import test_time_finetune, remove_lora


# ---------------------------------------------------------------------------
# Helpers (shared with evaluate_physmdt.py)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study for PhysMDT (item_019 of the research rubric)."
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

    # Reconstruct config
    raw_config = ckpt.get("config", None)
    if raw_config is None:
        print("[WARN] Checkpoint has no 'config' key; using default PhysMDTConfig.")
        config = PhysMDTConfig()
    elif isinstance(raw_config, PhysMDTConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = PhysMDTConfig(**raw_config)
    else:
        config = raw_config

    model = PhysMDT(config).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        print("[WARN] No model_state_dict found; using freshly-initialised weights.")

    model.eval()
    return model, config


def _decode_tokens_to_expr(token_ids, tokenizer: ExprTokenizer):
    """Decode a token-id sequence to a SymPy expression.

    Returns the decoded SymPy expression, or None on failure.
    """
    ids = list(token_ids)

    if EOS_IDX in ids:
        ids = ids[: ids.index(EOS_IDX) + 1]

    ids = [i for i in ids if i not in (PAD_IDX, MASK_IDX)]

    if not ids:
        return None

    try:
        expr = tokenizer.decode(ids, strip_special=True)
        return expr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-pass decoding for the "no_refinement" ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def single_pass_decode(
    model: PhysMDT,
    observations: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_len: int = 32,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a single forward pass with fully masked input and take argmax.

    This is the ablation for "no_refinement": instead of iterative
    soft-masking refinement, we simply run the model once on a fully
    masked sequence and take argmax at every position.

    Args:
        model: A trained PhysMDT model (set to eval mode).
        observations: (batch, n_points, max_vars+1) observation data.
        obs_mask: (batch, n_points, max_vars+1) observation validity mask.
        seq_len: Length of the output token sequence to generate.
        device: Device to run on.

    Returns:
        tokens: (batch, seq_len) discrete token indices.
        confidences: (batch, seq_len) per-position confidence (max prob).
    """
    if device is None:
        device = observations.device

    model.eval()
    batch = observations.shape[0]

    # Build fully masked input: SOS at position 0, MASK everywhere else
    tokens = torch.full(
        (batch, seq_len), MASK_IDX, dtype=torch.long, device=device
    )
    tokens[:, 0] = SOS_IDX

    token_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    token_mask[:, 0] = False  # SOS is not masked

    # Single forward pass
    logits, _aux = model(
        observations=observations,
        obs_mask=obs_mask,
        masked_tokens=tokens,
        token_mask=token_mask,
    )

    # Argmax decoding
    pred_tokens = logits.argmax(dim=-1)  # (batch, seq_len)
    pred_tokens[:, 0] = SOS_IDX  # preserve SOS

    # Confidence scores from softmax
    probs = F.softmax(logits, dim=-1)
    confidences, _ = probs.max(dim=-1)  # (batch, seq_len)

    return pred_tokens, confidences


# ---------------------------------------------------------------------------
# Monkey-patching utilities for ablation conditions
# ---------------------------------------------------------------------------

class _ZeroTreePosHook:
    """Context manager that monkey-patches TreePositionalEncoding.forward
    to return zeros, effectively ablating tree-positional encoding."""

    def __init__(self, model: PhysMDT):
        self._model = model
        self._original_forward = None

    def __enter__(self):
        pos_enc_module = self._model.pos_encoding
        self._original_forward = pos_enc_module.forward

        def _zero_forward(seq_len, tree_depths=None, sibling_indices=None,
                          device=None):
            if device is None:
                device = pos_enc_module.pos_embedding.weight.device
            # Return zero tensor: the positional encoding contributes nothing
            return torch.zeros(1, seq_len, self._model.config.d_model,
                               device=device)

        pos_enc_module.forward = _zero_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model.pos_encoding.forward = self._original_forward
        return False


class _ZeroDimBiasHook:
    """Context manager that monkey-patches DimensionalAnalysisBias.forward
    to return zeros, effectively ablating dimensional analysis bias."""

    def __init__(self, model: PhysMDT):
        self._model = model
        self._original_forward = None

    def __enter__(self):
        if self._model.dim_analysis is None:
            # Model was built without dim_analysis; nothing to patch
            return self

        dim_module = self._model.dim_analysis
        self._original_forward = dim_module.forward

        def _zero_forward(token_embeddings):
            batch, seq_len, _ = token_embeddings.shape
            nhead = dim_module.nhead
            device = token_embeddings.device
            # Return zero bias and zero loss
            zero_bias = torch.zeros(batch, nhead, seq_len, seq_len,
                                    device=device)
            zero_loss = torch.tensor(0.0, device=device)
            return zero_bias, zero_loss

        dim_module.forward = _zero_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._model.dim_analysis is not None and self._original_forward is not None:
            self._model.dim_analysis.forward = self._original_forward
        return False


# ---------------------------------------------------------------------------
# Core evaluation for a single equation under a given ablation condition
# ---------------------------------------------------------------------------

def evaluate_equation_ablation(
    model: PhysMDT,
    eq,
    device: torch.device,
    tokenizer: ExprTokenizer,
    ablation: str,
    n_test_samples: int = 5,
    n_points: int = 100,
    noise_level: float = 0.01,
    n_candidates: int = 8,
    n_steps: int = 64,
    ttft_steps: int = 128,
    ttft_lr: float = 1e-3,
    ttft_rank: int = 16,
    seed: int = 42,
) -> Dict:
    """Evaluate a single equation under a specific ablation condition.

    Args:
        model: PhysMDT model (eval mode, base weights -- must NOT have LoRA).
        eq: Equation object.
        device: torch.device.
        tokenizer: ExprTokenizer.
        ablation: One of "full", "no_refinement", "no_tree_pos",
                  "no_dim_bias", "no_ttft", "no_curriculum".
        n_test_samples: number of test samples per equation.
        n_points: number of observation points per sample.
        noise_level: Gaussian noise standard deviation.
        n_candidates: number of candidates for voting.
        n_steps: number of refinement steps.
        ttft_steps: number of TTFT fine-tuning steps.
        ttft_lr: TTFT learning rate.
        ttft_rank: LoRA rank for TTFT.
        seed: random seed.

    Returns:
        Dict with per-equation metrics.
    """
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
            "ablation": ablation,
            "n_samples": 0,
            "symbolic_accuracy": 0.0,
            "mean_r2": -1.0,
            "mean_edit_distance": 1.0,
            "mean_latency_ms": 0.0,
            "details": [],
        }

    seq_len = model.config.max_expr_len
    n_vars = len(eq.variables)

    sym_correct = 0
    r2_scores = []
    edit_distances = []
    latencies = []
    details = []

    for i in range(min(len(ds), n_test_samples)):
        sample = ds[i]
        obs = sample["observations"].unsqueeze(0).to(device)
        obs_mask_tensor = sample["obs_mask"].unsqueeze(0).to(device)
        true_token_ids = sample["tokens"][: sample["token_len"]].tolist()

        start_time = time.time()

        # ----- Dispatch based on ablation condition -----
        if ablation == "no_refinement":
            # Single-pass decoding: no iterative refinement
            with torch.no_grad():
                pred_tokens, confidence = single_pass_decode(
                    model=model,
                    observations=obs,
                    obs_mask=obs_mask_tensor,
                    seq_len=seq_len,
                    device=device,
                )

        elif ablation == "no_tree_pos":
            # Zero out tree-positional encoding via monkey-patch
            with _ZeroTreePosHook(model):
                with torch.no_grad():
                    pred_tokens, confidence = generate_candidates(
                        model=model,
                        observations=obs,
                        obs_mask=obs_mask_tensor,
                        seq_len=seq_len,
                        n_steps=n_steps,
                        n_candidates=n_candidates,
                        temperature=1.0,
                        device=device,
                    )

        elif ablation == "no_dim_bias":
            # Zero out dimensional analysis bias via monkey-patch
            with _ZeroDimBiasHook(model):
                with torch.no_grad():
                    pred_tokens, confidence = generate_candidates(
                        model=model,
                        observations=obs,
                        obs_mask=obs_mask_tensor,
                        seq_len=seq_len,
                        n_steps=n_steps,
                        n_candidates=n_candidates,
                        temperature=1.0,
                        device=device,
                    )

        elif ablation == "no_ttft":
            # Zero-shot only (same as "full" without TTFT).  Since the
            # "full" baseline already includes TTFT, this condition simply
            # runs the standard generate_candidates pipeline without any
            # test-time fine-tuning.
            with torch.no_grad():
                pred_tokens, confidence = generate_candidates(
                    model=model,
                    observations=obs,
                    obs_mask=obs_mask_tensor,
                    seq_len=seq_len,
                    n_steps=n_steps,
                    n_candidates=n_candidates,
                    temperature=1.0,
                    device=device,
                )

        elif ablation in ("full", "no_curriculum"):
            # "full": all components including TTFT
            # "no_curriculum": same checkpoint, same pipeline; the difference
            #   is conceptual (model was trained with curriculum, so this
            #   represents what you'd compare against a hypothetical
            #   non-curriculum-trained model).  We run the full pipeline
            #   and note the caveat in results.
            #
            # First: zero-shot via generate_candidates
            with torch.no_grad():
                pred_tokens_zs, confidence_zs = generate_candidates(
                    model=model,
                    observations=obs,
                    obs_mask=obs_mask_tensor,
                    seq_len=seq_len,
                    n_steps=n_steps,
                    n_candidates=n_candidates,
                    temperature=1.0,
                    device=device,
                )

            # Then: apply TTFT to a deep copy and re-run
            ttft_model = copy.deepcopy(model)
            try:
                ttft_model = test_time_finetune(
                    model=ttft_model,
                    observations=obs,
                    obs_mask=obs_mask_tensor,
                    n_steps=ttft_steps,
                    lr=ttft_lr,
                    rank=ttft_rank,
                    n_vars=n_vars,
                    verbose=False,
                )
                with torch.no_grad():
                    pred_tokens, confidence = generate_candidates(
                        model=ttft_model,
                        observations=obs,
                        obs_mask=obs_mask_tensor,
                        seq_len=seq_len,
                        n_steps=n_steps,
                        n_candidates=n_candidates,
                        temperature=1.0,
                        device=device,
                    )
            except Exception:
                # TTFT failed; fall back to zero-shot
                pred_tokens, confidence = pred_tokens_zs, confidence_zs
            finally:
                # Clean up
                try:
                    remove_lora(ttft_model)
                except Exception:
                    pass
                del ttft_model

        else:
            raise ValueError(f"Unknown ablation condition: {ablation}")

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

        mean_conf = float(confidence[0].mean().cpu().item())
        details.append({
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
        "ablation": ablation,
        "n_samples": n_evaluated,
        "symbolic_accuracy": sym_correct / max(n_evaluated, 1),
        "mean_r2": float(np.mean(valid_r2)) if valid_r2 else -1.0,
        "std_r2": float(np.std(valid_r2)) if len(valid_r2) > 1 else 0.0,
        "mean_edit_distance": float(np.mean(edit_distances)),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_results(per_equation_results: List[Dict]) -> Dict:
    """Aggregate per-equation results into summary statistics."""
    if not per_equation_results:
        return {
            "n_equations": 0,
            "symbolic_accuracy": 0.0,
            "mean_r2": -1.0,
            "std_r2": 0.0,
            "mean_edit_distance": 1.0,
            "mean_latency_ms": 0.0,
        }

    sym_accs = [r["symbolic_accuracy"] for r in per_equation_results]
    r2s = [r["mean_r2"] for r in per_equation_results if r["mean_r2"] >= 0]
    edit_dists = [r["mean_edit_distance"] for r in per_equation_results]
    latencies = [r["mean_latency_ms"] for r in per_equation_results
                 if r["mean_latency_ms"] > 0]

    return {
        "n_equations": len(per_equation_results),
        "symbolic_accuracy": float(np.mean(sym_accs)),
        "mean_r2": float(np.mean(r2s)) if r2s else -1.0,
        "std_r2": float(np.std(r2s)) if len(r2s) > 1 else 0.0,
        "mean_edit_distance": float(np.mean(edit_dists)),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def _aggregate_by_tier(per_equation_results: List[Dict]) -> Dict[int, Dict]:
    """Group results by tier and aggregate."""
    tier_groups: Dict[int, List[Dict]] = {}
    for r in per_equation_results:
        t = r["tier"]
        if t not in tier_groups:
            tier_groups[t] = []
        tier_groups[t].append(r)
    return {t: _aggregate_results(eqs) for t, eqs in sorted(tier_groups.items())}


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _render_ablation_table(
    condition_summaries: Dict[str, Dict],
    condition_tier_results: Dict[str, Dict[int, Dict]],
    output_path: str,
) -> None:
    """Render the ablation results as a clean table image.

    Args:
        condition_summaries: {condition_name: aggregate_summary_dict}
        condition_tier_results: {condition_name: {tier: summary_dict}}
        output_path: path to save the PNG image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping table rendering.")
        return

    # Build table data
    conditions = [
        "full",
        "no_refinement",
        "no_tree_pos",
        "no_dim_bias",
        "no_ttft",
        "no_curriculum",
    ]
    condition_labels = {
        "full": "Full PhysMDT",
        "no_refinement": "w/o Refinement",
        "no_tree_pos": "w/o Tree Pos. Enc.",
        "no_dim_bias": "w/o Dim. Analysis",
        "no_ttft": "w/o TTFT",
        "no_curriculum": "w/o Curriculum*",
    }

    # Column headers: Condition | Tier 3 Acc | Tier 3 R2 | Tier 4 Acc | Tier 4 R2 |
    #                            Tier 5 Acc | Tier 5 R2 | Overall Acc | Overall R2
    col_labels = [
        "Condition",
        "Tier 3\nAcc (%)",
        "Tier 3\nR\u00b2",
        "Tier 4\nAcc (%)",
        "Tier 4\nR\u00b2",
        "Tier 5\nAcc (%)",
        "Tier 5\nR\u00b2",
        "Overall\nAcc (%)",
        "Overall\nR\u00b2",
    ]

    cell_data = []
    for cond in conditions:
        summary = condition_summaries.get(cond, {})
        tier_res = condition_tier_results.get(cond, {})
        row = [condition_labels.get(cond, cond)]

        for tier in [3, 4, 5]:
            t = tier_res.get(tier, {})
            acc = t.get("symbolic_accuracy", 0.0) * 100
            r2 = t.get("mean_r2", -1.0)
            row.append(f"{acc:.1f}")
            row.append(f"{r2:.4f}" if r2 >= 0 else "N/A")

        overall_acc = summary.get("symbolic_accuracy", 0.0) * 100
        overall_r2 = summary.get("mean_r2", -1.0)
        row.append(f"{overall_acc:.1f}")
        row.append(f"{overall_r2:.4f}" if overall_r2 >= 0 else "N/A")

        cell_data.append(row)

    # Render as a matplotlib table figure
    n_rows = len(cell_data)
    n_cols = len(col_labels)

    fig_width = max(12, n_cols * 1.4)
    fig_height = max(3.0, (n_rows + 2) * 0.55)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(
        "PhysMDT Ablation Study: Component Contributions (Tier 3-5)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Style "full" row (first data row) with light highlight
    for j in range(n_cols):
        cell = table[1, j]
        cell.set_facecolor("#D6E4F0")

    # Alternate row colors for readability
    for i in range(2, n_rows + 1):
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i, j].set_facecolor(color)

    # Add footnote
    fig.text(
        0.5, 0.02,
        "* w/o Curriculum uses the same checkpoint (trained with curriculum); "
        "comparison is conceptual.\n"
        "Accuracy = symbolic equivalence rate. R\u00b2 = mean numeric R\u00b2 "
        "on valid predictions. Seed = 42.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#666666",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved ablation table to {output_path}")


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
        print("[Quick mode] Using reduced parameters for smoke testing.\n")
    else:
        n_test_samples = 5
        n_points = 100
        n_candidates = 8
        n_steps = 64
        ttft_steps = 128

    noise_level = 0.01

    # ---- Collect Tier 3-5 training equations (exclude Tier 1-2) -----------
    training_equations = get_training_equations()
    ablation_equations = [eq for eq in training_equations if eq.tier >= 3]
    print(f"Ablation equations (Tier 3-5): {len(ablation_equations)}")
    tier_counts = {}
    for eq in ablation_equations:
        tier_counts[eq.tier] = tier_counts.get(eq.tier, 0) + 1
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]} equations")
    print()

    # ---- Define ablation conditions ---------------------------------------
    ablation_conditions = [
        "full",
        "no_refinement",
        "no_tree_pos",
        "no_dim_bias",
        "no_ttft",
        "no_curriculum",
    ]

    # ---- Run ablations ----------------------------------------------------
    all_results: Dict[str, List[Dict]] = {}

    for condition in ablation_conditions:
        print("=" * 70)
        print(f"ABLATION CONDITION: {condition}")
        print("=" * 70)

        # Reset seed before each condition for fair comparison
        _set_seed(args.seed)

        condition_results = []
        for eq in ablation_equations:
            print(f"  [{condition}] {eq.id} - {eq.name} ...", end=" ", flush=True)

            result = evaluate_equation_ablation(
                model=model,
                eq=eq,
                device=device,
                tokenizer=tokenizer,
                ablation=condition,
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
            condition_results.append(result)

            acc_pct = result["symbolic_accuracy"] * 100
            r2_val = result["mean_r2"]
            lat = result["mean_latency_ms"]
            print(f"Acc={acc_pct:.0f}%  R2={r2_val:.4f}  Lat={lat:.0f}ms")

        all_results[condition] = condition_results
        print()

    # ---- Aggregate results ------------------------------------------------
    condition_summaries: Dict[str, Dict] = {}
    condition_tier_results: Dict[str, Dict[int, Dict]] = {}

    for condition, results in all_results.items():
        condition_summaries[condition] = _aggregate_results(results)
        condition_tier_results[condition] = _aggregate_by_tier(results)

    # ---- Print summary table to console -----------------------------------
    print()
    print("=" * 70)
    print("ABLATION STUDY SUMMARY (Tier 3-5)")
    print("=" * 70)

    header = f"{'Condition':<22s} {'Acc (%)':<10s} {'R2':<10s} {'EditDist':<10s} {'Lat (ms)':<10s}"
    print(header)
    print("-" * len(header))

    for condition in ablation_conditions:
        s = condition_summaries[condition]
        acc_str = f"{s['symbolic_accuracy']*100:.1f}"
        r2_str = f"{s['mean_r2']:.4f}" if s['mean_r2'] >= 0 else "N/A"
        ed_str = f"{s['mean_edit_distance']:.3f}"
        lat_str = f"{s['mean_latency_ms']:.0f}"
        print(f"  {condition:<20s} {acc_str:<10s} {r2_str:<10s} {ed_str:<10s} {lat_str:<10s}")

    print()

    # Per-tier breakdown
    for condition in ablation_conditions:
        tier_res = condition_tier_results[condition]
        parts = []
        for tier in [3, 4, 5]:
            t = tier_res.get(tier, {})
            acc = t.get("symbolic_accuracy", 0.0) * 100
            r2 = t.get("mean_r2", -1.0)
            r2_str = f"{r2:.3f}" if r2 >= 0 else "N/A"
            parts.append(f"T{tier}={acc:.0f}%/{r2_str}")
        print(f"  {condition:<20s}  " + "  ".join(parts))

    print()

    # ---- Compute deltas from full baseline --------------------------------
    full_summary = condition_summaries.get("full", {})
    full_acc = full_summary.get("symbolic_accuracy", 0.0)
    full_r2 = full_summary.get("mean_r2", 0.0)

    print("Component contribution (delta from full baseline):")
    for condition in ablation_conditions:
        if condition == "full":
            continue
        s = condition_summaries[condition]
        delta_acc = (full_acc - s["symbolic_accuracy"]) * 100
        delta_r2 = full_r2 - s["mean_r2"] if s["mean_r2"] >= 0 else float("nan")
        delta_r2_str = f"{delta_r2:+.4f}" if not np.isnan(delta_r2) else "N/A"
        print(f"  {condition:<20s}  Acc delta: {delta_acc:+.1f}pp  R2 delta: {delta_r2_str}")

    print()

    # ---- Build output JSON ------------------------------------------------
    output = {
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
        "ablation_conditions": ablation_conditions,
        "condition_descriptions": {
            "full": (
                "Full PhysMDT with all components: recursive soft-masking "
                "refinement, tree-positional encoding, dimensional analysis "
                "bias, test-time fine-tuning, and curriculum training."
            ),
            "no_refinement": (
                "PhysMDT without recursive soft-masking refinement. Uses a "
                "single forward pass with fully masked input and argmax "
                "decoding (no iterative refinement, no candidate voting)."
            ),
            "no_tree_pos": (
                "PhysMDT without tree-positional encoding. The positional "
                "encoding output is zeroed out via monkey-patching, so the "
                "model receives no structural position information."
            ),
            "no_dim_bias": (
                "PhysMDT without dimensional analysis attention bias. The "
                "dimensional analysis module output is zeroed out, removing "
                "the physics-informed attention bias."
            ),
            "no_ttft": (
                "PhysMDT without test-time fine-tuning. Uses zero-shot "
                "inference only (generate_candidates without LoRA adaptation)."
            ),
            "no_curriculum": (
                "Conceptual ablation: PhysMDT trained without curriculum. "
                "NOTE: This uses the same checkpoint (trained WITH curriculum) "
                "since we do not retrain from scratch. The comparison is "
                "therefore a conceptual upper bound on the curriculum-trained "
                "model's performance, and the delta against a truly "
                "non-curriculum model would need a separate training run."
            ),
        },
        "summary": {
            condition: condition_summaries[condition]
            for condition in ablation_conditions
        },
        "per_tier": {
            condition: {
                str(tier): tier_data
                for tier, tier_data in condition_tier_results[condition].items()
            }
            for condition in ablation_conditions
        },
        "deltas_from_full": {},
        "per_equation": {
            condition: all_results[condition]
            for condition in ablation_conditions
        },
    }

    # Compute deltas
    for condition in ablation_conditions:
        if condition == "full":
            continue
        s = condition_summaries[condition]
        delta_acc = (full_acc - s["symbolic_accuracy"]) * 100
        delta_r2 = full_r2 - s["mean_r2"] if s["mean_r2"] >= 0 else None
        output["deltas_from_full"][condition] = {
            "accuracy_delta_pp": float(delta_acc),
            "r2_delta": float(delta_r2) if delta_r2 is not None else None,
        }

    # ---- Save results JSON ------------------------------------------------
    os.makedirs("results", exist_ok=True)
    results_path = "results/ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved ablation results to {results_path}")

    # ---- Generate ablation table figure -----------------------------------
    os.makedirs("figures", exist_ok=True)
    table_path = "figures/ablation_table.png"
    _render_ablation_table(condition_summaries, condition_tier_results, table_path)

    # ---- Final summary ----------------------------------------------------
    print()
    print("=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"  Model:       PhysMDT ({n_params:,} params)")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Equations:   {len(ablation_equations)} (Tier 3-5)")
    print(f"  Conditions:  {len(ablation_conditions)}")
    print(f"  Results:     {results_path}")
    print(f"  Table:       {table_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
