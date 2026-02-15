"""
Resume PARR training from AR-only checkpoint with refinement phase.
Loads best AR checkpoint and trains 5 more epochs with AR+Refinement loss.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import PhysicsDataset
from src.data.equation_templates import EQUATION_VOCAB as EQ_VOCAB
from src.models.parr_transformer import create_parr_model

SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return {
        "observations": torch.stack([b["observations"] for b in batch]),
        "equations": torch.stack([b["equation"] for b in batch]),
        "tiers": [b["tier"] for b in batch],
        "n_vars": [b["n_input_vars"] for b in batch],
        "n_obs": [b["n_obs"] for b in batch],
    }


def quick_token_match(pred, gt):
    pad = EQ_VOCAB["[PAD]"]
    p = [t for t in pred if t != pad]
    g = [t for t in gt if t != pad]
    return p == g


def quick_validate(model, val_loader, device, max_batches=50, K=4):
    model.eval()
    tier_matches = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            obs = batch["observations"].to(device)
            eq_gt = batch["equations"]
            tiers = batch["tiers"]

            pred_tokens = model.generate(obs, K=K)
            pred_np = pred_tokens.cpu().numpy()
            gt_np = eq_gt.numpy()

            for i in range(len(pred_np)):
                tier = tiers[i]
                if quick_token_match(pred_np[i].tolist(), gt_np[i].tolist()):
                    tier_matches[tier][0] += 1
                tier_matches[tier][1] += 1

    model.train()
    tier_esm = {}
    for tier in [1, 2, 3, 4]:
        if tier_matches[tier][1] > 0:
            tier_esm[tier] = tier_matches[tier][0] / tier_matches[tier][1]
        else:
            tier_esm[tier] = 0.0
    return tier_esm


def train_refinement():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "checkpoints"
    results_dir = "results"

    print("Loading datasets...")
    train_ds = PhysicsDataset("data", "train")
    val_ds = PhysicsDataset("data", "val")

    indices = list(range(min(200000, len(train_ds))))
    train_subset_ds = Subset(train_ds, indices)

    # Fewer workers to avoid OOM during refinement
    train_loader = DataLoader(
        train_subset_ds, batch_size=48, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=48, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    steps_per_epoch = len(train_loader)

    print("Loading PARR model from AR checkpoint...")
    model = create_parr_model(d_model=512, K=8, device=device)
    checkpoint_path = f"{output_dir}/parr_best.pt"
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Lower LR for refinement fine-tuning
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=0.01
    )

    n_refinement_epochs = 5
    total_steps = n_refinement_epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    scaler = torch.amp.GradScaler('cuda')

    pad_idx = EQ_VOCAB["[PAD]"]
    global_step = 0
    start_time = time.time()
    best_val_esm = 0.774  # From AR-only training

    # First validate with K=0 (AR-only) and K=4 (with refinement)
    print("\n=== Pre-training validation ===")
    tier_esm_ar = quick_validate(model, val_loader, device, max_batches=50, K=0)
    overall_ar = np.mean(list(tier_esm_ar.values()))
    print(f"AR-only (K=0): " + " | ".join(f"T{t}:{tier_esm_ar[t]:.3f}" for t in [1, 2, 3, 4]))
    print(f"  Overall: {overall_ar:.3f}")

    tier_esm_ref = quick_validate(model, val_loader, device, max_batches=50, K=4)
    overall_ref = np.mean(list(tier_esm_ref.values()))
    print(f"With refinement (K=4): " + " | ".join(f"T{t}:{tier_esm_ref[t]:.3f}" for t in [1, 2, 3, 4]))
    print(f"  Overall: {overall_ref:.3f}")

    training_log = {
        "phase": "refinement",
        "losses": [],
        "val_metrics": [],
        "best_val_esm": best_val_esm,
        "gpu_memory_peak": 0.0,
    }

    for epoch in range(n_refinement_epochs):
        print(f"\nEpoch {epoch+1}/{n_refinement_epochs} | AR+Refinement")
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            obs = batch["observations"].to(device)
            eq = batch["equations"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                ar_logits, ref_logits, is_corrupted = model(obs, eq)
                loss = model.compute_loss(ar_logits, ref_logits, eq, is_corrupted)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9
                training_log["gpu_memory_peak"] = max(training_log["gpu_memory_peak"], mem)

            if global_step % 500 == 0:
                avg_loss = epoch_loss / n_batches
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Step {global_step} | Loss: {avg_loss:.4f} | "
                      f"LR: {lr_now:.6f} | Time: {elapsed:.0f}s")

            if global_step % 2000 == 0:
                # Validate with refinement
                tier_esm = quick_validate(model, val_loader, device, max_batches=50, K=4)
                overall_esm = np.mean(list(tier_esm.values()))
                print(f"  -> Val ESM (K=4): " + " | ".join(
                    f"T{t}:{tier_esm[t]:.3f}" for t in [1, 2, 3, 4]
                ))
                print(f"     Overall: {overall_esm:.3f}")

                # Also check AR-only
                tier_esm_ar = quick_validate(model, val_loader, device, max_batches=50, K=0)
                overall_ar = np.mean(list(tier_esm_ar.values()))
                print(f"  -> Val ESM (K=0): Overall {overall_ar:.3f}")

                training_log["val_metrics"].append({
                    "step": global_step,
                    "tier_esm_K4": tier_esm,
                    "overall_esm_K4": overall_esm,
                    "overall_esm_K0": overall_ar,
                })

                if overall_esm > best_val_esm:
                    best_val_esm = overall_esm
                    training_log["best_val_esm"] = best_val_esm
                    torch.save(model.state_dict(), f"{output_dir}/parr_best.pt")
                    print(f"  -> New best! Overall ESM: {overall_esm:.3f}")

        torch.save(model.state_dict(), f"{output_dir}/parr_ref_epoch{epoch+1}.pt")
        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.4f}")

    torch.save(model.state_dict(), f"{output_dir}/parr_final.pt")

    with open(f"{results_dir}/parr_refinement_log.json", "w") as f:
        json.dump(training_log, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"\nRefinement training complete in {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Best overall ESM: {best_val_esm:.3f}")
    print(f"Peak GPU memory: {training_log['gpu_memory_peak']:.1f}GB")


if __name__ == "__main__":
    train_refinement()
