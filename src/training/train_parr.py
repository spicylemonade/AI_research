"""
Training script for the PARR transformer with curriculum learning.
Implements masked diffusion training with progressive complexity.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from src.data.generator import PhysicsDataset
from src.data.equation_templates import EQUATION_VOCAB
from src.models.parr_transformer import create_parr_model, PARRTransformer
from src.training.curriculum import CurriculumScheduler
from src.evaluation.metrics import exact_symbolic_match

SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def quick_validate_parr(model, val_loader, device, max_batches=50):
    """Quick validation for PARR model."""
    model.eval()
    tier_matches = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            obs = batch["observations"].to(device)
            eq_gt = batch["equations"]
            tiers = batch["tiers"]

            pred_tokens = model.generate(obs)
            pred_np = pred_tokens.cpu().numpy()
            gt_np = eq_gt.numpy()

            for i in range(len(pred_np)):
                tier = tiers[i]
                if exact_symbolic_match(pred_np[i].tolist(), gt_np[i].tolist()):
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


def train_parr(
    data_dir="data",
    output_dir="checkpoints",
    results_dir="results",
    max_epochs=20,
    batch_size=32,
    lr=2e-4,
    max_train_time=28800,  # 8 hours max
    eval_every=2000,
    K=8,
    d_model=512,
    device="cuda",
):
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds = PhysicsDataset(data_dir, "train")
    val_ds = PhysicsDataset(data_dir, "val")

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    print("Creating PARR model...")
    model = create_parr_model(d_model=d_model, K=K, device=device)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01
    )

    total_steps = max_epochs * (len(train_ds) // batch_size)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    scaler = torch.amp.GradScaler('cuda')

    # Curriculum scheduler
    curriculum = CurriculumScheduler(train_ds)

    training_log = {
        "losses": [], "val_metrics": [], "curriculum": [],
        "best_val_esm": 0.0, "gpu_memory_peak": 0.0,
    }
    global_step = 0
    start_time = time.time()
    best_val_esm = 0.0

    for epoch in range(max_epochs):
        # Get dataloader for current curriculum phase
        train_loader = curriculum.get_dataloader(
            batch_size=batch_size, num_workers=4, collate_fn=collate_fn
        )

        print(f"\nEpoch {epoch+1}/{max_epochs} | Phase {curriculum.current_phase} | "
              f"Tiers: {curriculum.get_phase_tiers(curriculum.current_phase)} | "
              f"Samples: {len(curriculum.get_phase_indices(curriculum.current_phase))}")

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if time.time() - start_time > max_train_time:
                print(f"Time limit reached ({max_train_time}s). Stopping.")
                break

            obs = batch["observations"].to(device)
            eq = batch["equations"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                all_logits, is_masked = model(obs, eq)
                loss = model.compute_loss(all_logits, eq, is_masked)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Track GPU memory
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9
                training_log["gpu_memory_peak"] = max(training_log["gpu_memory_peak"], mem)

            if global_step % 500 == 0:
                avg_loss = epoch_loss / n_batches
                elapsed = time.time() - start_time
                print(f"  Step {global_step} | Loss: {avg_loss:.4f} | "
                      f"Phase: {curriculum.current_phase} | Time: {elapsed:.0f}s")
                training_log["losses"].append({
                    "step": global_step, "loss": avg_loss,
                    "phase": curriculum.current_phase, "elapsed": elapsed,
                })

            if global_step % eval_every == 0:
                tier_esm = quick_validate_parr(model, val_loader, device, max_batches=50)
                overall_esm = np.mean(list(tier_esm.values()))
                print(f"  → Val ESM: " + " | ".join(
                    f"T{t}:{tier_esm[t]:.3f}" for t in [1, 2, 3, 4]
                ))

                training_log["val_metrics"].append({
                    "step": global_step, "tier_esm": tier_esm,
                    "overall_esm": overall_esm,
                })

                # Check curriculum advancement
                if curriculum.should_advance(tier_esm):
                    curriculum.advance_phase()
                    training_log["curriculum"].append({
                        "step": global_step, "phase": curriculum.current_phase,
                    })

                if overall_esm > best_val_esm:
                    best_val_esm = overall_esm
                    training_log["best_val_esm"] = best_val_esm
                    torch.save(model.state_dict(), f"{output_dir}/parr_best.pt")
                    print(f"  → New best! Overall ESM: {overall_esm:.3f}")

        if time.time() - start_time > max_train_time:
            break

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.4f}")

    # Save final checkpoint
    torch.save(model.state_dict(), f"{output_dir}/parr_final.pt")

    # Save training log
    with open(f"{results_dir}/parr_training_log.json", "w") as f:
        json.dump(training_log, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Best overall ESM: {best_val_esm:.3f}")
    print(f"Peak GPU memory: {training_log['gpu_memory_peak']:.1f}GB")

    return model, training_log


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, log = train_parr(device=device)
