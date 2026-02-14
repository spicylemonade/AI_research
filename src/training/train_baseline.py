"""
Training script for the vanilla transformer baseline.
Trains on physics equation-observation pairs and evaluates per-tier.
"""
import os
import sys
import json
import time
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from src.data.generator import PhysicsDataset
from src.data.equation_templates import EQUATION_VOCAB
from src.models.baseline_transformer import create_baseline_model
from src.evaluation.metrics import (
    evaluate_model_predictions, exact_symbolic_match, r2_score, format_results_latex
)

# Timeout handling
class TrainTimeout(Exception):
    pass

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


def train_baseline(
    data_dir="data",
    output_dir="checkpoints",
    results_dir="results",
    max_epochs=15,
    batch_size=64,
    lr=3e-4,
    max_train_time=14400,  # 4 hours max
    eval_every=2000,
    device="cuda",
):
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds = PhysicsDataset(data_dir, "train")
    val_ds = PhysicsDataset(data_dir, "val")

    # Use subset for faster training (100K samples for baseline)
    rng = np.random.RandomState(SEED)
    train_indices = rng.choice(len(train_ds), min(200_000, len(train_ds)), replace=False)
    train_subset = Subset(train_ds, train_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    print("Creating model...")
    model = create_baseline_model(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98),
                            weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs * len(train_loader), eta_min=1e-5
    )

    pad_idx = EQUATION_VOCAB["[PAD]"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    scaler = torch.amp.GradScaler('cuda')

    training_log = {
        "losses": [], "val_metrics": [],
        "epoch_losses": [], "best_val_esm": 0.0,
    }
    global_step = 0
    start_time = time.time()
    best_val_esm = 0.0

    print(f"Training for {max_epochs} epochs, {len(train_loader)} steps/epoch...")
    print(f"Training subset: {len(train_subset)} samples")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if time.time() - start_time > max_train_time:
                print(f"Time limit reached ({max_train_time}s). Stopping training.")
                break

            obs = batch["observations"].to(device)
            eq = batch["equations"].to(device)

            # Teacher forcing: input is eq[:-1], target is eq[1:]
            eq_input = eq[:, :-1]
            eq_target = eq[:, 1:]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(obs, eq_input)
                loss = criterion(
                    logits.reshape(-1, model.vocab_size),
                    eq_target.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 500 == 0:
                avg_loss = epoch_loss / n_batches
                lr_current = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"  Step {global_step} | Loss: {avg_loss:.4f} | "
                      f"LR: {lr_current:.6f} | Time: {elapsed:.0f}s")
                training_log["losses"].append({
                    "step": global_step, "loss": avg_loss,
                    "lr": lr_current, "elapsed": elapsed,
                })

            if global_step % eval_every == 0:
                val_esm = quick_validate(model, val_loader, device, max_batches=50)
                print(f"  → Val ESM: {val_esm:.3f}")
                training_log["val_metrics"].append({
                    "step": global_step, "esm": val_esm,
                })

                if val_esm > best_val_esm:
                    best_val_esm = val_esm
                    training_log["best_val_esm"] = best_val_esm
                    torch.save(model.state_dict(), f"{output_dir}/baseline_best.pt")
                    print(f"  → New best! Saved checkpoint.")

        if time.time() - start_time > max_train_time:
            break

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        training_log["epoch_losses"].append({"epoch": epoch, "loss": avg_epoch_loss})
        print(f"Epoch {epoch+1}/{max_epochs} | Avg Loss: {avg_epoch_loss:.4f}")

    # Save final checkpoint
    torch.save(model.state_dict(), f"{output_dir}/baseline_final.pt")

    # Save training log
    with open(f"{results_dir}/baseline_training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Best validation ESM: {best_val_esm:.3f}")

    return model, training_log


def quick_validate(model, val_loader, device, max_batches=50):
    """Quick validation: compute ESM on a subset of validation data."""
    model.eval()
    total_match = 0
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            obs = batch["observations"].to(device)
            eq_gt = batch["equations"]

            # Generate predictions
            pred_tokens = model.generate(obs)
            pred_tokens = pred_tokens.cpu().numpy()
            gt_tokens = eq_gt.numpy()

            for i in range(len(pred_tokens)):
                if exact_symbolic_match(pred_tokens[i].tolist(), gt_tokens[i].tolist()):
                    total_match += 1
                total_count += 1

    model.train()
    return total_match / max(total_count, 1)


def full_evaluate(model, data_dir, results_dir, device="cuda", split="test"):
    """Full evaluation on test set with all metrics."""
    print(f"\n=== Full Evaluation on {split} set ===")
    test_ds = PhysicsDataset(data_dir, split)
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    model.eval()
    all_preds = []
    all_gts = []
    all_obs = []
    all_meta = []

    max_samples = 5000  # Evaluate on 5K samples for speed
    count = 0

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            if count >= max_samples:
                break

            obs = batch["observations"].to(device)
            eq_gt = batch["equations"]

            # Measure inference time
            start = time.time()
            pred_tokens = model.generate(obs)
            inference_time = time.time() - start

            pred_np = pred_tokens.cpu().numpy()
            gt_np = eq_gt.numpy()
            obs_np = batch["observations"].numpy()

            for i in range(len(pred_np)):
                if count >= max_samples:
                    break
                all_preds.append(pred_np[i].tolist())
                all_gts.append(gt_np[i].tolist())
                all_obs.append(obs_np[i])
                all_meta.append({
                    "tier": batch["tiers"][i],
                    "n_input_vars": batch["n_vars"][i],
                    "n_obs": batch["n_obs"][i],
                })
                count += 1

    print(f"Evaluating {len(all_preds)} predictions...")
    obs_array = np.array(all_obs)
    results = evaluate_model_predictions(
        all_preds, all_gts, obs_array, all_meta, max_eval=max_samples
    )

    # Add inference latency
    results["inference"] = {
        "samples_per_second": count / max(inference_time, 0.001),
        "avg_latency_ms": inference_time / max(count, 1) * 1000,
    }

    # Print results
    print("\n=== Results ===")
    for tier in sorted(results["per_tier"].keys()):
        tr = results["per_tier"][tier]
        print(f"Tier {tier}: ESM={tr['exact_match_rate']:.3f}, "
              f"R²={tr['r2_score']:.3f}, NTED={tr['tree_edit_distance']:.3f}")
    ov = results["overall"]
    print(f"Overall: ESM={ov['exact_match_rate']:.3f}, R²={ov['r2_score']:.3f}")

    # Save results
    with open(f"{results_dir}/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_dir}/baseline_results.json")

    # Print LaTeX table
    latex = format_results_latex(results)
    print(f"\n{latex}")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, log = train_baseline(device=device)
    results = full_evaluate(model, "data", "results", device)
