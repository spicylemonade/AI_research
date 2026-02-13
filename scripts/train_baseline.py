#!/usr/bin/env python3
"""
Training script for the autoregressive baseline transformer.

Uses AdamW optimizer, cosine LR schedule, gradient clipping, and
mixed-precision (AMP) training on CPU/GPU.

Usage:
    python scripts/train_baseline.py --n_samples 10000 --epochs 20 --batch_size 64
"""

import os
import sys
import json
import time
import signal
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.baseline_ar import build_baseline_model
from src.tokenizer import (
    encode, decode, VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX, MAX_SEQ_LEN,
    infix_to_prefix,
)
from data.generator import generate_dataset, split_dataset

SEED = 42


class ComputeTimeout(Exception):
    pass


class PhysicsEquationDataset(Dataset):
    """Dataset of physics equations with numerical observations."""

    def __init__(self, samples, max_vars=6, n_obs=20, max_seq_len=MAX_SEQ_LEN):
        self.samples = []
        self.max_vars = max_vars
        self.n_obs = n_obs
        self.max_seq_len = max_seq_len

        for s in samples:
            if s.get("token_ids") is not None:
                self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Encode equation
        token_ids = sample["token_ids"]
        tgt = torch.tensor(token_ids[:self.max_seq_len], dtype=torch.long)

        # Encode observations
        obs = torch.zeros(self.n_obs, self.max_vars + 1)
        if sample.get("numerical_data") and sample["numerical_data"]:
            nd = sample["numerical_data"]
            x_data = nd["x"]
            y_data = nd["y"]
            for i in range(min(len(x_data), self.n_obs)):
                # Fill variable values
                var_vals = list(x_data[i].values())
                for j in range(min(len(var_vals), self.max_vars)):
                    obs[i, j] = var_vals[j]
                obs[i, self.max_vars] = y_data[i]

        return obs, tgt


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                grad_clip=1.0, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    scaler = torch.amp.GradScaler('cpu', enabled=use_amp) if use_amp else None

    for batch_idx, (obs, tgt) in enumerate(dataloader):
        obs = obs.to(device)
        tgt = tgt.to(device)

        # Input: all tokens except last, Target: all tokens except first (shifted)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device.type):
                logits = model(obs, tgt_input)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
        else:
            logits = model(obs, tgt_input)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        mask = tgt_output != PAD_IDX
        preds = logits.argmax(dim=-1)
        total_correct += ((preds == tgt_output) & mask).sum().item()
        total_tokens += mask.sum().item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, loss={loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for obs, tgt in dataloader:
        obs = obs.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(obs, tgt_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))

        total_loss += loss.item()
        mask = tgt_output != PAD_IDX
        preds = logits.argmax(dim=-1)
        total_correct += ((preds == tgt_output) & mask).sum().item()
        total_tokens += mask.sum().item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default="results/baseline_ar")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Max training time in seconds")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data
    print(f"Generating {args.n_samples} samples...")
    samples = generate_dataset(args.n_samples, args.seed, include_numerical=True, n_points=20)
    train_samples, val_samples, test_samples = split_dataset(samples, seed=args.seed)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Build datasets
    train_ds = PhysicsEquationDataset(train_samples)
    val_ds = PhysicsEquationDataset(val_samples)
    test_ds = PhysicsEquationDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model
    model = build_baseline_model(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_enc_layers=args.n_layers,
        n_dec_layers=args.n_layers,
        d_ff=args.d_model * 4,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Training loop with timeout
    os.makedirs(args.output_dir, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        elapsed = time.time() - start_time
        if elapsed > args.timeout:
            print(f"Timeout reached ({args.timeout}s). Stopping training.")
            break

        print(f"\nEpoch {epoch+1}/{args.epochs} (elapsed: {elapsed:.0f}s)")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Save results
    results = {
        "train_history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_val_loss": best_val_loss,
        "n_samples": args.n_samples,
        "epochs_completed": len(history["train_loss"]),
        "model_params": model.count_parameters(),
        "training_time_seconds": time.time() - start_time,
    }

    with open(os.path.join(args.output_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
