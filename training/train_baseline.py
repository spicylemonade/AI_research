"""Training script for the autoregressive transformer baseline.

Trains the AR baseline on the synthetic physics dataset with:
- AdamW optimizer with cosine LR schedule
- Mixed-precision (bf16) training
- Gradient accumulation
- Checkpoint saving (excluded from git)
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.physics_generator import PhysicsDataset, generate_datasets
from data.tokenizer import ExprTokenizer, PAD_IDX, SOS_IDX, EOS_IDX
from models.ar_baseline import ARBaseline, ARBaselineConfig


def get_args():
    parser = argparse.ArgumentParser(description='Train AR baseline')
    parser.add_argument('--n_train', type=int, default=200000)
    parser.add_argument('--n_val', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true', help='Quick smoke test mode')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--max_hours', type=float, default=4.0, help='Max training hours')
    return parser.parse_args()


def cosine_lr_schedule(optimizer, step, warmup_steps, total_steps, lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        lr_scale = step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr * lr_scale
    return lr * lr_scale


def train_epoch(model, loader, optimizer, device, epoch, args, global_step,
                total_steps, scaler, log_data):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        obs = batch['observations'].to(device)
        obs_mask = batch['obs_mask'].to(device)
        tokens = batch['tokens'].to(device)

        # Teacher forcing: input is tokens[:-1], target is tokens[1:]
        tgt_input = tokens[:, :-1]
        tgt_target = tokens[:, 1:]

        with autocast(dtype=torch.bfloat16):
            logits = model(obs, obs_mask, tgt_input)
            # logits: (batch, seq_len, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_target.reshape(-1),
                ignore_index=PAD_IDX,
            )
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            current_lr = cosine_lr_schedule(
                optimizer, global_step, args.warmup_steps, total_steps, args.lr
            )

        total_loss += loss.item() * args.grad_accum
        n_batches += 1

        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / n_batches
            elapsed = time.time() - start_time
            samples_sec = (batch_idx + 1) * args.batch_size / elapsed
            print(f"  Epoch {epoch} | Step {batch_idx+1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | "
                  f"{samples_sec:.0f} samples/sec")

            log_data['train_steps'].append({
                'global_step': global_step,
                'epoch': epoch,
                'loss': avg_loss,
                'lr': current_lr,
                'samples_sec': samples_sec,
            })

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    tier_correct = {}
    tier_total = {}

    tokenizer = ExprTokenizer()

    for batch in loader:
        obs = batch['observations'].to(device)
        obs_mask = batch['obs_mask'].to(device)
        tokens = batch['tokens'].to(device)
        tiers = batch['tier']

        tgt_input = tokens[:, :-1]
        tgt_target = tokens[:, 1:]

        with autocast(dtype=torch.bfloat16):
            logits = model(obs, obs_mask, tgt_input)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_target.reshape(-1),
                ignore_index=PAD_IDX,
            )

        total_loss += loss.item()
        n_batches += 1

        # Check token-level accuracy (greedy)
        preds = logits.argmax(dim=-1)  # (batch, seq_len)
        for b in range(preds.size(0)):
            tier = tiers[b].item()
            # Compare non-pad tokens
            target = tgt_target[b]
            pred = preds[b]
            mask = target != PAD_IDX
            correct = (pred[mask] == target[mask]).all().item()

            tier_correct[tier] = tier_correct.get(tier, 0) + int(correct)
            tier_total[tier] = tier_total.get(tier, 0) + 1

    avg_loss = total_loss / max(n_batches, 1)
    tier_acc = {}
    for t in sorted(tier_total.keys()):
        tier_acc[t] = tier_correct.get(t, 0) / tier_total[t]

    return avg_loss, tier_acc


def main():
    args = get_args()

    if args.quick:
        args.n_train = 5000
        args.n_val = 500
        args.epochs = 3
        args.batch_size = 64
        args.log_interval = 20
        args.eval_interval = 100

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate data
    train_ds, val_ds = generate_datasets(
        n_train=args.n_train,
        n_val=args.n_val,
        noise_level=0.01,
        seed=args.seed,
        quick=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    config = ARBaselineConfig()
    model = ARBaseline(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98), eps=1e-8,
    )

    scaler = GradScaler()

    total_steps = len(train_loader) * args.epochs // args.grad_accum

    # Logging
    log_data = {
        'config': {
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_encoder_layers': config.num_encoder_layers,
            'num_decoder_layers': config.num_decoder_layers,
            'dim_feedforward': config.dim_feedforward,
            'n_train': args.n_train,
            'n_val': args.n_val,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'total_params': model.count_parameters(),
        },
        'train_steps': [],
        'val_epochs': [],
    }

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()
    max_seconds = args.max_hours * 3600

    print(f"\nStarting training: {args.epochs} epochs, {total_steps} steps")
    print(f"Max training time: {args.max_hours} hours")

    for epoch in range(1, args.epochs + 1):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_seconds:
            print(f"\nTime limit reached ({args.max_hours}h). Stopping.")
            break

        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, args,
            global_step, total_steps, scaler, log_data,
        )

        # Validation
        val_loss, tier_acc = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        for t in sorted(tier_acc.keys()):
            print(f"  Tier {t}: {tier_acc[t]*100:.1f}% accuracy")

        log_data['val_epochs'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'tier_accuracy': tier_acc,
            'elapsed_hours': (time.time() - start_time) / 3600,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, 'ar_baseline_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, ckpt_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    # Save training log
    total_time = time.time() - start_time
    log_data['total_time_hours'] = total_time / 3600
    log_data['best_val_loss'] = best_val_loss

    os.makedirs('results', exist_ok=True)
    with open('results/baseline_training_log.json', 'w') as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"\nTraining complete in {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # GPU memory info
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {mem:.2f} GB")
        log_data['peak_gpu_memory_gb'] = mem

    return model, log_data


if __name__ == '__main__':
    main()
