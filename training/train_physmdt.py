"""Curriculum training script for Physics Masked Diffusion Transformer (PhysMDT).

Three-phase curriculum:
  Phase 1: Tier 1-2 only (simple equations)
  Phase 2: Tier 1-3 with mixed batches
  Phase 3: Tier 1-4 with emphasis on harder tiers

Masking schedule: High (90-100%) annealing to variable (30-100%).
Mixed-precision bf16 with gradient checkpointing.
"""

import os
import sys
import json
import time
import math
import csv
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.physics_generator import PhysicsDataset
from data.equations import get_training_equations, get_equations_by_tier
from data.tokenizer import ExprTokenizer, PAD_IDX, SOS_IDX, EOS_IDX, MASK_IDX
from models.physmdt import PhysMDT, PhysMDTConfig, apply_random_mask


def get_args():
    parser = argparse.ArgumentParser(description='Train PhysMDT with curriculum')
    # Data
    parser.add_argument('--n_samples_per_phase', type=int, default=50000)
    parser.add_argument('--n_val', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    # Optimizer
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    # Curriculum
    parser.add_argument('--phase1_epochs', type=int, default=10)
    parser.add_argument('--phase2_epochs', type=int, default=10)
    parser.add_argument('--phase3_epochs', type=int, default=10)
    # Masking
    parser.add_argument('--mask_ratio_start', type=float, default=0.9)
    parser.add_argument('--mask_ratio_end', type=float, default=0.3)
    # General
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--max_hours', type=float, default=6.0)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--dim_loss_weight', type=float, default=0.05)
    return parser.parse_args()


def cosine_lr_schedule(optimizer, step, warmup_steps, total_steps, base_lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def get_mask_ratio(global_step, total_steps, start=0.9, end=0.3):
    """Anneal mask ratio from high to variable range."""
    progress = min(global_step / max(total_steps, 1), 1.0)
    # Minimum mask ratio decreases over time
    min_ratio = start - (start - end) * progress
    # Sample uniform between min_ratio and 1.0
    return random.uniform(min_ratio, 1.0)


def create_phase_dataset(tiers, n_samples, noise_level=0.01, seed=42):
    """Create dataset for a specific set of tiers."""
    all_eqs = get_training_equations()
    phase_eqs = [eq for eq in all_eqs if eq.tier in tiers]
    # Weight harder tiers more
    tier_weights = {t: t * 1.0 for t in tiers}
    return PhysicsDataset(
        equations=phase_eqs,
        n_samples=n_samples,
        noise_level=noise_level,
        seed=seed,
        tier_weights=tier_weights,
    )


def train_phase(model, train_loader, val_loader, optimizer, scaler, device,
                phase_name, n_epochs, args, global_step, total_steps, log_data,
                start_time, max_seconds):
    """Train one curriculum phase."""
    print(f"\n{'='*60}")
    print(f"Curriculum {phase_name}")
    print(f"{'='*60}")

    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        elapsed = time.time() - start_time
        if elapsed > max_seconds:
            print(f"Time limit reached. Stopping.")
            return global_step, best_val_loss

        model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            obs = batch['observations'].to(device)
            obs_mask = batch['obs_mask'].to(device)
            tokens = batch['tokens'].to(device)

            # Apply random masking with annealed ratio
            mask_ratio = get_mask_ratio(global_step, total_steps,
                                        args.mask_ratio_start, args.mask_ratio_end)
            masked_tokens, token_mask = apply_random_mask(tokens, mask_ratio)

            with autocast(dtype=torch.bfloat16):
                logits, aux = model(obs, obs_mask, masked_tokens, token_mask)
                loss = model.compute_loss(
                    logits, tokens, token_mask,
                    aux.get('dim_loss'),
                    args.dim_loss_weight,
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
                    optimizer, global_step, args.warmup_steps,
                    total_steps, args.lr,
                )

            total_loss += loss.item() * args.grad_accum
            n_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_loss / n_batches
                elapsed = time.time() - epoch_start
                samples_sec = (batch_idx + 1) * args.batch_size / elapsed
                print(f"  {phase_name} Epoch {epoch} | "
                      f"Step {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | MR: {mask_ratio:.2f} | "
                      f"{samples_sec:.0f} samp/s")

                log_data['train_steps'].append({
                    'global_step': global_step,
                    'phase': phase_name,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'mask_ratio': mask_ratio,
                    'lr': current_lr if 'current_lr' in dir() else args.lr,
                    'samples_sec': samples_sec,
                })

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        val_loss = evaluate_phase(model, val_loader, device)

        print(f"  {phase_name} Epoch {epoch} | "
              f"Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

        log_data['val_epochs'].append({
            'global_step': global_step,
            'phase': phase_name,
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'elapsed_hours': (time.time() - start_time) / 3600,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, 'physmdt_best.pt')
            torch.save({
                'global_step': global_step,
                'phase': phase_name,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  Saved best model (val={val_loss:.4f})")

    return global_step, best_val_loss


@torch.no_grad()
def evaluate_phase(model, loader, device):
    """Quick validation loss evaluation."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        obs = batch['observations'].to(device)
        obs_mask = batch['obs_mask'].to(device)
        tokens = batch['tokens'].to(device)

        # Use 50% mask ratio for validation
        masked_tokens, token_mask = apply_random_mask(tokens, 0.5)

        with autocast(dtype=torch.bfloat16):
            logits, aux = model(obs, obs_mask, masked_tokens, token_mask)
            loss = model.compute_loss(logits, tokens, token_mask)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = get_args()

    if args.quick:
        args.n_samples_per_phase = 3000
        args.n_val = 500
        args.phase1_epochs = 3
        args.phase2_epochs = 3
        args.phase3_epochs = 3
        args.batch_size = 32
        args.log_interval = 20
        args.max_hours = 0.5

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    config = PhysMDTConfig()
    model = PhysMDT(config).to(device)
    print(f"PhysMDT parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98), eps=1e-8,
    )
    scaler = GradScaler()

    # Compute total steps
    steps_per_phase = (args.n_samples_per_phase // args.batch_size)
    total_steps = steps_per_phase * (
        args.phase1_epochs + args.phase2_epochs + args.phase3_epochs
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    log_data = {
        'config': {
            'model_params': model.count_parameters(),
            'd_model': config.d_model,
            'n_samples_per_phase': args.n_samples_per_phase,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'total_steps': total_steps,
        },
        'train_steps': [],
        'val_epochs': [],
        'phase_results': [],
    }

    start_time = time.time()
    max_seconds = args.max_hours * 3600
    global_step = 0

    # ============ Phase 1: Tier 1-2 ============
    print("\nGenerating Phase 1 data (Tier 1-2)...")
    train_ds1 = create_phase_dataset(
        tiers=[1, 2], n_samples=args.n_samples_per_phase, seed=args.seed
    )
    val_ds = create_phase_dataset(
        tiers=[1, 2, 3, 4, 5], n_samples=args.n_val, seed=args.seed + 100
    )
    loader1 = DataLoader(train_ds1, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    global_step, p1_loss = train_phase(
        model, loader1, val_loader, optimizer, scaler, device,
        "Phase 1 (Tier 1-2)", args.phase1_epochs, args,
        global_step, total_steps, log_data, start_time, max_seconds,
    )
    log_data['phase_results'].append({
        'phase': 'Phase 1', 'tiers': [1, 2], 'best_val_loss': p1_loss
    })

    # ============ Phase 2: Tier 1-3 ============
    print("\nGenerating Phase 2 data (Tier 1-3)...")
    train_ds2 = create_phase_dataset(
        tiers=[1, 2, 3], n_samples=args.n_samples_per_phase, seed=args.seed + 1
    )
    loader2 = DataLoader(train_ds2, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, drop_last=True, pin_memory=True)

    global_step, p2_loss = train_phase(
        model, loader2, val_loader, optimizer, scaler, device,
        "Phase 2 (Tier 1-3)", args.phase2_epochs, args,
        global_step, total_steps, log_data, start_time, max_seconds,
    )
    log_data['phase_results'].append({
        'phase': 'Phase 2', 'tiers': [1, 2, 3], 'best_val_loss': p2_loss
    })

    # ============ Phase 3: Tier 1-4 (with emphasis on harder tiers) ============
    print("\nGenerating Phase 3 data (Tier 1-4)...")
    train_ds3 = create_phase_dataset(
        tiers=[1, 2, 3, 4], n_samples=args.n_samples_per_phase, seed=args.seed + 2
    )
    loader3 = DataLoader(train_ds3, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, drop_last=True, pin_memory=True)

    global_step, p3_loss = train_phase(
        model, loader3, val_loader, optimizer, scaler, device,
        "Phase 3 (Tier 1-4)", args.phase3_epochs, args,
        global_step, total_steps, log_data, start_time, max_seconds,
    )
    log_data['phase_results'].append({
        'phase': 'Phase 3', 'tiers': [1, 2, 3, 4], 'best_val_loss': p3_loss
    })

    # Save training log
    total_time = time.time() - start_time
    log_data['total_time_hours'] = total_time / 3600
    log_data['final_global_step'] = global_step

    if torch.cuda.is_available():
        log_data['peak_gpu_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9

    with open('results/physmdt_training_log.json', 'w') as f:
        json.dump(log_data, f, indent=2, default=str)

    # Also save as CSV for easy plotting
    with open('results/physmdt_training_steps.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['global_step', 'phase', 'loss', 'mask_ratio', 'lr'])
        writer.writeheader()
        for step in log_data['train_steps']:
            writer.writerow({
                'global_step': step['global_step'],
                'phase': step['phase'],
                'loss': step['loss'],
                'mask_ratio': step.get('mask_ratio', ''),
                'lr': step.get('lr', ''),
            })

    print(f"\nTraining complete in {total_time/3600:.2f} hours")
    print(f"Phase 1 best val loss: {p1_loss:.4f}")
    print(f"Phase 2 best val loss: {p2_loss:.4f}")
    print(f"Phase 3 best val loss: {p3_loss:.4f}")

    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == '__main__':
    main()
