"""Full PhysMDT training pipeline.

Pre-trains PhysMDT on FSReD + procedural Newtonian equations with masked
diffusion objective, tree-aware positional encoding, and physics augmentations.

Usage:
    python scripts/train_physmdt.py --config configs/physmdt_base.yaml
    python scripts/train_physmdt.py --config configs/physmdt_scaled.yaml
"""

import sys
import os
import json
import time
import argparse
import math

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '.')

from src.model.physmdt import PhysMDT
from src.data.tokenizer import EquationTokenizer
from src.data.dataset import create_dataloaders
from src.model.tree_positional_encoding import get_tree_positional_encoding
from src.data.physics_augmentations import compute_augmented_loss


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(step: int, warmup_steps: int, base_lr: float, max_steps: int) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(config: dict, max_steps_override: int = None):
    """Main training loop.

    Args:
        config: Configuration dictionary.
        max_steps_override: Override max_steps (for smoke tests).
    """
    # Unpack config
    model_cfg = config['model']
    train_cfg = config['training']
    data_cfg = config['data']
    aug_cfg = config.get('augmentation', {})
    ckpt_cfg = config.get('checkpointing', {})
    log_cfg = config.get('logging', {})

    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_steps = max_steps_override or train_cfg['max_steps']

    # Create tokenizer
    tokenizer = EquationTokenizer(max_seq_len=model_cfg['max_seq_len'])

    # Create data loaders
    print("Creating data loaders...")
    n_procedural = data_cfg.get('n_procedural', 50000)
    train_loader, val_loader, test_loader = create_dataloaders(
        include_fsred=data_cfg.get('include_fsred', True),
        include_procedural=data_cfg.get('include_procedural', True),
        n_procedural=n_procedural,
        n_data_points=data_cfg.get('n_data_points', 256),
        batch_size=train_cfg['batch_size'],
        noise_std=data_cfg.get('noise_std', 0.01),
        train_ratio=data_cfg.get('train_ratio', 0.8),
        val_ratio=data_cfg.get('val_ratio', 0.1),
        seed=seed,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    model = PhysMDT(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        ffn_dim=model_cfg['ffn_dim'],
        max_seq_len=model_cfg['max_seq_len'],
        dropout=model_cfg.get('dropout', 0.1),
        input_dim=model_cfg.get('input_dim', 10),
        encoder_layers=model_cfg.get('encoder_layers', 4),
        mask_token_id=model_cfg.get('mask_token_id', 3),
        use_tree_pe=model_cfg.get('use_tree_pe', False),
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )

    # Training state
    use_tree_pe = model_cfg.get('use_tree_pe', False)
    use_physics_prior = aug_cfg.get('use_physics_prior', False)
    lambda_physics = aug_cfg.get('lambda_physics', 0.1)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    log_every = log_cfg.get('log_every', 100)
    eval_every = log_cfg.get('eval_every', 1000)
    save_every = ckpt_cfg.get('save_every', 10000)
    results_dir = log_cfg.get('results_dir', 'results/training_curves')
    checkpoint_dir = ckpt_cfg.get('checkpoint_dir', 'checkpoints/physmdt')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_log = []
    step = 0
    epoch = 0
    best_val_loss = float('inf')
    start_time = time.time()

    print(f"Starting training for {max_steps} steps...")

    while step < max_steps:
        epoch += 1
        model.train()

        for batch in train_loader:
            if step >= max_steps:
                break

            data_matrix = batch['data_matrix'].to(device)
            token_ids = batch['token_ids'].to(device)

            # Compute positional encoding
            pos_encoding = None
            if use_tree_pe:
                pos_encoding = get_tree_positional_encoding(
                    token_ids, model_cfg['d_model']
                ).to(device)

            # Learning rate schedule
            lr = get_lr(step, train_cfg.get('warmup_steps', 1000),
                       train_cfg['lr'], max_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Compute loss
            if use_physics_prior:
                loss, info = compute_augmented_loss(
                    model, data_matrix, token_ids, tokenizer,
                    pos_encoding=pos_encoding,
                    lambda_physics=lambda_physics,
                    use_physics_prior=True,
                )
            else:
                loss, info = model.compute_loss(
                    data_matrix, token_ids,
                    pos_encoding=pos_encoding,
                )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1

            # Logging
            if step % log_every == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                      f"LR: {lr:.2e} | Time: {elapsed:.0f}s")

            # Evaluation
            if step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device, use_tree_pe,
                                   model_cfg['d_model'])
                training_log.append({
                    'step': step,
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'val_loss': val_loss,
                    'lr': lr,
                    'elapsed': time.time() - start_time,
                })
                print(f"  Val loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(),
                              os.path.join(checkpoint_dir, 'best_model.pt'))

            # Checkpointing
            if step % save_every == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(checkpoint_dir, f'checkpoint_{step}.pt'))

    # Final evaluation
    val_loss = evaluate(model, val_loader, device, use_tree_pe,
                       model_cfg['d_model'])
    training_log.append({
        'step': step,
        'epoch': epoch,
        'train_loss': loss.item(),
        'val_loss': val_loss,
        'lr': lr,
        'elapsed': time.time() - start_time,
    })

    # Save training log
    log_path = os.path.join(results_dir, 'physmdt_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Save final model
    torch.save(model.state_dict(),
              os.path.join(checkpoint_dir, 'final_model.pt'))
    print(f"Final model saved. Best val loss: {best_val_loss:.4f}")

    return model, training_log


def evaluate(model, val_loader, device, use_tree_pe, d_model):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            data_matrix = batch['data_matrix'].to(device)
            token_ids = batch['token_ids'].to(device)

            pos_encoding = None
            if use_tree_pe:
                pos_encoding = get_tree_positional_encoding(
                    token_ids, d_model
                ).to(device)

            loss, _ = model.compute_loss(data_matrix, token_ids,
                                         pos_encoding=pos_encoding)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PhysMDT')
    parser.add_argument('--config', type=str, default='configs/physmdt_base.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Override max training steps')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, max_steps_override=args.max_steps)
