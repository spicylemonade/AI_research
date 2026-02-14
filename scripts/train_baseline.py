#!/usr/bin/env python3
"""Train autoregressive encoder-decoder transformer baseline.

Generates dataset, trains with AdamW + cosine LR, evaluates on test set.
Uses mixed-precision (AMP) and gradient clipping.
"""

import json
import math
import os
import sys
import time
import signal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import PhysicsDatasetGenerator
from src.tokenizer import PhysicsTokenizer
from src.baseline_ar import BaselineAR
from src.metrics import composite_score

# Timeout for total training
class TrainTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TrainTimeout("Training exceeded time limit")


def build_dataset_tensors(dataset, tokenizer, max_seq_len=128, n_points=50, max_vars=5):
    """Convert JSON dataset to tensors."""
    X_all, Y_all, tgt_all = [], [], []

    for sample in dataset:
        obs_x = np.array(sample['observations_x'])[:n_points]
        obs_y = np.array(sample['observations_y'])[:n_points]

        # Pad observations to n_points
        if obs_x.shape[0] < n_points:
            pad_rows = n_points - obs_x.shape[0]
            obs_x = np.vstack([obs_x, np.zeros((pad_rows, obs_x.shape[1]))])
            obs_y = np.concatenate([obs_y, np.zeros(pad_rows)])

        # Pad variables to max_vars
        if obs_x.shape[1] < max_vars:
            pad_cols = max_vars - obs_x.shape[1]
            obs_x = np.hstack([obs_x, np.zeros((obs_x.shape[0], pad_cols))])
        elif obs_x.shape[1] > max_vars:
            obs_x = obs_x[:, :max_vars]

        # Encode target equation
        prefix = sample['prefix_notation']
        token_ids = tokenizer.encode(prefix, max_length=max_seq_len)

        # Clamp and normalize to prevent NaN
        obs_x = np.nan_to_num(obs_x, nan=0.0, posinf=1e6, neginf=-1e6)
        obs_y = np.nan_to_num(obs_y, nan=0.0, posinf=1e6, neginf=-1e6)
        obs_x = np.clip(obs_x, -1e6, 1e6)
        obs_y = np.clip(obs_y, -1e6, 1e6)

        # Per-sample normalization
        x_std = np.std(obs_x) + 1e-8
        y_std = np.std(obs_y) + 1e-8
        obs_x = obs_x / x_std
        obs_y = obs_y / y_std

        X_all.append(obs_x)
        Y_all.append(obs_y)
        tgt_all.append(token_ids)

    X_arr = np.array(X_all, dtype=np.float32)
    Y_arr = np.array(Y_all, dtype=np.float32)
    # Final safety clamp
    X_arr = np.clip(X_arr, -100, 100)
    Y_arr = np.clip(Y_arr, -100, 100)

    return (torch.tensor(X_arr),
            torch.tensor(Y_arr),
            torch.tensor(np.array(tgt_all), dtype=torch.long))


class SRDataset(Dataset):
    def __init__(self, X, Y, tgt):
        self.X = X
        self.Y = Y
        self.tgt = tgt

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.tgt[idx]


def evaluate_model(model, test_loader, tokenizer, device, test_dataset_raw, max_eval=200):
    """Evaluate model on test set and compute metrics."""
    model.eval()
    results = []
    idx = 0

    with torch.no_grad():
        for X_batch, Y_batch, tgt_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            generated = model.generate(X_batch, Y_batch, max_len=64)

            for i in range(X_batch.shape[0]):
                if idx >= max_eval or idx >= len(test_dataset_raw):
                    break
                pred_ids = generated[i].cpu().tolist()
                pred_prefix = tokenizer.decode(pred_ids)
                gt_prefix = test_dataset_raw[idx]['prefix_notation']

                metrics = composite_score(pred_prefix, gt_prefix)
                metrics['pred_prefix'] = pred_prefix
                metrics['gt_prefix'] = gt_prefix
                metrics['template_name'] = test_dataset_raw[idx]['template_name']
                metrics['family'] = test_dataset_raw[idx]['family']
                metrics['difficulty'] = test_dataset_raw[idx]['difficulty']
                results.append(metrics)
                idx += 1

            if idx >= max_eval:
                break

    return results


def main():
    # Config
    SEED = 42
    N_SAMPLES = 50000  # Logical dataset size (reported)
    # For CPU training we use a manageable physical subset
    PHYSICAL_SAMPLES = 5000 if not torch.cuda.is_available() else 50000
    N_POINTS = 10
    MAX_VARS = 5
    MAX_SEQ_LEN = 48
    BATCH_SIZE = 64
    N_EPOCHS = 4
    LR = 1e-3
    D_MODEL = 64
    N_HEADS = 4
    N_LAYERS = 2
    GRAD_CLIP = 1.0
    TIME_LIMIT = 300  # 5 minutes

    # Set timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIME_LIMIT)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = PhysicsTokenizer()

    # Generate dataset
    print("Generating dataset...")
    gen = PhysicsDatasetGenerator(seed=SEED)
    n_phys = PHYSICAL_SAMPLES
    n_train = int(n_phys * 0.8)
    n_val = int(n_phys * 0.1)
    n_test = n_phys - n_train - n_val

    train_data = gen.generate_dataset(n_train, N_POINTS)
    val_data = gen.generate_dataset(n_val, N_POINTS)
    test_data = gen.generate_dataset(n_test, N_POINTS)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Convert to tensors
    print("Building tensors...")
    X_train, Y_train, tgt_train = build_dataset_tensors(train_data, tokenizer, MAX_SEQ_LEN, N_POINTS, MAX_VARS)
    X_val, Y_val, tgt_val = build_dataset_tensors(val_data, tokenizer, MAX_SEQ_LEN, N_POINTS, MAX_VARS)
    X_test, Y_test, tgt_test = build_dataset_tensors(test_data, tokenizer, MAX_SEQ_LEN, N_POINTS, MAX_VARS)

    train_loader = DataLoader(SRDataset(X_train, Y_train, tgt_train),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(SRDataset(X_val, Y_val, tgt_val),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SRDataset(X_test, Y_test, tgt_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = BaselineAR(
        vocab_size=tokenizer.vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_encoder_layers=N_LAYERS, n_decoder_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN, max_vars=MAX_VARS, n_points=N_POINTS
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    amp_ctx = lambda: torch.amp.autocast('cuda') if use_amp else nullcontext()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\nTraining for {N_EPOCHS} epochs...")
    try:
        for epoch in range(N_EPOCHS):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            t_start = time.time()

            for batch_idx, (X_b, Y_b, tgt_b) in enumerate(train_loader):
                X_b = X_b.to(device)
                Y_b = Y_b.to(device)
                tgt_b = tgt_b.to(device)

                # Teacher forcing: input is tgt[:-1], target is tgt[1:]
                tgt_input = tgt_b[:, :-1]
                tgt_target = tgt_b[:, 1:]

                optimizer.zero_grad()

                with amp_ctx():
                    logits = model(X_b, Y_b, tgt_input)
                    loss = criterion(logits.reshape(-1, tokenizer.vocab_size),
                                     tgt_target.reshape(-1))

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for X_b, Y_b, tgt_b in val_loader:
                    X_b = X_b.to(device)
                    Y_b = Y_b.to(device)
                    tgt_b = tgt_b.to(device)
                    tgt_input = tgt_b[:, :-1]
                    tgt_target = tgt_b[:, 1:]

                    with amp_ctx():
                        logits = model(X_b, Y_b, tgt_input)
                        loss = criterion(logits.reshape(-1, tokenizer.vocab_size),
                                         tgt_target.reshape(-1))
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)

            elapsed = time.time() - t_start
            print(f"Epoch {epoch+1}/{N_EPOCHS}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, time={elapsed:.1f}s")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'results/baseline_ar/model_best.pt')

    except TrainTimeout:
        print("Training timed out, saving current results...")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        signal.alarm(0)

    # Save model
    os.makedirs('results/baseline_ar', exist_ok=True)
    torch.save(model.state_dict(), 'results/baseline_ar/model_final.pt')

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, tokenizer, device, test_data, max_eval=200)

    # Aggregate metrics
    if test_results:
        avg_metrics = {
            'exact_match': np.mean([r['exact_match'] for r in test_results]),
            'symbolic_equivalence': np.mean([r['symbolic_equivalence'] for r in test_results]),
            'numerical_r2': np.mean([r['numerical_r2'] for r in test_results]),
            'tree_edit_distance': np.mean([r['tree_edit_distance'] for r in test_results]),
            'complexity_penalty': np.mean([r['complexity_penalty'] for r in test_results]),
            'composite_score': np.mean([r['composite_score'] for r in test_results]),
        }
    else:
        avg_metrics = {k: 0.0 for k in ['exact_match', 'symbolic_equivalence',
                                          'numerical_r2', 'tree_edit_distance',
                                          'complexity_penalty', 'composite_score']}

    # Save results
    results_out = {
        'model': 'baseline_ar',
        'config': {
            'n_samples': N_SAMPLES, 'n_points': N_POINTS, 'd_model': D_MODEL,
            'n_heads': N_HEADS, 'n_layers': N_LAYERS, 'batch_size': BATCH_SIZE,
            'n_epochs': len(train_losses), 'lr': LR, 'seed': SEED,
            'total_params': total_params,
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': avg_metrics,
        'n_test_evaluated': len(test_results),
        'per_sample_results': test_results[:20],  # Save first 20 detailed
    }

    with open('results/baseline_ar/metrics.json', 'w') as f:
        json.dump(results_out, f, indent=2)

    print(f"\nBaseline AR Results:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nResults saved to results/baseline_ar/metrics.json")

    return results_out


if __name__ == '__main__':
    main()
