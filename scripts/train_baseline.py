"""Training script for the AR-Baseline model.

Usage:
    python scripts/train_baseline.py [--epochs 100] [--batch_size 16] [--smoke_test]
"""

import sys
import os
import json
import time
import argparse
import signal

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.ar_baseline import ARBaseline
from src.data.dataset import create_dataloaders, get_fsred_equations, EquationDataset, collate_fn
from src.data.tokenizer import EquationTokenizer, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

# Timeout handling
class TrainTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TrainTimeout()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in loader:
        data_matrix = batch['data_matrix'].to(device)
        token_ids = batch['token_ids'].to(device)

        # Teacher forcing: input is tokens[:-1], target is tokens[1:]
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]

        logits = model(data_matrix, input_ids)
        loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    for batch in loader:
        data_matrix = batch['data_matrix'].to(device)
        token_ids = batch['token_ids'].to(device)
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
        logits = model(data_matrix, input_ids)
        loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    # Set timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data
    if args.smoke_test:
        # Small dataset for smoke test
        from src.data.dataset import _generate_procedural_equations
        eqs = _generate_procedural_equations(n_equations=1000, seed=args.seed)
    else:
        eqs = get_fsred_equations()
        from src.data.dataset import _generate_procedural_equations
        eqs += _generate_procedural_equations(n_equations=50000, seed=args.seed)

    tokenizer = EquationTokenizer(max_seq_len=64)

    n_train = int(len(eqs) * 0.8)
    n_val = int(len(eqs) * 0.1)
    train_eqs = eqs[:n_train]
    val_eqs = eqs[n_train:n_train + n_val]

    train_ds = EquationDataset(train_eqs, tokenizer, n_data_points=64, seed=args.seed)
    val_ds = EquationDataset(val_eqs, tokenizer, n_data_points=64, seed=args.seed + 1)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    model = ARBaseline(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.d_model * 4,
        max_seq_len=64,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss = float('inf')
    training_log = []

    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = eval_epoch(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            log_entry = {
                'epoch': epoch + 1,
                'train_loss': round(train_loss, 4),
                'val_loss': round(val_loss, 4),
                'time_s': round(elapsed, 1),
            }
            training_log.append(log_entry)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}: "
                      f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                      f"time={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('results', exist_ok=True)
                torch.save(model.state_dict(), 'results/ar_baseline_best.pt')

    except TrainTimeout:
        print(f"Training timed out after {args.timeout}s — saving results so far")
    finally:
        signal.alarm(0)

    # Save training log
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_training.log', 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to results/baseline_training.log")

    return best_val_loss


if __name__ == '__main__':
    main()
