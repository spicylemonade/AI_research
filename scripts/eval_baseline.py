"""Train and evaluate AR-Baseline on FSReD benchmark.

Produces results/baseline_results.json with per-equation metrics.
"""

import sys
import os
import json
import time
import signal
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.ar_baseline import ARBaseline
from src.data.dataset import get_fsred_equations, EquationDataset, collate_fn, _generate_data_points
from src.data.tokenizer import EquationTokenizer, PAD_TOKEN
from src.evaluation.metrics import compute_all_metrics

class Timeout(Exception):
    pass

def _handler(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, _handler)

def main():
    signal.alarm(480)  # 8 minute total timeout

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = EquationTokenizer(max_seq_len=64)
    fsred_equations = get_fsred_equations()

    # Split: 96 train, 12 val, 12 test (but evaluate on all 120)
    train_eqs = fsred_equations[:96]
    val_eqs = fsred_equations[96:108]

    # Create datasets with smaller data points for speed
    train_ds = EquationDataset(train_eqs, tokenizer, n_data_points=64, seed=42)
    val_ds = EquationDataset(val_eqs, tokenizer, n_data_points=64, seed=43)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = ARBaseline(
        vocab_size=tokenizer.vocab_size, d_model=256, n_heads=4,
        n_layers=4, ffn_dim=1024, max_seq_len=64, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Training
    training_log = []
    best_val_loss = float('inf')
    print("Training AR-Baseline...")
    try:
        for epoch in range(50):
            model.train()
            total_loss = 0
            n_batches = 0
            for batch in train_loader:
                data_matrix = batch['data_matrix'].to(device)
                token_ids = batch['token_ids'].to(device)
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

            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    data_matrix = batch['data_matrix'].to(device)
                    token_ids = batch['token_ids'].to(device)
                    input_ids = token_ids[:, :-1]
                    target_ids = token_ids[:, 1:]
                    logits = model(data_matrix, input_ids)
                    loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
                    val_loss += loss.item()
                    val_batches += 1
            avg_val = val_loss / max(val_batches, 1)

            training_log.append({'epoch': epoch + 1, 'train_loss': round(avg_loss, 4), 'val_loss': round(avg_val, 4)})
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: train={avg_loss:.4f} val={avg_val:.4f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
    except Timeout:
        print("Training timed out, using model so far")

    # Save training log
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_training.log', 'w') as f:
        json.dump(training_log, f, indent=2)

    # Evaluate on ALL 120 FSReD equations
    print("\nEvaluating on 120 FSReD equations...")
    model.eval()
    results_per_eq = []
    difficulty_metrics = {'easy': [], 'medium': [], 'hard': []}

    for i, (expr_str, desc) in enumerate(fsred_equations):
        t0 = time.time()

        # Get difficulty
        if i < 30:
            difficulty = 'easy'
        elif i < 70:
            difficulty = 'medium'
        else:
            difficulty = 'hard'

        # Generate test data
        try:
            X, y = _generate_data_points(expr_str, n_points=100, seed=42 + i)
        except Exception:
            X, y = np.random.randn(100, 2).astype(np.float32), np.zeros(100, dtype=np.float32)

        # Prepare input
        n_vars = X.shape[1]
        X_norm = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)
        data_matrix = np.column_stack([X_norm, y_norm[:, None]])
        # Pad to 10 cols
        if data_matrix.shape[1] < 10:
            data_matrix = np.hstack([data_matrix, np.zeros((data_matrix.shape[0], 10 - data_matrix.shape[1]))])
        # Pad/truncate to 64 rows
        if len(data_matrix) < 64:
            data_matrix = np.vstack([data_matrix, np.zeros((64 - len(data_matrix), 10))])
        data_matrix = data_matrix[:64]

        dm_tensor = torch.tensor(data_matrix, dtype=torch.float32).unsqueeze(0).to(device)

        # Generate prediction
        with torch.no_grad():
            try:
                pred_ids = model.generate_greedy(dm_tensor, max_len=64)
                pred_tokens = pred_ids[0].cpu().tolist()
                pred_str = tokenizer.decode(pred_tokens)
            except Exception:
                pred_str = "0"
                pred_tokens = [1, 76, 2]  # BOS, 0, EOS

        true_tokens = tokenizer.encode(expr_str)
        elapsed = time.time() - t0

        # Compute metrics
        metrics = compute_all_metrics(
            pred_str=pred_str, true_str=expr_str,
            X=X, y_true=y,
            pred_tokens=pred_tokens, true_tokens=true_tokens,
            inference_time=elapsed,
        )

        eq_result = {
            'equation_id': i,
            'true_expr': expr_str,
            'pred_expr': pred_str,
            'difficulty': difficulty,
            'metrics': {k: (float(v) if v is not None else None) for k, v in metrics.items()},
        }
        results_per_eq.append(eq_result)
        difficulty_metrics[difficulty].append(metrics)

        if (i + 1) % 30 == 0:
            print(f"  Evaluated {i+1}/120 equations")

    signal.alarm(0)

    # Aggregate metrics by difficulty
    aggregate = {}
    for diff in ['easy', 'medium', 'hard']:
        mets = difficulty_metrics[diff]
        if mets:
            sr = np.mean([m['solution_rate'] for m in mets])
            r2_vals = [m['r_squared'] for m in mets if m['r_squared'] is not None]
            ned_vals = [m['ned'] for m in mets]
            sa_vals = [m['symbolic_accuracy'] for m in mets if m['symbolic_accuracy'] is not None]
            aggregate[diff] = {
                'count': len(mets),
                'solution_rate': round(float(sr), 4),
                'mean_r_squared': round(float(np.mean(r2_vals)), 4) if r2_vals else None,
                'mean_ned': round(float(np.mean(ned_vals)), 4),
                'mean_symbolic_accuracy': round(float(np.mean(sa_vals)), 4) if sa_vals else None,
            }

    # Overall aggregate
    all_mets = difficulty_metrics['easy'] + difficulty_metrics['medium'] + difficulty_metrics['hard']
    all_sr = np.mean([m['solution_rate'] for m in all_mets])
    all_r2 = [m['r_squared'] for m in all_mets if m['r_squared'] is not None]
    all_ned = [m['ned'] for m in all_mets]

    # Comparison with published SymbolicGPT
    comparison = {
        'AR_Baseline': {
            'overall_solution_rate': round(float(all_sr), 4),
            'easy_solution_rate': aggregate['easy']['solution_rate'],
            'medium_solution_rate': aggregate['medium']['solution_rate'],
            'hard_solution_rate': aggregate['hard']['solution_rate'],
            'mean_r_squared': round(float(np.mean(all_r2)), 4) if all_r2 else None,
            'mean_ned': round(float(np.mean(all_ned)), 4),
        },
        'SymbolicGPT_published': {
            'overall_solution_rate': 0.35,
            'easy_solution_rate': 0.62,
            'medium_solution_rate': 0.33,
            'hard_solution_rate': 0.18,
            'mean_r_squared': None,
            'mean_ned': None,
            'source': 'valipour2021symbolicgpt',
        },
    }

    baseline_results = {
        'model': 'AR-Baseline',
        'training_epochs': len(training_log),
        'best_val_loss': round(best_val_loss, 4),
        'per_equation': results_per_eq,
        'aggregate_by_difficulty': aggregate,
        'comparison': comparison,
    }

    with open('results/baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\nResults saved to results/baseline_results.json")
    print(f"Overall solution rate: {all_sr:.4f}")
    for diff in ['easy', 'medium', 'hard']:
        print(f"  {diff}: solution_rate={aggregate[diff]['solution_rate']}")
    print(f"SymbolicGPT published: easy=0.62, medium=0.33, hard=0.18")


if __name__ == '__main__':
    main()
