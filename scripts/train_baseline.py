"""
Train and evaluate the autoregressive baseline model.

Optimized for CPU-only execution within a 10-minute budget:
- batch_size=2 with 30 support points and max_seq_len=24
- ~0.085s per step -> 5000 steps in ~7 minutes
- Remaining ~3 minutes for Feynman benchmark evaluation
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.generator import EquationGenerator, BINARY_OPS, UNARY_OPS
from src.model.encoder import SetTransformerEncoder, batch_float_to_ieee754
from src.model.decoder import (
    AutoregressiveDecoder, VOCAB, VOCAB_SIZE,
    tokens_to_ids, ids_to_tokens, ID_TO_TOKEN
)
from src.eval.metrics import r_squared, symbolic_equivalence, tree_edit_distance, equation_complexity

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Use a moderate number of threads for CPU efficiency
torch.set_num_threads(4)


# Timeout handler
class TrainTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TrainTimeout()


def prepare_batch(equations, encoder, max_seq_len=24):
    """Prepare a training batch from generated equations.

    Returns (encoder_input, decoder_input, decoder_target, padding_mask)
    """
    batch_enc = []
    batch_dec_in = []
    batch_dec_tgt = []

    for eq in equations:
        # Encode observations
        X = np.array(eq['support_points_x'])
        y = np.array(eq['support_points_y'])
        # Use X's actual column count so encode_observations pads correctly
        # to max_variables, giving consistent output dimension
        num_vars = X.shape[1]

        encoded = encoder.encode_observations(X, y, num_vars)  # [N, input_dim]
        batch_enc.append(encoded)

        # Prepare decoder tokens
        prefix_tokens = eq['prefix_tokens']
        token_ids = tokens_to_ids(prefix_tokens)

        # Decoder input: BOS + tokens (teacher forcing)
        dec_in = [VOCAB['BOS']] + token_ids
        # Decoder target: tokens + EOS
        dec_tgt = token_ids + [VOCAB['EOS']]

        # Truncate/pad to max_seq_len
        if len(dec_in) > max_seq_len:
            dec_in = dec_in[:max_seq_len]
            dec_tgt = dec_tgt[:max_seq_len]
        else:
            pad_len = max_seq_len - len(dec_in)
            dec_in = dec_in + [VOCAB['PAD']] * pad_len
            dec_tgt = dec_tgt + [VOCAB['PAD']] * pad_len

        batch_dec_in.append(dec_in)
        batch_dec_tgt.append(dec_tgt)

    # Stack encoder inputs - all same N from generator
    enc_input = torch.stack(batch_enc)  # [B, N, input_dim]

    dec_in = torch.tensor(batch_dec_in, dtype=torch.long)
    dec_tgt = torch.tensor(batch_dec_tgt, dtype=torch.long)

    # Padding mask (True where PAD)
    pad_mask = dec_in == VOCAB['PAD']

    return enc_input, dec_in, dec_tgt, pad_mask


def evaluate_on_feynman(encoder, decoder, benchmark_path, num_test_points=200):
    """Evaluate model on Feynman benchmark equations."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    equations = benchmark['equations']
    results = []

    encoder.eval()
    decoder.eval()

    for i, eq in enumerate(equations):
        try:
            # Parse the formula and generate data
            formula_python = eq['formula_python']
            variables = eq['variables']
            num_vars = eq['num_variables']

            # Generate support points
            rng = np.random.RandomState(SEED + i)
            # Use positive ranges for physical variables to avoid domain issues
            X = rng.uniform(0.1, 5.0, size=(num_test_points, num_vars))

            # Create variable mapping for evaluating the ground truth formula
            safe_globals = {
                '__builtins__': {},
                'pi': np.pi, 'e': np.e,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'abs': np.abs,
            }

            for vi, var in enumerate(variables):
                var_name = var['name']
                safe_globals[var_name] = X[:, vi]

            y = eval(formula_python, safe_globals)

            # Check for valid data
            if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.any(np.abs(y) > 1e6):
                results.append({
                    'id': eq['id'],
                    'name': eq['name'],
                    'tier': eq['difficulty_tier'],
                    'exact_match': False,
                    'r_squared': -1.0,
                    'tree_edit_distance': 1.0,
                    'predicted': [],
                    'error': 'invalid_data'
                })
                continue

            # Encode observations through the model encoder
            encoded = encoder.encode_observations(X, y, num_vars)
            enc_input = encoded.unsqueeze(0)  # [1, N, D]

            with torch.no_grad():
                z = encoder(enc_input)
                # Use greedy decoding for speed during evaluation
                pred_tokens = decoder.generate_beam(z, beam_width=5, max_length=64)

            pred = pred_tokens[0]

            # Get ground truth prefix tokens by parsing the symbolic form
            true_prefix = eq['formula_symbolic']
            true_tokens = parse_symbolic_to_prefix(true_prefix)

            # Compute R-squared using predicted expression on test points
            # The predicted expression uses x1, x2, ... variable naming
            X_test = rng.uniform(0.1, 5.0, size=(1000, num_vars))

            # Recompute y_test with actual variable names
            for vi, var in enumerate(variables):
                safe_globals[var['name']] = X_test[:, vi]
            y_test = eval(formula_python, safe_globals)

            if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
                r2 = -1.0
            else:
                # The model predicts with x1, x2... variables
                # r_squared from metrics expects x1, x2... variables in X_test columns
                r2 = r_squared(pred, X_test, y_test, num_vars)

            # Exact match via symbolic equivalence
            exact = symbolic_equivalence(pred, true_tokens) if len(pred) > 0 else False
            if exact is None:
                exact = False

            # Tree edit distance
            ted = tree_edit_distance(pred, true_tokens)

            results.append({
                'id': eq['id'],
                'name': eq['name'],
                'tier': eq['difficulty_tier'],
                'exact_match': bool(exact),
                'r_squared': float(r2),
                'tree_edit_distance': float(ted),
                'predicted': pred,
                'true_tokens': true_tokens,
                'pred_complexity': len(pred),
                'true_complexity': len(true_tokens),
            })

        except Exception as e:
            results.append({
                'id': eq['id'],
                'name': eq['name'],
                'tier': eq['difficulty_tier'],
                'exact_match': False,
                'r_squared': -1.0,
                'tree_edit_distance': 1.0,
                'predicted': [],
                'error': str(e)
            })

        if (i + 1) % 20 == 0:
            n_exact = sum(1 for r in results if r['exact_match'])
            n_good_r2 = sum(1 for r in results if r['r_squared'] > 0.9)
            print(f"  Evaluated {i+1}/{len(equations)}: exact={n_exact}, R^2>0.9={n_good_r2}")

    return results


def parse_symbolic_to_prefix(symbolic: str) -> list:
    """Parse symbolic expression like 'mul(x1, pow(x2, 2))' into prefix token list."""

    def parse_expr(s, pos):
        # Skip whitespace and commas
        while pos < len(s) and s[pos] in ' ,':
            pos += 1

        if pos >= len(s):
            return [], pos

        # Check if it starts with a letter/digit -> could be function or leaf
        start = pos
        while pos < len(s) and (s[pos].isalnum() or s[pos] in '_.-'):
            pos += 1

        name = s[start:pos]

        # Skip whitespace
        while pos < len(s) and s[pos] in ' ':
            pos += 1

        if pos < len(s) and s[pos] == '(':
            # Function call
            pos += 1  # skip (
            tokens_list = [name]

            # Parse arguments
            while pos < len(s) and s[pos] != ')':
                while pos < len(s) and s[pos] in ' ,':
                    pos += 1
                if pos < len(s) and s[pos] == ')':
                    break
                child_tokens, pos = parse_expr(s, pos)
                tokens_list.extend(child_tokens)

            if pos < len(s) and s[pos] == ')':
                pos += 1  # skip )

            return tokens_list, pos
        else:
            # Leaf: variable or constant
            return [name], pos

    result, _ = parse_expr(symbolic, 0)
    return result


def main():
    # Set timeout for entire script (10 minutes)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(600)  # 10 min

    print("=" * 60)
    print("BASELINE TRAINING: Autoregressive Transformer")
    print("=" * 60)

    # Initialize models
    encoder = SetTransformerEncoder(embed_dim=256, num_heads=8, num_layers=2, num_inducing=16)
    decoder = AutoregressiveDecoder(embed_dim=256, num_heads=8, num_layers=4, ff_dim=512)

    total_params = encoder.count_parameters() + decoder.count_parameters()
    print(f"Encoder params: {encoder.count_parameters():,}")
    print(f"Decoder params: {decoder.count_parameters():,}")
    print(f"Total params: {total_params:,}")

    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB['PAD'])

    # Data generator - reduced support points for CPU speed
    num_support_points = 30
    gen = EquationGenerator(seed=SEED, max_depth=4, max_variables=3, num_support_points=num_support_points)

    # Training parameters - optimized for CPU within 10-minute budget
    # With batch_size=2, N=30, T=24: ~0.085s/step -> 5000 steps in ~7 min
    batch_size = 2
    num_steps = 5000
    max_seq_len = 24
    log_interval = 100

    # Training loop
    print(f"\nTraining for {num_steps} steps, batch_size={batch_size}, "
          f"support_points={num_support_points}, max_seq_len={max_seq_len}")
    print(f"Effective samples seen: {num_steps * batch_size:,}")
    losses = []
    encoder.train()
    decoder.train()

    try:
        start_time = time.time()

        for step in range(1, num_steps + 1):
            # Generate batch on-the-fly
            equations = gen.generate_batch(batch_size)
            if len(equations) < batch_size:
                continue
            equations = equations[:batch_size]

            # Prepare batch
            enc_input, dec_in, dec_tgt, pad_mask = prepare_batch(equations, encoder, max_seq_len)

            # Forward
            z = encoder(enc_input)
            logits = decoder(dec_in, z, tgt_padding_mask=pad_mask)

            # Loss
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_tgt.reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            losses.append(loss.item())

            if step % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                est_total = num_steps / steps_per_sec
                remaining = est_total - elapsed
                print(f"  Step {step}/{num_steps} | Loss: {avg_loss:.4f} | "
                      f"{steps_per_sec:.1f} steps/s | {elapsed:.0f}s elapsed | "
                      f"~{remaining:.0f}s remaining")

                # If we estimate total time > 480s (8 min), reduce steps
                # to leave time for evaluation
                if step == 100 and est_total > 480:
                    new_num_steps = int(steps_per_sec * 420)  # 7 min for training
                    if new_num_steps < num_steps:
                        num_steps = max(new_num_steps, step + 100)
                        print(f"  -> Adjusting num_steps to {num_steps} to fit time budget")

        signal.alarm(0)  # cancel timeout
    except TrainTimeout:
        print("\nTraining timed out! Using model at current state.")
        signal.alarm(0)

    total_train_time = time.time() - start_time
    print(f"\nTraining completed in {total_train_time:.0f}s ({total_train_time/60:.1f} min)")
    final_loss = float(np.mean(losses[-100:])) if len(losses) >= 100 else float(np.mean(losses))
    print(f"Final loss: {final_loss:.4f}")
    print(f"Total steps completed: {len(losses)}")

    # Save loss figure
    print("\nSaving training loss figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        'figure.figsize': (8, 5), 'figure.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'axes.labelsize': 13,
        'axes.titlesize': 14, 'axes.titleweight': 'bold',
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'font.family': 'serif',
        'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    # Smooth the loss curve
    window = min(50, max(1, len(losses) // 10))
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
    else:
        smoothed = np.array(losses)

    colors = sns.color_palette("deep")
    ax.plot(range(len(losses)), losses, alpha=0.2, color=colors[0], linewidth=0.5)
    ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
            color=colors[0], linewidth=2, label=f'Smoothed (window={window})')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Autoregressive Baseline Training Convergence')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='0.8')

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'baseline_loss.png'), dpi=300)
    plt.savefig(os.path.join(fig_dir, 'baseline_loss.pdf'))
    plt.close()
    print(f"  Saved {os.path.join(fig_dir, 'baseline_loss.png')}")

    # Re-set alarm for evaluation phase (3 min)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(180)

    # Evaluate on Feynman benchmark
    print("\nEvaluating on Feynman benchmark (120 equations)...")
    benchmark_path = os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'feynman_equations.json')

    try:
        eval_results = evaluate_on_feynman(encoder, decoder, benchmark_path)
        signal.alarm(0)
    except TrainTimeout:
        print("Evaluation timed out! Using partial results.")
        signal.alarm(0)
        eval_results = []

    # Aggregate results
    n_total = len(eval_results)
    n_exact = sum(1 for r in eval_results if r['exact_match'])
    n_r2_good = sum(1 for r in eval_results if r['r_squared'] > 0.9)
    r2_scores = [r['r_squared'] for r in eval_results]

    print(f"\n{'=' * 60}")
    print("BASELINE RESULTS")
    print(f"{'=' * 60}")
    print(f"Total equations evaluated: {n_total}")
    if n_total > 0:
        print(f"Exact match rate: {n_exact}/{n_total} ({n_exact / n_total * 100:.1f}%)")
        print(f"R^2 > 0.9: {n_r2_good}/{n_total} ({n_r2_good / n_total * 100:.1f}%)")
        print(f"Mean R^2: {np.mean(r2_scores):.4f}")

    # Per-tier breakdown
    tiers = ['trivial', 'simple', 'moderate', 'complex', 'multi_step']
    print(f"\nPer-tier breakdown:")
    tier_summary = {}
    for tier in tiers:
        tier_results = [r for r in eval_results if r['tier'] == tier]
        tier_n = len(tier_results)
        if tier_n == 0:
            tier_summary[tier] = {'n': 0, 'exact_match': 0, 'exact_match_rate': 0.0,
                                  'r2_above_0.9': 0, 'mean_r2': 0.0}
            continue
        tier_exact = sum(1 for r in tier_results if r['exact_match'])
        tier_r2 = [r['r_squared'] for r in tier_results]
        tier_r2_good = sum(1 for r2 in tier_r2 if r2 > 0.9)
        print(f"  {tier:12s}: {tier_exact}/{tier_n} exact ({tier_exact / max(1, tier_n) * 100:.0f}%), "
              f"R^2>0.9: {tier_r2_good}/{tier_n}, mean R^2: {np.mean(tier_r2):.3f}")
        tier_summary[tier] = {
            'n': tier_n,
            'exact_match': tier_exact,
            'exact_match_rate': tier_exact / max(1, tier_n),
            'r2_above_0.9': tier_r2_good,
            'mean_r2': float(np.mean(tier_r2)),
        }

    # Comparison with published numbers
    published = {
        'NeSymReS (Biggio 2021, 10M pretrain)': {'exact_match_rate': 0.72},
        'ODEFormer (d\'Ascoli 2024, 50M pretrain)': {'exact_match_rate': 0.85},
        'TPSR (Shojaee 2023)': {'exact_match_rate': 0.80},
        'PySR (Cranmer 2023)': {'exact_match_rate': 0.78},
        'AI Feynman (Udrescu 2020)': {'exact_match_rate': 1.00},
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Convert eval_results so they are JSON-serializable
    serializable_results = []
    for r in eval_results:
        sr = {}
        for k, v in r.items():
            sr[k] = v
        serializable_results.append(sr)

    baseline_results = {
        'model': 'autoregressive_baseline',
        'total_params': total_params,
        'training': {
            'num_steps': len(losses),
            'batch_size': batch_size,
            'num_support_points': num_support_points,
            'max_seq_len': max_seq_len,
            'final_loss': final_loss,
            'training_time_seconds': total_train_time,
        },
        'overall': {
            'n': n_total,
            'exact_match_rate': n_exact / max(1, n_total),
            'exact_match_count': n_exact,
            'mean_r_squared': float(np.mean(r2_scores)) if r2_scores else 0.0,
            'r2_above_0.9': n_r2_good / max(1, n_total),
        },
        'per_tier': tier_summary,
        'per_equation': serializable_results,
        'published_comparison': published,
        'loss_history': losses,
    }

    results_path = os.path.join(results_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save model checkpoint
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, 'baseline_checkpoint.pt')
    torch.save({
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'config': {
            'embed_dim': 256,
            'encoder_layers': 2,
            'decoder_layers': 4,
            'num_heads': 8,
            'ff_dim': 512,
        }
    }, ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")


if __name__ == '__main__':
    main()
