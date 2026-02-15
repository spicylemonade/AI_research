"""Evaluate the trained AR baseline and produce benchmark results.

Generates figures/baseline_loss.png and results/baseline_results.json.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.physics_generator import PhysicsDataset
from data.equations import get_training_equations, get_held_out_equations, get_equations_by_tier
from data.tokenizer import ExprTokenizer, PAD_IDX, SOS_IDX, EOS_IDX
from models.ar_baseline import ARBaseline, ARBaselineConfig
from evaluation.metrics import symbolic_equivalence, numeric_r2, token_edit_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Publication-quality figure setup
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
mpl.rcParams.update({
    'figure.figsize': (8, 5),
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'font.family': 'serif',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = ARBaseline(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def evaluate_on_equations(model, equations, device, n_samples_per_eq=20,
                          n_points=50, noise_level=0.0, seed=42):
    """Evaluate model on a set of equations, getting symbolic equivalence and R²."""
    tokenizer = ExprTokenizer()
    results = []

    for eq in equations:
        ds = PhysicsDataset(
            equations=[eq], n_samples=n_samples_per_eq, n_points=n_points,
            noise_level=noise_level, seed=seed,
        )
        if len(ds) == 0:
            continue

        eq_sym_correct = 0
        eq_r2_scores = []
        eq_latencies = []

        for i in range(min(len(ds), n_samples_per_eq)):
            sample = ds[i]
            obs = sample['observations'].unsqueeze(0).to(device)
            obs_mask = sample['obs_mask'].unsqueeze(0).to(device)
            true_tokens = sample['tokens'][:sample['token_len']].tolist()

            # Generate
            start_time = time.time()
            with torch.no_grad():
                generated = model.generate(obs, obs_mask, max_len=64, temperature=0.0)
            latency = time.time() - start_time
            eq_latencies.append(latency)

            gen_tokens = generated[0].cpu().tolist()
            # Strip after EOS
            if EOS_IDX in gen_tokens:
                gen_tokens = gen_tokens[:gen_tokens.index(EOS_IDX) + 1]

            # Decode
            try:
                pred_expr = tokenizer.decode(gen_tokens)
                true_expr = tokenizer.decode(true_tokens)

                # Symbolic equivalence
                is_equiv = symbolic_equivalence(pred_expr, true_expr)
                if is_equiv:
                    eq_sym_correct += 1

                # Numeric R²
                r2 = numeric_r2(pred_expr, true_expr)
                eq_r2_scores.append(r2)
            except Exception:
                eq_r2_scores.append(-1.0)

        results.append({
            'equation_id': eq.id,
            'equation_name': eq.name,
            'tier': eq.tier,
            'held_out': eq.held_out,
            'n_samples': min(len(ds), n_samples_per_eq),
            'symbolic_accuracy': eq_sym_correct / max(min(len(ds), n_samples_per_eq), 1),
            'mean_r2': float(np.mean([r for r in eq_r2_scores if r >= 0])) if any(r >= 0 for r in eq_r2_scores) else -1.0,
            'mean_latency_ms': float(np.mean(eq_latencies)) * 1000 if eq_latencies else 0,
        })

    return results


def plot_training_loss(log_path, output_path):
    """Plot training loss curves from the training log."""
    with open(log_path) as f:
        log = json.load(f)

    # Extract training loss per step
    steps = [s['global_step'] for s in log['train_steps']]
    losses = [s['loss'] for s in log['train_steps']]

    # Validation loss per epoch
    val_epochs = [v['epoch'] for v in log['val_epochs']]
    val_losses = [v['val_loss'] for v in log['val_epochs']]

    colors = sns.color_palette("deep")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Training loss
    ax1.plot(steps, losses, color=colors[0], alpha=0.7, linewidth=1.0,
             label='Training Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_yscale('log')
    ax1.set_title('AR Baseline Training Dynamics')

    # Validation loss on secondary x-axis mapped to steps
    steps_per_epoch = steps[-1] / val_epochs[-1] if val_epochs else 1
    val_steps = [e * steps_per_epoch for e in val_epochs]
    ax1.plot(val_steps, val_losses, color=colors[1], linewidth=2.0,
             marker='o', markersize=5, label='Validation Loss')

    ax1.legend(loc='upper right', frameon=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {output_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt_path = 'checkpoints/ar_baseline_best.pt'
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Train first.")
        return

    model, config = load_model(ckpt_path, device)
    print(f"Loaded model: {model.count_parameters():,} params")

    # Evaluate on training equations
    print("\nEvaluating on training equations...")
    train_eqs = get_training_equations()
    train_results = evaluate_on_equations(
        model, train_eqs, device, n_samples_per_eq=10, seed=42
    )

    # Aggregate by tier
    tier_results = {}
    for r in train_results:
        t = r['tier']
        if t not in tier_results:
            tier_results[t] = {'sym_accs': [], 'r2s': [], 'latencies': []}
        tier_results[t]['sym_accs'].append(r['symbolic_accuracy'])
        tier_results[t]['r2s'].append(r['mean_r2'])
        tier_results[t]['latencies'].append(r['mean_latency_ms'])

    print("\nPer-Tier Symbolic Equivalence Accuracy (AR Baseline):")
    tier_summary = {}
    for t in sorted(tier_results.keys()):
        acc = np.mean(tier_results[t]['sym_accs'])
        r2 = np.mean([r for r in tier_results[t]['r2s'] if r >= 0])
        lat = np.mean(tier_results[t]['latencies'])
        tier_summary[t] = {
            'symbolic_accuracy': float(acc),
            'mean_r2': float(r2),
            'mean_latency_ms': float(lat),
        }
        print(f"  Tier {t}: Acc={acc*100:.1f}%, R²={r2:.4f}, Latency={lat:.1f}ms")

    overall_acc = np.mean([r['symbolic_accuracy'] for r in train_results])
    overall_r2 = np.mean([r['mean_r2'] for r in train_results if r['mean_r2'] >= 0])
    overall_latency = np.mean([r['mean_latency_ms'] for r in train_results])

    # GPU memory
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # Save results
    baseline_results = {
        'model': 'AR Baseline',
        'total_params': model.count_parameters(),
        'overall_symbolic_accuracy': float(overall_acc),
        'overall_mean_r2': float(overall_r2),
        'overall_mean_latency_ms': float(overall_latency),
        'peak_gpu_memory_gb': float(gpu_mem),
        'tier_results': {str(k): v for k, v in tier_summary.items()},
        'per_equation': train_results,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)
    print(f"\nResults saved to results/baseline_results.json")

    # Plot training loss
    log_path = 'results/baseline_training_log.json'
    if os.path.exists(log_path):
        plot_training_loss(log_path, 'figures/baseline_loss.png')

    print(f"\nOverall: Acc={overall_acc*100:.1f}%, R²={overall_r2:.4f}")


if __name__ == '__main__':
    main()
