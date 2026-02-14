"""Run the full ablation study for PhysDiffuse."""

import torch
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.feynman_loader import generate_benchmark_data
from data.tokenizer import decode, PAD_ID, MAX_SEQ_LEN
from model.phys_diffuse import PhysDiffuse, create_phys_diffuse
from model.postprocess import postprocess_candidates
from evaluation.metrics import compute_all_metrics, tier_stratified_report

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VARS = 10
N_POINTS = 50


def prepare_benchmark():
    """Load benchmark equations with padding."""
    benchmark = generate_benchmark_data(n_points=N_POINTS, seed=SEED)
    tables = []
    meta = []
    for eq in benchmark:
        table = eq['table']
        if table.shape[0] < N_POINTS:
            pad = np.zeros((N_POINTS - table.shape[0], table.shape[1]))
            table = np.vstack([table, pad])
        elif table.shape[0] > N_POINTS:
            table = table[:N_POINTS]
        if table.shape[1] < MAX_VARS + 1:
            pad = np.zeros((table.shape[0], MAX_VARS + 1 - table.shape[1]))
            table = np.hstack([table, pad])
        tables.append(table)
        meta.append(eq)
    return np.array(tables, dtype=np.float32), meta


def evaluate_config(model, tables, meta, config):
    """Evaluate a single ablation configuration."""
    T = config.get('T', 32)
    R = config.get('R', 2)
    n_samples = config.get('n_samples', 64)
    use_mcts = config.get('use_mcts', False)
    use_augmentation = config.get('use_augmentation', False)
    use_postprocess = config.get('use_postprocess', True)

    results_list = []
    model.eval()

    for i in range(len(meta)):
        eq = meta[i]
        obs = torch.tensor(tables[i:i+1], device=DEVICE)

        pred_ids = model.generate(
            obs, T=T, R=R, n_samples=n_samples,
            use_mcts=use_mcts, use_augmentation=use_augmentation
        )
        pred_tokens = decode(pred_ids)
        gt_tokens = eq['prefix']

        if use_postprocess and pred_tokens:
            processed = postprocess_candidates([pred_tokens], eq['table'], top_k=1)
            if processed and processed[0]['mse'] < float('inf'):
                pred_tokens = processed[0]['tokens']

        metrics = compute_all_metrics(
            pred_tokens, gt_tokens, eq['table'],
            variable_units=eq.get('units', {}),
            gt_constants=eq.get('constants', {}),
        )

        result = {
            'id': eq['id'],
            'name': eq['name'],
            'tier': eq['tier'],
            **metrics,
        }
        results_list.append(result)

    tiers = [eq['tier'] for eq in meta]
    report = tier_stratified_report(results_list, tiers)
    return results_list, report


def run_ablations():
    """Run all ablation configurations."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model checkpoint
    checkpoint_path = 'results/checkpoints/phys_diffuse.pt'
    if not os.path.exists(checkpoint_path):
        print("ERROR: PhysDiffuse checkpoint not found. Train first.")
        return

    tables, meta = prepare_benchmark()
    print(f"Loaded {len(meta)} benchmark equations")

    # Define ablation configurations
    configs = {
        'full_system': {
            'T': 64, 'R': 2, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': True,
            'use_postprocess': True,
            'description': 'Full PhysDiffuse (T=64, R=2, aug, postprocess)',
        },
        'no_refinement': {
            'T': 1, 'R': 1, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': False,
            'use_postprocess': True,
            'description': 'No refinement (T=1, single pass)',
        },
        'T8': {
            'T': 8, 'R': 1, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': False,
            'use_postprocess': True,
            'description': 'Refinement T=8',
        },
        'T16': {
            'T': 16, 'R': 1, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': False,
            'use_postprocess': True,
            'description': 'Refinement T=16',
        },
        'T32': {
            'T': 32, 'R': 1, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': False,
            'use_postprocess': True,
            'description': 'Refinement T=32',
        },
        'T64_no_cold_restart': {
            'T': 64, 'R': 1, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': False,
            'use_postprocess': True,
            'description': 'T=64 without cold restart (R=1)',
        },
        'with_mcts': {
            'T': 64, 'R': 2, 'n_samples': 128,
            'use_mcts': True, 'use_augmentation': True,
            'use_postprocess': True,
            'description': 'Full + MCTS-guided selection',
        },
        'no_augmentation': {
            'T': 64, 'R': 2, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': False,
            'use_postprocess': True,
            'description': 'Full - augmentation',
        },
        'no_postprocess': {
            'T': 64, 'R': 2, 'n_samples': 128,
            'use_mcts': False, 'use_augmentation': True,
            'use_postprocess': False,
            'description': 'Full - post-processing',
        },
        'minimal': {
            'T': 16, 'R': 1, 'n_samples': 32,
            'use_mcts': False, 'use_augmentation': False,
            'use_postprocess': False,
            'description': 'Minimal: T=16, R=1, 32 samples, no postprocess',
        },
    }

    # Create model and load checkpoint
    model = create_phys_diffuse(d_model=512, n_heads=8, device=DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    all_results = {}
    for name, config in configs.items():
        print(f"\n--- Ablation: {config['description']} ---")
        t0 = time.time()
        per_eq, report = evaluate_config(model, tables, meta, config)
        elapsed = time.time() - t0

        all_results[name] = {
            'config': config,
            'report': report,
            'per_equation': per_eq,
            'time_seconds': elapsed,
        }

        overall = report.get('overall', {})
        print(f"  EM={overall.get('exact_match_rate', 0):.1%}, "
              f"NED={overall.get('mean_ned', 1):.3f}, "
              f"R²={overall.get('mean_r2', -1):.3f}, "
              f"Time={elapsed:.0f}s")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_study.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary markdown
    summary_lines = ["# Ablation Study Results\n"]
    summary_lines.append("| Configuration | EM | NED | R² | DimOK | Time |")
    summary_lines.append("|---|---|---|---|---|---|")

    for name, data in all_results.items():
        overall = data['report'].get('overall', {})
        summary_lines.append(
            f"| {data['config']['description']} | "
            f"{overall.get('exact_match_rate', 0):.1%} | "
            f"{overall.get('mean_ned', 1):.3f} | "
            f"{overall.get('mean_r2', -1):.3f} | "
            f"{overall.get('dim_consistency_rate', 0):.1%} | "
            f"{data['time_seconds']:.0f}s |"
        )

    with open('results/ablation_summary.md', 'w') as f:
        f.write('\n'.join(summary_lines))

    print("\n=== Ablation study complete ===")
    print(f"Results saved to results/ablation_study.json")
    print(f"Summary saved to results/ablation_summary.md")


if __name__ == '__main__':
    run_ablations()
