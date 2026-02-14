"""Evaluate PhysDiffuse with Test-Time Training on full test set."""

import torch
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.feynman_loader import generate_benchmark_data
from data.tokenizer import decode, PAD_ID, MAX_SEQ_LEN
from model.phys_diffuse import create_phys_diffuse
from model.ttt import ttt_generate
from model.postprocess import postprocess_candidates
from evaluation.metrics import compute_all_metrics, tier_stratified_report

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VARS = 10
N_POINTS = 50


def prepare_benchmark():
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


def run_ttt_evaluation():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    checkpoint_path = 'results/checkpoints/phys_diffuse.pt'
    if not os.path.exists(checkpoint_path):
        print("ERROR: PhysDiffuse checkpoint not found.")
        return

    tables, meta = prepare_benchmark()
    print(f"Loaded {len(meta)} benchmark equations")

    # Create model
    model = create_phys_diffuse(d_model=512, n_heads=8, device=DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    results_list = []
    ttt_times = []
    qualitative = {1: [], 2: [], 3: [], 4: []}

    print(f"\nEvaluating PhysDiffuse+TTT on {len(meta)} equations...")
    total_start = time.time()

    for i in range(len(meta)):
        eq = meta[i]
        obs = torch.tensor(tables[i:i+1], device=DEVICE)

        t0 = time.time()
        pred_ids = ttt_generate(
            model, obs, T=64, R=2, n_samples=128,
            n_ttt_steps=96, ttt_augmentations=64,
            ttt_rank=16, ttt_lr=1e-4, verbose=False
        )
        ttt_time = time.time() - t0
        ttt_times.append(ttt_time)

        pred_tokens = decode(pred_ids)
        gt_tokens = eq['prefix']

        # Post-process
        if pred_tokens:
            processed = postprocess_candidates([pred_tokens], eq['table'], top_k=1)
            if processed and processed[0]['mse'] < float('inf'):
                pred_tokens = processed[0]['tokens']

        metrics = compute_all_metrics(
            pred_tokens, gt_tokens, eq['table'],
            variable_units=eq.get('units', {}),
            gt_constants=eq.get('constants', {}),
        )

        tier = eq['tier']
        result = {
            'id': eq['id'],
            'name': eq['name'],
            'tier': tier,
            'formula': eq['formula'],
            'predicted': ' '.join(pred_tokens),
            'ground_truth': ' '.join(gt_tokens),
            'ttt_time_seconds': ttt_time,
            **metrics,
        }
        results_list.append(result)

        if tier in qualitative and len(qualitative[tier]) < 3:
            qualitative[tier].append({
                'name': eq['name'],
                'formula': eq['formula'],
                'predicted': ' '.join(pred_tokens),
                'ground_truth': ' '.join(gt_tokens),
                'exact_match': metrics['exact_match'],
                'r2': metrics['r2'],
            })

        status = "MATCH" if metrics['exact_match'] else f"NED={metrics['ned']:.2f}, R²={metrics['r2']:.3f}"
        print(f"  [{eq['id']}] {eq['name']}: {status} (TTT: {ttt_time:.1f}s)")

    total_time = time.time() - total_start
    tiers = [eq['tier'] for eq in meta]
    report = tier_stratified_report(results_list, tiers)

    full_results = {
        'n_equations': len(meta),
        'total_time_seconds': total_time,
        'mean_ttt_time_per_equation': np.mean(ttt_times),
        'report': report,
        'qualitative_examples': qualitative,
        'per_equation': results_list,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/phys_diffuse_ttt_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print("\n=== PhysDiffuse+TTT Results ===")
    for tier_key, tier_data in sorted(report.items()):
        print(f"  {tier_key}: EM={tier_data['exact_match_rate']:.1%}, "
              f"NED={tier_data['mean_ned']:.3f}, R²={tier_data['mean_r2']:.3f}, "
              f"DimOK={tier_data['dim_consistency_rate']:.1%}")
    print(f"\nTotal inference time: {total_time:.0f}s ({total_time/3600:.2f}h)")
    print(f"Mean TTT time per equation: {np.mean(ttt_times):.1f}s")


if __name__ == '__main__':
    run_ttt_evaluation()
