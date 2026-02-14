"""Generalization experiments: test on OOD equation families."""

import torch
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.feynman_loader import generate_ood_data
from data.tokenizer import decode, PAD_ID, MAX_SEQ_LEN
from model.phys_diffuse import create_phys_diffuse
from model.ttt import ttt_generate
from model.postprocess import postprocess_candidates
from evaluation.metrics import compute_all_metrics, tier_stratified_report
from baselines.autoregressive import AutoregressiveBaseline

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VARS = 10
N_POINTS = 50


def prepare_ood_data():
    """Load OOD equations with padding."""
    ood_eqs = generate_ood_data(n_points=N_POINTS, seed=SEED)
    tables = []
    meta = []
    for eq in ood_eqs:
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


def evaluate_model_on_ood(model, tables, meta, model_name, use_ttt=False):
    """Evaluate a model on OOD equations."""
    model.eval()
    results_list = []

    for i in range(len(meta)):
        eq = meta[i]
        obs = torch.tensor(tables[i:i+1], device=DEVICE)

        if use_ttt:
            pred_ids = ttt_generate(
                model, obs, T=64, R=2, n_samples=64,
                n_ttt_steps=64, ttt_augmentations=32,
                ttt_rank=16, verbose=False
            )
        else:
            pred_ids = model.generate(obs, T=64, R=2, n_samples=128)

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

        result = {
            'id': eq['id'],
            'name': eq['name'],
            'domain': eq.get('domain', 'unknown'),
            'formula': eq['formula'],
            'predicted': ' '.join(pred_tokens),
            'ground_truth': ' '.join(gt_tokens),
            **metrics,
        }
        results_list.append(result)

        status = "MATCH" if metrics['exact_match'] else f"NED={metrics['ned']:.2f}, R²={metrics['r2']:.3f}"
        print(f"  [{model_name}] {eq['name']}: {status}")

    return results_list


def evaluate_baseline_on_ood(tables, meta):
    """Evaluate autoregressive baseline on OOD."""
    checkpoint_path = 'results/checkpoints/baseline.pt'
    if not os.path.exists(checkpoint_path):
        print("Baseline checkpoint not found, skipping")
        return []

    model = AutoregressiveBaseline(
        d_model=512, n_heads=8, n_enc_layers=4, n_dec_layers=6,
        d_ff=2048, dropout=0.1,
    ).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    results_list = []
    for i in range(len(meta)):
        eq = meta[i]
        obs = torch.tensor(tables[i:i+1], device=DEVICE)

        pred_ids = model.beam_search(obs, beam_width=10, max_len=MAX_SEQ_LEN)
        pred_tokens = decode(pred_ids)
        gt_tokens = eq['prefix']

        metrics = compute_all_metrics(
            pred_tokens, gt_tokens, eq['table'],
            variable_units=eq.get('units', {}),
            gt_constants=eq.get('constants', {}),
        )

        result = {
            'id': eq['id'],
            'name': eq['name'],
            'formula': eq['formula'],
            'predicted': ' '.join(pred_tokens),
            'ground_truth': ' '.join(gt_tokens),
            **metrics,
        }
        results_list.append(result)

        status = "MATCH" if metrics['exact_match'] else f"NED={metrics['ned']:.2f}, R²={metrics['r2']:.3f}"
        print(f"  [Baseline] {eq['name']}: {status}")

    return results_list


def run_generalization():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    checkpoint_path = 'results/checkpoints/phys_diffuse.pt'
    if not os.path.exists(checkpoint_path):
        print("ERROR: PhysDiffuse checkpoint not found.")
        return

    tables, meta = prepare_ood_data()
    print(f"Loaded {len(meta)} OOD equations")

    # PhysDiffuse
    model = create_phys_diffuse(d_model=512, n_heads=8, device=DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    print("\n--- PhysDiffuse ---")
    pd_results = evaluate_model_on_ood(model, tables, meta, "PhysDiffuse")

    print("\n--- PhysDiffuse+TTT ---")
    pd_ttt_results = evaluate_model_on_ood(model, tables, meta, "PD+TTT", use_ttt=True)

    print("\n--- Baseline ---")
    baseline_results = evaluate_baseline_on_ood(tables, meta)

    # Compute summary stats
    def summarize(results):
        if not results:
            return {}
        return {
            'n_exact_match': sum(1 for r in results if r['exact_match']),
            'mean_ned': np.mean([r['ned'] for r in results]),
            'mean_r2': np.mean([r['r2'] for r in results]),
            'n_high_r2': sum(1 for r in results if r['r2'] > 0.9),
            'dim_consistency_rate': np.mean([r['dim_consistent'] for r in results]),
        }

    # Failure analysis
    failures = []
    for r in pd_ttt_results:
        if not r['exact_match'] and r['r2'] < 0.9:
            failures.append({
                'name': r['name'],
                'formula': r['formula'],
                'predicted': r['predicted'],
                'ground_truth': r['ground_truth'],
                'r2': r['r2'],
                'ned': r['ned'],
                'failure_mode': _classify_failure(r),
            })

    full_results = {
        'n_ood_equations': len(meta),
        'phys_diffuse': {'summary': summarize(pd_results), 'per_equation': pd_results},
        'phys_diffuse_ttt': {'summary': summarize(pd_ttt_results), 'per_equation': pd_ttt_results},
        'baseline': {'summary': summarize(baseline_results), 'per_equation': baseline_results},
        'failure_analysis': failures,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/generalization.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print("\n=== Generalization Summary ===")
    for model_name in ['phys_diffuse', 'phys_diffuse_ttt', 'baseline']:
        s = full_results[model_name]['summary']
        if s:
            print(f"  {model_name}: EM={s['n_exact_match']}/{len(meta)}, "
                  f"NED={s['mean_ned']:.3f}, R²={s['mean_r2']:.3f}, "
                  f"High R²={s['n_high_r2']}/{len(meta)}")


def _classify_failure(result):
    """Classify failure mode."""
    pred = result['predicted'].split()
    gt = result['ground_truth'].split()

    if len(pred) > len(gt) * 2:
        return 'excessive_complexity'
    if len(pred) < len(gt) * 0.5:
        return 'oversimplification'
    if result['ned'] > 0.9:
        return 'completely_wrong_structure'
    if result['r2'] > 0.5:
        return 'close_numerical_fit'
    return 'structural_mismatch'


if __name__ == '__main__':
    run_generalization()
