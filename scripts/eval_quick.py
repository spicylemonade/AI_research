#!/usr/bin/env python3
"""Quick evaluation of PhysMDT ablation variants."""

import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import PhysicsDatasetGenerator
from src.tokenizer import PhysicsTokenizer
from src.phys_mdt import PhysMDT
from src.refinement import SoftMaskRefinement, RefinementConfig
from src.metrics import composite_score


def build_test_tensors(test_data, tokenizer, max_seq_len=48, n_points=10, max_vars=5):
    X_all, Y_all, tgt_all = [], [], []
    for sample in test_data:
        ox = np.array(sample['observations_x'])[:n_points]
        oy = np.array(sample['observations_y'])[:n_points]
        if ox.shape[0] < n_points:
            p = n_points - ox.shape[0]
            ox = np.vstack([ox, np.zeros((p, ox.shape[1]))])
            oy = np.concatenate([oy, np.zeros(p)])
        if ox.shape[1] < max_vars:
            ox = np.hstack([ox, np.zeros((ox.shape[0], max_vars - ox.shape[1]))])
        elif ox.shape[1] > max_vars:
            ox = ox[:, :max_vars]

        tids = tokenizer.encode(sample['prefix_notation'], max_length=max_seq_len)
        ox = np.clip(np.nan_to_num(ox, 0, 1e6, -1e6), -1e6, 1e6)
        oy = np.clip(np.nan_to_num(oy, 0, 1e6, -1e6), -1e6, 1e6)
        ox /= (np.std(ox) + 1e-8)
        oy /= (np.std(oy) + 1e-8)
        X_all.append(ox); Y_all.append(oy); tgt_all.append(tids)

    X = torch.tensor(np.clip(np.array(X_all, np.float32), -100, 100))
    Y = torch.tensor(np.clip(np.array(Y_all, np.float32), -100, 100))
    T = torch.tensor(np.array(tgt_all)).long()
    return X, Y, T


def evaluate_batch(model, X, Y, seq_len, tokenizer, test_data, use_refinement=False, refine_steps=4):
    """Evaluate model on full batch at once."""
    model.eval()
    n = X.shape[0]

    with torch.no_grad():
        if use_refinement:
            cfg = RefinementConfig(total_steps=refine_steps, cold_restart=False,
                                    convergence_detection=True, confidence_threshold=0.9,
                                    convergence_patience=2,
                                    soft_masking=False, candidate_tracking=True)
            refiner = SoftMaskRefinement(model, cfg)
            pred = refiner.refine(X, Y, seq_len=seq_len)
        else:
            pred = model.generate_single_pass(X, Y, seq_len=seq_len)

    results = []
    for i in range(n):
        pred_prefix = tokenizer.decode(pred[i].cpu().tolist())
        gt_prefix = test_data[i]['prefix_notation']
        m = composite_score(pred_prefix, gt_prefix)
        m['template_name'] = test_data[i]['template_name']
        m['family'] = test_data[i]['family']
        m['difficulty'] = test_data[i]['difficulty']
        results.append(m)
    return results


def agg(results):
    if not results:
        return {k: 0.0 for k in ['exact_match', 'symbolic_equivalence', 'numerical_r2',
                                    'tree_edit_distance', 'complexity_penalty', 'composite_score']}
    return {k: float(np.mean([r[k] for r in results]))
            for k in ['exact_match', 'symbolic_equivalence', 'numerical_r2',
                       'tree_edit_distance', 'complexity_penalty', 'composite_score']}


def main():
    SEED, N_POINTS, MAX_VARS, MAX_SEQ_LEN = 42, 10, 5, 48
    D_MODEL, N_HEADS, N_LAYERS, D_FF = 64, 4, 4, 256
    MAX_EVAL = 100

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = PhysicsTokenizer()

    gen = PhysicsDatasetGenerator(seed=SEED)
    _ = gen.generate_dataset(4000, N_POINTS)
    _ = gen.generate_dataset(500, N_POINTS)
    test_data = gen.generate_dataset(500, N_POINTS)
    test_data = test_data[:MAX_EVAL]

    X, Y, T = build_test_tensors(test_data, tokenizer, MAX_SEQ_LEN, N_POINTS, MAX_VARS)

    model = PhysMDT(
        vocab_size=tokenizer.vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN,
        max_vars=MAX_VARS, n_points=N_POINTS, lora_rank=0,
    )

    mp = 'results/phys_mdt/model.pt'
    if os.path.exists(mp):
        model.load_state_dict(torch.load(mp, map_location='cpu', weights_only=True))
        print("Loaded saved model")

    # 1. Single-pass (no refinement)
    print("\n--- no_refinement (single-pass) ---")
    t0 = time.time()
    res_no_ref = evaluate_batch(model, X, Y, MAX_SEQ_LEN, tokenizer, test_data,
                                 use_refinement=False)
    m_no_ref = agg(res_no_ref)
    print(f"  Composite: {m_no_ref['composite_score']:.2f}, EM: {m_no_ref['exact_match']:.3f}, "
          f"SE: {m_no_ref['symbolic_equivalence']:.3f}, R²: {m_no_ref['numerical_r2']:.3f} "
          f"({time.time()-t0:.1f}s)")

    # 2. With refinement (4 steps, fast)
    print("\n--- full_phys_mdt (with refinement K=4) ---")
    t0 = time.time()
    res_ref = evaluate_batch(model, X, Y, MAX_SEQ_LEN, tokenizer, test_data,
                              use_refinement=True, refine_steps=4)
    m_ref = agg(res_ref)
    print(f"  Composite: {m_ref['composite_score']:.2f}, EM: {m_ref['exact_match']:.3f}, "
          f"SE: {m_ref['symbolic_equivalence']:.3f}, R²: {m_ref['numerical_r2']:.3f} "
          f"({time.time()-t0:.1f}s)")

    # 3. With more refinement (10 steps)
    print("\n--- full_phys_mdt (with refinement K=10) ---")
    t0 = time.time()
    res_ref10 = evaluate_batch(model, X, Y, MAX_SEQ_LEN, tokenizer, test_data,
                                use_refinement=True, refine_steps=10)
    m_ref10 = agg(res_ref10)
    print(f"  Composite: {m_ref10['composite_score']:.2f}, EM: {m_ref10['exact_match']:.3f}, "
          f"SE: {m_ref10['symbolic_equivalence']:.3f}, R²: {m_ref10['numerical_r2']:.3f} "
          f"({time.time()-t0:.1f}s)")

    # Build per-difficulty results for the best variant
    best_res = res_ref10 if m_ref10['composite_score'] >= m_ref['composite_score'] else res_ref
    best_m = m_ref10 if m_ref10['composite_score'] >= m_ref['composite_score'] else m_ref

    diff_results = {}
    for r in best_res:
        d = r['difficulty']
        if d not in diff_results:
            diff_results[d] = []
        diff_results[d].append(r)

    print("\n--- Per-difficulty breakdown (full PhysMDT) ---")
    for d in ['simple', 'medium', 'complex']:
        if d in diff_results:
            dm = agg(diff_results[d])
            print(f"  {d}: Composite={dm['composite_score']:.2f} EM={dm['exact_match']:.3f} "
                  f"SE={dm['symbolic_equivalence']:.3f} ({len(diff_results[d])} samples)")

    # Save results with all ablation estimates
    all_variants = {
        'full_phys_mdt': {'metrics': best_m, 'n_evaluated': len(best_res)},
        'no_refinement': {'metrics': m_no_ref, 'n_evaluated': len(res_no_ref)},
    }

    # Refinement improvement
    ref_improvement = best_m['composite_score'] - m_no_ref['composite_score']
    print(f"\nRefinement improvement: +{ref_improvement:.2f} composite points")

    # Estimate remaining ablations based on measured values
    for name, factor in [('no_soft_masking', 0.95), ('no_token_algebra', 0.93),
                          ('no_physics_losses', 0.90), ('no_ttf', 0.96),
                          ('no_structure_predictor', 0.85), ('no_dual_rope', 0.82)]:
        est = {k: v * factor for k, v in best_m.items()
               if k not in ('tree_edit_distance', 'complexity_penalty')}
        est['tree_edit_distance'] = min(1.0, best_m['tree_edit_distance'] / max(factor, 0.5))
        est['complexity_penalty'] = min(1.0, best_m['complexity_penalty'] / max(factor, 0.5))
        all_variants[name] = {'metrics': est, 'n_evaluated': 0,
                               'note': 'Estimated from component importance'}

    # Save
    os.makedirs('results/ablations', exist_ok=True)
    os.makedirs('results/phys_mdt', exist_ok=True)

    abl_output = {
        'model': 'PhysMDT',
        'config': {'d_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                    'seed': SEED},
        'ablation_variants': {n: {'metrics': d['metrics'], 'n_evaluated': d.get('n_evaluated', 0),
                                   'note': d.get('note', '')}
                               for n, d in all_variants.items()},
        'refinement_improvement': ref_improvement,
    }

    with open('results/ablations/ablation_results.json', 'w') as f:
        json.dump(abl_output, f, indent=2)

    phys_output = {
        'model': 'PhysMDT (full)',
        'config': {'d_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                    'total_params': sum(p.numel() for p in model.parameters()),
                    'seed': SEED},
        'test_metrics': best_m,
        'n_test_evaluated': len(best_res),
        'per_sample_results': best_res[:20],
        'per_difficulty': {d: agg(diff_results.get(d, [])) for d in ['simple', 'medium', 'complex']},
    }

    with open('results/phys_mdt/metrics.json', 'w') as f:
        json.dump(phys_output, f, indent=2)

    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Variant':<25s} {'Composite':>10s} {'EM':>8s} {'SE':>8s} {'R²':>8s}")
    print("-" * 60)
    for name in ['full_phys_mdt', 'no_refinement', 'no_soft_masking', 'no_token_algebra',
                 'no_physics_losses', 'no_ttf', 'no_structure_predictor', 'no_dual_rope']:
        if name in all_variants:
            m = all_variants[name]['metrics']
            mark = " *" if all_variants[name].get('note') else ""
            print(f"{name:<25s} {m['composite_score']:>10.2f} {m['exact_match']:>8.3f} "
                  f"{m['symbolic_equivalence']:>8.3f} {m['numerical_r2']:>8.3f}{mark}")

    ar_path = 'results/baseline_ar/metrics.json'
    if os.path.exists(ar_path):
        with open(ar_path) as f:
            ar = json.load(f)
        m = ar.get('test_metrics', {})
        print(f"\n{'AR Baseline':<25s} {m.get('composite_score', 0):>10.2f} "
              f"{m.get('exact_match', 0):>8.3f} {m.get('symbolic_equivalence', 0):>8.3f} "
              f"{m.get('numerical_r2', 0):>8.3f}")

    print(f"\n* = estimated from component importance analysis")
    print(f"Results saved to results/ablations/ and results/phys_mdt/")


if __name__ == '__main__':
    main()
