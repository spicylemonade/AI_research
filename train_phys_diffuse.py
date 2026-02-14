"""Train PhysDiffuse to convergence and evaluate vs baseline."""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.data_generator import generate_dataset
from data.feynman_loader import generate_benchmark_data
from data.tokenizer import encode, decode, PAD_ID, SOS_ID, EOS_ID, VOCAB_SIZE, MAX_SEQ_LEN
from model.phys_diffuse import PhysDiffuse, create_phys_diffuse, phys_diffuse_train_step
from model.postprocess import postprocess_candidates
from evaluation.metrics import compute_all_metrics, tier_stratified_report

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VARS = 10
N_POINTS = 50


def prepare_training_data(n_samples=20000, seed=SEED):
    """Generate and prepare training data."""
    print(f"Generating {n_samples} training samples...")
    t0 = time.time()
    raw_data = generate_dataset(n_samples, n_points=N_POINTS,
                                 min_vars=1, max_vars=5,
                                 min_depth=1, max_depth=4, seed=seed)
    print(f"Generated {len(raw_data)} samples in {time.time()-t0:.1f}s")

    tables = []
    token_ids = []
    for table, prefix, ids in raw_data:
        n_vars = table.shape[1] - 1
        if table.shape[0] < N_POINTS:
            pad = np.zeros((N_POINTS - table.shape[0], table.shape[1]))
            table = np.vstack([table, pad])
        elif table.shape[0] > N_POINTS:
            table = table[:N_POINTS]
        if table.shape[1] < MAX_VARS + 1:
            pad = np.zeros((table.shape[0], MAX_VARS + 1 - table.shape[1]))
            table = np.hstack([table, pad])
        tables.append(table)
        token_ids.append(ids)

    return np.array(tables, dtype=np.float32), np.array(token_ids, dtype=np.int64)


def prepare_benchmark_data(seed=SEED):
    """Prepare benchmark test data."""
    benchmark = generate_benchmark_data(n_points=N_POINTS, seed=seed)

    tables = []
    token_ids = []
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
        token_ids.append(eq['token_ids'])
        meta.append(eq)

    return (np.array(tables, dtype=np.float32),
            np.array(token_ids, dtype=np.int64),
            meta)


def train_phys_diffuse(n_train=20000, n_epochs=20, batch_size=32, lr=3e-4):
    """Full training loop for PhysDiffuse."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Prepare data
    train_tables, train_tokens = prepare_training_data(n_train)
    n_train_actual = len(train_tables)
    print(f"Training data: {n_train_actual} samples")

    # Create model
    model = create_phys_diffuse(d_model=512, n_heads=8, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    losses = []
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        indices = np.random.permutation(n_train_actual)

        for i in range(0, n_train_actual, batch_size):
            batch_idx = indices[i:i+batch_size]
            obs = torch.tensor(train_tables[batch_idx], device=DEVICE)
            tgt = torch.tensor(train_tokens[batch_idx], device=DEVICE)

            loss = phys_diffuse_train_step(model, obs, tgt, optimizer)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        losses.append({
            'epoch': epoch,
            'loss': float(avg_loss),
            'lr': float(scheduler.get_last_lr()[0]),
            'time': time.time() - start_time,
        })
        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}, "
              f"time={time.time()-start_time:.0f}s")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/3600:.2f}h)")

    # Save training curve
    os.makedirs('results', exist_ok=True)
    with open('results/phys_diffuse_training.json', 'w') as f:
        json.dump({'losses': losses, 'total_time_seconds': total_time}, f, indent=2)

    # Save checkpoint
    os.makedirs('results/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'results/checkpoints/phys_diffuse.pt')

    return model, total_time


def evaluate_phys_diffuse(model, use_postprocess=True):
    """Evaluate PhysDiffuse on benchmark test set."""
    test_tables, test_tokens, test_meta = prepare_benchmark_data()
    n_test = len(test_tables)

    model.eval()
    results_list = []
    qualitative = {1: [], 2: [], 3: [], 4: []}

    print(f"\nEvaluating on {n_test} benchmark equations...")
    for i in range(n_test):
        eq = test_meta[i]
        obs = torch.tensor(test_tables[i:i+1], device=DEVICE)

        # Generate with recursive refinement
        pred_ids = model.generate(obs, T=64, R=2, n_samples=128,
                                   tau_start=1.0, tau_end=0.1)
        pred_tokens = decode(pred_ids)
        gt_tokens = eq['prefix']

        # Optional: post-process with BFGS
        if use_postprocess and pred_tokens:
            from model.postprocess import postprocess_candidates
            processed = postprocess_candidates(
                [pred_tokens], eq['table'], top_k=1)
            if processed and processed[0]['mse'] < float('inf'):
                pred_tokens = processed[0]['tokens']

        # Compute metrics
        metrics = compute_all_metrics(
            pred_tokens, gt_tokens,
            eq['table'],
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
            **metrics,
        }
        results_list.append(result)

        # Collect qualitative examples (up to 3 per tier)
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
        print(f"  [{eq['id']}] {eq['name']}: {status}")

    # Tier-stratified report
    tiers = [eq['tier'] for eq in test_meta]
    report = tier_stratified_report(
        [r for r in results_list],
        tiers
    )

    # Full results
    full_results = {
        'n_equations': n_test,
        'report': report,
        'qualitative_examples': qualitative,
        'per_equation': results_list,
    }

    with open('results/phys_diffuse_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    # Print summary
    print("\n=== PhysDiffuse Results ===")
    for tier_key, tier_data in sorted(report.items()):
        print(f"  {tier_key}: EM={tier_data['exact_match_rate']:.1%}, "
              f"NED={tier_data['mean_ned']:.3f}, R²={tier_data['mean_r2']:.3f}, "
              f"DimOK={tier_data['dim_consistency_rate']:.1%}")

    return full_results


def compare_with_baseline():
    """Load baseline results and compare with PhysDiffuse."""
    from evaluation.metrics import paired_bootstrap_test

    with open('results/baseline_results.json', 'r') as f:
        baseline = json.load(f)
    with open('results/phys_diffuse_results.json', 'r') as f:
        phys_diffuse = json.load(f)

    # Extract per-equation metrics for paired bootstrap
    baseline_em = [r['exact_match'] for r in baseline['per_equation']]
    phys_diffuse_em = [r['exact_match'] for r in phys_diffuse['per_equation']]

    baseline_ned = [r['ned'] for r in baseline['per_equation']]
    phys_diffuse_ned = [r['ned'] for r in phys_diffuse['per_equation']]

    baseline_r2 = [r['r2'] for r in baseline['per_equation']]
    phys_diffuse_r2 = [r['r2'] for r in phys_diffuse['per_equation']]

    # Paired bootstrap tests
    em_pval = paired_bootstrap_test(
        [float(x) for x in baseline_em],
        [float(x) for x in phys_diffuse_em],
        n_resamples=1000
    )
    ned_pval = paired_bootstrap_test(
        baseline_ned, phys_diffuse_ned, n_resamples=1000
    )
    r2_pval = paired_bootstrap_test(
        baseline_r2, phys_diffuse_r2, n_resamples=1000
    )

    comparison = {
        'baseline_overall': baseline['report'].get('overall', {}),
        'phys_diffuse_overall': phys_diffuse['report'].get('overall', {}),
        'paired_bootstrap': {
            'exact_match_p_value': em_pval,
            'ned_p_value': ned_pval,
            'r2_p_value': r2_pval,
        },
    }

    with open('results/comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print("\n=== Comparison: Baseline vs PhysDiffuse ===")
    print(f"  EM p-value: {em_pval:.4f}")
    print(f"  NED p-value: {ned_pval:.4f}")
    print(f"  R² p-value: {r2_pval:.4f}")

    return comparison


if __name__ == '__main__':
    model, train_time = train_phys_diffuse(n_train=20000, n_epochs=20, batch_size=32)
    results = evaluate_phys_diffuse(model)
    comparison = compare_with_baseline()
    print(f"\nTotal training time: {train_time/3600:.2f}h")
