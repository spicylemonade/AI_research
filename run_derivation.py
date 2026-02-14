"""Derivation from scratch: derive fundamental laws from raw observation data."""

import torch
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.tokenizer import decode, encode, PAD_ID, MASK_ID, SOS_ID, EOS_ID, MAX_SEQ_LEN
from model.phys_diffuse import create_phys_diffuse
from model.ttt import ttt_generate
from model.postprocess import postprocess_candidates, bfgs_optimize_constants
from evaluation.metrics import compute_all_metrics, exact_symbolic_match

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VARS = 10
N_POINTS = 200  # More points for derivation experiments


def generate_physical_system(system_name, n_points=N_POINTS, seed=42):
    """Generate synthetic observation data for fundamental physical systems."""
    rng = np.random.default_rng(seed)

    systems = {
        'newton_2nd_law': {
            'description': "Newton's 2nd Law: F = m * a",
            'formula': 'F = m * a',
            'prefix': ['mul', 'x_0', 'x_1'],
            'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -2], 'y': [1, 1, -2]},
            'generate': lambda: _gen_fma(rng, n_points),
        },
        'universal_gravitation': {
            'description': "Universal Gravitation: F = G * m1 * m2 / r^2",
            'formula': 'F = c_0 * m1 * m2 / r^2',
            'prefix': ['mul', 'c_0', 'div', 'mul', 'x_0', 'x_1', 'pow', 'x_2', 'int_2'],
            'units': {'x_0': [1, 0, 0], 'x_1': [1, 0, 0], 'x_2': [0, 1, 0]},
            'generate': lambda: _gen_gravitation(rng, n_points),
        },
        'kinetic_energy': {
            'description': "Kinetic Energy: E = 0.5 * m * v^2",
            'formula': 'E = 0.5 * m * v^2',
            'prefix': ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],
            'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'y': [1, 2, -2]},
            'generate': lambda: _gen_kinetic_energy(rng, n_points),
        },
        'conservation_momentum': {
            'description': "Conservation of Momentum: p = m * v",
            'formula': 'p = m * v',
            'prefix': ['mul', 'x_0', 'x_1'],
            'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'y': [1, 1, -1]},
            'generate': lambda: _gen_momentum(rng, n_points),
        },
        'hooke_law': {
            'description': "Hooke's Law: F = -k * x",
            'formula': 'F = c_0 * x',
            'prefix': ['mul', 'c_0', 'x_0'],
            'units': {'x_0': [0, 1, 0], 'y': [1, 1, -2]},
            'generate': lambda: _gen_hooke(rng, n_points),
        },
    }

    if system_name not in systems:
        raise ValueError(f"Unknown system: {system_name}")

    sys_info = systems[system_name]
    table = sys_info['generate']()
    return table, sys_info


def _gen_fma(rng, n):
    m = rng.uniform(0.1, 100, n)
    a = rng.uniform(0.1, 20, n)
    F = m * a
    return np.column_stack([m, a, F]).astype(np.float32)


def _gen_gravitation(rng, n):
    G = 6.674e-11
    m1 = rng.uniform(1e20, 1e30, n)
    m2 = rng.uniform(1e20, 1e30, n)
    r = rng.uniform(1e6, 1e10, n)
    F = G * m1 * m2 / r**2
    # Normalize for numerical stability
    m1_norm = m1 / 1e25
    m2_norm = m2 / 1e25
    r_norm = r / 1e8
    F_norm = F / (6.674e-11 * 1e50 / 1e16)
    return np.column_stack([m1_norm, m2_norm, r_norm, F_norm]).astype(np.float32)


def _gen_kinetic_energy(rng, n):
    m = rng.uniform(0.1, 100, n)
    v = rng.uniform(0.1, 50, n)
    E = 0.5 * m * v**2
    return np.column_stack([m, v, E]).astype(np.float32)


def _gen_momentum(rng, n):
    m = rng.uniform(0.1, 100, n)
    v = rng.uniform(-50, 50, n)
    p = m * v
    return np.column_stack([m, v, p]).astype(np.float32)


def _gen_hooke(rng, n):
    k = 50.0  # Spring constant
    x = rng.uniform(-2, 2, n)
    F = -k * x
    return np.column_stack([x, F]).astype(np.float32)


def pad_table(table, n_points=50, max_vars=10):
    """Pad observation table to standard shape."""
    if table.shape[0] < n_points:
        pad = np.zeros((n_points - table.shape[0], table.shape[1]))
        table = np.vstack([table, pad])
    elif table.shape[0] > n_points:
        table = table[:n_points]
    if table.shape[1] < max_vars + 1:
        pad = np.zeros((table.shape[0], max_vars + 1 - table.shape[1]))
        table = np.hstack([table, pad])
    return table


def run_derivation_experiments():
    """Run derivation from scratch for 5 fundamental laws."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    checkpoint_path = 'results/checkpoints/phys_diffuse.pt'
    if not os.path.exists(checkpoint_path):
        print("ERROR: PhysDiffuse checkpoint not found.")
        return

    # Create model
    model = create_phys_diffuse(d_model=512, n_heads=8, device=DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    systems = [
        'newton_2nd_law',
        'universal_gravitation',
        'kinetic_energy',
        'conservation_momentum',
        'hooke_law',
    ]

    results = []
    refinement_traces = {}

    for sys_name in systems:
        print(f"\n=== Deriving: {sys_name} ===")
        table, info = generate_physical_system(sys_name, n_points=N_POINTS)
        padded = pad_table(table, n_points=50)  # Use 50 for model input
        obs = torch.tensor(padded[np.newaxis], dtype=torch.float32, device=DEVICE)

        # Track refinement steps
        trace = _trace_refinement(model, obs, n_steps=32)
        refinement_traces[sys_name] = trace

        # Full generation with TTT
        t0 = time.time()
        pred_ids = ttt_generate(
            model, obs, T=64, R=2, n_samples=128,
            n_ttt_steps=96, ttt_augmentations=64,
            ttt_rank=16, verbose=False
        )
        elapsed = time.time() - t0

        pred_tokens = decode(pred_ids)
        gt_tokens = info['prefix']

        # Post-process with BFGS
        if pred_tokens:
            processed = postprocess_candidates([pred_tokens], table, top_k=1)
            if processed and processed[0]['mse'] < float('inf'):
                pred_tokens = processed[0]['tokens']
                opt_consts = processed[0].get('constants', {})
            else:
                opt_consts = {}
        else:
            opt_consts = {}

        # Compute metrics
        metrics = compute_all_metrics(
            pred_tokens, gt_tokens, table,
            variable_units=info.get('units', {}),
        )

        result = {
            'system': sys_name,
            'description': info['description'],
            'formula': info['formula'],
            'predicted': ' '.join(pred_tokens),
            'ground_truth': ' '.join(gt_tokens),
            'optimized_constants': opt_consts,
            'time_seconds': elapsed,
            **metrics,
        }
        results.append(result)

        status = "DERIVED!" if metrics['exact_match'] else f"NED={metrics['ned']:.2f}, R²={metrics['r2']:.3f}"
        print(f"  GT: {info['formula']}")
        print(f"  Predicted: {' '.join(pred_tokens)}")
        print(f"  {status} (time: {elapsed:.1f}s)")

    # Summary
    n_derived = sum(1 for r in results if r['exact_match'])
    n_high_r2 = sum(1 for r in results if r['r2'] > 0.95)

    full_results = {
        'n_systems': len(systems),
        'n_exact_match': n_derived,
        'n_high_r2': n_high_r2,
        'results': results,
        'refinement_traces': refinement_traces,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/derivation_from_scratch.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n=== Derivation Summary ===")
    print(f"  Exact Match: {n_derived}/{len(systems)}")
    print(f"  High R² (>0.95): {n_high_r2}/{len(systems)}")


def _trace_refinement(model, obs, n_steps=32):
    """Trace the refinement process step by step."""
    import torch.nn.functional as F
    from data.tokenizer import VOCAB_SIZE

    device = obs.device
    L = MAX_SEQ_LEN
    n_samples = 32

    model.eval()
    memory = model.encoder(obs)
    memory = memory.expand(n_samples, -1, -1)

    x = torch.full((n_samples, L), MASK_ID, dtype=torch.long, device=device)
    x[:, 0] = SOS_ID

    trace = []
    for step in range(1, n_steps + 1):
        with torch.no_grad():
            logits = model.decoder(x, memory)
            norm = logits.norm(dim=-1, keepdim=True) / (model.d_model ** 0.5) + 1e-6
            logits = logits / norm
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
            logits = logits.clamp(-30, 30)
            tau = 1.0 * (0.1 / 1.0) ** (step / n_steps)
            probs = F.softmax(logits / tau, dim=-1)
            probs = torch.where(torch.isfinite(probs) & (probs > 0), probs, torch.full_like(probs, 1e-8))
            probs = probs / probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(n_samples, L)
            conf = probs.max(dim=-1).values
            n_unmask = max(1, int(L * step / n_steps))

            for b in range(n_samples):
                conf_masked = conf[b].clone()
                conf_masked[0] = -1
                _, top_idx = conf_masked.topk(n_unmask)
                x[b, top_idx] = sampled[b, top_idx]

        # Record current state of first sample
        current_tokens = decode(x[0].tolist())
        n_masked = (x[0] == MASK_ID).sum().item()
        trace.append({
            'step': step,
            'tokens': current_tokens[:10],
            'n_masked': n_masked,
            'n_unmasked': L - n_masked,
        })

    return trace


if __name__ == '__main__':
    run_derivation_experiments()
