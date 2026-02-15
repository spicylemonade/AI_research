"""Synthetic physics dataset generator with configurable complexity.

Generates paired (numerical_observations, symbolic_equation) samples from
the equation corpus with random variable instantiation and noise injection.
"""

import os
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sympy

from data.equations import (
    Equation, get_training_equations, get_held_out_equations,
    get_equations_by_tier, get_all_equations,
)
from data.tokenizer import ExprTokenizer, PAD_IDX


@dataclass
class PhysicsSample:
    """A single (observations, expression) sample."""
    observations: np.ndarray   # shape (n_points, n_vars + 1)
    token_ids: List[int]       # prefix notation token IDs
    equation_id: str           # equation identifier
    tier: int                  # complexity tier
    n_vars: int                # number of input variables


class PhysicsDataset(Dataset):
    """PyTorch dataset of physics equation samples.

    Each sample contains:
    - observations: tensor of shape (n_points, max_vars + 1)
    - obs_mask: tensor of shape (n_points, max_vars + 1), 1 for valid, 0 for pad
    - token_ids: tensor of shape (max_expr_len,), padded token sequence
    - tier: int
    - n_vars: int
    """

    def __init__(
        self,
        equations: Optional[List[Equation]] = None,
        n_samples: int = 10000,
        n_points: int = 50,
        noise_level: float = 0.0,
        max_vars: int = 5,
        max_expr_len: int = 64,
        seed: int = 42,
        tier_weights: Optional[Dict[int, float]] = None,
    ):
        """Initialize the dataset.

        Args:
            equations: List of equations to sample from. Defaults to training set.
            n_samples: Number of samples to generate.
            n_points: Number of observation points per sample.
            noise_level: Standard deviation of Gaussian noise (fraction of |y|).
            max_vars: Maximum number of variables (for padding).
            max_expr_len: Maximum expression token length (for padding).
            seed: Random seed for reproducibility.
            tier_weights: Optional per-tier sampling weights. Default: uniform.
        """
        super().__init__()
        self.equations = equations or get_training_equations()
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise_level = noise_level
        self.max_vars = max_vars
        self.max_expr_len = max_expr_len
        self.seed = seed
        self.tokenizer = ExprTokenizer()

        # Group equations by tier for weighted sampling
        self.tier_equations: Dict[int, List[Equation]] = {}
        for eq in self.equations:
            self.tier_equations.setdefault(eq.tier, []).append(eq)

        # Set tier weights
        if tier_weights is None:
            self.tier_weights = {t: 1.0 for t in self.tier_equations}
        else:
            self.tier_weights = tier_weights

        # Normalize weights
        total = sum(self.tier_weights.get(t, 0) for t in self.tier_equations)
        self.tier_probs = {t: self.tier_weights.get(t, 0) / total
                          for t in self.tier_equations}

        # Pre-generate samples for reproducibility and speed
        self._generate_all(seed)

    def _generate_all(self, seed: int):
        """Pre-generate all samples."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        self.samples = []
        tiers = sorted(self.tier_equations.keys())
        tier_probs = [self.tier_probs[t] for t in tiers]

        for i in range(self.n_samples):
            # Sample tier
            tier = rng.choices(tiers, weights=tier_probs, k=1)[0]
            # Sample equation from tier
            eq = rng.choice(self.tier_equations[tier])

            sample = self._generate_sample(eq, np_rng)
            if sample is not None:
                self.samples.append(sample)
            else:
                # Retry with a simpler equation
                fallback_eq = rng.choice(self.tier_equations[min(tiers)])
                sample = self._generate_sample(fallback_eq, np_rng)
                if sample is not None:
                    self.samples.append(sample)

        self.n_samples = len(self.samples)

    def _generate_sample(self, eq: Equation, rng: np.random.RandomState,
                         max_retries: int = 5) -> Optional[PhysicsSample]:
        """Generate a single sample from an equation."""
        n_vars = len(eq.variables)

        # Tokenize expression
        try:
            token_ids = self.tokenizer.encode(eq.symbolic_expr, add_sos_eos=True)
        except Exception:
            return None

        for attempt in range(max_retries):
            try:
                # Generate random input points
                x_points = np.zeros((self.n_points, n_vars))
                for j, var in enumerate(eq.variables):
                    var_name = str(var)
                    if var_name in eq.var_ranges:
                        lo, hi = eq.var_ranges[var_name]
                    else:
                        lo, hi = 0.1, 10.0
                    x_points[:, j] = rng.uniform(lo, hi, self.n_points)

                # Evaluate equation at each point
                # Build lambdified function for speed
                func = sympy.lambdify(
                    eq.variables, eq.symbolic_expr,
                    modules=['numpy', {'sqrt': np.sqrt, 'sin': np.sin,
                                       'cos': np.cos, 'tan': np.tan,
                                       'log': np.log, 'exp': np.exp,
                                       'pi': np.pi}]
                )

                # Substitute constants
                const_expr = eq.symbolic_expr
                for cname, cval in eq.constants.items():
                    # Find the symbol in expression
                    for sym in const_expr.free_symbols:
                        if str(sym) == cname:
                            const_expr = const_expr.subs(sym, cval)
                            break

                # Re-lambdify with constants substituted
                remaining_vars = sorted(const_expr.free_symbols,
                                        key=lambda s: str(s))
                if remaining_vars:
                    func = sympy.lambdify(
                        remaining_vars, const_expr,
                        modules=['numpy']
                    )
                    # Map remaining vars to columns
                    var_map = {str(v): i for i, v in enumerate(eq.variables)}
                    args = []
                    for rv in remaining_vars:
                        rv_name = str(rv)
                        if rv_name in var_map:
                            args.append(x_points[:, var_map[rv_name]])
                        else:
                            # Unknown variable, skip
                            return None
                    y_points = func(*args)
                else:
                    # All variables were constants — constant function
                    y_points = np.full(self.n_points, float(const_expr))

                # Check for valid output
                if np.any(np.isnan(y_points)) or np.any(np.isinf(y_points)):
                    continue
                if np.max(np.abs(y_points)) > 1e10:
                    continue

                # Add noise
                if self.noise_level > 0:
                    noise = rng.normal(0, self.noise_level, self.n_points)
                    y_points = y_points * (1 + noise)

                # Combine into observation matrix
                observations = np.column_stack([x_points, y_points])

                return PhysicsSample(
                    observations=observations.astype(np.float32),
                    token_ids=token_ids,
                    equation_id=eq.id,
                    tier=eq.tier,
                    n_vars=n_vars,
                )

            except Exception:
                continue

        return None

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Pad observations to max_vars + 1 columns
        obs = np.zeros((self.n_points, self.max_vars + 1), dtype=np.float32)
        obs_mask = np.zeros((self.n_points, self.max_vars + 1), dtype=np.float32)
        actual_cols = sample.observations.shape[1]
        obs[:, :actual_cols] = sample.observations
        obs_mask[:, :actual_cols] = 1.0

        # Pad token sequence
        tokens = sample.token_ids[:self.max_expr_len]
        token_tensor = torch.full((self.max_expr_len,), PAD_IDX, dtype=torch.long)
        token_tensor[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        token_len = len(tokens)

        return {
            'observations': torch.tensor(obs, dtype=torch.float32),
            'obs_mask': torch.tensor(obs_mask, dtype=torch.float32),
            'tokens': token_tensor,
            'token_len': token_len,
            'tier': sample.tier,
            'n_vars': sample.n_vars,
            'equation_id': sample.equation_id,
        }


def generate_datasets(
    n_train: int = 500000,
    n_val: int = 10000,
    n_points: int = 50,
    noise_level: float = 0.01,
    seed: int = 42,
    quick: bool = False,
) -> Tuple[PhysicsDataset, PhysicsDataset]:
    """Generate train and validation datasets.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_points: Observations per sample.
        noise_level: Gaussian noise level.
        seed: Random seed.
        quick: If True, use 1% of data for smoke testing.

    Returns:
        (train_dataset, val_dataset)
    """
    if quick:
        n_train = max(n_train // 100, 1000)
        n_val = max(n_val // 100, 200)

    train_equations = get_training_equations()
    # Emphasis on harder tiers for training
    tier_weights = {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}

    print(f"Generating {n_train} training samples...")
    train_ds = PhysicsDataset(
        equations=train_equations,
        n_samples=n_train,
        n_points=n_points,
        noise_level=noise_level,
        seed=seed,
        tier_weights=tier_weights,
    )

    print(f"Generating {n_val} validation samples...")
    val_ds = PhysicsDataset(
        equations=train_equations,
        n_samples=n_val,
        n_points=n_points,
        noise_level=0.0,  # No noise for validation
        seed=seed + 1,
    )

    print(f"Generated: train={len(train_ds)}, val={len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests():
    """Run unit tests for data generation."""
    from data.tokenizer import ExprTokenizer
    tokenizer = ExprTokenizer()
    equations = get_training_equations()
    rng = np.random.RandomState(42)

    passed = 0
    failed = 0

    for eq in equations:
        ds = PhysicsDataset(
            equations=[eq], n_samples=10, n_points=20, noise_level=0.0, seed=42
        )
        if len(ds) == 0:
            print(f"WARN: No valid samples for {eq.name}")
            continue

        sample = ds[0]
        obs = sample['observations'].numpy()
        n_vars = sample['n_vars']

        # Decode tokens and evaluate
        try:
            decoded = tokenizer.decode(sample['tokens'][:sample['token_len']].tolist())

            # Substitute constants
            for cname, cval in eq.constants.items():
                for sym in decoded.free_symbols:
                    if str(sym) == cname:
                        decoded = decoded.subs(sym, cval)

            remaining_vars = sorted(decoded.free_symbols, key=lambda s: str(s))
            if not remaining_vars:
                continue

            func = sympy.lambdify(remaining_vars, decoded, modules=['numpy'])
            var_map = {str(v): i for i, v in enumerate(eq.variables)}
            args = []
            for rv in remaining_vars:
                rv_name = str(rv)
                if rv_name in var_map:
                    args.append(obs[:, var_map[rv_name]])
                else:
                    continue

            if len(args) != len(remaining_vars):
                continue

            y_pred = func(*args)
            y_true = obs[:, n_vars]

            # Check numerical consistency
            max_err = np.max(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8))
            if max_err < 0.01:  # 1% tolerance
                passed += 1
            else:
                failed += 1
                print(f"FAIL {eq.name}: max_rel_err={max_err:.4f}")
        except Exception as e:
            failed += 1
            print(f"ERROR {eq.name}: {e}")

    print(f"\nData generation tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == '__main__':
    print("Running data generation tests...")
    run_tests()

    print("\nGenerating small dataset...")
    train_ds, val_ds = generate_datasets(n_train=1000, n_val=100, quick=False)
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")

    # Test DataLoader
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  observations: {batch['observations'].shape}")
    print(f"  obs_mask: {batch['obs_mask'].shape}")
    print(f"  tokens: {batch['tokens'].shape}")
    print(f"  token_len: {batch['token_len'].shape}")
    print(f"  tier: {batch['tier'][:5]}")
