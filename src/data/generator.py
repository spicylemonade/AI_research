"""
Synthetic data generation pipeline for physics equation-observation pairs.
Generates train/val/test splits with configurable noise and augmentation.
"""
import os
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .equation_templates import (
    get_all_templates, EquationTemplate, tokenize_prefix,
    VOCAB_SIZE, EQUATION_VOCAB, MAX_OBS_POINTS, MAX_INPUT_VARS, MAX_EQ_LENGTH
)

# Fixed random seed for reproducibility
SEED = 42


def generate_single_sample(
    template: EquationTemplate,
    rng: np.random.RandomState,
    n_obs: int = 50,
    noise_level: float = 0.0,
    scale_augment: bool = False,
) -> Optional[Dict]:
    """Generate a single equation-observation pair from a template.

    Returns dict with observations, equation tokens, metadata, or None if invalid.
    """
    # Sample coefficients
    coeffs = []
    for low, high in template.coeff_ranges:
        coeffs.append(rng.uniform(low, high))

    # Sample input variables
    n_vars = template.n_input_vars
    x = np.zeros((n_obs, n_vars))
    for i in range(n_vars):
        low, high = template.var_ranges[i]
        if template.positive_vars and i < len(template.positive_vars) and template.positive_vars[i]:
            x[:, i] = rng.uniform(max(low, 1e-6), high, size=n_obs)
        else:
            x[:, i] = rng.uniform(low, high, size=n_obs)

    # Optional scale augmentation
    if scale_augment:
        scale = rng.uniform(0.1, 10.0)
        x *= scale
        # Adjust coefficients inversely (simplified - just scale observations)

    # Evaluate equation
    try:
        y = template.eval_fn(x, coeffs)
    except (ZeroDivisionError, FloatingPointError, RuntimeWarning):
        return None

    # Check validity
    if not np.all(np.isfinite(y)):
        return None
    if np.std(y) < 1e-8:
        return None
    if np.any(np.abs(y) > 1e6):
        return None

    # Add noise
    if noise_level > 0:
        noise_std = noise_level * np.std(y)
        y = y + rng.normal(0, noise_std, size=n_obs)

    # Tokenize equation
    prefix_tokens = template.to_prefix(coeffs)
    eq_token_ids = tokenize_prefix(prefix_tokens)

    if len(eq_token_ids) > MAX_EQ_LENGTH:
        # Truncate (shouldn't happen with our templates)
        eq_token_ids = eq_token_ids[:MAX_EQ_LENGTH - 1] + [EQUATION_VOCAB["[EQ_END]"]]

    # Build observation array: shape (n_obs, MAX_INPUT_VARS + 1)
    obs = np.zeros((n_obs, MAX_INPUT_VARS + 1), dtype=np.float32)
    obs[:, :n_vars] = x
    obs[:, MAX_INPUT_VARS] = y  # Last column is y

    return {
        "observations": obs,
        "eq_tokens": eq_token_ids,
        "template_id": template.template_id,
        "tier": template.tier,
        "n_input_vars": n_vars,
        "coeffs": coeffs,
        "noise_level": noise_level,
        "description": template.description,
    }


def _generate_batch(args):
    """Worker function for parallel generation."""
    (template_idx, templates, n_samples, start_seed, noise_dist, augment_prob) = args
    template = templates[template_idx]
    rng = np.random.RandomState(start_seed)

    samples = []
    attempts = 0
    max_attempts = n_samples * 3  # Allow some failures

    while len(samples) < n_samples and attempts < max_attempts:
        # Sample noise level from distribution
        noise_r = rng.random()
        if noise_r < noise_dist[0]:
            noise_level = 0.0
        elif noise_r < noise_dist[0] + noise_dist[1]:
            noise_level = 0.01
        elif noise_r < noise_dist[0] + noise_dist[1] + noise_dist[2]:
            noise_level = 0.05
        else:
            noise_level = 0.10

        scale_augment = rng.random() < augment_prob

        sample = generate_single_sample(
            template, rng,
            n_obs=rng.randint(30, MAX_OBS_POINTS + 1),
            noise_level=noise_level,
            scale_augment=scale_augment,
        )
        if sample is not None:
            samples.append(sample)
        attempts += 1

    return samples


class PhysicsDatasetGenerator:
    """Generate and manage physics equation-observation datasets."""

    def __init__(
        self,
        output_dir: str = "data",
        n_train: int = 1_000_000,
        n_val: int = 50_000,
        n_test: int = 50_000,
        seed: int = SEED,
        noise_distribution: Tuple[float, ...] = (0.40, 0.25, 0.20, 0.15),
        augment_prob: float = 0.3,
        n_workers: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = seed
        self.noise_distribution = noise_distribution
        self.augment_prob = augment_prob
        self.n_workers = min(n_workers, cpu_count())

        self.templates = get_all_templates()
        self.tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for t in self.templates:
            self.tier_counts[t.tier] = self.tier_counts.get(t.tier, 0) + 1

        print(f"Loaded {len(self.templates)} templates: "
              f"T1={self.tier_counts[1]}, T2={self.tier_counts[2]}, "
              f"T3={self.tier_counts[3]}, T4={self.tier_counts[4]}")

    def _compute_tier_allocation(self, n_total: int) -> Dict[int, int]:
        """Compute number of samples per tier."""
        return {
            1: int(n_total * 0.30),
            2: int(n_total * 0.30),
            3: int(n_total * 0.25),
            4: n_total - int(n_total * 0.30) - int(n_total * 0.30) - int(n_total * 0.25),
        }

    def generate_split(self, n_samples: int, split_name: str, base_seed: int) -> List[Dict]:
        """Generate a dataset split."""
        tier_alloc = self._compute_tier_allocation(n_samples)
        all_samples = []

        for tier in [1, 2, 3, 4]:
            tier_templates = [t for t in self.templates if t.tier == tier]
            n_per_template = tier_alloc[tier] // len(tier_templates)
            remainder = tier_alloc[tier] % len(tier_templates)

            tasks = []
            for i, template in enumerate(tier_templates):
                n = n_per_template + (1 if i < remainder else 0)
                seed = base_seed + tier * 10000 + i * 100
                tasks.append((
                    self.templates.index(template),
                    self.templates,
                    n,
                    seed,
                    self.noise_distribution,
                    self.augment_prob,
                ))

            # Generate sequentially (lambdas not picklable)
            print(f"  Generating Tier {tier}: {tier_alloc[tier]} samples from {len(tier_templates)} templates...")
            for task in tqdm(tasks, desc=f"  Tier {tier}", leave=False):
                batch = _generate_batch(task)
                all_samples.extend(batch)

        # Shuffle
        rng = np.random.RandomState(base_seed)
        rng.shuffle(all_samples)

        print(f"  {split_name}: Generated {len(all_samples)} samples (target: {n_samples})")
        return all_samples[:n_samples]

    def _samples_to_arrays(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Convert sample dicts to numpy arrays for efficient storage."""
        n = len(samples)

        # Observations: (n, MAX_OBS_POINTS, MAX_INPUT_VARS + 1)
        obs_array = np.zeros((n, MAX_OBS_POINTS, MAX_INPUT_VARS + 1), dtype=np.float32)

        # Equation tokens: (n, MAX_EQ_LENGTH)
        eq_array = np.full((n, MAX_EQ_LENGTH), EQUATION_VOCAB["[PAD]"], dtype=np.int32)

        metadata = []

        for i, s in enumerate(samples):
            obs = s["observations"]
            n_obs = obs.shape[0]
            obs_array[i, :n_obs, :] = obs

            eq_toks = s["eq_tokens"]
            eq_array[i, :len(eq_toks)] = eq_toks

            metadata.append({
                "template_id": s["template_id"],
                "tier": s["tier"],
                "n_input_vars": s["n_input_vars"],
                "n_obs": n_obs,
                "coeffs": [float(c) for c in s["coeffs"]],
                "noise_level": s["noise_level"],
                "description": s["description"],
                "eq_length": len(eq_toks),
            })

        return obs_array, eq_array, metadata

    def verify_no_leakage(self, train_meta: List[Dict], test_meta: List[Dict]) -> bool:
        """Verify no coefficient configuration leaks between train and test."""
        def make_hash(m):
            key = f"{m['template_id']}_{','.join(f'{c:.2f}' for c in m['coeffs'])}"
            return hashlib.md5(key.encode()).hexdigest()

        train_hashes = set(make_hash(m) for m in train_meta)
        test_hashes = set(make_hash(m) for m in test_meta)

        overlap = train_hashes & test_hashes
        if overlap:
            print(f"  WARNING: {len(overlap)} overlapping configurations found!")
            return False
        print(f"  No-leakage verification passed. Train: {len(train_hashes)}, Test: {len(test_hashes)}")
        return True

    def generate_and_save(self):
        """Generate all splits and save to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=== Generating Training Set ===")
        train_samples = self.generate_split(self.n_train, "train", self.seed)
        train_obs, train_eq, train_meta = self._samples_to_arrays(train_samples)

        print("\n=== Generating Validation Set ===")
        val_samples = self.generate_split(self.n_val, "val", self.seed + 1000000)
        val_obs, val_eq, val_meta = self._samples_to_arrays(val_samples)

        print("\n=== Generating Test Set ===")
        test_samples = self.generate_split(self.n_test, "test", self.seed + 2000000)
        test_obs, test_eq, test_meta = self._samples_to_arrays(test_samples)

        # Verify no leakage
        print("\n=== Verifying No Leakage ===")
        self.verify_no_leakage(train_meta, test_meta)

        # Save arrays as memory-mapped numpy
        print("\n=== Saving to disk ===")
        for name, obs, eq, meta in [
            ("train", train_obs, train_eq, train_meta),
            ("val", val_obs, val_eq, val_meta),
            ("test", test_obs, test_eq, test_meta),
        ]:
            np.save(self.output_dir / f"{name}_obs.npy", obs)
            np.save(self.output_dir / f"{name}_eq.npy", eq)
            with open(self.output_dir / f"{name}_meta.json", "w") as f:
                json.dump(meta, f)
            print(f"  Saved {name}: obs={obs.shape}, eq={eq.shape}")

        # Save dataset info
        info = {
            "n_train": len(train_meta),
            "n_val": len(val_meta),
            "n_test": len(test_meta),
            "n_templates": len(self.templates),
            "vocab_size": VOCAB_SIZE,
            "max_obs_points": MAX_OBS_POINTS,
            "max_input_vars": MAX_INPUT_VARS,
            "max_eq_length": MAX_EQ_LENGTH,
            "seed": self.seed,
            "tier_distribution": {
                str(tier): sum(1 for m in train_meta if m["tier"] == tier)
                for tier in [1, 2, 3, 4]
            },
        }
        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Report sizes
        total_bytes = sum(
            os.path.getsize(self.output_dir / f)
            for f in os.listdir(self.output_dir) if f.endswith(('.npy', '.json'))
        )
        print(f"\n  Total dataset size: {total_bytes / 1e9:.2f} GB")
        return info


class PhysicsDataset:
    """PyTorch-compatible dataset wrapper for generated physics data."""

    def __init__(self, data_dir: str, split: str = "train"):
        import torch
        from torch.utils.data import Dataset

        self.data_dir = Path(data_dir)
        self.split = split

        # Load arrays
        self.obs = np.load(self.data_dir / f"{split}_obs.npy", mmap_mode='r')
        self.eq = np.load(self.data_dir / f"{split}_eq.npy", mmap_mode='r')

        with open(self.data_dir / f"{split}_meta.json", "r") as f:
            self.meta = json.load(f)

        with open(self.data_dir / "dataset_info.json", "r") as f:
            self.info = json.load(f)

        self.n_samples = len(self.meta)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        import torch

        obs = torch.from_numpy(self.obs[idx].copy()).float()
        eq = torch.from_numpy(self.eq[idx].copy()).long()
        tier = self.meta[idx]["tier"]
        n_vars = self.meta[idx]["n_input_vars"]
        n_obs = self.meta[idx]["n_obs"]

        return {
            "observations": obs,       # (MAX_OBS_POINTS, MAX_INPUT_VARS + 1)
            "equation": eq,             # (MAX_EQ_LENGTH,)
            "tier": tier,
            "n_input_vars": n_vars,
            "n_obs": n_obs,
        }

    def get_tier_indices(self, tier: int) -> List[int]:
        """Get indices of samples belonging to a specific tier."""
        return [i for i, m in enumerate(self.meta) if m["tier"] == tier]


if __name__ == "__main__":
    generator = PhysicsDatasetGenerator(
        output_dir="data",
        n_train=1_000_000,
        n_val=50_000,
        n_test=50_000,
        seed=42,
        n_workers=8,
    )
    generator.generate_and_save()
