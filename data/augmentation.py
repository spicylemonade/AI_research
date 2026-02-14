"""Data augmentation for physics equation symbolic regression."""

import numpy as np
from typing import Tuple


def noise_injection(table: np.ndarray, noise_level: float = 0.05,
                    rng: np.random.Generator = None) -> np.ndarray:
    """Add Gaussian noise to observation values.

    Args:
        table: (N, D+1) observation table
        noise_level: standard deviation of noise as fraction of column std
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = table.copy()
    for col in range(table.shape[1]):
        col_std = np.std(table[:, col])
        if col_std > 1e-10:
            noise = rng.normal(0, noise_level * col_std, size=table.shape[0])
            noisy[:, col] += noise
    return noisy


def variable_permutation(table: np.ndarray, prefix_tokens: list,
                         rng: np.random.Generator = None) -> Tuple[np.ndarray, list]:
    """Randomly permute the order of input variables.

    Args:
        table: (N, D+1) observation table (last col is target)
        prefix_tokens: list of prefix notation tokens
        rng: random number generator

    Returns:
        Permuted table and updated prefix tokens
    """
    if rng is None:
        rng = np.random.default_rng()

    n_vars = table.shape[1] - 1
    if n_vars <= 1:
        return table.copy(), prefix_tokens.copy()

    perm = rng.permutation(n_vars)
    new_table = table.copy()
    for i, j in enumerate(perm):
        new_table[:, i] = table[:, j]

    # Update tokens: rename x_i -> x_perm[i]
    inv_perm = np.argsort(perm)
    new_tokens = []
    for tok in prefix_tokens:
        if tok.startswith('x_'):
            idx = int(tok.split('_')[1])
            if idx < n_vars:
                new_tokens.append(f'x_{inv_perm[idx]}')
            else:
                new_tokens.append(tok)
        else:
            new_tokens.append(tok)

    return new_table, new_tokens


def unit_rescaling(table: np.ndarray, prefix_tokens: list,
                   rng: np.random.Generator = None) -> Tuple[np.ndarray, list]:
    """Rescale variables to simulate different unit systems.

    Multiplies each input column by a random scale factor and adjusts
    the target accordingly (since we can't easily adjust prefix tokens,
    we let the constant tokens absorb the change).

    Args:
        table: (N, D+1) observation table
        prefix_tokens: list of prefix notation tokens
        rng: random number generator

    Returns:
        Rescaled table and original prefix tokens (constants may need re-fitting)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_vars = table.shape[1] - 1
    new_table = table.copy()

    # Rescale input columns
    scales = rng.uniform(0.5, 2.0, size=n_vars)
    for i in range(n_vars):
        new_table[:, i] *= scales[i]

    # The target column stays the same (equation form doesn't change,
    # but constants need to absorb the scale changes during BFGS post-processing)

    return new_table, prefix_tokens.copy()


def observation_subsampling(table: np.ndarray, subsample_ratio: float = 0.7,
                            rng: np.random.Generator = None) -> np.ndarray:
    """Randomly subsample observation points.

    Args:
        table: (N, D+1) observation table
        subsample_ratio: fraction of points to keep
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    n = table.shape[0]
    n_keep = max(int(n * subsample_ratio), 10)  # keep at least 10 points
    idx = rng.choice(n, size=n_keep, replace=False)
    return table[idx]


def augment_sample(table: np.ndarray, prefix_tokens: list,
                   noise_level: float = 0.05,
                   subsample_ratio: float = 0.7,
                   permute: bool = True,
                   rescale: bool = True,
                   rng: np.random.Generator = None) -> Tuple[np.ndarray, list]:
    """Apply a random combination of augmentations.

    Args:
        table: (N, D+1) observation table
        prefix_tokens: list of prefix notation tokens
        noise_level: noise injection level
        subsample_ratio: observation subsampling ratio
        permute: whether to apply variable permutation
        rescale: whether to apply unit rescaling
        rng: random number generator

    Returns:
        Augmented table and (potentially) modified prefix tokens
    """
    if rng is None:
        rng = np.random.default_rng()

    aug_table = table.copy()
    aug_tokens = prefix_tokens.copy()

    # Apply augmentations
    aug_table = noise_injection(aug_table, noise_level, rng)

    if permute:
        aug_table, aug_tokens = variable_permutation(aug_table, aug_tokens, rng)

    if rescale:
        aug_table, aug_tokens = unit_rescaling(aug_table, aug_tokens, rng)

    aug_table = observation_subsampling(aug_table, subsample_ratio, rng)

    return aug_table, aug_tokens
