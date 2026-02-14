"""Dataset generation and loading pipeline for PhysMDT.

Generates (data_matrix, equation_tokens) pairs from FSReD equations and
procedurally generated Newtonian physics equations.
"""

import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import sympy
from sympy import Symbol, pi, E, sin, cos, tan, log, exp, sqrt, Abs
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.tokenizer import (
    EquationTokenizer, FSRED_EQUATIONS, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
)

# Symbols for evaluation
SYMBOLS = {f'x{i}': Symbol(f'x{i}') for i in range(1, 10)}


def _count_variables(expr_str: str) -> int:
    """Count how many distinct variables appear in an expression."""
    count = 0
    for i in range(1, 10):
        if f'x{i}' in expr_str:
            count += 1
    return max(count, 1)


def _generate_data_points(expr_str: str, n_points: int = 1000,
                          noise_std: float = 0.0,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) data points for a given equation.

    Args:
        expr_str: Symbolic expression string.
        n_points: Number of data points to generate.
        noise_std: Standard deviation of Gaussian noise (relative to signal).
        seed: Random seed for reproducibility.

    Returns:
        X: (n_points, n_vars) input data matrix.
        y: (n_points,) output values.
    """
    rng = np.random.RandomState(seed)
    n_vars = _count_variables(expr_str)

    # Parse expression
    local_dict = dict(SYMBOLS)
    local_dict['pi'] = pi
    local_dict['e'] = E
    try:
        expr = sympy.sympify(expr_str, locals=local_dict)
    except Exception:
        expr = sympy.parse_expr(expr_str, local_dict=local_dict)

    # Generate input values in physically reasonable ranges
    X = rng.uniform(0.1, 5.0, size=(n_points * 3, n_vars))

    # Evaluate expression
    y_list = []
    X_valid = []
    for i in range(len(X)):
        vals = {Symbol(f'x{j+1}'): float(X[i, j]) for j in range(n_vars)}
        try:
            val = complex(expr.subs(vals))
            if np.isfinite(val.real) and abs(val.imag) < 1e-10 and abs(val.real) < 1e10:
                y_list.append(val.real)
                X_valid.append(X[i])
                if len(y_list) >= n_points:
                    break
        except Exception:
            continue

    if len(y_list) < n_points:
        # Fill remaining with additional attempts in wider range
        X_extra = rng.uniform(0.5, 3.0, size=(n_points * 5, n_vars))
        for i in range(len(X_extra)):
            if len(y_list) >= n_points:
                break
            vals = {Symbol(f'x{j+1}'): float(X_extra[i, j]) for j in range(n_vars)}
            try:
                val = complex(expr.subs(vals))
                if np.isfinite(val.real) and abs(val.imag) < 1e-10 and abs(val.real) < 1e10:
                    y_list.append(val.real)
                    X_valid.append(X_extra[i])
            except Exception:
                continue

    if len(y_list) == 0:
        # Fallback: return dummy data
        X_valid = [rng.uniform(0.1, 5.0, size=n_vars) for _ in range(n_points)]
        y_list = [0.0] * n_points

    X_out = np.array(X_valid[:n_points], dtype=np.float32)
    y_out = np.array(y_list[:n_points], dtype=np.float32)

    # Add noise
    if noise_std > 0 and len(y_out) > 0:
        signal_std = np.std(y_out) + 1e-10
        noise = rng.normal(0, noise_std * signal_std, size=len(y_out))
        y_out = y_out + noise.astype(np.float32)

    return X_out, y_out


# ---------------------------------------------------------------------------
# Newtonian Physics Equation Templates
# ---------------------------------------------------------------------------

NEWTONIAN_TEMPLATES = [
    # Format: (template_str, n_vars, description)
    ("C1*x1*x2", 2, "F=ma"),
    ("C1*x1*x2**2/2", 2, "Kinetic energy"),
    ("C1*x1*x2/x3**2", 3, "Gravitational force"),
    ("-C1*x1*x2/x3", 3, "Gravitational PE"),
    ("sqrt(C1*x1/x2)", 2, "Orbital velocity"),
    ("C1*x1**2*x2", 2, "Centripetal force"),
    ("x1*sin(C1*x2+C2)", 2, "SHO position"),
    ("x1*exp(-C1*x2)*cos(C2*x2)", 2, "Damped oscillator"),
    ("C1*sqrt(x1/x2)", 2, "Pendulum period"),
    ("C1*x1*x2**2", 2, "Spring PE"),
    ("x1*x2**2*x3", 3, "Angular momentum"),
    ("C1*x1*x2**2", 2, "Moment of inertia"),
    ("x1**2*sin(C1*x2)/x3", 3, "Projectile range"),
    ("x1*x2*cos(x3)", 3, "Work"),
    ("x1**2/x2", 2, "Centripetal acceleration"),
    ("C1*x1*x2 + C2*x3*x4", 4, "Two-body energy"),
    ("x1*cos(x2)*x3", 3, "Component force"),
    ("sqrt(x1**2+x2**2)", 2, "Magnitude"),
    ("x1*x2/(x3+x4)", 4, "Coupled system"),
    ("C1*x1*exp(-C2*x2)*sin(C3*x2)", 2, "Driven oscillator"),
]


def _generate_procedural_equations(n_equations: int = 50000,
                                   seed: int = 42) -> List[Tuple[str, str]]:
    """Generate Newtonian physics equations with random coefficients.

    Returns: List of (equation_str, description) tuples.
    """
    rng = random.Random(seed)
    equations = []

    for i in range(n_equations):
        template_str, n_vars, desc = rng.choice(NEWTONIAN_TEMPLATES)

        # Random coefficients
        coeffs = {}
        for c_idx in range(1, 5):
            c_name = f'C{c_idx}'
            if c_name in template_str:
                # Generate physically meaningful constants
                coeff = rng.choice([1, 2, 3, 4, 5, -1, -2]) * rng.uniform(0.5, 5.0)
                coeff = round(coeff, 2)
                coeffs[c_name] = coeff

        # Substitute coefficients
        eq_str = template_str
        for c_name, c_val in coeffs.items():
            if c_val < 0:
                eq_str = eq_str.replace(c_name, f'({c_val})')
            else:
                eq_str = eq_str.replace(c_name, str(c_val))

        equations.append((eq_str, f"{desc}_variant_{i}"))

    return equations


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class EquationDataset(Dataset):
    """Dataset of (data_matrix, equation_tokens) pairs.

    Each item contains:
    - data_matrix: (n_data_points, n_vars+1) float tensor (X columns + y column)
    - token_ids: (max_seq_len,) long tensor of equation token IDs
    - n_vars: number of input variables
    - difficulty: 'easy', 'medium', or 'hard'
    """

    def __init__(self, equations: List[Tuple[str, str]],
                 tokenizer: EquationTokenizer,
                 n_data_points: int = 256,
                 noise_std: float = 0.0,
                 seed: int = 42):
        """
        Args:
            equations: List of (expr_str, description) tuples.
            tokenizer: EquationTokenizer instance.
            n_data_points: Number of data points per equation.
            noise_std: Gaussian noise level (relative).
            seed: Random seed.
        """
        self.equations = equations
        self.tokenizer = tokenizer
        self.n_data_points = n_data_points
        self.noise_std = noise_std
        self.seed = seed

    def __len__(self):
        return len(self.equations)

    def __getitem__(self, idx):
        expr_str, desc = self.equations[idx]
        n_vars = _count_variables(expr_str)

        # Generate data
        X, y = _generate_data_points(
            expr_str, n_points=self.n_data_points,
            noise_std=self.noise_std,
            seed=self.seed + idx
        )

        # Normalize data for stability
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X - X_mean) / X_std
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mean) / y_std

        # Combine into data matrix: (n_points, n_vars+1)
        data_matrix = np.column_stack([X_norm, y_norm[:, None] if y_norm.ndim == 1 else y_norm])
        data_matrix = data_matrix[:self.n_data_points]

        # Pad to fixed size if fewer points
        if len(data_matrix) < self.n_data_points:
            pad = np.zeros((self.n_data_points - len(data_matrix), data_matrix.shape[1]),
                           dtype=np.float32)
            data_matrix = np.vstack([data_matrix, pad])

        # Pad columns to max 10 (9 vars + 1 target)
        max_cols = 10
        if data_matrix.shape[1] < max_cols:
            pad_cols = np.zeros((data_matrix.shape[0], max_cols - data_matrix.shape[1]),
                                dtype=np.float32)
            data_matrix = np.hstack([data_matrix, pad_cols])

        # Tokenize equation
        token_ids = self.tokenizer.encode(expr_str)

        # Determine difficulty
        n_ops = sum(1 for c in expr_str if c in '+-*/^')
        if n_vars <= 2 and n_ops <= 3:
            difficulty = 'easy'
        elif n_vars <= 4 and n_ops <= 6:
            difficulty = 'medium'
        else:
            difficulty = 'hard'

        return {
            'data_matrix': torch.tensor(data_matrix, dtype=torch.float32),
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'n_vars': n_vars,
            'difficulty': difficulty,
            'expr_str': expr_str,
            'description': desc,
        }


def collate_fn(batch):
    """Custom collation for variable-size data matrices."""
    data_matrices = torch.stack([item['data_matrix'] for item in batch])
    token_ids = torch.stack([item['token_ids'] for item in batch])
    n_vars = [item['n_vars'] for item in batch]
    difficulties = [item['difficulty'] for item in batch]
    expr_strs = [item['expr_str'] for item in batch]
    descriptions = [item['description'] for item in batch]

    return {
        'data_matrix': data_matrices,
        'token_ids': token_ids,
        'n_vars': n_vars,
        'difficulty': difficulties,
        'expr_str': expr_strs,
        'description': descriptions,
    }


def get_fsred_equations() -> List[Tuple[str, str]]:
    """Get all 120 FSReD equations as (expr_str, description) tuples."""
    equations = []
    for i, eq in enumerate(FSRED_EQUATIONS[:120]):
        if i < 30:
            diff = 'easy'
        elif i < 70:
            diff = 'medium'
        else:
            diff = 'hard'
        equations.append((eq, f"fsred_{i:03d}_{diff}"))
    return equations


def create_dataloaders(
    include_fsred: bool = True,
    include_procedural: bool = True,
    n_procedural: int = 50000,
    n_data_points: int = 256,
    noise_std: float = 0.0,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_seq_len: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    Args:
        include_fsred: Include FSReD equations.
        include_procedural: Include procedurally generated equations.
        n_procedural: Number of procedural equations.
        n_data_points: Data points per equation.
        noise_std: Gaussian noise level.
        batch_size: Batch size.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        seed: Random seed.
        max_seq_len: Max token sequence length.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    all_equations = []
    if include_fsred:
        all_equations.extend(get_fsred_equations())
    if include_procedural:
        all_equations.extend(_generate_procedural_equations(n_procedural, seed))

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(all_equations)

    n_total = len(all_equations)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_eqs = all_equations[:n_train]
    val_eqs = all_equations[n_train:n_train + n_val]
    test_eqs = all_equations[n_train + n_val:]

    tokenizer = EquationTokenizer(max_seq_len=max_seq_len)

    train_ds = EquationDataset(train_eqs, tokenizer, n_data_points, noise_std, seed)
    val_ds = EquationDataset(val_eqs, tokenizer, n_data_points, noise_std, seed + 1)
    test_ds = EquationDataset(test_eqs, tokenizer, n_data_points, noise_std, seed + 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader, test_loader
