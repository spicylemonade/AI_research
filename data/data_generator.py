"""Procedural data generation for physics equation symbolic regression."""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from data.tokenizer import (ExprNode, ARITY, VARIABLES, CONSTANTS, INT_CONSTANTS,
                            SPECIAL_CONSTANTS, encode, MAX_SEQ_LEN)


# Physics-informed grammar weights
# Operators more common in Newtonian physics get higher weight
OPERATOR_WEIGHTS = {
    'add': 3.0, 'sub': 2.0, 'mul': 5.0, 'div': 3.0, 'pow': 2.0,
    'sqrt': 1.5, 'sin': 1.0, 'cos': 1.0, 'exp': 0.5, 'log': 0.5,
    'neg': 1.0, 'abs': 0.3, 'inv': 1.0,
}

LEAF_WEIGHTS = {
    'variable': 5.0,
    'constant': 2.0,
    'int_constant': 3.0,
    'special_constant': 1.0,
}

# Variable ranges for physics quantities
VARIABLE_RANGES = {
    0: (0.1, 100.0),    # mass-like
    1: (0.1, 50.0),     # velocity-like
    2: (0.1, 100.0),    # distance-like
    3: (0.01, 10.0),    # time-like
    4: (-np.pi, np.pi), # angle-like
    5: (0.1, 50.0),     # generic positive
    6: (0.1, 50.0),     # generic positive
    7: (-10.0, 10.0),   # generic
    8: (-10.0, 10.0),   # generic
    9: (0.1, 20.0),     # generic positive
}


def sample_operator(rng: np.random.Generator) -> str:
    ops = list(OPERATOR_WEIGHTS.keys())
    weights = np.array([OPERATOR_WEIGHTS[op] for op in ops])
    weights /= weights.sum()
    return ops[rng.choice(len(ops), p=weights)]


def sample_leaf(rng: np.random.Generator, n_vars: int, n_consts: int) -> ExprNode:
    categories = list(LEAF_WEIGHTS.keys())
    weights = np.array([LEAF_WEIGHTS[c] for c in categories])
    weights /= weights.sum()
    cat = categories[rng.choice(len(categories), p=weights)]

    if cat == 'variable':
        idx = rng.integers(0, n_vars)
        return ExprNode(VARIABLES[idx])
    elif cat == 'constant':
        idx = rng.integers(0, min(n_consts, 3))
        return ExprNode(CONSTANTS[idx])
    elif cat == 'int_constant':
        idx = rng.integers(0, 6)
        return ExprNode(INT_CONSTANTS[idx])
    else:  # special_constant
        idx = rng.integers(0, len(SPECIAL_CONSTANTS))
        return ExprNode(SPECIAL_CONSTANTS[idx])


def generate_random_tree(rng: np.random.Generator, max_depth: int, n_vars: int,
                         n_consts: int = 3, current_depth: int = 0) -> ExprNode:
    """Generate a random expression tree using physics-informed grammar."""
    # Probability of generating a leaf increases with depth
    leaf_prob = min(0.9, 0.1 + 0.2 * current_depth)

    if current_depth >= max_depth or rng.random() < leaf_prob:
        return sample_leaf(rng, n_vars, n_consts)

    op = sample_operator(rng)
    arity = ARITY[op]
    children = []
    for _ in range(arity):
        child = generate_random_tree(rng, max_depth, n_vars, n_consts, current_depth + 1)
        children.append(child)

    return ExprNode(op, children)


def tree_to_numpy_func(node: ExprNode, const_values: Dict[str, float]):
    """Convert expression tree to a numpy-evaluable function."""
    def _eval(node, x_dict):
        if node.token in VARIABLES:
            idx = int(node.token.split('_')[1])
            return x_dict[idx]
        elif node.token in CONSTANTS:
            return const_values.get(node.token, 1.0)
        elif node.token.startswith('int_'):
            return float(int(node.token.split('_')[1]))
        elif node.token == 'pi':
            return np.pi
        elif node.token == 'e_const':
            return np.e
        elif node.token == 'half':
            return 0.5
        elif node.token == 'third':
            return 1.0 / 3.0
        elif node.token == 'quarter':
            return 0.25

        children_vals = [_eval(c, x_dict) for c in node.children]

        op_map = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: np.where(np.abs(b) > 1e-10, a / b, np.full_like(a, np.nan)),
            'pow': lambda a, b: np.where((a > 0) | (b == np.floor(b)), np.power(np.abs(a) + 1e-10, b), np.full_like(a, np.nan)),
            'sqrt': lambda a: np.where(a >= 0, np.sqrt(a + 1e-10), np.full_like(a, np.nan)),
            'sin': lambda a: np.sin(a),
            'cos': lambda a: np.cos(a),
            'exp': lambda a: np.exp(np.clip(a, -20, 20)),
            'log': lambda a: np.where(a > 0, np.log(a + 1e-10), np.full_like(a, np.nan)),
            'neg': lambda a: -a,
            'abs': lambda a: np.abs(a),
            'inv': lambda a: np.where(np.abs(a) > 1e-10, 1.0 / a, np.full_like(a, np.nan)),
        }

        if node.token in op_map:
            return op_map[node.token](*children_vals)
        raise ValueError(f"Unknown token: {node.token}")

    return lambda x_dict: _eval(node, x_dict)


def generate_observation_table(tree: ExprNode, n_points: int, n_vars: int,
                                const_values: Dict[str, float],
                                rng: np.random.Generator) -> Optional[np.ndarray]:
    """Generate observation table (N, D+1) for an equation tree."""
    eval_func = tree_to_numpy_func(tree, const_values)

    # Sample input variables
    x_dict = {}
    x_cols = []
    for i in range(n_vars):
        lo, hi = VARIABLE_RANGES.get(i, (0.1, 10.0))
        x_i = rng.uniform(lo, hi, size=n_points)
        x_dict[i] = x_i
        x_cols.append(x_i)

    # Evaluate
    try:
        y = eval_func(x_dict)
        y = np.array(y, dtype=np.float64)
    except Exception:
        return None

    # Filter invalid values
    valid = np.isfinite(y) & (np.abs(y) < 1e6)
    if valid.sum() < n_points * 0.8:
        return None

    # Stack into table
    table = np.column_stack(x_cols + [y])

    # Keep only valid rows
    table = table[valid]
    if len(table) < n_points // 2:
        return None

    # Subsample to exact n_points if we have enough
    if len(table) > n_points:
        idx = rng.choice(len(table), n_points, replace=False)
        table = table[idx]

    return table.astype(np.float32)


def sample_constants(rng: np.random.Generator, n_consts: int = 3) -> Dict[str, float]:
    """Sample physically plausible constant values."""
    const_values = {}
    for i in range(n_consts):
        # Mix of different constant ranges
        choice = rng.integers(0, 4)
        if choice == 0:
            val = rng.uniform(0.1, 10.0)
        elif choice == 1:
            val = rng.uniform(-5.0, 5.0)
        elif choice == 2:
            # Powers of 10
            val = 10.0 ** rng.uniform(-2, 2)
        else:
            # Small integers (common in physics)
            val = float(rng.integers(1, 6))
        const_values[f'c_{i}'] = val
    return const_values


def generate_dataset(n_samples: int, n_points: int = 200,
                     min_vars: int = 1, max_vars: int = 5,
                     min_depth: int = 1, max_depth: int = 5,
                     seed: int = 42) -> List[Tuple[np.ndarray, List[str], List[int]]]:
    """Generate a dataset of (observation_table, prefix_tokens, token_ids) tuples."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    rng = np.random.default_rng(seed)
    dataset = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(dataset) < n_samples and attempts < max_attempts:
        attempts += 1

        n_vars = rng.integers(min_vars, max_vars + 1)
        depth = rng.integers(min_depth, max_depth + 1)

        # Generate random expression tree
        tree = generate_random_tree(rng, depth, n_vars)
        prefix = tree.to_prefix()

        # Skip if too long
        if len(prefix) + 2 > MAX_SEQ_LEN:  # +2 for SOS/EOS
            continue

        # Check it uses at least one variable
        if not tree.get_variables():
            continue

        # Sample constants and generate observation table
        const_values = sample_constants(rng)
        table = generate_observation_table(tree, n_points, n_vars, const_values, rng)

        if table is None:
            continue

        # Check output has reasonable variance
        y = table[:, -1]
        if np.std(y) < 1e-6 or np.std(y) > 1e5:
            continue

        # Encode tokens
        token_ids = encode(prefix)

        dataset.append((table, prefix, token_ids))

        if len(dataset) % 5000 == 0:
            print(f"Generated {len(dataset)}/{n_samples} samples ({attempts} attempts)")

    print(f"Generated {len(dataset)} samples in {attempts} attempts "
          f"(success rate: {len(dataset)/max(attempts,1)*100:.1f}%)")
    return dataset


if __name__ == '__main__':
    import time
    start = time.time()
    dataset = generate_dataset(1000, seed=42)
    elapsed = time.time() - start
    print(f"Generated {len(dataset)} samples in {elapsed:.1f}s")
    print(f"Example prefix: {dataset[0][1]}")
    print(f"Example token IDs: {dataset[0][2][:15]}...")
    print(f"Example table shape: {dataset[0][0].shape}")
