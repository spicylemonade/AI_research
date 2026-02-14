"""
Synthetic training data generator for symbolic regression.
Generates random expression trees and evaluates them on random input points
following the NeSymReS data generation protocol (Biggio et al., 2021).
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

# Fixed random seed for reproducibility
SEED = 42

# Operator definitions
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']
UNARY_OPS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'neg', 'abs']
ALL_OPS = BINARY_OPS + UNARY_OPS

# LaTeX mappings
LATEX_MAP = {
    'add': lambda a, b: f'({a} + {b})',
    'sub': lambda a, b: f'({a} - {b})',
    'mul': lambda a, b: f'({a} \\cdot {b})',
    'div': lambda a, b: f'\\frac{{{a}}}{{{b}}}',
    'pow': lambda a, b: f'{{{a}}}^{{{b}}}',
    'sin': lambda a: f'\\sin({a})',
    'cos': lambda a: f'\\cos({a})',
    'exp': lambda a: f'e^{{{a}}}',
    'log': lambda a: f'\\log({a})',
    'sqrt': lambda a: f'\\sqrt{{{a}}}',
    'neg': lambda a: f'(-{a})',
    'abs': lambda a: f'|{a}|',
}

# NumPy evaluation functions
NUMPY_OPS = {
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'div': np.divide,
    'pow': np.power,
    'sin': np.sin,
    'cos': np.cos,
    'exp': np.exp,
    'log': np.log,
    'sqrt': np.sqrt,
    'neg': np.negative,
    'abs': np.abs,
}


@dataclass
class ExprNode:
    """Node in an expression tree."""
    op: Optional[str] = None       # Operator name or None for leaf
    children: List['ExprNode'] = field(default_factory=list)
    value: Optional[str] = None    # Variable name (x1, x2...) or constant value

    def is_leaf(self) -> bool:
        return self.op is None

    def to_prefix(self) -> List[str]:
        """Convert to prefix notation token list."""
        if self.is_leaf():
            return [self.value]
        tokens = [self.op]
        for child in self.children:
            tokens.extend(child.to_prefix())
        return tokens

    def to_latex(self) -> str:
        """Convert to LaTeX string."""
        if self.is_leaf():
            if self.value.startswith('x'):
                return self.value
            return str(self.value)
        child_latex = [c.to_latex() for c in self.children]
        return LATEX_MAP[self.op](*child_latex)

    def evaluate(self, variables: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate expression on variable arrays."""
        if self.is_leaf():
            if self.value in variables:
                return variables[self.value]
            return np.full_like(list(variables.values())[0], float(self.value))
        child_vals = [c.evaluate(variables) for c in self.children]
        if self.op == 'pow':
            # Safe power: avoid complex numbers
            base = child_vals[0]
            exp_val = child_vals[1]
            # Only allow power when base > 0 or exponent is integer
            with np.errstate(all='ignore'):
                result = NUMPY_OPS[self.op](np.abs(base) + 1e-10, exp_val)
            return result
        elif self.op == 'div':
            with np.errstate(all='ignore'):
                denom = child_vals[1]
                denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
                return child_vals[0] / denom
        elif self.op == 'log':
            with np.errstate(all='ignore'):
                return np.log(np.abs(child_vals[0]) + 1e-10)
        elif self.op == 'sqrt':
            with np.errstate(all='ignore'):
                return np.sqrt(np.abs(child_vals[0]))
        elif self.op == 'exp':
            with np.errstate(all='ignore'):
                return np.exp(np.clip(child_vals[0], -10, 10))
        else:
            return NUMPY_OPS[self.op](*child_vals)

    def node_count(self) -> int:
        """Count total nodes in the expression tree."""
        if self.is_leaf():
            return 1
        return 1 + sum(c.node_count() for c in self.children)


class EquationGenerator:
    """Generates random symbolic equations following NeSymReS protocol."""

    def __init__(
        self,
        max_depth: int = 5,
        max_variables: int = 5,
        num_support_points: int = 200,
        input_range: Tuple[float, float] = (-5.0, 5.0),
        binary_ops: List[str] = None,
        unary_ops: List[str] = None,
        constant_range: Tuple[float, float] = (-5.0, 5.0),
        integer_constants: List[int] = None,
        max_output_value: float = 1e6,
        seed: int = SEED,
    ):
        self.max_depth = max_depth
        self.max_variables = max_variables
        self.num_support_points = num_support_points
        self.input_range = input_range
        self.binary_ops = binary_ops or BINARY_OPS
        self.unary_ops = unary_ops or UNARY_OPS
        self.constant_range = constant_range
        self.integer_constants = integer_constants or [0, 1, 2, 3, -1]
        self.max_output_value = max_output_value
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def _random_tree(self, depth: int, num_vars: int, p_leaf: float = 0.3) -> ExprNode:
        """Generate a random expression tree."""
        if depth <= 0 or (depth < self.max_depth and random.random() < p_leaf):
            # Leaf node: variable or constant
            if random.random() < 0.7:  # 70% chance of variable
                var_idx = random.randint(1, num_vars)
                return ExprNode(value=f'x{var_idx}')
            else:
                # Constant: integer or placeholder C
                if random.random() < 0.6:
                    c = random.choice(self.integer_constants)
                    return ExprNode(value=str(c))
                else:
                    return ExprNode(value='C')

        # Internal node: operator
        if random.random() < 0.6:  # 60% binary, 40% unary
            op = random.choice(self.binary_ops)
            left = self._random_tree(depth - 1, num_vars, p_leaf + 0.1)
            right = self._random_tree(depth - 1, num_vars, p_leaf + 0.1)
            return ExprNode(op=op, children=[left, right])
        else:
            op = random.choice(self.unary_ops)
            child = self._random_tree(depth - 1, num_vars, p_leaf + 0.1)
            return ExprNode(op=op, children=[child])

    def _replace_constants(self, node: ExprNode) -> ExprNode:
        """Replace C placeholders with random numerical constants."""
        if node.is_leaf():
            if node.value == 'C':
                val = self.rng.uniform(*self.constant_range)
                node.value = f'{val:.4f}'
            return node
        for child in node.children:
            self._replace_constants(child)
        return node

    def _check_valid(self, y: np.ndarray) -> bool:
        """Check if output values are valid (no NaN, Inf, or extreme values)."""
        if np.any(np.isnan(y)):
            return False
        if np.any(np.isinf(y)):
            return False
        if np.any(np.abs(y) > self.max_output_value):
            return False
        # Check for near-constant functions (boring)
        if np.std(y) < 1e-6:
            return False
        return True

    def _count_used_variables(self, node: ExprNode) -> set:
        """Find which variables are actually used in the expression."""
        if node.is_leaf():
            if node.value and node.value.startswith('x'):
                return {node.value}
            return set()
        result = set()
        for child in node.children:
            result |= self._count_used_variables(child)
        return result

    def generate_one(self, num_vars: int = None, depth: int = None) -> Optional[Dict[str, Any]]:
        """Generate a single (observations, equation) pair.

        Returns None if the equation is degenerate.
        """
        if num_vars is None:
            num_vars = random.randint(1, self.max_variables)
        if depth is None:
            depth = random.randint(1, self.max_depth)

        # Generate random tree
        tree = self._random_tree(depth, num_vars)

        # Get prefix tokens (with C placeholders)
        prefix_skeleton = tree.to_prefix()

        # Replace constants with numerical values for evaluation
        tree_eval = self._replace_constants(tree)

        # Generate support points
        X = self.rng.uniform(
            self.input_range[0], self.input_range[1],
            size=(self.num_support_points, num_vars)
        )

        # Create variable dict
        variables = {f'x{i+1}': X[:, i] for i in range(num_vars)}

        # Evaluate
        try:
            with np.errstate(all='ignore'):
                y = tree_eval.evaluate(variables)
        except (ValueError, OverflowError, ZeroDivisionError):
            return None

        # Validate
        if not self._check_valid(y):
            return None

        # Get actual used variables
        used_vars = self._count_used_variables(tree_eval)
        actual_num_vars = len(used_vars)
        if actual_num_vars == 0:
            return None

        # Build result
        prefix_tokens = tree_eval.to_prefix()
        latex_str = tree_eval.to_latex()

        return {
            'prefix_tokens': prefix_tokens,
            'latex': latex_str,
            'num_variables': actual_num_vars,
            'num_operators': tree_eval.node_count() - actual_num_vars,
            'support_points_x': X.tolist(),
            'support_points_y': y.tolist(),
            'variables_used': sorted(list(used_vars)),
            'tree_depth': depth,
            'node_count': tree_eval.node_count(),
        }

    def generate_batch(self, n: int, num_vars: int = None, depth: int = None) -> List[Dict[str, Any]]:
        """Generate a batch of valid equations."""
        results = []
        attempts = 0
        max_attempts = n * 10
        while len(results) < n and attempts < max_attempts:
            eq = self.generate_one(num_vars=num_vars, depth=depth)
            if eq is not None:
                results.append(eq)
            attempts += 1
        return results

    def generate_dataset(self, n: int, save_path: str = None) -> List[Dict[str, Any]]:
        """Generate a full dataset of n equations."""
        print(f"Generating {n} equations...")
        dataset = self.generate_batch(n)
        print(f"Generated {len(dataset)} valid equations from attempts")

        if save_path:
            import json
            # Convert numpy arrays to lists for JSON serialization
            with open(save_path, 'w') as f:
                json.dump(dataset, f)
            print(f"Saved to {save_path}")

        return dataset


def prefix_to_infix(tokens: List[str]) -> str:
    """Convert prefix token list to infix string for readability."""
    pos = [0]

    def parse():
        if pos[0] >= len(tokens):
            return '?'
        token = tokens[pos[0]]
        pos[0] += 1

        if token in BINARY_OPS:
            left = parse()
            right = parse()
            return f'({left} {token} {right})'
        elif token in UNARY_OPS:
            child = parse()
            return f'{token}({child})'
        else:
            return token

    return parse()


# Convenience function
def generate_training_data(n: int = 10000, seed: int = SEED, **kwargs) -> List[Dict[str, Any]]:
    """Generate n training equations with default settings."""
    gen = EquationGenerator(seed=seed, **kwargs)
    return gen.generate_batch(n)


if __name__ == '__main__':
    import time

    gen = EquationGenerator(seed=42)

    # Benchmark generation speed
    start = time.time()
    n_test = 1000
    data = gen.generate_batch(n_test)
    elapsed = time.time() - start

    rate = len(data) / elapsed * 3600
    print(f"Generated {len(data)} equations in {elapsed:.2f}s")
    print(f"Rate: {rate:.0f} equations/hour")
    print(f"Sample prefix: {data[0]['prefix_tokens']}")
    print(f"Sample latex: {data[0]['latex']}")
    print(f"Sample infix: {prefix_to_infix(data[0]['prefix_tokens'])}")
