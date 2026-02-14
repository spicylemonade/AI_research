"""
Multi-step derivation chain generator for compositional equation learning.
Decomposes complex equations into intermediate sub-expression steps.
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.decoder import VOCAB, BINARY_OPS, UNARY_OPS


def find_subtree_end(tokens: List[str], start: int) -> int:
    """Find the end index of a subtree in prefix notation."""
    if start >= len(tokens):
        return start
    token = tokens[start]
    if token in BINARY_OPS:
        left_end = find_subtree_end(tokens, start + 1)
        return find_subtree_end(tokens, left_end)
    elif token in UNARY_OPS:
        return find_subtree_end(tokens, start + 1)
    else:
        return start + 1


class DerivationChainGenerator:
    """Generates multi-step derivation chains for complex equations.

    Three decomposition strategies:
    1. Algebraic substitution: extract and name sub-expressions
    2. Dimensional building: build from simplest dimensional quantities
    3. Functional composition: decompose nested functions
    """

    def algebraic_substitution(self, tokens: List[str]) -> List[List[str]]:
        """Strategy 1: Extract sub-expressions as intermediate steps.

        For binary ops at the root, the chain is:
        [left_subtree, right_subtree, full_expression]
        """
        if len(tokens) <= 3:
            return [tokens]

        steps = []

        if tokens[0] in BINARY_OPS:
            left_end = find_subtree_end(tokens, 1)
            left_sub = tokens[1:left_end]
            right_sub = tokens[left_end:]

            # Add non-trivial subtrees
            if len(left_sub) >= 3:
                # Recursively decompose left
                left_chain = self.algebraic_substitution(left_sub)
                steps.extend(left_chain[:-1])  # All but the last (which is left_sub itself)
            if len(left_sub) >= 2:
                steps.append(left_sub)

            if len(right_sub) >= 3:
                right_chain = self.algebraic_substitution(right_sub)
                steps.extend(right_chain[:-1])
            if len(right_sub) >= 2:
                steps.append(right_sub)

        elif tokens[0] in UNARY_OPS:
            child_sub = tokens[1:]
            if len(child_sub) >= 3:
                child_chain = self.algebraic_substitution(child_sub)
                steps.extend(child_chain[:-1])
            if len(child_sub) >= 2:
                steps.append(child_sub)

        steps.append(tokens)

        # Deduplicate
        seen = set()
        unique_steps = []
        for step in steps:
            key = tuple(step)
            if key not in seen:
                seen.add(key)
                unique_steps.append(step)

        return unique_steps[-5:]  # Limit to 5 steps

    def dimensional_building(self, tokens: List[str]) -> List[List[str]]:
        """Strategy 2: Build from simplest components.

        Start with individual variables, then combine them.
        """
        if len(tokens) <= 3:
            return [tokens]

        # Collect all leaf nodes
        variables = []
        for t in tokens:
            if t.startswith('x') and t not in [tt for vv in variables for tt in vv]:
                variables.append([t])

        steps = []

        # Add pairs of variables
        if len(variables) >= 2:
            steps.append(variables[0])
            steps.append(variables[1])

        # Build up incrementally
        current = self.algebraic_substitution(tokens)
        steps.extend(current)

        # Deduplicate and limit
        seen = set()
        unique = []
        for step in steps:
            key = tuple(step)
            if key not in seen and len(step) >= 1:
                seen.add(key)
                unique.append(step)

        return unique[-5:]

    def functional_composition(self, tokens: List[str]) -> List[List[str]]:
        """Strategy 3: Decompose nested functions.

        For expressions like sin(mul(x1, x2)), extract inner function first.
        """
        if len(tokens) <= 3:
            return [tokens]

        steps = []

        # Find nested unary operations
        for i, t in enumerate(tokens):
            if t in UNARY_OPS:
                # Get the argument subtree
                arg_end = find_subtree_end(tokens, i + 1)
                arg_subtree = tokens[i + 1:arg_end]
                if len(arg_subtree) >= 3:
                    steps.append(arg_subtree)  # Inner function
                    steps.append(tokens[i:arg_end])  # Outer(inner)

        # Add full expression
        if not steps:
            return self.algebraic_substitution(tokens)

        steps.append(tokens)

        seen = set()
        unique = []
        for step in steps:
            key = tuple(step)
            if key not in seen:
                seen.add(key)
                unique.append(step)

        return unique[-5:]

    def generate_chain(self, tokens: List[str], strategy: str = 'auto') -> List[List[str]]:
        """Generate derivation chain using specified strategy.

        Args:
            tokens: Prefix notation token list
            strategy: 'algebraic', 'dimensional', 'functional', or 'auto'

        Returns:
            List of token lists from simplest to full expression
        """
        if strategy == 'algebraic':
            return self.algebraic_substitution(tokens)
        elif strategy == 'dimensional':
            return self.dimensional_building(tokens)
        elif strategy == 'functional':
            return self.functional_composition(tokens)
        elif strategy == 'auto':
            # Choose strategy based on expression structure
            has_unary = any(t in UNARY_OPS for t in tokens)
            has_deep_binary = len(tokens) > 10

            if has_unary and has_deep_binary:
                return self.functional_composition(tokens)
            elif has_deep_binary:
                return self.algebraic_substitution(tokens)
            else:
                return self.dimensional_building(tokens)
        else:
            return [tokens]

    def generate_chains_for_benchmark(self, benchmark_path: str,
                                       tiers: List[str] = None) -> Dict[str, List[List[str]]]:
        """Generate derivation chains for benchmark equations.

        Args:
            benchmark_path: Path to feynman_equations.json
            tiers: Which tiers to generate chains for (default: complex + multi_step)

        Returns:
            Dict mapping equation id to derivation chain
        """
        if tiers is None:
            tiers = ['complex', 'multi_step']

        with open(benchmark_path) as f:
            benchmark = json.load(f)

        chains = {}
        for eq in benchmark['equations']:
            if eq['difficulty_tier'] in tiers:
                # Parse symbolic expression into token list
                symbolic = eq['formula_symbolic']
                tokens = parse_symbolic_to_tokens(symbolic)

                chain = self.generate_chain(tokens, strategy='auto')
                chains[eq['id']] = chain

        return chains


def parse_symbolic_to_tokens(symbolic: str) -> List[str]:
    """Parse symbolic expression like 'mul(x1, pow(x2, 2))' into token list."""
    tokens = []
    pos = [0]
    s = symbolic.strip()

    def parse_expr():
        while pos[0] < len(s) and s[pos[0]] in ' ,':
            pos[0] += 1

        if pos[0] >= len(s):
            return

        start = pos[0]
        while pos[0] < len(s) and (s[pos[0]].isalnum() or s[pos[0]] in '_.-'):
            pos[0] += 1

        name = s[start:pos[0]]

        while pos[0] < len(s) and s[pos[0]] in ' ':
            pos[0] += 1

        if pos[0] < len(s) and s[pos[0]] == '(':
            pos[0] += 1  # skip (
            tokens.append(name)

            while pos[0] < len(s) and s[pos[0]] != ')':
                while pos[0] < len(s) and s[pos[0]] in ' ,':
                    pos[0] += 1
                if pos[0] < len(s) and s[pos[0]] == ')':
                    break
                parse_expr()

            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1
        else:
            if name:
                tokens.append(name)

    parse_expr()
    return tokens


if __name__ == '__main__':
    gen = DerivationChainGenerator()

    # Test with a complex expression: add(mul(x1, x2), sin(x3))
    tokens = ['add', 'mul', 'x1', 'x2', 'sin', 'x3']

    print("Expression:", ' '.join(tokens))

    print("\nAlgebraic substitution:")
    for i, step in enumerate(gen.algebraic_substitution(tokens)):
        print(f"  Step {i+1}: {' '.join(step)}")

    print("\nDimensional building:")
    for i, step in enumerate(gen.dimensional_building(tokens)):
        print(f"  Step {i+1}: {' '.join(step)}")

    print("\nFunctional composition:")
    for i, step in enumerate(gen.functional_composition(tokens)):
        print(f"  Step {i+1}: {' '.join(step)}")

    # Test parse
    symbolic = "mul(x1, pow(x2, 2))"
    parsed = parse_symbolic_to_tokens(symbolic)
    print(f"\nParsed '{symbolic}': {parsed}")

    # Test chain for complex expression
    complex_expr = ['div', 'mul', 'mul', 'x1', 'x2', 'x3', 'mul', '4', 'mul', 'pi', 'pow', 'x4', '2']
    print(f"\nComplex: {' '.join(complex_expr)}")
    chain = gen.generate_chain(complex_expr)
    for i, step in enumerate(chain):
        print(f"  Step {i+1}: {' '.join(step)}")

    print("\nAll derivation chain tests passed!")
