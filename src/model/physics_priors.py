"""
Physics-informed structural priors for symbolic regression.
Implements dimensional analysis, arity constraints, symmetry augmentation,
and compositionality priors for equation derivation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import copy
import random

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.decoder import VOCAB, BINARY_OPS, UNARY_OPS, ID_TO_TOKEN


@dataclass
class PhysicsPriorsConfig:
    """Configuration for physics priors. All are toggleable."""
    enable_dimensional_analysis: bool = True
    enable_arity_constraints: bool = True
    enable_symmetry_augmentation: bool = True
    enable_compositionality: bool = True
    dimensional_loss_weight: float = 0.1
    compositionality_loss_weight: float = 0.3


# ============================================================
# 1. Dimensional Analysis
# ============================================================

# Base SI dimensions: [M, L, T, I, Theta, N, J] (mass, length, time, current, temperature, amount, luminosity)
# We use a simplified 3D system: [M, L, T] for mechanics

DIMENSION_MAP = {
    # Common physical quantities and their dimensions [M, L, T]
    'mass': [1, 0, 0],
    'length': [0, 1, 0],
    'time': [0, 0, 1],
    'velocity': [0, 1, -1],
    'acceleration': [0, 1, -2],
    'force': [1, 1, -2],
    'energy': [1, 2, -2],
    'power': [1, 2, -3],
    'pressure': [1, -1, -2],
    'charge': [0, 0, 1],  # simplified
    'voltage': [1, 2, -3],  # simplified
    'dimensionless': [0, 0, 0],
}


def check_dimensional_consistency(prefix_tokens: List[str], var_dimensions: Dict[str, List[int]]) -> bool:
    """Check if an expression is dimensionally consistent.

    Rules:
    - add/sub: operands must have same dimensions
    - mul: dimensions add
    - div: dimensions subtract
    - pow: base dimensions multiply by exponent (must be dimensionless or integer)
    - sin/cos/exp/log: argument must be dimensionless, result is dimensionless
    - sqrt: dimensions halved

    Returns True if consistent, False if violation detected.
    """
    pos = [0]

    def get_dimensions():
        if pos[0] >= len(prefix_tokens):
            return None
        token = prefix_tokens[pos[0]]
        pos[0] += 1

        if token in BINARY_OPS:
            left = get_dimensions()
            right = get_dimensions()
            if left is None or right is None:
                return None

            if token in ('add', 'sub'):
                # Must have same dimensions
                if left != right:
                    return None  # Dimensional violation!
                return left
            elif token == 'mul':
                return [l + r for l, r in zip(left, right)]
            elif token == 'div':
                return [l - r for l, r in zip(left, right)]
            elif token == 'pow':
                # Simplified: if right is dimensionless, multiply left dims by the numeric value
                if right == [0, 0, 0]:
                    return left  # Approximate: can't determine exact power
                return left  # Simplified

        elif token in UNARY_OPS:
            child = get_dimensions()
            if child is None:
                return None

            if token in ('sin', 'cos', 'tan', 'exp', 'log', 'asin', 'acos', 'atan'):
                # Argument should be dimensionless
                if child != [0, 0, 0]:
                    return None  # Dimensional violation!
                return [0, 0, 0]
            elif token == 'sqrt':
                return [d / 2.0 for d in child]
            elif token in ('neg', 'abs'):
                return child

        elif token in var_dimensions:
            return var_dimensions[token]
        else:
            # Constants are dimensionless
            return [0, 0, 0]

    try:
        result = get_dimensions()
        return result is not None
    except Exception:
        return False


def dimensional_analysis_loss(logits: torch.Tensor, token_ids: torch.Tensor,
                               var_dimensions: Optional[Dict[str, List[int]]] = None) -> torch.Tensor:
    """Compute soft dimensional analysis penalty.

    For each predicted sequence, check if add/sub operations have
    matching dimensions. Returns a penalty term.
    """
    # Simplified: penalize when add/sub is predicted but children have different predicted types
    # This is approximate since we don't know future tokens during generation
    B, T, V = logits.shape

    # Get predicted tokens
    pred_tokens = logits.argmax(dim=-1)  # [B, T]

    penalty = torch.zeros(1, device=logits.device)

    add_id = VOCAB.get('add', -1)
    sub_id = VOCAB.get('sub', -1)

    # For each add/sub token, check if the softmax distribution over the
    # two children positions shows similar operator types
    # This is a soft heuristic that encourages dimensional consistency
    for b in range(B):
        for t in range(T - 2):
            if pred_tokens[b, t].item() in (add_id, sub_id):
                # Soft constraint: the distributions for the two children
                # should be similar (encouraging same-type operands)
                if t + 1 < T and t + 2 < T:
                    dist1 = F.softmax(logits[b, t + 1], dim=-1)
                    dist2 = F.softmax(logits[b, t + 2], dim=-1)
                    # KL divergence as penalty
                    kl = F.kl_div(dist1.log(), dist2, reduction='sum')
                    penalty = penalty + kl

    return penalty / max(B, 1)


# ============================================================
# 2. Operator Arity Constraints
# ============================================================

def get_valid_next_tokens(prefix_so_far: List[str], max_depth: int = 10) -> Set[int]:
    """Get valid next token IDs based on arity constraints.

    In prefix notation:
    - Binary ops consume 2 subtrees
    - Unary ops consume 1 subtree
    - Leaves (variables, constants) consume 0

    We track the "arity stack" to know how many more tokens are needed.
    """
    # Count remaining needed subtrees
    needed = 1  # We need 1 subtree (the whole expression)
    depth = 0

    for token in prefix_so_far:
        if token in BINARY_OPS:
            needed += 1  # binary op needs 2 subtrees, consumed 1 spot, net +1
            depth += 1
        elif token in UNARY_OPS:
            depth += 1  # unary needs 1 subtree, consumed 1 spot, net 0
        else:
            needed -= 1  # leaf consumes 1 needed spot

    if needed <= 0:
        # Expression is complete, only EOS is valid
        return {VOCAB['EOS']}

    valid = set()

    # Variables and constants are always valid as leaves
    for token_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                        'C', '0', '1', '2', '3', '4', '5', 'pi', 'e', '-1', '-2']:
        if token_name in VOCAB:
            valid.add(VOCAB[token_name])

    # Operators are valid if we haven't exceeded max depth
    if depth < max_depth:
        for op in list(BINARY_OPS) + list(UNARY_OPS):
            if op in VOCAB:
                valid.add(VOCAB[op])

    return valid


def apply_arity_mask(logits: torch.Tensor, prefix_so_far: List[List[str]]) -> torch.Tensor:
    """Apply arity constraint mask to logits.

    Sets logits of invalid tokens to -inf.

    Args:
        logits: [B, V] next-token logits
        prefix_so_far: List of token string lists for each batch element

    Returns:
        Masked logits [B, V]
    """
    B, V = logits.shape
    masked_logits = logits.clone()

    for b in range(B):
        valid_ids = get_valid_next_tokens(prefix_so_far[b])
        mask = torch.ones(V, dtype=torch.bool, device=logits.device)
        for vid in valid_ids:
            if 0 <= vid < V:
                mask[vid] = False
        masked_logits[b, mask] = float('-inf')

    return masked_logits


# ============================================================
# 3. Symmetry Augmentation
# ============================================================

def augment_commutative(prefix_tokens: List[str]) -> List[List[str]]:
    """Generate commutative variants of an expression.

    For add and mul operations, swap operands.
    Returns list of augmented token lists including the original.
    """
    variants = [prefix_tokens]

    # Find commutative operations and generate swapped versions
    def find_subtree_end(tokens, start):
        """Find the end index of a subtree starting at position start."""
        if start >= len(tokens):
            return start
        token = tokens[start]
        if token in BINARY_OPS:
            # Skip op, then find end of left subtree, then right subtree
            left_end = find_subtree_end(tokens, start + 1)
            right_end = find_subtree_end(tokens, left_end)
            return right_end
        elif token in UNARY_OPS:
            return find_subtree_end(tokens, start + 1)
        else:
            return start + 1

    commutative_ops = {'add', 'mul'}

    for i, token in enumerate(prefix_tokens):
        if token in commutative_ops:
            # Find the two subtrees
            left_start = i + 1
            left_end = find_subtree_end(prefix_tokens, left_start)
            right_start = left_end
            right_end = find_subtree_end(prefix_tokens, right_start)

            if right_end <= len(prefix_tokens):
                left_subtree = prefix_tokens[left_start:left_end]
                right_subtree = prefix_tokens[right_start:right_end]

                # Create swapped version
                swapped = (prefix_tokens[:left_start] +
                          right_subtree + left_subtree +
                          prefix_tokens[right_end:])
                if swapped != prefix_tokens:
                    variants.append(swapped)

    return variants


def augment_rescale(prefix_tokens: List[str], scale_range: Tuple[float, float] = (0.1, 10.0)) -> List[str]:
    """Generate unit-rescaled variant by wrapping expression in mul(C, expr).

    This simulates a change of units.
    """
    scale = random.uniform(*scale_range)
    return ['mul', f'{scale:.4f}'] + prefix_tokens


# ============================================================
# 4. Compositionality Prior
# ============================================================

def decompose_expression(prefix_tokens: List[str]) -> List[List[str]]:
    """Decompose a complex expression into sub-expression derivation chain.

    Strategy: extract subtrees as intermediate steps.
    Returns list of (sub_expression_tokens) from simplest to full.
    """
    if len(prefix_tokens) <= 3:
        return [prefix_tokens]

    steps = []

    def find_subtree_end(tokens, start):
        if start >= len(tokens):
            return start
        token = tokens[start]
        if token in BINARY_OPS:
            left_end = find_subtree_end(tokens, start + 1)
            right_end = find_subtree_end(tokens, left_end)
            return right_end
        elif token in UNARY_OPS:
            return find_subtree_end(tokens, start + 1)
        else:
            return start + 1

    # Extract meaningful subtrees (at least 3 tokens)
    def extract_subtrees(tokens, start=0, depth=0):
        if start >= len(tokens):
            return
        token = tokens[start]
        if token in BINARY_OPS:
            left_start = start + 1
            left_end = find_subtree_end(tokens, left_start)
            right_end = find_subtree_end(tokens, left_end)

            # Add subtrees if non-trivial
            left_subtree = tokens[left_start:left_end]
            right_subtree = tokens[left_end:right_end]

            if len(left_subtree) >= 3:
                steps.append(left_subtree)
            if len(right_subtree) >= 3:
                steps.append(right_subtree)

            extract_subtrees(tokens, left_start, depth + 1)
            extract_subtrees(tokens, left_end, depth + 1)

        elif token in UNARY_OPS:
            child_end = find_subtree_end(tokens, start + 1)
            child_subtree = tokens[start + 1:child_end]
            if len(child_subtree) >= 3:
                steps.append(child_subtree)
            extract_subtrees(tokens, start + 1, depth + 1)

    extract_subtrees(prefix_tokens)

    # Sort by complexity (length) and deduplicate
    seen = set()
    unique_steps = []
    for step in sorted(steps, key=len):
        step_tuple = tuple(step)
        if step_tuple not in seen:
            seen.add(step_tuple)
            unique_steps.append(step)

    # Add the full expression at the end
    unique_steps.append(prefix_tokens)

    # Limit to 5 steps
    if len(unique_steps) > 5:
        # Keep first, last, and evenly spaced middle steps
        indices = [0] + list(np.linspace(1, len(unique_steps)-2, 3, dtype=int)) + [len(unique_steps)-1]
        unique_steps = [unique_steps[i] for i in sorted(set(indices))]

    return unique_steps


def compositionality_loss(logits_chain: List[torch.Tensor],
                          target_chain: List[torch.Tensor],
                          weight: float = 0.3) -> torch.Tensor:
    """Compute chain-of-thought loss on intermediate derivation steps.

    Args:
        logits_chain: List of [B, T, V] logits for each step
        target_chain: List of [B, T] targets for each step
        weight: Weight for intermediate steps (final step gets weight 1.0)

    Returns:
        Weighted loss
    """
    total_loss = torch.tensor(0.0, device=logits_chain[0].device)
    n_steps = len(logits_chain)

    for i, (logits, targets) in enumerate(zip(logits_chain, target_chain)):
        step_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=VOCAB['PAD'],
        )

        if i < n_steps - 1:
            total_loss = total_loss + weight * step_loss
        else:
            total_loss = total_loss + step_loss

    return total_loss / n_steps


# ============================================================
# Unit Tests
# ============================================================

def run_tests():
    print("Running physics priors tests...")

    # Test 1: Dimensional consistency - valid: m*a (force = mass * acceleration)
    valid = check_dimensional_consistency(
        ['mul', 'x1', 'x2'],
        {'x1': [1, 0, 0], 'x2': [0, 1, -2]}  # mass * acceleration
    )
    assert valid == True, "Test 1 failed: m*a should be dimensionally consistent"
    print("  [PASS] m*a is dimensionally consistent")

    # Test 2: Dimensional violation - invalid: m + a (adding mass to acceleration)
    invalid = check_dimensional_consistency(
        ['add', 'x1', 'x2'],
        {'x1': [1, 0, 0], 'x2': [0, 1, -2]}  # mass + acceleration
    )
    assert invalid == False, "Test 2 failed: m+a should be dimensionally inconsistent"
    print("  [PASS] m+a detected as dimensionally inconsistent")

    # Test 3: sin of dimensionless quantity - valid
    valid = check_dimensional_consistency(
        ['sin', 'x1'],
        {'x1': [0, 0, 0]}  # dimensionless
    )
    assert valid == True, "Test 3 failed: sin(dimensionless) should be valid"
    print("  [PASS] sin(dimensionless) is valid")

    # Test 4: sin of dimensional quantity - invalid
    invalid = check_dimensional_consistency(
        ['sin', 'x1'],
        {'x1': [0, 1, 0]}  # length
    )
    assert invalid == False, "Test 4 failed: sin(length) should be invalid"
    print("  [PASS] sin(length) detected as invalid")

    # Test 5: Arity constraints
    valid_ids = get_valid_next_tokens([])  # Empty prefix - need one subtree
    assert VOCAB['x1'] in valid_ids, "Test 5a failed: x1 should be valid first token"
    assert VOCAB['add'] in valid_ids, "Test 5b failed: add should be valid first token"
    print("  [PASS] Arity constraints for empty prefix")

    valid_ids = get_valid_next_tokens(['add', 'x1', 'x2'])  # Complete expression
    assert VOCAB['EOS'] in valid_ids, "Test 5c failed: EOS should be valid after complete expression"
    print("  [PASS] EOS valid after complete expression")

    # Test 6: Commutative augmentation
    variants = augment_commutative(['add', 'x1', 'x2'])
    assert ['add', 'x1', 'x2'] in variants, "Test 6a failed: original should be in variants"
    assert ['add', 'x2', 'x1'] in variants, "Test 6b failed: swapped should be in variants"
    print(f"  [PASS] Commutative augmentation: {len(variants)} variants")

    # Test 7: Expression decomposition
    steps = decompose_expression(['add', 'mul', 'x1', 'x2', 'sin', 'x3'])
    assert len(steps) >= 2, f"Test 7 failed: should have >=2 steps, got {len(steps)}"
    assert steps[-1] == ['add', 'mul', 'x1', 'x2', 'sin', 'x3'], "Test 7: last step should be full expression"
    print(f"  [PASS] Expression decomposition: {len(steps)} steps")

    # Test 8: Arity mask application
    logits = torch.randn(2, len(VOCAB))
    prefixes = [['add', 'x1'], ['mul', 'x1', 'x2']]
    masked = apply_arity_mask(logits, prefixes)
    # First batch element needs one more leaf or subtree (needed=1)
    # Second batch element is complete (needed=0), only EOS valid
    assert masked[1, VOCAB['EOS']] != float('-inf'), "Test 8a failed: EOS should be valid for complete expr"
    assert masked[1, VOCAB['add']] == float('-inf'), "Test 8b failed: add should be invalid for complete expr"
    print("  [PASS] Arity mask application")

    # Test 9: Rescale augmentation
    random.seed(42)
    rescaled = augment_rescale(['x1'])
    assert rescaled[0] == 'mul', "Test 9a failed: rescaled should start with mul"
    assert len(rescaled) == 3, f"Test 9b failed: rescaled should have 3 tokens, got {len(rescaled)}"
    print("  [PASS] Rescale augmentation")

    # Test 10: Dimensional analysis loss
    B, T, V = 2, 5, len(VOCAB)
    logits = torch.randn(B, T, V)
    token_ids = torch.randint(0, V, (B, T))
    loss = dimensional_analysis_loss(logits, token_ids)
    assert loss.shape == (1,), f"Test 10 failed: loss shape {loss.shape}"
    print("  [PASS] Dimensional analysis loss computation")

    # Test 11: Compositionality loss
    B, T, V = 1, 4, len(VOCAB)
    logits_chain = [torch.randn(B, T, V), torch.randn(B, T, V)]
    target_chain = [torch.randint(0, V, (B, T)), torch.randint(0, V, (B, T))]
    closs = compositionality_loss(logits_chain, target_chain)
    assert closs.dim() == 0 or closs.shape == (1,) or closs.numel() == 1, "Test 11 failed: compositionality loss shape"
    print("  [PASS] Compositionality loss computation")

    # Test 12: Config toggleable
    config = PhysicsPriorsConfig(enable_dimensional_analysis=False)
    assert config.enable_dimensional_analysis == False
    assert config.enable_arity_constraints == True  # default
    print("  [PASS] Config flags are toggleable")

    # Test 13: Dimensional consistency for deeper expressions
    # E = mc^2 -> mul(m, pow(c, 2))
    valid = check_dimensional_consistency(
        ['mul', 'x1', 'pow', 'x2', 'x3'],
        {'x1': [1, 0, 0], 'x2': [0, 1, -1], 'x3': [0, 0, 0]}  # mass * velocity^(dimensionless)
    )
    assert valid == True, "Test 13 failed: E=mc^2 pattern should be valid"
    print("  [PASS] Deeper expression dimensional consistency (E=mc^2 pattern)")

    # Test 14: Commutative augmentation for nested expression
    variants = augment_commutative(['mul', 'add', 'x1', 'x2', 'x3'])
    assert ['mul', 'add', 'x1', 'x2', 'x3'] in variants, "Test 14a: original"
    # mul is commutative: swap (add x1 x2) and x3
    assert ['mul', 'x3', 'add', 'x1', 'x2'] in variants, "Test 14b: swapped mul children"
    # add is commutative: swap x1 and x2
    assert ['mul', 'add', 'x2', 'x1', 'x3'] in variants, "Test 14c: swapped add children"
    print(f"  [PASS] Nested commutative augmentation: {len(variants)} variants")

    # Test 15: Arity constraints - unary op needs one more child
    valid_ids = get_valid_next_tokens(['sin'])  # sin needs one child
    assert VOCAB['x1'] in valid_ids, "Test 15a: x1 valid after sin"
    assert VOCAB['add'] in valid_ids, "Test 15b: add valid after sin (would create subtree)"
    print("  [PASS] Arity constraints for unary operator")

    # Test 16: DIMENSION_MAP has expected entries
    assert DIMENSION_MAP['force'] == [1, 1, -2], "Test 16: force dimensions"
    assert DIMENSION_MAP['energy'] == [1, 2, -2], "Test 16: energy dimensions"
    print("  [PASS] DIMENSION_MAP entries correct")

    print("\nAll physics priors tests passed!")


if __name__ == '__main__':
    run_tests()
