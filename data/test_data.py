"""Unit tests for data pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.tokenizer import (encode, decode, _parse_prefix, ExprNode, tree_to_sympy,
                            VOCAB_SIZE, MAX_SEQ_LEN, PAD_ID, SOS_ID, EOS_ID, MASK_ID,
                            TOKEN_TO_ID, ALL_TOKENS)
from data.data_generator import (generate_random_tree, generate_observation_table,
                                  sample_constants, generate_dataset)
from data.augmentation import (noise_injection, variable_permutation,
                                observation_subsampling, unit_rescaling, augment_sample)
from data.feynman_loader import generate_benchmark_data, generate_ood_data


def test_vocab():
    """Test vocabulary is complete and valid."""
    assert VOCAB_SIZE <= 128, f"Vocabulary too large: {VOCAB_SIZE}"
    assert len(ALL_TOKENS) == len(set(ALL_TOKENS)), "Duplicate tokens!"
    assert '<SOS>' in TOKEN_TO_ID
    assert '<EOS>' in TOKEN_TO_ID
    assert '<PAD>' in TOKEN_TO_ID
    assert '<MASK>' in TOKEN_TO_ID
    print("  [PASS] Vocabulary test")


def test_roundtrip_encoding():
    """Test that encoding and decoding are inverse operations."""
    test_cases = [
        ['mul', 'x_0', 'x_1'],  # F = m * a
        ['add', 'x_0', 'mul', 'x_1', 'x_2'],  # v = v0 + a*t
        ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],  # E_k = 0.5*m*v^2
        ['div', 'mul', 'mul', 'c_0', 'x_0', 'x_1', 'pow', 'x_2', 'int_2'],  # F = G*m1*m2/r^2
        ['sqrt', 'div', 'mul', 'c_0', 'x_0', 'x_1'],  # v = sqrt(G*M/r)
    ]

    for tokens in test_cases:
        ids = encode(tokens)
        decoded = decode(ids)
        assert decoded == tokens, f"Roundtrip failed: {tokens} -> {ids} -> {decoded}"

    print("  [PASS] Roundtrip encoding test")


def test_prefix_parsing():
    """Test prefix notation to tree parsing."""
    # F = m * a
    tree, next_pos = _parse_prefix(['mul', 'x_0', 'x_1'], 0)
    assert tree.token == 'mul'
    assert len(tree.children) == 2
    assert tree.children[0].token == 'x_0'
    assert tree.children[1].token == 'x_1'
    assert next_pos == 3

    # E_k = 0.5 * m * v^2
    tree, next_pos = _parse_prefix(['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'], 0)
    assert tree.token == 'mul'
    assert tree.depth() == 3
    assert tree.num_operators() == 3
    assert tree.get_variables() == {'x_0', 'x_1'}

    print("  [PASS] Prefix parsing test")


def test_tree_to_sympy():
    """Test conversion from expression tree to SymPy expression."""
    import sympy as sp

    # F = m * a -> mul x_0 x_1
    tree, _ = _parse_prefix(['mul', 'x_0', 'x_1'], 0)
    expr = tree_to_sympy(tree)
    x0, x1 = sp.Symbol('x0'), sp.Symbol('x1')
    assert sp.simplify(expr - x0 * x1) == 0, f"SymPy conversion failed: {expr}"

    # E_k = 0.5 * m * v^2
    tree, _ = _parse_prefix(['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'], 0)
    expr = tree_to_sympy(tree)
    expected = sp.Rational(1, 2) * x0 * x1**2
    assert sp.simplify(expr - expected) == 0, f"SymPy conversion failed: {expr}"

    print("  [PASS] Tree to SymPy test")


def test_data_generation():
    """Test that data generation produces valid samples."""
    rng = np.random.default_rng(42)
    tree = generate_random_tree(rng, max_depth=3, n_vars=3)
    prefix = tree.to_prefix()

    assert len(prefix) > 0
    assert tree.get_variables()

    const_values = sample_constants(rng)
    table = generate_observation_table(tree, 200, 3, const_values, rng)
    # table might be None for some random trees, which is fine
    if table is not None:
        assert table.shape[1] == 4  # 3 vars + 1 target
        assert np.all(np.isfinite(table))

    print("  [PASS] Data generation test")


def test_dataset_generation():
    """Test batch dataset generation."""
    dataset = generate_dataset(100, n_points=50, seed=42)
    assert len(dataset) >= 50, f"Expected at least 50 samples, got {len(dataset)}"

    for table, prefix, token_ids in dataset[:5]:
        assert table.ndim == 2
        assert len(token_ids) == MAX_SEQ_LEN
        assert token_ids[0] == SOS_ID
        # Check EOS exists somewhere
        assert EOS_ID in token_ids

    print("  [PASS] Dataset generation test")


def test_benchmark_data():
    """Test Feynman benchmark data generation."""
    data = generate_benchmark_data(n_points=50, seed=42)
    assert len(data) >= 30, f"Expected at least 30 benchmark equations, got {len(data)}"

    tiers = {1: 0, 2: 0, 3: 0, 4: 0}
    for d in data:
        tier = d['tier']
        tiers[tier] = tiers.get(tier, 0) + 1
        assert d['table'].shape[0] > 0, f"Empty table for {d['id']}"
        assert len(d['prefix']) > 0

    print(f"  [PASS] Benchmark data test (tiers: {tiers})")


def test_ood_data():
    """Test OOD data generation."""
    data = generate_ood_data(n_points=50, seed=42)
    assert len(data) >= 6, f"Expected at least 6 OOD equations, got {len(data)}"
    print(f"  [PASS] OOD data test ({len(data)} equations)")


def test_augmentation():
    """Test all 4 augmentation types."""
    rng = np.random.default_rng(42)
    table = rng.uniform(0, 10, size=(100, 4)).astype(np.float32)  # 3 vars + 1 target
    prefix = ['mul', 'x_0', 'add', 'x_1', 'x_2']

    # 1. Noise injection
    noisy = noise_injection(table, 0.05, rng)
    assert noisy.shape == table.shape
    assert not np.allclose(noisy, table)  # Should be different
    assert np.allclose(noisy, table, atol=5.0)  # But not too different

    # 2. Variable permutation
    perm_table, perm_tokens = variable_permutation(table, prefix, rng)
    assert perm_table.shape == table.shape
    assert set(tok for tok in perm_tokens if tok.startswith('x_')) == set(tok for tok in prefix if tok.startswith('x_'))

    # 3. Observation subsampling
    sub_table = observation_subsampling(table, 0.7, rng)
    assert sub_table.shape[0] < table.shape[0]
    assert sub_table.shape[1] == table.shape[1]

    # 4. Unit rescaling
    scaled_table, scaled_tokens = unit_rescaling(table, prefix, rng)
    assert scaled_table.shape == table.shape
    assert not np.allclose(scaled_table[:, :-1], table[:, :-1])  # Input cols changed

    # Combined augmentation
    aug_table, aug_tokens = augment_sample(table, prefix, rng=rng)
    assert aug_table.shape[1] == table.shape[1]

    print("  [PASS] Augmentation test (all 4 types)")


if __name__ == '__main__':
    print("Running data pipeline unit tests...")
    test_vocab()
    test_roundtrip_encoding()
    test_prefix_parsing()
    test_tree_to_sympy()
    test_data_generation()
    test_dataset_generation()
    test_benchmark_data()
    test_ood_data()
    test_augmentation()
    print("\nAll tests passed!")
