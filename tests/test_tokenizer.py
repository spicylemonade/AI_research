#!/usr/bin/env python3
"""Unit tests for the physics tokenizer."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenizer import PhysicsTokenizer
from data.generator import build_templates


class TestTokenizerBasics(unittest.TestCase):
    def setUp(self):
        self.tok = PhysicsTokenizer()

    def test_vocab_size(self):
        self.assertGreaterEqual(self.tok.vocab_size, 140)

    def test_special_tokens(self):
        self.assertEqual(self.tok.token_to_id['[PAD]'], 0)
        self.assertEqual(self.tok.token_to_id['[BOS]'], 1)
        self.assertEqual(self.tok.token_to_id['[EOS]'], 2)
        self.assertEqual(self.tok.token_to_id['[MASK]'], 3)

    def test_encode_simple(self):
        ids = self.tok.encode("mul m a", add_special=True)
        self.assertEqual(ids[0], self.tok.bos_id)
        self.assertEqual(ids[-1], self.tok.eos_id)
        self.assertEqual(len(ids), 5)  # BOS mul m a EOS

    def test_decode_simple(self):
        ids = self.tok.encode("mul m a")
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, "mul m a")

    def test_roundtrip_simple(self):
        original = "add v0 mul a t"
        ids = self.tok.encode(original)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, original)

    def test_roundtrip_complex(self):
        original = "div mul G_const mul m1 m2 pow r INT_2"
        ids = self.tok.encode(original)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, original)


class TestRoundTripFidelity(unittest.TestCase):
    """Test round-trip fidelity on all generator templates."""

    def setUp(self):
        self.tok = PhysicsTokenizer()
        self.templates = build_templates()

    def test_all_templates_roundtrip(self):
        """Every template's prefix notation should survive encode/decode."""
        failures = []
        for t in self.templates:
            prefix = t.prefix_notation
            ids = self.tok.encode(prefix)
            decoded = self.tok.decode(ids)
            if decoded != prefix:
                failures.append((t.name, prefix, decoded))
        self.assertEqual(len(failures), 0,
                         f"Round-trip failures: {failures[:5]}")


class TestTreeDepth(unittest.TestCase):
    def setUp(self):
        self.tok = PhysicsTokenizer()

    def test_simple_depth(self):
        depths = self.tok.get_tree_depths("mul m a")
        self.assertEqual(depths, [0, 1, 1])

    def test_nested_depth(self):
        depths = self.tok.get_tree_depths("add v0 mul a t")
        # add is root (0), v0 is arg1 (1), mul is arg2 (1), a and t are under mul (2)
        self.assertEqual(depths, [0, 1, 1, 2, 2])


class TestPadding(unittest.TestCase):
    def setUp(self):
        self.tok = PhysicsTokenizer()

    def test_padding(self):
        ids = self.tok.encode("mul m a", max_length=10)
        self.assertEqual(len(ids), 10)
        self.assertEqual(ids[-1], self.tok.pad_id)

    def test_truncation(self):
        long_expr = "add mul a b mul c d"
        ids = self.tok.encode(long_expr, max_length=5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[-1], self.tok.eos_id)


class TestSkeleton(unittest.TestCase):
    def setUp(self):
        self.tok = PhysicsTokenizer()

    def test_skeleton_encoding(self):
        skel = self.tok.encode_skeleton("mul m a")
        # STRUCT_BOS, SKEL_mul, LEAF_VAR, LEAF_VAR, STRUCT_EOS
        self.assertEqual(len(skel), 5)


if __name__ == '__main__':
    unittest.main()
