#!/usr/bin/env python3
"""Unit tests for token algebra module."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phys_mdt import PhysMDT
from src.token_algebra import TokenAlgebra, _is_all_special, _SPECIAL_IDS


def _make_model(**kwargs):
    defaults = dict(
        vocab_size=147, d_model=64, n_heads=4, n_layers=2,
        d_ff=128, max_seq_len=48, max_vars=5, n_points=10,
        lora_rank=0, dropout=0.1,
    )
    defaults.update(kwargs)
    return PhysMDT(**defaults)


class TestTokenAlgebra(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.model.eval()
        self.algebra = TokenAlgebra(self.model)

    def test_get_embedding(self):
        emb = self.algebra.get_embedding(10)
        self.assertEqual(emb.shape, (64,))

    def test_get_embedding_out_of_range(self):
        with self.assertRaises(ValueError):
            self.algebra.get_embedding(200)

    def test_interpolate(self):
        emb = self.algebra.interpolate(10, 20, alpha=0.5)
        self.assertEqual(emb.shape, (64,))
        # alpha=0 should give token 10 embedding
        emb_a = self.algebra.interpolate(10, 20, alpha=0.0)
        self.assertTrue(torch.allclose(emb_a, self.algebra.get_embedding(10)))
        # alpha=1 should give token 20 embedding
        emb_b = self.algebra.interpolate(10, 20, alpha=1.0)
        self.assertTrue(torch.allclose(emb_b, self.algebra.get_embedding(20)))

    def test_interpolate_invalid_alpha(self):
        with self.assertRaises(ValueError):
            self.algebra.interpolate(10, 20, alpha=1.5)

    def test_analogy(self):
        token_id, sim = self.algebra.analogy(10, 11, 12)
        self.assertIsInstance(token_id, int)
        self.assertIsInstance(sim, float)
        self.assertNotIn(token_id, _SPECIAL_IDS)
        # Should exclude input tokens
        self.assertNotIn(token_id, {10, 11, 12})

    def test_project_nearest(self):
        emb = self.algebra.get_embedding(15)
        results = self.algebra.project_nearest(emb, top_k=5)
        self.assertEqual(len(results), 5)
        # First result should be token 15 itself (highest similarity)
        self.assertEqual(results[0][0], 15)
        # Similarity should be ~1.0 for same token
        self.assertGreater(results[0][1], 0.99)

    def test_cosine_similarity(self):
        sim = self.algebra.cosine_similarity(10, 10)
        self.assertAlmostEqual(sim, 1.0, places=4)
        sim2 = self.algebra.cosine_similarity(10, 20)
        self.assertIsInstance(sim2, float)
        self.assertGreaterEqual(sim2, -1.0)
        self.assertLessEqual(sim2, 1.0)

    def test_pairwise_similarity(self):
        mat = self.algebra.pairwise_similarity([10, 20, 30])
        self.assertEqual(mat.shape, (3, 3))
        # Diagonal should be ~1.0
        for i in range(3):
            self.assertAlmostEqual(mat[i, i].item(), 1.0, places=4)

    def test_refresh(self):
        # Should not error
        self.algebra.refresh()
        emb = self.algebra.get_embedding(10)
        self.assertEqual(emb.shape, (64,))

    def test_interpolation_path(self):
        path = self.algebra.interpolation_path(10, 20, steps=5, top_k=1)
        self.assertEqual(len(path), 5)
        for step in path:
            self.assertEqual(len(step), 1)
            self.assertIsInstance(step[0][0], int)

    def test_compute_algebra_bias(self):
        token_ids = torch.tensor([[1, 10, 11, 12, 2]])
        bias = self.algebra.compute_algebra_bias(token_ids, neighbourhood_k=5, bias_strength=0.1)
        self.assertEqual(bias.shape, (1, 5, 147))


class TestIsAllSpecial(unittest.TestCase):
    def test_all_special(self):
        col = torch.tensor([0, 1, 2, 3])
        self.assertTrue(_is_all_special(col))

    def test_not_all_special(self):
        col = torch.tensor([0, 1, 10])
        self.assertFalse(_is_all_special(col))


if __name__ == '__main__':
    unittest.main()
