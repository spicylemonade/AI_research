#!/usr/bin/env python3
"""Unit tests for structure predictor module."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.structure_predictor import (
    StructurePredictor, STRUCTURE_VOCAB, STRUCTURE_VOCAB_SIZE,
    SKELETON_OPS, SKELETON_LEAVES, FULL_TO_SKEL,
    compute_skeleton_loss,
)


class TestStructureVocab(unittest.TestCase):
    def test_vocab_size(self):
        self.assertEqual(STRUCTURE_VOCAB_SIZE, len(STRUCTURE_VOCAB))
        self.assertEqual(STRUCTURE_VOCAB_SIZE, 24)

    def test_special_tokens(self):
        self.assertEqual(STRUCTURE_VOCAB[0], 'STRUCT_PAD')
        self.assertEqual(STRUCTURE_VOCAB[1], 'STRUCT_BOS')
        self.assertEqual(STRUCTURE_VOCAB[2], 'STRUCT_EOS')
        self.assertEqual(STRUCTURE_VOCAB[3], 'STRUCT_MASK')

    def test_skeleton_ops(self):
        self.assertIn('SKEL_add', SKELETON_OPS)
        self.assertIn('OP_BINARY', SKELETON_OPS)
        self.assertIn('OP_UNARY', SKELETON_OPS)

    def test_skeleton_leaves(self):
        self.assertIn('LEAF_VAR', SKELETON_LEAVES)
        self.assertIn('LEAF_CONST', SKELETON_LEAVES)

    def test_full_to_skel_mapping(self):
        self.assertEqual(FULL_TO_SKEL['add'], 'SKEL_add')
        self.assertEqual(FULL_TO_SKEL['sin'], 'SKEL_sin')
        self.assertEqual(len(FULL_TO_SKEL), 12)


class TestStructurePredictor(unittest.TestCase):
    def setUp(self):
        self.model = StructurePredictor(
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, max_vars=5, n_points=10, max_skel_len=16,
        )

    def test_forward_shape(self):
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        skel_ids = torch.randint(0, STRUCTURE_VOCAB_SIZE, (2, 8))
        logits = self.model(X, Y, skel_ids)
        self.assertEqual(logits.shape, (2, 8, STRUCTURE_VOCAB_SIZE))

    def test_encode_observations(self):
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        memory = self.model.encode_observations(X, Y)
        self.assertEqual(memory.shape, (2, 10, 64))

    def test_generate_shape(self):
        self.model.eval()
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        generated = self.model.generate(X, Y, max_len=12)
        self.assertEqual(generated.shape[0], 2)
        self.assertLessEqual(generated.shape[1], 12)
        # First token should be BOS (1)
        self.assertTrue((generated[:, 0] == 1).all())

    def test_skeleton_to_mask_constraints(self):
        skeleton_ids = torch.tensor([[1, 10, 6, 7, 2]])  # BOS, SKEL_add, LEAF_VAR, LEAF_CONST, EOS
        constraints = self.model.skeleton_to_mask_constraints(skeleton_ids, full_vocab_size=147)
        self.assertEqual(constraints.shape, (1, 5, 147))
        # All positions should have at least some True values
        for j in range(5):
            self.assertTrue(constraints[0, j].any())

    def test_gradient_flow(self):
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        skel_ids = torch.randint(0, STRUCTURE_VOCAB_SIZE, (2, 8))
        logits = self.model(X, Y, skel_ids)
        loss = logits.mean()
        loss.backward()
        has_grad = False
        for p in self.model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "No gradients flowing")

    def test_parameter_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 10000)


class TestSkeletonLoss(unittest.TestCase):
    def test_loss_computation(self):
        pred = torch.randn(2, 8, STRUCTURE_VOCAB_SIZE)
        target = torch.randint(0, STRUCTURE_VOCAB_SIZE, (2, 8))
        loss = compute_skeleton_loss(pred, target)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_loss_with_padding(self):
        pred = torch.randn(2, 8, STRUCTURE_VOCAB_SIZE)
        target = torch.zeros(2, 8, dtype=torch.long)  # All PAD
        target[:, 0] = 1  # At least one non-pad token
        loss = compute_skeleton_loss(pred, target, pad_id=0)
        self.assertFalse(torch.isnan(loss))


if __name__ == '__main__':
    unittest.main()
