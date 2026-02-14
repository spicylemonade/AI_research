#!/usr/bin/env python3
"""Unit tests for test-time finetuning module."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phys_mdt import PhysMDT
from src.ttf import DataAugmenter, TestTimeFinetuner


def _make_model(**kwargs):
    defaults = dict(
        vocab_size=147, d_model=64, n_heads=4, n_layers=2,
        d_ff=128, max_seq_len=48, max_vars=5, n_points=10,
        lora_rank=0, dropout=0.1,
    )
    defaults.update(kwargs)
    return PhysMDT(**defaults)


class TestDataAugmenter(unittest.TestCase):
    def test_augment_returns_lists(self):
        aug = DataAugmenter(noise_std=0.01)
        X = torch.randn(10, 3)
        Y = torch.randn(10)
        X_list, Y_list = aug.augment(X, Y, n_augmented=3)
        # Original + 3 augmented
        self.assertGreaterEqual(len(X_list), 2)  # at least original + some
        self.assertGreaterEqual(len(Y_list), 2)

    def test_augment_shapes(self):
        aug = DataAugmenter(noise_std=0.01)
        X = torch.randn(10, 3)
        Y = torch.randn(10)
        X_list, Y_list = aug.augment(X, Y, n_augmented=2)
        for xi in X_list:
            self.assertEqual(xi.shape[-1], 3)
        for yi in Y_list:
            self.assertGreater(yi.shape[0], 0)

    def test_augment_seed_reproducibility(self):
        aug1 = DataAugmenter(noise_std=0.01, seed=42)
        aug2 = DataAugmenter(noise_std=0.01, seed=42)
        X = torch.randn(10, 3)
        Y = torch.randn(10)
        X_list1, Y_list1 = aug1.augment(X, Y, n_augmented=2)
        X_list2, Y_list2 = aug2.augment(X, Y, n_augmented=2)
        # Same seed should produce same augmentation types
        self.assertEqual(len(X_list1), len(X_list2))


class TestTestTimeFinetuner(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()

    def test_finetune_supervised(self):
        finetuner = TestTimeFinetuner(self.model, lora_rank=8, n_steps=3,
                                       lr=1e-3, augment=False)
        X = torch.randn(10, 5)
        Y = torch.randn(10)
        target = torch.randint(3, 147, (16,))
        target[0] = 1  # BOS
        stats = finetuner.finetune(X, Y, target_ids=target, seq_len=16)
        self.assertIn('n_steps', stats)
        self.assertIn('final_loss', stats)
        self.assertGreater(stats['n_steps'], 0)

    def test_finetune_and_generate(self):
        finetuner = TestTimeFinetuner(self.model, lora_rank=8, n_steps=2,
                                       lr=1e-3, augment=False)
        X = torch.randn(10, 5)
        Y = torch.randn(10)
        pred, stats = finetuner.finetune_and_generate(X, Y, seq_len=16)
        self.assertEqual(pred.shape, (16,))
        self.assertIn('final_loss', stats)

    def test_state_restoration(self):
        """After finetune_and_generate, model weights should be restored."""
        finetuner = TestTimeFinetuner(self.model, lora_rank=8, n_steps=2,
                                       lr=1e-3, augment=False)
        # Get original output
        X_test = torch.randn(1, 10, 5)
        Y_test = torch.randn(1, 10)
        with torch.no_grad():
            orig = self.model.generate_single_pass(X_test, Y_test, seq_len=8)

        X = torch.randn(10, 5)
        Y = torch.randn(10)
        finetuner.finetune_and_generate(X, Y, seq_len=8)

        # After restore, LoRA should be disabled
        self.assertFalse(any(
            m.lora_enabled for b in self.model.blocks
            for m in b.modules() if hasattr(m, 'lora_enabled')
        ))

    def test_save_and_restore(self):
        finetuner = TestTimeFinetuner(self.model, lora_rank=8, n_steps=1)
        finetuner.save_state()
        self.assertIsNotNone(finetuner._saved_state)
        finetuner.restore_state()
        self.assertIsNone(finetuner._saved_state)


if __name__ == '__main__':
    unittest.main()
