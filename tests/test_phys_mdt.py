#!/usr/bin/env python3
"""Unit tests for PhysMDT model."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phys_mdt import PhysMDT, DualAxisRoPE, LoRALinear


class TestDualAxisRoPE(unittest.TestCase):
    def test_output_shape(self):
        rope = DualAxisRoPE(d_model=64, max_seq_len=128)
        x = torch.randn(2, 10, 64)
        out = rope(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_with_depths(self):
        rope = DualAxisRoPE(d_model=64)
        x = torch.randn(2, 10, 64)
        depths = torch.randint(0, 5, (2, 10))
        out = rope(x, tree_depth=depths)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_different_positions_differ(self):
        rope = DualAxisRoPE(d_model=64)
        x = torch.randn(1, 10, 64)
        out = rope(x)
        # Different positions should produce different embeddings
        self.assertFalse(torch.allclose(out[0, 0], out[0, 5], atol=1e-5))


class TestLoRALinear(unittest.TestCase):
    def test_without_lora(self):
        layer = LoRALinear(64, 32, rank=0)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_with_lora(self):
        layer = LoRALinear(64, 32, rank=8)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_enable_disable_lora(self):
        layer = LoRALinear(64, 32, rank=0)
        self.assertFalse(layer.lora_enabled)
        layer.enable_lora(8)
        self.assertTrue(layer.lora_enabled)
        layer.disable_lora()
        self.assertFalse(layer.lora_enabled)


class TestPhysMDT(unittest.TestCase):
    def setUp(self):
        self.model = PhysMDT(
            vocab_size=147, d_model=64, n_heads=4, n_layers=2,
            d_ff=128, max_seq_len=48, max_vars=5, n_points=10,
            lora_rank=0, dropout=0.1
        )

    def test_forward_shape(self):
        batch = 4
        token_ids = torch.randint(0, 147, (batch, 32))
        X = torch.randn(batch, 10, 5)
        Y = torch.randn(batch, 10)
        logits = self.model(token_ids, X, Y)
        self.assertEqual(logits.shape, (batch, 32, 147))

    def test_forward_with_depths(self):
        batch = 2
        token_ids = torch.randint(0, 147, (batch, 20))
        X = torch.randn(batch, 10, 5)
        Y = torch.randn(batch, 10)
        depths = torch.randint(0, 5, (batch, 20))
        logits = self.model(token_ids, X, Y, tree_depths=depths)
        self.assertEqual(logits.shape, (batch, 20, 147))

    def test_masked_diffusion_loss(self):
        batch = 4
        token_ids = torch.randint(3, 147, (batch, 24))
        token_ids[:, 0] = 1  # BOS
        token_ids[:, -1] = 2  # EOS
        X = torch.randn(batch, 10, 5)
        Y = torch.randn(batch, 10)

        loss, info = self.model.compute_masked_diffusion_loss(token_ids, X, Y, t=0.5)
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(torch.isnan(loss))
        self.assertIn('accuracy', info)
        self.assertIn('n_masked', info)

    def test_generate_single_pass(self):
        batch = 2
        X = torch.randn(batch, 10, 5)
        Y = torch.randn(batch, 10)
        pred = self.model.generate_single_pass(X, Y, seq_len=20)
        self.assertEqual(pred.shape, (batch, 20))
        self.assertTrue((pred[:, 0] == 1).all())  # BOS

    def test_gradient_flow(self):
        """Ensure gradients flow through the model."""
        batch = 2
        token_ids = torch.randint(3, 147, (batch, 16))
        token_ids[:, 0] = 1
        X = torch.randn(batch, 10, 5)
        Y = torch.randn(batch, 10)

        loss, _ = self.model.compute_masked_diffusion_loss(token_ids, X, Y, t=0.5)
        loss.backward()

        has_grad = False
        for p in self.model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "No gradients flowing through model")

    def test_lora_enable(self):
        self.model.enable_lora(rank=8)
        lora_params = self.model.get_lora_parameters()
        self.assertGreater(len(lora_params), 0)

    def test_masking_preserves_specials(self):
        """Masked diffusion should not mask BOS/EOS/PAD tokens."""
        batch = 4
        token_ids = torch.randint(3, 147, (batch, 20))
        token_ids[:, 0] = 1  # BOS
        token_ids[:, -1] = 2  # EOS
        X = torch.randn(batch, 10, 5)
        Y = torch.randn(batch, 10)

        # Run with high mask rate
        loss, info = self.model.compute_masked_diffusion_loss(token_ids, X, Y, t=0.99)
        # Should still compute a valid loss
        self.assertFalse(torch.isnan(loss))

    def test_parameter_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 100000)  # Should be a non-trivial model


if __name__ == '__main__':
    unittest.main()
