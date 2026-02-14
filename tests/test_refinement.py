#!/usr/bin/env python3
"""Unit tests for iterative soft-mask refinement module."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phys_mdt import PhysMDT
from src.refinement import (
    RefinementConfig, CandidateTracker, SoftMaskRefinement,
    PAD_ID, BOS_ID, EOS_ID, MASK_ID, _ids_to_key,
)


def _make_model(**kwargs):
    defaults = dict(
        vocab_size=147, d_model=64, n_heads=4, n_layers=2,
        d_ff=128, max_seq_len=48, max_vars=5, n_points=10,
        lora_rank=0, dropout=0.1,
    )
    defaults.update(kwargs)
    return PhysMDT(**defaults)


class TestCandidateTracker(unittest.TestCase):
    def test_record_and_best(self):
        tracker = CandidateTracker(top_k=2)
        tracker.reset(2)
        ids = torch.tensor([[1, 10, 11, 2], [1, 20, 21, 2]])
        tracker.record(ids)
        tracker.record(ids)
        best = tracker.best(torch.device('cpu'))
        self.assertEqual(best.shape, (2, 4))
        self.assertTrue(torch.equal(best, ids))

    def test_empty_tracker(self):
        tracker = CandidateTracker(top_k=2)
        tracker.reset(1)
        best = tracker.best(torch.device('cpu'))
        # No candidates -> should return MASK tokens
        self.assertEqual(best.shape, (1, 0))  # seq_len=0 since no candidates

    def test_top_candidates(self):
        tracker = CandidateTracker(top_k=2)
        tracker.reset(1)
        a = torch.tensor([[1, 10, 11, 2]])
        b = torch.tensor([[1, 20, 21, 2]])
        for _ in range(3):
            tracker.record(a)
        for _ in range(5):
            tracker.record(b)
        tops = tracker.top_candidates(torch.device('cpu'))
        self.assertEqual(len(tops), 1)
        self.assertEqual(len(tops[0]), 2)
        # b should be most visited
        self.assertTrue(torch.equal(tops[0][0][0], b.squeeze(0)))
        self.assertEqual(tops[0][0][1], 5)

    def test_ids_to_key(self):
        ids = torch.tensor([1, 2, 3])
        key = _ids_to_key(ids)
        self.assertEqual(key, (1, 2, 3))


class TestRefinementConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = RefinementConfig()
        self.assertEqual(cfg.total_steps, 50)
        self.assertTrue(cfg.cold_restart)
        self.assertTrue(cfg.convergence_detection)
        self.assertTrue(cfg.soft_masking)
        self.assertTrue(cfg.candidate_tracking)

    def test_custom_config(self):
        cfg = RefinementConfig(total_steps=10, cold_restart=False)
        self.assertEqual(cfg.total_steps, 10)
        self.assertFalse(cfg.cold_restart)


class TestSoftMaskRefinement(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.model.eval()

    def test_refine_output_shape(self):
        cfg = RefinementConfig(total_steps=4, cold_restart=False,
                                convergence_detection=False,
                                soft_masking=False, candidate_tracking=True)
        refiner = SoftMaskRefinement(self.model, cfg)
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        pred = refiner.refine(X, Y, seq_len=16)
        self.assertEqual(pred.shape, (2, 16))
        self.assertTrue((pred[:, 0] == BOS_ID).all())

    def test_refine_with_cold_restart(self):
        cfg = RefinementConfig(total_steps=4, cold_restart=True,
                                convergence_detection=False,
                                soft_masking=False, candidate_tracking=True)
        refiner = SoftMaskRefinement(self.model, cfg)
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        pred = refiner.refine(X, Y, seq_len=16)
        self.assertEqual(pred.shape, (2, 16))

    def test_refine_with_soft_masking(self):
        cfg = RefinementConfig(total_steps=4, cold_restart=False,
                                convergence_detection=False,
                                soft_masking=True, candidate_tracking=True)
        refiner = SoftMaskRefinement(self.model, cfg)
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        pred = refiner.refine(X, Y, seq_len=16)
        self.assertEqual(pred.shape, (2, 16))

    def test_refine_with_convergence(self):
        cfg = RefinementConfig(total_steps=10, cold_restart=False,
                                convergence_detection=True,
                                confidence_threshold=0.0,  # very low = converge fast
                                convergence_patience=1,
                                soft_masking=False, candidate_tracking=False)
        refiner = SoftMaskRefinement(self.model, cfg)
        X = torch.randn(1, 10, 5)
        Y = torch.randn(1, 10)
        pred = refiner.refine(X, Y, seq_len=8)
        self.assertEqual(pred.shape, (1, 8))

    def test_refine_without_candidate_tracking(self):
        cfg = RefinementConfig(total_steps=4, cold_restart=False,
                                convergence_detection=False,
                                soft_masking=False, candidate_tracking=False)
        refiner = SoftMaskRefinement(self.model, cfg)
        X = torch.randn(2, 10, 5)
        Y = torch.randn(2, 10)
        pred = refiner.refine(X, Y, seq_len=16)
        self.assertEqual(pred.shape, (2, 16))
        self.assertTrue((pred[:, 0] == BOS_ID).all())

    def test_refine_gumbel_noise(self):
        cfg = RefinementConfig(total_steps=2, cold_restart=False,
                                convergence_detection=False,
                                soft_masking=False, candidate_tracking=False,
                                gumbel_noise=0.5)
        refiner = SoftMaskRefinement(self.model, cfg)
        X = torch.randn(1, 10, 5)
        Y = torch.randn(1, 10)
        pred = refiner.refine(X, Y, seq_len=8)
        self.assertEqual(pred.shape, (1, 8))


if __name__ == '__main__':
    unittest.main()
