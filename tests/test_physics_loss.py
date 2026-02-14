#!/usr/bin/env python3
"""Unit tests for physics-informed loss module (>=6 required by item_016)."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics_loss import (
    check_dimensional_consistency, get_units, _compute_units,
    DimensionalConsistencyLoss, ConservationRegularizer, SymmetryLoss,
    PhysicsLoss, UNIT_TABLE, OPERATOR_ARITIES,
)


class TestGetUnits(unittest.TestCase):
    def test_known_variable(self):
        self.assertEqual(get_units('F'), (1, 1, -2))
        self.assertEqual(get_units('m'), (1, 0, 0))
        self.assertEqual(get_units('v'), (0, 1, -1))

    def test_dimensionless(self):
        self.assertEqual(get_units('theta'), (0, 0, 0))
        self.assertEqual(get_units('pi'), (0, 0, 0))

    def test_integer_token(self):
        self.assertEqual(get_units('INT_2'), (0, 0, 0))

    def test_constant_token(self):
        self.assertEqual(get_units('CONST_START'), (0, 0, 0))

    def test_operator_returns_none(self):
        self.assertIsNone(get_units('add'))
        self.assertIsNone(get_units('mul'))


class TestDimensionalConsistency(unittest.TestCase):
    def test_consistent_f_ma(self):
        # F = m * a -> mul m a should be consistent (1,1,-2) = (1,0,0) * (0,1,-2)
        penalty = check_dimensional_consistency("mul m a")
        self.assertEqual(penalty, 0.0)

    def test_consistent_kinetic_energy(self):
        # KE = 0.5 * m * v^2 -> mul m mul v v
        penalty = check_dimensional_consistency("mul m mul v v")
        self.assertEqual(penalty, 0.0)

    def test_inconsistent_add_m_v(self):
        # m + v has incompatible units
        penalty = check_dimensional_consistency("add m v")
        self.assertGreater(penalty, 0.0)

    def test_trig_dimensionless_arg(self):
        # sin(theta) should be consistent
        penalty = check_dimensional_consistency("sin theta")
        self.assertEqual(penalty, 0.0)

    def test_trig_dimensional_arg(self):
        # sin(F) should be inconsistent (F has dimensions)
        penalty = check_dimensional_consistency("sin F")
        self.assertGreater(penalty, 0.0)

    def test_empty_string(self):
        penalty = check_dimensional_consistency("")
        # Should return moderate or high penalty
        self.assertGreater(penalty, 0.0)


class TestDimensionalConsistencyLoss(unittest.TestCase):
    def test_enabled(self):
        loss = DimensionalConsistencyLoss(weight=1.0, enabled=True)
        result = loss(["mul m a", "add m v"])
        self.assertIsInstance(result.item(), float)
        self.assertGreater(result.item(), 0.0)  # second equation is inconsistent

    def test_disabled(self):
        loss = DimensionalConsistencyLoss(weight=1.0, enabled=False)
        result = loss(["add m v"])
        self.assertEqual(result.item(), 0.0)

    def test_empty_list(self):
        loss = DimensionalConsistencyLoss(weight=1.0, enabled=True)
        result = loss([])
        self.assertEqual(result.item(), 0.0)


class TestConservationRegularizer(unittest.TestCase):
    def test_constant_predictions(self):
        loss = ConservationRegularizer(weight=1.0, enabled=True)
        # Constant predictions should have zero variance
        pred = torch.ones(4, 20) * 5.0
        result = loss(pred)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_varying_predictions(self):
        loss = ConservationRegularizer(weight=1.0, enabled=True)
        pred = torch.randn(4, 20)
        result = loss(pred)
        self.assertGreater(result.item(), 0.0)

    def test_disabled(self):
        loss = ConservationRegularizer(weight=1.0, enabled=False)
        pred = torch.randn(4, 20)
        result = loss(pred)
        self.assertEqual(result.item(), 0.0)


class TestSymmetryLoss(unittest.TestCase):
    def test_even_function(self):
        loss = SymmetryLoss(weight=1.0, enabled=True)
        X = torch.randn(2, 10, 3)
        Y = torch.randn(2, 10)
        # Even function: f(x) = x^2
        model_fn = lambda x: (x ** 2).sum(dim=-1)
        result = loss(model_fn, X, Y)
        self.assertIsInstance(result.item(), float)

    def test_disabled(self):
        loss = SymmetryLoss(weight=1.0, enabled=False)
        X = torch.randn(2, 10, 3)
        Y = torch.randn(2, 10)
        model_fn = lambda x: x.sum(dim=-1)
        result = loss(model_fn, X, Y)
        self.assertEqual(result.item(), 0.0)


class TestPhysicsLoss(unittest.TestCase):
    def test_combined_loss(self):
        ploss = PhysicsLoss(dim_weight=0.1, conserv_weight=0.1, sym_weight=0.05)
        total, components = ploss(
            pred_prefixes=["mul m a", "add m v"],
            pred_values=torch.randn(2, 10),
        )
        self.assertIn('dim_consistency', components)
        self.assertIn('conservation', components)
        self.assertIn('total_physics_loss', components)

    def test_selective_enable(self):
        ploss = PhysicsLoss(enable_dim=True, enable_conserv=False, enable_sym=False)
        total, components = ploss(
            pred_prefixes=["mul m a"],
            pred_values=torch.randn(2, 10),
        )
        self.assertIn('dim_consistency', components)
        self.assertEqual(components.get('conservation', 0.0), 0.0)

    def test_all_disabled(self):
        ploss = PhysicsLoss(enable_dim=False, enable_conserv=False, enable_sym=False)
        total, components = ploss(
            pred_prefixes=["mul m a"],
            pred_values=torch.randn(2, 10),
        )
        self.assertAlmostEqual(total.item(), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
