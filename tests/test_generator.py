#!/usr/bin/env python3
"""Unit tests for the physics equation dataset generator."""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import PhysicsDatasetGenerator, build_templates, format_constant, EquationTemplate


class TestTemplates(unittest.TestCase):
    """Tests for equation templates."""

    def setUp(self):
        self.templates = build_templates()

    def test_at_least_60_templates(self):
        self.assertGreaterEqual(len(self.templates), 60)

    def test_seven_families(self):
        families = set(t.family for t in self.templates)
        expected = {'kinematics', 'dynamics', 'energy', 'rotational',
                    'gravitation', 'oscillations', 'fluid_statics'}
        self.assertEqual(families, expected)

    def test_each_family_has_templates(self):
        family_counts = {}
        for t in self.templates:
            family_counts[t.family] = family_counts.get(t.family, 0) + 1
        for fam, count in family_counts.items():
            self.assertGreaterEqual(count, 5, f"Family {fam} has only {count} templates")

    def test_three_difficulty_levels(self):
        difficulties = set(t.difficulty for t in self.templates)
        self.assertEqual(difficulties, {'simple', 'medium', 'complex'})

    def test_each_difficulty_has_templates(self):
        diff_counts = {}
        for t in self.templates:
            diff_counts[t.difficulty] = diff_counts.get(t.difficulty, 0) + 1
        for diff in ['simple', 'medium', 'complex']:
            self.assertGreaterEqual(diff_counts.get(diff, 0), 5,
                                    f"Difficulty {diff} has too few templates")

    def test_template_has_required_fields(self):
        for t in self.templates:
            self.assertTrue(t.name, f"Template missing name")
            self.assertTrue(t.family, f"Template {t.name} missing family")
            self.assertTrue(t.difficulty, f"Template {t.name} missing difficulty")
            self.assertTrue(t.prefix_notation, f"Template {t.name} missing prefix_notation")
            self.assertTrue(t.eval_func, f"Template {t.name} missing eval_func")
            self.assertIsInstance(t.variables, list)

    def test_unique_template_names(self):
        names = [t.name for t in self.templates]
        self.assertEqual(len(names), len(set(names)), "Duplicate template names found")


class TestFormatConstant(unittest.TestCase):
    """Tests for constant formatting."""

    def test_integer_constants(self):
        for i in range(10):
            self.assertEqual(format_constant(float(i)), f"INT_{i}")

    def test_pi(self):
        self.assertEqual(format_constant(math.pi), "pi")

    def test_g_accel(self):
        self.assertEqual(format_constant(9.81), "g_accel")

    def test_negative_constant(self):
        result = format_constant(-3.14)
        self.assertIn("C_-", result)
        self.assertIn("CONST_START", result)
        self.assertIn("CONST_END", result)


class TestGenerator(unittest.TestCase):
    """Tests for the dataset generator."""

    def setUp(self):
        self.gen = PhysicsDatasetGenerator(seed=42)

    def test_generate_sample_structure(self):
        sample = self.gen.generate_sample()
        self.assertIn('template_name', sample)
        self.assertIn('family', sample)
        self.assertIn('difficulty', sample)
        self.assertIn('prefix_notation', sample)
        self.assertIn('observations_x', sample)
        self.assertIn('observations_y', sample)
        self.assertIn('n_points', sample)

    def test_observation_shape(self):
        sample = self.gen.generate_sample(n_points=100)
        X = sample['observations_x']
        Y = sample['observations_y']
        self.assertEqual(len(X), 100)
        self.assertEqual(len(Y), 100)

    def test_reproducibility(self):
        gen1 = PhysicsDatasetGenerator(seed=42)
        gen2 = PhysicsDatasetGenerator(seed=42)
        s1 = gen1.generate_sample()
        s2 = gen2.generate_sample()
        self.assertEqual(s1['template_name'], s2['template_name'])
        self.assertEqual(s1['prefix_notation'], s2['prefix_notation'])

    def test_generate_dataset_size(self):
        dataset = self.gen.generate_dataset(100)
        self.assertEqual(len(dataset), 100)

    def test_filter_by_difficulty(self):
        dataset = self.gen.generate_dataset(50, difficulty='simple')
        for s in dataset:
            self.assertEqual(s['difficulty'], 'simple')

    def test_filter_by_family(self):
        dataset = self.gen.generate_dataset(50, family='kinematics')
        for s in dataset:
            self.assertEqual(s['family'], 'kinematics')

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            self.gen.generate_and_save(10, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 10)
        finally:
            os.unlink(path)

    def test_observations_finite(self):
        """Y values should be finite for valid equations."""
        for _ in range(20):
            sample = self.gen.generate_sample()
            Y = sample['observations_y']
            finite_count = sum(1 for y in Y if math.isfinite(y))
            self.assertGreater(finite_count, len(Y) * 0.5,
                               f"Too many non-finite Y values for {sample['template_name']}")

    def test_coefficient_ranges(self):
        """Coefficients should be within defined ranges."""
        for template in self.gen.templates:
            coeffs = template.generate_coefficients(self.gen.rng)
            for name, val in coeffs.items():
                lo, hi = template.coeff_ranges[name]
                self.assertGreaterEqual(val, lo, f"{template.name}: {name}={val} < {lo}")
                self.assertLessEqual(val, hi, f"{template.name}: {name}={val} > {hi}")


if __name__ == '__main__':
    unittest.main()
