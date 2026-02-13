"""Unit tests for the physics equation dataset generator."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from data.generator import (
    EQUATION_TEMPLATES, NUM_TEMPLATES, instantiate_equation,
    generate_numerical_data, generate_dataset, split_dataset,
    get_family_distribution, get_difficulty_distribution,
)
import random


class TestTemplates:
    def test_minimum_templates(self):
        assert NUM_TEMPLATES >= 50, f"Only {NUM_TEMPLATES} templates, need >=50"

    def test_all_families_present(self):
        families = set(t[1] for t in EQUATION_TEMPLATES)
        required = {"kinematics", "dynamics", "energy", "rotational",
                    "gravitation", "oscillations", "fluid"}
        assert required.issubset(families), f"Missing families: {required - families}"

    def test_all_difficulties_present(self):
        difficulties = set(t[2] for t in EQUATION_TEMPLATES)
        assert difficulties == {"simple", "medium", "complex"}

    def test_templates_have_required_fields(self):
        for t in EQUATION_TEMPLATES:
            assert len(t) == 6, f"Template {t[0]} has {len(t)} fields, expected 6"
            name, family, diff, template, variables, coeffs = t
            assert isinstance(name, str)
            assert isinstance(family, str)
            assert diff in ("simple", "medium", "complex")
            assert isinstance(template, str)
            assert isinstance(variables, dict)
            assert isinstance(coeffs, dict)


class TestInstantiation:
    def test_instantiate_produces_valid_dict(self):
        rng = random.Random(42)
        for template in EQUATION_TEMPLATES[:10]:
            result = instantiate_equation(template, rng)
            assert "name" in result
            assert "family" in result
            assert "difficulty" in result
            assert "infix" in result
            assert "prefix" in result
            assert "coefficients" in result

    def test_different_seeds_produce_different_coefficients(self):
        t = EQUATION_TEMPLATES[0]
        r1 = instantiate_equation(t, random.Random(1))
        r2 = instantiate_equation(t, random.Random(2))
        assert r1["coefficients"] != r2["coefficients"]

    def test_reproducibility(self):
        t = EQUATION_TEMPLATES[0]
        r1 = instantiate_equation(t, random.Random(42))
        r2 = instantiate_equation(t, random.Random(42))
        assert r1["coefficients"] == r2["coefficients"]


class TestNumericalData:
    def test_numerical_data_generated_for_variable_equations(self):
        rng = random.Random(42)
        # uniform_velocity has variable t
        result = instantiate_equation(EQUATION_TEMPLATES[0], rng)
        num = generate_numerical_data(result, n_points=20, rng=rng)
        assert num is not None
        assert len(num["x"]) > 0
        assert len(num["y"]) > 0
        assert len(num["x"]) == len(num["y"])

    def test_no_numerical_data_for_constant_equations(self):
        rng = random.Random(42)
        # newton_second has no free variables
        for t in EQUATION_TEMPLATES:
            if not t[4]:  # no variables
                result = instantiate_equation(t, rng)
                num = generate_numerical_data(result, n_points=20, rng=rng)
                assert num is None
                break


class TestDatasetGeneration:
    def test_correct_size(self):
        samples = generate_dataset(n_samples=100, seed=42, include_numerical=False)
        assert len(samples) == 100

    def test_all_families_represented(self):
        samples = generate_dataset(n_samples=1000, seed=42, include_numerical=False)
        families = set(s["family"] for s in samples)
        required = {"kinematics", "dynamics", "energy", "rotational",
                    "gravitation", "oscillations", "fluid"}
        assert required.issubset(families)

    def test_all_difficulties_represented(self):
        samples = generate_dataset(n_samples=1000, seed=42, include_numerical=False)
        diffs = set(s["difficulty"] for s in samples)
        assert diffs == {"simple", "medium", "complex"}

    def test_split_ratios(self):
        samples = generate_dataset(n_samples=1000, seed=42, include_numerical=False)
        train, val, test = split_dataset(samples, seed=42)
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100

    def test_token_ids_present(self):
        samples = generate_dataset(n_samples=100, seed=42, include_numerical=False)
        has_tokens = sum(1 for s in samples if s.get("token_ids") is not None)
        assert has_tokens > 80  # Most should have valid token_ids


class TestDistributions:
    def test_family_distribution(self):
        samples = generate_dataset(n_samples=1000, seed=42, include_numerical=False)
        dist = get_family_distribution(samples)
        assert len(dist) == 7
        for family, count in dist.items():
            assert count > 50  # Each family should have reasonable representation

    def test_difficulty_distribution(self):
        samples = generate_dataset(n_samples=1000, seed=42, include_numerical=False)
        dist = get_difficulty_distribution(samples)
        assert len(dist) == 3
        for diff, count in dist.items():
            assert count > 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
