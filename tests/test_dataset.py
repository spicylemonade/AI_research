"""Tests for the dataset generation and loading pipeline."""

import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, '.')
from src.data.dataset import (
    EquationDataset, create_dataloaders, get_fsred_equations,
    _generate_data_points, _generate_procedural_equations,
    _count_variables, collate_fn,
)
from src.data.tokenizer import EquationTokenizer


@pytest.fixture
def tokenizer():
    return EquationTokenizer(max_seq_len=64)


class TestDataGeneration:
    def test_count_variables(self):
        assert _count_variables("x1*x2") == 2
        assert _count_variables("x1+x2+x3") == 3
        assert _count_variables("x1") == 1
        assert _count_variables("x1*x2*x3*x4*x5") == 5

    def test_generate_data_points_simple(self):
        X, y = _generate_data_points("x1*x2", n_points=100, seed=42)
        assert X.shape[0] == 100
        assert X.shape[1] == 2
        assert y.shape[0] == 100
        assert np.all(np.isfinite(y))

    def test_generate_data_points_noise(self):
        X1, y1 = _generate_data_points("x1*x2", n_points=100, noise_std=0.0, seed=42)
        X2, y2 = _generate_data_points("x1*x2", n_points=100, noise_std=0.05, seed=42)
        # Same X, different y due to noise
        np.testing.assert_array_equal(X1, X2)
        assert not np.allclose(y1, y2)

    def test_generate_data_points_complex(self):
        X, y = _generate_data_points("sin(x1)*cos(x2)+x3", n_points=50, seed=42)
        assert X.shape[0] == 50
        assert X.shape[1] == 3
        assert np.all(np.isfinite(y))

    def test_generate_data_points_reproducible(self):
        X1, y1 = _generate_data_points("x1**2", n_points=50, seed=42)
        X2, y2 = _generate_data_points("x1**2", n_points=50, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestProceduralGeneration:
    def test_generate_procedural_count(self):
        eqs = _generate_procedural_equations(n_equations=100, seed=42)
        assert len(eqs) == 100

    def test_generate_procedural_format(self):
        eqs = _generate_procedural_equations(n_equations=10, seed=42)
        for expr_str, desc in eqs:
            assert isinstance(expr_str, str)
            assert isinstance(desc, str)
            assert len(expr_str) > 0

    def test_large_procedural(self):
        eqs = _generate_procedural_equations(n_equations=50000, seed=42)
        assert len(eqs) == 50000


class TestFSReD:
    def test_fsred_count(self):
        eqs = get_fsred_equations()
        assert len(eqs) == 120

    def test_fsred_format(self):
        eqs = get_fsred_equations()
        for expr_str, desc in eqs:
            assert isinstance(expr_str, str)
            assert 'fsred_' in desc


class TestEquationDataset:
    def test_dataset_length(self, tokenizer):
        eqs = [("x1*x2", "test1"), ("x1+x2", "test2")]
        ds = EquationDataset(eqs, tokenizer, n_data_points=32, seed=42)
        assert len(ds) == 2

    def test_dataset_item_shape(self, tokenizer):
        eqs = [("x1*x2", "test1")]
        ds = EquationDataset(eqs, tokenizer, n_data_points=32, seed=42)
        item = ds[0]
        assert item['data_matrix'].shape == (32, 10)
        assert item['token_ids'].shape == (64,)
        assert isinstance(item['n_vars'], int)
        assert item['difficulty'] in ('easy', 'medium', 'hard')

    def test_dataset_item_values(self, tokenizer):
        eqs = [("x1*x2", "test1")]
        ds = EquationDataset(eqs, tokenizer, n_data_points=32, seed=42)
        item = ds[0]
        assert torch.all(torch.isfinite(item['data_matrix']))
        assert item['token_ids'][0] == 1  # BOS


class TestDataLoaders:
    def test_create_dataloaders(self):
        train_loader, val_loader, test_loader = create_dataloaders(
            include_fsred=True,
            include_procedural=False,
            n_data_points=32,
            batch_size=4,
            seed=42,
        )
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(test_loader.dataset) > 0
        total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        assert total == 120

    def test_dataloader_batch(self):
        train_loader, _, _ = create_dataloaders(
            include_fsred=True,
            include_procedural=False,
            n_data_points=32,
            batch_size=4,
            seed=42,
        )
        batch = next(iter(train_loader))
        assert batch['data_matrix'].shape[0] == 4
        assert batch['data_matrix'].shape[1] == 32
        assert batch['data_matrix'].shape[2] == 10
        assert batch['token_ids'].shape[0] == 4
        assert batch['token_ids'].shape[1] == 64

    def test_split_ratios(self):
        train_loader, val_loader, test_loader = create_dataloaders(
            include_fsred=True,
            include_procedural=False,
            n_data_points=32,
            batch_size=4,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_test = len(test_loader.dataset)
        assert n_train == 96  # 80% of 120
        assert n_val == 12   # 10% of 120
        assert n_test == 12  # 10% of 120


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
