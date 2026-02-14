#!/usr/bin/env python3
"""Test-time finetuning (TTF) module for PhysMDT.

Per-equation LoRA adaptation with data augmentation for improved
equation discovery at inference time.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple


class DataAugmenter:
    """Data augmentation for test-time finetuning."""

    def __init__(self, noise_std: float = 0.01, scale_range: Tuple[float, float] = (0.5, 2.0),
                 seed: int = 42):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rng = np.random.RandomState(seed)

    def augment(self, X: torch.Tensor, Y: torch.Tensor,
                n_augmented: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate augmented versions of observation data.

        Args:
            X: (n_points, n_vars) observations
            Y: (n_points,) targets
            n_augmented: number of augmented copies

        Returns:
            (X_aug, Y_aug) with original + augmented data stacked
        """
        X_list = [X]
        Y_list = [Y]

        for _ in range(n_augmented):
            aug_type = self.rng.choice(['noise', 'scale', 'subsample'])

            if aug_type == 'noise':
                noise_x = torch.randn_like(X) * self.noise_std
                noise_y = torch.randn_like(Y) * self.noise_std * Y.abs().mean().clamp(min=1e-8)
                X_list.append(X + noise_x)
                Y_list.append(Y + noise_y)

            elif aug_type == 'scale':
                scale = self.rng.uniform(*self.scale_range)
                X_list.append(X * scale)
                Y_list.append(Y * (scale ** 2))  # approximate scaling

            elif aug_type == 'subsample':
                n = X.shape[0]
                idx = torch.randperm(n)[:max(n // 2, 2)]
                X_list.append(X[idx])
                Y_list.append(Y[idx])

        return X_list, Y_list


class TestTimeFinetuner:
    """Per-equation test-time finetuning with LoRA.

    Finetunes PhysMDT on a single equation's observation data
    using LoRA adapters, then restores base weights after evaluation.
    """

    def __init__(self, model, lora_rank: int = 32, n_steps: int = 64,
                 lr: float = 1e-3, augment: bool = True, noise_std: float = 0.01):
        self.base_model = model
        self.lora_rank = lora_rank
        self.n_steps = n_steps
        self.lr = lr
        self.augment = augment
        self.augmenter = DataAugmenter(noise_std=noise_std) if augment else None
        self._saved_state = None

    def save_state(self):
        """Save model state for later restoration."""
        self._saved_state = copy.deepcopy(self.base_model.state_dict())

    def restore_state(self):
        """Restore model to saved state."""
        if self._saved_state is not None:
            self.base_model.load_state_dict(self._saved_state, strict=False)
            self._saved_state = None

    def finetune(self, X: torch.Tensor, Y: torch.Tensor,
                 target_ids: Optional[torch.Tensor] = None,
                 seq_len: int = 48) -> Dict[str, float]:
        """Finetune the model on a single equation's data.

        Args:
            X: (n_points, n_vars) observation inputs
            Y: (n_points,) observation outputs
            target_ids: (seq_len,) optional ground-truth token IDs for supervised TTF
            seq_len: sequence length for self-supervised TTF

        Returns:
            Dict with finetuning stats (loss, steps, etc.)
        """
        device = next(self.base_model.parameters()).device

        # Save base state
        self.save_state()

        # Enable LoRA
        self.base_model.enable_lora(self.lora_rank)

        # Get LoRA parameters
        lora_params = self.base_model.get_lora_parameters()
        if not lora_params:
            # Fallback: finetune all parameters
            lora_params = list(self.base_model.parameters())

        optimizer = torch.optim.Adam(lora_params, lr=self.lr)

        # Prepare data
        X_batch = X.unsqueeze(0).to(device)  # (1, n_points, n_vars)
        Y_batch = Y.unsqueeze(0).to(device)  # (1, n_points)

        # Augment
        if self.augment and self.augmenter is not None:
            X_list, Y_list = self.augmenter.augment(X, Y, n_augmented=3)
            X_batch = torch.stack([xi.to(device) for xi in X_list if xi.shape[0] == X.shape[0]])
            Y_batch = torch.stack([yi.to(device) for yi in Y_list if yi.shape[0] == Y.shape[0]])
            if X_batch.shape[0] == 0:
                X_batch = X.unsqueeze(0).to(device)
                Y_batch = Y.unsqueeze(0).to(device)

        # Finetuning loop
        self.base_model.train()
        losses = []

        for step in range(self.n_steps):
            optimizer.zero_grad()

            if target_ids is not None:
                # Supervised: use ground-truth tokens
                tgt = target_ids.unsqueeze(0).expand(X_batch.shape[0], -1).to(device)
                loss, info = self.base_model.compute_masked_diffusion_loss(
                    tgt, X_batch, Y_batch, t=None
                )
            else:
                # Self-supervised: generate predictions and refine
                with torch.no_grad():
                    pred = self.base_model.generate_single_pass(X_batch, Y_batch, seq_len=seq_len)
                loss, info = self.base_model.compute_masked_diffusion_loss(
                    pred, X_batch, Y_batch, t=0.3  # Light masking for refinement
                )

            if torch.isnan(loss):
                break

            loss.backward()
            nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            losses.append(loss.item())

        self.base_model.eval()

        return {
            'n_steps': len(losses),
            'final_loss': losses[-1] if losses else float('nan'),
            'avg_loss': sum(losses) / len(losses) if losses else float('nan'),
            'min_loss': min(losses) if losses else float('nan'),
        }

    def finetune_and_generate(self, X: torch.Tensor, Y: torch.Tensor,
                               seq_len: int = 48,
                               target_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Finetune then generate predictions.

        Returns predictions and finetuning stats, then restores base weights.
        """
        device = next(self.base_model.parameters()).device

        stats = self.finetune(X, Y, target_ids=target_ids, seq_len=seq_len)

        # Generate after finetuning
        with torch.no_grad():
            X_batch = X.unsqueeze(0).to(device)
            Y_batch = Y.unsqueeze(0).to(device)
            pred = self.base_model.generate_single_pass(X_batch, Y_batch, seq_len=seq_len)

        # Restore base weights
        self.restore_state()
        self.base_model.disable_lora()

        return pred.squeeze(0), stats
