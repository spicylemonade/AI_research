"""
Test-Time Finetuning (TTF) for In-Context Equation Adaptation.

For each test equation, applies LoRA finetuning on the observation pairs
to specialize the model before running the refinement loop.

Protocol adapted from ARC 2025:
    1. Take numerical observation pairs as a few-shot task
    2. Apply LoRA rank-32 finetuning for 64-128 steps
    3. Per-step data augmentation (noise, variable renaming, scaling)
    4. Run refinement loop post-TTF
    5. Restore base weights

References:
    - arc2025architects: test-time finetuning with LoRA
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import MASK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, MAX_SEQ_LEN, VOCAB_SIZE
from src.phys_mdt import LoRALinear


class TTFAugmenter:
    """Data augmentation for test-time finetuning."""

    def __init__(self, noise_std: float = 0.05, scale_range: Tuple = (0.8, 1.2),
                 seed: int = 42):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rng = np.random.RandomState(seed)

    def augment(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to observation data.

        Args:
            obs: (batch, n_obs, dim) observation data

        Returns:
            Augmented observations
        """
        aug = obs.clone()

        # Noise injection
        noise = torch.randn_like(aug) * self.noise_std
        aug = aug + noise

        # Random scaling
        scale = self.rng.uniform(*self.scale_range)
        aug = aug * scale

        return aug


# Fix the missing import
from typing import Tuple


class TestTimeFinetuner:
    """Test-time finetuning with LoRA for per-equation adaptation."""

    def __init__(
        self,
        lora_rank: int = 32,
        n_steps: int = 64,
        lr: float = 1e-3,
        noise_std: float = 0.05,
    ):
        self.lora_rank = lora_rank
        self.n_steps = n_steps
        self.lr = lr
        self.augmenter = TTFAugmenter(noise_std=noise_std)

    def finetune(self, model, obs: torch.Tensor, target_tokens: torch.Tensor,
                 ) -> Dict[str, float]:
        """Apply TTF to a model for a specific equation.

        Args:
            model: PhysMDT model
            obs: (1, n_obs, dim) observation data for this equation
            target_tokens: (1, seq_len) target token indices (partially known)

        Returns:
            Dict with TTF metrics (loss curve, time)
        """
        start_time = time.time()

        # Save original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Create LoRA parameters if not present
        lora_params = []
        for block in model.blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear) and module.in_features == model.d_model:
                    if not hasattr(module, '_lora_A'):
                        module._lora_A = nn.Parameter(
                            torch.randn(module.in_features, self.lora_rank,
                                        device=obs.device) * 0.01
                        )
                        module._lora_B = nn.Parameter(
                            torch.zeros(self.lora_rank, module.out_features,
                                        device=obs.device)
                        )
                    lora_params.extend([module._lora_A, module._lora_B])

        if not lora_params:
            # Fallback: finetune all parameters with small LR
            optimizer = optim.Adam(model.parameters(), lr=self.lr * 0.1)
        else:
            optimizer = optim.Adam(lora_params, lr=self.lr)

        losses = []
        model.train()

        for step in range(self.n_steps):
            # Augment observations
            aug_obs = self.augmenter.augment(obs)

            # Compute masked loss
            loss, metrics = model.compute_loss(aug_obs, target_tokens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        ttf_time = time.time() - start_time

        return {
            "losses": losses,
            "final_loss": losses[-1] if losses else 0.0,
            "ttf_time_seconds": ttf_time,
            "n_steps": self.n_steps,
        }

    def restore(self, model, original_state: Dict[str, torch.Tensor]):
        """Restore model to pre-TTF state."""
        model.load_state_dict(original_state)

    def finetune_and_predict(self, model, obs: torch.Tensor,
                              target_tokens: torch.Tensor,
                              refinement_module=None,
                              max_len: int = MAX_SEQ_LEN) -> Dict:
        """Full TTF pipeline: finetune, predict, restore.

        Args:
            model: PhysMDT model
            obs: (1, n_obs, dim) observations
            target_tokens: (1, seq_len) partial target (for finetuning)
            refinement_module: optional IterativeRefinement module
            max_len: max sequence length for generation

        Returns:
            Dict with predictions and metrics
        """
        # Save state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Finetune
        ttf_metrics = self.finetune(model, obs, target_tokens)

        # Generate
        if refinement_module:
            result = refinement_module.refine(model, obs, max_len=max_len)
            prediction = result["tokens"]
        else:
            prediction = model.generate(obs, max_len=max_len)

        # Restore
        self.restore(model, original_state)

        return {
            "prediction": prediction,
            "ttf_metrics": ttf_metrics,
        }
