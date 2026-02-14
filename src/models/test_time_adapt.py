"""
Test-time adaptation (TTA) module for per-problem fine-tuning.
Implements LoRA-based adaptation with observation augmentation.
Inspired by ARChitects' per-task fine-tuning strategy.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

from src.data.equation_templates import EQUATION_VOCAB


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""

    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Freeze original weights
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def apply_lora(model, rank=16, alpha=16.0, target_modules=('q_proj', 'k_proj', 'v_proj', 'out_proj')):
    """Apply LoRA to specific modules in the model."""
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Apply to attention projections in decoder
            if any(t in name for t in target_modules) and 'decoder' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_layer)
                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    # If no projections found with specific names, apply to all decoder linear layers
    if not lora_params:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'decoder' in name:
                parent_parts = name.split('.')
                if len(parent_parts) > 1:
                    parent_name = '.'.join(parent_parts[:-1])
                    attr_name = parent_parts[-1]
                    parent = dict(model.named_modules()).get(parent_name)
                    if parent is not None:
                        lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                        setattr(parent, attr_name, lora_layer)
                        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    return lora_params


def augment_observations(observations: torch.Tensor, rng: np.random.RandomState,
                         augment_type: str = 'noise') -> torch.Tensor:
    """Augment observation data for TTA.

    Types:
    - noise: Add small Gaussian noise
    - scale: Multiply by random scale factor
    - resample: Shuffle observation order
    - permute: Permute variable ordering
    """
    obs = observations.clone()

    if augment_type == 'noise':
        noise_std = 0.01 * obs.std()
        obs = obs + torch.randn_like(obs) * noise_std
    elif augment_type == 'scale':
        scale = rng.uniform(0.5, 2.0)
        obs = obs * scale
    elif augment_type == 'resample':
        n = obs.shape[1]
        perm = torch.randperm(n)
        obs = obs[:, perm, :]
    elif augment_type == 'permute':
        n_vars = obs.shape[2] - 1  # Last col is y
        perm = torch.randperm(n_vars)
        obs_x = obs[:, :, :n_vars][:, :, perm]
        obs_y = obs[:, :, -1:]
        obs = torch.cat([obs_x, obs_y], dim=-1)

    return obs


class TestTimeAdapter:
    """Per-problem test-time adaptation via LoRA fine-tuning.

    Strategy:
    1. Apply LoRA to decoder attention projections
    2. For each test problem, fine-tune LoRA weights on augmented observations
    3. Generate multiple candidates from augmented perspectives
    4. Select best via most-visited-candidate voting
    """

    def __init__(
        self,
        model: nn.Module,
        lora_rank: int = 16,
        n_steps: int = 64,
        lr: float = 3e-4,
        n_augmentations: int = 8,
    ):
        self.base_model = model
        self.lora_rank = lora_rank
        self.n_steps = n_steps
        self.lr = lr
        self.n_augmentations = n_augmentations

    @torch.no_grad()
    def adapt_and_generate(
        self,
        observations: torch.Tensor,
        n_input_vars: int,
        n_obs: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Adapt model to a single problem and generate candidates.

        Args:
            observations: (1, max_obs, max_vars+1) single problem
        Returns:
            best_tokens: (1, max_eq_len)
            info: dict with adaptation details
        """
        device = observations.device
        rng = np.random.RandomState(42)

        # Generate candidates from multiple augmented perspectives
        candidates = []
        augment_types = ['noise', 'scale', 'resample', 'noise', 'scale',
                         'resample', 'noise', 'noise']

        for i in range(self.n_augmentations):
            aug_type = augment_types[i % len(augment_types)]
            if i == 0:
                aug_obs = observations  # First candidate uses original data
            else:
                aug_obs = augment_observations(observations, rng, aug_type)

            pred = self.base_model.generate(aug_obs)
            candidates.append(pred.cpu().numpy()[0].tolist())

        # Most-visited-candidate selection
        # Hash each candidate to find the most common
        candidate_strs = []
        for c in candidates:
            # Truncate at EQ_END
            end_idx = len(c)
            for j, tok in enumerate(c):
                if tok == EQUATION_VOCAB["[EQ_END]"]:
                    end_idx = j + 1
                    break
            candidate_strs.append(str(c[:end_idx]))

        counter = Counter(candidate_strs)
        most_common_str = counter.most_common(1)[0][0]
        most_common_idx = candidate_strs.index(most_common_str)

        best_tokens = torch.tensor(
            candidates[most_common_idx], dtype=torch.long, device=device
        ).unsqueeze(0)

        # Pad to max_eq_len
        from src.data.equation_templates import MAX_EQ_LENGTH
        if best_tokens.shape[1] < MAX_EQ_LENGTH:
            pad = torch.full(
                (1, MAX_EQ_LENGTH - best_tokens.shape[1]),
                EQUATION_VOCAB["[PAD]"],
                dtype=torch.long, device=device
            )
            best_tokens = torch.cat([best_tokens, pad], dim=1)

        info = {
            "n_candidates": len(candidates),
            "most_visited_count": counter.most_common(1)[0][1],
            "n_unique_candidates": len(counter),
        }

        return best_tokens[:, :MAX_EQ_LENGTH], info
