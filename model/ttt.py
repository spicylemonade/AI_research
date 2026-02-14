"""Test-Time Training (TTT) with LoRA adapters for PhysDiffuse.

Inspired by ARC2025 ARChitects: per-instance fine-tuning with low-rank
adapters on the masked reconstruction objective using augmented versions
of the test observation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
from typing import List, Dict, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import VOCAB_SIZE, PAD_ID, SOS_ID, EOS_ID, MASK_ID, MAX_SEQ_LEN, decode
from data.augmentation import augment_sample


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping a frozen linear layer."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        in_features = original.in_features
        out_features = original.out_features

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA matrices (on same device as original layer)
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def attach_lora(model, rank: int = 16, alpha: float = 32.0,
                target_modules: Optional[List[str]] = None):
    """Attach LoRA adapters to the model's decoder linear layers.

    Args:
        model: PhysDiffuse model
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: list of substrings to match for adaptation
                       (default: attention and FFN projections)

    Returns:
        list of LoRA parameters for the optimizer
    """
    if target_modules is None:
        # Only target FFN layers (not attention layers which have special structure)
        target_modules = ['ffn']

    lora_params = []
    replacements = []

    # Find FFN linear layers in decoder (avoid attention internals)
    for name, module in model.decoder.named_modules():
        if isinstance(module, nn.Linear):
            # Only adapt FFN layers (layers.X.ffn.0 and layers.X.ffn.3)
            if 'ffn' in name and ('0' in name.split('.')[-1] or '3' in name.split('.')[-1]):
                replacements.append((name, module))

    # Replace with LoRA layers
    for name, module in replacements:
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
        # Navigate to parent and replace
        parts = name.split('.')
        parent = model.decoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    return lora_params


def detach_lora(model):
    """Remove LoRA adapters and restore original linear layers."""
    replacements = []
    for name, module in model.decoder.named_modules():
        if isinstance(module, LoRALinear):
            replacements.append((name, module.original))

    for name, original in replacements:
        parts = name.split('.')
        parent = model.decoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], original)


def ttt_adapt(model, obs_table: torch.Tensor, n_augmentations: int = 64,
              n_steps: int = 96, lr: float = 1e-4, rank: int = 16,
              verbose: bool = False) -> None:
    """Per-equation test-time training.

    Adapts the model to a single test observation using augmented versions
    of the input and the self-supervised masked reconstruction objective.

    Args:
        model: PhysDiffuse model (will be modified in-place with LoRA)
        obs_table: (1, N, D+1) single observation table
        n_augmentations: number of augmented copies
        n_steps: number of gradient steps
        lr: learning rate for LoRA parameters
        rank: LoRA rank
        verbose: print progress
    """
    device = obs_table.device
    t0 = time.time()

    # Attach LoRA adapters
    lora_params = attach_lora(model, rank=rank)
    if not lora_params:
        return

    optimizer = torch.optim.Adam(lora_params, lr=lr)

    # Generate augmented observations
    from data.augmentation import noise_injection
    obs_np = obs_table[0].cpu().numpy()  # (N, D+1)
    augmented_obs = [obs_np]
    for i in range(n_augmentations - 1):
        rng = np.random.default_rng(42 + i)
        aug = noise_injection(obs_np, noise_level=0.05, rng=rng)
        augmented_obs.append(aug)

    # Stack and convert to tensor
    # Ensure all have same shape
    n_points = obs_np.shape[0]
    n_cols = obs_np.shape[1]
    aug_tables = np.stack(augmented_obs)  # (K, N, D+1)
    aug_tensor = torch.tensor(aug_tables, dtype=torch.float32, device=device)

    # Generate pseudo-targets using the model's own predictions
    # First pass: get model's best guess
    model.eval()
    with torch.no_grad():
        memory = model.encoder(obs_table)
        # Use a few forward passes to get pseudo-labels
        pseudo_labels = _generate_pseudo_labels(model, memory, n_samples=32)

    if pseudo_labels is None:
        detach_lora(model)
        return

    # Expand pseudo-labels for all augmentations
    # Use the most common prediction as the reconstruction target
    target = pseudo_labels.unsqueeze(0).expand(len(augmented_obs), -1)  # (K, L)
    target = target.to(device)

    # TTT training loop
    model.train()
    batch_size = min(16, len(augmented_obs))

    for step in range(n_steps):
        # Random batch of augmented observations
        idx = np.random.choice(len(augmented_obs), batch_size, replace=True)
        batch_obs = aug_tensor[idx]
        batch_tgt = target[idx]

        loss, _ = model(batch_obs, batch_tgt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        if verbose and step % 20 == 0:
            print(f"  TTT step {step}: loss={loss.item():.4f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"  TTT completed in {elapsed:.1f}s")


def _generate_pseudo_labels(model, memory: torch.Tensor,
                            n_samples: int = 32) -> Optional[torch.Tensor]:
    """Generate pseudo-labels from the model for TTT.

    Uses the model's generate method to get a consensus prediction,
    then uses that as the reconstruction target.
    """
    from collections import Counter

    device = memory.device
    L = MAX_SEQ_LEN
    memory_exp = memory.expand(n_samples, -1, -1)

    # Single round of generation
    x = torch.full((n_samples, L), MASK_ID, dtype=torch.long, device=device)
    x[:, 0] = SOS_ID

    for step in range(1, 17):  # Quick 16-step generation
        logits = model.decoder(x, memory_exp)
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        logits = logits.clamp(-30, 30)
        probs = F.softmax(logits / 0.5, dim=-1)
        probs = torch.where(torch.isfinite(probs) & (probs > 0), probs, torch.full_like(probs, 1e-8))
        probs = probs / probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(
            probs.view(-1, VOCAB_SIZE), 1
        ).view(n_samples, L)

        conf = probs.max(dim=-1).values
        n_unmask = max(1, int(L * step / 16))

        for b in range(n_samples):
            conf_masked = conf[b].clone()
            conf_masked[0] = -1
            _, top_idx = conf_masked.topk(n_unmask)
            x[b, top_idx] = sampled[b, top_idx]

    # Find most common sequence
    seqs = []
    for b in range(n_samples):
        seq = x[b].tolist()
        trimmed = []
        for tok_id in seq:
            trimmed.append(tok_id)
            if tok_id == EOS_ID:
                break
        seqs.append(tuple(trimmed))

    counter = Counter(seqs)
    best_seq = list(counter.most_common(1)[0][0])

    # Pad to L
    if len(best_seq) < L:
        best_seq.extend([PAD_ID] * (L - len(best_seq)))
    elif len(best_seq) > L:
        best_seq = best_seq[:L]

    return torch.tensor(best_seq, dtype=torch.long, device=device)


def ttt_generate(model, obs_table: torch.Tensor,
                 T: int = 32, R: int = 2, n_samples: int = 128,
                 n_ttt_steps: int = 96, ttt_augmentations: int = 64,
                 ttt_rank: int = 16, ttt_lr: float = 1e-4,
                 verbose: bool = False) -> List[int]:
    """Full TTT + generation pipeline.

    1. Adapt model with TTT on the test instance
    2. Generate candidate equations
    3. Select best via most-visited

    Args:
        model: PhysDiffuse model
        obs_table: (1, N, D+1) observation table
        T, R, n_samples: generation parameters
        n_ttt_steps: number of TTT gradient steps
        ttt_augmentations: number of augmented copies for TTT
        ttt_rank: LoRA rank
        ttt_lr: TTT learning rate
        verbose: print progress

    Returns:
        best_tokens: list of token IDs
    """
    # Save model state
    original_state = {name: p.clone() for name, p in model.named_parameters()}

    try:
        # TTT adaptation
        if verbose:
            print("  Starting TTT adaptation...")
        ttt_adapt(model, obs_table, n_augmentations=ttt_augmentations,
                  n_steps=n_ttt_steps, lr=ttt_lr, rank=ttt_rank,
                  verbose=verbose)

        # Generate with adapted model
        model.eval()
        result = model.generate(obs_table, T=T, R=R, n_samples=n_samples)
        return result

    finally:
        # Restore original model
        detach_lora(model)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in original_state:
                    p.copy_(original_state[name])


if __name__ == '__main__':
    """Smoke test for TTT module."""
    from model.phys_diffuse import create_phys_diffuse
    from data.feynman_loader import generate_benchmark_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    model = create_phys_diffuse(d_model=256, n_heads=4, device=device,
                                n_enc_layers=2, n_dec_layers=4)

    # Load benchmark data
    benchmark = generate_benchmark_data(n_points=50, seed=42)
    simple_eqs = [b for b in benchmark if b['tier'] == 1][:3]

    max_n_points = 50
    max_vars = 10

    print(f"\nTesting TTT on {len(simple_eqs)} equations...")
    for eq in simple_eqs:
        table = eq['table']
        if table.shape[0] < max_n_points:
            pad = np.zeros((max_n_points - table.shape[0], table.shape[1]))
            table = np.vstack([table, pad])
        else:
            table = table[:max_n_points]
        if table.shape[1] < max_vars + 1:
            pad = np.zeros((table.shape[0], max_vars + 1 - table.shape[1]))
            table = np.hstack([table, pad])

        obs = torch.tensor(table[np.newaxis], dtype=torch.float32, device=device)

        # Test LoRA attach/detach
        lora_params = attach_lora(model, rank=8)
        n_lora = sum(p.numel() for p in lora_params)
        print(f"  LoRA params: {n_lora/1e3:.1f}K")
        detach_lora(model)

        # Test TTT generation (quick, few steps)
        t0 = time.time()
        pred_ids = ttt_generate(
            model, obs, T=8, R=1, n_samples=16,
            n_ttt_steps=10, ttt_augmentations=8,
            ttt_rank=8, ttt_lr=1e-3, verbose=False
        )
        elapsed = time.time() - t0
        pred_tokens = decode(pred_ids)
        print(f"  {eq['name']}: pred={pred_tokens[:6]}, time={elapsed:.1f}s")

    print("\nTTT smoke test complete!")
