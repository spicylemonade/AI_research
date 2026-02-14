"""Per-equation test-time finetuning (TTF) with LoRA for PhysMDT.

Finetunes PhysMDT on each test equation's data at inference time
using low-rank adaptation, then runs soft-masking inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping a frozen linear layer."""

    def __init__(self, linear: nn.Linear, rank: int = 32, alpha: float = 64.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Freeze original weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        original = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original + lora_out


def apply_lora(model, rank: int = 32, alpha: float = 64.0) -> List[nn.Parameter]:
    """Apply LoRA to FFN linear layers in all transformer layers.

    Wraps the first linear layer (up-projection) and last linear layer
    (down-projection) in each FFN block with LoRALinear, so that LoRA
    parameters are part of the computation graph.

    Args:
        model: PhysMDT model.
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.

    Returns:
        List of LoRA parameters for the optimizer.
    """
    lora_params = []

    for layer in model.layers:
        # Wrap FFN up-projection (layer.ffn[0])
        ffn_up = layer.ffn[0]
        lora_up = LoRALinear(ffn_up, rank=rank, alpha=alpha)
        layer.ffn[0] = lora_up
        lora_params.extend([lora_up.lora_A, lora_up.lora_B])

        # Wrap FFN down-projection (layer.ffn[3])
        ffn_down = layer.ffn[3]
        lora_down = LoRALinear(ffn_down, rank=rank, alpha=alpha)
        layer.ffn[3] = lora_down
        lora_params.extend([lora_down.lora_A, lora_down.lora_B])

    return lora_params


def remove_lora(model):
    """Remove LoRA adaptations and restore original linear layers.

    Unwraps any LoRALinear modules back to their original nn.Linear layers
    and re-enables gradient computation for all parameters.
    """
    for layer in model.layers:
        for idx in [0, 3]:
            module = layer.ffn[idx]
            if isinstance(module, LoRALinear):
                layer.ffn[idx] = module.linear
                module.linear.weight.requires_grad_(True)
                if module.linear.bias is not None:
                    module.linear.bias.requires_grad_(True)

    for param in model.parameters():
        param.requires_grad_(True)


def test_time_finetune(
    model,
    data_matrix: torch.Tensor,
    token_ids: torch.Tensor,
    n_steps: int = 128,
    lora_rank: int = 32,
    lora_alpha: float = 64.0,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    noise_aug_std: float = 0.01,
    scale_aug_range: float = 0.1,
    mask_rate_range: Tuple[float, float] = (0.3, 0.9),
    seed: int = 42,
    pos_encoding: Optional[torch.Tensor] = None,
) -> Dict:
    """Per-equation test-time finetuning with LoRA.

    Args:
        model: PhysMDT model.
        data_matrix: (1, n_points, input_dim) equation's data.
        token_ids: (1, seq_len) equation's token IDs.
        n_steps: Number of TTF steps.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        lr: Learning rate.
        weight_decay: Weight decay.
        noise_aug_std: Noise augmentation standard deviation.
        scale_aug_range: Variable scaling range.
        mask_rate_range: Masking rate range for training.
        seed: Random seed.
        pos_encoding: Optional positional encoding.

    Returns:
        info dict with TTF metrics.
    """
    torch.manual_seed(seed)
    device = data_matrix.device

    # Apply LoRA
    lora_params = apply_lora(model, rank=lora_rank, alpha=lora_alpha)

    if not lora_params:
        # Fallback: finetune output projection only
        lora_params = list(model.output_proj.parameters())

    # Freeze non-LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    for p in lora_params:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

    losses = []
    for step in range(n_steps):
        # Random augmentation per step
        aug_data = data_matrix.clone()

        # Noise augmentation
        noise_std = noise_aug_std * torch.rand(1).item()
        aug_data = aug_data + torch.randn_like(aug_data) * noise_std

        # Variable scaling
        scale = 1.0 + scale_aug_range * (torch.rand(1).item() - 0.5)
        aug_data = aug_data * scale

        # Random mask rate
        mask_rate = mask_rate_range[0] + torch.rand(1).item() * (mask_rate_range[1] - mask_rate_range[0])

        # Compute loss
        loss, info = model.compute_loss(aug_data, token_ids, mask_rate=mask_rate,
                                         pos_encoding=pos_encoding)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        losses.append(loss.item())

    # Re-enable all gradients
    for param in model.parameters():
        param.requires_grad_(True)

    ttf_info = {
        'initial_loss': losses[0] if losses else None,
        'final_loss': losses[-1] if losses else None,
        'n_steps': n_steps,
        'lora_rank': lora_rank,
        'n_lora_params': sum(p.numel() for p in lora_params),
    }

    return ttf_info


def ttf_then_infer(
    model,
    data_matrix: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    ttf_steps: int = 128,
    infer_steps: int = 50,
    lora_rank: int = 32,
    pos_encoding=None,
    **ttf_kwargs,
) -> Tuple[str, Dict]:
    """Run TTF followed by soft-masking inference.

    Args:
        model: PhysMDT model.
        data_matrix: (1, n_points, input_dim) data.
        token_ids: (1, seq_len) tokens.
        tokenizer: EquationTokenizer.
        ttf_steps: TTF adaptation steps.
        infer_steps: Soft-masking inference steps.
        lora_rank: LoRA rank.
        pos_encoding: Optional PE.

    Returns:
        pred_str: Predicted equation string.
        info: Combined TTF + inference info.
    """
    from src.model.soft_masking import soft_masking_inference

    # TTF phase
    ttf_info = test_time_finetune(
        model, data_matrix, token_ids,
        n_steps=ttf_steps, lora_rank=lora_rank,
        pos_encoding=pos_encoding, **ttf_kwargs,
    )

    # Inference phase
    pred_tokens, infer_info = soft_masking_inference(
        model, data_matrix, seq_len=token_ids.shape[1],
        num_steps=infer_steps, pos_encoding=pos_encoding,
    )

    pred_str = tokenizer.decode(pred_tokens)

    combined_info = {**ttf_info, **infer_info}
    return pred_str, combined_info
