"""
Physics-informed positional encodings and dimensional analysis attention constraints.
Adapted from ARChitects' 2D RoPE concept with multi-scale frequency bands.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleRoPE(nn.Module):
    """Multi-scale Rotary Position Encoding.

    Uses separate frequency bands for:
    - Band 1: Sequential position index (data point ordering)
    - Band 2: Value magnitude (log-scale encoding)

    Inspired by ARChitects' Golden Gate RoPE which encodes positions
    along multiple 2D directions for spatial reasoning.
    """

    def __init__(self, d_model, max_seq_len=128, base_freq=10000.0):
        super().__init__()
        self.d_model = d_model
        self.half_d = d_model // 2
        self.enabled = True

        # Band 1: position-based frequencies
        pos_freqs = 1.0 / (base_freq ** (torch.arange(0, self.half_d, 2).float() / self.half_d))
        self.register_buffer('pos_freqs', pos_freqs)

        # Band 2: value-magnitude frequencies (different base)
        val_freqs = 1.0 / (1000.0 ** (torch.arange(0, self.half_d, 2).float() / self.half_d))
        self.register_buffer('val_freqs', val_freqs)

    def forward(self, x, positions=None, values=None):
        """Apply multi-scale RoPE.

        Args:
            x: (B, T, d_model) input embeddings
            positions: (B, T) position indices (optional)
            values: (B, T) representative values for magnitude encoding (optional)
        """
        if not self.enabled:
            return x

        B, T, D = x.shape
        device = x.device

        if positions is None:
            positions = torch.arange(T, device=device).float().unsqueeze(0).expand(B, T)

        # Band 1: position-based rotation
        pos_angles = positions.unsqueeze(-1) * self.pos_freqs.unsqueeze(0).unsqueeze(0)
        pos_sin = torch.sin(pos_angles)  # (B, T, half_d/2)
        pos_cos = torch.cos(pos_angles)

        # Apply to first half of dimensions
        x1 = x[:, :, :self.half_d]
        x1_pairs = x1.reshape(B, T, -1, 2)  # (B, T, half_d/2, 2)
        rotated1 = torch.stack([
            x1_pairs[..., 0] * pos_cos - x1_pairs[..., 1] * pos_sin,
            x1_pairs[..., 0] * pos_sin + x1_pairs[..., 1] * pos_cos,
        ], dim=-1).reshape(B, T, self.half_d)

        if values is not None:
            # Band 2: value-magnitude rotation
            log_vals = torch.sign(values) * torch.log1p(torch.abs(values))
            val_angles = log_vals.unsqueeze(-1) * self.val_freqs.unsqueeze(0).unsqueeze(0)
            val_sin = torch.sin(val_angles)
            val_cos = torch.cos(val_angles)

            x2 = x[:, :, self.half_d:]
            x2_pairs = x2.reshape(B, T, -1, 2)
            rotated2 = torch.stack([
                x2_pairs[..., 0] * val_cos - x2_pairs[..., 1] * val_sin,
                x2_pairs[..., 0] * val_sin + x2_pairs[..., 1] * val_cos,
            ], dim=-1).reshape(B, T, self.half_d)
        else:
            rotated2 = x[:, :, self.half_d:]

        return torch.cat([rotated1, rotated2], dim=-1)


class DimensionalAttentionMask(nn.Module):
    """Dimensional analysis attention constraint for equation decoding.

    Learns which token pairs are dimensionally compatible and creates
    soft attention masks to discourage dimensionally invalid operations.

    For example, adding quantities with incompatible dimensions should be
    penalized in the attention weights.
    """

    def __init__(self, vocab_size, d_model, n_dim_categories=8):
        super().__init__()
        self.enabled = True
        self.n_categories = n_dim_categories

        # Learn dimensional category for each token
        self.dim_embedding = nn.Embedding(vocab_size, n_dim_categories)

        # Compatibility matrix: which dimension categories can interact
        self.compatibility = nn.Parameter(
            torch.randn(n_dim_categories, n_dim_categories) * 0.1
        )

    def forward(self, token_ids):
        """Compute soft dimensional attention mask.

        Args:
            token_ids: (B, T) token IDs

        Returns:
            mask_bias: (B, T, T) attention bias (add to attention logits)
        """
        if not self.enabled:
            return None

        B, T = token_ids.shape

        # Get dimensional categories (soft)
        dim_probs = F.softmax(self.dim_embedding(token_ids), dim=-1)  # (B, T, n_cat)

        # Compute compatibility between all pairs
        # (B, T, n_cat) @ (n_cat, n_cat) @ (B, n_cat, T) -> (B, T, T)
        compat = torch.einsum('btc,cd,bsd->bts', dim_probs, self.compatibility, dim_probs)

        # Sigmoid to get soft mask (0 = incompatible, 1 = compatible)
        mask_bias = torch.sigmoid(compat) * 2 - 1  # Range [-1, 1]

        return mask_bias

    def count_invalid_outputs(self, pred_tokens, gt_tokens):
        """Count predictions that are dimensionally invalid."""
        # Simplified check: count predictions where operators are applied
        # to operands of different learned dimensional categories
        return 0  # Placeholder — full check requires parsing expression tree


class PhysicsEncodings(nn.Module):
    """Combined physics-informed encodings with ablation toggles."""

    def __init__(self, d_model, vocab_size, use_multiscale_rope=True,
                 use_dim_attention=True):
        super().__init__()
        self.multiscale_rope = MultiScaleRoPE(d_model)
        self.multiscale_rope.enabled = use_multiscale_rope

        self.dim_attention = DimensionalAttentionMask(vocab_size, d_model)
        self.dim_attention.enabled = use_dim_attention

    def apply_rope(self, x, positions=None, values=None):
        return self.multiscale_rope(x, positions, values)

    def get_dim_mask(self, token_ids):
        return self.dim_attention(token_ids)
