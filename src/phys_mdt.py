#!/usr/bin/env python3
"""PhysMDT: Physics Masked Diffusion Transformer.

8-layer transformer with masked diffusion training, dual-axis RoPE,
cross-attention observation encoder, and LoRA support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Dual-Axis Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class DualAxisRoPE(nn.Module):
    """Rotary Position Embedding encoding both sequence position and tree depth."""

    def __init__(self, d_model: int, max_seq_len: int = 256, max_depth: int = 16):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for dual-axis RoPE"
        self.d_model = d_model
        half_dim = d_model // 2  # each axis gets half

        # Precompute frequency bands for each axis
        inv_freq_seq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        inv_freq_depth = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))

        self.register_buffer('inv_freq_seq', inv_freq_seq)
        self.register_buffer('inv_freq_depth', inv_freq_depth)
        self.max_seq_len = max_seq_len
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor, seq_pos: Optional[torch.Tensor] = None,
                tree_depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply dual-axis RoPE to input tensor.

        Args:
            x: (batch, seq_len, d_model) input
            seq_pos: (batch, seq_len) sequence positions (default: 0..seq_len-1)
            tree_depth: (batch, seq_len) tree depths (default: all 0)

        Returns:
            (batch, seq_len, d_model) with RoPE applied
        """
        batch, seq_len, d = x.shape
        half = d // 2

        if seq_pos is None:
            seq_pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        if tree_depth is None:
            tree_depth = torch.zeros(batch, seq_len, device=x.device, dtype=torch.long)

        # Sequence axis RoPE (first half of d_model)
        seq_pos_float = seq_pos.float()
        sincos_seq = self._compute_sincos(seq_pos_float, self.inv_freq_seq, half)

        # Depth axis RoPE (second half of d_model)
        depth_float = tree_depth.float()
        sincos_depth = self._compute_sincos(depth_float, self.inv_freq_depth, half)

        # Apply to respective halves
        x_seq = x[..., :half]
        x_depth = x[..., half:]

        x_seq = self._apply_rope(x_seq, sincos_seq)
        x_depth = self._apply_rope(x_depth, sincos_depth)

        return torch.cat([x_seq, x_depth], dim=-1)

    def _compute_sincos(self, positions, inv_freq, dim):
        """Compute sin/cos for RoPE."""
        # positions: (batch, seq_len)
        # inv_freq: (dim//2,)
        freqs = torch.einsum('bi,j->bij', positions, inv_freq)  # (batch, seq_len, dim//2)
        sin_val = freqs.sin()
        cos_val = freqs.cos()
        return sin_val, cos_val

    def _apply_rope(self, x, sincos):
        """Apply rotary embedding to x."""
        sin_val, cos_val = sincos
        d = x.shape[-1]
        half_d = d // 2

        x1 = x[..., :half_d]
        x2 = x[..., half_d:]

        # Expand sin/cos to match x dimensions
        sin_val = sin_val[..., :half_d]
        cos_val = cos_val[..., :half_d]

        out1 = x1 * cos_val - x2 * sin_val
        out2 = x1 * sin_val + x2 * cos_val
        return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Linear layer with optional LoRA adaptation."""

    def __init__(self, in_features: int, out_features: int, rank: int = 0,
                 bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = rank
        self.lora_enabled = rank > 0

        if self.lora_enabled:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # B initialized to zero so LoRA starts as identity

    def forward(self, x):
        out = self.linear(x)
        if self.lora_enabled:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + lora_out
        return out

    def enable_lora(self, rank: int):
        """Enable or reconfigure LoRA."""
        if not self.lora_enabled or self.rank != rank:
            self.rank = rank
            self.lora_enabled = True
            device = self.linear.weight.device
            self.lora_A = nn.Parameter(torch.zeros(rank, self.linear.in_features, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.linear.out_features, rank, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def disable_lora(self):
        self.lora_enabled = False


# ---------------------------------------------------------------------------
# Observation Encoder with Cross-Attention
# ---------------------------------------------------------------------------

class CrossAttentionObsEncoder(nn.Module):
    """Encode (x, y) observation pairs and inject into equation representations via cross-attention."""

    def __init__(self, max_vars: int = 5, d_model: int = 256, n_heads: int = 8,
                 n_points: int = 20, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(max_vars + 1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_points, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

        # Self-attention layers to process observations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.obs_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, n_points, n_vars)
            Y: (batch, n_points)

        Returns:
            (batch, n_points, d_model) observation memory for cross-attention
        """
        XY = torch.cat([X, Y.unsqueeze(-1)], dim=-1)
        if XY.shape[-1] < self.input_proj.in_features:
            pad = torch.zeros(*XY.shape[:-1], self.input_proj.in_features - XY.shape[-1],
                              device=XY.device, dtype=XY.dtype)
            XY = torch.cat([XY, pad], dim=-1)
        elif XY.shape[-1] > self.input_proj.in_features:
            XY = XY[..., :self.input_proj.in_features]

        h = self.input_proj(XY)
        n_pts = h.shape[1]
        h = h + self.pos_encoding[:, :n_pts, :]
        h = self.norm(h)
        return self.obs_encoder(h)


# ---------------------------------------------------------------------------
# PhysMDT Transformer Block
# ---------------------------------------------------------------------------

class PhysMDTBlock(nn.Module):
    """Single transformer block with self-attention, cross-attention to observations, and FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 lora_rank: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            LoRALinear(d_model, d_ff, rank=lora_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(d_ff, d_model, rank=lora_rank),
            nn.Dropout(dropout),
        )

        # LoRA on attention projections
        self.q_proj = LoRALinear(d_model, d_model, rank=lora_rank)
        self.v_proj = LoRALinear(d_model, d_model, rank=lora_rank)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (bidirectional for masked diffusion — no causal mask)
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.self_attn(x, x, x)
        x = residual + x_attn

        # Cross-attention to observations
        residual = x
        x = self.norm2(x)
        x_cross, _ = self.cross_attn(x, memory, memory)
        x = residual + x_cross

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)

        return x


# ---------------------------------------------------------------------------
# PhysMDT Main Model
# ---------------------------------------------------------------------------

class PhysMDT(nn.Module):
    """Physics Masked Diffusion Transformer.

    Args:
        vocab_size: Size of token vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        max_vars: Maximum number of input variables
        n_points: Number of observation points
        lora_rank: LoRA rank (0 = disabled)
        dropout: Dropout rate
    """

    def __init__(self, vocab_size: int = 147, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 8, d_ff: int = 1024, max_seq_len: int = 128,
                 max_vars: int = 5, n_points: int = 20, lora_rank: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.mask_token_id = 3  # [MASK] token ID

        # Dual-axis RoPE
        self.rope = DualAxisRoPE(d_model, max_seq_len)

        # Observation encoder
        self.obs_encoder = CrossAttentionObsEncoder(
            max_vars=max_vars, d_model=d_model, n_heads=n_heads,
            n_points=n_points, n_layers=2
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PhysMDTBlock(d_model, n_heads, d_ff, dropout, lora_rank)
            for _ in range(n_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                mask_positions: Optional[torch.Tensor] = None,
                tree_depths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for masked diffusion training.

        Args:
            token_ids: (batch, seq_len) token IDs with some positions masked
            X: (batch, n_points, n_vars) observation inputs
            Y: (batch, n_points) observation outputs
            mask_positions: (batch, seq_len) boolean mask (True = masked)
            tree_depths: (batch, seq_len) tree depth for each position

        Returns:
            (batch, seq_len, vocab_size) logits for all positions
        """
        batch, seq_len = token_ids.shape

        # Token embeddings
        h = self.token_embedding(token_ids)  # (batch, seq_len, d_model)

        # Apply dual-axis RoPE
        h = self.rope(h, tree_depth=tree_depths)

        # Encode observations
        memory = self.obs_encoder(X, Y)  # (batch, n_points, d_model)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, memory)

        # Output projection
        h = self.output_norm(h)
        logits = self.output_proj(h)  # (batch, seq_len, vocab_size)

        return logits

    def compute_masked_diffusion_loss(self, token_ids: torch.Tensor,
                                       X: torch.Tensor, Y: torch.Tensor,
                                       tree_depths: Optional[torch.Tensor] = None,
                                       t: Optional[float] = None) -> Tuple[torch.Tensor, dict]:
        """Compute the masked diffusion training loss.

        Randomly masks tokens at rate t, predicts masked tokens.

        Args:
            token_ids: (batch, seq_len) ground-truth token IDs (with BOS/EOS)
            X, Y: observation data
            tree_depths: optional tree depths
            t: masking rate (if None, sampled from U(0.1, 1.0))

        Returns:
            (loss, info_dict) where loss is scalar and info contains accuracy etc.
        """
        batch, seq_len = token_ids.shape

        # Sample masking rate
        if t is None:
            t = torch.empty(1).uniform_(0.1, 1.0).item()

        # Create mask: True where token should be masked
        mask = torch.bernoulli(torch.full((batch, seq_len), t, device=token_ids.device)).bool()

        # Don't mask special tokens (PAD=0, BOS=1, EOS=2)
        special_mask = (token_ids <= 2)
        mask = mask & ~special_mask

        # Replace masked positions with [MASK] token
        masked_ids = token_ids.clone()
        masked_ids[mask] = self.mask_token_id

        # Forward pass
        logits = self.forward(masked_ids, X, Y, mask_positions=mask, tree_depths=tree_depths)

        # Loss only on masked positions
        if mask.any():
            masked_logits = logits[mask]  # (n_masked, vocab_size)
            masked_targets = token_ids[mask]  # (n_masked,)
            loss = F.cross_entropy(masked_logits, masked_targets)

            # Compute accuracy on masked positions
            with torch.no_grad():
                preds = masked_logits.argmax(dim=-1)
                accuracy = (preds == masked_targets).float().mean().item()
        else:
            loss = torch.tensor(0.0, device=token_ids.device)
            accuracy = 0.0

        info = {
            'mask_rate': t,
            'n_masked': mask.sum().item(),
            'accuracy': accuracy,
        }
        return loss, info

    @torch.no_grad()
    def generate_single_pass(self, X: torch.Tensor, Y: torch.Tensor,
                              seq_len: int = 48,
                              tree_depths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single-pass generation: fully masked -> predict all tokens.

        Args:
            X: (batch, n_points, n_vars)
            Y: (batch, n_points)
            seq_len: length of sequence to generate

        Returns:
            (batch, seq_len) predicted token IDs
        """
        self.eval()
        batch = X.shape[0]
        device = X.device

        # Start with all [MASK] except BOS at position 0
        token_ids = torch.full((batch, seq_len), self.mask_token_id,
                               dtype=torch.long, device=device)
        token_ids[:, 0] = 1  # BOS

        logits = self.forward(token_ids, X, Y, tree_depths=tree_depths)
        # Argmax decode at all positions except BOS
        pred_ids = logits.argmax(dim=-1)
        pred_ids[:, 0] = 1  # Keep BOS
        return pred_ids

    def enable_lora(self, rank: int = 32):
        """Enable LoRA on all compatible layers."""
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, LoRALinear):
                    module.enable_lora(rank)

    def disable_lora(self):
        """Disable LoRA on all layers."""
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, LoRALinear):
                    module.disable_lora()

    def get_lora_parameters(self):
        """Get only LoRA parameters for finetuning."""
        params = []
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, LoRALinear) and module.lora_enabled:
                    params.extend([module.lora_A, module.lora_B])
        return params
