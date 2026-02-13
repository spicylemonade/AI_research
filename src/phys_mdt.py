"""
PhysMDT: Masked Diffusion Transformer for Physics Equation Derivation.

A masked diffusion transformer inspired by the ARC 2025 LLaDA architecture
(nie2025llada), adapted for symbolic equation generation from numerical
observations of physical systems.

Architecture:
    - 8-layer transformer with 8 attention heads, d_model=512
    - Masked diffusion training objective (random masking with variable ratio)
    - Dual-axis positional encoding (sequence position + expression tree depth)
    - Observation encoder for numerical (x, y) pairs
    - LoRA support for parameter-efficient finetuning

References:
    - nie2025llada: LLaDA — Large Language Diffusion Models
    - sahoo2024simple: MDLM — Simple and Effective Masked Diffusion Language Models
    - arc2025architects: ARC 2025 Solution by the ARChitects
    - lample2020deep: Deep Learning for Symbolic Mathematics
    - su2024roformer: RoFormer — Rotary Position Embedding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX, MASK_IDX, MAX_SEQ_LEN


# ─── Dual-Axis RoPE ─────────────────────────────────────────────────────────

class DualAxisRoPE(nn.Module):
    """Dual-axis Rotary Position Embedding encoding both sequence position
    and expression tree depth.

    Inspired by the ARC 2025 Golden Gate RoPE variant that encodes positions
    along multiple directional axes. For equations:
    - Axis 1: Left-to-right sequence position (standard RoPE)
    - Axis 2: Expression tree depth (how deep in the parse tree)

    Reference: su2024roformer, arc2025architects
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, max_depth: int = 32):
        super().__init__()
        self.d_model = d_model
        self.half_dim = d_model // 2

        # Precompute frequency bases for both axes
        # Axis 1: sequence position frequencies
        seq_freqs = 1.0 / (10000.0 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer('seq_freqs', seq_freqs)

        # Axis 2: tree depth frequencies
        depth_freqs = 1.0 / (10000.0 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer('depth_freqs', depth_freqs)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None,
                depths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply dual-axis RoPE to input tensor.

        Args:
            x: (batch, seq_len, d_model) input tensor
            positions: (batch, seq_len) sequence positions (default: 0,1,2,...)
            depths: (batch, seq_len) tree depths (default: all zeros)

        Returns:
            (batch, seq_len, d_model) position-encoded tensor
        """
        batch, seq_len, _ = x.shape
        device = x.device

        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        if depths is None:
            depths = torch.zeros(batch, seq_len, device=device)

        # Split x into two halves for the two axes
        x_seq = x[:, :, :self.half_dim]  # First half for sequence position
        x_dep = x[:, :, self.half_dim:]  # Second half for tree depth

        # Apply RoPE to first half (sequence position)
        x_seq = self._apply_rope(x_seq, positions.float(), self.seq_freqs)

        # Apply RoPE to second half (tree depth)
        x_dep = self._apply_rope(x_dep, depths.float(), self.depth_freqs)

        return torch.cat([x_seq, x_dep], dim=-1)

    def _apply_rope(self, x: torch.Tensor, pos: torch.Tensor,
                    freqs: torch.Tensor) -> torch.Tensor:
        """Apply standard RoPE rotation."""
        # pos: (batch, seq_len), freqs: (dim//2,)
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (batch, seq, dim//2)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        d = x.shape[-1]
        x1 = x[:, :, :d // 2]
        x2 = x[:, :, d // 2:]

        # Pad if dimensions don't match
        min_d = min(x1.shape[-1], cos_vals.shape[-1])
        x1 = x1[:, :, :min_d]
        x2 = x2[:, :, :min_d]
        cos_vals = cos_vals[:, :, :min_d]
        sin_vals = sin_vals[:, :, :min_d]

        out1 = x1 * cos_vals - x2 * sin_vals
        out2 = x1 * sin_vals + x2 * cos_vals

        return torch.cat([out1, out2], dim=-1)


# ─── LoRA Module ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for nn.Linear.

    Reference: arc2025architects — used rank-512 for pretraining, rank-32 for TTF
    """

    def __init__(self, linear: nn.Linear, rank: int = 32, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.scaling = alpha / rank
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        if self.enabled:
            lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
            return base_out + lora_out
        return base_out

    def reset_lora(self):
        """Reset LoRA weights to zero (for TTF)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


# ─── Observation Encoder ─────────────────────────────────────────────────────

class ObservationEncoder(nn.Module):
    """Encode numerical (x, y) observation pairs into continuous embeddings."""

    def __init__(self, d_model: int, max_vars: int = 6, n_obs: int = 20):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(max_vars + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(obs)


# ─── PhysMDT Transformer Block ──────────────────────────────────────────────

class PhysMDTBlock(nn.Module):
    """Transformer block for PhysMDT with bidirectional attention (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                cross_attn_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (bidirectional, no causal mask)
        normed = self.norm1(x)
        if cross_attn_kv is not None:
            attn_out, _ = self.attn(normed, cross_attn_kv, cross_attn_kv,
                                    key_padding_mask=key_padding_mask)
        else:
            attn_out, _ = self.attn(normed, normed, normed,
                                    key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# ─── Main PhysMDT Model ─────────────────────────────────────────────────────

class PhysMDT(nn.Module):
    """Masked Diffusion Transformer for Physics Equation Derivation.

    Unlike autoregressive models, PhysMDT uses bidirectional attention
    and a masked diffusion training objective: randomly mask output equation
    tokens at varying ratios and train to predict masked positions conditioned
    on observations.

    Architecture:
        - Observation encoder: processes (x, y) pairs
        - Token embedding + dual-axis RoPE
        - 8 bidirectional transformer blocks
        - Cross-attention from equation tokens to observation embeddings
        - Output projection to vocabulary logits

    The masked diffusion objective follows LLaDA (nie2025llada):
        1. Sample masking ratio t ~ Uniform(0, 1)
        2. Mask each output token independently with probability t
        3. Replace masked tokens with MASK embedding
        4. Train to predict original tokens at masked positions

    Reference: nie2025llada, arc2025architects, sahoo2024simple
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = MAX_SEQ_LEN,
        max_vars: int = 6,
        n_obs: int = 20,
        pad_idx: int = PAD_IDX,
        mask_idx: int = MASK_IDX,
        lora_rank: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Observation encoder
        self.obs_encoder = ObservationEncoder(d_model, max_vars, n_obs)

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Dual-axis positional encoding
        self.rope = DualAxisRoPE(d_model, max_seq_len)

        # Bidirectional transformer blocks
        self.blocks = nn.ModuleList([
            PhysMDTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Cross-attention blocks (interleaved)
        self.cross_blocks = nn.ModuleList([
            PhysMDTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.out_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # LoRA support
        self.lora_modules = []
        if lora_rank > 0:
            self._apply_lora(lora_rank)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Special init for mask embedding
        nn.init.normal_(self.token_embedding.weight[self.mask_idx], mean=0, std=0.02)

    def _apply_lora(self, rank: int):
        """Apply LoRA to all linear layers in attention modules."""
        for block in self.blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear) and module.in_features == self.d_model:
                    lora = LoRALinear(module, rank=rank)
                    self.lora_modules.append(lora)

    def get_mask_embedding(self) -> torch.Tensor:
        """Return the MASK token embedding."""
        return self.token_embedding.weight[self.mask_idx]

    def mask_tokens(self, token_ids: torch.Tensor, mask_ratio: Optional[float] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to token sequences.

        Following LLaDA, sample mask ratio uniformly from [0, 1] unless specified.

        Args:
            token_ids: (batch, seq_len) original token indices
            mask_ratio: fixed mask ratio (None = sample randomly per batch)

        Returns:
            masked_ids: token_ids with some positions replaced by MASK
            mask: boolean tensor indicating which positions were masked
        """
        batch, seq_len = token_ids.shape

        if mask_ratio is None:
            mask_ratio = torch.rand(1).item()

        # Don't mask BOS, EOS, or PAD tokens
        can_mask = (token_ids != self.pad_idx) & (token_ids != BOS_IDX) & (token_ids != EOS_IDX)

        # Generate random mask
        rand_vals = torch.rand(batch, seq_len, device=token_ids.device)
        mask = (rand_vals < mask_ratio) & can_mask

        # Apply mask
        masked_ids = token_ids.clone()
        masked_ids[mask] = self.mask_idx

        return masked_ids, mask

    def forward(self, obs: torch.Tensor, token_ids: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                depths: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Forward pass for masked diffusion training.

        Args:
            obs: (batch, n_obs, max_vars+1) observation data
            token_ids: (batch, seq_len) token indices (some may be MASK)
            positions: (batch, seq_len) sequence positions
            depths: (batch, seq_len) tree depths

        Returns:
            (batch, seq_len, vocab_size) logits for all positions
        """
        # Encode observations
        obs_emb = self.obs_encoder(obs)  # (batch, n_obs, d_model)

        # Embed tokens
        tok_emb = self.token_embedding(token_ids)  # (batch, seq_len, d_model)

        # Apply dual-axis RoPE
        tok_emb = self.rope(tok_emb, positions, depths)

        # Key padding mask for observations (all valid)
        # and for token sequence
        token_pad_mask = (token_ids == self.pad_idx)

        # Interleaved self-attention and cross-attention
        h = tok_emb
        for self_block, cross_block in zip(self.blocks, self.cross_blocks):
            h = self_block(h, key_padding_mask=token_pad_mask)
            h = cross_block(h, cross_attn_kv=obs_emb)

        # Output projection
        h = self.out_norm(h)
        logits = self.output_proj(h)

        return logits

    def compute_loss(self, obs: torch.Tensor, token_ids: torch.Tensor,
                     positions: Optional[torch.Tensor] = None,
                     depths: Optional[torch.Tensor] = None,
                     mask_ratio: Optional[float] = None,
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute masked diffusion loss.

        1. Mask random subset of tokens
        2. Forward pass to predict masked positions
        3. Cross-entropy loss only on masked positions

        Returns:
            loss: scalar loss tensor
            metrics: dict with loss value and mask ratio
        """
        # Apply masking
        masked_ids, mask = self.mask_tokens(token_ids, mask_ratio)

        # Forward
        logits = self.forward(obs, masked_ids, positions, depths)

        # Loss only on masked positions
        if mask.any():
            masked_logits = logits[mask]  # (n_masked, vocab_size)
            masked_targets = token_ids[mask]  # (n_masked,)
            loss = F.cross_entropy(masked_logits, masked_targets)
        else:
            loss = torch.tensor(0.0, device=obs.device, requires_grad=True)

        metrics = {
            "loss": loss.item(),
            "mask_ratio": mask.float().mean().item(),
            "n_masked": mask.sum().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def generate(self, obs: torch.Tensor, max_len: int = MAX_SEQ_LEN,
                 n_steps: int = 50) -> torch.Tensor:
        """Generate equation tokens using iterative unmasking.

        Start with all MASK tokens, iteratively reveal the most confident.

        Args:
            obs: (batch, n_obs, max_vars+1)
            max_len: maximum sequence length
            n_steps: number of unmasking steps

        Returns:
            (batch, max_len) generated token indices
        """
        self.eval()
        batch = obs.size(0)
        device = obs.device

        # Initialize with BOS + all MASK + EOS
        tokens = torch.full((batch, max_len), self.mask_idx, dtype=torch.long, device=device)
        tokens[:, 0] = BOS_IDX

        for step in range(n_steps):
            logits = self.forward(obs, tokens)

            # Get confidence for each position
            probs = F.softmax(logits, dim=-1)
            max_probs, max_tokens = probs.max(dim=-1)

            # Only update masked positions
            is_mask = tokens == self.mask_idx
            if not is_mask.any():
                break

            # Fraction of positions to unmask this step
            unmask_frac = (step + 1) / n_steps
            n_to_unmask = max(1, int(is_mask.float().sum(dim=1).max().item() * unmask_frac))

            # For each batch element, unmask the most confident masked positions
            for b in range(batch):
                mask_positions = is_mask[b].nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    continue

                confidences = max_probs[b, mask_positions]
                n_unmask = min(n_to_unmask, len(mask_positions))
                top_k_idx = confidences.topk(n_unmask).indices
                unmask_pos = mask_positions[top_k_idx]
                tokens[b, unmask_pos] = max_tokens[b, unmask_pos]

        # Set EOS at first remaining MASK position
        for b in range(batch):
            mask_pos = (tokens[b] == self.mask_idx).nonzero(as_tuple=True)[0]
            if len(mask_pos) > 0:
                tokens[b, mask_pos[0]] = EOS_IDX
                tokens[b, mask_pos[1:]] = PAD_IDX

        return tokens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_lora_parameters(self) -> int:
        count = 0
        for m in self.lora_modules:
            count += m.lora_A.numel() + m.lora_B.numel()
        return count


def build_phys_mdt(d_model=512, n_layers=8, n_heads=8, lora_rank=0, **kwargs) -> PhysMDT:
    """Build PhysMDT with specified configuration."""
    return PhysMDT(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        lora_rank=lora_rank,
        **kwargs,
    )


if __name__ == '__main__':
    # Test model
    model = build_phys_mdt(d_model=256, n_layers=4, n_heads=8)
    print(f"PhysMDT")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  d_model: {model.d_model}")
    print(f"  n_layers: {model.n_layers}")

    # Test forward pass
    batch = 4
    obs = torch.randn(batch, 20, 7)
    token_ids = torch.randint(0, VOCAB_SIZE, (batch, 30))
    token_ids[:, 0] = BOS_IDX

    logits = model.forward(obs, token_ids)
    print(f"  Forward: obs {obs.shape} + tokens {token_ids.shape} -> {logits.shape}")

    # Test loss computation
    loss, metrics = model.compute_loss(obs, token_ids)
    print(f"  Loss: {loss.item():.4f}, mask_ratio: {metrics['mask_ratio']:.3f}")

    # Test generation
    gen = model.generate(obs, max_len=30, n_steps=10)
    print(f"  Generate: obs {obs.shape} -> {gen.shape}")
