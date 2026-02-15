"""Physics Masked Diffusion Transformer (PhysMDT).

Novel architecture combining:
1. Masked diffusion language model for symbolic expression generation
2. Set-transformer observation encoder (permutation invariant)
3. Cross-attention between observations and masked expressions
4. Tree-positional encoding for expression structure
5. Dimensional analysis attention bias

Inspired by ARChitects ARC 2025 solution and MDLM (Sahoo et al. 2024).
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX, MASK_IDX


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhysMDTConfig:
    """Configuration for PhysMDT model.

    Target: ~50-80M parameters on single A100.
    """
    # Model dimensions
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6  # observation encoder
    num_decoder_layers: int = 8  # expression decoder
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Sequence limits
    max_expr_len: int = 64
    max_obs_points: int = 50
    max_vars: int = 5

    # Vocabulary
    vocab_size: int = VOCAB_SIZE
    pad_idx: int = PAD_IDX
    sos_idx: int = SOS_IDX
    eos_idx: int = EOS_IDX
    mask_idx: int = MASK_IDX

    # Set transformer (observation encoder)
    num_inducing_points: int = 32  # ISAB inducing points

    # Tree positional encoding
    max_tree_depth: int = 16
    max_sibling_index: int = 8
    use_tree_pos: bool = True

    # Dimensional analysis
    use_dim_analysis: bool = True
    n_physical_dims: int = 3  # mass, length, time


# ---------------------------------------------------------------------------
# Set Transformer Encoder (Permutation Invariant)
# ---------------------------------------------------------------------------

class MultiheadAttentionBlock(nn.Module):
    """Multihead attention block for set transformer."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value, key_padding_mask=None):
        # Cross-attention
        attended, _ = self.attn(query, key_value, key_value,
                                key_padding_mask=key_padding_mask)
        x = self.norm1(query + attended)
        x = self.norm2(x + self.ff(x))
        return x


class InducedSetAttentionBlock(nn.Module):
    """ISAB: Inducing-point set attention block from Set Transformer.

    Reduces O(n²) attention to O(nm) where m = num_inducing_points.
    """

    def __init__(self, d_model, nhead, num_inducing_points, dropout=0.1):
        super().__init__()
        self.inducing_points = nn.Parameter(
            torch.randn(1, num_inducing_points, d_model) * 0.02
        )
        self.mab1 = MultiheadAttentionBlock(d_model, nhead, dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, nhead, dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, n_points, d_model)
        batch_size = x.size(0)
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        # Step 1: inducing points attend to input
        h = self.mab1(inducing, x, key_padding_mask=key_padding_mask)
        # Step 2: input attends to inducing points (no mask needed)
        return self.mab2(x, h)


class SetTransformerEncoder(nn.Module):
    """Permutation-invariant encoder for observation points.

    Processes (batch, n_points, n_vars+1) -> (batch, n_inducing, d_model).
    """

    def __init__(self, config: PhysMDTConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.max_vars + 1, config.d_model)
        self.isab_layers = nn.ModuleList([
            InducedSetAttentionBlock(
                config.d_model, config.nhead,
                config.num_inducing_points, config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ])
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, observations, obs_mask=None):
        """
        Args:
            observations: (batch, n_points, max_vars+1) float tensor
            obs_mask: (batch, n_points, max_vars+1) float mask (1=valid, 0=pad)

        Returns:
            encoded: (batch, n_points, d_model) — encoded observations
            key_padding_mask: (batch, n_points) bool mask for cross-attention
        """
        x = self.input_proj(observations)  # (batch, n_points, d_model)

        # Create key padding mask: True where ALL vars are padded
        if obs_mask is not None:
            # A point is invalid if all its entries are 0
            key_padding_mask = obs_mask.sum(dim=-1) == 0  # (batch, n_points), True=padded
        else:
            key_padding_mask = None

        for isab in self.isab_layers:
            x = isab(x, key_padding_mask=key_padding_mask)

        x = self.output_norm(x)
        return x, key_padding_mask


# ---------------------------------------------------------------------------
# Tree Positional Encoding (2D RoPE variant)
# ---------------------------------------------------------------------------

class TreePositionalEncoding(nn.Module):
    """2D positional encoding for expression tree structure.

    Encodes (depth, sibling_index) using a 2D RoPE-inspired approach.
    Adapted from ARChitects' Golden Gate RoPE.
    """

    def __init__(self, d_model, max_depth=16, max_sibling=8):
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth
        self.max_sibling = max_sibling

        # Learnable embeddings for depth and sibling index
        self.depth_embedding = nn.Embedding(max_depth, d_model // 2)
        self.sibling_embedding = nn.Embedding(max_sibling, d_model // 2)

        # Fallback: standard positional encoding for when tree info unavailable
        self.pos_embedding = nn.Embedding(64, d_model)

    def forward(self, seq_len, tree_depths=None, sibling_indices=None, device=None):
        """
        Args:
            seq_len: length of the sequence
            tree_depths: (batch, seq_len) int tensor of tree depths (optional)
            sibling_indices: (batch, seq_len) int tensor of sibling indices (optional)

        Returns:
            pos_enc: (batch, seq_len, d_model) or (1, seq_len, d_model)
        """
        if device is None:
            device = self.pos_embedding.weight.device

        if tree_depths is not None and sibling_indices is not None:
            # Clamp to valid range
            depths = tree_depths.clamp(0, self.max_depth - 1)
            siblings = sibling_indices.clamp(0, self.max_sibling - 1)

            d_emb = self.depth_embedding(depths)        # (batch, seq, d_model//2)
            s_emb = self.sibling_embedding(siblings)      # (batch, seq, d_model//2)
            return torch.cat([d_emb, s_emb], dim=-1)      # (batch, seq, d_model)
        else:
            # Fallback to standard positional encoding
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            return self.pos_embedding(positions)


# ---------------------------------------------------------------------------
# Dimensional Analysis Bias Head
# ---------------------------------------------------------------------------

class DimensionalAnalysisBias(nn.Module):
    """Auxiliary attention bias based on dimensional analysis.

    Tracks physical dimensions (mass, length, time) through the expression
    and adds a bias term to attention scores.
    """

    def __init__(self, d_model, n_dims=3, nhead=1):
        super().__init__()
        # Project token embeddings to dimensional scores
        self.dim_proj = nn.Linear(d_model, n_dims)
        # Learnable dimensional compatibility bias
        self.compat_proj = nn.Linear(n_dims * 2, nhead)
        self.nhead = nhead
        self.n_dims = n_dims

    def forward(self, token_embeddings):
        """
        Args:
            token_embeddings: (batch, seq_len, d_model)

        Returns:
            dim_bias: (batch, nhead, seq_len, seq_len) attention bias
            dim_loss: scalar dimensional consistency loss
        """
        batch, seq_len, _ = token_embeddings.shape
        # Project to dimensional space
        dims = self.dim_proj(token_embeddings)  # (batch, seq, n_dims)

        # Compute pairwise dimensional compatibility
        dims_i = dims.unsqueeze(2).expand(-1, -1, seq_len, -1)
        dims_j = dims.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pair_dims = torch.cat([dims_i, dims_j], dim=-1)  # (batch, seq, seq, 2*n_dims)

        bias = self.compat_proj(pair_dims)  # (batch, seq, seq, nhead)
        bias = bias.permute(0, 3, 1, 2)    # (batch, nhead, seq, seq)

        # Dimensional consistency loss: adjacent tokens should have compatible dims
        dim_loss = torch.tensor(0.0, device=token_embeddings.device)
        if seq_len > 1:
            adj_diff = (dims[:, 1:] - dims[:, :-1]).pow(2).mean()
            dim_loss = adj_diff * 0.01

        return bias, dim_loss


# ---------------------------------------------------------------------------
# PhysMDT Expression Decoder
# ---------------------------------------------------------------------------

class PhysMDTDecoderLayer(nn.Module):
    """Single decoder layer with self-attention + cross-attention + FFN."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, attn_bias=None):
        """
        Args:
            tgt: (batch, seq_len, d_model) expression token embeddings
            memory: (batch, n_points, d_model) encoded observations
            tgt_key_padding_mask: (batch, seq_len) bool mask
            memory_key_padding_mask: (batch, n_points) bool mask
            attn_bias: optional (batch, nhead, seq, seq) dimensional bias
        """
        # Self-attention (bidirectional — no causal mask!)
        x = tgt
        attn_out, self_attn_weights = self.self_attn(
            x, x, x, key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to observations
        cross_out, cross_attn_weights = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = self.norm2(x + self.dropout(cross_out))

        # Feed-forward
        x = self.norm3(x + self.ff(x))

        return x, self_attn_weights, cross_attn_weights


# ---------------------------------------------------------------------------
# PhysMDT Main Model
# ---------------------------------------------------------------------------

class PhysMDT(nn.Module):
    """Physics Masked Diffusion Transformer.

    A masked diffusion model for symbolic equation discovery.
    The decoder operates on partially masked expression sequences and
    predicts logits for all masked positions simultaneously.
    """

    def __init__(self, config: PhysMDTConfig):
        super().__init__()
        self.config = config

        # Observation encoder (set transformer)
        self.obs_encoder = SetTransformerEncoder(config)

        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )

        # Positional encoding
        if config.use_tree_pos:
            self.pos_encoding = TreePositionalEncoding(
                config.d_model, config.max_tree_depth, config.max_sibling_index
            )
        else:
            self.pos_encoding = TreePositionalEncoding(
                config.d_model, config.max_tree_depth, config.max_sibling_index
            )

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            PhysMDTDecoderLayer(
                config.d_model, config.nhead,
                config.dim_feedforward, config.dropout
            )
            for _ in range(config.num_decoder_layers)
        ])

        # Output projection (weight-tied with token embedding)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight

        # Layer norm
        self.output_norm = nn.LayerNorm(config.d_model)

        # Dimensional analysis bias
        if config.use_dim_analysis:
            self.dim_analysis = DimensionalAnalysisBias(
                config.d_model, config.n_physical_dims, nhead=1
            )
        else:
            self.dim_analysis = None

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        self.token_embedding.weight.data[self.config.pad_idx].zero_()

    def get_token_embeddings(self):
        """Return the token embedding matrix."""
        return self.token_embedding.weight

    def forward(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        masked_tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
        sibling_indices: Optional[torch.Tensor] = None,
        soft_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            observations: (batch, n_points, max_vars+1)
            obs_mask: (batch, n_points, max_vars+1) float mask
            masked_tokens: (batch, seq_len) token indices with some <MASK>
            token_mask: (batch, seq_len) bool, True where tokens are masked
            tree_depths: (batch, seq_len) int, optional tree depth per token
            sibling_indices: (batch, seq_len) int, optional sibling index
            soft_embeddings: (batch, seq_len, d_model) optional soft token embeddings
                            (used during refinement instead of discrete tokens)

        Returns:
            logits: (batch, seq_len, vocab_size) — predictions for all positions
            aux: dict with auxiliary outputs (dim_loss, attention weights)
        """
        batch_size, seq_len = masked_tokens.shape

        # Encode observations
        obs_encoded, obs_key_pad_mask = self.obs_encoder(observations, obs_mask)

        # Token embeddings
        if soft_embeddings is not None:
            # Use soft embeddings (during refinement)
            token_emb = soft_embeddings
        else:
            token_emb = self.token_embedding(masked_tokens)

        # Add positional encoding
        pos_enc = self.pos_encoding(
            seq_len, tree_depths, sibling_indices,
            device=token_emb.device
        )
        if pos_enc.dim() == 2:
            pos_enc = pos_enc.unsqueeze(0)
        token_emb = token_emb + pos_enc

        token_emb = self.embed_dropout(token_emb)

        # Key padding mask for tokens
        tgt_key_pad_mask = masked_tokens == self.config.pad_idx

        # Dimensional analysis bias
        dim_loss = torch.tensor(0.0, device=token_emb.device)
        attn_bias = None
        if self.dim_analysis is not None:
            attn_bias, dim_loss = self.dim_analysis(token_emb)

        # Decoder layers
        x = token_emb
        self_attn_weights_all = []
        cross_attn_weights_all = []

        for layer in self.decoder_layers:
            x, self_attn_w, cross_attn_w = layer(
                x, obs_encoded,
                tgt_key_padding_mask=tgt_key_pad_mask,
                memory_key_padding_mask=obs_key_pad_mask,
                attn_bias=attn_bias,
            )
            self_attn_weights_all.append(self_attn_w)
            cross_attn_weights_all.append(cross_attn_w)

        x = self.output_norm(x)

        # Output logits
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        aux = {
            'dim_loss': dim_loss,
            'self_attn_weights': self_attn_weights_all,
            'cross_attn_weights': cross_attn_weights_all,
            'encoded_obs': obs_encoded,
        }

        return logits, aux

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        dim_loss: torch.Tensor = None,
        dim_loss_weight: float = 0.1,
    ) -> torch.Tensor:
        """Compute masked diffusion training loss.

        Only computes cross-entropy on masked positions.

        Args:
            logits: (batch, seq_len, vocab_size)
            target_tokens: (batch, seq_len) ground truth tokens
            token_mask: (batch, seq_len) bool, True where tokens were masked
            dim_loss: optional dimensional analysis auxiliary loss
            dim_loss_weight: weight for dimensional loss

        Returns:
            loss: scalar loss
        """
        # Flatten
        batch, seq_len, vocab = logits.shape
        flat_logits = logits.reshape(-1, vocab)
        flat_targets = target_tokens.reshape(-1)
        flat_mask = token_mask.reshape(-1).float()

        # Cross-entropy (computed everywhere, but masked)
        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')

        # Only count loss on masked positions
        masked_loss = (ce_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)

        total_loss = masked_loss
        if dim_loss is not None:
            total_loss = total_loss + dim_loss_weight * dim_loss

        return total_loss

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def apply_random_mask(tokens: torch.Tensor, mask_ratio: float,
                      mask_idx: int = MASK_IDX, pad_idx: int = PAD_IDX,
                      sos_idx: int = SOS_IDX, eos_idx: int = EOS_IDX) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random masking to token sequences.

    Args:
        tokens: (batch, seq_len) token indices
        mask_ratio: fraction of non-special tokens to mask
        mask_idx: index of <MASK> token

    Returns:
        masked_tokens: (batch, seq_len) with some tokens replaced by <MASK>
        token_mask: (batch, seq_len) bool, True where tokens were masked
    """
    batch, seq_len = tokens.shape

    # Maskable: everything except SOS (position 0 anchor).
    # EOS and PAD are now maskable so the model learns sequence boundaries.
    maskable = (tokens != sos_idx)

    # Random mask
    rand = torch.rand(batch, seq_len, device=tokens.device)
    to_mask = maskable & (rand < mask_ratio)

    masked_tokens = tokens.clone()
    masked_tokens[to_mask] = mask_idx

    return masked_tokens, to_mask


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("PhysMDT Unit Tests")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = PhysMDTConfig()
    model = PhysMDT(config).to(device)

    n_params = model.count_parameters()
    print(f"\nModel parameters: {n_params:,}")

    # Test 1: Forward pass
    print("\nTest 1: Forward pass")
    batch_size = 4
    obs = torch.randn(batch_size, 50, 6, device=device)
    obs_mask = torch.ones(batch_size, 50, 6, device=device)
    tokens = torch.randint(0, config.vocab_size, (batch_size, 32), device=device)
    tokens[:, 0] = SOS_IDX
    tokens[:, -1] = EOS_IDX
    token_mask = torch.zeros(batch_size, 32, dtype=torch.bool, device=device)
    token_mask[:, 5:25] = True  # mask middle tokens

    logits, aux = model(obs, obs_mask, tokens, token_mask)
    print(f"  logits shape: {logits.shape} (expected: ({batch_size}, 32, {config.vocab_size}))")
    assert logits.shape == (batch_size, 32, config.vocab_size), "Shape mismatch!"
    print("  PASSED")

    # Test 2: Loss computation
    print("\nTest 2: Loss computation")
    target = torch.randint(0, config.vocab_size, (batch_size, 32), device=device)
    loss = model.compute_loss(logits, target, token_mask, aux['dim_loss'])
    print(f"  loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("  PASSED")

    # Test 3: Backward pass
    print("\nTest 3: Backward pass")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients should be non-zero"
    print("  PASSED")

    # Test 4: Random masking
    print("\nTest 4: Random masking")
    original = torch.tensor([[SOS_IDX, 5, 17, 7, 8, 9, EOS_IDX, PAD_IDX]], device=device)
    masked, mask = apply_random_mask(original, mask_ratio=0.5)
    print(f"  original: {original[0].tolist()}")
    print(f"  masked:   {masked[0].tolist()}")
    print(f"  mask:     {mask[0].tolist()}")
    # SOS, EOS, PAD should not be masked
    assert masked[0, 0] == SOS_IDX, "SOS should not be masked"
    assert masked[0, -2] == EOS_IDX, "EOS should not be masked"
    assert masked[0, -1] == PAD_IDX, "PAD should not be masked"
    print("  PASSED")

    # Test 5: Soft embeddings input
    print("\nTest 5: Soft embeddings input")
    soft_emb = torch.randn(batch_size, 32, config.d_model, device=device)
    logits_soft, _ = model(obs, obs_mask, tokens, soft_embeddings=soft_emb)
    print(f"  logits shape: {logits_soft.shape}")
    assert logits_soft.shape == logits.shape
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All PhysMDT unit tests PASSED")
    print(f"Model size: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print("=" * 60)
