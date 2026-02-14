"""PhysMDT: Physics Masked Diffusion Transformer.

Core masked diffusion backbone for symbolic equation generation.
Bidirectional transformer that predicts masked tokens given unmasked
context and data embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.ar_baseline import DeepSetsEncoder


class BidirectionalTransformerLayer(nn.Module):
    """Single bidirectional transformer layer (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                pos_encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (bidirectional - no mask)
        residual = x
        x = self.norm1(x)
        if pos_encoding is not None:
            q = k = x + pos_encoding
        else:
            q = k = x
        x_attn, _ = self.self_attn(q, k, x)
        x = residual + self.dropout(x_attn)

        # Cross-attention with data encoding
        residual = x
        x = self.norm2(x)
        x_cross, _ = self.cross_attn(x, memory, memory)
        x = residual + self.dropout(x_cross)

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)

        return x


class PhysMDT(nn.Module):
    """Physics Masked Diffusion Transformer.

    Bidirectional transformer that operates on masked equation token sequences.
    Predicts original tokens at masked positions given unmasked context
    and data embeddings.
    """

    def __init__(self, vocab_size: int = 200, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 8,
                 ffn_dim: int = 2048, max_seq_len: int = 64,
                 dropout: float = 0.1, input_dim: int = 10,
                 encoder_layers: int = 4, mask_token_id: int = 3,
                 use_tree_pe: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.use_tree_pe = use_tree_pe

        # Data encoder
        self.data_encoder = DeepSetsEncoder(
            input_dim=input_dim, d_model=d_model,
            n_heads=min(n_heads, 4), n_layers=encoder_layers,
        )

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Bidirectional transformer layers
        self.layers = nn.ModuleList([
            BidirectionalTransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data_matrix: torch.Tensor,
                masked_token_ids: torch.Tensor,
                pos_encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            data_matrix: (batch, n_points, input_dim) data points.
            masked_token_ids: (batch, seq_len) token IDs with some masked.
            pos_encoding: Optional (batch, seq_len, d_model) positional encoding.

        Returns:
            logits: (batch, seq_len, vocab_size) predictions for all positions.
        """
        batch_size, seq_len = masked_token_ids.shape

        # Encode data
        context = self.data_encoder(data_matrix)  # (batch, d_model)
        memory = context.unsqueeze(1)  # (batch, 1, d_model)

        # Token embeddings
        x = self.token_embedding(masked_token_ids) * math.sqrt(self.d_model)

        # Positional encoding
        if pos_encoding is not None:
            x = x + pos_encoding
        else:
            positions = torch.arange(seq_len, device=masked_token_ids.device)
            x = x + self.pos_embedding(positions)

        # Bidirectional transformer layers
        for layer in self.layers:
            x = layer(x, memory, pos_encoding)

        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)

        return logits

    def forward_soft(self, data_matrix: torch.Tensor,
                     soft_embeddings: torch.Tensor,
                     pos_encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with continuous soft embeddings (no token lookup).

        Used during soft-masking recursion inference.

        Args:
            data_matrix: (batch, n_points, input_dim) data points.
            soft_embeddings: (batch, seq_len, d_model) continuous embeddings.
            pos_encoding: Optional positional encoding.

        Returns:
            logits: (batch, seq_len, vocab_size) predictions.
        """
        x = soft_embeddings * math.sqrt(self.d_model)

        # Positional encoding
        if pos_encoding is not None:
            x = x + pos_encoding
        else:
            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=x.device)
            x = x + self.pos_embedding(positions)

        # Encode data
        context = self.data_encoder(data_matrix)
        memory = context.unsqueeze(1)

        # Bidirectional layers
        for layer in self.layers:
            x = layer(x, memory, pos_encoding)

        x = self.output_norm(x)
        logits = self.output_proj(x)
        return logits

    def compute_loss(self, data_matrix: torch.Tensor,
                     token_ids: torch.Tensor,
                     mask_rate: Optional[float] = None,
                     pos_encoding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Compute masked diffusion training loss.

        Args:
            data_matrix: (batch, n_points, input_dim)
            token_ids: (batch, seq_len) original token IDs.
            mask_rate: Masking probability. If None, sampled from Uniform(0.1, 0.9).
            pos_encoding: Optional positional encoding.

        Returns:
            loss: Scalar cross-entropy loss on masked positions only.
            info: Dict with additional information.
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Sample mask rate if not provided
        if mask_rate is None:
            mask_rate = torch.empty(1).uniform_(0.1, 0.9).item()

        # Create mask (True = masked)
        mask = torch.rand(batch_size, seq_len, device=device) < mask_rate
        # Don't mask PAD, BOS, EOS tokens
        special_mask = (token_ids == 0) | (token_ids == 1) | (token_ids == 2)
        mask = mask & ~special_mask

        # Apply mask
        masked_tokens = token_ids.clone()
        masked_tokens[mask] = self.mask_token_id

        # Forward pass
        logits = self.forward(data_matrix, masked_tokens, pos_encoding)

        # Loss on masked positions only
        if mask.sum() > 0:
            masked_logits = logits[mask]  # (n_masked, vocab)
            masked_targets = token_ids[mask]  # (n_masked,)
            loss = F.cross_entropy(masked_logits, masked_targets)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        info = {
            'mask_rate': mask_rate,
            'n_masked': int(mask.sum().item()),
            'n_total': int((~special_mask).sum().item()),
        }

        return loss, info
