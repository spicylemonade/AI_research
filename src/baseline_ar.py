#!/usr/bin/env python3
"""Autoregressive encoder-decoder transformer baseline for symbolic regression.

6-layer encoder-decoder transformer (8 heads, d_model=512) that takes
numerical observation pairs as input and outputs symbolic equations via
autoregressive decoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ObservationEncoder(nn.Module):
    """Encode numerical (x, y) observation pairs into a sequence of embeddings."""

    def __init__(self, max_vars: int = 5, d_model: int = 512, n_points: int = 50):
        super().__init__()
        self.d_model = d_model
        # Project each observation point (x_1, ..., x_d, y) -> d_model
        self.input_proj = nn.Linear(max_vars + 1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_points, d_model) * 0.02)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, n_points, n_vars)
            Y: (batch, n_points)

        Returns:
            (batch, n_points, d_model) encoded observations
        """
        # Concatenate X and Y
        XY = torch.cat([X, Y.unsqueeze(-1)], dim=-1)  # (batch, n_points, n_vars+1)
        # Pad to max_vars+1 if needed
        if XY.shape[-1] < self.input_proj.in_features:
            pad = torch.zeros(*XY.shape[:-1], self.input_proj.in_features - XY.shape[-1],
                              device=XY.device, dtype=XY.dtype)
            XY = torch.cat([XY, pad], dim=-1)
        elif XY.shape[-1] > self.input_proj.in_features:
            XY = XY[..., :self.input_proj.in_features]

        h = self.input_proj(XY)  # (batch, n_points, d_model)
        n_pts = h.shape[1]
        h = h + self.pos_encoding[:, :n_pts, :]
        return self.layer_norm(h)


class BaselineAR(nn.Module):
    """Standard autoregressive encoder-decoder transformer for symbolic regression."""

    def __init__(self, vocab_size: int = 147, d_model: int = 512, n_heads: int = 8,
                 n_encoder_layers: int = 6, n_decoder_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 128, max_vars: int = 5, n_points: int = 50):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Observation encoder
        self.obs_encoder = ObservationEncoder(max_vars, d_model, n_points)

        # Transformer encoder (processes observation embeddings)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Token embedding for decoder
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len * 2, d_model)  # extra room for generation

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_observations(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Encode numerical observations."""
        obs_emb = self.obs_encoder(X, Y)  # (batch, n_points, d_model)
        memory = self.transformer_encoder(obs_emb)  # (batch, n_points, d_model)
        return memory

    def decode_step(self, tgt_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Run decoder one step.

        Args:
            tgt_ids: (batch, seq_len) target token IDs
            memory: (batch, n_points, d_model) encoder output

        Returns:
            (batch, seq_len, vocab_size) logits
        """
        seq_len = tgt_ids.shape[1]
        positions = torch.arange(seq_len, device=tgt_ids.device).unsqueeze(0)

        tgt_emb = self.token_embedding(tgt_ids) + self.pos_embedding(positions)
        tgt_emb = tgt_emb * math.sqrt(self.d_model)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tgt_ids.device
        )

        decoded = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=causal_mask
        )
        logits = self.output_proj(decoded)
        return logits

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                tgt_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Args:
            X: (batch, n_points, n_vars)
            Y: (batch, n_points)
            tgt_ids: (batch, seq_len) target token IDs (teacher forced)

        Returns:
            (batch, seq_len, vocab_size) logits
        """
        memory = self.encode_observations(X, Y)
        logits = self.decode_step(tgt_ids, memory)
        return logits

    @torch.no_grad()
    def generate(self, X: torch.Tensor, Y: torch.Tensor,
                 bos_id: int = 1, eos_id: int = 2,
                 max_len: int = 64, temperature: float = 1.0) -> torch.Tensor:
        """Greedy autoregressive generation.

        Args:
            X: (batch, n_points, n_vars)
            Y: (batch, n_points)

        Returns:
            (batch, generated_len) token IDs
        """
        self.eval()
        batch_size = X.shape[0]
        memory = self.encode_observations(X, Y)

        # Start with BOS
        generated = torch.full((batch_size, 1), bos_id,
                               dtype=torch.long, device=X.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=X.device)

        for _ in range(max_len - 1):
            logits = self.decode_step(generated, memory)
            next_logits = logits[:, -1, :] / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Set finished sequences to PAD
            next_token = next_token.masked_fill(finished.unsqueeze(-1), 0)
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return generated
