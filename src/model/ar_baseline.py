"""Autoregressive Transformer Baseline for Symbolic Regression.

Implements a SymbolicGPT-style causal transformer decoder that takes
data point embeddings as prefix and autoregressively generates equation tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant set encoder using DeepSets + MHA.

    Encodes a variable-size set of (x, y) data points into a fixed-size
    context vector for conditioning the decoder.
    """

    def __init__(self, input_dim: int = 10, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, data_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data_matrix: (batch, n_points, input_dim) data matrix.

        Returns:
            context: (batch, d_model) fixed-size context vector.
        """
        x = self.input_proj(data_matrix)
        for layer in self.layers:
            x = layer(x)
        # Pool over data points
        x = x.transpose(1, 2)  # (batch, d_model, n_points)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        return self.output_proj(x)


class ARBaseline(nn.Module):
    """Autoregressive transformer decoder for symbolic regression.

    Takes a data encoding as prefix and generates equation tokens
    autoregressively using causal masked self-attention.
    """

    def __init__(self, vocab_size: int = 200, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 4,
                 ffn_dim: int = 1024, max_seq_len: int = 64,
                 dropout: float = 0.1, input_dim: int = 10,
                 encoder_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Data encoder
        self.data_encoder = DeepSetsEncoder(
            input_dim=input_dim, d_model=d_model,
            n_heads=n_heads, n_layers=encoder_layers,
        )

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()
        return mask

    def forward(self, data_matrix: torch.Tensor,
                token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data_matrix: (batch, n_points, input_dim) data matrix.
            token_ids: (batch, seq_len) target token IDs (teacher forcing).

        Returns:
            logits: (batch, seq_len, vocab_size) output logits.
        """
        batch_size, seq_len = token_ids.shape

        # Encode data
        context = self.data_encoder(data_matrix)  # (batch, d_model)
        memory = context.unsqueeze(1)  # (batch, 1, d_model)

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        tok_emb = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        pos_emb = self.pos_embedding(positions)
        tgt = tok_emb + pos_emb

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, token_ids.device)

        # Decode
        output = self.decoder(tgt, memory, tgt_mask=causal_mask)

        # Project to vocabulary
        logits = self.output_proj(output)
        return logits

    @torch.no_grad()
    def generate_greedy(self, data_matrix: torch.Tensor,
                        max_len: int = 64, bos_token: int = 1,
                        eos_token: int = 2) -> torch.Tensor:
        """Greedy autoregressive generation."""
        batch_size = data_matrix.shape[0]
        device = data_matrix.device

        context = self.data_encoder(data_matrix)
        memory = context.unsqueeze(1)

        generated = torch.full((batch_size, 1), bos_token,
                               dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            seq_len = generated.shape[1]
            positions = torch.arange(seq_len, device=device)
            tok_emb = self.token_embedding(generated) * math.sqrt(self.d_model)
            pos_emb = self.pos_embedding(positions)
            tgt = tok_emb + pos_emb
            causal_mask = self._generate_causal_mask(seq_len, device)
            output = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token).all():
                break

        return generated

    @torch.no_grad()
    def generate_beam(self, data_matrix: torch.Tensor,
                      beam_width: int = 5, max_len: int = 64,
                      bos_token: int = 1, eos_token: int = 2) -> torch.Tensor:
        """Beam search decoding.

        Args:
            data_matrix: (1, n_points, input_dim) single example.
            beam_width: Number of beams.
            max_len: Maximum generation length.

        Returns:
            best_sequence: (1, seq_len) best beam result.
        """
        device = data_matrix.device
        context = self.data_encoder(data_matrix)
        memory = context.unsqueeze(1)  # (1, 1, d_model)

        # Initialize beams: (beam_width, seq_len)
        beams = torch.full((beam_width, 1), bos_token, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_width, device=device)
        beam_scores[1:] = -float('inf')  # Only first beam active initially

        memory_expanded = memory.expand(beam_width, -1, -1)

        for step in range(max_len - 1):
            seq_len = beams.shape[1]
            positions = torch.arange(seq_len, device=device)
            tok_emb = self.token_embedding(beams) * math.sqrt(self.d_model)
            pos_emb = self.pos_embedding(positions)
            tgt = tok_emb + pos_emb
            causal_mask = self._generate_causal_mask(seq_len, device)
            output = self.decoder(tgt, memory_expanded, tgt_mask=causal_mask)
            logits = self.output_proj(output[:, -1, :])  # (beam_width, vocab)
            log_probs = F.log_softmax(logits, dim=-1)

            # Expand scores
            next_scores = beam_scores.unsqueeze(1) + log_probs  # (beam, vocab)
            next_scores = next_scores.view(-1)  # (beam * vocab)

            # Top-k
            topk_scores, topk_ids = next_scores.topk(beam_width, dim=0)
            beam_ids = topk_ids // self.vocab_size
            token_ids = topk_ids % self.vocab_size

            # Update beams
            beams = torch.cat([beams[beam_ids], token_ids.unsqueeze(1)], dim=1)
            beam_scores = topk_scores

            # Check if all beams ended
            if (beams[:, -1] == eos_token).all():
                break

        # Return best beam
        best_idx = beam_scores.argmax()
        return beams[best_idx].unsqueeze(0)
