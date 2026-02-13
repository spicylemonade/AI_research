"""
Autoregressive transformer baseline for symbolic equation derivation.

Standard encoder-decoder transformer that takes numerical observation pairs
as input and outputs symbolic equations in prefix notation via autoregressive
decoding. Serves as the performance floor for comparison against PhysMDT.

Architecture: 6 layers, 8 heads, d_model=512

References:
    - vaswani2017attention: Attention is All You Need
    - lample2020deep: Deep Learning for Symbolic Mathematics
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX, MAX_SEQ_LEN


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ObservationEncoder(nn.Module):
    """Encode numerical observation pairs (x, y) into transformer-compatible embeddings.

    Each observation pair is projected through a learnable embedding layer.
    """

    def __init__(self, d_model: int, max_vars: int = 6, n_obs: int = 20):
        super().__init__()
        # Each observation: (x1, ..., x_d, y) -> project to d_model
        self.input_proj = nn.Linear(max_vars + 1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_vars = max_vars

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, n_obs, max_vars + 1) padded observation pairs

        Returns:
            (batch, n_obs, d_model) observation embeddings
        """
        return self.norm(self.input_proj(obs))


class BaselineARTransformer(nn.Module):
    """Encoder-decoder transformer for equation derivation.

    Encoder processes numerical observations.
    Decoder generates equation tokens autoregressively.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 512,
        n_heads: int = 8,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = MAX_SEQ_LEN,
        max_vars: int = 6,
        n_obs: int = 20,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len

        # Encoder: processes observation pairs
        self.obs_encoder = ObservationEncoder(d_model, max_vars, n_obs)
        self.enc_pos = PositionalEncoding(d_model, max_len=n_obs + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # Decoder: generates equation tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dec_pos = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation pairs.

        Args:
            obs: (batch, n_obs, max_vars+1)

        Returns:
            (batch, n_obs, d_model) encoder output
        """
        enc_emb = self.obs_encoder(obs)
        enc_emb = self.enc_pos(enc_emb)
        return self.encoder(enc_emb)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target tokens given encoder memory.

        Args:
            tgt: (batch, tgt_len) target token indices
            memory: (batch, src_len, d_model) encoder output
            tgt_mask: (tgt_len, tgt_len) causal mask

        Returns:
            (batch, tgt_len, vocab_size) logits
        """
        tgt_emb = self.token_embedding(tgt)
        tgt_emb = self.dec_pos(tgt_emb)

        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)

        tgt_key_padding_mask = (tgt == self.pad_idx)

        dec_out = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(dec_out)

    def forward(self, obs: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Args:
            obs: (batch, n_obs, max_vars+1) observation data
            tgt: (batch, tgt_len) target token indices (with BOS prefix)

        Returns:
            (batch, tgt_len, vocab_size) logits
        """
        memory = self.encode(obs)
        return self.decode(tgt, memory)

    @torch.no_grad()
    def generate(self, obs: torch.Tensor, max_len: int = MAX_SEQ_LEN,
                 temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            obs: (batch, n_obs, max_vars+1)
            max_len: maximum generation length
            temperature: sampling temperature (1.0 = greedy after argmax)

        Returns:
            (batch, max_len) generated token indices
        """
        self.eval()
        batch_size = obs.size(0)
        device = obs.device

        memory = self.encode(obs)

        # Start with BOS token
        generated = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decode(generated, memory)
            next_logits = logits[:, -1, :] / temperature

            # Greedy decoding
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Replace with PAD for finished sequences
            next_token = next_token.masked_fill(finished.unsqueeze(1), PAD_IDX)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == EOS_IDX)
            if finished.all():
                break

        # Pad to max_len
        if generated.size(1) < max_len:
            pad = torch.full(
                (batch_size, max_len - generated.size(1)),
                PAD_IDX, dtype=torch.long, device=device
            )
            generated = torch.cat([generated, pad], dim=1)

        return generated[:, :max_len]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline_model(**kwargs) -> BaselineARTransformer:
    """Build the baseline AR model with default hyperparameters."""
    return BaselineARTransformer(**kwargs)


if __name__ == '__main__':
    model = build_baseline_model()
    print(f"Baseline AR Transformer")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  d_model: {model.d_model}")
    print(f"  Vocab size: {model.vocab_size}")

    # Test forward pass
    batch = 4
    obs = torch.randn(batch, 20, 7)  # 20 obs, 6 vars + 1 target
    tgt = torch.randint(0, VOCAB_SIZE, (batch, 30))
    tgt[:, 0] = BOS_IDX

    logits = model(obs, tgt)
    print(f"  Forward pass: obs {obs.shape} + tgt {tgt.shape} -> logits {logits.shape}")

    # Test generation
    gen = model.generate(obs, max_len=30)
    print(f"  Generation: obs {obs.shape} -> generated {gen.shape}")
