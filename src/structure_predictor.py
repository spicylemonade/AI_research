"""
Dual-Model Architecture: Equation Structure Predictor.

A smaller transformer that predicts the structural skeleton of the target
equation (operator tree without leaf values) before the main PhysMDT
fills in variables and constants.

Pipeline:
    1. Structure model predicts equation template: '+ * ? ? * ? ? ?'
    2. PhysMDT fills in template conditioned on observations + structure

Hypothesis: Decomposing structure from content aids complex equation derivation,
analogous to how the ARC 2025 solution used a dedicated shape-prediction model.

References:
    - arc2025architects: dedicated shape-prediction model
    - lample2020deep: tree-structured equation generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import (
    OPERATOR_ARITY, TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE,
    PAD_IDX, BOS_IDX, EOS_IDX, MAX_SEQ_LEN,
    ARITHMETIC_OPS, TRIG_FUNCTIONS, TRANSCENDENTAL,
)


# Structure vocabulary: operators + placeholder '?' for leaves
STRUCT_TOKENS = ['<BOS>', '<EOS>', '<PAD>', '?'] + ARITHMETIC_OPS + TRIG_FUNCTIONS + TRANSCENDENTAL
STRUCT_TOKEN_TO_IDX = {tok: idx for idx, tok in enumerate(STRUCT_TOKENS)}
STRUCT_IDX_TO_TOKEN = {idx: tok for idx, tok in enumerate(STRUCT_TOKENS)}
STRUCT_VOCAB_SIZE = len(STRUCT_TOKENS)
STRUCT_PAD_IDX = STRUCT_TOKEN_TO_IDX['<PAD>']
STRUCT_BOS_IDX = STRUCT_TOKEN_TO_IDX['<BOS>']
STRUCT_EOS_IDX = STRUCT_TOKEN_TO_IDX['<EOS>']
STRUCT_LEAF_IDX = STRUCT_TOKEN_TO_IDX['?']


def equation_to_structure(prefix_tokens: List[str]) -> List[str]:
    """Convert a prefix-notation equation to its structural skeleton.

    Replaces all leaf tokens (variables, constants) with '?' placeholder.

    Example: ['+', '*', 'a', 'm', '^', 'v', '2'] -> ['+', '*', '?', '?', '^', '?', '?']
    """
    all_ops = set(ARITHMETIC_OPS + TRIG_FUNCTIONS + TRANSCENDENTAL)
    structure = []
    for tok in prefix_tokens:
        if tok in all_ops:
            structure.append(tok)
        else:
            structure.append('?')
    return structure


def encode_structure(prefix_tokens: List[str], max_len: int = MAX_SEQ_LEN) -> List[int]:
    """Encode equation structure as token indices."""
    struct = equation_to_structure(prefix_tokens)
    indices = [STRUCT_BOS_IDX]
    for tok in struct:
        idx = STRUCT_TOKEN_TO_IDX.get(tok, STRUCT_LEAF_IDX)
        indices.append(idx)
    indices.append(STRUCT_EOS_IDX)

    # Pad
    while len(indices) < max_len:
        indices.append(STRUCT_PAD_IDX)
    return indices[:max_len]


def decode_structure(indices: List[int]) -> List[str]:
    """Decode structure indices to token list."""
    tokens = []
    for idx in indices:
        if idx == STRUCT_BOS_IDX:
            continue
        if idx == STRUCT_EOS_IDX:
            break
        if idx == STRUCT_PAD_IDX:
            continue
        tokens.append(STRUCT_IDX_TO_TOKEN.get(idx, '?'))
    return tokens


class StructurePredictor(nn.Module):
    """Smaller transformer that predicts equation structure from observations.

    4-layer transformer, lighter than the main PhysMDT.
    Predicts operator tree skeleton without leaf values.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_vars: int = 6,
        n_obs: int = 20,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Observation encoder
        self.obs_proj = nn.Sequential(
            nn.Linear(max_vars + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Structure token embedding
        self.struct_embedding = nn.Embedding(STRUCT_VOCAB_SIZE, d_model,
                                             padding_idx=STRUCT_PAD_IDX)

        # Positional encoding
        self.pos_enc = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder (processes observations)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Transformer decoder (generates structure)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # Output
        self.output_proj = nn.Linear(d_model, STRUCT_VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs: torch.Tensor, struct_tokens: torch.Tensor
                ) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: (batch, n_obs, max_vars+1) observations
            struct_tokens: (batch, seq_len) structure token indices (teacher forcing)

        Returns:
            (batch, seq_len, struct_vocab_size) logits
        """
        # Encode observations
        obs_emb = self.obs_proj(obs)
        enc_out = self.encoder(obs_emb)

        # Decode structure
        struct_emb = self.struct_embedding(struct_tokens)
        pos = torch.arange(struct_tokens.size(1), device=struct_tokens.device)
        struct_emb = struct_emb + self.pos_enc(pos).unsqueeze(0)

        # Causal mask for autoregressive structure generation
        tgt_mask = torch.triu(
            torch.ones(struct_tokens.size(1), struct_tokens.size(1),
                       device=struct_tokens.device), diagonal=1
        ).bool()

        dec_out = self.decoder(
            struct_emb, enc_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(struct_tokens == STRUCT_PAD_IDX),
        )

        return self.output_proj(dec_out)

    @torch.no_grad()
    def predict_structure(self, obs: torch.Tensor, max_len: int = 64
                          ) -> torch.Tensor:
        """Generate structure autoregressively.

        Args:
            obs: (batch, n_obs, max_vars+1)
            max_len: max structure length

        Returns:
            (batch, max_len) structure token indices
        """
        self.eval()
        batch = obs.size(0)
        device = obs.device

        obs_emb = self.obs_proj(obs)
        enc_out = self.encoder(obs_emb)

        # Start with BOS
        struct = torch.full((batch, 1), STRUCT_BOS_IDX, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            struct_emb = self.struct_embedding(struct)
            pos = torch.arange(struct.size(1), device=device)
            struct_emb = struct_emb + self.pos_enc(pos).unsqueeze(0)

            tgt_mask = torch.triu(
                torch.ones(struct.size(1), struct.size(1), device=device), diagonal=1
            ).bool()

            dec_out = self.decoder(struct_emb, enc_out, tgt_mask=tgt_mask)
            logits = self.output_proj(dec_out[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            struct = torch.cat([struct, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if (next_token == STRUCT_EOS_IDX).all():
                break

        # Pad to max_len
        if struct.size(1) < max_len:
            pad = torch.full((batch, max_len - struct.size(1)), STRUCT_PAD_IDX,
                             dtype=torch.long, device=device)
            struct = torch.cat([struct, pad], dim=1)

        return struct[:, :max_len]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = StructurePredictor()
    print(f"Structure Predictor")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Structure vocab size: {STRUCT_VOCAB_SIZE}")

    # Test
    obs = torch.randn(2, 20, 7)
    struct = torch.randint(0, STRUCT_VOCAB_SIZE, (2, 20))
    struct[:, 0] = STRUCT_BOS_IDX
    logits = model(obs, struct)
    print(f"  Forward: {logits.shape}")

    # Test structure extraction
    prefix = ['+', '*', 'a', 'm', '^', 'v', '2']
    struct_tokens = equation_to_structure(prefix)
    print(f"  Structure: {prefix} -> {struct_tokens}")
