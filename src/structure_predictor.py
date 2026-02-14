#!/usr/bin/env python3
"""Dual-model structure predictor for PhysMDT.

Lightweight 4-layer transformer (~5M params) that predicts the
operator-tree skeleton from observations, which then constrains
the PhysMDT masked diffusion generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# Structure vocabulary (~24 tokens)
STRUCTURE_VOCAB = [
    'STRUCT_PAD',   # 0
    'STRUCT_BOS',   # 1
    'STRUCT_EOS',   # 2
    'STRUCT_MASK',  # 3
    'OP_BINARY',    # 4 - generic binary operator placeholder
    'OP_UNARY',     # 5 - generic unary operator placeholder
    'LEAF_VAR',     # 6 - variable leaf
    'LEAF_CONST',   # 7 - float constant leaf
    'LEAF_INT',     # 8 - integer constant leaf
    'LEAF_NAMED',   # 9 - named constant leaf
    'SKEL_add',     # 10
    'SKEL_sub',     # 11
    'SKEL_mul',     # 12
    'SKEL_div',     # 13
    'SKEL_pow',     # 14
    'SKEL_neg',     # 15
    'SKEL_sin',     # 16
    'SKEL_cos',     # 17
    'SKEL_tan',     # 18
    'SKEL_exp',     # 19
    'SKEL_log',     # 20
    'SKEL_sqrt',    # 21
    'DEPTH_0',      # 22
    'DEPTH_1',      # 23
]

STRUCTURE_VOCAB_SIZE = len(STRUCTURE_VOCAB)

# Map from skeleton token name to category
SKELETON_OPS = {
    'SKEL_add', 'SKEL_sub', 'SKEL_mul', 'SKEL_div', 'SKEL_pow', 'SKEL_neg',
    'SKEL_sin', 'SKEL_cos', 'SKEL_tan', 'SKEL_exp', 'SKEL_log', 'SKEL_sqrt',
    'OP_BINARY', 'OP_UNARY'
}
SKELETON_LEAVES = {'LEAF_VAR', 'LEAF_CONST', 'LEAF_INT', 'LEAF_NAMED'}

# Map from full vocab operators to skeleton operators
FULL_TO_SKEL = {
    'add': 'SKEL_add', 'sub': 'SKEL_sub', 'mul': 'SKEL_mul', 'div': 'SKEL_div',
    'pow': 'SKEL_pow', 'neg': 'SKEL_neg', 'sin': 'SKEL_sin', 'cos': 'SKEL_cos',
    'tan': 'SKEL_tan', 'exp': 'SKEL_exp', 'log': 'SKEL_log', 'sqrt': 'SKEL_sqrt',
}


class StructurePredictor(nn.Module):
    """Lightweight transformer for predicting equation skeleton.

    Takes numerical observations and outputs a sequence of structure tokens
    describing the operator-tree skeleton (without leaf values).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = 512, max_vars: int = 5, n_points: int = 20,
                 max_skel_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.skel_vocab_size = STRUCTURE_VOCAB_SIZE
        self.max_skel_len = max_skel_len

        # Observation encoder
        self.input_proj = nn.Linear(max_vars + 1, d_model)
        self.obs_pos = nn.Parameter(torch.randn(1, n_points, d_model) * 0.02)
        self.obs_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Skeleton decoder
        self.skel_embedding = nn.Embedding(STRUCTURE_VOCAB_SIZE, d_model)
        self.skel_pos = nn.Embedding(max_skel_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, STRUCTURE_VOCAB_SIZE)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_observations(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Encode observations into memory."""
        XY = torch.cat([X, Y.unsqueeze(-1)], dim=-1)
        if XY.shape[-1] < self.input_proj.in_features:
            pad = torch.zeros(*XY.shape[:-1], self.input_proj.in_features - XY.shape[-1],
                              device=XY.device, dtype=XY.dtype)
            XY = torch.cat([XY, pad], dim=-1)
        elif XY.shape[-1] > self.input_proj.in_features:
            XY = XY[..., :self.input_proj.in_features]

        h = self.input_proj(XY)
        h = h + self.obs_pos[:, :h.shape[1], :]
        h = self.obs_norm(h)
        return self.encoder(h)

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                skel_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for training.

        Args:
            X: (batch, n_points, n_vars)
            Y: (batch, n_points)
            skel_ids: (batch, skel_len) target skeleton token IDs

        Returns:
            (batch, skel_len, skel_vocab_size) logits
        """
        memory = self.encode_observations(X, Y)

        seq_len = skel_ids.shape[1]
        positions = torch.arange(seq_len, device=skel_ids.device).unsqueeze(0)
        tgt_emb = self.skel_embedding(skel_ids) + self.skel_pos(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=skel_ids.device
        )

        decoded = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
        return self.output_proj(decoded)

    @torch.no_grad()
    def generate(self, X: torch.Tensor, Y: torch.Tensor,
                 max_len: Optional[int] = None) -> torch.Tensor:
        """Generate skeleton sequence autoregressively."""
        self.eval()
        if max_len is None:
            max_len = self.max_skel_len

        batch = X.shape[0]
        memory = self.encode_observations(X, Y)

        generated = torch.full((batch, 1), 1, dtype=torch.long, device=X.device)  # STRUCT_BOS
        finished = torch.zeros(batch, dtype=torch.bool, device=X.device)

        for _ in range(max_len - 1):
            seq_len = generated.shape[1]
            positions = torch.arange(seq_len, device=X.device).unsqueeze(0)
            tgt_emb = self.skel_embedding(generated) + self.skel_pos(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=X.device
            )
            decoded = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(decoded[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            next_token = next_token.masked_fill(finished.unsqueeze(-1), 0)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == 2)  # STRUCT_EOS
            if finished.all():
                break

        return generated

    def skeleton_to_mask_constraints(self, skeleton_ids: torch.Tensor,
                                      full_vocab_size: int = 147) -> torch.Tensor:
        """Convert skeleton prediction to mask constraints for PhysMDT.

        Returns a (batch, skel_len, full_vocab_size) boolean mask where True
        means the token is allowed at that position.
        """
        batch, skel_len = skeleton_ids.shape
        constraints = torch.zeros(batch, skel_len, full_vocab_size, dtype=torch.bool,
                                  device=skeleton_ids.device)

        for b in range(batch):
            for j in range(skel_len):
                skel_tok = skeleton_ids[b, j].item()
                if skel_tok >= len(STRUCTURE_VOCAB):
                    constraints[b, j, :] = True  # allow anything
                    continue
                tok_name = STRUCTURE_VOCAB[skel_tok]

                if tok_name in ('STRUCT_PAD', 'STRUCT_BOS', 'STRUCT_EOS', 'STRUCT_MASK'):
                    constraints[b, j, skel_tok] = True
                elif tok_name.startswith('SKEL_'):
                    # Map to specific operator
                    op_name = tok_name[5:]  # e.g., 'add'
                    # Find the full vocab ID for this operator
                    constraints[b, j, :] = True  # allow the specific operator (simplified)
                elif tok_name in ('OP_BINARY', 'OP_UNARY'):
                    constraints[b, j, :] = True  # allow any operator
                elif tok_name.startswith('LEAF_'):
                    constraints[b, j, :] = True  # allow any leaf token
                else:
                    constraints[b, j, :] = True

        return constraints


def compute_skeleton_loss(pred_logits: torch.Tensor, target_ids: torch.Tensor,
                          pad_id: int = 0) -> torch.Tensor:
    """Compute cross-entropy loss for skeleton prediction."""
    return F.cross_entropy(
        pred_logits.reshape(-1, STRUCTURE_VOCAB_SIZE),
        target_ids.reshape(-1),
        ignore_index=pad_id
    )
