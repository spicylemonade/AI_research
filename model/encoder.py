"""Set-Transformer encoder for numerical observation data."""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        Q = self.W_q(query).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.W_o(out)


class ISAB(nn.Module):
    """Induced Set Attention Block (Lee et al., 2019)."""
    def __init__(self, d_model, n_heads, n_inducing=32, dropout=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, n_heads, dropout)
        self.mha2 = MultiHeadAttention(d_model, n_heads, dropout)
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B = x.shape[0]
        I = self.inducing_points.expand(B, -1, -1)

        # Inducing points attend to input
        h = self.norm1(I)
        h = I + self.mha1(h, self.norm1(x), self.norm1(x))
        h = h + self.ffn1(self.norm2(h))

        # Input attends to inducing points
        out = self.norm3(x)
        out = x + self.mha2(out, self.norm3(h), self.norm3(h))
        out = out + self.ffn2(self.norm4(out))

        return out


class PMA(nn.Module):
    """Pooling by Multi-head Attention."""
    def __init__(self, d_model, n_heads, n_seeds=16, dropout=0.1):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B = x.shape[0]
        S = self.seed_vectors.expand(B, -1, -1)
        h = S + self.mha(self.norm1(S), self.norm1(x), self.norm1(x))
        h = h + self.ffn(self.norm2(h))
        return h


class SetTransformerEncoder(nn.Module):
    """Set-Transformer encoder for numerical observation data.

    Processes a variable-size set of (x, y) observation points into
    a fixed-size set of K summary vectors.

    Args:
        max_vars: maximum number of input variables
        d_model: model dimension
        n_heads: number of attention heads
        n_isab_layers: number of ISAB layers
        n_inducing: number of inducing points per ISAB
        n_seeds: number of output summary vectors
        dropout: dropout rate
    """
    def __init__(self, max_vars=10, d_model=512, n_heads=8,
                 n_isab_layers=4, n_inducing=32, n_seeds=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(max_vars + 1, d_model)
        self.var_embedding = nn.Embedding(max_vars + 1, d_model)

        self.isab_layers = nn.ModuleList([
            ISAB(d_model, n_heads, n_inducing, dropout)
            for _ in range(n_isab_layers)
        ])
        self.pma = PMA(d_model, n_heads, n_seeds, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, obs_table):
        """
        Args:
            obs_table: (B, N, D+1) observation data, zero-padded if D < max_vars

        Returns:
            (B, K, d_model) summary vectors
        """
        B, N, D = obs_table.shape

        # Project input
        # Pad to max_vars + 1 if needed
        if D < self.input_proj.in_features:
            pad = torch.zeros(B, N, self.input_proj.in_features - D,
                             device=obs_table.device)
            obs_table = torch.cat([obs_table, pad], dim=-1)

        h = self.input_proj(obs_table)  # (B, N, d_model)

        # Add variable-position embedding (same for all points)
        var_ids = torch.arange(min(D, self.var_embedding.num_embeddings),
                              device=obs_table.device)
        var_emb = self.var_embedding(var_ids)  # (D, d_model)
        # Broadcast: mean of variable embeddings added to each point
        h = h + var_emb[:D].mean(dim=0, keepdim=True).unsqueeze(0)

        # ISAB layers
        for isab in self.isab_layers:
            h = isab(h)

        # Pool to fixed-size output
        h = self.pma(h)  # (B, K, d_model)
        h = self.final_norm(h)
        return h
