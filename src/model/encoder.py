"""
Set Transformer encoder for symbolic regression.
Maps variable-length (x, y) observation sets to a fixed-dimensional latent vector
using IEEE-754 multi-hot bit representation (per NeSymReS, Biggio et al. 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
from typing import Optional


def float_to_ieee754_half(x: float) -> list:
    """Convert a float to IEEE-754 half-precision 16-bit representation."""
    # Clamp to half-precision range
    x = max(min(x, 65504.0), -65504.0)
    # Convert to half precision bytes
    half = np.float16(x)
    # Get the 16 bits
    bits = np.frombuffer(half.tobytes(), dtype=np.uint8)
    # Convert to individual bits (16 total)
    bit_list = []
    for byte in bits:
        for i in range(8):
            bit_list.append((byte >> i) & 1)
    return bit_list[:16]


def batch_float_to_ieee754(values: np.ndarray) -> np.ndarray:
    """Convert array of floats to IEEE-754 half-precision multi-hot encoding.

    Args:
        values: numpy array of shape (...,)

    Returns:
        numpy array of shape (..., 16) with binary values
    """
    # Convert to float16
    half_values = values.astype(np.float16)
    # View as uint16 to get bit patterns
    uint_values = half_values.view(np.uint16)
    # Extract individual bits
    bits = np.zeros((*values.shape, 16), dtype=np.float32)
    for i in range(16):
        bits[..., i] = (uint_values >> i) & 1
    return bits


class MultiHeadAttention(nn.Module):
    """Multi-head attention using PyTorch's optimized scaled_dot_product_attention."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, N, D = query.shape
        _, M, _ = key.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's optimized SDPA (fused kernels on CPU/GPU)
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer, Lee et al. 2019)."""

    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        h = self.norm1(query + self.attn(query, key_value, key_value))
        return self.norm2(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block (Set Transformer, Lee et al. 2019).

    Uses inducing points for O(N*M) complexity instead of O(N^2).
    """

    def __init__(self, dim: int, num_heads: int = 8, num_inducing: int = 32, dropout: float = 0.0):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, dim))
        self.mab1 = MAB(dim, num_heads, dropout=dropout)  # inducing <- input
        self.mab2 = MAB(dim, num_heads, dropout=dropout)  # input <- inducing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        inducing = self.inducing_points.expand(B, -1, -1)
        h = self.mab1(inducing, x)  # [B, M, D] attend to input
        return self.mab2(x, h)       # [B, N, D] attend to compressed


class PMA(nn.Module):
    """Pooling by Multihead Attention (Set Transformer, Lee et al. 2019).

    Produces fixed-size output from variable-size input set.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        seeds = self.seed_vectors.expand(B, -1, -1)
        return self.mab(seeds, x)  # [B, num_seeds, D]


class SetTransformerEncoder(nn.Module):
    """Set Transformer encoder for symbolic regression.

    Takes numerical observations and produces a fixed-size latent vector.
    Uses IEEE-754 half-precision multi-hot encoding for input representation.

    Architecture:
        Input projection -> ISAB layers -> PMA pooling -> Output projection

    Args:
        max_variables: Maximum number of input variables (1-9)
        embed_dim: Embedding dimension (default 256)
        num_heads: Number of attention heads (default 8)
        num_layers: Number of ISAB layers (default 4)
        num_inducing: Number of inducing points per ISAB (default 32)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        max_variables: int = 9,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_inducing: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_variables = max_variables
        self.embed_dim = embed_dim

        # Input: (max_variables + 1) * 16 bits per support point
        # +1 for the output variable y
        input_dim = (max_variables + 1) * 16  # 160 for max_variables=9

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ISAB layers
        self.isab_layers = nn.ModuleList([
            ISAB(embed_dim, num_heads, num_inducing, dropout)
            for _ in range(num_layers)
        ])

        # Pooling to single vector
        self.pma = PMA(embed_dim, num_heads, num_seeds=1, dropout=dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_observations(self, X: np.ndarray, y: np.ndarray, num_vars: int) -> torch.Tensor:
        """Encode raw observations to IEEE-754 multi-hot representation.

        Args:
            X: Input points, shape [N, num_vars]
            y: Output values, shape [N]
            num_vars: Actual number of variables used

        Returns:
            Tensor of shape [N, (max_variables+1)*16]
        """
        N = X.shape[0]

        # Pad X to max_variables columns
        if num_vars < self.max_variables:
            padding = np.zeros((N, self.max_variables - num_vars))
            X_padded = np.concatenate([X, padding], axis=1)
        else:
            X_padded = X[:, :self.max_variables]

        # Concatenate X and y
        Xy = np.concatenate([X_padded, y.reshape(-1, 1)], axis=1)  # [N, max_vars+1]

        # Convert to IEEE-754 multi-hot
        bits = batch_float_to_ieee754(Xy)  # [N, max_vars+1, 16]
        bits_flat = bits.reshape(N, -1)    # [N, (max_vars+1)*16]

        return torch.FloatTensor(bits_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: IEEE-754 encoded observations, shape [B, N, input_dim]
               where N is number of support points

        Returns:
            Latent vector z of shape [B, embed_dim]
        """
        # Project input
        h = self.input_proj(x)  # [B, N, embed_dim]

        # Apply ISAB layers
        for isab in self.isab_layers:
            h = isab(h)

        # Pool to single vector
        z = self.pma(h)  # [B, 1, embed_dim]
        z = z.squeeze(1)  # [B, embed_dim]

        # Output projection
        z = self.output_proj(z)  # [B, embed_dim]

        return z

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    import time

    # For single-sample CPU inference, fewer threads avoids oversubscription
    # overhead.  Tune torch.set_num_threads() for your machine; 1-4 is
    # typically optimal for batch_size=1 on modern multi-core CPUs.
    torch.set_num_threads(1)

    # Test encoder
    encoder = SetTransformerEncoder(max_variables=9, embed_dim=256, num_heads=8, num_layers=4)
    encoder.eval()
    print(f"Encoder parameters: {encoder.count_parameters():,}")

    # Test forward pass timing
    batch_size = 1
    num_points = 200
    input_dim = (9 + 1) * 16  # 160

    x = torch.randn(batch_size, num_points, input_dim)

    # Warmup (multiple runs to stabilise)
    with torch.no_grad():
        for _ in range(10):
            _ = encoder(x)

    # Time it
    n_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            z = encoder(x)
    elapsed = (time.time() - start) / n_runs * 1000

    print(f"Output shape: {z.shape}")
    print(f"Forward pass: {elapsed:.1f}ms (target: <100ms)")
    print(f"Parameters within budget: {encoder.count_parameters() <= 15_000_000}")

    # Test encode_observations helper
    X_test = np.random.randn(100, 3).astype(np.float64)
    y_test = np.random.randn(100).astype(np.float64)
    encoded = encoder.encode_observations(X_test, y_test, num_vars=3)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded values are binary: {torch.all((encoded == 0) | (encoded == 1)).item()}")

    # Test with batch
    encoded_batch = encoded.unsqueeze(0)  # [1, 100, 160]
    z_out = encoder(encoded_batch)
    print(f"Batch output shape: {z_out.shape}")

    # Test float_to_ieee754_half
    bits = float_to_ieee754_half(3.14)
    print(f"IEEE-754 half bits for 3.14: {bits}")
    print(f"Number of bits: {len(bits)}")
