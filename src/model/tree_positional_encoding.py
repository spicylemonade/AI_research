"""Tree-aware 2D positional encoding for symbolic equation tokens.

Adapts the ARC2025 Golden Gate RoPE concept to operate on expression-tree
structure rather than 2D image grids.  Each token in an RPN (reverse-Polish
notation) sequence is assigned a 2D coordinate ``(depth, h_index)`` that
reflects its position in the reconstructed expression tree.  Those
coordinates are then encoded with rotary positional embeddings using four
directional frequency bases – depth, horizontal, diagonal, and
anti-diagonal – following the Golden Gate RoPE formulation.

Usage
-----
>>> from src.model.tree_positional_encoding import (
...     rpn_to_tree_positions,
...     TreeAwareRoPE,
...     get_tree_positional_encoding,
... )

The module is designed as a **drop-in replacement** for standard sinusoidal
or learned positional encodings in :class:`PhysMDT`.  It can be toggled
on/off through the ``use_tree_pe`` configuration flag.

References
----------
* Su et al. 2021 – "RoFormer: Enhanced Transformer with Rotary Position
  Embedding"
* ARChitects ARC2025 – 2D Golden Gate RoPE for grid-structured inputs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Token vocabulary constants (mirrored from src.data.tokenizer)
# ---------------------------------------------------------------------------

PAD_TOKEN: int = 0
BOS_TOKEN: int = 1
EOS_TOKEN: int = 2
MASK_TOKEN: int = 3

# Binary operators: token IDs 4-8
BINARY_OPS: Dict[str, int] = {"+": 4, "-": 5, "*": 6, "/": 7, "^": 8}

# Unary operators: token IDs 9-15
UNARY_OPS: Dict[str, int] = {
    "sqrt": 9,
    "sin": 10,
    "cos": 11,
    "tan": 12,
    "log": 13,
    "exp": 14,
    "abs": 15,
}

# Convenience sets for fast lookup
_BINARY_IDS: frozenset = frozenset(BINARY_OPS.values())
_UNARY_IDS: frozenset = frozenset(UNARY_OPS.values())
_SPECIAL_IDS: frozenset = frozenset({PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN})

# Number of directional frequency bases (depth, horizontal, diagonal,
# anti-diagonal) inspired by the Golden Gate RoPE formulation.
NUM_DIRECTIONS: int = 4

# Default base period for rotary positional embeddings.
_ROPE_BASE: float = 10_000.0


# ---------------------------------------------------------------------------
# Lightweight tree node used only during position assignment
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    """Internal tree node produced during RPN stack simulation."""
    token_idx: int          # position in the original RPN sequence
    depth: int = 0          # depth in the expression tree (root = 0)
    h_index: int = 0        # horizontal index at the node's depth level
    children: Optional[List["_TreeNode"]] = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


# ---------------------------------------------------------------------------
# Core function: RPN → tree positions
# ---------------------------------------------------------------------------

def rpn_to_tree_positions(
    token_ids: Union[List[int], torch.Tensor],
) -> Tuple[List[int], List[int]]:
    """Parse an RPN token sequence and assign 2D tree positions.

    The algorithm simulates the standard RPN evaluation stack:

    * **Operands** (variables, constants, numbers, specials) are pushed onto
      the stack as leaf nodes.
    * **Binary operators** pop two children, create a parent node, and push
      it back.
    * **Unary operators** pop one child, create a parent node, and push it
      back.

    After the tree is fully built, a top-down traversal assigns each node a
    depth (distance from root, root = 0) and a horizontal index (left-to-
    right position among nodes at the same depth).

    Special tokens (PAD, BOS, EOS, MASK) receive depth = 0 and a horizontal
    index equal to their sequential position so they sort stably but do not
    perturb the tree structure.

    Parameters
    ----------
    token_ids : list[int] | torch.Tensor
        1-D sequence of integer token IDs (no batch dimension).

    Returns
    -------
    depths : list[int]
        Per-token depth in the expression tree.
    h_indices : list[int]
        Per-token horizontal index at its depth level.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    seq_len: int = len(token_ids)
    depths: List[int] = [0] * seq_len
    h_indices: List[int] = [0] * seq_len

    # ----- Phase 1: build the tree via RPN stack simulation -----
    stack: List[_TreeNode] = []
    tree_nodes: List[Optional[_TreeNode]] = [None] * seq_len  # index-aligned
    special_indices: List[int] = []  # positions of special / PAD tokens

    for idx, tid in enumerate(token_ids):
        if tid in _SPECIAL_IDS:
            # Special tokens are not part of the expression tree.
            special_indices.append(idx)
            continue

        if tid in _BINARY_IDS:
            node = _TreeNode(token_idx=idx)
            if len(stack) >= 2:
                right = stack.pop()
                left = stack.pop()
                node.children = [left, right]
            elif len(stack) == 1:
                # Gracefully handle malformed sequences
                child = stack.pop()
                node.children = [child]
            stack.append(node)
            tree_nodes[idx] = node

        elif tid in _UNARY_IDS:
            node = _TreeNode(token_idx=idx)
            if len(stack) >= 1:
                child = stack.pop()
                node.children = [child]
            stack.append(node)
            tree_nodes[idx] = node

        else:
            # Operand (variable, constant, number, or unknown) → leaf node
            node = _TreeNode(token_idx=idx)
            stack.append(node)
            tree_nodes[idx] = node

    # If multiple independent sub-trees remain on the stack (rare but
    # possible with malformed/padded sequences), treat the last one as root.
    # We still assign positions to all sub-trees.
    roots: List[_TreeNode] = list(stack)

    # ----- Phase 2: assign depth via top-down BFS / DFS -----
    # We count nodes per depth level to assign horizontal indices.
    depth_counters: Dict[int, int] = {}

    def _assign_positions(node: _TreeNode, depth: int) -> None:
        """Recursive pre-order traversal assigning (depth, h_index)."""
        node.depth = depth

        count = depth_counters.get(depth, 0)
        node.h_index = count
        depth_counters[depth] = count + 1

        for child in node.children:
            _assign_positions(child, depth + 1)

    for root in roots:
        _assign_positions(root, depth=0)

    # ----- Phase 3: write back into flat arrays -----
    for idx in range(seq_len):
        node = tree_nodes[idx]
        if node is not None:
            depths[idx] = node.depth
            h_indices[idx] = node.h_index

    # Assign special tokens (PAD/BOS/EOS/MASK) depth=0, h_index by position
    # so they remain ordered but do not interfere with real tree positions.
    # We use negative h_index values (offset by -seq_len) so they never
    # collide with real tree indices.  The embedding functions handle them
    # transparently since RoPE uses continuous positions.
    for si in special_indices:
        depths[si] = 0
        h_indices[si] = si  # sequential ordering for specials

    return depths, h_indices


# ---------------------------------------------------------------------------
# Rotary embedding helpers
# ---------------------------------------------------------------------------

def _compute_freqs(
    positions: torch.Tensor,
    dim: int,
    base: float = _ROPE_BASE,
) -> torch.Tensor:
    """Compute sinusoidal frequency vectors for given scalar positions.

    Parameters
    ----------
    positions : Tensor, shape ``(seq_len,)``
        Scalar position values (int or float).
    dim : int
        Embedding dimension for this frequency set (should be even).
    base : float
        Base period for frequency calculation.

    Returns
    -------
    freqs : Tensor, shape ``(seq_len, dim)``
        Frequency vectors ready for rotation (interleaved cos/sin pairs).
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    # theta_i = 1 / base^(2i / dim)  for i in [0, half)
    freq_seq = torch.arange(half, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (base ** (2.0 * freq_seq / dim))  # (half,)
    # Outer product: (seq_len,) x (half,) → (seq_len, half)
    angles = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    # Interleave cos and sin → (seq_len, dim)
    cos_vals = angles.cos()
    sin_vals = angles.sin()
    freqs = torch.stack([cos_vals, sin_vals], dim=-1)  # (seq_len, half, 2)
    return freqs.reshape(positions.shape[0], dim)


def _apply_rotary_to_pair(
    x: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D rotation to consecutive dimension pairs.

    Parameters
    ----------
    x : Tensor, shape ``(..., dim)``  where ``dim`` is even.
    cos_theta, sin_theta : Tensor, broadcastable to ``(..., dim // 2)``.

    Returns
    -------
    Rotated tensor of same shape as *x*.
    """
    d = x.shape[-1]
    x1 = x[..., 0::2]  # even dims
    x2 = x[..., 1::2]  # odd dims
    out1 = x1 * cos_theta - x2 * sin_theta
    out2 = x1 * sin_theta + x2 * cos_theta
    # Interleave back
    out = torch.stack([out1, out2], dim=-1).reshape(*x.shape[:-1], d)
    return out


# ---------------------------------------------------------------------------
# TreeAwareRoPE module
# ---------------------------------------------------------------------------

class TreeAwareRoPE(nn.Module):
    """Tree-aware 2D Rotary Positional Embedding.

    Splits the model dimension into four equal groups corresponding to four
    directional frequency bases:

    ======  ==================  ==============
    Group   Direction           Scalar input
    ======  ==================  ==============
    0       Depth               ``d``
    1       Horizontal          ``h``
    2       Diagonal            ``d + h``
    3       Anti-diagonal       ``d − h``
    ======  ==================  ==============

    For each group, standard RoPE frequency vectors are computed from the
    scalar input and applied as 2D rotations to consecutive dimension pairs
    within that group.

    Parameters
    ----------
    d_model : int
        Total embedding / hidden dimension.  Must be divisible by
        ``4 * 2 = 8`` (four directions, each needing even dimension count).
    base : float
        Base period for the rotary frequency computation (default 10 000).

    Notes
    -----
    This module does **not** contain learnable parameters.  It is purely
    functional and computed on-the-fly from the tree-derived positions.
    """

    def __init__(self, d_model: int, base: float = _ROPE_BASE) -> None:
        super().__init__()
        if d_model % (NUM_DIRECTIONS * 2) != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"{NUM_DIRECTIONS * 2} (4 directions × 2 for cos/sin pairs)."
            )
        self.d_model = d_model
        self.d_per_dir = d_model // NUM_DIRECTIONS
        self.base = base

    # ---- internal helpers ------------------------------------------------

    def _freqs_for_direction(
        self,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape ``(seq_len, d_per_dir // 2)``."""
        half = self.d_per_dir // 2
        freq_seq = torch.arange(half, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (self.base ** (2.0 * freq_seq / self.d_per_dir))
        angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # (*, half)
        return angles.cos(), angles.sin()

    # ---- public API ------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        depths: torch.Tensor,
        h_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply tree-aware 2D RoPE rotations to *x*.

        Parameters
        ----------
        x : Tensor, shape ``(batch, seq_len, d_model)``
            Input embeddings (could be queries *or* keys).
        depths : Tensor, shape ``(batch, seq_len)``
            Integer depths from :func:`rpn_to_tree_positions`.
        h_indices : Tensor, shape ``(batch, seq_len)``
            Integer horizontal indices from :func:`rpn_to_tree_positions`.

        Returns
        -------
        Tensor of same shape as *x* with rotary encoding applied.
        """
        B, S, D = x.shape
        assert D == self.d_model, (
            f"Expected last dim {self.d_model}, got {D}"
        )
        d = self.d_per_dir

        # Compute scalar positions for each direction
        # depths, h_indices: (B, S)
        pos_depth = depths          # direction 0
        pos_horiz = h_indices       # direction 1
        pos_diag = depths + h_indices   # direction 2
        pos_anti = depths - h_indices   # direction 3

        # Split x into 4 groups along last dim
        x0 = x[..., 0*d : 1*d]
        x1 = x[..., 1*d : 2*d]
        x2 = x[..., 2*d : 3*d]
        x3 = x[..., 3*d : 4*d]

        # Flatten batch × seq for frequency computation, then reshape back
        def _rotate(chunk: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            """Rotate a (B, S, d_per_dir) chunk by scalar positions (B, S)."""
            flat_pos = positions.reshape(-1)           # (B*S,)
            cos_t, sin_t = self._freqs_for_direction(flat_pos)  # (B*S, d//2)
            cos_t = cos_t.view(B, S, -1)
            sin_t = sin_t.view(B, S, -1)
            return _apply_rotary_to_pair(chunk, cos_t, sin_t)

        r0 = _rotate(x0, pos_depth)
        r1 = _rotate(x1, pos_horiz)
        r2 = _rotate(x2, pos_diag)
        r3 = _rotate(x3, pos_anti)

        return torch.cat([r0, r1, r2, r3], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_per_dir={self.d_per_dir}, "
            f"base={self.base}, num_directions={NUM_DIRECTIONS}"
        )


# ---------------------------------------------------------------------------
# Convenience: get_tree_positional_encoding
# ---------------------------------------------------------------------------

def get_tree_positional_encoding(
    token_ids: torch.Tensor,
    d_model: int,
    base: float = _ROPE_BASE,
) -> torch.Tensor:
    """Compute a tree-aware additive positional encoding tensor.

    Unlike :class:`TreeAwareRoPE` (which is applied **multiplicatively** via
    rotation), this function returns a standard **(batch, seq_len, d_model)**
    additive encoding that can be summed with token embeddings.  This makes
    it a direct drop-in replacement for ``nn.Embedding``-based positional
    encodings used in :class:`PhysMDT`.

    The encoding is constructed by computing sinusoidal features from the
    four directional scalars (depth, horizontal, diagonal, anti-diagonal)
    and concatenating them along the model dimension.

    Parameters
    ----------
    token_ids : Tensor, shape ``(batch, seq_len)``
        Integer token IDs.
    d_model : int
        Model hidden dimension (must be divisible by 8).
    base : float
        Base period for sinusoidal frequencies.

    Returns
    -------
    Tensor of shape ``(batch, seq_len, d_model)``.
    """
    if d_model % (NUM_DIRECTIONS * 2) != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by "
            f"{NUM_DIRECTIONS * 2}."
        )

    B, S = token_ids.shape
    device = token_ids.device
    d_per_dir = d_model // NUM_DIRECTIONS
    half = d_per_dir // 2

    # Pre-compute inverse frequencies (shared across batch)
    freq_seq = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (2.0 * freq_seq / d_per_dir))  # (half,)

    # Collect tree positions for each sample in the batch
    all_encodings: List[torch.Tensor] = []
    for b in range(B):
        depths, h_indices = rpn_to_tree_positions(token_ids[b])
        depth_t = torch.tensor(depths, dtype=torch.float32, device=device)   # (S,)
        horiz_t = torch.tensor(h_indices, dtype=torch.float32, device=device)

        # Scalar positions for each direction
        pos_list = [
            depth_t,                  # depth
            horiz_t,                  # horizontal
            depth_t + horiz_t,        # diagonal
            depth_t - horiz_t,        # anti-diagonal
        ]

        parts: List[torch.Tensor] = []
        for pos in pos_list:
            # angles: (S, half)
            angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
            # interleave sin and cos → (S, d_per_dir)
            enc = torch.stack([angles.sin(), angles.cos()], dim=-1)
            parts.append(enc.reshape(S, d_per_dir))

        # Concatenate directions → (S, d_model)
        all_encodings.append(torch.cat(parts, dim=-1))

    return torch.stack(all_encodings, dim=0)  # (B, S, d_model)


# ---------------------------------------------------------------------------
# Integration helper for PhysMDT
# ---------------------------------------------------------------------------

def build_tree_positions_batch(
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (depths, h_indices) tensors for a batched token ID tensor.

    Parameters
    ----------
    token_ids : Tensor, shape ``(batch, seq_len)``

    Returns
    -------
    depths : LongTensor, shape ``(batch, seq_len)``
    h_indices : LongTensor, shape ``(batch, seq_len)``
    """
    B, S = token_ids.shape
    device = token_ids.device

    depth_list: List[List[int]] = []
    h_list: List[List[int]] = []
    for b in range(B):
        d, h = rpn_to_tree_positions(token_ids[b])
        depth_list.append(d)
        h_list.append(h)

    depths = torch.tensor(depth_list, dtype=torch.long, device=device)
    h_indices = torch.tensor(h_list, dtype=torch.long, device=device)
    return depths, h_indices
