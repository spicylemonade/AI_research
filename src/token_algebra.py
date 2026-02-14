#!/usr/bin/env python3
"""Token algebra: symbolic manipulation in continuous embedding space for PhysMDT.

Implements four capabilities for operating on token embeddings as continuous
vectors rather than discrete symbols:

    1. **Linear interpolation** between symbol embeddings -- smoothly blend two
       token representations (useful for exploring intermediate concepts).
    2. **Symbolic analogy** via vector arithmetic -- the classic *a - b + c*
       operation (e.g. ``F - mul(m, a) + E`` should approximate the kinetic
       energy direction in embedding space).
    3. **Nearest-neighbour projection** back to the discrete token vocabulary
       using cosine similarity.
    4. **Integration hook** into the iterative soft-mask refinement loop so that
       token-algebraic suggestions can be injected as soft-logit biases during
       refinement.

The module is designed around a single ``TokenAlgebra`` class that wraps a
trained ``PhysMDT`` model (specifically its ``token_embedding`` weight matrix).

References:
    - Mikolov et al. (2013) -- king - man + woman = queen analogy framework
    - ARC 2025 ARChitects -- token algebra for discrete puzzle solving
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Special token IDs (consistent with src/tokenizer.py and src/refinement.py)
# ---------------------------------------------------------------------------

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3

# IDs that should never be suggested as nearest-neighbour results because they
# are structural / meta tokens rather than meaningful equation symbols.
_SPECIAL_IDS = frozenset({PAD_ID, BOS_ID, EOS_ID, MASK_ID, 4, 5})
# 4 = [SEP], 5 = [UNK]


# ---------------------------------------------------------------------------
# TokenAlgebra
# ---------------------------------------------------------------------------

class TokenAlgebra:
    """Symbolic manipulation in continuous embedding space.

    This class extracts the ``(vocab_size, d_model)`` embedding weight matrix
    from a trained ``PhysMDT`` model and provides vector-arithmetic operations
    over those embeddings together with a nearest-neighbour decoder that maps
    continuous vectors back to the discrete vocabulary.

    Args:
        model: A ``PhysMDT`` instance (see ``src/phys_mdt.py``).  Only the
            ``token_embedding`` attribute is used; the rest of the model is
            needed only for the ``suggest_alternatives`` refinement hook.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.vocab_size: int = model.vocab_size
        self.d_model: int = model.d_model

        # Cache the embedding weight on whatever device the model lives on.
        # Shape: (vocab_size, d_model).
        self._emb: torch.Tensor = model.token_embedding.weight.detach()

        # Pre-compute L2-normalised embeddings for cosine-similarity queries.
        # We keep this as a buffer that can be refreshed after fine-tuning.
        self._emb_norm: torch.Tensor = self._l2_normalise(self._emb)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-cache embeddings from the model (call after fine-tuning)."""
        self._emb = self.model.token_embedding.weight.detach()
        self._emb_norm = self._l2_normalise(self._emb)

    def get_embedding(self, token_id: int) -> torch.Tensor:
        """Return the raw embedding vector for a single token ID.

        Returns:
            ``(d_model,)`` tensor.
        """
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(
                f"token_id {token_id} out of range [0, {self.vocab_size})"
            )
        return self._emb[token_id]

    # ------------------------------------------------------------------
    # 1. Linear interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        tok_a_id: int,
        tok_b_id: int,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Linearly interpolate between two token embeddings.

        Computes ``(1 - alpha) * emb(a) + alpha * emb(b)``.

        Args:
            tok_a_id: Token ID of the starting symbol.
            tok_b_id: Token ID of the ending symbol.
            alpha: Interpolation factor in ``[0, 1]``.  ``alpha=0`` returns the
                embedding of *a*, ``alpha=1`` returns the embedding of *b*.

        Returns:
            ``(d_model,)`` interpolated embedding vector.

        Raises:
            ValueError: If *alpha* is outside ``[0, 1]`` or token IDs are
                out of range.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        emb_a = self.get_embedding(tok_a_id)
        emb_b = self.get_embedding(tok_b_id)
        return (1.0 - alpha) * emb_a + alpha * emb_b

    # ------------------------------------------------------------------
    # 2. Symbolic analogy via vector arithmetic
    # ------------------------------------------------------------------

    def analogy(
        self,
        a_id: int,
        b_id: int,
        c_id: int,
        exclude_input: bool = True,
    ) -> Tuple[int, float]:
        """Compute the analogy ``a - b + c`` and find the nearest token.

        This mirrors the classic word2vec analogy: *a is to b as ? is to c*.
        The result vector ``emb(a) - emb(b) + emb(c)`` is projected back onto
        the vocabulary via cosine similarity.

        Args:
            a_id: Token ID for *a*.
            b_id: Token ID for *b*.
            c_id: Token ID for *c*.
            exclude_input: If ``True`` (default), the three input token IDs are
                excluded from the candidate set so the answer is non-trivial.

        Returns:
            ``(nearest_token_id, cosine_similarity)`` -- the vocabulary token
            closest to the analogy vector and its cosine similarity score.
        """
        vec = self._analogy_vector(a_id, b_id, c_id)
        exclude = frozenset({a_id, b_id, c_id}) if exclude_input else frozenset()
        results = self._nearest(vec, top_k=1, exclude_ids=_SPECIAL_IDS | exclude)
        token_id, sim = results[0]
        return token_id, sim

    def analogy_weighted(
        self,
        positive: List[Tuple[int, float]],
        negative: List[Tuple[int, float]],
        exclude_input: bool = True,
    ) -> Tuple[int, float]:
        """Generalised weighted analogy: ``sum(w_i * emb(p_i)) - sum(w_j * emb(n_j))``.

        This allows compound expressions such as ``F - 1.0*m*a + 0.5*E`` to be
        represented as separate positive and negative weighted terms.

        Args:
            positive: List of ``(token_id, weight)`` pairs to add.
            negative: List of ``(token_id, weight)`` pairs to subtract.
            exclude_input: Exclude all input token IDs from the result.

        Returns:
            ``(nearest_token_id, cosine_similarity)``.
        """
        device = self._emb.device
        vec = torch.zeros(self.d_model, device=device)
        input_ids: set = set()
        for tid, w in positive:
            vec = vec + w * self.get_embedding(tid)
            input_ids.add(tid)
        for tid, w in negative:
            vec = vec - w * self.get_embedding(tid)
            input_ids.add(tid)

        exclude = frozenset(input_ids) if exclude_input else frozenset()
        results = self._nearest(vec, top_k=1, exclude_ids=_SPECIAL_IDS | exclude)
        return results[0]

    # ------------------------------------------------------------------
    # 3. Nearest-neighbour projection
    # ------------------------------------------------------------------

    def project_nearest(
        self,
        embedding: torch.Tensor,
        top_k: int = 5,
        exclude_special: bool = True,
    ) -> List[Tuple[int, float]]:
        """Project a continuous embedding back to the nearest vocabulary tokens.

        Uses cosine similarity to rank all tokens in the vocabulary and returns
        the *top_k* closest.

        Args:
            embedding: ``(d_model,)`` or ``(1, d_model)`` vector to project.
            top_k: Number of nearest neighbours to return.
            exclude_special: If ``True`` (default), special tokens (``PAD``,
                ``BOS``, ``EOS``, ``MASK``, ``SEP``, ``UNK``) are excluded.

        Returns:
            List of ``(token_id, cosine_similarity)`` tuples sorted by
            decreasing similarity.
        """
        if embedding.dim() == 2:
            embedding = embedding.squeeze(0)
        exclude = _SPECIAL_IDS if exclude_special else frozenset()
        return self._nearest(embedding, top_k=top_k, exclude_ids=exclude)

    # ------------------------------------------------------------------
    # 4. Integration hook into refinement loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def suggest_alternatives(
        self,
        token_ids: torch.Tensor,
        position: int,
        model: nn.Module,
        X: torch.Tensor,
        Y: torch.Tensor,
        top_k: int = 5,
        tree_depths: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, float]]:
        """Suggest alternative tokens at a given position using embedding-space
        neighbourhood combined with model likelihood.

        This is the primary integration hook for the refinement loop.  The
        procedure:

        1. Extract the contextual hidden representation at *position* by
           running a forward pass through the model with the current
           ``token_ids``.
        2. Identify the *top_k_embed* nearest embedding-space neighbours of
           that contextual vector (wider than *top_k* to allow re-ranking).
        3. For each candidate, substitute it into the sequence, run a forward
           pass, and score by the model's own log-probability at that position.
        4. Return the *top_k* candidates sorted by the combined score:

           ``score = beta * cosine_sim + (1 - beta) * log_prob``

           where ``beta = 0.3`` weights embedding proximity vs. model fit.

        Args:
            token_ids: ``(1, seq_len)`` or ``(seq_len,)`` current token ID
                sequence (batch size must be 1).
            position: Index in the sequence to suggest alternatives for.
            model: The ``PhysMDT`` model instance to score candidates through.
            X: ``(1, n_points, n_vars)`` observation inputs.
            Y: ``(1, n_points)`` observation outputs.
            top_k: Number of suggestions to return.
            tree_depths: Optional ``(1, seq_len)`` tree depth per position.

        Returns:
            List of ``(token_id, combined_score)`` sorted by decreasing score.
        """
        beta = 0.3  # weight for embedding similarity vs model log-prob
        top_k_embed = max(top_k * 3, 15)  # broader initial neighbourhood

        # Ensure shapes.
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # (1, seq_len)

        model.eval()

        # ------- Step 1: contextual hidden vector at *position* -------
        # Run the full model to get logits, then extract the contextual
        # representation just before the output projection.
        # We hook into the model internals to get the hidden state.
        hidden = self._get_hidden_at_position(
            token_ids, model, X, Y, tree_depths, position,
        )
        # hidden: (d_model,)

        # ------- Step 2: embedding-space nearest neighbours -------
        current_id = token_ids[0, position].item()
        exclude = _SPECIAL_IDS | {current_id}
        embed_neighbours = self._nearest(
            hidden, top_k=top_k_embed, exclude_ids=exclude,
        )
        # List of (token_id, cosine_sim)

        # ------- Step 3: score each candidate via model log-prob -------
        scored: List[Tuple[int, float]] = []
        for cand_id, cosine_sim in embed_neighbours:
            # Substitute candidate into the sequence.
            modified = token_ids.clone()
            modified[0, position] = cand_id

            logits = model.forward(
                modified, X, Y,
                mask_positions=None,
                tree_depths=tree_depths,
            )
            # logits: (1, seq_len, vocab_size)

            # Log-probability of the candidate at this position.
            log_probs = F.log_softmax(logits[0, position], dim=-1)
            lp = log_probs[cand_id].item()

            # Normalise log-prob to [0, 1] range for combining with cosine sim.
            # Since log_probs are in (-inf, 0], we use sigmoid-like mapping.
            norm_lp = 1.0 / (1.0 + math.exp(-lp))  # maps ~0 for very negative, ~0.5 for lp=0

            combined = beta * cosine_sim + (1.0 - beta) * norm_lp
            scored.append((cand_id, combined))

        # ------- Step 4: sort and return top_k -------
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Refinement-loop bias injection
    # ------------------------------------------------------------------

    def compute_algebra_bias(
        self,
        token_ids: torch.Tensor,
        positions: Optional[List[int]] = None,
        neighbourhood_k: int = 10,
        bias_strength: float = 0.1,
    ) -> torch.Tensor:
        """Compute a soft logit bias tensor that can be added to model logits
        during a refinement step.

        For each specified position, the method takes the current token's
        embedding, finds its *neighbourhood_k* nearest neighbours, and produces
        a bias that gently pushes probability toward those neighbours.  This
        encourages the refinement loop to explore semantically related tokens.

        Args:
            token_ids: ``(batch, seq_len)`` current token IDs.
            positions: List of sequence positions to apply bias at.  If
                ``None``, bias is applied at all non-special positions.
            neighbourhood_k: Number of embedding neighbours to include.
            bias_strength: Scale of the bias logits.

        Returns:
            ``(batch, seq_len, vocab_size)`` bias tensor (same shape as model
            logits).  Add this to logits before softmax during refinement.
        """
        batch, seq_len = token_ids.shape
        device = token_ids.device
        bias = torch.zeros(batch, seq_len, self.vocab_size, device=device)

        if positions is None:
            # All non-special positions.
            positions = [
                p for p in range(seq_len)
                if not _is_all_special(token_ids[:, p])
            ]

        for pos in positions:
            for b in range(batch):
                tid = token_ids[b, pos].item()
                if tid in _SPECIAL_IDS:
                    continue

                emb = self.get_embedding(tid)
                neighbours = self._nearest(
                    emb, top_k=neighbourhood_k, exclude_ids=_SPECIAL_IDS,
                )

                for nbr_id, sim in neighbours:
                    # Bias proportional to similarity.
                    bias[b, pos, nbr_id] = bias_strength * sim

        return bias

    # ------------------------------------------------------------------
    # Embedding-space analysis utilities
    # ------------------------------------------------------------------

    def cosine_similarity(self, id_a: int, id_b: int) -> float:
        """Compute cosine similarity between two token embeddings.

        Returns:
            Scalar cosine similarity in ``[-1, 1]``.
        """
        a = self._emb_norm[id_a]
        b = self._emb_norm[id_b]
        return float(torch.dot(a, b).item())

    def pairwise_similarity(
        self,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity matrix for a set of tokens.

        Args:
            token_ids: List of *N* token IDs.

        Returns:
            ``(N, N)`` symmetric similarity matrix.
        """
        vecs = self._emb_norm[token_ids]  # (N, d_model)
        return torch.mm(vecs, vecs.t())  # (N, N)

    def interpolation_path(
        self,
        tok_a_id: int,
        tok_b_id: int,
        steps: int = 10,
        top_k: int = 1,
    ) -> List[List[Tuple[int, float]]]:
        """Trace a linear path between two tokens and decode at each step.

        Args:
            tok_a_id: Starting token ID.
            tok_b_id: Ending token ID.
            steps: Number of interpolation steps (inclusive of endpoints).
            top_k: Number of nearest neighbours to report per step.

        Returns:
            List of length *steps*, each element a list of ``(token_id, sim)``
            tuples.
        """
        results: List[List[Tuple[int, float]]] = []
        for i in range(steps):
            alpha = i / max(steps - 1, 1)
            emb = self.interpolate(tok_a_id, tok_b_id, alpha)
            neighbours = self.project_nearest(emb, top_k=top_k)
            results.append(neighbours)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise(x: torch.Tensor) -> torch.Tensor:
        """L2-normalise along the last dimension, handling zero vectors."""
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return x / norms

    def _analogy_vector(self, a_id: int, b_id: int, c_id: int) -> torch.Tensor:
        """Compute the raw analogy vector ``emb(a) - emb(b) + emb(c)``."""
        return self.get_embedding(a_id) - self.get_embedding(b_id) + self.get_embedding(c_id)

    def _nearest(
        self,
        vec: torch.Tensor,
        top_k: int = 5,
        exclude_ids: frozenset = frozenset(),
    ) -> List[Tuple[int, float]]:
        """Find the *top_k* nearest vocabulary tokens to *vec* by cosine similarity.

        Args:
            vec: ``(d_model,)`` query vector.
            top_k: Number of results.
            exclude_ids: Token IDs to exclude from results.

        Returns:
            List of ``(token_id, cosine_similarity)`` sorted descending.
        """
        # Normalise query.
        vec_norm = vec / vec.norm().clamp(min=1e-12)  # (d_model,)

        # Cosine similarities with all vocabulary tokens.
        sims = torch.mv(self._emb_norm, vec_norm)  # (vocab_size,)

        # Mask excluded IDs by setting their similarity to -inf.
        if exclude_ids:
            mask_indices = torch.tensor(
                list(exclude_ids), dtype=torch.long, device=sims.device,
            )
            sims[mask_indices] = float('-inf')

        # Top-k.
        k = min(top_k, self.vocab_size - len(exclude_ids))
        k = max(k, 1)
        topk_vals, topk_idx = torch.topk(sims, k)

        results: List[Tuple[int, float]] = []
        for i in range(k):
            results.append((int(topk_idx[i].item()), float(topk_vals[i].item())))
        return results

    def _get_hidden_at_position(
        self,
        token_ids: torch.Tensor,
        model: nn.Module,
        X: torch.Tensor,
        Y: torch.Tensor,
        tree_depths: Optional[torch.Tensor],
        position: int,
    ) -> torch.Tensor:
        """Run the model forward and extract the hidden representation at
        *position* from just before the output projection.

        Returns:
            ``(d_model,)`` hidden vector.
        """
        # Embed tokens.
        h = model.token_embedding(token_ids)  # (1, seq_len, d_model)

        # Apply dual-axis RoPE.
        h = model.rope(h, tree_depth=tree_depths)

        # Encode observations.
        memory = model.obs_encoder(X, Y)  # (1, n_points, d_model)

        # Transformer blocks.
        for block in model.blocks:
            h = block(h, memory)

        # Layer norm (before output projection).
        h = model.output_norm(h)  # (1, seq_len, d_model)

        return h[0, position]  # (d_model,)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _is_all_special(col: torch.Tensor) -> bool:
    """Return True if every element in a 1-D tensor is a special token ID."""
    for val in col.tolist():
        if val not in _SPECIAL_IDS:
            return False
    return True
