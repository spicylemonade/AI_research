#!/usr/bin/env python3
"""Iterative soft-mask refinement inference for PhysMDT.

Implements recursive soft-mask refinement adapted from the ARC 2025 ARChitects
approach, with convergence detection, cold-restart, and candidate tracking for
physics symbolic regression.

Components:
    1. Initial forward pass producing logit distributions
    2. Soft-mask injection at all positions for iterative refinement
    3. Configurable refinement steps (default K=50)
    4. Cold-restart mechanism (2 rounds of K/2 steps)
    5. Convergence detection at >95% confidence threshold
    6. Candidate tracking selecting top-2 most visited equation candidates
    7. Ablation flags to disable each component individually
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Special token IDs (consistent with src/tokenizer.py)
# ---------------------------------------------------------------------------

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RefinementConfig:
    """Configuration for iterative soft-mask refinement.

    Attributes:
        total_steps: Total number of refinement steps K (default 50).
        cold_restart: Whether to use the cold-restart mechanism.  When enabled
            the procedure runs 2 rounds of ``total_steps // 2`` steps.  After
            the first round the sequence is re-initialised to fully-masked
            tokens but the best candidate from round 1 is injected as soft
            logits to bias the second round.
        convergence_detection: Whether to stop early when the model is highly
            confident (>95 % at every position for 3 consecutive steps).
        confidence_threshold: Per-position confidence required for convergence
            (probability of the top token).
        convergence_patience: Number of consecutive steps that must exceed
            the threshold before declaring convergence.
        soft_masking: Whether to feed soft (probability-weighted) embeddings
            back into the model during refinement rather than hard argmax
            token IDs.
        candidate_tracking: Whether to track and return the most frequently
            visited candidate across all refinement steps.
        top_k_candidates: Number of most-visited candidates to keep during
            tracking.
        temperature: Softmax temperature applied to logits when computing
            soft masks.  Lower values sharpen the distribution toward argmax.
        gumbel_noise: Scale of optional Gumbel noise added to logits for
            diversity (0.0 = deterministic).
    """

    total_steps: int = 50
    cold_restart: bool = True
    convergence_detection: bool = True
    confidence_threshold: float = 0.95
    convergence_patience: int = 3
    soft_masking: bool = True
    candidate_tracking: bool = True
    top_k_candidates: int = 2
    temperature: float = 1.0
    gumbel_noise: float = 0.0


# ---------------------------------------------------------------------------
# Candidate bookkeeping
# ---------------------------------------------------------------------------

def _ids_to_key(ids: torch.Tensor) -> Tuple[int, ...]:
    """Convert a 1-D tensor of token IDs to an immutable hashable key."""
    return tuple(ids.tolist())


class CandidateTracker:
    """Tracks the most-visited equation candidates across refinement steps.

    Each time a full sequence is produced (via argmax of the current logits)
    it is recorded.  At the end of the procedure the top-K most frequently
    seen sequences are returned together with their visit counts.
    """

    def __init__(self, top_k: int = 2):
        self.top_k = top_k
        # Per-batch-element counters: list (over batch) of Counter dicts.
        self._counters: List[Counter] = []

    def reset(self, batch_size: int) -> None:
        self._counters = [Counter() for _ in range(batch_size)]

    def record(self, token_ids: torch.Tensor) -> None:
        """Record a batch of candidate sequences.

        Args:
            token_ids: ``(batch, seq_len)`` predicted token IDs.
        """
        batch = token_ids.shape[0]
        if len(self._counters) == 0:
            self.reset(batch)
        for b in range(batch):
            key = _ids_to_key(token_ids[b])
            self._counters[b][key] += 1

    def best(self, device: torch.device) -> torch.Tensor:
        """Return the most-visited candidate for each element in the batch.

        Returns:
            ``(batch, seq_len)`` tensor of token IDs.  If no candidates have
            been recorded the result is all-MASK tokens.
        """
        results = []
        for counter in self._counters:
            if not counter:
                results.append(None)
                continue
            top = counter.most_common(1)[0][0]  # tuple of ints
            results.append(torch.tensor(top, dtype=torch.long, device=device))

        # Determine seq_len from any non-None result
        seq_len = 0
        for r in results:
            if r is not None:
                seq_len = r.shape[0]
                break

        out = []
        for r in results:
            if r is None:
                out.append(torch.full((seq_len,), MASK_ID, dtype=torch.long, device=device))
            else:
                out.append(r)
        return torch.stack(out, dim=0)

    def top_candidates(self, device: torch.device) -> List[List[Tuple[torch.Tensor, int]]]:
        """Return top-K candidates with counts for each batch element.

        Returns:
            List (batch) of lists of ``(token_ids, count)`` tuples, sorted by
            visit count descending.
        """
        all_results = []
        for counter in self._counters:
            top = counter.most_common(self.top_k)
            batch_cands = []
            for key, count in top:
                batch_cands.append(
                    (torch.tensor(key, dtype=torch.long, device=device), count)
                )
            all_results.append(batch_cands)
        return all_results


# ---------------------------------------------------------------------------
# Core refinement class
# ---------------------------------------------------------------------------

class SoftMaskRefinement:
    """Iterative soft-mask refinement inference wrapper for PhysMDT.

    This class wraps a trained ``PhysMDT`` model and runs an iterative
    inference loop that progressively refines a fully-masked sequence.  The
    refinement procedure is the key inference-time innovation adapted from the
    ARC 2025 ARChitects solution.

    Usage::

        model = PhysMDT(...)
        # ... load weights ...
        config = RefinementConfig(total_steps=50)
        refiner = SoftMaskRefinement(model, config)
        pred_ids = refiner.refine(X, Y, seq_len=48)

    Args:
        model: A trained ``PhysMDT`` instance.
        config: A ``RefinementConfig`` controlling the refinement behaviour.
            If ``None`` the defaults are used.
    """

    def __init__(self, model: nn.Module, config: Optional[RefinementConfig] = None):
        self.model = model
        self.config = config if config is not None else RefinementConfig()

        # Cache the token embedding weight matrix for soft-mask injection.
        # This is a (vocab_size, d_model) matrix.
        self._emb_weight: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def refine(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        seq_len: int = 48,
        tree_depths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run iterative soft-mask refinement and return predicted token IDs.

        Args:
            X: ``(batch, n_points, n_vars)`` observation inputs.
            Y: ``(batch, n_points)`` observation outputs.
            seq_len: Length of the equation token sequence to generate.
            tree_depths: Optional ``(batch, seq_len)`` tree depth per position.

        Returns:
            ``(batch, seq_len)`` predicted token IDs.
        """
        self.model.eval()
        cfg = self.config
        device = X.device
        batch = X.shape[0]

        # Lazily cache the embedding weight on the correct device.
        self._emb_weight = self.model.token_embedding.weight.detach()  # (V, d)

        # Set up candidate tracker.
        tracker = CandidateTracker(top_k=cfg.top_k_candidates)
        tracker.reset(batch)

        if cfg.cold_restart:
            # ---------- cold-restart: 2 rounds of K/2 steps ----------
            steps_per_round = cfg.total_steps // 2

            # Round 1
            logits_r1, ids_r1 = self._run_refinement_round(
                X, Y, seq_len, tree_depths, steps_per_round, tracker,
                prior_logits=None,
            )

            # Round 2: re-initialise from scratch but inject round-1
            # best candidate as soft-logit prior.
            prior_logits = logits_r1  # (batch, seq_len, vocab_size)
            logits_r2, ids_r2 = self._run_refinement_round(
                X, Y, seq_len, tree_depths, steps_per_round, tracker,
                prior_logits=prior_logits,
            )

            final_logits = logits_r2
        else:
            # ---------- single run of K steps ----------
            final_logits, _ = self._run_refinement_round(
                X, Y, seq_len, tree_depths, cfg.total_steps, tracker,
                prior_logits=None,
            )

        # Select final output.
        if cfg.candidate_tracking:
            pred_ids = tracker.best(device)
        else:
            pred_ids = final_logits.argmax(dim=-1)
            pred_ids[:, 0] = BOS_ID  # preserve BOS

        return pred_ids

    # ------------------------------------------------------------------
    # Internal: single refinement round
    # ------------------------------------------------------------------

    def _run_refinement_round(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        seq_len: int,
        tree_depths: Optional[torch.Tensor],
        num_steps: int,
        tracker: CandidateTracker,
        prior_logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one refinement round (possibly one of two in cold-restart).

        Args:
            X, Y: Observation data.
            seq_len: Sequence length to generate.
            tree_depths: Optional tree depths.
            num_steps: Number of refinement iterations in this round.
            tracker: ``CandidateTracker`` shared across rounds.
            prior_logits: Soft logits from a previous round used to bias the
                initial state (cold-restart conditioning).  Shape
                ``(batch, seq_len, vocab_size)`` or ``None``.

        Returns:
            ``(logits, pred_ids)`` — the final logits and argmax token IDs
            from this round.
        """
        cfg = self.config
        device = X.device
        batch = X.shape[0]
        vocab_size = self.model.vocab_size

        # Step 0 — initial forward pass with fully-masked input.
        token_ids = torch.full(
            (batch, seq_len), MASK_ID, dtype=torch.long, device=device,
        )
        token_ids[:, 0] = BOS_ID

        logits = self._forward_pass(token_ids, X, Y, tree_depths)
        # logits: (batch, seq_len, vocab_size)

        # If we have a prior from the previous cold-restart round, blend it
        # into the initial logits to bias toward the previous best candidate.
        if prior_logits is not None:
            logits = 0.5 * logits + 0.5 * prior_logits

        # Record the initial argmax candidate.
        pred_ids = logits.argmax(dim=-1)  # (batch, seq_len)
        pred_ids[:, 0] = BOS_ID
        if cfg.candidate_tracking:
            tracker.record(pred_ids)

        # Convergence tracking state.
        consecutive_converged = 0

        # Iterative refinement loop.
        for step in range(num_steps):
            if cfg.soft_masking:
                # -- Soft-mask injection --
                # Build soft embeddings by weighting the token embedding
                # matrix with the softmax probabilities from the current
                # logits.  Position 0 (BOS) is always kept hard.
                token_ids = self._soft_to_hard_ids(logits)
                token_ids[:, 0] = BOS_ID
                logits_new = self._forward_with_soft_input(
                    logits, X, Y, tree_depths,
                )
            else:
                # -- Hard-argmax fallback (ablation: soft_masking=False) --
                token_ids = logits.argmax(dim=-1)
                token_ids[:, 0] = BOS_ID
                logits_new = self._forward_pass(token_ids, X, Y, tree_depths)

            # Add optional Gumbel noise for diversity.
            if cfg.gumbel_noise > 0.0:
                gumbel = -torch.log(-torch.log(
                    torch.rand_like(logits_new).clamp(min=1e-20)
                ).clamp(min=1e-20))
                logits_new = logits_new + cfg.gumbel_noise * gumbel

            logits = logits_new

            # Record current argmax prediction.
            pred_ids = logits.argmax(dim=-1)
            pred_ids[:, 0] = BOS_ID
            if cfg.candidate_tracking:
                tracker.record(pred_ids)

            # -- Convergence detection --
            if cfg.convergence_detection:
                probs = F.softmax(logits / max(cfg.temperature, 1e-8), dim=-1)
                max_probs, _ = probs.max(dim=-1)  # (batch, seq_len)
                # Check positions 1..end (skip BOS which is always confident).
                confident = (max_probs[:, 1:] > cfg.confidence_threshold).all()
                if confident:
                    consecutive_converged += 1
                else:
                    consecutive_converged = 0
                if consecutive_converged >= cfg.convergence_patience:
                    break

        return logits, pred_ids

    # ------------------------------------------------------------------
    # Internal: model forward helpers
    # ------------------------------------------------------------------

    def _forward_pass(
        self,
        token_ids: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        tree_depths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard hard-token forward pass through the model.

        Returns ``(batch, seq_len, vocab_size)`` logits.
        """
        return self.model.forward(
            token_ids, X, Y,
            mask_positions=None,
            tree_depths=tree_depths,
        )

    def _forward_with_soft_input(
        self,
        logits: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        tree_depths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass using soft (probability-weighted) embeddings instead of
        discrete token IDs.

        The key idea: instead of committing to a hard argmax at each position,
        we compute a probability distribution over the vocabulary and form the
        input embedding as a weighted sum of all token embeddings.  This
        preserves gradient-like information that helps the model refine
        uncertain positions.

        Steps:
            1. Convert logits to probabilities via temperature-scaled softmax.
            2. Multiply probabilities by the embedding weight matrix to get
               soft embeddings ``(batch, seq_len, d_model)``.
            3. Override position 0 with the hard BOS embedding.
            4. Feed through the same model forward path (RoPE, cross-attention
               to observations, transformer blocks, output projection).

        Returns:
            ``(batch, seq_len, vocab_size)`` logits from the model.
        """
        cfg = self.config
        batch, seq_len, vocab_size = logits.shape
        device = logits.device

        # Temperature-scaled softmax to produce soft weights.
        probs = F.softmax(logits / max(cfg.temperature, 1e-8), dim=-1)  # (B, S, V)

        # Weighted sum of token embeddings -> soft input embeddings.
        emb_weight = self._emb_weight  # (V, d_model)
        soft_emb = torch.matmul(probs, emb_weight)  # (B, S, d_model)

        # Hard-set BOS at position 0.
        bos_emb = emb_weight[BOS_ID].unsqueeze(0).expand(batch, -1)  # (B, d_model)
        soft_emb[:, 0, :] = bos_emb

        # -- Run the model internals manually to avoid the token_embedding
        # lookup (which requires discrete IDs). --
        model = self.model

        # Apply dual-axis RoPE.
        h = model.rope(soft_emb, tree_depth=tree_depths)

        # Encode observations.
        memory = model.obs_encoder(X, Y)

        # Transformer blocks.
        for block in model.blocks:
            h = block(h, memory)

        # Output projection.
        h = model.output_norm(h)
        out_logits = model.output_proj(h)  # (B, S, V)

        return out_logits

    # ------------------------------------------------------------------
    # Internal: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_to_hard_ids(logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to hard token IDs via argmax.

        This is a convenience used when we need discrete IDs for bookkeeping
        even though the actual forward pass uses soft embeddings.
        """
        return logits.argmax(dim=-1)
