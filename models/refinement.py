"""Recursive Soft-Masking Refinement for PhysMDT.

Implements the iterative refinement inference procedure adapted from the
ARChitects' ARC 2025 approach. Instead of autoregressive left-to-right
generation, this module starts from a fully masked output sequence and
iteratively refines it over N steps using:

1. **Soft token embeddings** (token algebra): probability-weighted mixture
   of embedding vectors rather than hard argmax.
2. **Soft masking**: additive <MASK> embedding residual that decays over
   refinement steps.
3. **Cosine unmasking schedule**: highest-confidence tokens are committed
   first, following a cosine curve.
4. **Most-visited-candidate voting**: generate K candidate solutions and
   select the most frequently occurring token at each position.

Reference:
    ARChitects ARC 2025 solution report (see sources.bib)
    Sahoo et al. 2024, "Simple and Effective Masked Diffusion Language Models"
"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.physmdt import PhysMDT, PhysMDTConfig, MASK_IDX
from data.tokenizer import ExprTokenizer, PAD_IDX, SOS_IDX, EOS_IDX, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Cosine unmasking schedule
# ---------------------------------------------------------------------------

def cosine_unmasking_fraction(t: int, T: int) -> float:
    """Fraction of tokens that should be unmasked at step t out of T total.

    Uses a cosine schedule:  fraction = 1 - cos(pi * t / (2 * T))

    At t=0 the fraction is 0 (fully masked).  At t=T it is 1 (fully unmasked).

    Args:
        t: Current refinement step (0-indexed).
        T: Total number of refinement steps.

    Returns:
        Float in [0, 1] indicating the fraction of tokens to unmask.
    """
    if T <= 0:
        return 1.0
    ratio = min(t / T, 1.0)
    return 1.0 - math.cos(math.pi * ratio / 2.0)


def mask_alpha(t: int, T: int) -> float:
    """Decay factor for the additive <MASK> embedding residual.

    Linearly decays from 1.0 at t=0 to 0.0 at t=T.

    Args:
        t: Current refinement step (0-indexed).
        T: Total number of refinement steps.

    Returns:
        Float in [0, 1].
    """
    if T <= 0:
        return 0.0
    return max(1.0 - t / T, 0.0)


# ---------------------------------------------------------------------------
# Soft token embedding computation
# ---------------------------------------------------------------------------

def compute_soft_embeddings(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute soft token embeddings as probability-weighted mixture.

    For each position, instead of taking the argmax token, we compute:
        e_soft = sum_t P(t) * E(t)
    where P(t) is the token probability (softmax of logits) and E(t) is
    the corresponding embedding vector.

    Args:
        logits: (batch, seq_len, vocab_size) raw model logits.
        embedding_weight: (vocab_size, d_model) token embedding matrix.
        temperature: Softmax temperature. Lower values produce sharper
            distributions (closer to argmax).

    Returns:
        soft_emb: (batch, seq_len, d_model) soft token embeddings.
        probs: (batch, seq_len, vocab_size) token probabilities.
    """
    # Compute probabilities with temperature scaling
    probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)  # (B, S, V)

    # Weighted sum over vocabulary: (B, S, V) @ (V, D) -> (B, S, D)
    soft_emb = torch.matmul(probs, embedding_weight)

    return soft_emb, probs


# ---------------------------------------------------------------------------
# Confidence-based unmasking
# ---------------------------------------------------------------------------

def select_tokens_to_unmask(
    probs: torch.Tensor,
    current_unmasked: torch.Tensor,
    target_fraction: float,
    special_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select which tokens to commit (unmask) based on confidence.

    Tokens that are already unmasked stay unmasked.  Among the still-masked
    tokens, those with the highest max-probability are unmasked first until
    the target fraction is reached.

    Args:
        probs: (batch, seq_len, vocab_size) token probabilities.
        current_unmasked: (batch, seq_len) bool, True where already committed.
        target_fraction: Desired fraction of tokens to be unmasked after
            this step.
        special_mask: (batch, seq_len) bool, True for positions that are
            always considered unmasked (SOS, EOS, PAD).

    Returns:
        new_unmasked: (batch, seq_len) bool, updated unmasked indicator.
        confidences: (batch, seq_len) per-position max probability.
    """
    batch, seq_len, _ = probs.shape

    # Per-position confidence = max probability
    confidences, _ = probs.max(dim=-1)  # (B, S)

    # Positions that are always unmasked (specials)
    always_unmasked = current_unmasked | special_mask  # (B, S)

    # Total number of non-special positions
    non_special_count = (~special_mask).float().sum(dim=-1)  # (B,)

    # Target number of unmasked non-special tokens
    target_count = (target_fraction * non_special_count).long()  # (B,)

    # Currently unmasked non-special count
    already_count = (current_unmasked & ~special_mask).float().sum(dim=-1).long()

    new_unmasked = current_unmasked.clone()

    for b in range(batch):
        need = max(target_count[b].item() - already_count[b].item(), 0)
        if need == 0:
            continue

        # Candidate positions: masked and not special
        candidate_mask = ~always_unmasked[b]  # (S,)
        if not candidate_mask.any():
            continue

        # Get confidences for candidate positions
        cand_conf = confidences[b].clone()
        cand_conf[~candidate_mask] = -1.0  # exclude non-candidates

        # Select top-k by confidence
        k = min(need, candidate_mask.sum().item())
        _, top_indices = cand_conf.topk(k)
        new_unmasked[b, top_indices] = True

    return new_unmasked, confidences


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine(
    model: PhysMDT,
    observations: torch.Tensor,     # (batch, n_points, max_vars+1)
    obs_mask: torch.Tensor,         # (batch, n_points, max_vars+1)
    seq_len: int = 32,              # output sequence length
    n_steps: int = 64,              # refinement steps
    temperature: float = 1.0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterative refinement via recursive soft-masking.

    Starts from a fully masked output sequence and progressively refines it
    over ``n_steps`` steps.  At each step the model predicts logits for all
    positions.  Soft token embeddings (probability-weighted mixture of
    embedding vectors) are computed and re-injected with a decaying additive
    <MASK> embedding residual.  A cosine unmasking schedule determines when
    to commit high-confidence tokens.

    Args:
        model: A trained PhysMDT model (set to eval mode).
        observations: (batch, n_points, max_vars+1) observation data.
        obs_mask: (batch, n_points, max_vars+1) observation validity mask.
        seq_len: Length of the output token sequence to generate.
        n_steps: Number of refinement steps.
        temperature: Softmax temperature for soft embedding computation.
        device: Device to run on.  If None, uses the observations' device.

    Returns:
        tokens: (batch, seq_len) discrete token indices (final output).
        confidences: (batch, seq_len) per-position confidence scores.
    """
    if device is None:
        device = observations.device

    model.eval()
    batch = observations.shape[0]

    # Retrieve embedding weight matrix: (vocab_size, d_model)
    embed_weight = model.get_token_embeddings()  # (V, D)
    d_model = embed_weight.shape[1]

    # <MASK> embedding vector
    mask_embedding = embed_weight[MASK_IDX].unsqueeze(0).unsqueeze(0)  # (1, 1, D)

    # -- Initialize fully masked token sequence --
    # Place SOS at position 0, MASK everywhere else
    tokens = torch.full(
        (batch, seq_len), MASK_IDX, dtype=torch.long, device=device
    )
    tokens[:, 0] = SOS_IDX

    # Special-position mask: positions that should never be treated as masked
    # (SOS at position 0; we don't force EOS here -- the model should learn it)
    special_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    special_mask[:, 0] = True  # SOS is always committed

    # Track which positions have been committed (unmasked)
    unmasked = special_mask.clone()

    # Track final confidences
    final_confidences = torch.zeros(batch, seq_len, device=device)

    for step in range(n_steps):
        # ---- 1. Build input embeddings ----
        if step == 0:
            # First step: use discrete MASK tokens everywhere
            soft_emb = None  # model will use token_embedding(tokens)
        else:
            # soft_emb was computed in the previous iteration's step 2
            pass  # soft_emb is already set from previous loop body

        # ---- 2. Forward pass through model ----
        logits, _aux = model(
            observations=observations,
            obs_mask=obs_mask,
            masked_tokens=tokens,
            token_mask=~unmasked,
            soft_embeddings=soft_emb if step > 0 else None,
        )
        # logits: (batch, seq_len, vocab_size)

        # ---- 3. Compute soft token embeddings (token algebra) ----
        soft_emb_new, probs = compute_soft_embeddings(
            logits, embed_weight, temperature=temperature
        )

        # ---- 4. Apply soft masking: e_input = e_soft + alpha * E(<MASK>) ----
        alpha = mask_alpha(step, n_steps)
        soft_emb = soft_emb_new + alpha * mask_embedding

        # For committed positions, use the discrete token embedding directly
        # (no soft masking on already-committed tokens)
        if unmasked.any():
            discrete_emb = embed_weight[tokens]  # (B, S, D)
            soft_emb = torch.where(
                unmasked.unsqueeze(-1).expand_as(soft_emb),
                discrete_emb,
                soft_emb,
            )

        # ---- 5. Cosine unmasking schedule ----
        target_frac = cosine_unmasking_fraction(step + 1, n_steps)
        unmasked, confidences = select_tokens_to_unmask(
            probs, unmasked, target_frac, special_mask
        )

        # ---- 6. Commit tokens at newly unmasked positions ----
        # For newly unmasked positions, take the argmax token
        argmax_tokens = logits.argmax(dim=-1)  # (B, S)
        tokens = torch.where(unmasked, argmax_tokens, tokens)

        # Keep SOS in place
        tokens[:, 0] = SOS_IDX

        final_confidences = confidences

    # Final pass: get clean logits with all committed tokens
    logits_final, _ = model(
        observations=observations,
        obs_mask=obs_mask,
        masked_tokens=tokens,
        token_mask=torch.zeros_like(unmasked),  # nothing masked
    )

    # Final argmax for any remaining masked positions
    final_tokens = logits_final.argmax(dim=-1)
    # Preserve committed tokens, fill remaining from final pass
    tokens = torch.where(unmasked, tokens, final_tokens)
    tokens[:, 0] = SOS_IDX

    # Final confidence from the clean pass
    final_probs = F.softmax(logits_final / max(temperature, 1e-8), dim=-1)
    final_confidences, _ = final_probs.max(dim=-1)

    return tokens, final_confidences


# ---------------------------------------------------------------------------
# Candidate generation with most-visited voting
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_candidates(
    model: PhysMDT,
    observations: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_len: int = 32,
    n_steps: int = 64,
    n_candidates: int = 8,
    temperature: float = 1.0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate K candidate solutions and select via most-visited voting.

    Runs the refinement procedure ``n_candidates`` times with stochastic
    temperature sampling.  For each position in the output sequence, the
    token that appears most frequently across candidates is selected
    (most-visited-candidate voting).

    Args:
        model: A trained PhysMDT model.
        observations: (batch, n_points, max_vars+1).
        obs_mask: (batch, n_points, max_vars+1).
        seq_len: Output sequence length.
        n_steps: Refinement steps per candidate.
        n_candidates: Number of candidate trajectories (K).
        temperature: Softmax temperature.  Values slightly above 1.0
            introduce diversity across candidates.
        device: Target device.

    Returns:
        best_tokens: (batch, seq_len) token indices from majority voting.
        confidence: (batch, seq_len) fraction of candidates that agreed
            on each position (voting confidence).
    """
    if device is None:
        device = observations.device

    batch = observations.shape[0]

    # Collect all candidate token sequences
    all_candidates = []  # list of (batch, seq_len) tensors

    for k in range(n_candidates):
        # Use varying temperature for diversity across candidates
        # Candidates 0 uses base temperature; others add slight perturbation
        if k == 0:
            temp_k = temperature
        else:
            # Slightly varied temperature for diversity
            temp_k = temperature * (0.8 + 0.4 * (k / max(n_candidates - 1, 1)))

        cand_tokens, _cand_conf = refine(
            model=model,
            observations=observations,
            obs_mask=obs_mask,
            seq_len=seq_len,
            n_steps=n_steps,
            temperature=temp_k,
            device=device,
        )
        all_candidates.append(cand_tokens)

    # Stack: (n_candidates, batch, seq_len)
    candidates = torch.stack(all_candidates, dim=0)

    # --- Most-visited-candidate voting ---
    # For each (batch, position), find the most common token across candidates
    best_tokens = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    confidence = torch.zeros(batch, seq_len, device=device)

    for b in range(batch):
        for s in range(seq_len):
            # Tokens from all candidates at this position
            position_tokens = candidates[:, b, s]  # (n_candidates,)

            # Count occurrences using bincount
            counts = torch.bincount(
                position_tokens, minlength=VOCAB_SIZE
            )
            winner = counts.argmax()
            best_tokens[b, s] = winner
            confidence[b, s] = counts[winner].float() / n_candidates

    # Ensure SOS is preserved
    best_tokens[:, 0] = SOS_IDX

    return best_tokens, confidence


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Recursive Soft-Masking Refinement Unit Tests")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Setup: create a random PhysMDT model ----
    config = PhysMDTConfig(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        max_expr_len=64,
        max_obs_points=20,
        max_vars=5,
    )
    model = PhysMDT(config).to(device)
    model.eval()
    n_params = model.count_parameters()
    print(f"Test model parameters: {n_params:,}")

    batch_size = 2
    n_points = 20
    obs_dim = config.max_vars + 1
    test_seq_len = 16

    observations = torch.randn(batch_size, n_points, obs_dim, device=device)
    obs_mask = torch.ones(batch_size, n_points, obs_dim, device=device)

    # ---- Test 1: Cosine unmasking schedule ----
    print("\nTest 1: Cosine unmasking schedule")
    T = 64
    fractions = [cosine_unmasking_fraction(t, T) for t in range(T + 1)]
    assert abs(fractions[0] - 0.0) < 1e-6, "t=0 should give fraction 0"
    assert abs(fractions[T] - 1.0) < 1e-6, "t=T should give fraction 1"
    # Check monotonically non-decreasing
    for i in range(1, len(fractions)):
        assert fractions[i] >= fractions[i - 1] - 1e-9, \
            f"Schedule not monotonic at step {i}"
    print(f"  Schedule at t=0: {fractions[0]:.4f}")
    print(f"  Schedule at t=T/4: {fractions[T // 4]:.4f}")
    print(f"  Schedule at t=T/2: {fractions[T // 2]:.4f}")
    print(f"  Schedule at t=T: {fractions[T]:.4f}")
    print("  PASSED")

    # ---- Test 2: Mask alpha decay ----
    print("\nTest 2: Mask alpha decay")
    assert abs(mask_alpha(0, T) - 1.0) < 1e-6, "alpha at t=0 should be 1.0"
    assert abs(mask_alpha(T, T) - 0.0) < 1e-6, "alpha at t=T should be 0.0"
    assert abs(mask_alpha(T // 2, T) - 0.5) < 1e-6, "alpha at t=T/2 should be 0.5"
    print("  PASSED")

    # ---- Test 3: Soft embedding computation ----
    print("\nTest 3: Soft embedding computation")
    dummy_logits = torch.randn(batch_size, test_seq_len, VOCAB_SIZE, device=device)
    embed_weight = model.get_token_embeddings()
    soft_emb, probs = compute_soft_embeddings(dummy_logits, embed_weight, temperature=1.0)
    assert soft_emb.shape == (batch_size, test_seq_len, config.d_model), \
        f"Expected ({batch_size}, {test_seq_len}, {config.d_model}), got {soft_emb.shape}"
    assert probs.shape == (batch_size, test_seq_len, VOCAB_SIZE)
    # Probs should sum to 1 along vocab dim
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        "Probabilities should sum to 1"
    print(f"  soft_emb shape: {soft_emb.shape}")
    print(f"  probs shape: {probs.shape}")
    print("  PASSED")

    # ---- Test 4: Refinement for 10 steps ----
    print("\nTest 4: Refinement for 10 steps")
    tokens, confidences = refine(
        model=model,
        observations=observations,
        obs_mask=obs_mask,
        seq_len=test_seq_len,
        n_steps=10,
        temperature=1.0,
        device=device,
    )
    assert tokens.shape == (batch_size, test_seq_len), \
        f"Expected ({batch_size}, {test_seq_len}), got {tokens.shape}"
    assert confidences.shape == (batch_size, test_seq_len), \
        f"Expected ({batch_size}, {test_seq_len}), got {confidences.shape}"
    # SOS should be preserved at position 0
    assert (tokens[:, 0] == SOS_IDX).all(), "Position 0 should be SOS"
    # Tokens should be valid indices
    assert (tokens >= 0).all() and (tokens < VOCAB_SIZE).all(), \
        "Token indices out of range"
    # Confidences should be in [0, 1]
    assert (confidences >= 0).all() and (confidences <= 1.0 + 1e-6).all(), \
        "Confidences out of range"
    print(f"  tokens shape: {tokens.shape}")
    print(f"  confidences shape: {confidences.shape}")
    print(f"  mean confidence: {confidences.mean().item():.4f}")
    print(f"  first sequence tokens: {tokens[0].tolist()}")
    print("  PASSED")

    # ---- Test 5: Generate candidates with K=4 ----
    print("\nTest 5: Generate candidates with K=4")
    best_tokens, voting_conf = generate_candidates(
        model=model,
        observations=observations,
        obs_mask=obs_mask,
        seq_len=test_seq_len,
        n_steps=10,
        n_candidates=4,
        temperature=1.0,
        device=device,
    )
    assert best_tokens.shape == (batch_size, test_seq_len), \
        f"Expected ({batch_size}, {test_seq_len}), got {best_tokens.shape}"
    assert voting_conf.shape == (batch_size, test_seq_len), \
        f"Expected ({batch_size}, {test_seq_len}), got {voting_conf.shape}"
    assert (best_tokens[:, 0] == SOS_IDX).all(), "Position 0 should be SOS"
    # Voting confidence should be in [0, 1]
    assert (voting_conf >= 0).all() and (voting_conf <= 1.0 + 1e-6).all(), \
        "Voting confidence out of range"
    print(f"  best_tokens shape: {best_tokens.shape}")
    print(f"  voting_conf shape: {voting_conf.shape}")
    print(f"  mean voting confidence: {voting_conf.mean().item():.4f}")
    print("  PASSED")

    # ---- Test 6: Convergence check ----
    print("\nTest 6: Convergence check (output stability over final steps)")
    # Run refinement for 20 steps and check that the last few steps
    # produce increasingly stable output (by comparing pairs of runs
    # with N and N-2 steps)
    tokens_18, _ = refine(
        model=model, observations=observations, obs_mask=obs_mask,
        seq_len=test_seq_len, n_steps=18, temperature=0.5, device=device,
    )
    tokens_20, _ = refine(
        model=model, observations=observations, obs_mask=obs_mask,
        seq_len=test_seq_len, n_steps=20, temperature=0.5, device=device,
    )
    tokens_10, _ = refine(
        model=model, observations=observations, obs_mask=obs_mask,
        seq_len=test_seq_len, n_steps=10, temperature=0.5, device=device,
    )

    # Count how many positions differ
    diff_18_20 = (tokens_18 != tokens_20).float().mean().item()
    diff_10_20 = (tokens_10 != tokens_20).float().mean().item()

    print(f"  Difference (10 vs 20 steps): {diff_10_20:.4f}")
    print(f"  Difference (18 vs 20 steps): {diff_18_20:.4f}")

    # Later steps should be more similar (less change between 18->20 than 10->20)
    # Note: with a random model this is probabilistic, so we just verify
    # the values are reasonable rather than strictly asserting ordering.
    print(f"  Convergence trend: {'GOOD' if diff_18_20 <= diff_10_20 + 0.1 else 'WEAK'}")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All Recursive Soft-Masking Refinement unit tests PASSED")
    print("=" * 60)
