"""
PhysDiffuser: Masked Discrete Diffusion Transformer for Physics Equation Derivation.

Inspired by LLaDA (Nie et al., 2025) and ARChitects ARC 2025 solution.
Uses masked diffusion with token algebra soft-masking for iterative refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import Counter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.decoder import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokens_to_ids, BINARY_OPS, UNARY_OPS


class DiffusionTransformerLayer(nn.Module):
    """Transformer layer for masked diffusion with cross-attention to encoder."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, key_padding_mask=None):
        # Bidirectional self-attention (no causal mask - this is diffusion, not AR)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_out))

        x = self.norm3(x + self.ff(x))
        return x


class PhysDiffuser(nn.Module):
    """Masked Discrete Diffusion Transformer for equation derivation.

    Forward process: randomly replace equation tokens with MASK token at ratio t ~ U[0,1]
    Reverse process: predict original tokens at masked positions
    Inference: iterative refinement via soft-masking (token algebra)

    Key innovation: token algebra soft-masking adds learnable mask embeddings
    to ALL positions (not just masked ones), allowing the model to express
    uncertainty and iteratively refine its predictions.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB['PAD'])
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Learnable mask embedding for soft-masking (token algebra)
        self.mask_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Timestep embedding (for diffusion schedule)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Memory projection
        self.memory_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Transformer layers (bidirectional - no causal mask)
        self.layers = nn.ModuleList([
            DiffusionTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def mask_tokens(self, token_ids: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion: mask tokens at given ratio.

        Args:
            token_ids: [B, T] token indices
            mask_ratio: fraction of non-PAD tokens to mask (0.0 to 1.0)

        Returns:
            masked_ids: [B, T] with some tokens replaced by MASK
            mask_positions: [B, T] boolean tensor, True where masked
        """
        B, T = token_ids.shape
        # Don't mask PAD, BOS, or EOS
        non_special = (token_ids != VOCAB['PAD']) & (token_ids != VOCAB['BOS']) & (token_ids != VOCAB['EOS'])

        # Random mask
        rand = torch.rand(B, T, device=token_ids.device)
        mask_positions = (rand < mask_ratio) & non_special

        masked_ids = token_ids.clone()
        masked_ids[mask_positions] = VOCAB['MASK']

        return masked_ids, mask_positions

    def forward(
        self,
        token_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        mask_ratio: Optional[float] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Args:
            token_ids: Ground truth token ids [B, T]
            encoder_output: Encoder latent z [B, D]
            mask_ratio: If None, sample t ~ U[0,1] per batch element
            padding_mask: [B, T] True where PAD

        Returns:
            logits: [B, T, vocab_size]
            mask_positions: [B, T] boolean
        """
        B, T = token_ids.shape
        device = token_ids.device

        # Sample masking ratio if not provided (LLaDA-style)
        if mask_ratio is None:
            # Sample t ~ U[0,1] for each batch element
            t = torch.rand(B, 1, device=device)
            # Apply different mask ratios per sample
            non_special = (token_ids != VOCAB['PAD']) & (token_ids != VOCAB['BOS']) & (token_ids != VOCAB['EOS'])
            rand = torch.rand(B, T, device=device)
            mask_positions = (rand < t) & non_special
            masked_ids = token_ids.clone()
            masked_ids[mask_positions] = VOCAB['MASK']
        else:
            masked_ids, mask_positions = self.mask_tokens(token_ids, mask_ratio)
            t = torch.full((B, 1), mask_ratio, device=device)

        # Token + position embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(masked_ids) + self.pos_embed(positions)

        # Add soft mask embedding to ALL positions (token algebra)
        # Scale by mask ratio t to inform model of noise level
        soft_mask = self.mask_embed.expand(B, T, -1) * t.unsqueeze(-1)
        h = h + soft_mask

        # Add timestep information
        time_emb = self.time_embed(t)  # [B, 1, D]
        h = h + time_emb.unsqueeze(1)

        # Prepare memory
        memory = self.memory_proj(encoder_output).unsqueeze(1)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, memory, key_padding_mask=padding_mask)

        # Output
        h = self.output_norm(h)
        logits = self.output_head(h)

        return logits, mask_positions

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss on masked positions only.

        Args:
            logits: [B, T, V]
            targets: [B, T] ground truth token ids
            mask_positions: [B, T] boolean, True where masked

        Returns:
            Scalar loss
        """
        if mask_positions.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Only compute loss on masked positions
        masked_logits = logits[mask_positions]  # [N_masked, V]
        masked_targets = targets[mask_positions]  # [N_masked]

        return F.cross_entropy(masked_logits, masked_targets)

    @torch.no_grad()
    def generate_refinement(
        self,
        encoder_output: torch.Tensor,
        num_steps: int = 50,
        seq_len: int = 30,
        temperature: float = 0.8,
        alpha_start: float = 1.0,
        alpha_end: float = 0.0,
    ) -> List[List[str]]:
        """Iterative refinement via recursive soft-masking.

        Starts from fully masked sequence and iteratively unmasks tokens.
        Uses cosine schedule for mask ratio decay.

        Args:
            encoder_output: [B, D]
            num_steps: Number of refinement steps
            seq_len: Target sequence length
            temperature: Sampling temperature
            alpha_start: Initial mask ratio (1.0 = fully masked)
            alpha_end: Final mask ratio (0.0 = fully unmasked)
        """
        self.eval()
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Start with fully masked sequence: BOS + MASK*seq_len + EOS
        tokens = torch.full((B, seq_len + 2), VOCAB['MASK'], dtype=torch.long, device=device)
        tokens[:, 0] = VOCAB['BOS']
        tokens[:, -1] = VOCAB['EOS']

        # Cosine schedule for mask ratio
        for step in range(num_steps):
            # Cosine decay of mask ratio
            progress = step / max(num_steps - 1, 1)
            mask_ratio = alpha_start + (alpha_end - alpha_start) * (1 - np.cos(progress * np.pi)) / 2

            # Forward pass with current tokens
            positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0).expand(B, -1)
            h = self.token_embed(tokens) + self.pos_embed(positions)

            # Soft mask embedding scaled by current mask ratio
            t = torch.full((B, 1), mask_ratio, device=device)
            soft_mask = self.mask_embed.expand(B, tokens.shape[1], -1) * t.unsqueeze(-1)
            h = h + soft_mask

            time_emb = self.time_embed(t)
            h = h + time_emb.unsqueeze(1)

            memory = self.memory_proj(encoder_output).unsqueeze(1)

            for layer in self.layers:
                h = layer(h, memory)

            h = self.output_norm(h)
            logits = self.output_head(h) / temperature

            # For positions that are MASK, sample new token
            probs = F.softmax(logits, dim=-1)

            # Confidence-based unmasking: unmask tokens with highest confidence first
            is_mask = (tokens == VOCAB['MASK'])

            if is_mask.any():
                # Get max probability for each position
                max_probs, predictions = probs.max(dim=-1)

                # Number of tokens to unmask this step
                n_masked = is_mask.sum(dim=1)
                n_to_unmask = torch.ceil(n_masked.float() / max(num_steps - step, 1)).long()
                n_to_unmask = torch.clamp(n_to_unmask, min=1)

                for b in range(B):
                    mask_idx = is_mask[b].nonzero(as_tuple=True)[0]
                    if len(mask_idx) == 0:
                        continue

                    # Sort by confidence (highest first)
                    confidences = max_probs[b, mask_idx]
                    sorted_idx = confidences.argsort(descending=True)

                    # Unmask top-k
                    k = min(n_to_unmask[b].item(), len(mask_idx))
                    unmask_positions = mask_idx[sorted_idx[:k]]
                    tokens[b, unmask_positions] = predictions[b, unmask_positions]

        # Convert to token strings
        results = []
        for b in range(B):
            seq = []
            for t_id in tokens[b].tolist():
                if t_id == VOCAB['EOS'] or t_id == VOCAB['PAD']:
                    break
                if t_id == VOCAB['BOS'] or t_id == VOCAB['MASK']:
                    continue
                seq.append(ID_TO_TOKEN.get(t_id, '?'))
            results.append(seq)

        return results

    @torch.no_grad()
    def generate_with_voting(
        self,
        encoder_output: torch.Tensor,
        num_trajectories: int = 8,
        num_steps: int = 50,
        seq_len: int = 30,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """Generate with most-visited-candidate selection.

        Runs multiple refinement trajectories and selects the most common result.
        """
        self.eval()
        B = encoder_output.shape[0]

        all_candidates = []
        for traj in range(num_trajectories):
            # Use different temperature for diversity
            temp = temperature * (0.8 + 0.4 * traj / max(num_trajectories - 1, 1))
            result = self.generate_refinement(
                encoder_output, num_steps=num_steps, seq_len=seq_len,
                temperature=temp,
            )
            all_candidates.append(result)

        # Most-visited-candidate selection per batch element
        final_results = []
        for b in range(B):
            candidates = [tuple(all_candidates[t][b]) for t in range(num_trajectories)]
            counter = Counter(candidates)
            most_common = counter.most_common(1)[0][0]
            final_results.append(list(most_common))

        return final_results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    import time

    model = PhysDiffuser(embed_dim=256, num_heads=8, num_layers=4, ff_dim=512)
    print(f"PhysDiffuser parameters: {model.count_parameters():,}")

    # Test training forward
    B, T = 4, 32
    token_ids = torch.randint(4, VOCAB_SIZE, (B, T))
    token_ids[:, 0] = VOCAB['BOS']
    z = torch.randn(B, 256)

    logits, mask_pos = model(token_ids, z)
    loss = model.compute_loss(logits, token_ids, mask_pos)
    print(f"Training loss: {loss.item():.4f}")
    print(f"Masked positions: {mask_pos.sum().item()}/{B*T}")

    # Test inference
    z = torch.randn(1, 256)
    start = time.time()
    result = model.generate_refinement(z, num_steps=20, seq_len=20)
    elapsed = (time.time() - start) * 1000
    print(f"Refinement (20 steps): {elapsed:.0f}ms, result: {result[0][:10]}...")

    # Test voting
    start = time.time()
    result = model.generate_with_voting(z, num_trajectories=4, num_steps=10, seq_len=15)
    elapsed = (time.time() - start) * 1000
    print(f"Voting (4 traj, 10 steps): {elapsed:.0f}ms, result: {result[0][:10]}...")
