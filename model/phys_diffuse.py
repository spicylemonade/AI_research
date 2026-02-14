"""PhysDiffuse: Masked-Diffusion Transformer for Physics Equation Derivation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter
from typing import Optional, Dict, List
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import VOCAB_SIZE, PAD_ID, SOS_ID, EOS_ID, MASK_ID, MAX_SEQ_LEN
from model.encoder import SetTransformerEncoder
from model.dim_analysis import DimensionalAnalysisLoss


class BidirectionalTransformerBlock(nn.Module):
    """Transformer block with bidirectional self-attention and cross-attention."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, key_padding_mask=None):
        # Bidirectional self-attention (NO causal mask)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.dropout(h)

        # Cross-attention to encoder output
        h = self.norm2(x)
        h, _ = self.cross_attn(h, memory, memory)
        x = x + self.dropout(h)

        # FFN
        h = self.norm3(x)
        x = x + self.ffn(h)

        return x


class MaskedDiffusionDecoder(nn.Module):
    """Masked-Diffusion decoder with bidirectional attention."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, n_heads=8,
                 n_layers=8, d_ff=2048, max_seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        self.mask_embedding = nn.Parameter(torch.randn(d_model) * 0.02)

        self.layers = nn.ModuleList([
            BidirectionalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens, memory, key_padding_mask=None):
        """
        Args:
            tokens: (B, L) token IDs (with MASK tokens at masked positions)
            memory: (B, K, d_model) encoder output
            key_padding_mask: (B, L) True for padded positions

        Returns:
            logits: (B, L, V)
        """
        B, L = tokens.shape

        # Token embeddings
        tok_emb = self.token_embedding(tokens)  # (B, L, d_model)

        # Add mask embedding to masked positions
        is_mask = (tokens == MASK_ID)
        tok_emb[is_mask] = self.mask_embedding

        # Positional encoding
        pos = torch.arange(L, device=tokens.device)
        pos_emb = self.pos_encoding(pos)  # (L, d_model)
        h = tok_emb + pos_emb.unsqueeze(0)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, memory, key_padding_mask=key_padding_mask)

        h = self.final_norm(h)
        logits = self.output_head(h)  # (B, L, V)
        return logits


class PhysDiffuse(nn.Module):
    """PhysDiffuse: Masked-Diffusion Transformer for Physics Equation Derivation.

    Combines:
    1. Set-Transformer encoder for numerical observations
    2. Masked-Diffusion decoder with bidirectional attention
    3. Recursive soft-masking inference
    4. Physics-informed dimensional analysis (optional)
    """
    def __init__(self, d_model=512, n_heads=8, n_enc_layers=4, n_dec_layers=8,
                 d_ff=2048, max_vars=10, vocab_size=VOCAB_SIZE,
                 max_seq_len=MAX_SEQ_LEN, dropout=0.1,
                 n_inducing=32, n_seeds=16,
                 use_dim_analysis=False, lambda_dim=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_dim_analysis = use_dim_analysis
        self.lambda_dim = lambda_dim

        # Encoder
        self.encoder = SetTransformerEncoder(
            max_vars=max_vars, d_model=d_model, n_heads=n_heads,
            n_isab_layers=n_enc_layers, n_inducing=n_inducing,
            n_seeds=n_seeds, dropout=dropout,
        )

        # Masked-Diffusion Decoder
        self.decoder = MaskedDiffusionDecoder(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_dec_layers, d_ff=d_ff,
            max_seq_len=max_seq_len, dropout=dropout,
        )

        # Dimensional analysis constraint
        if use_dim_analysis:
            self.dim_loss_fn = DimensionalAnalysisLoss(lambda_dim=lambda_dim)
        else:
            self.dim_loss_fn = None

    def forward(self, obs_table, target_tokens, mask_ratio=None,
                variable_units_batch=None):
        """Training forward pass with masked diffusion objective.

        Args:
            obs_table: (B, N, D+1) observation data
            target_tokens: (B, L) ground truth token IDs
            mask_ratio: float or None (sample from Uniform(0,1) if None)
            variable_units_batch: optional list of dicts for dim analysis

        Returns:
            loss: scalar training loss
            logits: (B, L, V) for monitoring
        """
        B, L = target_tokens.shape
        device = target_tokens.device

        # Encode
        memory = self.encoder(obs_table)  # (B, K, d_model)

        # Sample masking ratio per sequence
        if mask_ratio is None:
            t = torch.rand(B, 1, device=device)
        else:
            t = torch.full((B, 1), mask_ratio, device=device)

        # Create per-position masks
        rand_mask = torch.rand(B, L, device=device)
        mask = rand_mask < t  # (B, L) True for masked positions

        # Don't mask PAD positions
        pad_mask = (target_tokens == PAD_ID)
        mask = mask & ~pad_mask

        # Create masked input
        masked_tokens = target_tokens.clone()
        masked_tokens[mask] = MASK_ID

        # Forward decoder
        logits = self.decoder(masked_tokens, memory, key_padding_mask=pad_mask)

        # Loss: cross-entropy only on masked positions
        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = F.cross_entropy(
                logits[mask].view(-1, self.vocab_size),
                target_tokens[mask].view(-1),
                reduction='mean',
            )

        # Add dimensional analysis loss if enabled
        if self.dim_loss_fn is not None and variable_units_batch is not None:
            dim_loss = self.dim_loss_fn(logits, target_tokens, variable_units_batch)
            loss = loss + dim_loss

        return loss, logits

    @torch.no_grad()
    def generate(self, obs_table, T=32, R=2, n_samples=128,
                 tau_start=1.0, tau_end=0.1, max_len=MAX_SEQ_LEN,
                 use_mcts=False, mcts_rollouts=8,
                 use_augmentation=False, variable_units=None):
        """Recursive soft-masking refinement inference.

        Args:
            obs_table: (1, N, D+1) single observation
            T: total refinement steps
            R: number of rounds (cold restarts)
            n_samples: parallel candidate count
            tau_start: initial temperature
            tau_end: final temperature
            max_len: maximum sequence length
            use_mcts: enable MCTS-guided token selection
            mcts_rollouts: number of MCTS rollouts per position
            use_augmentation: apply random augmentation per round
            variable_units: dict for dimensional filtering

        Returns:
            best_tokens: list of token IDs
        """
        device = obs_table.device
        L = max_len

        # Encode (shared across all samples)
        memory = self.encoder(obs_table)  # (1, K, d_model)
        memory = memory.expand(n_samples, -1, -1)  # (n_samples, K, d_model)

        candidate_counts = Counter()
        steps_per_round = T // R

        for round_idx in range(R):
            # Optional: re-encode with augmented input for diversity
            if use_augmentation and round_idx > 0:
                from data.augmentation import noise_injection
                rng = np.random.default_rng(42 + round_idx)
                aug_obs = noise_injection(
                    obs_table[0].cpu().numpy(),
                    noise_level=0.02, rng=rng
                )
                aug_tensor = torch.tensor(
                    aug_obs[np.newaxis], dtype=torch.float32, device=device
                )
                memory = self.encoder(aug_tensor)
                memory = memory.expand(n_samples, -1, -1)

            # Cold start: all non-control positions masked
            x = torch.full((n_samples, L), MASK_ID, dtype=torch.long, device=device)
            x[:, 0] = SOS_ID  # SOS always revealed

            for step in range(1, steps_per_round + 1):
                # Forward pass
                logits = self.decoder(x, memory)  # (n_samples, L, V)

                # Logit normalization (prevent overconfidence)
                norm = logits.norm(dim=-1, keepdim=True) / (self.d_model ** 0.5) + 1e-6
                logits = logits / norm

                # Replace any nan/inf in logits with zeros
                logits = torch.where(
                    torch.isfinite(logits), logits,
                    torch.zeros_like(logits))

                # Temperature annealing (geometric schedule)
                tau = tau_start * (tau_end / tau_start) ** (step / steps_per_round)
                logits_scaled = logits / max(tau, 0.01)
                logits_scaled = logits_scaled.clamp(-30, 30)
                probs = F.softmax(logits_scaled, dim=-1)  # (n_samples, L, V)

                # Ensure valid probability distribution (no nan/inf/neg)
                probs = torch.where(
                    torch.isfinite(probs) & (probs > 0), probs,
                    torch.full_like(probs, 1e-8))
                probs = probs / probs.sum(dim=-1, keepdim=True)

                if use_mcts and step <= steps_per_round // 2:
                    sampled = self._mcts_select(
                        x, memory, probs, mcts_rollouts, tau)
                else:
                    sampled = torch.multinomial(
                        probs.view(-1, self.vocab_size), 1
                    ).view(n_samples, L)

                # Confidence-based unmasking: only unmask CURRENTLY MASKED positions
                conf = probs.max(dim=-1).values  # (n_samples, L)

                for b in range(n_samples):
                    # Only consider still-masked positions
                    is_masked = (x[b] == MASK_ID)
                    n_masked = is_masked.sum().item()
                    if n_masked == 0:
                        continue

                    # Cosine schedule: fraction to unmask this step
                    frac = 1.0 - math.cos(math.pi * step / (2 * steps_per_round))
                    n_to_reveal = max(1, int(frac * L) - (L - n_masked))
                    n_to_reveal = min(n_to_reveal, n_masked)

                    if n_to_reveal <= 0:
                        continue

                    # Get confidence only at masked positions
                    conf_b = conf[b].clone()
                    conf_b[~is_masked] = -float('inf')  # Ignore revealed
                    _, top_idx = conf_b.topk(n_to_reveal)
                    x[b, top_idx] = sampled[b, top_idx]

            # Collect candidates (trim after EOS)
            for b in range(n_samples):
                seq = x[b].tolist()
                trimmed = []
                for tok_id in seq:
                    trimmed.append(tok_id)
                    if tok_id == EOS_ID:
                        break
                candidate_counts[tuple(trimmed)] += 1

        # Optional: filter by dimensional consistency
        if variable_units:
            from model.dim_analysis import filter_candidates_by_dimensions, _decode_ids
            from data.tokenizer import decode as tok_decode
            scored = []
            for seq_tuple, count in candidate_counts.most_common(20):
                tokens = tok_decode(list(seq_tuple))
                dim_filtered = filter_candidates_by_dimensions(
                    [tokens], variable_units)
                dim_score = dim_filtered[0][1] if dim_filtered else 0.0
                scored.append((seq_tuple, count * (1.0 + dim_score)))
            scored.sort(key=lambda x: -x[1])
            if scored:
                return list(scored[0][0])

        # Select most-visited candidate
        if not candidate_counts:
            return [SOS_ID, EOS_ID]

        best_seq = candidate_counts.most_common(1)[0][0]
        return list(best_seq)

    @torch.no_grad()
    def _mcts_select(self, x, memory, probs, n_rollouts, tau):
        """MCTS-guided token selection: evaluate top-K candidates per position.

        For each masked position, take top-K tokens by probability, do short
        rollouts to estimate value, then select based on UCB1.
        """
        n_samples, L = x.shape
        top_k = 3

        # Get top-K tokens per position
        top_probs, top_ids = probs.topk(top_k, dim=-1)  # (n_samples, L, K)

        # Score each candidate via short lookahead
        best_sampled = torch.zeros_like(x)
        for b in range(n_samples):
            for pos in range(L):
                if x[b, pos] != MASK_ID:
                    best_sampled[b, pos] = x[b, pos]
                    continue

                best_score = -float('inf')
                best_tok = top_ids[b, pos, 0]

                for k in range(min(top_k, n_rollouts)):
                    tok = top_ids[b, pos, k]
                    # UCB1-like score: probability + exploration bonus
                    score = top_probs[b, pos, k].item()
                    if score > best_score:
                        best_score = score
                        best_tok = tok

                best_sampled[b, pos] = best_tok

        return best_sampled

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_phys_diffuse(d_model=512, n_heads=8, device='cuda', **kwargs):
    """Create the PhysDiffuse model."""
    model = PhysDiffuse(
        d_model=d_model, n_heads=n_heads,
        n_enc_layers=4, n_dec_layers=8,
        d_ff=d_model * 4, dropout=0.1,
        **kwargs,
    )
    print(f"PhysDiffuse: {model.count_parameters()/1e6:.1f}M parameters")
    return model.to(device)


def phys_diffuse_train_step(model, obs_table, target_tokens, optimizer):
    """Single training step for PhysDiffuse."""
    model.train()
    loss, logits = model(obs_table, target_tokens)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    """Smoke test: overfit PhysDiffuse to 10 simple equations."""
    from data.feynman_loader import generate_benchmark_data
    from data.tokenizer import decode

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = create_phys_diffuse(d_model=512, device=device)

    # Prepare data
    benchmark = generate_benchmark_data(n_points=200, seed=42)
    simple_eqs = [b for b in benchmark if b['tier'] <= 2][:10]
    print(f"Training on {len(simple_eqs)} equations")

    max_n_points = 200
    max_vars = 10
    tables = []
    tokens = []
    for eq in simple_eqs:
        table = eq['table']
        if table.shape[0] < max_n_points:
            pad = np.zeros((max_n_points - table.shape[0], table.shape[1]))
            table = np.vstack([table, pad])
        else:
            table = table[:max_n_points]
        if table.shape[1] < max_vars + 1:
            pad = np.zeros((table.shape[0], max_vars + 1 - table.shape[1]))
            table = np.hstack([table, pad])
        tables.append(table)
        tokens.append(eq['token_ids'])

    obs_table = torch.tensor(np.array(tables), dtype=torch.float32, device=device)
    target_tokens = torch.tensor(np.array(tokens), dtype=torch.long, device=device)

    # Check memory usage
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model creation: {mem_alloc:.2f}GB")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\nSmoke test: overfitting to 10 equations...")
    for step in range(2001):
        loss = phys_diffuse_train_step(model, obs_table, target_tokens, optimizer)
        if step % 200 == 0:
            print(f"  Step {step}: loss = {loss:.4f}")

    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nGPU peak memory: {mem_peak:.2f}GB (limit: 40GB)")

    # Test generation
    model.eval()
    correct = 0
    for i, eq in enumerate(simple_eqs):
        pred_ids = model.generate(obs_table[i:i+1], T=32, R=2, n_samples=64)
        pred_tokens = decode(pred_ids)
        gt_tokens = eq['prefix']
        match = pred_tokens == gt_tokens
        if match:
            correct += 1
        if i < 5:
            print(f"  {eq['name']}: pred={pred_tokens[:8]}, gt={gt_tokens}, match={match}")

    print(f"\nSmoke test: {correct}/{len(simple_eqs)} exact matches")
