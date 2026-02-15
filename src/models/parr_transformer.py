"""
Physics-Aware Recursive Refinement (PARR) Transformer.
Novel architecture combining autoregressive generation with iterative
bidirectional refinement, ConvSwiGLU FFN, and Token Algebra.

Training: Teacher-forced decoding with random noise injection and K refinement passes.
Generation: Autoregressive first pass → K bidirectional refinement passes.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.equation_templates import VOCAB_SIZE, EQUATION_VOCAB, MAX_EQ_LENGTH


class ConvSwiGLU(nn.Module):
    """Convolutional SwiGLU feed-forward block (from URM).

    Uses depthwise 1D convolutions with gated activation for stronger
    nonlinearity and local pattern capture in equation token sequences.
    """

    def __init__(self, d_model, d_ff, kernel_size=5, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        padding = kernel_size // 2
        self.conv_gate = nn.Conv1d(d_ff, d_ff, kernel_size, padding=padding, groups=d_ff)
        self.conv_up = nn.Conv1d(d_ff, d_ff, kernel_size, padding=padding, groups=d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.conv_gate(self.w_gate(x).transpose(1, 2)).transpose(1, 2)
        up = self.conv_up(self.w_up(x).transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.w_down(F.silu(gate) * up))


class TokenAlgebra(nn.Module):
    """Token algebra layer for continuous-space refinement.

    Adds a learned refinement signal modulated by step progress,
    allowing the model to smoothly transition from coarse to fine predictions.
    """

    def __init__(self, d_model):
        super().__init__()
        self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.blend_gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.Sigmoid(),
        )

    def forward(self, token_embeds, step_fraction):
        B, T, D = token_embeds.shape
        step_input = torch.full((B, T, 1), step_fraction,
                                device=token_embeds.device, dtype=token_embeds.dtype)
        gate_input = torch.cat([token_embeds, step_input], dim=-1)
        gate = self.blend_gate(gate_input)
        mask_strength = 1.0 - step_fraction
        mask_signal = self.mask_embed * mask_strength
        return token_embeds + gate * mask_signal


class PARRRefinementBlock(nn.Module):
    """Shared refinement block applied K times in the PARR loop.

    Bidirectional self-attention + cross-attention + ConvSwiGLU + Token Algebra.
    """

    def __init__(self, d_model, n_heads, d_ff, kernel_size=5, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        self.ffn = ConvSwiGLU(d_model, d_ff, kernel_size, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.token_algebra = TokenAlgebra(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask=None, step_fraction=0.0):
        # Bidirectional self-attention
        residual = x
        x = self.self_attn_norm(x)
        x_attn, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(x_attn)

        # Cross-attention to encoder
        residual = x
        x = self.cross_attn_norm(x)
        x_cross, self._last_attn_weights = self.cross_attn(
            x, memory, memory, key_padding_mask=memory_key_padding_mask
        )
        x = residual + self.dropout(x_cross)

        # ConvSwiGLU FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        # Token algebra
        x = self.token_algebra(x, step_fraction)
        return x


class NumericalEncoderPARR(nn.Module):
    """Encoder for numerical observations."""

    def __init__(self, d_model=512, max_obs=50, max_vars=7):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(max_vars, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embedding = nn.Embedding(max_obs, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, observations):
        B, N, V = observations.shape
        sign = torch.sign(observations)
        log_obs = sign * torch.log1p(torch.abs(observations))
        projected = self.value_proj(log_obs)
        pos_ids = torch.arange(N, device=observations.device).unsqueeze(0).expand(B, N)
        encoded = self.norm(projected + self.pos_embedding(pos_ids))
        padding_mask = (observations.abs().sum(dim=-1) == 0)
        return encoded, padding_mask


class PARRTransformer(nn.Module):
    """Physics-Aware Recursive Refinement Transformer.

    Architecture:
    1. Numerical encoder (6-layer Transformer)
    2. Autoregressive decoder (6-layer causal Transformer) for initial prediction
    3. Shared refinement block applied K times (bidirectional, ConvSwiGLU, Token Algebra)
    4. TBPTL for memory-efficient refinement training

    Training: teacher-forced AR decoder + K refinement steps on noisy/corrupted input
    Generation: AR greedy decode → K bidirectional refinement passes
    """

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
        d_ff=2048,
        K=8,
        K_bp=3,
        kernel_size=5,
        dropout=0.1,
        max_obs=50,
        max_vars=7,
        vocab_size=VOCAB_SIZE,
        max_eq_len=MAX_EQ_LENGTH,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_eq_len = max_eq_len
        self.K = K
        self.K_bp = K_bp

        # Encoder
        self.numerical_encoder = NumericalEncoderPARR(d_model, max_obs, max_vars)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_enc_layers
        )

        # Autoregressive decoder (causal)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.ar_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_dec_layers
        )

        # Shared refinement block (applied K times)
        self.refinement_block = PARRRefinementBlock(
            d_model, n_heads, d_ff, kernel_size, dropout
        )

        # Embeddings
        self.eq_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_eq_len, d_model)
        self.step_embedding = nn.Embedding(K + 1, d_model)

        # Output
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Causal mask (cached)
        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(max_eq_len),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, observations):
        enc_input, padding_mask = self.numerical_encoder(observations)
        memory = self.transformer_encoder(
            enc_input, src_key_padding_mask=padding_mask
        )
        return memory, padding_mask

    def _embed_equation(self, eq_tokens):
        """Embed equation tokens with position encoding."""
        B, T = eq_tokens.shape
        eq_embeds = self.eq_embedding(eq_tokens) * math.sqrt(self.d_model)
        pos_ids = torch.arange(T, device=eq_tokens.device).unsqueeze(0).expand(B, T)
        return eq_embeds + self.pos_embedding(pos_ids)

    def forward(self, observations, eq_tokens, use_tbptl=True):
        """Training forward pass.

        Phase 1: Teacher-forced AR decoder produces initial representation.
        Phase 2: K refinement steps on corrupted version of the target.

        Returns (ar_logits, refinement_logits_list, corruption_mask).
        """
        B, T = eq_tokens.shape
        device = eq_tokens.device

        memory, enc_mask = self.encode(observations)

        # --- Phase 1: AR decoder (teacher-forced) ---
        eq_embeds = self._embed_equation(eq_tokens)
        tgt_mask = self.causal_mask[:T, :T]
        pad_mask = (eq_tokens == EQUATION_VOCAB["[PAD]"])

        ar_output = self.ar_decoder(
            eq_embeds, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=enc_mask,
        )
        ar_logits = self.output_proj(self.output_norm(ar_output))

        # --- Phase 2: Refinement on corrupted input ---
        # Corrupt the target sequence (random token replacement)
        corrupt_rate = torch.rand(B, 1, device=device) * 0.5  # 0-50% corruption
        corrupt_prob = torch.rand(B, T, device=device)
        is_corrupted = corrupt_prob < corrupt_rate
        is_corrupted = is_corrupted & ~pad_mask
        # Don't corrupt EQ_START
        is_corrupted[:, 0] = False

        corrupted_tokens = eq_tokens.clone()
        # Replace corrupted tokens with random vocabulary tokens
        random_tokens = torch.randint(0, self.vocab_size, (B, T), device=device)
        corrupted_tokens[is_corrupted] = random_tokens[is_corrupted]

        eq_embeds = self._embed_equation(corrupted_tokens)

        refinement_logits = []
        for k in range(self.K):
            step_frac = k / max(self.K - 1, 1)

            # TBPTL: detach early steps
            if use_tbptl and k == self.K - self.K_bp and k > 0:
                eq_embeds = eq_embeds.detach().requires_grad_(True)

            step_emb = self.step_embedding(
                torch.full((B, T), k, device=device, dtype=torch.long)
            )
            step_input = eq_embeds + step_emb

            eq_embeds = self.refinement_block(
                step_input, memory, enc_mask, step_fraction=step_frac
            )

            logits = self.output_proj(self.output_norm(eq_embeds))
            refinement_logits.append(logits)

        return ar_logits, refinement_logits, is_corrupted

    def compute_loss(self, ar_logits, refinement_logits, targets, is_corrupted):
        """Combined AR + refinement loss."""
        pad_idx = EQUATION_VOCAB["[PAD]"]
        B, T = targets.shape

        # --- AR loss (standard cross-entropy, shifted) ---
        # Predict next token: logits[t] predicts targets[t+1]
        ar_loss = F.cross_entropy(
            ar_logits[:, :-1].reshape(-1, self.vocab_size),
            targets[:, 1:].reshape(-1),
            ignore_index=pad_idx,
        )

        # --- Refinement loss (predict correct tokens at all positions) ---
        K = len(refinement_logits)
        ref_loss = 0.0
        for k, logits in enumerate(refinement_logits):
            weight = (k + 1) / K
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=pad_idx,
            )
            ref_loss += weight * loss
        ref_loss /= K

        # Combined: AR loss + refinement loss
        return ar_loss + ref_loss

    @torch.no_grad()
    def generate(self, observations, K=None, temperature=1.0):
        """Generate equations: AR first pass → K refinement passes."""
        if K is None:
            K = self.K

        B = observations.shape[0]
        device = observations.device

        memory, enc_mask = self.encode(observations)

        # --- Phase 1: Autoregressive generation ---
        eq_tokens = torch.full(
            (B, self.max_eq_len), EQUATION_VOCAB["[PAD]"],
            dtype=torch.long, device=device
        )
        eq_tokens[:, 0] = EQUATION_VOCAB["[EQ_START]"]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(1, self.max_eq_len):
            eq_embeds = self._embed_equation(eq_tokens[:, :t])
            tgt_mask = self.causal_mask[:t, :t]

            dec_output = self.ar_decoder(
                eq_embeds, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=enc_mask,
            )

            logits = self.output_proj(self.output_norm(dec_output[:, -1:])) / temperature
            next_token = logits.argmax(dim=-1).squeeze(-1)

            # Only write tokens for unfinished samples
            eq_tokens[:, t] = torch.where(finished, eq_tokens[:, t], next_token)
            finished = finished | (next_token == EQUATION_VOCAB["[EQ_END]"])

            if finished.all():
                break

        # --- Phase 2: Bidirectional refinement ---
        eq_embeds = self._embed_equation(eq_tokens)

        for k in range(K):
            step_frac = k / max(K - 1, 1)
            step_emb = self.step_embedding(
                torch.full((B, self.max_eq_len), min(k, self.K - 1),
                           device=device, dtype=torch.long)
            )
            step_input = eq_embeds + step_emb

            eq_embeds = self.refinement_block(
                step_input, memory, enc_mask, step_fraction=step_frac
            )

            # Update tokens from refinement logits
            logits = self.output_proj(self.output_norm(eq_embeds)) / temperature
            refined_tokens = logits.argmax(dim=-1)

            # Only update non-PAD positions (preserve structure)
            non_pad = (eq_tokens != EQUATION_VOCAB["[PAD]"])
            eq_tokens[non_pad] = refined_tokens[non_pad]

            # Re-embed for next iteration
            eq_embeds = self._embed_equation(eq_tokens)

        return eq_tokens

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_parr_model(d_model=512, K=4, device='cuda'):
    """Create the PARR model."""
    model = PARRTransformer(
        d_model=d_model, n_heads=8, n_enc_layers=6, n_dec_layers=6,
        d_ff=2048, K=K, K_bp=min(3, K),
        kernel_size=5, dropout=0.1,
    )
    print(f"PARR model: {model.count_parameters() / 1e6:.1f}M parameters (K={K} refinement steps)")
    return model.to(device)
