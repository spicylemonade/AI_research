"""
Physics-Aware Recursive Refinement (PARR) Transformer.
Novel architecture combining masked diffusion decoding with recurrent refinement,
inspired by LLaDA's token algebra and URM's ConvSwiGLU.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.equation_templates import VOCAB_SIZE, EQUATION_VOCAB, MAX_EQ_LENGTH


class ConvSwiGLU(nn.Module):
    """Convolutional SwiGLU feed-forward block (from URM).

    Uses depthwise 1D convolutions with gated activation for stronger
    nonlinearity and local pattern capture.
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
        # x: (B, T, d_model)
        gate = self.conv_gate(self.w_gate(x).transpose(1, 2)).transpose(1, 2)
        up = self.conv_up(self.w_up(x).transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.w_down(F.silu(gate) * up))


class TokenAlgebra(nn.Module):
    """Token algebra layer for continuous-space refinement.

    Adds a learned mask signal to token embeddings, triggering the model
    to refine predictions at those positions (inspired by ARChitects).
    """

    def __init__(self, d_model):
        super().__init__()
        self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.blend_gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.Sigmoid(),
        )

    def forward(self, token_embeds, step_fraction):
        """
        Args:
            token_embeds: (B, T, d_model)
            step_fraction: float in [0, 1] indicating refinement progress
        """
        B, T, D = token_embeds.shape
        step_input = torch.full((B, T, 1), step_fraction,
                                device=token_embeds.device, dtype=token_embeds.dtype)
        gate_input = torch.cat([token_embeds, step_input], dim=-1)
        gate = self.blend_gate(gate_input)

        # Decreasing mask signal as refinement progresses
        mask_strength = 1.0 - step_fraction
        mask_signal = self.mask_embed * mask_strength
        return token_embeds + gate * mask_signal


class PARRDecoderBlock(nn.Module):
    """Single shared decoder block for the PARR refinement loop.

    Contains: self-attention, cross-attention, ConvSwiGLU FFN, token algebra.
    """

    def __init__(self, d_model, n_heads, d_ff, kernel_size=5, dropout=0.1):
        super().__init__()

        # Self-attention (bidirectional — no causal mask)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        # Cross-attention to encoder memory
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # ConvSwiGLU FFN
        self.ffn = ConvSwiGLU(d_model, d_ff, kernel_size, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

        # Token algebra
        self.token_algebra = TokenAlgebra(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask=None, step_fraction=0.0):
        """
        Args:
            x: (B, T, d_model) equation embeddings
            memory: (B, S, d_model) encoder output
            memory_key_padding_mask: (B, S) True = ignore
            step_fraction: float in [0, 1]
        """
        # Self-attention (bidirectional)
        residual = x
        x = self.self_attn_norm(x)
        x_attn, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(x_attn)

        # Cross-attention
        residual = x
        x = self.cross_attn_norm(x)
        x_cross, self._last_attn_weights = self.cross_attn(
            x, memory, memory, key_padding_mask=memory_key_padding_mask
        )
        x = residual + self.dropout(x_cross)

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        # Token algebra
        x = self.token_algebra(x, step_fraction)

        return x


class NumericalEncoderPARR(nn.Module):
    """Encoder for numerical observations (shared with baseline)."""

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

    Key innovations:
    1. Masked diffusion decoding with progressive unmasking
    2. Shared decoder block applied K times (recurrent refinement)
    3. ConvSwiGLU FFN for local pattern capture
    4. Token algebra for continuous-space blending
    5. TBPTL for memory-efficient training
    """

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        d_ff=2048,
        K=8,  # Number of refinement steps
        K_bp=3,  # Steps to backpropagate through (TBPTL)
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

        # Shared decoder block (applied K times)
        self.decoder_block = PARRDecoderBlock(
            d_model, n_heads, d_ff, kernel_size, dropout
        )

        # Equation embeddings
        self.eq_embedding = nn.Embedding(vocab_size, d_model)
        self.mask_token_id = EQUATION_VOCAB["[MASK]"]

        # Timestep encoding for refinement steps
        self.step_embedding = nn.Embedding(K + 1, d_model)

        # Position encoding for equation tokens
        self.pos_embedding = nn.Embedding(max_eq_len, d_model)

        # Output heads
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Confidence scoring for progressive unmasking
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, observations):
        """Encode observations."""
        enc_input, padding_mask = self.numerical_encoder(observations)
        memory = self.transformer_encoder(
            enc_input, src_key_padding_mask=padding_mask
        )
        return memory, padding_mask

    def forward(self, observations, eq_tokens, use_tbptl=True):
        """Training forward pass with masked diffusion.

        Randomly masks equation tokens, then refines over K steps.
        Returns per-step logits for computing loss at each refinement step.
        """
        B, T = eq_tokens.shape
        device = eq_tokens.device

        memory, enc_mask = self.encode(observations)

        # Random masking (following LLaDA's variable masking rate)
        mask_rate = torch.rand(B, 1, device=device)  # Per-sample mask rate
        mask_prob = torch.rand(B, T, device=device)
        is_masked = mask_prob < mask_rate  # True = masked

        # Don't mask PAD tokens
        pad_mask = (eq_tokens == EQUATION_VOCAB["[PAD]"])
        is_masked = is_masked & ~pad_mask

        # Initialize with masked tokens
        masked_tokens = eq_tokens.clone()
        masked_tokens[is_masked] = self.mask_token_id

        # Get initial embeddings
        eq_embeds = self.eq_embedding(masked_tokens) * math.sqrt(self.d_model)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        eq_embeds = eq_embeds + self.pos_embedding(pos_ids)

        # Refinement loop with TBPTL
        all_logits = []

        for k in range(self.K):
            step_frac = k / max(self.K - 1, 1)

            # TBPTL: detach early steps
            if use_tbptl and k == self.K - self.K_bp and k > 0:
                eq_embeds = eq_embeds.detach().requires_grad_(True)

            # Add step embedding
            step_emb = self.step_embedding(
                torch.full((B, T), k, device=device, dtype=torch.long)
            )
            step_input = eq_embeds + step_emb

            # Apply shared decoder block
            eq_embeds = self.decoder_block(
                step_input, memory, enc_mask, step_fraction=step_frac
            )

            # Compute logits at this step
            logits = self.output_proj(self.output_norm(eq_embeds))
            all_logits.append(logits)

        return all_logits, is_masked

    def compute_loss(self, all_logits, targets, is_masked):
        """Compute weighted loss across refinement steps.

        Later steps weighted more heavily (linear weighting).
        """
        K = len(all_logits)
        total_loss = 0.0
        pad_idx = EQUATION_VOCAB["[PAD]"]

        for k, logits in enumerate(all_logits):
            # Weight increases linearly
            weight = (k + 1) / K

            # Loss only on masked positions
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=pad_idx,
                reduction='none'
            )

            # Mask the loss to only masked positions
            mask_flat = is_masked.reshape(-1).float()
            loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            total_loss += weight * loss

        return total_loss / K

    @torch.no_grad()
    def generate(self, observations, K=None, temperature=1.0):
        """Generate equations via iterative refinement.

        Starts from fully masked sequence, progressively unmasks.
        """
        if K is None:
            K = self.K

        B = observations.shape[0]
        device = observations.device

        memory, enc_mask = self.encode(observations)

        # Start fully masked
        eq_tokens = torch.full(
            (B, self.max_eq_len), self.mask_token_id,
            dtype=torch.long, device=device
        )
        # Set first token to [EQ_START]
        eq_tokens[:, 0] = EQUATION_VOCAB["[EQ_START]"]

        is_unmasked = torch.zeros(B, self.max_eq_len, dtype=torch.bool, device=device)
        is_unmasked[:, 0] = True  # EQ_START is always unmasked

        eq_embeds = self.eq_embedding(eq_tokens) * math.sqrt(self.d_model)
        pos_ids = torch.arange(self.max_eq_len, device=device).unsqueeze(0).expand(B, self.max_eq_len)
        eq_embeds = eq_embeds + self.pos_embedding(pos_ids)

        for k in range(K):
            step_frac = k / max(K - 1, 1)

            # Add step embedding
            step_emb = self.step_embedding(
                torch.full((B, self.max_eq_len), min(k, self.K - 1),
                           device=device, dtype=torch.long)
            )
            step_input = eq_embeds + step_emb

            # Apply decoder block
            eq_embeds = self.decoder_block(
                step_input, memory, enc_mask, step_fraction=step_frac
            )

            # Get logits and confidence
            logits = self.output_proj(self.output_norm(eq_embeds)) / temperature
            confidence = self.confidence_head(eq_embeds).squeeze(-1)

            # Progressive unmasking: unmask top-p fraction
            p = (k + 1) / K
            n_to_unmask = int(p * self.max_eq_len)

            # Get new predictions
            new_tokens = logits.argmax(dim=-1)

            # Select positions to unmask (highest confidence among still-masked)
            masked_confidence = confidence.clone()
            masked_confidence[is_unmasked] = -1  # Already unmasked

            _, unmask_order = masked_confidence.sort(dim=-1, descending=True)
            for b in range(B):
                for j in range(min(n_to_unmask, self.max_eq_len)):
                    pos = unmask_order[b, j]
                    if not is_unmasked[b, pos]:
                        eq_tokens[b, pos] = new_tokens[b, pos]
                        is_unmasked[b, pos] = True
                        eq_embeds[b, pos] = self.eq_embedding(new_tokens[b, pos]) * math.sqrt(self.d_model)
                        eq_embeds[b, pos] += self.pos_embedding.weight[pos]

        # Final pass: unmask everything remaining
        final_logits = self.output_proj(self.output_norm(eq_embeds))
        final_tokens = final_logits.argmax(dim=-1)
        eq_tokens[~is_unmasked] = final_tokens[~is_unmasked]

        return eq_tokens

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_parr_model(d_model=512, K=8, device='cuda'):
    """Create the PARR model."""
    model = PARRTransformer(
        d_model=d_model, n_heads=8, n_enc_layers=6,
        d_ff=2048, K=K, K_bp=3,
        kernel_size=5, dropout=0.1,
    )
    print(f"PARR model: {model.count_parameters() / 1e6:.1f}M parameters (K={K} refinement steps)")
    return model.to(device)
