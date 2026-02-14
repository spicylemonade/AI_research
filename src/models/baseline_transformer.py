"""
Vanilla encoder-decoder transformer baseline for symbolic regression.
Standard architecture: 6 layers, 512 dim, 8 heads, ~50M params.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.equation_templates import VOCAB_SIZE, EQUATION_VOCAB, MAX_EQ_LENGTH


class NumericalEncoder(nn.Module):
    """Encode numerical observation pairs into embeddings.

    Each observation point (a vector of max_vars+1 values) is projected
    as a single token. This gives a sequence of max_obs tokens.
    """

    def __init__(self, d_model=512, max_obs=50, max_vars=7):
        super().__init__()
        self.d_model = d_model
        # Project entire observation vector to d_model
        self.value_proj = nn.Sequential(
            nn.Linear(max_vars, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.obs_pos_embedding = nn.Embedding(max_obs, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, observations, n_obs_mask=None):
        """
        Args:
            observations: (batch, max_obs, max_vars+1) float tensor
        Returns:
            encoded: (batch, max_obs, d_model)
            mask: (batch, max_obs) bool - True means IGNORE this position
        """
        B, N, V = observations.shape

        # Normalize input values with log-scaling for stability
        sign = torch.sign(observations)
        log_obs = sign * torch.log1p(torch.abs(observations))

        # Project each observation point as a whole vector
        projected = self.value_proj(log_obs)  # (B, N, d_model)

        # Add position embedding
        pos_ids = torch.arange(N, device=observations.device).unsqueeze(0).expand(B, N)
        pos_emb = self.obs_pos_embedding(pos_ids)

        encoded = self.layer_norm(projected + pos_emb)

        # Padding mask: rows where all values are zero are padding
        row_nonzero = (observations.abs().sum(dim=-1) > 0)  # (B, N) True = valid
        padding_mask = ~row_nonzero  # True = ignore

        return encoded, padding_mask


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaselineTransformer(nn.Module):
    """Standard encoder-decoder transformer for symbolic regression."""

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
        d_ff=2048,
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

        # Encoder
        self.numerical_encoder = NumericalEncoder(d_model, max_obs, max_vars)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_enc_layers
        )

        # Decoder
        self.eq_embedding = nn.Embedding(vocab_size, d_model)
        self.eq_pos_encoding = PositionalEncoding(d_model, max_eq_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_dec_layers
        )

        # Output head
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device):
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def encode(self, observations):
        """Encode numerical observations."""
        enc_input, padding_mask = self.numerical_encoder(observations)
        # padding_mask: True = ignore position
        memory = self.transformer_encoder(
            enc_input, src_key_padding_mask=padding_mask
        )
        return memory, padding_mask

    def decode(self, eq_tokens, memory, memory_key_padding_mask=None):
        """Decode equation tokens given encoder memory."""
        B, T = eq_tokens.shape
        tgt_emb = self.eq_embedding(eq_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.eq_pos_encoding(tgt_emb)

        tgt_mask = self._generate_square_subsequent_mask(T, eq_tokens.device)

        output = self.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_proj(output)
        return logits

    def forward(self, observations, eq_tokens):
        """Full forward pass: observations -> logits.

        Args:
            observations: (B, max_obs, max_vars+1)
            eq_tokens: (B, max_eq_len) - teacher-forced equation tokens
        Returns:
            logits: (B, max_eq_len, vocab_size)
        """
        memory, enc_key_padding_mask = self.encode(observations)
        logits = self.decode(eq_tokens, memory, enc_key_padding_mask)
        return logits

    @torch.no_grad()
    def generate(self, observations, max_len=None, temperature=1.0):
        """Autoregressive generation from observations.

        Args:
            observations: (B, max_obs, max_vars+1)
        Returns:
            tokens: (B, max_len) generated token IDs
        """
        if max_len is None:
            max_len = self.max_eq_len

        B = observations.shape[0]
        device = observations.device

        memory, enc_key_padding_mask = self.encode(observations)

        # Start with [EQ_START] token
        generated = torch.full(
            (B, 1), EQUATION_VOCAB["[EQ_START]"],
            dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            logits = self.decode(generated, memory, enc_key_padding_mask)
            next_logits = logits[:, -1, :] / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have generated [EQ_END]
            if (next_token == EQUATION_VOCAB["[EQ_END]"]).all():
                break

        # Pad to max_eq_len
        if generated.shape[1] < self.max_eq_len:
            pad = torch.full(
                (B, self.max_eq_len - generated.shape[1]),
                EQUATION_VOCAB["[PAD]"],
                dtype=torch.long, device=device
            )
            generated = torch.cat([generated, pad], dim=1)

        return generated[:, :self.max_eq_len]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_baseline_model(device='cuda'):
    """Create the baseline model with standard config."""
    model = BaselineTransformer(
        d_model=512, n_heads=8, n_enc_layers=6, n_dec_layers=6,
        d_ff=2048, dropout=0.1,
    )
    print(f"Baseline model: {model.count_parameters() / 1e6:.1f}M parameters")
    return model.to(device)
