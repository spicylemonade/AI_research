"""Autoregressive encoder-decoder transformer baseline for symbolic regression.

This module implements a standard encoder-decoder transformer that takes
numerical observation pairs as input and generates symbolic expression tokens
autoregressively. It serves as the baseline model against which the PhysMDT
masked diffusion transformer is compared.

Architecture overview:
    Encoder: Flattened observation points -> linear projection -> positional
             embeddings -> transformer encoder layers -> memory
    Decoder: Token embeddings + positional embeddings -> transformer decoder
             layers with cross-attention to encoder memory -> logits

Target parameter count: ~30M parameters.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX, MASK_IDX


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ARBaselineConfig:
    """Hyperparameters for the autoregressive baseline transformer.

    Attributes:
        d_model: Dimensionality of the model's hidden representations.
        nhead: Number of attention heads.
        num_encoder_layers: Number of transformer encoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: Dimensionality of the feedforward sublayer.
        dropout: Dropout rate applied throughout the model.
        max_expr_len: Maximum length of generated expression token sequences.
        max_obs_points: Maximum number of observation points per sample.
        max_vars: Maximum number of input variables (columns = max_vars + 1
            to include the output variable y).
        vocab_size: Size of the token vocabulary.
        pad_idx: Index of the <PAD> token.
        sos_idx: Index of the <SOS> token.
        eos_idx: Index of the <EOS> token.
        mask_idx: Index of the <MASK> token.
    """

    d_model: int = 384
    nhead: int = 8
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_expr_len: int = 64
    max_obs_points: int = 50
    max_vars: int = 5
    vocab_size: int = VOCAB_SIZE
    pad_idx: int = PAD_IDX
    sos_idx: int = SOS_IDX
    eos_idx: int = EOS_IDX
    mask_idx: int = MASK_IDX


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ARBaseline(nn.Module):
    """Standard encoder-decoder transformer for symbolic regression.

    The encoder ingests flattened numerical observation pairs (x_1, ..., x_k, y)
    for each of *n_points* observations and produces a contextualised memory
    tensor. The decoder generates symbolic expression tokens autoregressively,
    attending to the encoder memory via cross-attention.

    Args:
        config: An ``ARBaselineConfig`` instance containing all hyperparameters.
    """

    def __init__(self, config: ARBaselineConfig | None = None):
        super().__init__()
        if config is None:
            config = ARBaselineConfig()
        self.config = config

        # ---- Encoder --------------------------------------------------------
        # Each observation point is a vector of length (max_vars + 1).
        # We project it to d_model, then add learned positional embeddings
        # over the observation-point index.
        obs_dim = config.max_vars + 1
        self.obs_proj = nn.Linear(obs_dim, config.d_model)
        self.enc_pos_emb = nn.Embedding(config.max_obs_points, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # ---- Decoder --------------------------------------------------------
        self.tok_emb = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.dec_pos_emb = nn.Embedding(config.max_expr_len, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # ---- Output head ----------------------------------------------------
        self.out_proj = nn.Linear(config.d_model, config.vocab_size)

        # Share weights between token embedding and output projection
        # (weight tying improves generalisation and saves parameters).
        self.out_proj.weight = self.tok_emb.weight

        # ---- Dropout --------------------------------------------------------
        self.dropout = nn.Dropout(config.dropout)

        # ---- Initialisation -------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Xavier-uniform initialisation for linear layers, normal for embeddings."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Embedding-specific initialisation
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.enc_pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dec_pos_emb.weight, mean=0.0, std=0.02)

        # Zero out pad embedding
        with torch.no_grad():
            self.tok_emb.weight[self.config.pad_idx].zero_()

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode numerical observations into a memory tensor.

        Args:
            observations: Float tensor of shape ``(batch, n_points, max_vars+1)``
                containing the raw observation data. Unused variable columns
                should be zero-padded.
            obs_mask: Float tensor of shape ``(batch, n_points, max_vars+1)``
                with 1.0 for valid entries and 0.0 for padding.

        Returns:
            Encoder memory tensor of shape ``(batch, n_points, d_model)``.
        """
        batch_size, n_points, _ = observations.shape

        # Mask observations: zero out padded entries before projection
        x = observations * obs_mask  # (batch, n_points, obs_dim)

        # Linear projection to d_model
        x = self.obs_proj(x)  # (batch, n_points, d_model)

        # Add learned positional embeddings over the point index
        positions = torch.arange(n_points, device=x.device)  # (n_points,)
        x = x + self.enc_pos_emb(positions).unsqueeze(0)  # broadcast over batch

        x = self.dropout(x)

        # Build a key-padding mask for the encoder: True where the point is
        # entirely padding (all columns masked). Shape: (batch, n_points).
        # A point is valid if *any* entry in its row is unmasked.
        src_key_padding_mask = (obs_mask.sum(dim=-1) == 0)  # True = padded

        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    # ------------------------------------------------------------------
    # Causal mask helper
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generate a causal (upper-triangular) attention mask.

        Args:
            sz: Sequence length.
            device: Target device.

        Returns:
            Float mask of shape ``(sz, sz)`` with ``-inf`` above the diagonal
            and ``0.0`` on and below, suitable for ``nn.TransformerDecoder``.
        """
        return nn.Transformer.generate_square_subsequent_mask(
            sz, device=device
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a full forward pass (encoder + decoder) and return logits.

        During training the target tokens are provided in teacher-forcing mode.

        Args:
            observations: Float tensor ``(batch, n_points, max_vars+1)``.
            obs_mask: Float tensor ``(batch, n_points, max_vars+1)``,
                1.0 for valid entries, 0.0 for padding.
            tgt_tokens: Long tensor ``(batch, seq_len)`` of target token
                indices. Typically starts with ``<SOS>`` and includes all
                tokens up to (but not including) the final prediction target.
            tgt_mask: Optional boolean tensor ``(batch, seq_len)``, ``True``
                where the target token is padding. If ``None``, a mask is
                derived from ``PAD_IDX``.

        Returns:
            Logits tensor of shape ``(batch, seq_len, vocab_size)``.
        """
        # --- Encode observations ---
        memory, memory_key_padding_mask = self.encode(observations, obs_mask)

        # --- Decode ---
        batch_size, seq_len = tgt_tokens.shape

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=tgt_tokens.device)
        tgt = self.tok_emb(tgt_tokens) + self.dec_pos_emb(positions).unsqueeze(0)
        tgt = self.dropout(tgt)

        # Causal mask: prevent attending to future tokens
        causal_mask = self._generate_causal_mask(seq_len, tgt_tokens.device)

        # Padding mask for target tokens
        if tgt_mask is None:
            tgt_key_padding_mask = (tgt_tokens == self.config.pad_idx)
        else:
            tgt_key_padding_mask = tgt_mask

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.out_proj(output)  # (batch, seq_len, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Greedy / temperature-based autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        max_len: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate a symbolic expression token sequence.

        Generation starts with the ``<SOS>`` token and terminates when all
        sequences in the batch have produced ``<EOS>`` or ``max_len`` is
        reached.

        Args:
            observations: Float tensor ``(batch, n_points, max_vars+1)``.
            obs_mask: Float tensor ``(batch, n_points, max_vars+1)``.
            max_len: Maximum number of tokens to generate (including ``<SOS>``).
            temperature: Sampling temperature. Use 1.0 for standard sampling;
                values < 1.0 sharpen the distribution (greedy at 0); values
                > 1.0 increase diversity.

        Returns:
            Long tensor ``(batch, generated_len)`` of token indices, starting
            with ``<SOS>``.
        """
        device = observations.device
        batch_size = observations.shape[0]

        # Encode once
        memory, memory_key_padding_mask = self.encode(observations, obs_mask)

        # Start with <SOS>
        generated = torch.full(
            (batch_size, 1), self.config.sos_idx, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(1, max_len):
            seq_len = generated.shape[1]
            positions = torch.arange(seq_len, device=device)

            tgt = self.tok_emb(generated) + self.dec_pos_emb(positions).unsqueeze(0)
            causal_mask = self._generate_causal_mask(seq_len, device)

            output = self.decoder(
                tgt,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            # Take the logits for the last position
            next_logits = output[:, -1, :]  # (batch, d_model)
            next_logits = self.out_proj(next_logits)  # (batch, vocab_size)

            # Apply temperature
            if temperature <= 0:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # For finished sequences, force PAD
            next_token = next_token.squeeze(-1)  # (batch,)
            next_token = torch.where(finished, torch.tensor(self.config.pad_idx, device=device), next_token)

            generated = torch.cat(
                [generated, next_token.unsqueeze(-1)], dim=1
            )

            # Update finished mask
            finished = finished | (next_token == self.config.eos_idx)
            if finished.all():
                break

        return generated

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_unit_tests():
    """Verify model construction, forward-pass shapes, and generation."""
    print("=" * 60)
    print("ARBaseline unit tests")
    print("=" * 60)

    config = ARBaselineConfig()
    model = ARBaseline(config)
    n_params = model.count_parameters()
    print(f"\nModel created with {n_params:,} trainable parameters "
          f"(target: ~30M)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 4
    n_points = config.max_obs_points
    obs_dim = config.max_vars + 1
    seq_len = 20

    # Random observations and mask
    observations = torch.randn(batch_size, n_points, obs_dim, device=device)
    obs_mask = torch.ones(batch_size, n_points, obs_dim, device=device)
    # Simulate some padding: last 2 variable columns are unused
    obs_mask[:, :, -2:] = 0.0

    # Random target tokens (starting with SOS)
    tgt_tokens = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    tgt_tokens[:, 0] = config.sos_idx

    # --- Test forward pass ---
    print(f"\nForward pass (batch={batch_size}, n_points={n_points}, "
          f"seq_len={seq_len}):")
    logits = model(observations, obs_mask, tgt_tokens)
    print(f"  logits shape: {logits.shape}  "
          f"(expected: ({batch_size}, {seq_len}, {config.vocab_size}))")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Shape mismatch: {logits.shape}"
    print("  PASSED")

    # --- Test that loss can be computed ---
    # Shift target: predict next token from previous tokens
    # Input:  <SOS> t1 t2 ... t_{n-1}
    # Target: t1    t2 t3 ... t_n
    target = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        target.reshape(-1),
        ignore_index=config.pad_idx,
    )
    print(f"\n  Cross-entropy loss: {loss.item():.4f}")
    loss.backward()
    print("  Backward pass: PASSED")

    # --- Test generation ---
    model.eval()
    max_gen_len = 32
    print(f"\nGeneration (max_len={max_gen_len}, temperature=1.0):")
    generated = model.generate(
        observations, obs_mask, max_len=max_gen_len, temperature=1.0
    )
    print(f"  generated shape: {generated.shape}  "
          f"(expected: ({batch_size}, <=32))")
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= max_gen_len
    assert (generated[:, 0] == config.sos_idx).all(), \
        "First token should be <SOS>"
    print("  PASSED")

    # Greedy generation
    print(f"\nGeneration (greedy, temperature=0):")
    generated_greedy = model.generate(
        observations, obs_mask, max_len=max_gen_len, temperature=0.0
    )
    print(f"  generated shape: {generated_greedy.shape}")
    assert generated_greedy.shape[0] == batch_size
    print("  PASSED")

    # --- Test with minimal batch ---
    print(f"\nMinimal forward pass (batch=1, n_points=5, seq_len=3):")
    obs_small = torch.randn(1, 5, obs_dim, device=device)
    mask_small = torch.ones(1, 5, obs_dim, device=device)
    tgt_small = torch.tensor([[config.sos_idx, 10, 15]], device=device)
    logits_small = model(obs_small, mask_small, tgt_small)
    print(f"  logits shape: {logits_small.shape}  "
          f"(expected: (1, 3, {config.vocab_size}))")
    assert logits_small.shape == (1, 3, config.vocab_size)
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All unit tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_unit_tests()
