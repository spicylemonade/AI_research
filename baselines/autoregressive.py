"""Autoregressive encoder-decoder transformer baseline (E2ESR-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import VOCAB_SIZE, PAD_ID, SOS_ID, EOS_ID, MAX_SEQ_LEN
from model.encoder import SetTransformerEncoder


class TransformerDecoderLayer(nn.Module):
    """Standard transformer decoder layer with causal self-attention + cross-attention."""
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

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention (causal)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout(h)

        # Cross-attention to encoder
        h = self.norm2(x)
        h, _ = self.cross_attn(h, memory, memory)
        x = x + self.dropout(h)

        # FFN
        h = self.norm3(x)
        x = x + self.ffn(h)

        return x


class AutoregressiveBaseline(nn.Module):
    """Autoregressive encoder-decoder transformer for symbolic regression.

    Architecture: Set-Transformer encoder + causal transformer decoder.
    ~50M parameters total.
    """
    def __init__(self, d_model=512, n_heads=8, n_enc_layers=4, n_dec_layers=6,
                 d_ff=2048, max_vars=10, vocab_size=VOCAB_SIZE,
                 max_seq_len=MAX_SEQ_LEN, dropout=0.1,
                 n_inducing=32, n_seeds=16):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Encoder: Set-Transformer
        self.encoder = SetTransformerEncoder(
            max_vars=max_vars, d_model=d_model, n_heads=n_heads,
            n_isab_layers=n_enc_layers, n_inducing=n_inducing,
            n_seeds=n_seeds, dropout=dropout
        )

        # Decoder: Causal transformer
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, obs_table, target_tokens):
        """
        Args:
            obs_table: (B, N, D+1) observation data
            target_tokens: (B, L) target token IDs (with SOS prefix)

        Returns:
            logits: (B, L, V) log-probabilities over vocabulary
        """
        # Encode observations
        memory = self.encoder(obs_table)  # (B, K, d_model)

        # Decode with teacher forcing
        B, L = target_tokens.shape
        device = target_tokens.device

        # Embeddings
        tok_emb = self.token_embedding(target_tokens)  # (B, L, d_model)
        pos = torch.arange(L, device=device)
        pos_emb = self.pos_encoding(pos)  # (L, d_model)
        h = tok_emb + pos_emb.unsqueeze(0)

        # Causal mask
        causal_mask = self._generate_causal_mask(L, device)
        pad_mask = (target_tokens == PAD_ID)

        for layer in self.decoder_layers:
            h = layer(h, memory, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask)

        h = self.decoder_norm(h)
        logits = self.output_head(h)  # (B, L, V)
        return logits

    @torch.no_grad()
    def beam_search(self, obs_table, beam_width=10, max_len=MAX_SEQ_LEN):
        """Beam search decoding.

        Args:
            obs_table: (1, N, D+1) single observation table
            beam_width: number of beams
            max_len: maximum sequence length

        Returns:
            best_tokens: list of token IDs (best beam)
        """
        device = obs_table.device
        enc_out = self.encoder(obs_table)  # (1, K, d_model)

        # Initialize with single beam
        beams = torch.full((1, 1), SOS_ID, dtype=torch.long, device=device)
        beam_scores = torch.zeros(1, device=device)
        finished = []

        for step in range(max_len - 1):
            B_active = beams.shape[0]
            if B_active == 0:
                break

            # Expand encoder output to match active beams
            mem = enc_out.expand(B_active, -1, -1)

            tok_emb = self.token_embedding(beams)
            pos = torch.arange(beams.shape[1], device=device)
            pos_emb = self.pos_encoding(pos)
            h = tok_emb + pos_emb.unsqueeze(0)

            causal_mask = self._generate_causal_mask(beams.shape[1], device)

            for layer in self.decoder_layers:
                h = layer(h, mem, tgt_mask=causal_mask)
            h = self.decoder_norm(h)
            logits = self.output_head(h[:, -1, :])  # (B_active, V)
            log_probs = F.log_softmax(logits, dim=-1)

            # Expand beams
            scores = beam_scores.unsqueeze(-1) + log_probs  # (B_active, V)
            scores_flat = scores.view(-1)

            # Select top-k
            k = min(beam_width, scores_flat.shape[0])
            topk_scores, topk_idx = scores_flat.topk(k)
            beam_idx = topk_idx // self.vocab_size
            token_idx = topk_idx % self.vocab_size

            # Update beams
            new_beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(-1)], dim=-1)
            new_scores = topk_scores

            # Separate finished and active beams
            active_mask = token_idx != EOS_ID
            for i in range(k):
                if not active_mask[i]:
                    finished.append((new_scores[i].item(), new_beams[i].tolist()))

            beams = new_beams[active_mask]
            beam_scores = new_scores[active_mask]

            if len(beams) == 0:
                break

        # Add remaining active beams
        for i in range(len(beams)):
            finished.append((beam_scores[i].item(), beams[i].tolist()))

        if not finished:
            return [SOS_ID, EOS_ID]

        # Return best
        finished.sort(key=lambda x: x[0], reverse=True)
        return finished[0][1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_baseline_model(d_model=512, n_heads=8, device='cuda'):
    """Create the autoregressive baseline model."""
    model = AutoregressiveBaseline(
        d_model=d_model, n_heads=n_heads,
        n_enc_layers=4, n_dec_layers=6,
        d_ff=d_model * 4, dropout=0.1,
    )
    print(f"Autoregressive baseline: {model.count_parameters() / 1e6:.1f}M parameters")
    return model.to(device)


def train_step(model, obs_table, target_tokens, optimizer):
    """Single training step with teacher forcing.

    Args:
        model: AutoregressiveBaseline
        obs_table: (B, N, D+1) observation data
        target_tokens: (B, L) target token IDs (with SOS at start)

    Returns:
        loss value
    """
    model.train()

    # Input: all tokens except last, Target: all tokens except first
    input_tokens = target_tokens[:, :-1]
    target = target_tokens[:, 1:]

    logits = model(obs_table, input_tokens)  # (B, L-1, V)

    # Compute loss (ignore PAD)
    loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        target.reshape(-1),
        ignore_index=PAD_ID,
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    """Smoke test: overfit to 10 simple equations."""
    import numpy as np
    from data.feynman_loader import generate_benchmark_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    model = create_baseline_model(d_model=512, device=device)

    # Generate data for 10 simple equations (Tier 1 + first 2 Tier 2)
    benchmark = generate_benchmark_data(n_points=200, seed=42)
    simple_eqs = [b for b in benchmark if b['tier'] <= 2][:10]
    print(f"Training on {len(simple_eqs)} equations")

    # Prepare data
    max_n_points = 200
    max_vars = 10
    tables = []
    tokens = []
    for eq in simple_eqs:
        table = eq['table']
        n_vars = table.shape[1] - 1
        # Pad to fixed size
        if table.shape[0] < max_n_points:
            pad = np.zeros((max_n_points - table.shape[0], table.shape[1]))
            table = np.vstack([table, pad])
        else:
            table = table[:max_n_points]
        # Pad variables
        if n_vars + 1 < max_vars + 1:
            pad = np.zeros((table.shape[0], max_vars + 1 - table.shape[1]))
            table = np.hstack([table, pad])
        tables.append(table)
        tokens.append(eq['token_ids'])

    obs_table = torch.tensor(np.array(tables), dtype=torch.float32, device=device)
    target_tokens = torch.tensor(np.array(tokens), dtype=torch.long, device=device)

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\nSmoke test: overfitting to 10 equations...")
    for step in range(1001):
        loss = train_step(model, obs_table, target_tokens, optimizer)
        if step % 100 == 0:
            print(f"  Step {step}: loss = {loss:.4f}")

    # Check if model can predict the simple equations
    model.eval()
    from data.tokenizer import decode
    correct = 0
    for i, eq in enumerate(simple_eqs):
        pred_ids = model.beam_search(obs_table[i:i+1], beam_width=5)
        pred_tokens = decode(pred_ids)
        gt_tokens = eq['prefix']
        match = pred_tokens == gt_tokens
        if match:
            correct += 1
        if i < 5:
            print(f"  {eq['name']}: pred={pred_tokens}, gt={gt_tokens}, match={match}")

    print(f"\nSmoke test: {correct}/{len(simple_eqs)} exact matches")
    print(f"Expected: model should overfit to at least 5/10 equations")
