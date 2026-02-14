"""
Autoregressive transformer decoder for symbolic regression.
Generates prefix-notation equation tokens conditioned on encoder latent vector.

Uses KV-caching and batched beam inference for efficient incremental decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict

# Token vocabulary
VOCAB = {
    # Special tokens
    'PAD': 0, 'BOS': 1, 'EOS': 2, 'MASK': 3,
    # Binary operators
    'add': 4, 'sub': 5, 'mul': 6, 'div': 7, 'pow': 8,
    # Unary operators
    'sin': 9, 'cos': 10, 'tan': 11, 'exp': 12, 'log': 13,
    'sqrt': 14, 'neg': 15, 'abs': 16, 'asin': 17, 'acos': 18, 'atan': 19,
    # Variables
    'x1': 20, 'x2': 21, 'x3': 22, 'x4': 23, 'x5': 24,
    'x6': 25, 'x7': 26, 'x8': 27, 'x9': 28,
    # Constants
    'C': 29,
    '0': 30, '1': 31, '2': 32, '3': 33, '4': 34, '5': 35,
    'pi': 36, 'e': 37,
    '-1': 38, '-2': 39,
    # Derivation tokens
    'STEP': 40, 'END_STEP': 41,
}

ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

# Operator arities
BINARY_OPS = {'add', 'sub', 'mul', 'div', 'pow'}
UNARY_OPS = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'neg', 'abs', 'asin', 'acos', 'atan'}


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention.

    Supports an optional KV-cache for efficient incremental decoding.
    When use_cache=True, the layer accepts and returns cached key/value tensors
    for the self-attention sublayer so that only the new token position is computed.
    """

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

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor [B, T, D] (T=1 during cached incremental decoding)
            memory: Encoder memory [B, S, D]
            tgt_mask: Causal mask [T, T] (unused when using cache with T=1)
            tgt_key_padding_mask: Padding mask [B, T]
            past_kv: Cached (key, value) from previous steps, each [B, T_prev, D]
            use_cache: Whether to return updated cache

        Returns:
            (output, new_kv) where new_kv is None if use_cache=False
        """
        new_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if past_kv is not None:
            # Incremental mode: x is [B, 1, D], append to cached K/V
            past_k, past_v = past_kv
            k = torch.cat([past_k, x], dim=1)
            v = torch.cat([past_v, x], dim=1)
            if use_cache:
                new_kv = (k, v)
            attn_out, _ = self.self_attn(x, k, v)
        else:
            attn_out, _ = self.self_attn(
                x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )
            if use_cache:
                new_kv = (x.clone(), x.clone())

        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to encoder output
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_out))

        # Feed-forward
        x = self.norm3(x + self.ff(x))
        return x, new_kv


# Type alias for layer caches (list over layers, each a K/V tuple)
LayerCaches = List[Tuple[torch.Tensor, torch.Tensor]]


class AutoregressiveDecoder(nn.Module):
    """Autoregressive transformer decoder for equation generation.

    Architecture: 8-layer, 8-head, dim-256 transformer decoder with cross-attention
    to encoder output, learned positional encodings, and a vocabulary of ~50 tokens.

    Supports three inference modes:
    - Teacher forcing (forward): full-sequence parallel computation for training.
    - Greedy decoding (generate_greedy): incremental decoding with KV-cache.
    - Beam search (generate_beam): batched beam search with KV-cache.

    Args:
        vocab_size: Size of token vocabulary
        embed_dim: Embedding dimension (default 256)
        num_heads: Number of attention heads (default 8)
        num_layers: Number of decoder layers (default 8)
        ff_dim: Feed-forward dimension (default 1024)
        max_seq_len: Maximum sequence length (default 64)
        dropout: Dropout rate (default 0.1)
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
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB['PAD'])
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Memory projection (encoder output z -> memory for cross-attention)
        self.memory_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for teacher forcing training.

        Args:
            tgt_tokens: Target token ids, shape [B, T]
            encoder_output: Encoder latent vector z, shape [B, D]
            tgt_padding_mask: Padding mask, shape [B, T], True for padded positions

        Returns:
            Logits of shape [B, T, vocab_size]
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        # Token + position embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(tgt_tokens) + self.pos_embed(positions)
        h = self.dropout(h)

        # Prepare memory (expand z to sequence for cross-attention)
        memory = self.memory_proj(encoder_output).unsqueeze(1)  # [B, 1, D]

        # Causal mask
        causal_mask = self._generate_causal_mask(T, device)

        # Apply decoder layers (no cache in training)
        for layer in self.layers:
            h, _ = layer(h, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_padding_mask)

        # Output logits
        h = self.output_norm(h)
        logits = self.output_head(h)  # [B, T, vocab_size]

        return logits

    def _step_with_cache(
        self,
        token_ids: torch.Tensor,
        position: int,
        memory: torch.Tensor,
        past_kvs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, LayerCaches]:
        """Single-step batched forward with KV-cache.

        Args:
            token_ids: Token tensor [B, 1]
            position: Current position index
            memory: Projected encoder memory [B, 1, D]
            past_kvs: List of cached (K, V) per layer (or None for first step)

        Returns:
            (logits [B, vocab_size], updated_past_kvs)
        """
        device = token_ids.device
        B = token_ids.shape[0]

        pos = torch.full((B, 1), position, dtype=torch.long, device=device)
        h = self.token_embed(token_ids) + self.pos_embed(pos)  # [B, 1, D]

        new_kvs: LayerCaches = []
        for i, layer in enumerate(self.layers):
            h, kv = layer(h, memory, past_kv=past_kvs[i], use_cache=True)
            assert kv is not None
            new_kvs.append(kv)

        h = self.output_norm(h)
        logits = self.output_head(h[:, 0, :])  # [B, vocab_size]
        return logits, new_kvs

    @torch.no_grad()
    def generate_greedy(
        self,
        encoder_output: torch.Tensor,
        max_length: int = 64,
        temperature: float = 1.0,
    ) -> List[List[str]]:
        """Greedy decoding with KV-cache for efficiency.

        Args:
            encoder_output: Encoder latent [B, D]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no change)

        Returns:
            List of token-string lists, one per batch element.
        """
        self.eval()
        B = encoder_output.shape[0]
        device = encoder_output.device

        memory = self.memory_proj(encoder_output).unsqueeze(1)  # [B, 1, D]

        cur_token = torch.full((B, 1), VOCAB['BOS'], dtype=torch.long, device=device)
        all_tokens = [cur_token]

        past_kvs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            logits, past_kvs = self._step_with_cache(cur_token, step, memory, past_kvs)
            next_logits = logits / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            all_tokens.append(next_token)
            cur_token = next_token

            finished = finished | (next_token.squeeze(-1) == VOCAB['EOS'])
            if finished.all():
                break

        tokens = torch.cat(all_tokens, dim=1)

        results = []
        for b in range(B):
            seq = []
            for t in tokens[b].tolist():
                if t == VOCAB['EOS']:
                    break
                if t == VOCAB['BOS']:
                    continue
                seq.append(ID_TO_TOKEN.get(t, '?'))
            results.append(seq)

        return results

    @torch.no_grad()
    def generate_beam(
        self,
        encoder_output: torch.Tensor,
        beam_width: int = 5,
        max_length: int = 64,
    ) -> List[List[str]]:
        """Beam search decoding with batched KV-cache.

        All active beams are processed in a single batched forward pass per step,
        making beam search only marginally slower than greedy decoding.

        Args:
            encoder_output: shape [1, D] -- only B=1 supported
            beam_width: Number of beams (1-10)
            max_length: Maximum sequence length

        Returns:
            List containing the best decoded token-string list.
        """
        assert encoder_output.shape[0] == 1, "Beam search only supports batch size 1"
        assert 1 <= beam_width <= 10, "Beam width must be between 1 and 10"
        self.eval()
        device = encoder_output.device
        K = beam_width

        memory_single = self.memory_proj(encoder_output).unsqueeze(1)  # [1, 1, D]

        # -- Step 0: process BOS for a single beam --
        bos_token = torch.full((1, 1), VOCAB['BOS'], dtype=torch.long, device=device)
        logits_0, kvs_0 = self._step_with_cache(
            bos_token, 0, memory_single,
            [None] * self.num_layers,
        )  # logits_0: [1, V], kvs_0: list of (K_cache[1,1,D], V_cache[1,1,D])

        log_probs_0 = F.log_softmax(logits_0[0], dim=-1)  # [V]
        topk_lp, topk_ids = log_probs_0.topk(K)  # each [K]

        # Expand caches to K beams (replicate along batch dim)
        # Each cache entry: [1, T_cache, D] -> [K, T_cache, D]
        beam_kvs: LayerCaches = [
            (kk.expand(K, -1, -1).contiguous(), vv.expand(K, -1, -1).contiguous())
            for kk, vv in kvs_0
        ]

        # Track beam state
        beam_scores = topk_lp.tolist()          # list of K floats
        beam_tokens: List[List[int]] = [[VOCAB['BOS'], topk_ids[i].item()] for i in range(K)]
        beam_finished = [topk_ids[i].item() == VOCAB['EOS'] for i in range(K)]

        # Expand memory for K beams
        memory_beams = memory_single.expand(K, -1, -1)  # [K, 1, D]

        completed: List[Tuple[float, List[int]]] = []

        for step in range(1, max_length - 1):
            # Collect active beam indices
            active_indices = [i for i in range(len(beam_scores)) if not beam_finished[i]]
            if not active_indices:
                break

            n_active = len(active_indices)

            # Build batched input: last token of each active beam
            last_tokens = torch.tensor(
                [[beam_tokens[i][-1]] for i in active_indices],
                dtype=torch.long, device=device,
            )  # [n_active, 1]

            # Gather caches for active beams
            active_kvs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
            for layer_idx in range(self.num_layers):
                k_all, v_all = beam_kvs[layer_idx]
                active_kvs.append((
                    k_all[active_indices],
                    v_all[active_indices],
                ))

            active_memory = memory_beams[:n_active]  # [n_active, 1, D]

            # Single batched forward for all active beams
            logits_step, new_kvs_active = self._step_with_cache(
                last_tokens, step, active_memory, active_kvs,
            )  # logits_step: [n_active, V]

            next_log_probs = F.log_softmax(logits_step, dim=-1)  # [n_active, V]

            # Expand: for each active beam, get top-K candidates
            topk_lps, topk_tids = next_log_probs.topk(K, dim=-1)  # [n_active, K]

            # Build candidate list
            cand_scores: List[float] = []
            cand_tokens: List[List[int]] = []
            cand_active_idx: List[int] = []  # which active beam this came from
            cand_tok_choice: List[int] = []  # which of the K choices

            for ai, bi in enumerate(active_indices):
                for ki in range(K):
                    score = beam_scores[bi] + topk_lps[ai, ki].item()
                    tid = topk_tids[ai, ki].item()
                    cand_scores.append(score)
                    cand_tokens.append(beam_tokens[bi] + [tid])
                    cand_active_idx.append(ai)
                    cand_tok_choice.append(ki)

            # Also carry over completed beams from previous iteration
            # (they were already added to completed at the start of this step)

            # Select top-K candidates by score
            sorted_indices = sorted(range(len(cand_scores)), key=lambda x: cand_scores[x], reverse=True)
            selected = sorted_indices[:K]

            # Build new beam state and new caches
            new_beam_scores: List[float] = []
            new_beam_tokens: List[List[int]] = []
            new_beam_finished: List[bool] = []

            # For cache: we need to pick the right active-beam's cache row
            # new_kvs_active has shape [n_active, T_cache, D] per layer
            # We need to index by cand_active_idx for selected candidates
            selected_active_rows = [cand_active_idx[s] for s in selected]

            new_beam_kvs: LayerCaches = []
            for layer_idx in range(self.num_layers):
                nk, nv = new_kvs_active[layer_idx]  # [n_active, T_cache, D]
                row_idx = torch.tensor(selected_active_rows, dtype=torch.long, device=device)
                new_beam_kvs.append((nk[row_idx], nv[row_idx]))

            for s in selected:
                sc = cand_scores[s]
                toks = cand_tokens[s]
                fin = toks[-1] == VOCAB['EOS']
                if fin:
                    completed.append((sc, toks))
                new_beam_scores.append(sc)
                new_beam_tokens.append(toks)
                new_beam_finished.append(fin)

            beam_scores = new_beam_scores
            beam_tokens = new_beam_tokens
            beam_finished = new_beam_finished
            beam_kvs = new_beam_kvs
            memory_beams = memory_single.expand(K, -1, -1)

            if all(beam_finished):
                break

        # Add remaining active beams to completed
        for i in range(len(beam_scores)):
            if not beam_finished[i]:
                completed.append((beam_scores[i], beam_tokens[i]))

        if not completed:
            return [[]]

        best_score, best_seq = max(completed, key=lambda x: x[0])

        result: List[str] = []
        for t in best_seq:
            if t == VOCAB['EOS']:
                break
            if t == VOCAB['BOS']:
                continue
            result.append(ID_TO_TOKEN.get(t, '?'))

        return [result]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def tokens_to_ids(tokens: List[str]) -> List[int]:
    """Convert token strings to vocabulary IDs."""
    return [VOCAB.get(t, VOCAB['C']) for t in tokens]


def ids_to_tokens(ids: List[int]) -> List[str]:
    """Convert vocabulary IDs to token strings."""
    return [ID_TO_TOKEN.get(i, '?') for i in ids]


if __name__ == '__main__':
    import time

    decoder = AutoregressiveDecoder()
    print(f"Decoder parameters: {decoder.count_parameters():,}")
    assert decoder.count_parameters() <= 30_000_000, "Parameter budget exceeded!"

    # Test forward pass (teacher forcing)
    B, T = 2, 30
    tgt = torch.randint(0, VOCAB_SIZE, (B, T))
    z = torch.randn(B, 256)

    logits = decoder(tgt, z)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (B, T, VOCAB_SIZE)

    # Switch to eval mode for inference
    decoder.eval()

    z_single = torch.randn(1, 256)

    # Warmup run (JIT/memory allocation)
    _ = decoder.generate_greedy(z_single, max_length=5)

    # Test greedy decoding speed
    start = time.time()
    result = decoder.generate_greedy(z_single, max_length=30)
    greedy_time = (time.time() - start) * 1000
    print(f"Greedy decoding (30 tokens): {greedy_time:.1f}ms, tokens: {result[0][:10]}...")

    # Warmup beam search
    _ = decoder.generate_beam(z_single, beam_width=2, max_length=5)

    # Test beam search speed
    start = time.time()
    result = decoder.generate_beam(z_single, beam_width=5, max_length=30)
    beam_time = (time.time() - start) * 1000
    print(f"Beam search (width=5, 30 tokens): {beam_time:.1f}ms, tokens: {result[0][:10]}...")

    # Test beam search with different widths
    for bw in [1, 3, 5, 10]:
        start = time.time()
        result = decoder.generate_beam(z_single, beam_width=bw, max_length=30)
        t_ms = (time.time() - start) * 1000
        print(f"  beam_width={bw}: {t_ms:.1f}ms, len={len(result[0])}")

    print(f"\nVocab size: {VOCAB_SIZE}")
    print(f"Parameters within budget (<=30M): {decoder.count_parameters() <= 30_000_000}")
    print(f"Beam search (width=5) within 500ms: {beam_time < 500}")
