"""
Token Algebra for Symbolic Manipulation in Embedding Space.

Implements continuous embedding space operations inspired by the ARC 2025
ARChitects' discovery that tokens are points in continuous space, enabling
algebraic operations like interpolation and analogy.

Features:
    - Linear interpolation between symbol embeddings
    - Symbolic analogy via vector arithmetic
    - Nearest-neighbor projection back to discrete vocabulary
    - Integration into the refinement loop

References:
    - arc2025architects: token algebra in continuous space
    - nie2025llada: continuous token operations in LLaDA
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE


class TokenAlgebra:
    """Algebraic operations in the continuous token embedding space."""

    def __init__(self, embedding_weight: torch.Tensor):
        """
        Args:
            embedding_weight: (vocab_size, d_model) embedding matrix from the model
        """
        self.embeddings = embedding_weight.detach()
        self.vocab_size = self.embeddings.shape[0]
        self.d_model = self.embeddings.shape[1]

        # Normalize embeddings for cosine similarity
        self.normalized = F.normalize(self.embeddings, dim=-1)

    def get_embedding(self, token: str) -> Optional[torch.Tensor]:
        """Get the embedding vector for a token string."""
        idx = TOKEN_TO_IDX.get(token)
        if idx is None:
            return None
        return self.embeddings[idx]

    def interpolate(self, token_a: str, token_b: str, alpha: float = 0.5
                    ) -> torch.Tensor:
        """Linear interpolation between two token embeddings.

        Args:
            token_a, token_b: token strings
            alpha: interpolation weight (0.0 = token_a, 1.0 = token_b)

        Returns:
            Interpolated embedding vector
        """
        emb_a = self.get_embedding(token_a)
        emb_b = self.get_embedding(token_b)
        if emb_a is None or emb_b is None:
            raise ValueError(f"Token not in vocabulary: {token_a if emb_a is None else token_b}")
        return (1 - alpha) * emb_a + alpha * emb_b

    def analogy(self, a: str, b: str, c: str) -> torch.Tensor:
        """Compute vector analogy: a is to b as c is to ???

        Returns embedding(b) - embedding(a) + embedding(c)

        Example: velocity - position + force ≈ acceleration
        """
        emb_a = self.get_embedding(a)
        emb_b = self.get_embedding(b)
        emb_c = self.get_embedding(c)
        if any(e is None for e in [emb_a, emb_b, emb_c]):
            raise ValueError("Token not in vocabulary")
        return emb_b - emb_a + emb_c

    def nearest_neighbor(self, query: torch.Tensor, top_k: int = 5,
                         exclude: Optional[List[str]] = None
                         ) -> List[Tuple[str, float]]:
        """Find nearest tokens in embedding space.

        Args:
            query: (d_model,) query embedding
            top_k: number of neighbors to return
            exclude: tokens to exclude from results

        Returns:
            List of (token_string, cosine_similarity) tuples
        """
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        similarities = (query_norm @ self.normalized.T).squeeze(0)

        exclude_idx = set()
        if exclude:
            for tok in exclude:
                idx = TOKEN_TO_IDX.get(tok)
                if idx is not None:
                    exclude_idx.add(idx)

        # Zero out excluded
        for idx in exclude_idx:
            similarities[idx] = -1.0

        top_vals, top_idx = similarities.topk(top_k)

        results = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            token = IDX_TO_TOKEN.get(idx, f"<unk_{idx}>")
            results.append((token, val))

        return results

    def cosine_similarity(self, token_a: str, token_b: str) -> float:
        """Compute cosine similarity between two token embeddings."""
        emb_a = self.get_embedding(token_a)
        emb_b = self.get_embedding(token_b)
        if emb_a is None or emb_b is None:
            return 0.0
        return F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()

    def test_physics_analogies(self) -> Dict[str, float]:
        """Test physically meaningful analogies in the embedding space.

        Returns dict of analogy descriptions to cosine similarity scores.
        """
        analogies = [
            # (a, b, c, expected_d, description)
            ("m", "F", "v", "p", "mass:force :: velocity:momentum"),
            ("v", "a", "x", "v", "velocity:acceleration :: position:velocity"),
            ("+", "*", "sin", "cos", "add:multiply :: sin:cos"),
            ("m", "E", "v", "K", "mass:energy :: velocity:kinetic"),
            ("x", "v", "v", "a", "position:velocity :: velocity:acceleration"),
        ]

        results = {}
        for a, b, c, expected_d, desc in analogies:
            try:
                analogy_emb = self.analogy(a, b, c)
                neighbors = self.nearest_neighbor(analogy_emb, top_k=5, exclude=[a, b, c])
                best_token, best_sim = neighbors[0]

                # Check if expected is in top-5
                expected_found = any(tok == expected_d for tok, _ in neighbors)
                expected_sim = self.cosine_similarity(
                    expected_d,
                    IDX_TO_TOKEN.get(
                        TOKEN_TO_IDX.get(expected_d, 0), ""
                    )
                )

                results[desc] = {
                    "predicted": best_token,
                    "expected": expected_d,
                    "cosine_sim": best_sim,
                    "expected_in_top5": expected_found,
                }
            except Exception as e:
                results[desc] = {"error": str(e)}

        return results

    def get_similarity_matrix(self, tokens: List[str]) -> torch.Tensor:
        """Compute pairwise cosine similarity matrix for a list of tokens.

        Returns:
            (n, n) similarity matrix
        """
        embeddings = []
        for tok in tokens:
            emb = self.get_embedding(tok)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(torch.zeros(self.d_model))

        emb_matrix = torch.stack(embeddings)
        emb_norm = F.normalize(emb_matrix, dim=-1)
        return emb_norm @ emb_norm.T
