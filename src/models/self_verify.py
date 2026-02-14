"""
Equation tree consistency verification and self-validation mechanism.
Validates generated equations against observations and re-ranks candidates.
"""
import numpy as np
import torch
import sympy as sp
from typing import List, Dict, Optional, Tuple

from src.evaluation.metrics import tokens_to_prefix_str, prefix_to_sympy


def numerical_verify(pred_tokens: List[int], observations: np.ndarray,
                     n_input_vars: int, n_obs: int) -> float:
    """Verify a predicted equation against observations by computing R².

    Returns R² score, or -1.0 if the equation is invalid.
    """
    prefix = tokens_to_prefix_str(pred_tokens)
    expr = prefix_to_sympy(prefix)

    if expr is None:
        return -1.0

    variables = {f"x{i}": sp.Symbol(f"x{i}") for i in range(1, 7)}
    var_symbols = [variables[f"x{i}"] for i in range(1, n_input_vars + 1)]

    try:
        func = sp.lambdify(var_symbols, expr, modules=['numpy'])
        x = observations[:n_obs, :n_input_vars]
        y_true = observations[:n_obs, 6]

        args = [x[:, i] for i in range(n_input_vars)]
        y_pred = np.array(func(*args), dtype=np.float64)

        if not np.all(np.isfinite(y_pred)):
            return -1.0

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else -1.0

        return float(np.clip(1 - ss_res / ss_tot, -1.0, 1.0))
    except Exception:
        return -1.0


class SelfVerifier:
    """Self-verification module that generates multiple candidates,
    verifies against observations, and re-ranks by fit quality.
    """

    def __init__(self, model, n_candidates=8, r2_threshold=0.5):
        self.model = model
        self.n_candidates = n_candidates
        self.r2_threshold = r2_threshold

    @torch.no_grad()
    def generate_and_verify(
        self, observations: torch.Tensor,
        n_input_vars: List[int], n_obs: List[int],
        temperatures: List[float] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Generate multiple candidates and rank by R².

        Args:
            observations: (B, max_obs, max_vars+1)
            n_input_vars: list of n_vars per sample
            n_obs: list of n_obs per sample
            temperatures: list of temperatures for diverse generation

        Returns:
            best_tokens: (B, max_eq_len) best candidate tokens
            verification_info: list of dicts with candidate info
        """
        if temperatures is None:
            temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
            temperatures = temperatures[:self.n_candidates]

        B = observations.shape[0]
        device = observations.device
        obs_np = observations.cpu().numpy()

        all_candidates = []  # (n_candidates, B, max_eq_len)
        all_r2 = []  # (n_candidates, B)

        for temp in temperatures:
            # Generate with this temperature
            if hasattr(self.model, 'generate'):
                preds = self.model.generate(observations, temperature=temp)
            else:
                preds = self.model.generate(observations)
            all_candidates.append(preds.cpu())

            # Compute R² for each sample
            r2_scores = []
            pred_np = preds.cpu().numpy()
            for i in range(B):
                r2 = numerical_verify(
                    pred_np[i].tolist(), obs_np[i],
                    n_input_vars[i], n_obs[i]
                )
                r2_scores.append(r2)
            all_r2.append(r2_scores)

        # Select best candidate per sample
        all_r2 = np.array(all_r2)  # (n_candidates, B)
        all_candidates = torch.stack(all_candidates)  # (n_candidates, B, T)

        best_idx = all_r2.argmax(axis=0)  # (B,)
        best_tokens = torch.zeros_like(all_candidates[0])
        verification_info = []

        for i in range(B):
            best_tokens[i] = all_candidates[best_idx[i], i]
            info = {
                "best_r2": float(all_r2[best_idx[i], i]),
                "best_candidate_idx": int(best_idx[i]),
                "all_r2": all_r2[:, i].tolist(),
                "n_valid_candidates": int((all_r2[:, i] > self.r2_threshold).sum()),
            }
            verification_info.append(info)

        return best_tokens.to(device), verification_info

    def compute_invalid_reduction(
        self, verified_tokens: torch.Tensor,
        unverified_tokens: torch.Tensor,
        observations: np.ndarray,
        n_input_vars: List[int], n_obs: List[int],
    ) -> float:
        """Compute reduction in numerically invalid outputs."""
        n = len(n_input_vars)
        ver_np = verified_tokens.cpu().numpy()
        unver_np = unverified_tokens.cpu().numpy()

        ver_invalid = 0
        unver_invalid = 0

        for i in range(n):
            r2_ver = numerical_verify(ver_np[i].tolist(), observations[i],
                                       n_input_vars[i], n_obs[i])
            r2_unver = numerical_verify(unver_np[i].tolist(), observations[i],
                                         n_input_vars[i], n_obs[i])
            if r2_ver < 0.5:
                ver_invalid += 1
            if r2_unver < 0.5:
                unver_invalid += 1

        if unver_invalid == 0:
            return 0.0
        return (unver_invalid - ver_invalid) / unver_invalid
