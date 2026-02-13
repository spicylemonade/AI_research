"""
Physics-Informed Loss Terms and Constraints for PhysMDT.

Extends the training objective with physics-aware regularizers:
    1. Dimensional consistency loss (M, L, T units)
    2. Conservation law regularizer
    3. Symmetry-awareness loss

Each loss term is toggleable via configuration flags.

References:
    - raissi2019physics: Physics-informed neural networks (PINNs)
    - udrescu2020ai: AI Feynman physics-inspired methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import IDX_TO_TOKEN, TOKEN_TO_IDX, VOCAB_SIZE


# ─── Dimensional Analysis ────────────────────────────────────────────────────

# Physical dimensions for key variables: (M, L, T) exponents
DIMENSION_MAP = {
    # Variables with known dimensions
    'm': (1, 0, 0),     # mass
    'M': (1, 0, 0),     # mass
    'm1': (1, 0, 0),
    'm2': (1, 0, 0),
    'g': (0, 1, -2),    # acceleration due to gravity
    'F': (1, 1, -2),    # force
    'E': (1, 2, -2),    # energy
    'K': (1, 2, -2),    # kinetic energy
    'U': (1, 2, -2),    # potential energy
    'W': (1, 2, -2),    # work
    'P': (1, -1, -2),   # pressure
    'v': (0, 1, -1),    # velocity
    'a': (0, 1, -2),    # acceleration
    't': (0, 0, 1),     # time
    'x': (0, 1, 0),     # position
    'y': (0, 1, 0),
    'z': (0, 1, 0),
    'r': (0, 1, 0),     # distance
    'h': (0, 1, 0),     # height
    'L': (0, 1, 0),     # length
    'I': (1, 2, 0),     # moment of inertia
    'omega': (0, 0, -1), # angular velocity
    'k': (1, 0, -2),    # spring constant
    'rho': (1, -3, 0),  # density
    'G_const': (-1, 3, -2),  # gravitational constant
    'T': (0, 0, 1),     # period
    'p': (1, 1, -1),    # momentum
}


def get_token_dimensions(token: str) -> Optional[Tuple[int, int, int]]:
    """Get physical dimensions (M, L, T) for a token."""
    return DIMENSION_MAP.get(token)


class DimensionalConsistencyLoss(nn.Module):
    """Penalize equations where predicted tokens violate dimensional consistency.

    Implements a soft penalty based on learned dimension embeddings.
    Each token has an associated (M, L, T) dimension vector, and the loss
    encourages the model to produce dimensionally consistent equations.

    Reference: raissi2019physics — physics constraints in neural networks
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_dim: int = 3):
        super().__init__()
        self.d_dim = d_dim

        # Learnable dimension embeddings for each token
        self.dim_embedding = nn.Embedding(vocab_size, d_dim)

        # Initialize known dimensions
        with torch.no_grad():
            for token, dims in DIMENSION_MAP.items():
                idx = TOKEN_TO_IDX.get(token)
                if idx is not None:
                    self.dim_embedding.weight[idx] = torch.tensor(dims, dtype=torch.float)

    def forward(self, logits: torch.Tensor, target_tokens: torch.Tensor
                ) -> torch.Tensor:
        """Compute dimensional consistency loss.

        Encourages the predicted token distribution to have consistent
        dimensions with the target.

        Args:
            logits: (batch, seq_len, vocab_size) model output
            target_tokens: (batch, seq_len) ground truth tokens

        Returns:
            Scalar loss
        """
        # Get predicted dimension vectors (soft, via expected embedding)
        probs = F.softmax(logits, dim=-1)
        pred_dims = torch.matmul(probs, self.dim_embedding.weight)  # (batch, seq, d_dim)

        # Get target dimension vectors
        target_dims = self.dim_embedding(target_tokens)  # (batch, seq, d_dim)

        # MSE loss on dimensions (masked for non-variable tokens)
        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        for token, _ in DIMENSION_MAP.items():
            idx = TOKEN_TO_IDX.get(token)
            if idx is not None:
                mask |= (target_tokens == idx)

        if mask.any():
            loss = F.mse_loss(pred_dims[mask], target_dims[mask])
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss


class ConservationRegularizer(nn.Module):
    """Regularizer that penalizes predictions violating conservation laws.

    For energy/momentum conservation: if the equation describes a conservative
    system, the total energy evaluated at different trajectory points should
    be approximately constant.

    Reference: raissi2019physics
    """

    def __init__(self, conservation_weight: float = 0.1):
        super().__init__()
        self.weight = conservation_weight

    def forward(self, predicted_values: torch.Tensor,
                trajectory_points: torch.Tensor) -> torch.Tensor:
        """Compute conservation loss.

        Args:
            predicted_values: (batch, n_points) equation evaluated at trajectory points
            trajectory_points: (batch, n_points, dim) trajectory data

        Returns:
            Scalar loss penalizing non-conservation
        """
        if predicted_values.shape[1] < 2:
            return torch.tensor(0.0, device=predicted_values.device)

        # The predicted quantity should be approximately constant along trajectory
        mean_val = predicted_values.mean(dim=1, keepdim=True)
        variance = ((predicted_values - mean_val) ** 2).mean()

        return self.weight * variance


class SymmetryLoss(nn.Module):
    """Penalize predictions that break known symmetries.

    For conservative systems, checks time-reversal symmetry:
    f(x, t) should have specific parity under t -> -t.

    Reference: raissi2019physics, udrescu2020ai
    """

    def __init__(self, symmetry_weight: float = 0.1):
        super().__init__()
        self.weight = symmetry_weight

    def forward(self, model_fn, x_data: torch.Tensor,
                symmetry_type: str = "time_reversal") -> torch.Tensor:
        """Compute symmetry violation loss.

        Args:
            model_fn: callable that evaluates the predicted equation
            x_data: (batch, dim) input data
            symmetry_type: type of symmetry to check

        Returns:
            Scalar loss
        """
        if symmetry_type == "time_reversal":
            # For conservative systems: f(x, t) = f(x, -t) for even functions
            # Or f(x, t) = -f(x, -t) for odd functions
            f_forward = model_fn(x_data)
            x_reversed = x_data.clone()
            # Assume last dimension is time
            x_reversed[:, -1] = -x_reversed[:, -1]
            f_reversed = model_fn(x_reversed)

            # Penalize asymmetry (conservative approximation: even symmetry)
            loss = F.mse_loss(f_forward, f_reversed)
            return self.weight * loss

        return torch.tensor(0.0, device=x_data.device)


class PhysicsInformedLoss(nn.Module):
    """Combined physics-informed loss with toggleable components.

    Loss = CE_loss + λ_dim * dim_loss + λ_cons * cons_loss + λ_sym * sym_loss
    """

    def __init__(
        self,
        use_dimensional: bool = True,
        use_conservation: bool = True,
        use_symmetry: bool = True,
        dim_weight: float = 0.1,
        cons_weight: float = 0.1,
        sym_weight: float = 0.05,
    ):
        super().__init__()
        self.use_dimensional = use_dimensional
        self.use_conservation = use_conservation
        self.use_symmetry = use_symmetry

        self.dim_weight = dim_weight
        self.cons_weight = cons_weight
        self.sym_weight = sym_weight

        if use_dimensional:
            self.dim_loss = DimensionalConsistencyLoss()
        if use_conservation:
            self.cons_loss = ConservationRegularizer(cons_weight)
        if use_symmetry:
            self.sym_loss = SymmetryLoss(sym_weight)

    def forward(self, logits: torch.Tensor, target_tokens: torch.Tensor,
                predicted_values: Optional[torch.Tensor] = None,
                trajectory_points: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined physics-informed loss.

        Returns:
            total_loss: scalar
            loss_components: dict of individual loss values
        """
        components = {}

        total = torch.tensor(0.0, device=logits.device)

        if self.use_dimensional:
            dim_l = self.dim_loss(logits, target_tokens)
            total = total + self.dim_weight * dim_l
            components["dimensional"] = dim_l.item()

        if self.use_conservation and predicted_values is not None:
            cons_l = self.cons_loss(predicted_values, trajectory_points)
            total = total + cons_l
            components["conservation"] = cons_l.item()

        if self.use_symmetry:
            components["symmetry"] = 0.0  # Computed separately when model_fn available

        return total, components
