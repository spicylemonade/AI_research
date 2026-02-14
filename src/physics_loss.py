#!/usr/bin/env python3
"""Physics-informed loss functions for PhysMDT training.

Three toggleable loss components:
1. Dimensional consistency loss (M, L, T units)
2. Conservation regularizer (energy/momentum)
3. Symmetry enforcement loss (time-reversal, spatial)

Reference: Raissi et al. 2019 (PINNs) - raissi2019pinns in sources.bib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List


# ---------------------------------------------------------------------------
# Dimensional Analysis Units (MLT system)
# ---------------------------------------------------------------------------

# Each variable/constant has units [M^a, L^b, T^c]
UNIT_TABLE = {
    # Variables
    'x': (0, 1, 0), 'y': (0, 1, 0), 'z': (0, 1, 0), 'd': (0, 1, 0),
    'x0': (0, 1, 0), 'h': (0, 1, 0), 'l_length': (0, 1, 0), 'r': (0, 1, 0), 'R': (0, 1, 0),
    't': (0, 0, 1),
    'v': (0, 1, -1), 'v0': (0, 1, -1), 'vx': (0, 1, -1), 'vy': (0, 1, -1), 'vz': (0, 1, -1),
    'a': (0, 1, -2), 'ax': (0, 1, -2), 'ay': (0, 1, -2), 'az': (0, 1, -2),
    'F': (1, 1, -2), 'Fx': (1, 1, -2), 'Fy': (1, 1, -2), 'Fz': (1, 1, -2),
    'm': (1, 0, 0), 'm1': (1, 0, 0), 'm2': (1, 0, 0),
    'theta': (0, 0, 0), 'phi': (0, 0, 0),  # dimensionless angles
    'omega': (0, 0, -1), 'alpha': (0, 0, -2),
    'tau': (1, 2, -2),  # torque = N*m
    'I_inertia': (1, 2, 0),
    'L_angular': (1, 2, -1),
    'E_energy': (1, 2, -2), 'KE': (1, 2, -2), 'PE': (1, 2, -2), 'W_work': (1, 2, -2),
    'P_power': (1, 2, -3),
    'p_momentum': (1, 1, -1),
    'rho': (1, -3, 0),
    'P_pressure': (1, -1, -2),
    'V_volume': (0, 3, 0),
    'A_area': (0, 2, 0),
    'k_spring': (1, 0, -2),
    'mu': (0, 0, 0),  # dimensionless
    # Constants
    'g_accel': (0, 1, -2),
    'G_const': (-1, 3, -2),
    'pi': (0, 0, 0),
    'euler': (0, 0, 0),
    # Integer/float constants are dimensionless
}

# Operator arity
OPERATOR_ARITIES = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2,
    'neg': 1, 'sin': 1, 'cos': 1, 'tan': 1, 'exp': 1,
    'log': 1, 'sqrt': 1, 'abs': 1,
    'asin': 1, 'acos': 1, 'atan': 1, 'sinh': 1, 'cosh': 1,
}


def get_units(token: str) -> Optional[Tuple[int, int, int]]:
    """Get MLT units for a token. Returns None for operators."""
    if token in UNIT_TABLE:
        return UNIT_TABLE[token]
    if token.startswith('INT_') or token.startswith('CONST_') or token.startswith('C_') or \
       token.startswith('D_') or token.startswith('E_') or token == 'DOT' or \
       token == 'CONST_START' or token == 'CONST_END':
        return (0, 0, 0)  # dimensionless
    return None


def check_dimensional_consistency(prefix_str: str) -> float:
    """Check if a prefix-notation equation is dimensionally consistent.

    Returns 0.0 if consistent, positive penalty otherwise.
    """
    tokens = prefix_str.strip().split()
    try:
        units, _ = _compute_units(tokens, 0)
        if units is None:
            return 0.5  # can't determine = moderate penalty
        return 0.0  # dimensionally consistent
    except Exception:
        return 1.0  # error = high penalty


def _compute_units(tokens, idx):
    """Recursively compute units of a prefix expression."""
    if idx >= len(tokens):
        return None, idx

    tok = tokens[idx]

    if tok in OPERATOR_ARITIES:
        arity = OPERATOR_ARITIES[tok]
        if arity == 1:
            arg_units, next_idx = _compute_units(tokens, idx + 1)
            if arg_units is None:
                return None, next_idx

            if tok in ('sin', 'cos', 'tan', 'exp', 'log', 'asin', 'acos', 'atan', 'sinh', 'cosh'):
                # Argument must be dimensionless
                if arg_units != (0, 0, 0):
                    return None, next_idx  # inconsistent
                return (0, 0, 0), next_idx
            elif tok == 'sqrt':
                return (arg_units[0] / 2, arg_units[1] / 2, arg_units[2] / 2), next_idx
            elif tok in ('neg', 'abs'):
                return arg_units, next_idx
            return arg_units, next_idx

        elif arity == 2:
            left_units, next_idx = _compute_units(tokens, idx + 1)
            right_units, next_idx = _compute_units(tokens, next_idx)
            if left_units is None or right_units is None:
                return None, next_idx

            if tok == 'add' or tok == 'sub':
                if left_units == right_units:
                    return left_units, next_idx
                return None, next_idx  # inconsistent
            elif tok == 'mul':
                return tuple(l + r for l, r in zip(left_units, right_units)), next_idx
            elif tok == 'div':
                return tuple(l - r for l, r in zip(left_units, right_units)), next_idx
            elif tok == 'pow':
                # right must be dimensionless
                if right_units != (0, 0, 0):
                    return None, next_idx
                return left_units, next_idx  # simplified
            return None, next_idx
    else:
        units = get_units(tok)
        return units, idx + 1


# ---------------------------------------------------------------------------
# Loss Components
# ---------------------------------------------------------------------------

class DimensionalConsistencyLoss(nn.Module):
    """Penalize equations with incompatible M,L,T units."""

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__()
        self.weight = weight
        self.enabled = enabled

    def forward(self, pred_prefixes: List[str]) -> torch.Tensor:
        if not self.enabled or not pred_prefixes:
            return torch.tensor(0.0)

        penalties = []
        for prefix in pred_prefixes:
            p = check_dimensional_consistency(prefix)
            penalties.append(p)

        return torch.tensor(sum(penalties) / len(penalties)) * self.weight


class ConservationRegularizer(nn.Module):
    """Enforce energy/momentum conservation on sampled trajectories.

    For time-dependent equations, check that total energy is approximately constant.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__()
        self.weight = weight
        self.enabled = enabled

    def forward(self, pred_values: torch.Tensor, time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_values: (batch, n_points) predicted equation values over time
            time_steps: (batch, n_points) time values (if available)

        Returns:
            Conservation loss (lower = more conserved)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=pred_values.device)

        # Check variance of predictions across time (should be low for conserved quantities)
        variance = pred_values.var(dim=-1).mean()  # average variance across batch
        return variance * self.weight


class SymmetryLoss(nn.Module):
    """Penalize violations of known symmetries.

    Time-reversal symmetry: f(t) should relate to f(-t) for conservative systems.
    Spatial symmetry: f(x) == f(-x) for symmetric potentials.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__()
        self.weight = weight
        self.enabled = enabled

    def forward(self, model_fn, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_fn: callable that takes X and returns predictions
            X: (batch, n_points, n_vars) input observations
            Y: (batch, n_points) output observations

        Returns:
            Symmetry violation loss
        """
        if not self.enabled:
            return torch.tensor(0.0, device=X.device)

        # Time-reversal: negate the time variable (assumed to be last dim)
        X_reversed = X.clone()
        X_reversed[..., -1] = -X_reversed[..., -1]

        try:
            Y_original = model_fn(X)
            Y_reversed = model_fn(X_reversed)

            # For time-symmetric systems: even functions f(t) == f(-t)
            # For odd functions: f(t) == -f(-t)
            # We penalize the minimum of both violations
            even_violation = F.mse_loss(Y_original, Y_reversed)
            odd_violation = F.mse_loss(Y_original, -Y_reversed)
            sym_loss = torch.min(even_violation, odd_violation)

            return sym_loss * self.weight
        except Exception:
            return torch.tensor(0.0, device=X.device)


class PhysicsLoss(nn.Module):
    """Combined physics-informed loss with toggleable components.

    L_physics = w1 * L_dim + w2 * L_conserv + w3 * L_sym
    """

    def __init__(self, dim_weight: float = 0.1, conserv_weight: float = 0.1,
                 sym_weight: float = 0.05,
                 enable_dim: bool = True, enable_conserv: bool = True,
                 enable_sym: bool = True):
        super().__init__()
        self.dim_loss = DimensionalConsistencyLoss(dim_weight, enable_dim)
        self.conserv_loss = ConservationRegularizer(conserv_weight, enable_conserv)
        self.sym_loss = SymmetryLoss(sym_weight, enable_sym)

    def forward(self, pred_prefixes: Optional[List[str]] = None,
                pred_values: Optional[torch.Tensor] = None,
                model_fn=None,
                X: Optional[torch.Tensor] = None,
                Y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined physics loss.

        Returns (total_loss, component_dict)
        """
        total = torch.tensor(0.0)
        components = {}

        if pred_prefixes is not None:
            dim = self.dim_loss(pred_prefixes)
            total = total + dim
            components['dim_consistency'] = dim.item()

        if pred_values is not None:
            conserv = self.conserv_loss(pred_values)
            total = total + conserv
            components['conservation'] = conserv.item()

        if model_fn is not None and X is not None:
            sym = self.sym_loss(model_fn, X, Y)
            total = total + sym
            components['symmetry'] = sym.item()

        components['total_physics_loss'] = total.item()
        return total, components
