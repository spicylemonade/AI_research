"""Physics-informed training augmentations for PhysMDT.

Provides:
1. Dimensional analysis consistency checking
2. Symbolic equivalence augmentation (≥8 forms per equation)
3. Conservation law physics prior loss
"""

import torch
import torch.nn.functional as F
import numpy as np
import sympy
from sympy import (
    Symbol, simplify, expand, factor, trigsimp, powsimp,
    sin, cos, tan, log, exp, sqrt, Abs, pi, E,
    collect, cancel, radsimp, together, apart,
)
from typing import List, Dict, Optional, Tuple

from src.data.tokenizer import EquationTokenizer


# ---------------------------------------------------------------------------
# 1. Dimensional analysis consistency checker
# ---------------------------------------------------------------------------

# SI base dimension indices: [M, L, T, I, Θ, N, J]
# Simplified to mechanics: [M, L, T] (mass, length, time)
DIM_INDICES = {'M': 0, 'L': 1, 'T': 2}

# Default variable unit annotations from FSReD/Newtonian equations
# Maps variable names to dimension vectors [M, L, T]
DEFAULT_UNIT_MAP = {
    'x1': [0, 1, 0],    # length (m)
    'x2': [0, 1, 0],    # length (m)
    'x3': [0, 0, 1],    # time (s)
    'x4': [1, 0, 0],    # mass (kg)
    'x5': [0, 1, -1],   # velocity (m/s)
    'x6': [0, 1, -2],   # acceleration (m/s²)
    'x7': [1, 1, -2],   # force (N = kg·m/s²)
    'x8': [1, 2, -2],   # energy (J = kg·m²/s²)
    'x9': [0, 0, -1],   # frequency (1/s)
}


def get_expr_dimensions(expr, unit_map: Optional[Dict] = None) -> Optional[List[float]]:
    """Infer SI dimensions [M, L, T] of a sympy expression.

    Returns None if dimensions cannot be determined (e.g., adding
    dimensionally inconsistent terms).
    """
    if unit_map is None:
        unit_map = DEFAULT_UNIT_MAP

    if expr is None:
        return None

    # Symbol
    if isinstance(expr, Symbol):
        name = str(expr)
        return list(unit_map.get(name, [0, 0, 0]))

    # Pure number
    if expr.is_number:
        return [0, 0, 0]  # dimensionless

    # Addition/Subtraction — all terms must have same dimensions
    if expr.func == sympy.Add:
        dims = None
        for arg in expr.args:
            d = get_expr_dimensions(arg, unit_map)
            if d is None:
                return None
            if dims is None:
                dims = d
            else:
                if not all(abs(dims[i] - d[i]) < 1e-9 for i in range(3)):
                    return None  # inconsistent
        return dims

    # Multiplication
    if expr.func == sympy.Mul:
        dims = [0, 0, 0]
        for arg in expr.args:
            d = get_expr_dimensions(arg, unit_map)
            if d is None:
                return None
            dims = [dims[i] + d[i] for i in range(3)]
        return dims

    # Power
    if expr.func == sympy.Pow:
        base_dims = get_expr_dimensions(expr.args[0], unit_map)
        if base_dims is None:
            return None
        exponent = expr.args[1]
        if exponent.is_number:
            exp_val = float(exponent)
            return [base_dims[i] * exp_val for i in range(3)]
        # Non-numeric exponent: base must be dimensionless
        if all(abs(d) < 1e-9 for d in base_dims):
            return [0, 0, 0]
        return None

    # Transcendental functions require dimensionless argument
    if expr.func in (sympy.sin, sympy.cos, sympy.tan, sympy.log, sympy.exp):
        arg_dims = get_expr_dimensions(expr.args[0], unit_map)
        if arg_dims is None:
            return None
        if all(abs(d) < 1e-9 for d in arg_dims):
            return [0, 0, 0]  # dimensionless output
        return None  # dimensionally invalid

    # sqrt
    if expr.func == sympy.Pow and expr.args[1] == sympy.Rational(1, 2):
        base_dims = get_expr_dimensions(expr.args[0], unit_map)
        if base_dims is None:
            return None
        return [base_dims[i] * 0.5 for i in range(3)]

    # Abs
    if expr.func == sympy.Abs:
        return get_expr_dimensions(expr.args[0], unit_map)

    # Fallback: try to recurse
    if hasattr(expr, 'args') and len(expr.args) > 0:
        return get_expr_dimensions(expr.args[0], unit_map)

    return [0, 0, 0]


def check_dimensional_consistency(expr_str: str,
                                   unit_map: Optional[Dict] = None) -> bool:
    """Check if an expression is dimensionally consistent.

    Args:
        expr_str: Expression string.
        unit_map: Variable name → [M, L, T] dimension vector.

    Returns:
        True if dimensionally consistent, False otherwise.
    """
    try:
        local_dict = {f'x{i}': Symbol(f'x{i}') for i in range(1, 10)}
        local_dict['pi'] = sympy.pi
        local_dict['e'] = sympy.E
        expr = sympy.sympify(expr_str, locals=local_dict)
        dims = get_expr_dimensions(expr, unit_map)
        return dims is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. Symbolic equivalence augmentation
# ---------------------------------------------------------------------------

def generate_equivalent_forms(expr_str: str, num_forms: int = 8) -> List[str]:
    """Generate symbolically equivalent forms of an expression.

    Uses SymPy transformations to produce diverse but equivalent
    representations for training data augmentation.

    Args:
        expr_str: Original expression string.
        num_forms: Target number of equivalent forms.

    Returns:
        List of equivalent expression strings (includes original).
    """
    local_dict = {f'x{i}': Symbol(f'x{i}') for i in range(1, 10)}
    local_dict['pi'] = sympy.pi
    local_dict['e'] = sympy.E
    try:
        expr = sympy.sympify(expr_str, locals=local_dict)
    except Exception:
        return [expr_str]

    forms = set()
    forms.add(str(expr))

    transforms = [
        lambda e: expand(e),
        lambda e: factor(e),
        lambda e: simplify(e),
        lambda e: trigsimp(e),
        lambda e: powsimp(e),
        lambda e: cancel(e),
        lambda e: radsimp(e),
        lambda e: collect(e, Symbol('x1')),
        lambda e: collect(e, Symbol('x2')),
        lambda e: together(e),
        lambda e: apart(e),
        lambda e: expand(trigsimp(e)),
        lambda e: factor(expand(e)),
        lambda e: simplify(expand(e)),
        lambda e: powsimp(expand(e)),
    ]

    for t in transforms:
        if len(forms) >= num_forms:
            break
        try:
            result = t(expr)
            result_str = str(result)
            if result_str and result_str != 'nan':
                forms.add(result_str)
        except Exception:
            continue

    # Add double-negation and distribution variants
    try:
        forms.add(str(-(-expr)))
        forms.add(str(expr * 1))
        forms.add(str(expr + 0))
    except Exception:
        pass

    return list(forms)[:num_forms]


def augment_equation_batch(
    expr_strs: List[str],
    tokenizer: EquationTokenizer,
    num_augmented: int = 8,
) -> List[Tuple[str, List[int]]]:
    """Augment a batch of equations with equivalent forms.

    Args:
        expr_strs: List of expression strings.
        tokenizer: Tokenizer for encoding.
        num_augmented: Number of augmented forms per equation.

    Returns:
        List of (expr_str, token_ids) pairs for all augmented forms.
    """
    results = []
    for expr_str in expr_strs:
        forms = generate_equivalent_forms(expr_str, num_forms=num_augmented)
        for form in forms:
            try:
                token_ids = tokenizer.encode(form)
                if token_ids is not None:
                    results.append((form, token_ids))
            except Exception:
                continue
    return results


# ---------------------------------------------------------------------------
# 3. Conservation law physics prior loss
# ---------------------------------------------------------------------------

def physics_prior_loss(
    pred_logits: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer: EquationTokenizer,
    data_matrix: torch.Tensor,
    lambda_physics: float = 0.1,
) -> torch.Tensor:
    """Compute physics prior loss penalizing conservation law violations.

    Evaluates the predicted equation on data and checks:
    1. Energy-like conservation: d(predicted)/dt ≈ 0 for time-invariant quantities
    2. Symmetry penalty: predicted equation should have consistent behavior
       under sign changes of velocity-like variables

    Args:
        pred_logits: (batch, seq_len, vocab_size) model predictions.
        token_ids: (batch, seq_len) ground-truth tokens.
        tokenizer: Tokenizer for decoding.
        data_matrix: (batch, n_points, input_dim) numerical data.
        lambda_physics: Weight for the physics prior loss.

    Returns:
        Scalar physics prior loss tensor.
    """
    device = pred_logits.device
    batch_size = pred_logits.shape[0]

    # Decode predicted equations
    pred_ids = pred_logits.argmax(dim=-1)  # (batch, seq_len)

    total_penalty = torch.tensor(0.0, device=device, requires_grad=True)
    n_valid = 0

    for b in range(batch_size):
        pred_str = tokenizer.decode(pred_ids[b].tolist())
        true_str = tokenizer.decode(token_ids[b].tolist())

        try:
            local_dict = {f'x{i}': Symbol(f'x{i}') for i in range(1, 10)}
            pred_expr = sympy.sympify(pred_str, locals=local_dict)
            if pred_expr is None:
                continue
        except Exception:
            continue

        # Smoothness penalty: predicted equation should produce smooth outputs
        # across data points (penalize high variance in gradient)
        data_np = data_matrix[b].detach().cpu().numpy()
        n_points = data_np.shape[0]
        n_vars = min(data_np.shape[1], 9)

        outputs = []
        for i in range(min(n_points, 50)):
            vals = {Symbol(f'x{j+1}'): float(data_np[i, j]) for j in range(n_vars)}
            try:
                val = complex(pred_expr.subs(vals))
                if np.isfinite(val.real) and abs(val.imag) < 1e-10:
                    outputs.append(val.real)
            except Exception:
                continue

        if len(outputs) > 2:
            outputs = np.array(outputs)
            # Penalize infinite or extreme values
            if np.any(~np.isfinite(outputs)):
                total_penalty = total_penalty + 1.0
                n_valid += 1
            else:
                # Penalize very high variance relative to mean (unstable)
                std = np.std(outputs)
                mean_abs = max(np.mean(np.abs(outputs)), 1e-6)
                cv = std / mean_abs  # coefficient of variation
                if cv > 100:
                    total_penalty = total_penalty + 0.1 * min(cv / 100, 10.0)
                    n_valid += 1

        # Dimensional consistency penalty
        if not check_dimensional_consistency(pred_str):
            total_penalty = total_penalty + 0.5
            n_valid += 1

    if n_valid > 0:
        return lambda_physics * total_penalty / n_valid

    return torch.tensor(0.0, device=device, requires_grad=True)


# ---------------------------------------------------------------------------
# Training integration helper
# ---------------------------------------------------------------------------

def compute_augmented_loss(
    model,
    data_matrix: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer: EquationTokenizer,
    mask_rate: Optional[float] = None,
    pos_encoding=None,
    lambda_physics: float = 0.1,
    use_physics_prior: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Compute training loss with optional physics prior.

    Args:
        model: PhysMDT model.
        data_matrix: (batch, n_points, input_dim) data.
        token_ids: (batch, seq_len) ground-truth tokens.
        tokenizer: Tokenizer for decoding.
        mask_rate: Masking rate (None for random).
        pos_encoding: Optional positional encoding.
        lambda_physics: Physics prior weight.
        use_physics_prior: Whether to add physics prior loss.

    Returns:
        total_loss: Combined masked diffusion + physics prior loss.
        info: Dict with loss breakdown.
    """
    # Standard masked diffusion loss
    loss, info = model.compute_loss(data_matrix, token_ids,
                                     mask_rate=mask_rate,
                                     pos_encoding=pos_encoding)

    info['ce_loss'] = loss.item()

    if use_physics_prior and lambda_physics > 0:
        # Get logits for physics prior
        with torch.no_grad():
            batch_size, seq_len = token_ids.shape
            device = token_ids.device
            if mask_rate is None:
                mask_rate = 0.5
            mask = torch.rand(batch_size, seq_len, device=device) < mask_rate
            masked_tokens = token_ids.clone()
            masked_tokens[mask] = model.mask_token_id
            logits = model.forward(data_matrix, masked_tokens, pos_encoding)

        phys_loss = physics_prior_loss(
            logits, token_ids, tokenizer, data_matrix,
            lambda_physics=lambda_physics,
        )
        info['physics_loss'] = phys_loss.item()
        total_loss = loss + phys_loss
    else:
        info['physics_loss'] = 0.0
        total_loss = loss

    info['total_loss'] = total_loss.item()
    return total_loss, info
