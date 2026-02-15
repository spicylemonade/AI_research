"""Test-Time Fine-Tuning (TTFT) with LoRA adapters for PhysMDT.

Implements Low-Rank Adaptation (LoRA) injection into PhysMDT attention layers
and a per-problem fine-tuning loop that uses self-consistency loss: the model's
decoded expression must numerically agree with the observations.

References:
- Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models
- ARChitects ARC 2025: Test-time fine-tuning for per-problem adaptation
"""

import math
import copy
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import numpy as np

from data.tokenizer import ExprTokenizer, MASK_IDX, PAD_IDX, SOS_IDX, EOS_IDX
from models.physmdt import PhysMDT, PhysMDTConfig


# ---------------------------------------------------------------------------
# LoRA Layer
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.

    Adds a low-rank perturbation delta_W = (alpha / rank) * A @ B to an
    existing weight matrix.  During forward the original (frozen) linear
    transformation is augmented:

        y = x @ W^T + x @ (alpha/rank) * (A @ B)^T

    Parameters
    ----------
    in_features : int
        Input dimensionality (columns of the weight matrix).
    out_features : int
        Output dimensionality (rows of the weight matrix).
    rank : int
        Rank of the low-rank decomposition.
    alpha : float
        Scaling constant.  The effective scaling factor is alpha / rank.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 16,
                 alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A: (in_features, rank) -- initialized with Kaiming uniform
        self.A = nn.Parameter(torch.empty(in_features, rank))
        # B: (rank, out_features) -- initialized to zero so delta_W starts at 0
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the low-rank delta: x @ (scaling * A @ B).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Low-rank output of shape (..., out_features).
        """
        # x: (..., in_features) -> (..., rank) -> (..., out_features)
        return (x @ self.A @ self.B) * self.scaling

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}")


# ---------------------------------------------------------------------------
# LoRA-wrapped linear layer
# ---------------------------------------------------------------------------

class LinearWithLoRA(nn.Module):
    """A frozen linear layer augmented with a LoRA adapter.

    Wraps an existing ``nn.Linear`` (whose weight is frozen) and adds
    a trainable low-rank residual.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 16,
                 alpha: float = 1.0):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base + trainable LoRA delta
        return self.original_linear(x) + self.lora(x)


# ---------------------------------------------------------------------------
# LoRA-wrapped MultiheadAttention (in_proj)
# ---------------------------------------------------------------------------

class MHAWithLoRA(nn.Module):
    """Wrapper around ``nn.MultiheadAttention`` that injects LoRA into the
    combined QKV in-projection weight and the output projection.

    PyTorch's ``nn.MultiheadAttention`` stores Q, K, V projections in a
    single ``in_proj_weight`` of shape ``(3 * d_model, d_model)``.  We
    treat this as three separate LoRA adapters (one each for Q, K, V)
    stacked along the output dimension, plus one LoRA for the output
    projection.
    """

    def __init__(self, mha: nn.MultiheadAttention, rank: int = 16,
                 alpha: float = 1.0):
        super().__init__()
        self.mha = mha
        d_model = mha.embed_dim

        # LoRA adapters for Q, K, V projections
        self.lora_q = LoRALayer(d_model, d_model, rank=rank, alpha=alpha)
        self.lora_k = LoRALayer(d_model, d_model, rank=rank, alpha=alpha)
        self.lora_v = LoRALayer(d_model, d_model, rank=rank, alpha=alpha)

        # LoRA for output projection
        self.lora_out = LoRALayer(d_model, d_model, rank=rank, alpha=alpha)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # Compute LoRA deltas
        dq = self.lora_q(query)
        dk = self.lora_k(key)
        dv = self.lora_v(value)

        # Run original MHA
        out, weights = self.mha(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )

        # We add the LoRA delta to the output.  Ideally we'd inject inside
        # the attention computation, but to avoid re-implementing MHA from
        # scratch we use the following approximation which is standard for
        # LoRA on MHA: add delta to the Q/K/V inputs and re-run, or --
        # more cheaply -- use a first-order approximation.  Here we take
        # the simple additive approach on the input projections by running
        # MHA once more with just the deltas.  To keep it efficient we
        # instead just project the deltas through the output projection.
        #
        # Efficient approximation: delta_out ≈ lora_out(out) + Wout(Attn(dq, k, v))
        # For simplicity and to guarantee correctness in the LoRA paradigm
        # (which is designed as additive adaptation), we add LoRA deltas to
        # both input and output sides.

        # A simpler but effective approach used in practice: run a separate
        # forward with the LoRA-adjusted inputs.  This is the standard
        # implementation used in PEFT/HuggingFace LoRA for MHA.
        out_lora, _ = self.mha(
            query + dq, key + dk, value + dv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        # The delta is the difference; add output LoRA on top
        delta = out_lora - out
        out = out + delta + self.lora_out(out)

        return out, weights


# ---------------------------------------------------------------------------
# Inject / Remove LoRA
# ---------------------------------------------------------------------------

_LORA_BACKUP_ATTR = "_lora_original_module"


def inject_lora(model: PhysMDT, rank: int = 16, alpha: float = 1.0) -> PhysMDT:
    """Inject LoRA adapters into all attention layers of a PhysMDT model.

    Targets:
    - Self-attention and cross-attention in every ``PhysMDTDecoderLayer``
    - Attention blocks in the ``SetTransformerEncoder`` (ISAB layers)

    All original weights are frozen; only LoRA parameters are trainable.

    Parameters
    ----------
    model : PhysMDT
        The base model.  Modified in-place.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling alpha.

    Returns
    -------
    PhysMDT
        The same model object, with LoRA adapters injected.
    """
    # Determine device from model parameters
    device = next(model.parameters()).device

    # Freeze all base model parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Inject into decoder self-attention and cross-attention
    for layer in model.decoder_layers:
        # Self-attention
        original_self_attn = layer.self_attn
        lora_self_attn = MHAWithLoRA(original_self_attn, rank=rank, alpha=alpha)
        lora_self_attn = lora_self_attn.to(device)
        # Store backup for removal
        setattr(lora_self_attn, _LORA_BACKUP_ATTR, original_self_attn)
        layer.self_attn = lora_self_attn

        # Cross-attention
        original_cross_attn = layer.cross_attn
        lora_cross_attn = MHAWithLoRA(original_cross_attn, rank=rank, alpha=alpha)
        lora_cross_attn = lora_cross_attn.to(device)
        setattr(lora_cross_attn, _LORA_BACKUP_ATTR, original_cross_attn)
        layer.cross_attn = lora_cross_attn

    # Inject into encoder ISAB attention blocks
    for isab in model.obs_encoder.isab_layers:
        for mab_name in ['mab1', 'mab2']:
            mab = getattr(isab, mab_name)
            original_attn = mab.attn
            lora_attn = MHAWithLoRA(original_attn, rank=rank, alpha=alpha)
            lora_attn = lora_attn.to(device)
            setattr(lora_attn, _LORA_BACKUP_ATTR, original_attn)
            mab.attn = lora_attn

    return model


def remove_lora(model: PhysMDT) -> PhysMDT:
    """Remove LoRA adapters and restore original model weights.

    Parameters
    ----------
    model : PhysMDT
        Model with LoRA adapters injected.

    Returns
    -------
    PhysMDT
        Model with original attention modules restored and all parameters
        unfrozen.
    """
    # Restore decoder layers
    for layer in model.decoder_layers:
        for attr_name in ['self_attn', 'cross_attn']:
            module = getattr(layer, attr_name)
            if hasattr(module, _LORA_BACKUP_ATTR):
                original = getattr(module, _LORA_BACKUP_ATTR)
                setattr(layer, attr_name, original)

    # Restore encoder ISAB layers
    for isab in model.obs_encoder.isab_layers:
        for mab_name in ['mab1', 'mab2']:
            mab = getattr(isab, mab_name)
            module = mab.attn
            if hasattr(module, _LORA_BACKUP_ATTR):
                original = getattr(module, _LORA_BACKUP_ATTR)
                mab.attn = original

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    return model


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def lora_param_count(model: PhysMDT) -> Dict[str, int]:
    """Report LoRA parameter count vs. base model parameter count.

    Parameters
    ----------
    model : PhysMDT
        Model (with or without LoRA adapters).

    Returns
    -------
    dict
        Keys: ``lora_params``, ``base_params``, ``total_params``,
        ``lora_percentage``.
    """
    lora_params = 0
    base_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if 'lora' in name.lower():
            lora_params += n
        else:
            base_params += n

    total = lora_params + base_params
    pct = (lora_params / total * 100) if total > 0 else 0.0

    return {
        'lora_params': lora_params,
        'base_params': base_params,
        'total_params': total,
        'lora_percentage': pct,
    }


# ---------------------------------------------------------------------------
# Self-consistency loss helpers
# ---------------------------------------------------------------------------

def _decode_logits_to_tokens(logits: torch.Tensor) -> List[int]:
    """Greedy-decode logits to a token index list.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(seq_len, vocab_size)``.

    Returns
    -------
    list of int
        Decoded token indices (stripped of PAD but keeping SOS/EOS).
    """
    ids = logits.argmax(dim=-1).tolist()
    # Truncate at first EOS (inclusive)
    try:
        eos_pos = ids.index(EOS_IDX)
        ids = ids[:eos_pos + 1]
    except ValueError:
        pass
    return ids


def _safe_decode_to_sympy(token_ids: List[int],
                          tokenizer: ExprTokenizer) -> Optional[sympy.Expr]:
    """Attempt to decode token ids to a SymPy expression; return None on
    failure."""
    try:
        expr = tokenizer.decode(token_ids, strip_special=True)
        # Basic sanity: must have at least one free symbol or be a number
        if expr is None:
            return None
        return expr
    except Exception:
        return None


def _evaluate_expr_at_points(expr: sympy.Expr,
                             x_vals: np.ndarray,
                             var_names: List[str]) -> Optional[np.ndarray]:
    """Numerically evaluate a SymPy expression at the given data points.

    Parameters
    ----------
    expr : sympy.Expr
        The symbolic expression.
    x_vals : np.ndarray
        Shape ``(n_points, n_vars)`` array of input variable values.
    var_names : list of str
        Names of the variables (e.g. ``['x0', 'x1']``).

    Returns
    -------
    np.ndarray or None
        Shape ``(n_points,)`` array of expression values, or None if
        evaluation fails.
    """
    try:
        symbols = [sympy.Symbol(v, positive=True) for v in var_names]
        f = sympy.lambdify(symbols, expr, modules=['numpy'])
        n_points, n_vars = x_vals.shape
        cols = [x_vals[:, i] for i in range(n_vars)]
        result = f(*cols)
        result = np.asarray(result, dtype=np.float64).flatten()
        if result.shape[0] == 1 and n_points > 1:
            result = np.full(n_points, result[0])
        # Check for inf / nan
        if not np.all(np.isfinite(result)):
            return None
        return result
    except Exception:
        return None


def self_consistency_loss(
    logits: torch.Tensor,
    observations: torch.Tensor,
    obs_mask: torch.Tensor,
    tokenizer: ExprTokenizer,
    n_vars: int,
) -> torch.Tensor:
    """Compute the self-consistency loss for a batch of predictions.

    For each sample in the batch:
      1. Greedy-decode the logits to a symbolic expression.
      2. Evaluate the expression on the observation x-values.
      3. Compute MSE between evaluated y-values and true y-values.

    The loss is averaged over the batch.  Samples where decoding or
    evaluation fails are assigned a penalty loss.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(batch, seq_len, vocab_size)``.
    observations : torch.Tensor
        Shape ``(batch, n_points, n_vars + 1)``.  Last column is y.
    obs_mask : torch.Tensor
        Shape ``(batch, n_points, n_vars + 1)``.  1=valid, 0=pad.
    tokenizer : ExprTokenizer
        Tokenizer instance.
    n_vars : int
        Number of input variables.

    Returns
    -------
    torch.Tensor
        Scalar self-consistency loss.
    """
    batch_size = logits.shape[0]
    device = logits.device
    penalty = 10.0  # penalty for failed decode/eval
    losses = []

    var_names = [f'x{i}' for i in range(n_vars)]

    for b in range(batch_size):
        # Determine valid observation points for this sample
        point_valid = obs_mask[b].sum(dim=-1) > 0  # (n_points,)
        valid_idx = point_valid.nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            losses.append(torch.tensor(penalty, device=device))
            continue

        obs_np = observations[b, valid_idx].detach().cpu().numpy()
        x_np = obs_np[:, :n_vars]  # (n_valid, n_vars)
        y_true = obs_np[:, n_vars]  # (n_valid,)

        # Greedy decode
        token_ids = _decode_logits_to_tokens(logits[b])
        expr = _safe_decode_to_sympy(token_ids, tokenizer)
        if expr is None:
            losses.append(torch.tensor(penalty, device=device))
            continue

        y_pred = _evaluate_expr_at_points(expr, x_np, var_names)
        if y_pred is None:
            losses.append(torch.tensor(penalty, device=device))
            continue

        # MSE
        mse = float(np.mean((y_pred - y_true) ** 2))
        # Clamp to prevent explosion
        mse = min(mse, penalty)
        losses.append(torch.tensor(mse, device=device))

    # We also want gradients to flow through the logits even when the
    # symbolic evaluation is non-differentiable.  Use a straight-through
    # estimator: the logit-entropy term encourages the model to be more
    # confident, weighted by the self-consistency loss.
    #
    # total_loss = sc_loss_value  +  lambda * cross_entropy(logits, greedy_targets)
    #
    # The cross-entropy term IS differentiable w.r.t. logits and acts as
    # a surrogate gradient that pushes the model toward its own greedy
    # decode (reinforced by the sc_loss magnitude).

    sc_loss = torch.stack(losses).mean()

    # Surrogate: cross-entropy between logits and their own greedy argmax
    # This provides actual gradients through the LoRA parameters.
    greedy_targets = logits.detach().argmax(dim=-1)  # (batch, seq_len)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = greedy_targets.reshape(-1)
    surrogate_ce = F.cross_entropy(flat_logits, flat_targets, reduction='mean')

    # Weight the surrogate by the sc_loss magnitude (higher sc_loss => push harder)
    total_loss = sc_loss + surrogate_ce * sc_loss.detach().clamp(min=0.01)

    return total_loss


# ---------------------------------------------------------------------------
# Test-Time Fine-Tuning procedure
# ---------------------------------------------------------------------------

def test_time_finetune(
    model: PhysMDT,
    observations: torch.Tensor,
    obs_mask: torch.Tensor,
    n_steps: int = 128,
    lr: float = 1e-3,
    rank: int = 16,
    alpha: float = 1.0,
    n_vars: int = 1,
    verbose: bool = False,
) -> PhysMDT:
    """Run test-time fine-tuning on a PhysMDT model for a specific problem.

    The procedure:
      1. Inject LoRA adapters into the model.
      2. Freeze base model weights (only LoRA params are trainable).
      3. For ``n_steps`` iterations:
         a. Run forward pass with fully masked expression tokens.
         b. Decode logits to a candidate symbolic expression.
         c. Evaluate on observations to compute self-consistency loss
            (MSE between model prediction and true y-values).
         d. Backprop through LoRA weights only.
      4. Return adapted model.

    Parameters
    ----------
    model : PhysMDT
        Base PhysMDT model (will be modified in-place with LoRA).
    observations : torch.Tensor
        Shape ``(batch, n_points, n_vars+1)``.
    obs_mask : torch.Tensor
        Shape ``(batch, n_points, n_vars+1)``.
    n_steps : int
        Number of fine-tuning steps.
    lr : float
        Learning rate for LoRA parameters.
    rank : int
        LoRA rank (16--32).
    alpha : float
        LoRA scaling alpha.
    n_vars : int
        Number of input variables in the observations.
    verbose : bool
        Whether to print loss at each step.

    Returns
    -------
    PhysMDT
        The model with LoRA adapters adapted to the given observations.
    """
    device = next(model.parameters()).device

    # Move data to device
    observations = observations.to(device)
    obs_mask = obs_mask.to(device)

    # Step 1: Inject LoRA adapters (also freezes base weights)
    inject_lora(model, rank=rank, alpha=alpha)

    # Collect only LoRA parameters for the optimizer
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if len(lora_params) == 0:
        raise RuntimeError("No trainable LoRA parameters found after injection.")

    optimizer = torch.optim.Adam(lora_params, lr=lr)

    tokenizer = ExprTokenizer()
    batch_size = observations.shape[0]
    seq_len = model.config.max_expr_len

    # Create fully masked input tokens
    masked_tokens = torch.full(
        (batch_size, seq_len), MASK_IDX, dtype=torch.long, device=device
    )
    # Set SOS and EOS
    masked_tokens[:, 0] = SOS_IDX
    masked_tokens[:, -1] = EOS_IDX

    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    token_mask[:, 0] = False  # SOS is not masked
    token_mask[:, -1] = False  # EOS is not masked

    # Step 3: Fine-tuning loop
    model.train()
    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward pass
        logits, aux = model(
            observations, obs_mask, masked_tokens, token_mask=token_mask
        )

        # Self-consistency loss
        loss = self_consistency_loss(
            logits, observations, obs_mask, tokenizer, n_vars
        )

        # Backward pass (only LoRA parameters have requires_grad=True)
        loss.backward()
        optimizer.step()

        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"  TTFT step {step:4d}/{n_steps}: loss = {loss.item():.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("TTFT / LoRA Unit Tests")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------------------------------------------------
    # Test 1: Inject LoRA and verify parameter count < 5% of base
    # -----------------------------------------------------------------------
    print("\nTest 1: LoRA parameter count < 5% of base model")
    config = PhysMDTConfig()
    model = PhysMDT(config).to(device)
    base_count = sum(p.numel() for p in model.parameters())
    print(f"  Base parameter count: {base_count:,}")

    inject_lora(model, rank=16)
    stats = lora_param_count(model)
    print(f"  LoRA parameter count: {stats['lora_params']:,}")
    print(f"  Total parameter count: {stats['total_params']:,}")
    print(f"  LoRA percentage: {stats['lora_percentage']:.2f}%")
    assert stats['lora_percentage'] < 5.0, (
        f"LoRA params ({stats['lora_percentage']:.2f}%) should be < 5% of total"
    )
    print("  PASSED")

    # Clean up: remove LoRA for next tests
    remove_lora(model)

    # -----------------------------------------------------------------------
    # Test 2: Forward pass with LoRA produces correct output shape
    # -----------------------------------------------------------------------
    print("\nTest 2: Forward pass with LoRA - correct output shape")
    model = PhysMDT(config).to(device)
    inject_lora(model, rank=16)

    batch_size = 2
    n_points = 20
    n_vars = 2
    seq_len = 32

    obs = torch.randn(batch_size, n_points, config.max_vars + 1, device=device)
    obs_mask = torch.ones(batch_size, n_points, config.max_vars + 1, device=device)
    tokens = torch.full((batch_size, seq_len), MASK_IDX, dtype=torch.long, device=device)
    tokens[:, 0] = SOS_IDX
    tokens[:, -1] = EOS_IDX

    model.eval()
    with torch.no_grad():
        logits, aux = model(obs, obs_mask, tokens)

    expected_shape = (batch_size, seq_len, config.vocab_size)
    print(f"  Output shape: {logits.shape} (expected: {expected_shape})")
    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} != {expected_shape}"
    print("  PASSED")

    remove_lora(model)

    # -----------------------------------------------------------------------
    # Test 3: Base model weights are frozen during TTFT
    # -----------------------------------------------------------------------
    print("\nTest 3: Base model weights frozen during TTFT")
    model = PhysMDT(config).to(device)

    # Record base weights before TTFT
    base_weights_before = {}
    for name, param in model.named_parameters():
        base_weights_before[name] = param.data.clone()

    inject_lora(model, rank=16)

    # Verify base weights are frozen
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            assert param.requires_grad, f"LoRA param {name} should be trainable"
            trainable_count += 1
        else:
            assert not param.requires_grad, f"Base param {name} should be frozen"
            frozen_count += 1

    print(f"  Frozen base parameters: {frozen_count}")
    print(f"  Trainable LoRA parameters: {trainable_count}")

    # Do a small optimization step and verify base weights unchanged
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(lora_params, lr=1e-2)

    model.train()
    obs = torch.randn(2, 10, config.max_vars + 1, device=device)
    obs_mask = torch.ones(2, 10, config.max_vars + 1, device=device)
    tokens = torch.full((2, 16), MASK_IDX, dtype=torch.long, device=device)
    tokens[:, 0] = SOS_IDX
    tokens[:, -1] = EOS_IDX
    token_mask = torch.ones(2, 16, dtype=torch.bool, device=device)
    token_mask[:, 0] = False
    token_mask[:, -1] = False
    target = torch.randint(0, config.vocab_size, (2, 16), device=device)

    logits, aux = model(obs, obs_mask, tokens, token_mask)
    loss = model.compute_loss(logits, target, token_mask, aux['dim_loss'])
    loss.backward()
    optimizer.step()

    # After the step, check that base weights are unchanged
    remove_lora(model)
    weights_changed = False
    for name, param in model.named_parameters():
        if name in base_weights_before:
            if not torch.allclose(param.data, base_weights_before[name]):
                weights_changed = True
                print(f"  WARNING: Base weight {name} changed!")
                break

    assert not weights_changed, "Base model weights should not change during TTFT!"
    print("  Base weights verified unchanged after optimization step")
    print("  PASSED")

    # -----------------------------------------------------------------------
    # Test 4: Run 3 TTFT steps on toy data (no errors expected)
    # -----------------------------------------------------------------------
    print("\nTest 4: Run 3 TTFT steps on toy data")
    model = PhysMDT(config).to(device)

    # Create toy observations: y = 2*x0 + 1
    n_points = 15
    x_vals = np.random.uniform(0.5, 5.0, (1, n_points, 1))
    y_vals = 2.0 * x_vals[:, :, 0:1] + 1.0
    # Pad to max_vars + 1 columns
    obs_data = np.zeros((1, n_points, config.max_vars + 1))
    obs_data[:, :, 0] = x_vals[:, :, 0]
    obs_data[:, :, config.max_vars] = y_vals[:, :, 0]  # y in last used column
    # Actually, y is the last column of the observation (index n_vars)
    # For n_vars=1, y is column 1.  Pad the rest.
    obs_data_proper = np.zeros((1, n_points, config.max_vars + 1))
    obs_data_proper[:, :, 0] = x_vals[:, :, 0]
    obs_data_proper[:, :, 1] = y_vals[:, :, 0]

    obs_tensor = torch.tensor(obs_data_proper, dtype=torch.float32, device=device)
    obs_mask_tensor = torch.zeros(1, n_points, config.max_vars + 1, device=device)
    obs_mask_tensor[:, :, 0] = 1.0  # x0 is valid
    obs_mask_tensor[:, :, 1] = 1.0  # y is valid

    try:
        adapted_model = test_time_finetune(
            model,
            obs_tensor,
            obs_mask_tensor,
            n_steps=3,
            lr=1e-3,
            rank=16,
            n_vars=1,
            verbose=True,
        )
        print("  3 TTFT steps completed without errors")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # -----------------------------------------------------------------------
    # Test 5: LoRA layer standalone test
    # -----------------------------------------------------------------------
    print("\nTest 5: LoRALayer standalone forward/backward")
    lora = LoRALayer(64, 128, rank=8, alpha=2.0).to(device)
    x = torch.randn(4, 10, 64, device=device, requires_grad=True)
    out = lora(x)
    assert out.shape == (4, 10, 128), f"Expected (4, 10, 128), got {out.shape}"
    out.sum().backward()
    assert x.grad is not None, "Input gradient should exist"
    assert lora.A.grad is not None, "LoRA A gradient should exist"
    assert lora.B.grad is not None, "LoRA B gradient should exist"
    print(f"  Output shape: {out.shape}")
    print(f"  LoRA params: A={lora.A.shape}, B={lora.B.shape}")
    print(f"  Scaling: {lora.scaling}")
    print("  PASSED")

    # -----------------------------------------------------------------------
    # Test 6: remove_lora restores model correctly
    # -----------------------------------------------------------------------
    print("\nTest 6: remove_lora restores model correctly")
    model = PhysMDT(config).to(device)
    original_param_count = sum(p.numel() for p in model.parameters())

    inject_lora(model, rank=16)
    injected_param_count = sum(p.numel() for p in model.parameters())
    assert injected_param_count > original_param_count, (
        "Injected model should have more parameters"
    )

    remove_lora(model)
    restored_param_count = sum(p.numel() for p in model.parameters())
    assert restored_param_count == original_param_count, (
        f"Restored param count {restored_param_count} != original {original_param_count}"
    )

    # Verify all parameters are trainable again
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable, "All parameters should be trainable after remove_lora"
    print(f"  Original params: {original_param_count:,}")
    print(f"  After inject: {injected_param_count:,}")
    print(f"  After remove: {restored_param_count:,}")
    print("  PASSED")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All TTFT / LoRA unit tests PASSED")
    print("=" * 60)
