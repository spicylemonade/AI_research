#!/usr/bin/env python3
"""
Attention Visualization and Interpretability Analysis for PhysDiffuser.

Generates attention heatmaps for encoder (ISAB), diffuser (self-attention and
cross-attention), and decoder components.  Visualises diffusion refinement
trajectories showing how masked tokens are progressively resolved.

Item 022 of the research rubric.

Usage:
    python scripts/visualize_attention.py

Outputs figures to figures/attention_maps/
"""

import signal, sys, os, json, math, textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ---- Timeout guard (180 s) -----------------------------------------------
signal.alarm(180)

# ---- Paths ----------------------------------------------------------------
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
FIG_DIR = os.path.join(REPO, "figures", "attention_maps")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Model imports --------------------------------------------------------
from src.model.encoder import SetTransformerEncoder, batch_float_to_ieee754
from src.model.phys_diffuser import PhysDiffuser, DiffusionTransformerLayer
from src.model.decoder import (
    AutoregressiveDecoder,
    VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokens_to_ids,
    TransformerDecoderLayer,
)

# ---- Plotting defaults ----------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass

DPI = 300
CMAP_ATTN = "viridis"
CMAP_DIFF = "magma"

# ---- Load benchmark equations ---------------------------------------------
with open(os.path.join(REPO, "benchmarks", "feynman_equations.json")) as f:
    bench = json.load(f)
EQUATIONS = bench["equations"]

TIERS = ["trivial", "simple", "moderate", "complex", "multi_step"]

def pick_equations(n_per_tier=2):
    """Select n_per_tier representative equations from each difficulty tier."""
    selected = []
    for tier in TIERS:
        tier_eqs = [e for e in EQUATIONS if e["difficulty_tier"] == tier]
        # Pick first and last to get range of complexity within tier
        if len(tier_eqs) >= n_per_tier:
            idx = np.linspace(0, len(tier_eqs) - 1, n_per_tier, dtype=int)
            for i in idx:
                selected.append(tier_eqs[i])
        else:
            selected.extend(tier_eqs[:n_per_tier])
    return selected

SELECTED_EQUATIONS = pick_equations(2)  # 10 total

# ---- Helpers: synthetic observation data ----------------------------------

def generate_observations(eq, n_points=50):
    """Generate synthetic (X, y) observations for an equation.

    Uses the equation's Python formula evaluated on random inputs.
    """
    num_vars = eq["num_variables"]
    formula = eq["formula_python"]
    rng = np.random.default_rng(42)
    X = rng.uniform(0.5, 3.0, size=(n_points, num_vars))

    # Build variable mapping x1..xN -> columns of X
    var_names = [v["name"] for v in eq["variables"]]
    y = np.zeros(n_points)
    for i in range(n_points):
        local = {}
        for vi, vn in enumerate(var_names):
            local[vn] = X[i, vi]
        local["pi"] = np.pi
        local["e_const"] = np.e
        # Add common constants
        for const_name in ["G", "h", "c", "k_B", "epsilon", "mu_0", "sigma",
                           "q", "m_e", "C1", "R"]:
            if const_name not in local:
                local[const_name] = 1.0  # placeholder
        try:
            y[i] = eval(formula, {"__builtins__": {}}, {**local, "pi": np.pi,
                        "sin": np.sin, "cos": np.cos, "tan": np.tan,
                        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                        "abs": abs, "pow": pow})
        except Exception:
            y[i] = 0.0
    # Replace nan/inf
    mask = np.isfinite(y)
    if not mask.all():
        y[~mask] = 0.0
    return X, y


def tokenize_equation(eq):
    """Return token ID sequence for an equation's prefix-notation formula."""
    formula = eq["formula_symbolic"]
    # Parse prefix notation into tokens
    raw = formula.replace("(", " ").replace(")", " ").replace(",", " ").split()
    ids = [VOCAB["BOS"]]
    for tok in raw:
        tok_stripped = tok.strip()
        if tok_stripped in VOCAB:
            ids.append(VOCAB[tok_stripped])
        elif tok_stripped.startswith("C"):
            ids.append(VOCAB["C"])
        else:
            # Try mapping variable names to x1..x9
            ids.append(VOCAB.get(tok_stripped, VOCAB["C"]))
    ids.append(VOCAB["EOS"])
    return ids

# ==========================================================================
# 1. Hook-based attention extraction
# ==========================================================================

class AttentionCapture:
    """Registers forward hooks on nn.MultiheadAttention modules to capture
    attention weights during a single forward pass."""

    def __init__(self):
        self.weights = {}  # name -> [B, H, T_q, T_k]
        self._hooks = []

    def register(self, model, prefix=""):
        for name, mod in model.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                full_name = f"{prefix}.{name}" if prefix else name
                self._hooks.append(
                    mod.register_forward_hook(self._make_hook(full_name))
                )

    def _make_hook(self, name):
        def hook(module, input_args, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # We need to call it with need_weights=True to get weights.
            # Since the model code doesn't pass need_weights, the default is True
            # and batch_first=True, so output[1] is [B, T_q, T_k] or None.
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self.weights[name] = output[1].detach().cpu()
        return hook

    def clear(self):
        self.weights.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class EncoderAttentionCapture:
    """Captures attention weights from the encoder's custom MultiHeadAttention
    modules that use F.scaled_dot_product_attention (which does not return
    weights).  We hook into the forward method and manually compute softmax(QK^T/sqrt(d))."""

    def __init__(self):
        self.weights = {}
        self._hooks = []

    def register(self, model, prefix=""):
        from src.model.encoder import MultiHeadAttention as EncMHA
        for name, mod in model.named_modules():
            if isinstance(mod, EncMHA):
                full_name = f"{prefix}.{name}" if prefix else name
                self._hooks.append(
                    mod.register_forward_hook(self._make_hook(full_name, mod))
                )

    def _make_hook(self, name, module):
        def hook(mod, input_args, output):
            # Re-compute QK^T attention weights from the stored projections
            query, key, value = input_args[0], input_args[1], input_args[2]
            B, N, D = query.shape
            _, M, _ = key.shape
            num_heads = mod.num_heads
            head_dim = mod.head_dim

            q = mod.q_proj(query).reshape(B, N, num_heads, head_dim).transpose(1, 2)
            k = mod.k_proj(key).reshape(B, M, num_heads, head_dim).transpose(1, 2)

            scale = math.sqrt(head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            self.weights[name] = attn_weights.detach().cpu()
        return hook

    def clear(self):
        self.weights.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def enable_attn_weights(model):
    """Monkey-patch MultiheadAttention forward calls so need_weights=True
    is always passed.  This is necessary because the model code does not
    request attention weights by default."""
    for mod in model.modules():
        if isinstance(mod, nn.MultiheadAttention):
            _orig_forward = mod.forward
            def patched(query, key, value, *a, _orig=_orig_forward, **kw):
                kw["need_weights"] = True
                kw["average_attn_weights"] = False  # per-head
                return _orig(query, key, value, *a, **kw)
            mod.forward = patched


# ==========================================================================
# 2. Instantiate models (random weights -- methodology demo)
# ==========================================================================

torch.manual_seed(2026)
np.random.seed(2026)

EMBED_DIM = 256
NUM_HEADS = 8

encoder = SetTransformerEncoder(
    max_variables=9, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
    num_layers=2, num_inducing=16, dropout=0.0,
)
diffuser = PhysDiffuser(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
    num_layers=4, ff_dim=512, max_seq_len=64, dropout=0.0,
)
decoder = AutoregressiveDecoder(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
    num_layers=4, ff_dim=512, max_seq_len=64, dropout=0.0,
)
encoder.eval()
diffuser.eval()
decoder.eval()

# ==========================================================================
# 3. Generate attention maps for 10 representative equations
# ==========================================================================

def plot_attention_heatmap(attn_matrix, title, xlabel, ylabel, tokens_x=None,
                           tokens_y=None, save_path=None, cmap=CMAP_ATTN):
    """Plot a single attention heatmap with labeled axes and colorbar."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if tokens_x is not None:
        ax.set_xticks(range(len(tokens_x)))
        ax.set_xticklabels(tokens_x, rotation=45, ha="right", fontsize=7)
    if tokens_y is not None:
        ax.set_yticks(range(len(tokens_y)))
        ax.set_yticklabels(tokens_y, fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_multi_head_attention(attn_weights, heads_to_show, title_prefix,
                               xlabel, ylabel, tokens_x, tokens_y, save_path,
                               cmap=CMAP_ATTN):
    """Plot attention maps for multiple heads in a grid."""
    n_heads = len(heads_to_show)
    ncols = min(4, n_heads)
    nrows = math.ceil(n_heads / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, h in enumerate(heads_to_show):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        mat = attn_weights[h]  # [T_q, T_k]
        im = ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest",
                        vmin=0, vmax=max(0.01, mat.max()))
        ax.set_title(f"Head {h}", fontsize=9)
        if tokens_x is not None and len(tokens_x) <= 20:
            ax.set_xticks(range(len(tokens_x)))
            ax.set_xticklabels(tokens_x, rotation=45, ha="right", fontsize=6)
        else:
            ax.set_xlabel(xlabel, fontsize=8)
        if tokens_y is not None and len(tokens_y) <= 20:
            ax.set_yticks(range(len(tokens_y)))
            ax.set_yticklabels(tokens_y, fontsize=6)
        else:
            ax.set_ylabel(ylabel, fontsize=8)

    # Hide unused subplots
    for idx in range(n_heads, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(title_prefix, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


print("Generating attention maps for 10 representative equations...")

all_equation_data = []  # Store data for analysis

for eq_idx, eq in enumerate(SELECTED_EQUATIONS):
    eq_id = eq["id"]
    eq_name = eq["name"]
    tier = eq["difficulty_tier"]
    latex = eq["formula_latex"]
    token_ids_list = tokenize_equation(eq)
    token_labels = [ID_TO_TOKEN.get(t, "?") for t in token_ids_list]

    print(f"  [{eq_idx+1}/10] {eq_id}: {eq_name} ({tier})")

    # --- Prepare encoder input ---
    X, y = generate_observations(eq, n_points=50)
    encoded = encoder.encode_observations(X, y, num_vars=eq["num_variables"])
    encoder_input = encoded.unsqueeze(0)  # [1, 50, 160]

    # --- Encoder forward with attention capture ---
    enc_capture = EncoderAttentionCapture()
    enc_capture.register(encoder, prefix="encoder")
    with torch.no_grad():
        z = encoder(encoder_input)  # [1, 256]
    enc_capture.remove_hooks()

    # --- Diffuser forward with attention capture ---
    token_ids = torch.tensor([token_ids_list], dtype=torch.long)
    T_len = token_ids.shape[1]
    # Pad to at least seq_len
    if T_len < 32:
        pad_len = 32 - T_len
        token_ids = F.pad(token_ids, (0, pad_len), value=VOCAB["PAD"])
    padding_mask = (token_ids == VOCAB["PAD"])

    diff_capture = AttentionCapture()
    enable_attn_weights(diffuser)
    diff_capture.register(diffuser, prefix="diffuser")
    with torch.no_grad():
        logits, mask_pos = diffuser(token_ids, z, mask_ratio=0.5, padding_mask=padding_mask)
    diff_capture.remove_hooks()

    # --- Decoder forward with attention capture ---
    dec_capture = AttentionCapture()
    enable_attn_weights(decoder)
    dec_capture.register(decoder, prefix="decoder")
    dec_token_ids = torch.tensor([token_ids_list], dtype=torch.long)
    T_dec = dec_token_ids.shape[1]
    if T_dec < 32:
        dec_token_ids = F.pad(dec_token_ids, (0, 32 - T_dec), value=VOCAB["PAD"])
    dec_padding_mask = (dec_token_ids == VOCAB["PAD"])
    with torch.no_grad():
        dec_logits = decoder(dec_token_ids, z, tgt_padding_mask=dec_padding_mask)
    dec_capture.remove_hooks()

    # Collect data for later analysis
    entry = {
        "eq": eq,
        "token_labels": token_labels[:T_len],
        "enc_weights": dict(enc_capture.weights),
        "diff_weights": dict(diff_capture.weights),
        "dec_weights": dict(dec_capture.weights),
    }
    all_equation_data.append(entry)

    # ---- Plot diffuser self-attention (layer 0) ----
    diff_self_keys = [k for k in diff_capture.weights if "self_attn" in k]
    if diff_self_keys:
        key = diff_self_keys[0]
        w = diff_capture.weights[key][0]  # [H, T_q, T_k]
        if w.dim() == 3:
            # Crop to actual token length
            w_crop = w[:, :T_len, :T_len]
            heads_to_plot = list(range(min(NUM_HEADS, 4)))
            save_name = f"{eq_id}_diffuser_self_attn.png"
            plot_multi_head_attention(
                w_crop, heads_to_plot,
                f"Diffuser Self-Attention: {eq_name}\n({tier} tier, {latex})",
                "Key Position (Token)", "Query Position (Token)",
                token_labels[:T_len], token_labels[:T_len],
                os.path.join(FIG_DIR, save_name),
            )

    # ---- Plot diffuser cross-attention (layer 0) ----
    diff_cross_keys = [k for k in diff_capture.weights if "cross_attn" in k]
    if diff_cross_keys:
        key = diff_cross_keys[0]
        w = diff_capture.weights[key][0]  # [H, T_q, T_k_memory]
        if w.dim() == 3:
            w_crop = w[:, :T_len, :]
            # Cross-attn: query=token positions, key=memory (encoder output)
            avg_cross = w_crop.mean(dim=0)  # [T_q, T_k_mem]
            save_name = f"{eq_id}_diffuser_cross_attn.png"
            plot_attention_heatmap(
                avg_cross.numpy(),
                f"Diffuser Cross-Attention (avg heads): {eq_name}\n({tier}, {latex})",
                "Encoder Memory Position", "Token Position (Query)",
                tokens_y=token_labels[:T_len],
                save_path=os.path.join(FIG_DIR, save_name),
                cmap=CMAP_ATTN,
            )

    # ---- Plot decoder self-attention (layer 0, causal) ----
    dec_self_keys = [k for k in dec_capture.weights if "self_attn" in k]
    if dec_self_keys:
        key = dec_self_keys[0]
        w = dec_capture.weights[key][0]
        if w.dim() == 3:
            w_crop = w[:, :T_dec, :T_dec] if T_dec <= w.shape[1] else w
            actual_len = min(T_dec, w_crop.shape[1], len(token_labels))
            w_crop = w_crop[:, :actual_len, :actual_len]
            heads_to_plot = list(range(min(NUM_HEADS, 4)))
            save_name = f"{eq_id}_decoder_self_attn.png"
            plot_multi_head_attention(
                w_crop, heads_to_plot,
                f"Decoder Causal Self-Attention: {eq_name}\n({tier}, {latex})",
                "Key Position (Token)", "Query Position (Token)",
                token_labels[:actual_len], token_labels[:actual_len],
                os.path.join(FIG_DIR, save_name),
            )

    # ---- Plot decoder cross-attention ----
    dec_cross_keys = [k for k in dec_capture.weights if "cross_attn" in k]
    if dec_cross_keys:
        key = dec_cross_keys[0]
        w = dec_capture.weights[key][0]
        if w.dim() == 3:
            actual_len = min(T_dec, w.shape[1], len(token_labels))
            w_crop = w[:, :actual_len, :]
            avg_cross = w_crop.mean(dim=0)
            save_name = f"{eq_id}_decoder_cross_attn.png"
            plot_attention_heatmap(
                avg_cross.numpy(),
                f"Decoder Cross-Attention (avg heads): {eq_name}\n({tier}, {latex})",
                "Encoder Memory Position", "Token Position (Query)",
                tokens_y=token_labels[:actual_len],
                save_path=os.path.join(FIG_DIR, save_name),
                cmap=CMAP_ATTN,
            )

    enc_capture.clear()
    diff_capture.clear()
    dec_capture.clear()


# ==========================================================================
# 4. Encoder attention analysis figure (ISAB inducing points)
# ==========================================================================

print("Generating encoder ISAB attention analysis...")

# Re-run encoder with hooks for a moderate equation to show ISAB patterns
eq_mod = [e for e in EQUATIONS if e["difficulty_tier"] == "moderate"][0]
X_mod, y_mod = generate_observations(eq_mod, n_points=50)
enc_in = encoder.encode_observations(X_mod, y_mod, num_vars=eq_mod["num_variables"]).unsqueeze(0)

enc_cap2 = EncoderAttentionCapture()
enc_cap2.register(encoder, prefix="enc")
with torch.no_grad():
    z_mod = encoder(enc_in)
enc_cap2.remove_hooks()

# Plot all encoder attention maps in one figure
enc_attn_keys = sorted(enc_cap2.weights.keys())
if enc_attn_keys:
    n_panels = min(len(enc_attn_keys), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx in range(n_panels):
        key = enc_attn_keys[idx]
        w = enc_cap2.weights[key][0]  # [H, T_q, T_k]
        if w.dim() == 3:
            avg = w.mean(dim=0).numpy()
        else:
            avg = w.numpy()
        ax = axes[idx]
        im = ax.imshow(avg, cmap=CMAP_ATTN, aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        short_name = key.split(".")[-2] + "." + key.split(".")[-1] if "." in key else key
        ax.set_title(f"Encoder: {short_name}", fontsize=9)
        ax.set_xlabel("Key Dim", fontsize=8)
        ax.set_ylabel("Query Dim", fontsize=8)
    for idx in range(n_panels, 6):
        axes[idx].axis("off")
    fig.suptitle(
        f"Encoder (ISAB) Attention Patterns\n{eq_mod['name']} ({eq_mod['formula_latex']})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "encoder_isab_attention.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)

enc_cap2.clear()

# ==========================================================================
# 5. Diffusion Refinement Trajectory Visualization (3 equations, 20 steps)
# ==========================================================================

print("Generating diffusion refinement trajectories...")

# Select 3 equations: one simple, one moderate, one complex
trajectory_equations = [
    [e for e in EQUATIONS if e["difficulty_tier"] == "simple"][0],
    [e for e in EQUATIONS if e["difficulty_tier"] == "moderate"][2],
    [e for e in EQUATIONS if e["difficulty_tier"] == "complex"][0],
]

NUM_STEPS = 20
SEQ_LEN = 20  # tokens in the refinement sequence

for traj_idx, eq in enumerate(trajectory_equations):
    eq_name = eq["name"]
    eq_id = eq["id"]
    tier = eq["difficulty_tier"]
    latex = eq["formula_latex"]
    print(f"  Trajectory [{traj_idx+1}/3]: {eq_name} ({tier})")

    # Get encoder output
    X_t, y_t = generate_observations(eq, n_points=50)
    enc_in_t = encoder.encode_observations(X_t, y_t, num_vars=eq["num_variables"]).unsqueeze(0)
    with torch.no_grad():
        z_t = encoder(enc_in_t)

    # Run diffusion refinement step by step, recording the token state at each step
    diffuser.eval()
    device = z_t.device
    B = 1

    tokens = torch.full((B, SEQ_LEN + 2), VOCAB["MASK"], dtype=torch.long, device=device)
    tokens[:, 0] = VOCAB["BOS"]
    tokens[:, -1] = VOCAB["EOS"]

    trajectory_matrix = np.zeros((NUM_STEPS + 1, SEQ_LEN + 2), dtype=int)
    trajectory_matrix[0] = tokens[0].numpy()

    confidence_matrix = np.zeros((NUM_STEPS, SEQ_LEN + 2))

    for step in range(NUM_STEPS):
        progress = step / max(NUM_STEPS - 1, 1)
        mask_ratio = 1.0 + (0.0 - 1.0) * (1 - np.cos(progress * np.pi)) / 2

        positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
        h = diffuser.token_embed(tokens) + diffuser.pos_embed(positions)
        t_tensor = torch.full((B, 1), mask_ratio, device=device)
        soft_mask = diffuser.mask_embed.expand(B, tokens.shape[1], -1) * t_tensor.unsqueeze(-1)
        h = h + soft_mask
        time_emb = diffuser.time_embed(t_tensor)
        h = h + time_emb.unsqueeze(1)
        memory = diffuser.memory_proj(z_t).unsqueeze(1)

        with torch.no_grad():
            for layer in diffuser.layers:
                h = layer(h, memory)
            h = diffuser.output_norm(h)
            step_logits = diffuser.output_head(h) / 0.8

        probs = F.softmax(step_logits, dim=-1)
        max_probs, predictions = probs.max(dim=-1)

        confidence_matrix[step] = max_probs[0].numpy()

        is_mask = (tokens == VOCAB["MASK"])
        if is_mask.any():
            n_masked = is_mask.sum(dim=1)
            n_to_unmask = torch.clamp(
                torch.ceil(n_masked.float() / max(NUM_STEPS - step, 1)).long(), min=1
            )
            for b in range(B):
                mask_idx = is_mask[b].nonzero(as_tuple=True)[0]
                if len(mask_idx) == 0:
                    continue
                confidences = max_probs[b, mask_idx]
                sorted_idx = confidences.argsort(descending=True)
                k = min(n_to_unmask[b].item(), len(mask_idx))
                unmask_positions = mask_idx[sorted_idx[:k]]
                tokens[b, unmask_positions] = predictions[b, unmask_positions]

        trajectory_matrix[step + 1] = tokens[0].numpy()

    # ---- Plot trajectory heatmap ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={"height_ratios": [3, 2]})

    # Token identity trajectory
    traj_display = trajectory_matrix.copy().astype(float)
    # Color: MASK=0, PAD=0.1, BOS/EOS=0.2, operators=0.5-0.7, variables=0.8-1.0
    color_matrix = np.zeros_like(traj_display)
    for r in range(traj_display.shape[0]):
        for c in range(traj_display.shape[1]):
            tid = int(traj_display[r, c])
            if tid == VOCAB["MASK"]:
                color_matrix[r, c] = 0.0  # black for mask
            elif tid == VOCAB["PAD"]:
                color_matrix[r, c] = 0.05
            elif tid == VOCAB["BOS"] or tid == VOCAB["EOS"]:
                color_matrix[r, c] = 0.15
            elif tid in range(4, 9):  # binary ops
                color_matrix[r, c] = 0.45
            elif tid in range(9, 20):  # unary ops
                color_matrix[r, c] = 0.6
            elif tid in range(20, 29):  # variables
                color_matrix[r, c] = 0.85
            else:  # constants
                color_matrix[r, c] = 0.95

    # Custom colormap
    colors_list = [
        (0.0, "#1a1a2e"),   # MASK - dark
        (0.05, "#16213e"),  # PAD
        (0.15, "#0f3460"),  # BOS/EOS
        (0.45, "#e94560"),  # binary ops
        (0.6, "#f39c12"),   # unary ops
        (0.85, "#27ae60"),  # variables
        (0.95, "#3498db"),  # constants
        (1.0, "#3498db"),
    ]
    cmap_traj = LinearSegmentedColormap.from_list("traj", colors_list, N=256)

    im1 = ax1.imshow(color_matrix, cmap=cmap_traj, aspect="auto", interpolation="nearest",
                      vmin=0, vmax=1)

    # Annotate cells with token text for steps 0, N/4, N/2, 3N/4, N
    milestone_steps = [0, NUM_STEPS // 4, NUM_STEPS // 2, 3 * NUM_STEPS // 4, NUM_STEPS]
    for r in milestone_steps:
        for c in range(min(SEQ_LEN + 2, traj_display.shape[1])):
            tid = int(trajectory_matrix[r, c])
            tok_str = ID_TO_TOKEN.get(tid, "?")
            if tok_str == "MASK":
                tok_str = "[M]"
            elif tok_str == "PAD":
                tok_str = ""
            elif tok_str == "BOS":
                tok_str = "<s>"
            elif tok_str == "EOS":
                tok_str = "</s>"
            if tok_str:
                ax1.text(c, r, tok_str, ha="center", va="center", fontsize=5,
                         color="white" if color_matrix[r, c] < 0.4 else "black",
                         fontweight="bold")

    ax1.set_xlabel("Sequence Position", fontsize=10)
    ax1.set_ylabel("Refinement Step", fontsize=10)
    ax1.set_title(
        f"Diffusion Refinement Trajectory: {eq_name}\n"
        f"({tier} tier, {latex}), {NUM_STEPS} steps, MASK -> tokens",
        fontsize=11,
    )
    ax1.set_yticks(range(0, NUM_STEPS + 1, max(1, NUM_STEPS // 5)))

    # Legend for token types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1a1a2e", label="MASK"),
        Patch(facecolor="#e94560", label="Binary Op"),
        Patch(facecolor="#f39c12", label="Unary Op"),
        Patch(facecolor="#27ae60", label="Variable"),
        Patch(facecolor="#3498db", label="Constant"),
        Patch(facecolor="#0f3460", label="BOS/EOS"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=7,
               framealpha=0.9, ncol=3)

    # Confidence evolution
    im2 = ax2.imshow(confidence_matrix[:, 1:-1], cmap="inferno", aspect="auto",
                      interpolation="nearest", vmin=0, vmax=1)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Prediction Confidence", fontsize=9)
    ax2.set_xlabel("Sequence Position (interior tokens)", fontsize=10)
    ax2.set_ylabel("Refinement Step", fontsize=10)
    ax2.set_title("Model Confidence per Position over Refinement", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"trajectory_{eq_id}.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)

# ==========================================================================
# 6. Aggregate summary figure: attention entropy across tiers
# ==========================================================================

print("Generating aggregate analysis figures...")

# Compute attention entropy per tier for the 10 selected equations
tier_entropies = {t: [] for t in TIERS}
tier_cross_entropies = {t: [] for t in TIERS}

for entry in all_equation_data:
    tier = entry["eq"]["difficulty_tier"]
    # Diffuser self-attention entropy
    for key, w in entry["diff_weights"].items():
        if "self_attn" in key:
            if w.dim() == 4:
                w_avg = w[0].mean(dim=0)  # avg over heads -> [T_q, T_k]
            elif w.dim() == 3:
                w_avg = w[0]
            else:
                continue
            # Entropy of attention distribution per query position
            eps = 1e-10
            ent = -(w_avg * (w_avg + eps).log()).sum(dim=-1)
            tier_entropies[tier].append(ent.mean().item())
            break
    # Decoder cross-attention entropy
    for key, w in entry["dec_weights"].items():
        if "cross_attn" in key:
            if w.dim() == 4:
                w_avg = w[0].mean(dim=0)
            elif w.dim() == 3:
                w_avg = w[0]
            else:
                continue
            ent = -(w_avg * (w_avg + eps).log()).sum(dim=-1)
            tier_cross_entropies[tier].append(ent.mean().item())
            break

# Bar chart of attention entropy by tier
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

tier_labels = [t.replace("_", "\n") for t in TIERS]
self_means = [np.mean(tier_entropies[t]) if tier_entropies[t] else 0 for t in TIERS]
cross_means = [np.mean(tier_cross_entropies[t]) if tier_cross_entropies[t] else 0 for t in TIERS]

colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]

bars1 = ax1.bar(tier_labels, self_means, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Mean Attention Entropy (nats)", fontsize=10)
ax1.set_title("Diffuser Self-Attention Entropy by Tier", fontsize=11)
ax1.set_xlabel("Difficulty Tier", fontsize=10)
for bar, val in zip(bars1, self_means):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)

bars2 = ax2.bar(tier_labels, cross_means, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Mean Attention Entropy (nats)", fontsize=10)
ax2.set_title("Decoder Cross-Attention Entropy by Tier", fontsize=11)
ax2.set_xlabel("Difficulty Tier", fontsize=10)
for bar, val in zip(bars2, cross_means):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "attention_entropy_by_tier.png"), dpi=DPI,
            bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 7. Token-type attention preference analysis
# ==========================================================================

print("Generating token-type attention preference analysis...")

# For each equation, compute how much attention operators get vs variables vs constants
token_type_names = ["BinaryOp", "UnaryOp", "Variable", "Constant", "Special"]

def classify_token(tid):
    if tid in range(4, 9):
        return 0  # binary op
    elif tid in range(9, 20):
        return 1  # unary op
    elif tid in range(20, 29):
        return 2  # variable
    elif tid in range(29, 40):
        return 3  # constant
    else:
        return 4  # special

tier_type_attention = {t: np.zeros(5) for t in TIERS}
tier_type_counts = {t: 0 for t in TIERS}

for entry in all_equation_data:
    tier = entry["eq"]["difficulty_tier"]
    token_ids = tokenize_equation(entry["eq"])
    n_tokens = len(token_ids)

    # Get diffuser self-attention
    for key, w in entry["diff_weights"].items():
        if "self_attn" in key:
            if w.dim() == 4:
                attn = w[0].mean(dim=0)[:n_tokens, :n_tokens]
            elif w.dim() == 3:
                attn = w[:n_tokens, :n_tokens]
            else:
                continue
            # Compute attention received by each token type
            attn_received = attn.sum(dim=0)  # sum over query positions
            type_attn = np.zeros(5)
            type_count = np.zeros(5)
            for pos, tid in enumerate(token_ids):
                if pos < attn_received.shape[0]:
                    tt = classify_token(tid)
                    type_attn[tt] += attn_received[pos].item()
                    type_count[tt] += 1
            # Normalize by count
            for tt in range(5):
                if type_count[tt] > 0:
                    type_attn[tt] /= type_count[tt]
            tier_type_attention[tier] += type_attn
            tier_type_counts[tier] += 1
            break

# Normalize and plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(token_type_names[:4]))
width = 0.15
for i, tier in enumerate(TIERS):
    if tier_type_counts[tier] > 0:
        vals = tier_type_attention[tier][:4] / tier_type_counts[tier]
    else:
        vals = np.zeros(4)
    ax.bar(x + i * width, vals, width, label=tier.replace("_", " ").title(),
           color=colors[i], edgecolor="black", linewidth=0.5)

ax.set_ylabel("Normalized Attention Received", fontsize=10)
ax.set_xlabel("Token Type", fontsize=10)
ax.set_title("Self-Attention Received by Token Type Across Tiers", fontsize=12)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(token_type_names[:4], fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "token_type_attention.png"), dpi=DPI,
            bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 8. Summary statistics for the analysis document
# ==========================================================================

print("\nGenerating summary statistics...")

stats = {
    "equations_analyzed": len(SELECTED_EQUATIONS),
    "trajectories_generated": len(trajectory_equations),
    "num_refinement_steps": NUM_STEPS,
    "tier_entropies_self": {t: float(np.mean(tier_entropies[t])) if tier_entropies[t] else 0
                            for t in TIERS},
    "tier_entropies_cross": {t: float(np.mean(tier_cross_entropies[t]))
                              if tier_cross_entropies[t] else 0
                              for t in TIERS},
    "equations_list": [
        {"id": e["eq"]["id"], "name": e["eq"]["name"], "tier": e["eq"]["difficulty_tier"],
         "formula": e["eq"]["formula_latex"]}
        for e in all_equation_data
    ],
    "trajectory_equations": [
        {"id": e["id"], "name": e["name"], "tier": e["difficulty_tier"],
         "formula": e["formula_latex"]}
        for e in trajectory_equations
    ],
}

stats_path = os.path.join(REPO, "results", "attention_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved {len(os.listdir(FIG_DIR))} figures to {FIG_DIR}/")
print(f"Saved attention statistics to {stats_path}")
print("Done.")
