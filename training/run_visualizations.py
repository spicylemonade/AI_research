"""Qualitative visualizations for PhysMDT (Research Rubric item 021).

Generates three publication-quality figures:
    1. figures/refinement_heatmap_*.png  -- refinement trajectory heatmaps
    2. figures/embedding_space.png       -- UMAP/t-SNE of encoder output embeddings
    3. figures/attention_patterns.png    -- cross-attention weight heatmaps

Usage:
    python training/run_visualizations.py
    python training/run_visualizations.py --checkpoint checkpoints/physmdt_best.pt
    python training/run_visualizations.py --quick   # smoke test with fewer samples
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from data.equations import (
    Equation,
    get_training_equations,
    get_equations_by_tier,
)
from data.physics_generator import PhysicsDataset
from data.tokenizer import (
    ExprTokenizer,
    IDX2TOKEN,
    MASK_IDX,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    VOCAB_SIZE,
)
from models.physmdt import PhysMDT, PhysMDTConfig
from models.refinement import (
    compute_soft_embeddings,
    cosine_unmasking_fraction,
    mask_alpha,
    select_tokens_to_unmask,
)

# ---------------------------------------------------------------------------
# Matplotlib configuration -- publication quality, serif fonts, no seaborn
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ===================================================================
# CLI
# ===================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative visualizations for PhysMDT."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/physmdt_best.pt",
        help="Path to trained PhysMDT checkpoint.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: fewer equations, fewer refinement steps, fewer samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures",
        help="Output directory for figures (default: figures/).",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================================================================
# Model loading (mirrors evaluate_physmdt.py)
# ===================================================================

def _load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[Optional[PhysMDT], Optional[PhysMDTConfig]]:
    """Load a PhysMDT model from a checkpoint file.

    Returns (model, config) on success, or (None, None) if the checkpoint
    cannot be loaded.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        print("  Will use a freshly-initialised model for demonstration.")
        config = PhysMDTConfig()
        model = PhysMDT(config).to(device)
        model.eval()
        return model, config

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"[ERROR] Failed to load checkpoint: {exc}")
        return None, None

    raw_config = ckpt.get("config", None)
    if raw_config is None:
        config = PhysMDTConfig()
    elif isinstance(raw_config, PhysMDTConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = PhysMDTConfig(**raw_config)
    else:
        config = raw_config

    model = PhysMDT(config).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        print("[WARN] No model_state_dict found; using freshly-initialised weights.")
    model.eval()
    return model, config


# ===================================================================
# Equation family classification (for colouring embedding plots)
# ===================================================================

_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    "mechanics": [
        "Newton", "force", "momentum", "impulse", "friction", "drag",
        "work", "kinetic", "energy", "displacement", "velocity",
        "acceleration", "speed", "torque", "power", "incline",
        "projectile", "center-of-mass",
    ],
    "gravity": [
        "gravit", "orbital", "Kepler", "escape", "Schwarz", "tidal",
        "lensing",
    ],
    "oscillation": [
        "pendulum", "SHM", "spring", "oscillat", "wave",
        "frequency", "period",
    ],
    "electricity": [
        "Coulomb", "Lorentz", "LC circuit", "charge",
    ],
    "thermodynamics": [
        "heat", "Stefan", "Boltzmann", "thermal", "temperature", "RMS",
    ],
    "rotation": [
        "angular", "rotat", "moment of inertia", "centripetal",
        "Coriolis",
    ],
    "fluid": [
        "density", "pressure", "terminal", "Bernoulli",
    ],
    "relativity": [
        "time dilation", "relativit", "de Broglie", "Doppler",
        "rocket",
    ],
}


def classify_equation_family(eq: Equation) -> str:
    """Return a coarse family label for an equation based on its name/description."""
    text = f"{eq.name} {eq.description}".lower()
    for family, keywords in _FAMILY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                return family
    return "other"


# ===================================================================
# 1. Refinement Trajectory Heatmaps
# ===================================================================

@torch.no_grad()
def _refine_with_trajectory(
    model: PhysMDT,
    observations: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_len: int = 32,
    n_steps: int = 64,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Modified refinement loop that records the confidence matrix at every step.

    Returns:
        tokens: (batch, seq_len) final discrete tokens.
        confidence_matrix: np.ndarray of shape (n_steps, seq_len) -- max
            probability at each position at each refinement step (batch=0).
    """
    if device is None:
        device = observations.device

    model.eval()
    batch = observations.shape[0]

    embed_weight = model.get_token_embeddings()  # (V, D)
    d_model = embed_weight.shape[1]
    mask_embedding = embed_weight[MASK_IDX].unsqueeze(0).unsqueeze(0)

    tokens = torch.full(
        (batch, seq_len), MASK_IDX, dtype=torch.long, device=device
    )
    tokens[:, 0] = SOS_IDX

    special_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    special_mask[:, 0] = True
    unmasked = special_mask.clone()

    confidence_history: List[np.ndarray] = []
    soft_emb = None

    for step in range(n_steps):
        logits, _aux = model(
            observations=observations,
            obs_mask=obs_mask,
            masked_tokens=tokens,
            token_mask=~unmasked,
            soft_embeddings=soft_emb if step > 0 else None,
        )

        soft_emb_new, probs = compute_soft_embeddings(
            logits, embed_weight, temperature=temperature
        )

        # Record per-position confidence (max prob) for batch element 0
        confidences_step, _ = probs[0].max(dim=-1)  # (seq_len,)
        confidence_history.append(confidences_step.cpu().numpy())

        alpha = mask_alpha(step, n_steps)
        soft_emb = soft_emb_new + alpha * mask_embedding

        if unmasked.any():
            discrete_emb = embed_weight[tokens]
            soft_emb = torch.where(
                unmasked.unsqueeze(-1).expand_as(soft_emb),
                discrete_emb,
                soft_emb,
            )

        target_frac = cosine_unmasking_fraction(step + 1, n_steps)
        unmasked, _confs = select_tokens_to_unmask(
            probs, unmasked, target_frac, special_mask
        )

        argmax_tokens = logits.argmax(dim=-1)
        tokens = torch.where(unmasked, argmax_tokens, tokens)
        tokens[:, 0] = SOS_IDX

    # Final clean pass
    logits_final, _ = model(
        observations=observations,
        obs_mask=obs_mask,
        masked_tokens=tokens,
        token_mask=torch.zeros_like(unmasked),
    )
    final_tokens = logits_final.argmax(dim=-1)
    tokens = torch.where(unmasked, tokens, final_tokens)
    tokens[:, 0] = SOS_IDX

    # Record final confidence
    final_probs = F.softmax(logits_final / max(temperature, 1e-8), dim=-1)
    final_conf, _ = final_probs[0].max(dim=-1)
    confidence_history.append(final_conf.cpu().numpy())

    confidence_matrix = np.stack(confidence_history, axis=0)  # (n_steps+1, seq_len)
    return tokens, confidence_matrix


def _select_refinement_equations(quick: bool = False) -> List[Equation]:
    """Pick one Tier 2, one Tier 3, one Tier 4 training equation."""
    training = get_training_equations()
    t2 = [eq for eq in training if eq.tier == 2]
    t3 = [eq for eq in training if eq.tier == 3]
    t4 = [eq for eq in training if eq.tier == 4]

    selected = []
    # Tier 2: Kinetic energy (t2_01) -- 0.5*m*v^2
    if t2:
        selected.append(t2[0])
    # Tier 3: Newton's law of gravitation (t3_01) -- G*m1*m2/r^2
    if t3:
        selected.append(t3[0])
    # Tier 4: Simple pendulum period (t4_01) -- 2*pi*sqrt(L/g)
    if t4:
        selected.append(t4[0])

    if quick:
        selected = selected[:2]

    return selected


def generate_refinement_heatmaps(
    model: PhysMDT,
    device: torch.device,
    outdir: str,
    n_steps: int = 64,
    quick: bool = False,
    seed: int = 42,
) -> None:
    """Generate refinement trajectory heatmaps for selected equations."""
    print("\n--- Generating refinement trajectory heatmaps ---")
    tokenizer = ExprTokenizer()
    equations = _select_refinement_equations(quick=quick)

    if quick:
        n_steps = min(n_steps, 16)

    seq_len = model.config.max_expr_len

    for eq in equations:
        print(f"  Processing: {eq.id} -- {eq.name}")

        # Generate a single observation sample
        ds = PhysicsDataset(
            equations=[eq],
            n_samples=1,
            n_points=model.config.max_obs_points,
            noise_level=0.0,
            seed=seed,
        )
        if len(ds) == 0:
            print(f"    [SKIP] Could not generate data for {eq.name}")
            continue

        sample = ds[0]
        obs = sample["observations"].unsqueeze(0).to(device)
        obs_mask = sample["obs_mask"].unsqueeze(0).to(device)

        tokens, conf_matrix = _refine_with_trajectory(
            model=model,
            observations=obs,
            obs_mask=obs_mask,
            seq_len=seq_len,
            n_steps=n_steps,
            temperature=1.0,
            device=device,
        )

        # Determine the actual token length (up to first PAD after EOS)
        tok_list = tokens[0].cpu().tolist()
        display_len = seq_len
        for idx, t in enumerate(tok_list):
            if t == PAD_IDX:
                display_len = idx
                break
            if t == EOS_IDX:
                display_len = idx + 1
                break
        display_len = max(display_len, 4)
        display_len = min(display_len, seq_len)

        # Trim the confidence matrix to the display length
        conf_display = conf_matrix[:, :display_len]  # (n_steps+1, display_len)

        # Get final token names for annotation
        final_token_names = [
            tokenizer.get_token_str(tok_list[j]) for j in range(display_len)
        ]

        # --- Plot ---
        fig, ax = plt.subplots(
            figsize=(max(display_len * 0.42, 5), max(conf_display.shape[0] * 0.12, 3))
        )
        im = ax.imshow(
            conf_display,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
        )

        ax.set_xlabel("Token Position")
        ax.set_ylabel("Refinement Step")
        ax.set_title(
            f"Refinement Trajectory: {eq.name}\n({eq.id}, Tier {eq.tier})",
            fontsize=11,
        )

        # X-axis: token positions with final token names
        xtick_positions = list(range(display_len))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(final_token_names, rotation=60, ha="right", fontsize=7)

        # Y-axis: refinement step labels (show a subset to avoid crowding)
        n_rows = conf_display.shape[0]
        if n_rows <= 20:
            ytick_positions = list(range(n_rows))
        else:
            ytick_positions = list(range(0, n_rows, max(1, n_rows // 15)))
            if (n_rows - 1) not in ytick_positions:
                ytick_positions.append(n_rows - 1)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([str(y) for y in ytick_positions], fontsize=7)

        # Annotate final committed tokens on the last row
        last_row_idx = n_rows - 1
        for j in range(display_len):
            val = conf_display[last_row_idx, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(
                j,
                last_row_idx,
                final_token_names[j],
                ha="center",
                va="center",
                fontsize=5,
                color=text_color,
                fontweight="bold",
            )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Confidence (max prob)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        fig.tight_layout()
        fname = os.path.join(outdir, f"refinement_heatmap_{eq.id}.png")
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"    Saved: {fname}")


# ===================================================================
# 2. Embedding Space Visualization
# ===================================================================

def _try_import_umap():
    """Attempt to import UMAP; fall back to t-SNE from sklearn."""
    try:
        from umap import UMAP  # type: ignore
        return UMAP, "UMAP"
    except ImportError:
        pass
    try:
        from sklearn.manifold import TSNE  # type: ignore
        return TSNE, "t-SNE"
    except ImportError:
        pass
    return None, None


@torch.no_grad()
def _collect_embeddings(
    model: PhysMDT,
    device: torch.device,
    n_samples_per_eq: int = 5,
    n_points: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    """Collect encoder output embeddings for all training equations.

    Returns:
        embeddings: (N, d_model) array of mean-pooled encoder outputs.
        tiers: list of tier labels per sample.
        families: list of family labels per sample.
        eq_ids: list of equation IDs per sample.
    """
    training_eqs = get_training_equations()
    all_embeddings = []
    all_tiers: List[int] = []
    all_families: List[str] = []
    all_eq_ids: List[str] = []

    for eq in training_eqs:
        ds = PhysicsDataset(
            equations=[eq],
            n_samples=n_samples_per_eq,
            n_points=n_points,
            noise_level=0.0,
            seed=seed,
        )
        if len(ds) == 0:
            continue

        family = classify_equation_family(eq)

        for i in range(min(len(ds), n_samples_per_eq)):
            sample = ds[i]
            obs = sample["observations"].unsqueeze(0).to(device)
            obs_mask = sample["obs_mask"].unsqueeze(0).to(device)

            obs_encoded, obs_key_pad_mask = model.obs_encoder(obs, obs_mask)
            # obs_encoded: (1, n_points, d_model)

            # Mean-pool over non-padded positions
            if obs_key_pad_mask is not None:
                valid = (~obs_key_pad_mask).unsqueeze(-1).float()  # (1, n_points, 1)
                pooled = (obs_encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            else:
                pooled = obs_encoded.mean(dim=1)
            # pooled: (1, d_model)

            all_embeddings.append(pooled[0].cpu().numpy())
            all_tiers.append(eq.tier)
            all_families.append(family)
            all_eq_ids.append(eq.id)

    embeddings = np.stack(all_embeddings, axis=0)  # (N, d_model)
    return embeddings, all_tiers, all_families, all_eq_ids


def generate_embedding_space(
    model: PhysMDT,
    device: torch.device,
    outdir: str,
    quick: bool = False,
    seed: int = 42,
) -> None:
    """Generate 2D embedding space visualization."""
    print("\n--- Generating embedding space visualization ---")

    ReducerClass, reducer_name = _try_import_umap()
    if ReducerClass is None:
        print("  [SKIP] Neither UMAP nor sklearn.manifold.TSNE available.")
        print("         Install umap-learn or scikit-learn to generate this figure.")
        return

    n_samples_per_eq = 5 if not quick else 2
    n_points = model.config.max_obs_points

    print(f"  Collecting encoder embeddings ({n_samples_per_eq} samples/equation)...")
    embeddings, tiers, families, eq_ids = _collect_embeddings(
        model, device, n_samples_per_eq=n_samples_per_eq, n_points=n_points, seed=seed
    )
    print(f"  Collected {len(embeddings)} embedding vectors.")

    if len(embeddings) < 10:
        print("  [SKIP] Too few samples for dimensionality reduction.")
        return

    # Run dimensionality reduction
    print(f"  Running {reducer_name}...")
    if reducer_name == "UMAP":
        reducer = ReducerClass(
            n_components=2,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            random_state=seed,
        )
    else:
        perplexity = min(30.0, max(5.0, len(embeddings) / 5.0))
        reducer = ReducerClass(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            n_iter=1000,
        )
    coords_2d = reducer.fit_transform(embeddings)  # (N, 2)

    # --- Colour by tier ---
    tier_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}
    unique_tiers = sorted(set(tiers))

    # --- Marker by family ---
    family_markers = {
        "mechanics": "o",
        "gravity": "^",
        "oscillation": "s",
        "electricity": "D",
        "thermodynamics": "P",
        "rotation": "X",
        "fluid": "v",
        "relativity": "*",
        "other": "h",
    }
    unique_families = sorted(set(families))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(embeddings)):
        c = tier_colors.get(tiers[i], "#333333")
        m = family_markers.get(families[i], "o")
        ax.scatter(
            coords_2d[i, 0],
            coords_2d[i, 1],
            c=c,
            marker=m,
            s=30,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
        )

    # Legend: tiers
    tier_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=tier_colors[t],
            markersize=8,
            label=f"Tier {t}",
        )
        for t in unique_tiers
    ]
    # Legend: families
    family_handles = [
        Line2D(
            [0], [0],
            marker=family_markers.get(f, "o"),
            color="gray",
            markerfacecolor="gray",
            markersize=7,
            label=f.capitalize(),
            linestyle="None",
        )
        for f in unique_families
    ]

    legend1 = ax.legend(
        handles=tier_handles,
        title="Tier",
        loc="upper left",
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=family_handles,
        title="Family",
        loc="upper right",
        framealpha=0.9,
        fontsize=7,
        title_fontsize=8,
    )

    ax.set_xlabel(f"{reducer_name} Dimension 1")
    ax.set_ylabel(f"{reducer_name} Dimension 2")
    ax.set_title(
        f"Observation Encoder Embedding Space ({reducer_name})\n"
        f"{len(embeddings)} samples from {len(get_training_equations())} training equations"
    )

    fig.tight_layout()
    fname = os.path.join(outdir, "embedding_space.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ===================================================================
# 3. Attention Pattern Visualization
# ===================================================================

@torch.no_grad()
def _collect_attention_weights(
    model: PhysMDT,
    observations: torch.Tensor,
    obs_mask: torch.Tensor,
    tokens: torch.Tensor,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Run a forward pass and collect cross-attention weights from all layers.

    Uses forward hooks on PhysMDTDecoderLayer.cross_attn modules to capture
    the attention weight matrices.

    Returns:
        self_attn_weights: list of (nhead, seq_len, seq_len) arrays per layer.
        cross_attn_weights: list of (nhead, seq_len, n_obs) arrays per layer.
    """
    cross_weights_by_layer: List[np.ndarray] = []
    self_weights_by_layer: List[np.ndarray] = []
    hooks = []

    def _make_cross_hook(storage):
        def hook_fn(module, args, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # attn_weights: (batch, tgt_len, src_len) when batch_first=True
            # or could be (batch, nhead, tgt_len, src_len) for average_attn_weights=False
            attn_w = output[1]
            if attn_w is not None:
                storage.append(attn_w[0].cpu().numpy())  # batch=0
        return hook_fn

    def _make_self_hook(storage):
        def hook_fn(module, args, output):
            attn_w = output[1]
            if attn_w is not None:
                storage.append(attn_w[0].cpu().numpy())
        return hook_fn

    # Register hooks on each decoder layer's cross_attn and self_attn
    for layer in model.decoder_layers:
        h_cross = layer.cross_attn.register_forward_hook(
            _make_cross_hook(cross_weights_by_layer)
        )
        h_self = layer.self_attn.register_forward_hook(
            _make_self_hook(self_weights_by_layer)
        )
        hooks.extend([h_cross, h_self])

    # Run forward pass
    token_mask = torch.zeros_like(tokens, dtype=torch.bool)
    model(
        observations=observations,
        obs_mask=obs_mask,
        masked_tokens=tokens,
        token_mask=token_mask,
    )

    # Remove hooks
    for h in hooks:
        h.remove()

    return self_weights_by_layer, cross_weights_by_layer


def _select_attention_equations() -> List[Equation]:
    """Pick 2 equations for attention visualization (one simple, one complex)."""
    training = get_training_equations()
    # Pick first Tier 2 and first Tier 4
    t2 = [eq for eq in training if eq.tier == 2]
    t4 = [eq for eq in training if eq.tier == 4]
    selected = []
    if t2:
        selected.append(t2[0])
    if t4:
        selected.append(t4[0])
    if not selected and training:
        selected.append(training[0])
    return selected


def generate_attention_patterns(
    model: PhysMDT,
    device: torch.device,
    outdir: str,
    quick: bool = False,
    seed: int = 42,
) -> None:
    """Generate cross-attention weight visualizations."""
    print("\n--- Generating attention pattern visualizations ---")
    tokenizer = ExprTokenizer()
    equations = _select_attention_equations()

    if not equations:
        print("  [SKIP] No equations available for attention visualization.")
        return

    n_eqs = len(equations)
    # We show the last decoder layer's cross-attention for each equation
    # Layout: one row per equation, each showing the cross-attention heatmap
    fig, axes = plt.subplots(n_eqs, 1, figsize=(10, 4.0 * n_eqs), squeeze=False)

    for eq_idx, eq in enumerate(equations):
        print(f"  Processing: {eq.id} -- {eq.name}")

        ds = PhysicsDataset(
            equations=[eq],
            n_samples=1,
            n_points=model.config.max_obs_points,
            noise_level=0.0,
            seed=seed,
        )
        if len(ds) == 0:
            print(f"    [SKIP] Could not generate data for {eq.name}")
            continue

        sample = ds[0]
        obs = sample["observations"].unsqueeze(0).to(device)
        obs_mask = sample["obs_mask"].unsqueeze(0).to(device)
        token_ids = sample["tokens"].unsqueeze(0).to(device)

        _self_w, cross_w = _collect_attention_weights(model, obs, obs_mask, token_ids)

        if not cross_w:
            print("    [SKIP] No cross-attention weights captured.")
            continue

        # Use the last layer's cross-attention
        attn_map = cross_w[-1]  # shape (seq_len, n_obs) or (nhead, seq_len, n_obs)
        if attn_map.ndim == 3:
            # Average over heads
            attn_map = attn_map.mean(axis=0)
        # attn_map: (seq_len, n_obs)

        # Determine display lengths
        tok_list = token_ids[0].cpu().tolist()
        tok_display_len = len(tok_list)
        for j, t in enumerate(tok_list):
            if t == PAD_IDX:
                tok_display_len = j
                break
            if t == EOS_IDX:
                tok_display_len = j + 1
                break
        tok_display_len = max(tok_display_len, 2)

        # Determine observation display length (how many obs points are non-padded)
        obs_mask_np = obs_mask[0].cpu().numpy()
        obs_valid = obs_mask_np.sum(axis=-1)  # (n_points,)
        n_obs_valid = int((obs_valid > 0).sum())
        n_obs_display = min(n_obs_valid, attn_map.shape[1])
        # Cap for readability
        n_obs_cap = 30 if not quick else 15
        n_obs_display = min(n_obs_display, n_obs_cap)

        attn_display = attn_map[:tok_display_len, :n_obs_display]

        token_names = [tokenizer.get_token_str(tok_list[j]) for j in range(tok_display_len)]

        ax = axes[eq_idx, 0]
        im = ax.imshow(
            attn_display,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
        )

        ax.set_xlabel("Observation Point Index")
        ax.set_ylabel("Expression Token")
        ax.set_title(
            f"Cross-Attention (last layer): {eq.name}\n({eq.id}, Tier {eq.tier})",
            fontsize=10,
        )

        # Y-axis: token names
        ax.set_yticks(range(tok_display_len))
        ax.set_yticklabels(token_names, fontsize=7)

        # X-axis: observation indices (subset)
        if n_obs_display <= 20:
            ax.set_xticks(range(n_obs_display))
            ax.set_xticklabels(range(n_obs_display), fontsize=7)
        else:
            step = max(1, n_obs_display // 10)
            xticks = list(range(0, n_obs_display, step))
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Attention Weight", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fname = os.path.join(outdir, "attention_patterns.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ===================================================================
# Main
# ===================================================================

def main():
    args = _parse_args()
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Quick mode: {args.quick}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {args.outdir}")

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    model, config = _load_checkpoint(args.checkpoint, device)
    if model is None:
        print("[FATAL] Could not load model. Exiting.")
        sys.exit(1)

    n_params = model.count_parameters()
    print(f"Model loaded: {n_params:,} parameters ({n_params / 1e6:.1f}M)")

    # --- 1. Refinement Trajectory Heatmaps ---
    n_refine_steps = 64 if not args.quick else 16
    generate_refinement_heatmaps(
        model=model,
        device=device,
        outdir=args.outdir,
        n_steps=n_refine_steps,
        quick=args.quick,
        seed=args.seed,
    )

    # --- 2. Embedding Space ---
    generate_embedding_space(
        model=model,
        device=device,
        outdir=args.outdir,
        quick=args.quick,
        seed=args.seed,
    )

    # --- 3. Attention Patterns ---
    generate_attention_patterns(
        model=model,
        device=device,
        outdir=args.outdir,
        quick=args.quick,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Visualization generation complete.")
    print(f"Output directory: {args.outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
