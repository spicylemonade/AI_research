#!/usr/bin/env python3
"""Analyze learned token embeddings from a trained PhysMDT model.

Produces:
  - figures/embedding_tsne.png   -- t-SNE scatter of all token embeddings
                                    coloured by category
  - figures/embedding_heatmap.png -- cosine-similarity heatmap for selected
                                     physics-related tokens
  - results/embedding_analysis.json -- analogy test results and supporting data

Usage:
    python scripts/analyze_embeddings.py
"""

import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.tokenizer import PhysicsTokenizer
from src.phys_mdt import PhysMDT
from src.token_algebra import TokenAlgebra

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(REPO_ROOT, "results", "phys_mdt", "model.pt")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
D_FF = 256
MAX_SEQ_LEN = 48
MAX_VARS = 5
N_POINTS = 10

# Tokens for the cosine-similarity heatmap
HEATMAP_TOKENS = [
    "F", "m", "a", "v", "x", "t",
    "E_energy", "KE", "PE", "p_momentum",
    "omega", "g_accel", "G_const",
    "add", "mul", "div", "pow",
    "sin", "cos", "sqrt",
]

# Category definitions for t-SNE colouring
# Ranges reference _VOCAB_LIST in src/tokenizer.py
CATEGORY_RANGES = {
    "special":   (0, 5),     # [PAD]..[UNK]
    "operators": (6, 11),    # add..neg
    "functions": (12, 23),   # sin..cosh
    "variables": (24, 68),   # x..x0
    "constants": (69, 113),  # g_accel..CONST_END
}

CATEGORY_COLOURS = {
    "operators": "#D62728",   # red  (colorblind-safe)
    "variables": "#1F77B4",   # blue
    "constants": "#2CA02C",   # green
    "special":   "#7F7F7F",   # gray
    "functions": "#FF7F0E",   # orange
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_category(token_id: int) -> str:
    """Return the category name for a given token ID."""
    for cat, (lo, hi) in CATEGORY_RANGES.items():
        if lo <= token_id <= hi:
            return cat
    # Anything beyond the explicit ranges (structure tokens 114+)
    return "special"


def _publication_style() -> None:
    """Apply publication-quality matplotlib styling."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


# ---------------------------------------------------------------------------
# 1 & 2. Load model and extract embeddings
# ---------------------------------------------------------------------------

def load_model_and_embeddings(tokenizer: PhysicsTokenizer):
    """Load trained PhysMDT and return the model plus the embedding matrix."""
    model = PhysMDT(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        max_vars=MAX_VARS,
        n_points=N_POINTS,
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # vocab_size x d_model
    embedding_matrix = model.token_embedding.weight.detach().cpu().numpy()
    return model, embedding_matrix


# ---------------------------------------------------------------------------
# 3. Token name lookup
# ---------------------------------------------------------------------------

def get_token_names(tokenizer: PhysicsTokenizer):
    """Return a list of human-readable token names indexed by token ID."""
    return [tokenizer.id_to_token[i] for i in range(tokenizer.vocab_size)]


# ---------------------------------------------------------------------------
# 4. t-SNE visualisation
# ---------------------------------------------------------------------------

def create_tsne_plot(
    embedding_matrix: np.ndarray,
    token_names: list,
    save_path: str,
) -> np.ndarray:
    """Create and save a t-SNE scatter plot of token embeddings.

    Returns the 2-D t-SNE coordinates (vocab_size x 2).
    """
    _publication_style()

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(token_names) - 1),
        random_state=42,
        max_iter=1000,
    )
    coords = tsne.fit_transform(embedding_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each category separately for a proper legend
    for cat in ["operators", "functions", "variables", "constants", "special"]:
        indices = [
            i for i in range(len(token_names)) if _token_category(i) == cat
        ]
        if not indices:
            continue
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            c=CATEGORY_COLOURS[cat],
            label=cat.capitalize(),
            s=30,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.3,
        )

        # Annotate a representative subset to avoid clutter
        # For variables/operators/functions annotate all; for constants
        # annotate only named ones (not digit/float tokens).
        for idx in indices:
            name = token_names[idx]
            if cat == "constants" and name.startswith(("C_", "D_", "E_", "INT_", "CONST", "DOT")):
                continue  # skip float-encoding digit tokens
            if cat == "special" and idx > 5:
                continue  # skip structure tokens for clarity
            ax.annotate(
                name,
                (coords[idx, 0], coords[idx, 1]),
                fontsize=7,
                alpha=0.8,
                textcoords="offset points",
                xytext=(3, 3),
            )

    ax.set_title("t-SNE of PhysMDT Token Embeddings")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", framealpha=0.9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved t-SNE plot to {save_path}")
    return coords


# ---------------------------------------------------------------------------
# 5. Cosine-similarity heatmap
# ---------------------------------------------------------------------------

def create_heatmap(
    model: PhysMDT,
    tokenizer: PhysicsTokenizer,
    save_path: str,
) -> np.ndarray:
    """Create and save a cosine-similarity heatmap for selected tokens.

    Returns the similarity matrix (N x N) as a numpy array.
    """
    _publication_style()

    algebra = TokenAlgebra(model)

    # Resolve token IDs; skip any token not in the vocabulary
    valid_tokens = []
    valid_ids = []
    for tok in HEATMAP_TOKENS:
        tid = tokenizer.token_to_id.get(tok)
        if tid is not None:
            valid_tokens.append(tok)
            valid_ids.append(tid)

    sim_matrix = algebra.pairwise_similarity(valid_ids).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(len(valid_tokens)))
    ax.set_xticklabels(valid_tokens, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(valid_tokens)))
    ax.set_yticklabels(valid_tokens, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=11)

    ax.set_title("Cosine Similarity of Physics Token Embeddings")

    # Annotate each cell with the rounded value
    for i in range(len(valid_tokens)):
        for j in range(len(valid_tokens)):
            val = sim_matrix[i, j]
            colour = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=colour)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved heatmap to {save_path}")
    return sim_matrix


# ---------------------------------------------------------------------------
# 6. Analogy tests
# ---------------------------------------------------------------------------

def _resolve_ids(tokenizer: PhysicsTokenizer, names: list) -> list:
    """Convert a list of token name strings to IDs, raising on miss."""
    ids = []
    for n in names:
        tid = tokenizer.token_to_id.get(n)
        if tid is None:
            raise KeyError(f"Token '{n}' not found in vocabulary")
        ids.append(tid)
    return ids


def run_analogy_tests(
    model: PhysMDT,
    tokenizer: PhysicsTokenizer,
) -> list:
    """Run 5 physics analogy tests and return structured results.

    Each analogy is computed via vector arithmetic in embedding space using
    ``TokenAlgebra``.  For each test we report the top-5 nearest neighbours.
    """
    algebra = TokenAlgebra(model)
    token_names = get_token_names(tokenizer)

    # Helper: compute a - b + c, return top-5 neighbours
    def _analogy_top5(a_name, b_name, c_name, description):
        a_id, b_id, c_id = _resolve_ids(tokenizer, [a_name, b_name, c_name])
        vec = algebra.get_embedding(a_id) - algebra.get_embedding(b_id) + algebra.get_embedding(c_id)
        neighbours = algebra.project_nearest(vec, top_k=5, exclude_special=True)
        return {
            "description": description,
            "formula": f"{a_name} - {b_name} + {c_name}",
            "a": a_name,
            "b": b_name,
            "c": c_name,
            "top5": [
                {"token": token_names[tid], "token_id": tid, "similarity": round(sim, 4)}
                for tid, sim in neighbours
            ],
        }

    # Helper: similarity between two tokens + top-5 neighbours of midpoint
    def _similarity_top5(a_name, b_name, description):
        a_id, b_id = _resolve_ids(tokenizer, [a_name, b_name])
        sim = algebra.cosine_similarity(a_id, b_id)
        midpoint = algebra.interpolate(a_id, b_id, alpha=0.5)
        neighbours = algebra.project_nearest(midpoint, top_k=5, exclude_special=True)
        return {
            "description": description,
            "formula": f"similarity({a_name}, {b_name})",
            "a": a_name,
            "b": b_name,
            "cosine_similarity": round(sim, 4),
            "midpoint_top5": [
                {"token": token_names[tid], "token_id": tid, "similarity": round(s, 4)}
                for tid, s in neighbours
            ],
        }

    results = []

    # ----- Analogy 1: F:mul(m,a) :: E_energy:mul(m,mul(v,v)) -----
    # "Force is to F=ma as energy is to E=mv^2"
    # Vector: F - mul + E_energy  (operator-relation analogy)
    results.append(_analogy_top5(
        "E_energy", "F", "mul",
        "F:mul(m,a) :: E_energy:mul(m,mul(v,v)) -- force is to F=ma as energy is to E=mv^2",
    ))

    # ----- Analogy 2: v:div(x,t) :: a:div(v,t) -----
    # "Velocity is to x/t as acceleration is to v/t"
    # Vector: a - v + div  (derivative-chain analogy)
    results.append(_analogy_top5(
        "a", "v", "div",
        "v:div(x,t) :: a:div(v,t) -- velocity is to x/t as acceleration is to v/t",
    ))

    # ----- Analogy 3: KE:mul :: PE:mul -----
    # Both kinetic and potential energy involve products
    # Vector: PE - KE + mul
    results.append(_analogy_top5(
        "PE", "KE", "mul",
        "KE:mul :: PE:mul -- both kinetic and potential energy are products",
    ))

    # ----- Analogy 4: sin:cos (function similarity) -----
    results.append(_similarity_top5(
        "sin", "cos",
        "sin:cos -- trigonometric function similarity",
    ))

    # ----- Analogy 5: add:sub :: mul:div -----
    # Arithmetic analogies: add is to sub as mul is to div
    # Vector: mul - add + sub
    results.append(_analogy_top5(
        "mul", "add", "sub",
        "add:sub :: mul:div -- arithmetic analogies",
    ))

    return results


# ---------------------------------------------------------------------------
# 7. Save everything to JSON
# ---------------------------------------------------------------------------

def save_results(
    analogy_results: list,
    tsne_coords: np.ndarray,
    sim_matrix: np.ndarray,
    token_names: list,
    save_path: str,
) -> None:
    """Persist all analysis artefacts to a single JSON file."""
    output = {
        "analogy_tests": analogy_results,
        "tsne_coordinates": {
            token_names[i]: [round(float(tsne_coords[i, 0]), 4),
                             round(float(tsne_coords[i, 1]), 4)]
            for i in range(len(token_names))
        },
        "heatmap_tokens": HEATMAP_TOKENS,
        "similarity_matrix": [
            [round(float(v), 4) for v in row] for row in sim_matrix
        ],
        "token_categories": {
            token_names[i]: _token_category(i) for i in range(len(token_names))
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved analysis results to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PhysMDT Embedding Analysis")
    print("=" * 60)

    # Step 1-2: Load model and extract embedding matrix
    tokenizer = PhysicsTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model, embedding_matrix = load_model_and_embeddings(tokenizer)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Step 3: Get token names
    token_names = get_token_names(tokenizer)

    # Step 4: t-SNE visualisation
    tsne_path = os.path.join(FIGURES_DIR, "embedding_tsne.png")
    tsne_coords = create_tsne_plot(embedding_matrix, token_names, tsne_path)

    # Step 5: Cosine-similarity heatmap
    heatmap_path = os.path.join(FIGURES_DIR, "embedding_heatmap.png")
    sim_matrix = create_heatmap(model, tokenizer, heatmap_path)

    # Step 6: Analogy tests
    print("\nRunning analogy tests...")
    analogy_results = run_analogy_tests(model, tokenizer)
    for i, res in enumerate(analogy_results, 1):
        print(f"\n  Analogy {i}: {res['description']}")
        if "formula" in res and "top5" in res:
            print(f"    Vector: {res['formula']}")
            print(f"    Top-5 nearest neighbours:")
            for nbr in res["top5"]:
                print(f"      {nbr['token']:>15s}  (sim={nbr['similarity']:.4f})")
        elif "cosine_similarity" in res:
            print(f"    Cosine similarity: {res['cosine_similarity']:.4f}")
            print(f"    Midpoint top-5 nearest neighbours:")
            for nbr in res["midpoint_top5"]:
                print(f"      {nbr['token']:>15s}  (sim={nbr['similarity']:.4f})")

    # Step 7: Save everything
    results_path = os.path.join(RESULTS_DIR, "embedding_analysis.json")
    save_results(analogy_results, tsne_coords, sim_matrix, token_names, results_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
