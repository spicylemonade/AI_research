#!/usr/bin/env python3
"""
Embedding Analysis for PhysMDT (Item 022 of research rubric).

Analyzes the learned token embeddings from the trained PhysMDT model to
investigate physics knowledge emergence in the embedding space.

Produces:
    - figures/embedding_tsne.png       : t-SNE visualization by physical category
    - figures/embedding_similarity.png : Cosine similarity heatmap of key tokens
    - results/embeddings/analysis.json : Quantitative results (analogies, cluster stats)

References:
    - arc2025architects: token algebra in continuous embedding space
    - lample2020deep: Deep Learning for Symbolic Mathematics
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Matplotlib backend must be set before importing pyplot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional: sklearn for t-SNE
# ---------------------------------------------------------------------------
try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARNING] scikit-learn not installed; t-SNE plot will be skipped.")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so that src/ is importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from src.tokenizer import (
    TOKEN_TO_IDX,
    IDX_TO_TOKEN,
    VOCAB_SIZE,
    STRUCTURAL_TOKENS,
    ARITHMETIC_OPS,
    TRIG_FUNCTIONS,
    TRANSCENDENTAL,
    PHYSICS_VARS,
    NUMERIC_CONSTANTS,
    FLOAT_TOKENS,
)
from src.phys_mdt import build_phys_mdt

# ===========================================================================
# Configuration
# ===========================================================================

MODEL_PATH = os.path.join(REPO_ROOT, 'results', 'phys_mdt', 'model.pt')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'embeddings')

# Model architecture (must match checkpoint)
MODEL_D_MODEL = 128
MODEL_N_LAYERS = 3
MODEL_N_HEADS = 4

# Physical category definitions for colouring the t-SNE plot
CATEGORY_OPERATORS = ARITHMETIC_OPS                 # +, -, *, /, ^, neg
CATEGORY_TRIG = TRIG_FUNCTIONS                      # sin, cos, tan, ...
CATEGORY_PHYSICS = PHYSICS_VARS                     # m, M, g, F, ...
CATEGORY_NUMERIC = NUMERIC_CONSTANTS                # 0..9, pi, e_const, ...
CATEGORY_TRANSCENDENTAL = TRANSCENDENTAL            # exp, log, ln, sqrt, abs

# Colour map for categories
CATEGORY_COLOURS = {
    'operators':      'red',
    'trig':           'blue',
    'physics_vars':   'green',
    'numeric':        'orange',
    'transcendental': 'purple',
}

# Key physics tokens for the cosine similarity heatmap
HEATMAP_TOKENS = [
    'm', 'M', 'g', 'F', 'E', 'v', 'a', 't', 'x', 'r',
    'p', 'omega', 'k', 'sin', 'cos', '+', '*', '^',
]

# Tokens to annotate on the t-SNE plot
LABEL_TOKENS = [
    'm', 'M', 'g', 'F', 'E', 'v', 'a', 't', 'x', 'r', 'p', 'omega', 'k',
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
    '+', '-', '*', '/', '^',
    '0', '1', '2', 'pi', 'e_const',
]


# ===========================================================================
# Helper utilities
# ===========================================================================

def _token_category(token: str) -> str | None:
    """Return the category name for a token, or None for structural/float."""
    if token in CATEGORY_OPERATORS:
        return 'operators'
    if token in CATEGORY_TRIG:
        return 'trig'
    if token in CATEGORY_TRANSCENDENTAL:
        return 'transcendental'
    if token in CATEGORY_PHYSICS:
        return 'physics_vars'
    if token in CATEGORY_NUMERIC:
        return 'numeric'
    return None  # structural or float placeholder -- skip


def load_model() -> torch.nn.Module:
    """Load the trained PhysMDT model from disk."""
    print(f"Loading model from {MODEL_PATH} ...")
    model = build_phys_mdt(
        d_model=MODEL_D_MODEL,
        n_layers=MODEL_N_LAYERS,
        n_heads=MODEL_N_HEADS,
    )
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded  --  {model.count_parameters():,} parameters")
    return model


def extract_embeddings(model: torch.nn.Module) -> torch.Tensor:
    """Return the (vocab_size, d_model) token embedding weight matrix."""
    return model.token_embedding.weight.detach().clone()


# ===========================================================================
# (a) t-SNE Visualisation
# ===========================================================================

def run_tsne_visualisation(embeddings: torch.Tensor) -> None:
    """Produce a t-SNE scatter plot coloured by physical category."""
    if not HAS_SKLEARN:
        print("[SKIP] t-SNE visualisation -- sklearn not available.")
        return

    print("Running t-SNE visualisation ...")

    # Collect non-structural tokens with their categories
    indices = []
    categories = []
    token_labels = []

    for idx in range(VOCAB_SIZE):
        token = IDX_TO_TOKEN[idx]
        cat = _token_category(token)
        if cat is None:
            continue  # skip structural / float placeholder tokens
        indices.append(idx)
        categories.append(cat)
        token_labels.append(token)

    emb_subset = embeddings[indices].numpy()

    # Adaptive perplexity: must be less than the number of samples
    n_samples = len(indices)
    perplexity = min(30, max(5, n_samples // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        learning_rate='auto',
        init='pca',
        n_iter=1000,
    )
    coords = tsne.fit_transform(emb_subset)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    for cat_name, colour in CATEGORY_COLOURS.items():
        mask = [c == cat_name for c in categories]
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys, c=colour, label=cat_name, alpha=0.7, s=60, edgecolors='k', linewidths=0.3)

    # Label selected tokens
    for i, tok in enumerate(token_labels):
        if tok in LABEL_TOKENS:
            ax.annotate(
                tok,
                (coords[i, 0], coords[i, 1]),
                fontsize=8,
                fontweight='bold',
                ha='center',
                va='bottom',
                textcoords='offset points',
                xytext=(0, 5),
            )

    ax.set_title('t-SNE of PhysMDT Token Embeddings (coloured by physical category)', fontsize=14)
    ax.set_xlabel('t-SNE component 1', fontsize=12)
    ax.set_ylabel('t-SNE component 2', fontsize=12)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'embedding_tsne.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved t-SNE plot -> {out_path}")


# ===========================================================================
# (b) Cosine Similarity Heatmap
# ===========================================================================

def run_similarity_heatmap(embeddings: torch.Tensor) -> np.ndarray:
    """Compute and plot a pairwise cosine similarity heatmap for key tokens."""
    print("Computing cosine similarity heatmap ...")

    # Gather embeddings for the requested tokens
    valid_tokens = []
    emb_list = []
    for tok in HEATMAP_TOKENS:
        idx = TOKEN_TO_IDX.get(tok)
        if idx is not None:
            valid_tokens.append(tok)
            emb_list.append(embeddings[idx])

    emb_matrix = torch.stack(emb_list)                       # (n, d)
    emb_normed = F.normalize(emb_matrix, dim=-1)
    sim_matrix = (emb_normed @ emb_normed.T).numpy()          # (n, n)

    # Plot
    n = len(valid_tokens)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(valid_tokens, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(valid_tokens, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            colour = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color=colour)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=11)

    ax.set_title('Pairwise Cosine Similarity of Key Physics Token Embeddings', fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, 'embedding_similarity.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved similarity heatmap -> {out_path}")

    return sim_matrix


# ===========================================================================
# (c) Vector Analogy Tests
# ===========================================================================

def _cosine_sim_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def run_analogy_tests(embeddings: torch.Tensor) -> list:
    """
    Test physics vector analogies and return ranked results.

    Each analogy is expressed as: result = tok_A - tok_B + tok_C
    We then find the nearest vocabulary tokens to the result vector.
    """
    print("Running vector analogy tests ...")

    normed = F.normalize(embeddings, dim=-1)

    def _get(tok: str) -> torch.Tensor:
        return embeddings[TOKEN_TO_IDX[tok]]

    def _nearest(query: torch.Tensor, exclude_tokens: list, top_k: int = 10):
        q_normed = F.normalize(query.unsqueeze(0), dim=-1)
        sims = (q_normed @ normed.T).squeeze(0)
        # Exclude source tokens
        for tok in exclude_tokens:
            idx = TOKEN_TO_IDX.get(tok)
            if idx is not None:
                sims[idx] = -2.0
            # Also exclude structural tokens
        for tok in STRUCTURAL_TOKENS + FLOAT_TOKENS:
            idx = TOKEN_TO_IDX.get(tok)
            if idx is not None:
                sims[idx] = -2.0
        top_vals, top_idx = sims.topk(top_k)
        return [(IDX_TO_TOKEN[i.item()], v.item()) for v, i in zip(top_vals, top_idx)]

    # Define analogies as (description, expression_fn, source_tokens, expected)
    analogies = [
        {
            'description': 'F - m + v ≈ p?  (force minus mass plus velocity ≈ momentum)',
            'compute':     lambda: _get('F') - _get('m') + _get('v'),
            'exclude':     ['F', 'm', 'v'],
            'expected':    'p',
        },
        {
            'description': 'E - m + v ≈ ?   (energy minus mass + velocity ≈ kinetic?)',
            'compute':     lambda: _get('E') - _get('m') + _get('v'),
            'exclude':     ['E', 'm', 'v'],
            'expected':    'K',
        },
        {
            'description': 'v - a + F ≈ ?   (velocity - acceleration + force)',
            'compute':     lambda: _get('v') - _get('a') + _get('F'),
            'exclude':     ['v', 'a', 'F'],
            'expected':    'p',
        },
        {
            'description': 'sin - cos ≈ ?   (difference between trig functions)',
            'compute':     lambda: _get('sin') - _get('cos'),
            'exclude':     ['sin', 'cos'],
            'expected':    'tan',
        },
        {
            'description': 'E - K + U ≈ ?   (energy - kinetic + potential)',
            'compute':     lambda: _get('E') - _get('K') + _get('U'),
            'exclude':     ['E', 'K', 'U'],
            'expected':    'E',
        },
        {
            'description': 'omega - t + x ≈ ?  (angular freq - time + position)',
            'compute':     lambda: _get('omega') - _get('t') + _get('x'),
            'exclude':     ['omega', 't', 'x'],
            'expected':    'k',
        },
        {
            'description': 'a * embedding - F/m direction test: a - g ≈ ?',
            'compute':     lambda: _get('a') - _get('g'),
            'exclude':     ['a', 'g'],
            'expected':    'F',
        },
        {
            'description': 'p - m + F ≈ ?   (momentum - mass + force)',
            'compute':     lambda: _get('p') - _get('m') + _get('F'),
            'exclude':     ['p', 'm', 'F'],
            'expected':    'a',
        },
        {
            'description': 'exp - log ≈ ?   (inverse transcendental pair)',
            'compute':     lambda: _get('exp') - _get('log'),
            'exclude':     ['exp', 'log'],
            'expected':    'sqrt',
        },
        {
            'description': 'v - t + x ≈ ?   (velocity - time + position)',
            'compute':     lambda: _get('v') - _get('t') + _get('x'),
            'exclude':     ['v', 't', 'x'],
            'expected':    'a',
        },
    ]

    results = []
    for entry in analogies:
        query = entry['compute']()
        neighbours = _nearest(query, entry['exclude'], top_k=10)
        expected = entry['expected']

        # Check if expected token appears in top-10
        expected_rank = None
        expected_sim = None
        for rank, (tok, sim) in enumerate(neighbours, 1):
            if tok == expected:
                expected_rank = rank
                expected_sim = sim
                break

        # Also compute direct cosine similarity with expected token
        if expected in TOKEN_TO_IDX:
            direct_sim = _cosine_sim_vec(query, _get(expected))
        else:
            direct_sim = None

        result_entry = {
            'description': entry['description'],
            'expected': expected,
            'expected_rank': expected_rank,
            'expected_cosine_sim': round(expected_sim, 6) if expected_sim is not None else None,
            'direct_cosine_sim': round(direct_sim, 6) if direct_sim is not None else None,
            'top_10': [(tok, round(sim, 6)) for tok, sim in neighbours],
        }
        results.append(result_entry)

        # Pretty print
        top1_tok, top1_sim = neighbours[0]
        exp_info = ''
        if expected_rank is not None:
            exp_info = f' (expected "{expected}" found at rank {expected_rank}, sim={expected_sim:.4f})'
        else:
            direct_str = f', direct sim={direct_sim:.4f}' if direct_sim is not None else ''
            exp_info = f' (expected "{expected}" NOT in top-10{direct_str})'
        print(f"  {entry['description']}")
        print(f"    -> top-1: {top1_tok} (sim={top1_sim:.4f}){exp_info}")

    return results


# ===========================================================================
# (d) Cluster Statistics
# ===========================================================================

def compute_cluster_stats(embeddings: torch.Tensor) -> dict:
    """
    Compute mean within-cluster and between-cluster cosine similarity
    for each physical category.
    """
    print("Computing cluster statistics ...")

    categories_map = {}  # cat_name -> list of token indices
    for idx in range(VOCAB_SIZE):
        tok = IDX_TO_TOKEN[idx]
        cat = _token_category(tok)
        if cat is not None:
            categories_map.setdefault(cat, []).append(idx)

    normed = F.normalize(embeddings, dim=-1)

    within_sims = {}
    for cat, idx_list in categories_map.items():
        if len(idx_list) < 2:
            within_sims[cat] = 1.0
            continue
        cat_emb = normed[idx_list]       # (n_cat, d)
        sim = (cat_emb @ cat_emb.T).numpy()
        n = len(idx_list)
        # Mean off-diagonal similarity
        mask = ~np.eye(n, dtype=bool)
        within_sims[cat] = float(np.mean(sim[mask]))

    # Between-cluster: mean similarity between tokens of different categories
    all_cat_names = sorted(categories_map.keys())
    between_sims = {}
    for i, cat_a in enumerate(all_cat_names):
        for cat_b in all_cat_names[i + 1:]:
            emb_a = normed[categories_map[cat_a]]
            emb_b = normed[categories_map[cat_b]]
            cross_sim = (emb_a @ emb_b.T).numpy()
            key = f"{cat_a} vs {cat_b}"
            between_sims[key] = float(np.mean(cross_sim))

    overall_between = float(np.mean(list(between_sims.values()))) if between_sims else 0.0

    stats = {
        'within_cluster_similarity': {k: round(v, 6) for k, v in within_sims.items()},
        'between_cluster_similarity': {k: round(v, 6) for k, v in between_sims.items()},
        'mean_within_cluster': round(float(np.mean(list(within_sims.values()))), 6),
        'mean_between_cluster': round(overall_between, 6),
    }

    print(f"  Mean within-cluster similarity:  {stats['mean_within_cluster']:.4f}")
    print(f"  Mean between-cluster similarity: {stats['mean_between_cluster']:.4f}")
    for cat, val in within_sims.items():
        print(f"    {cat:20s} within = {val:.4f}")

    return stats


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # Ensure output directories exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load model and extract embeddings
    model = load_model()
    embeddings = extract_embeddings(model)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # 2a. t-SNE visualisation
    run_tsne_visualisation(embeddings)

    # 2b. Cosine similarity heatmap
    sim_matrix = run_similarity_heatmap(embeddings)

    # 2c. Vector analogy tests
    analogy_results = run_analogy_tests(embeddings)

    # 2d. Cluster statistics
    cluster_stats = compute_cluster_stats(embeddings)

    # 3. Save all results to JSON
    output = {
        'model_path': MODEL_PATH,
        'd_model': MODEL_D_MODEL,
        'vocab_size': VOCAB_SIZE,
        'top_10_analogies': analogy_results,
        'cluster_statistics': cluster_stats,
        'heatmap_tokens': HEATMAP_TOKENS,
        'heatmap_similarity_matrix': sim_matrix.tolist() if sim_matrix is not None else None,
    }

    json_path = os.path.join(RESULTS_DIR, 'analysis.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved -> {json_path}")

    # Summary
    print("\n=== Embedding Analysis Summary ===")
    print(f"  Figures:  {FIGURES_DIR}/embedding_tsne.png")
    print(f"           {FIGURES_DIR}/embedding_similarity.png")
    print(f"  Results:  {json_path}")
    n_good = sum(
        1 for a in analogy_results
        if a.get('direct_cosine_sim') is not None and a['direct_cosine_sim'] > 0.6
    )
    print(f"  Analogies with cosine sim > 0.6: {n_good}/{len(analogy_results)}")
    print("Done.")


if __name__ == '__main__':
    main()
