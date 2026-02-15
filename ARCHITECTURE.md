# PhysMDT: Physics Masked Diffusion Transformer — Architecture Document

## Hardware Budget
- **GPU**: Single NVIDIA A100-SXM4-40GB
- **VRAM**: 40 GB (HBM2e)
- **System RAM**: Available system memory
- **Disk**: ~2 TB
- **Constraint**: All training and inference must complete on this single GPU

## Project File Layout

```
repo/
├── ARCHITECTURE.md              # This document
├── README.md                    # Project overview and usage
├── research_rubric.json         # Experiment tracking rubric
├── sources.bib                  # BibTeX references
├── requirements.txt             # Pinned Python dependencies
├── run_all.sh                   # End-to-end pipeline script
│
├── data/
│   ├── __init__.py
│   ├── physics_generator.py     # Synthetic physics dataset generator
│   ├── tokenizer.py             # Symbolic expression tokenizer (prefix notation)
│   └── equations.py             # Equation corpus definition (50+ equations, 5 tiers)
│
├── models/
│   ├── __init__.py
│   ├── ar_baseline.py           # Autoregressive encoder-decoder transformer baseline
│   ├── physmdt.py               # PhysMDT: masked diffusion transformer core
│   ├── refinement.py            # Recursive soft-masking refinement inference
│   └── ttft.py                  # Test-time fine-tuning with LoRA
│
├── training/
│   ├── __init__.py
│   ├── train_baseline.py        # AR baseline training script
│   └── train_physmdt.py         # PhysMDT curriculum training script
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               # Evaluation metrics (symbolic equiv, R², edit dist, etc.)
│
├── figures/                     # Publication-quality figures (PNG + PDF)
├── results/                     # JSON experiment results
├── checkpoints/                 # Model checkpoints (gitignored)
└── paper/
    └── main.tex                 # LaTeX research paper draft
```

## Module Responsibilities and Interfaces

### 1. Data Generation (`data/`)

**`data/equations.py`** — Equation Corpus
- Defines 50+ Newtonian physics equations across 5 complexity tiers
- Each equation: symbolic expression (SymPy), variable names, physical units, tier label
- Designates 10+ held-out equations for zero-shot discovery evaluation
- Interface: `get_training_equations() -> List[Equation]`, `get_held_out_equations() -> List[Equation]`

**`data/physics_generator.py`** — Synthetic Data Generator
- Generates (numerical_observations, symbolic_equation) pairs
- Configurable: noise level (0–20%), num observations (5–200), variable ranges
- Random variable instantiation with Gaussian noise injection
- Outputs memory-mapped PyTorch-compatible datasets
- Interface: `PhysicsDataset(equations, n_samples, noise_level, n_points) -> Dataset`

**`data/tokenizer.py`** — Symbolic Expression Tokenizer
- Prefix notation encoding of symbolic expressions
- Vocabulary: operators, variables (x0–x9), numeric constants, special tokens
- Special tokens: `<SOS>`, `<EOS>`, `<PAD>`, `<MASK>`, `<SEP>`
- Vocabulary size < 200 tokens
- Interface: `encode(expr) -> List[int]`, `decode(tokens) -> sympy.Expr`

### 2. Models (`models/`)

**`models/ar_baseline.py`** — Autoregressive Baseline
- Standard encoder-decoder transformer (~30M params)
- Encoder: processes flattened numerical observation sequences
- Decoder: generates symbolic tokens autoregressively
- Interface: `ARBaseline(config) -> nn.Module`

**`models/physmdt.py`** — Physics Masked Diffusion Transformer
- Core novel architecture (~50–80M params)
- Set-transformer observation encoder (permutation invariant)
- Masked diffusion expression decoder with cross-attention
- Tree-positional encoding (2D RoPE variant for expression trees)
- Dimensional analysis attention bias head
- Interface: `PhysMDT(config) -> nn.Module`

**`models/refinement.py`** — Recursive Soft-Masking Refinement
- Iterative denoising: fully masked → progressively unmasked
- Soft token embeddings via probability-weighted mixture
- Cosine confidence-based unmasking schedule
- K-candidate generation with most-visited voting
- Interface: `refine(model, observations, n_steps=64, n_candidates=8) -> Expression`

**`models/ttft.py`** — Test-Time Fine-Tuning
- LoRA adapters (rank 16–32) on attention layers
- Self-consistency loss: decoded expression must match observations
- Per-problem adaptation in < 60 seconds
- Interface: `test_time_finetune(model, observations, n_steps=128) -> model`

### 3. Training (`training/`)

**`training/train_baseline.py`** — Baseline Training
- AdamW + cosine LR schedule, bf16 mixed precision
- Target: convergence on Tier 1–2 in ~4 hours

**`training/train_physmdt.py`** — Curriculum Training
- Phase 1: Tier 1–2 only
- Phase 2: add Tier 3 with mixed batches
- Phase 3: full Tier 1–4 with emphasis on harder tiers
- Masking schedule: 90–100% → 30–100% annealing
- Gradient checkpointing for memory efficiency
- Target: < 12 hours total on single A100

### 4. Evaluation (`evaluation/`)

**`evaluation/metrics.py`** — Metrics Suite
- Symbolic Equivalence Accuracy (SymPy simplify)
- Numeric R² Score
- Complexity-Weighted Score (per-tier breakdown)
- Token Edit Distance (Levenshtein)
- Novel Discovery Rate (held-out equations)
- Interface: `evaluate(predictions, ground_truth) -> MetricsDict`

### 5. Visualization (`figures/`)
- Training loss curves
- Per-tier accuracy comparisons
- Refinement trajectory heatmaps
- Embedding space t-SNE/UMAP
- Attention pattern visualizations
- Noise robustness and data efficiency curves
- Ablation results tables and radar charts

## Key Design Decisions

1. **Masked Diffusion over Autoregressive**: Parallel token prediction enables iterative refinement, better for capturing global structure of equations
2. **Prefix Notation**: Unambiguous, no parentheses needed, natural tree structure
3. **Set-Transformer Encoder**: Permutation invariance over observation points reflects the physics (order of measurements doesn't matter)
4. **Tree-Positional Encoding**: Respects hierarchical structure of mathematical expressions
5. **Curriculum Training**: Gradual complexity increase mirrors how humans learn physics
6. **Test-Time Fine-Tuning**: Adapts to specific physical systems without retraining the full model
