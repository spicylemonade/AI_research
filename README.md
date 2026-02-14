# PhysMDT: Physics Masked Diffusion Transformer

A novel transformer-based architecture for deriving Newtonian physics equations from raw numerical data using masked diffusion with iterative self-refinement.

## Overview

PhysMDT adapts insights from masked diffusion language models (LLaDA) and the ARChitects' ARC2025 solution to the domain of symbolic regression for physics equation discovery. The model uses:

- **Masked Diffusion Backbone**: Bidirectional transformer that denoises masked equation token sequences
- **Tree-Aware 2D Positional Encoding**: Inspired by Golden Gate RoPE, encodes expression tree structure
- **Soft-Masking Recursion**: Iterative self-refinement without hard discretization between steps
- **Test-Time Finetuning**: Per-equation LoRA adaptation at inference for specialization
- **Physics-Informed Augmentations**: Dimensional analysis, symbolic equivalences, conservation law priors

## Key Results

| Method | Easy SR | Medium SR | Hard SR | Overall SR |
|--------|---------|-----------|---------|------------|
| SymbolicGPT | 53% | 32% | 15% | 33% |
| AR-Baseline (ours) | 53% | 53% | 30% | 43% |
| ODEFormer | 65% | 48% | 30% | 48% |
| PhyE2E | 72% | 55% | 38% | 55% |
| AI-Feynman-2.0 | 80% | 60% | 35% | 58% |
| **PhysMDT-base+SM+TTF** | **77%** | **60%** | **42%** | **60%** |
| **PhysMDT-scaled+SM+TTF** | **83%** | **68%** | **50%** | **67%** |

PhysMDT-scaled with soft-masking and TTF achieves 67% overall solution rate on FSReD, exceeding PhyE2E (55%) and approaching AI-Feynman-2.0 (58%) while operating as a general-purpose learned model rather than a search-based method.

## Project Structure

```
PhysMDT/
├── src/
│   ├── model/
│   │   ├── physmdt.py                  # PhysMDT masked diffusion backbone
│   │   ├── ar_baseline.py              # AR-Baseline (SymbolicGPT-style)
│   │   ├── tree_positional_encoding.py # Tree-aware 2D Golden Gate RoPE
│   │   └── soft_masking.py             # Soft-masking recursion inference
│   ├── data/
│   │   ├── tokenizer.py                # RPN equation tokenizer (200 vocab)
│   │   ├── dataset.py                  # FSReD + procedural dataset pipeline
│   │   └── physics_augmentations.py    # Physics-informed augmentations
│   ├── training/
│   │   └── test_time_finetune.py       # LoRA TTF for per-equation adaptation
│   └── evaluation/
│       └── metrics.py                  # 6-metric evaluation suite
├── configs/
│   ├── physmdt_base.yaml               # Base model (~45M params)
│   └── physmdt_scaled.yaml             # Scaled model (~180M params)
├── scripts/
│   ├── train_physmdt.py                # PhysMDT training pipeline
│   ├── train_baseline.py               # AR-Baseline training
│   ├── eval_physmdt.py                 # Full FSReD evaluation
│   ├── eval_baseline.py                # Baseline evaluation
│   ├── generate_figures.py             # Publication figure generation
│   └── run_all.sh                      # Single-command reproduction
├── tests/
│   ├── test_tokenizer.py               # 32 tests
│   ├── test_dataset.py                 # 16 tests
│   └── test_metrics.py                 # 27 tests
├── results/
│   ├── main_experiment.json            # Main FSReD results
│   ├── ablation_study.json             # Component ablation study
│   ├── newtonian_showcase.json         # 18 Newtonian equation showcase
│   ├── robustness.json                 # Noise/sparsity/OOD evaluation
│   ├── efficiency.json                 # Computational efficiency
│   ├── baseline_results.json           # AR-Baseline results
│   └── training_curves/                # Training logs
├── figures/                            # Publication-quality figures (PNG/PDF)
├── docs/
│   ├── arc2025_analysis.md             # ARChitects solution analysis
│   ├── architecture.md                 # PhysMDT architecture design
│   ├── benchmarks.md                   # Benchmark documentation
│   ├── problem_statement.md            # Research problem formalization
│   └── interpretability_analysis.md    # Model interpretability
├── sources.bib                         # 20 BibTeX references
├── paper.md                            # Research paper
├── requirements.txt                    # Python dependencies
└── research_rubric.json                # Research progress tracking
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with CUDA (recommended for training)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd PhysMDT

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v --timeout=60
```

## Quick Start

### Run all experiments (single command)

```bash
bash scripts/run_all.sh
```

### Train PhysMDT-base

```bash
python scripts/train_physmdt.py --config configs/physmdt_base.yaml
```

### Train PhysMDT-scaled

```bash
python scripts/train_physmdt.py --config configs/physmdt_scaled.yaml
```

### Evaluate on FSReD

```bash
python scripts/eval_physmdt.py --epochs 100
```

### Generate figures

```bash
python scripts/generate_figures.py
```

## Expected Results

Results from `results/main_experiment.json`:

| Configuration | Easy | Medium | Hard | Overall |
|--------------|------|--------|------|---------|
| PhysMDT-base (single-pass) | 60% | 45% | 25% | 43% |
| PhysMDT-base + Soft-Masking | 70% | 55% | 35% | 53% |
| PhysMDT-base + TTF | 67% | 52% | 32% | 50% |
| PhysMDT-base + SM + TTF | 77% | 60% | 42% | 60% |
| PhysMDT-scaled + SM + TTF | 83% | 68% | 50% | 67% |

### Ablation Results (from `results/ablation_study.json`)

| Ablated Component | Overall SR | Drop |
|-------------------|-----------|------|
| Full model (PhysMDT-base+SM+TTF) | 60% | -- |
| w/o Soft-Masking Recursion | 50% | -10% |
| w/o Tree-Aware PE | 52% | -8% |
| w/o Test-Time Finetuning | 53% | -7% |
| w/o Physics Augmentations | 56% | -4% |

## Hardware Requirements

| Task | Hardware | Time |
|------|----------|------|
| Unit tests | CPU | ~2 min |
| AR-Baseline training (50 epochs) | CPU/GPU | ~10 min |
| PhysMDT-base training (100K steps) | 1x GPU | ~1 hour |
| PhysMDT-scaled training (500K steps) | 1x GPU | ~4 hours |
| Full evaluation (120 equations) | 1x GPU | ~30 min |
| Figure generation | CPU | ~1 min |

## Random Seeds

All experiments use fixed random seed 42 for reproducibility:
- `torch.manual_seed(42)`
- `numpy.random.seed(42)`
- Dataset splits use `seed=42`
- TTF uses `seed=42`

## Evaluation Metrics

1. **Solution Rate**: Exact symbolic match after SymPy simplification
2. **R²**: Coefficient of determination on test data points
3. **RMSE**: Root mean square error on test outputs
4. **NED**: Normalized edit distance on expression trees
5. **Symbolic Accuracy**: Token-level accuracy (ignoring PAD)
6. **Inference Time**: Wall-clock time per equation

## Citation

See `sources.bib` for all referenced works, including:
- LLaDA (Nie et al., 2025)
- ARChitects ARC2025 Solution
- SymbolicGPT (Valipour et al., 2021)
- AI Feynman (Udrescu & Tegmark, 2020)
- PhyE2E (2025)

## License

Research use only.
