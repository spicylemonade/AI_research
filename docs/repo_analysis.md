# Repository Analysis

## Overview

This repository implements **PhysMDT** (Physics Masked Diffusion Transformer), a novel transformer-based system for deriving Newtonian physics equations from numerical observations. The project is organized as a research codebase with clear separation between data generation, model implementation, training scripts, evaluation, and documentation.

## Directory Structure

### `src/` — Core Source Modules

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/tokenizer.py` | Prefix-notation tokenizer for physics equations | `PhysicsTokenizer.encode()`, `.decode()`, vocabulary of ≥140 tokens |
| `src/baseline_ar.py` | Autoregressive encoder-decoder transformer baseline | `BaselineAR`, `ObservationEncoder`, `EquationDecoder` |
| `src/phys_mdt.py` | Physics Masked Diffusion Transformer (core model) | `PhysMDT`, `DualAxisRoPE`, `MaskedDiffusionObjective`, `CrossAttentionEncoder` |
| `src/refinement.py` | Iterative soft-mask refinement inference | `SoftMaskRefinement`, `ColdRestart`, `ConvergenceDetector`, `CandidateTracker` |
| `src/token_algebra.py` | Symbolic manipulation in embedding space | `TokenAlgebra.interpolate()`, `.analogy()`, `.project_nearest()` |
| `src/ttf.py` | Test-time finetuning with LoRA adaptation | `TestTimeFinetuner`, `LoRAAdapter`, `DataAugmenter` |
| `src/physics_loss.py` | Physics-informed loss functions | `DimensionalConsistencyLoss`, `ConservationRegularizer`, `SymmetryLoss` |
| `src/structure_predictor.py` | Dual-model skeleton predictor | `StructurePredictor`, `SkeletonDecoder` |
| `src/metrics.py` | Evaluation metrics suite | `exact_match()`, `symbolic_equivalence()`, `numerical_r2()`, `tree_edit_distance()`, `complexity_penalty()`, `composite_score()` |

### `data/` — Data Generation

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `data/generator.py` | Physics equation dataset generator | `PhysicsDatasetGenerator`, `EquationTemplate`, 60+ templates across 7 families |

### `scripts/` — Training, Evaluation, and Utility Scripts

| File | Purpose |
|------|---------|
| `scripts/train_baseline.py` | Train autoregressive baseline model |
| `scripts/train_phys_mdt.py` | Train full PhysMDT with ablation variants |
| `scripts/eval_sr_baseline.py` | Evaluate PySR/gplearn baselines |
| `scripts/eval_benchmarks.py` | Evaluate on AI Feynman, Nguyen, Strogatz |
| `scripts/eval_challenge.py` | Evaluate on complex challenge equations |
| `scripts/analyze_embeddings.py` | Token embedding analysis and visualization |
| `scripts/statistical_tests.py` | Statistical significance testing |
| `scripts/generate_figures.py` | Publication-quality figure generation |
| `scripts/reproduce.sh` | Full reproducibility script |
| `scripts/update_rubric.py` | Research rubric status updater |

### `tests/` — Unit Tests

| File | Purpose |
|------|---------|
| `tests/test_generator.py` | Tests for data generator (≥15 tests) |
| `tests/test_tokenizer.py` | Tests for tokenizer (≥6 tests) |
| `tests/test_metrics.py` | Tests for evaluation metrics (≥20 tests) |
| `tests/test_phys_mdt.py` | Tests for PhysMDT model (≥8 tests) |

### `configs/` — Configuration Files

| File | Purpose |
|------|---------|
| `configs/` | Training and model configuration files (YAML/JSON) |

### `docs/` — Documentation

| File | Purpose |
|------|---------|
| `docs/repo_analysis.md` | This file — repository structure analysis |
| `docs/literature_review.md` | Comprehensive literature review |
| `docs/benchmarks.md` | SOTA benchmark comparison tables |
| `docs/problem_statement.md` | Formal problem definition and hypotheses |
| `docs/tokenization_design.md` | Tokenization scheme design |
| `docs/arc_analysis.md` | ARC 2025 architecture analysis |
| `docs/key_findings.md` | Synthesized experimental findings |
| `docs/peer_review.md` | Internal peer review |
| `docs/reproducibility_check.md` | Reproducibility verification |

### `results/` — Experimental Results (JSON)

| Directory | Contents |
|-----------|----------|
| `results/baseline_ar/` | AR baseline metrics |
| `results/sr_baseline/` | PySR/gplearn metrics |
| `results/ablations/` | 8-variant ablation results |
| `results/refinement_depth/` | Refinement step sweep results |
| `results/ai_feynman/` | AI Feynman benchmark results |
| `results/nguyen/` | Nguyen benchmark results |
| `results/strogatz/` | Strogatz benchmark results |
| `results/challenge/` | Challenge set results |
| `results/statistical_tests.json` | Statistical significance results |

### `figures/` — Publication-Quality Figures (PNG + PDF)

All figures use consistent seaborn/matplotlib styling with colorblind-friendly palettes.

### Root Files

| File | Purpose |
|------|---------|
| `research_rubric.json` | Research progress tracker (28 items across 5 phases) |
| `sources.bib` | BibTeX bibliography for all cited sources |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview |
| `.gitignore` | Git ignore patterns |

## Inter-Module Dependencies

```
data/generator.py
    └── src/tokenizer.py (uses tokenizer for prefix notation)

src/baseline_ar.py
    ├── src/tokenizer.py (encode/decode equations)
    └── src/metrics.py (evaluation)

src/phys_mdt.py
    ├── src/tokenizer.py (vocabulary, encode/decode)
    ├── src/physics_loss.py (training losses)
    └── src/structure_predictor.py (skeleton constraints)

src/refinement.py
    ├── src/phys_mdt.py (model forward passes)
    └── src/token_algebra.py (embedding manipulation)

src/ttf.py
    └── src/phys_mdt.py (LoRA adaptation of base model)

scripts/train_phys_mdt.py
    ├── data/generator.py (dataset creation)
    ├── src/phys_mdt.py (model)
    ├── src/physics_loss.py (losses)
    ├── src/structure_predictor.py (skeleton model)
    ├── src/refinement.py (inference)
    ├── src/ttf.py (test-time finetuning)
    └── src/metrics.py (evaluation)

scripts/eval_benchmarks.py
    ├── src/phys_mdt.py + src/refinement.py + src/ttf.py
    └── src/metrics.py
```

## Current State

- **Branch**: `research-lab-1771049657`
- **Status**: Fresh repository with rubric defined; implementation pending
- **Existing files**: `research_rubric.json`, `sources.bib`, `scripts/update_rubric.py`
- **Empty directories**: `src/`, `data/`, `tests/`, `configs/`, `figures/`, `results/`
