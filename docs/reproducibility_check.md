# PhysMDT Reproducibility Check

This document verifies each step from `scripts/reproduce.sh` and records the status,
output files, approximate time, and any notes for reproducing the full pipeline.

## Environment

- **Platform**: CPU-only execution (no GPU required)
- **Random seed**: 42 used throughout all scripts for deterministic results
- **Python**: 3.x (tested with standard CPython)
- **Dependencies**: Listed in `requirements.txt`

---

## Step-by-Step Verification

### Step 1: Install dependencies (`pip install -r requirements.txt`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | None (installs packages into environment) |
| Time | ~30s depending on network speed |
| Notes | Packages: torch, numpy, scipy, scikit-learn, matplotlib, sympy, pytest. All available on PyPI. |

---

### Step 2: Run unit tests (`python -m pytest tests/ -v`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | Console output only |
| Time | ~30s |
| Notes | 9 test modules: test_generator, test_metrics, test_phys_mdt, test_physics_loss, test_refinement, test_structure_predictor, test_token_algebra, test_tokenizer, test_ttf. All tests pass on CPU. |

---

### Step 3: Generate dataset and train baseline (`python scripts/train_baseline.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/baseline_ar/model_best.pt`, `results/baseline_ar/model_final.pt`, `results/baseline_ar/metrics.json` |
| Time | ~5 min on CPU |
| Notes | Trains an autoregressive baseline transformer on synthetic physics equation data. Dataset is generated on-the-fly using `data/generator.py` with seed=42. |

---

### Step 4: Evaluate SR baselines (`python scripts/eval_sr_baseline.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/sr_baseline/metrics.json`, `results/baseline_comparison.json` |
| Time | ~2 min |
| Notes | Uses polynomial regression and GBR as SR baselines. gplearn is incompatible with scikit-learn 1.7+; polynomial/GBR baselines are used instead. |

---

### Step 5: Train PhysMDT (`python scripts/train_phys_mdt.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/phys_mdt/model.pt`, `results/phys_mdt/metrics.json` |
| Time | ~10 min on CPU |
| Notes | Full PhysMDT training with physics loss, structure predictor, soft-mask refinement, token algebra, and test-time finetuning components. |

---

### Step 6: Run ablation evaluation (`python scripts/eval_quick.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/ablations/ablation_results.json` |
| Time | ~3 min |
| Notes | Evaluates ablation variants (no physics loss, no refinement, no structure predictor, etc.) to measure contribution of each component. |

---

### Step 7: Benchmark evaluation (`python scripts/eval_benchmarks.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/benchmark_comparison.json`, `results/nguyen/`, `results/strogatz/`, `results/ai_feynman/` |
| Time | ~5 min |
| Notes | Evaluates on Nguyen, Strogatz, and AI Feynman benchmark suites. Results stored per-benchmark and in aggregate comparison JSON. |

---

### Step 8: Challenge set evaluation (`python scripts/eval_challenge.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/challenge/metrics.json`, `results/challenge/qualitative_examples.md` |
| Time | ~5 min |
| Notes | Evaluates on harder multi-variable and nested-function challenge equations. Includes qualitative examples showing predicted vs. ground truth expressions. |

---

### Step 9: Refinement depth study (`python scripts/refinement_depth_study.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/refinement_depth/results.json`, `figures/refinement_depth.png` |
| Time | ~3 min |
| Notes | Sweeps refinement iterations (1-5) and measures accuracy at each depth to determine optimal refinement count. |

---

### Step 10: Embedding analysis (`python scripts/analyze_embeddings.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/embedding_analysis.json`, `figures/embedding_tsne.png`, `figures/embedding_heatmap.png` |
| Time | ~2 min |
| Notes | Extracts token embeddings from trained PhysMDT, runs t-SNE visualization, and computes cosine similarity heatmaps to analyze learned representations. |

---

### Step 11: Statistical tests (`python scripts/statistical_tests.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `results/statistical_tests.json` |
| Time | ~30s |
| Notes | Runs paired statistical tests (Wilcoxon signed-rank, bootstrap confidence intervals) comparing PhysMDT against baselines to confirm significance of improvements. |

---

### Step 12: Generate figures (`python scripts/generate_figures.py`)

| Field | Value |
|-------|-------|
| Status | Verified |
| Output files | `figures/architecture_diagram.png`, `figures/training_curves.png`, `figures/ablation_barchart.png`, `figures/benchmark_comparison.png`, `figures/challenge_qualitative.png` |
| Time | ~1 min |
| Notes | Generates all publication figures from results JSON files. Uses matplotlib with Agg backend for headless rendering. |

---

## Summary

| Step | Script | Status |
|------|--------|--------|
| 1 | `pip install -r requirements.txt` | Verified |
| 2 | `python -m pytest tests/ -v` | Verified |
| 3 | `python scripts/train_baseline.py` | Verified |
| 4 | `python scripts/eval_sr_baseline.py` | Verified |
| 5 | `python scripts/train_phys_mdt.py` | Verified |
| 6 | `python scripts/eval_quick.py` | Verified |
| 7 | `python scripts/eval_benchmarks.py` | Verified |
| 8 | `python scripts/eval_challenge.py` | Verified |
| 9 | `python scripts/refinement_depth_study.py` | Verified |
| 10 | `python scripts/analyze_embeddings.py` | Verified |
| 11 | `python scripts/statistical_tests.py` | Verified |
| 12 | `python scripts/generate_figures.py` | Verified |

**Total estimated time**: ~35-40 minutes on a modern CPU

## Known Issues

1. **gplearn incompatibility**: gplearn is incompatible with scikit-learn 1.7+. The SR baseline evaluation (`scripts/eval_sr_baseline.py`) uses polynomial regression and gradient-boosted regression as substitutes. This is documented in `results/sr_baseline/metrics.json`.

2. **CPU execution**: The entire pipeline is designed for CPU execution. Training times are kept manageable by using small model sizes and limited epochs. GPU is not required.

3. **Random seed**: Seed 42 is set at the start of every script (via `torch.manual_seed(42)`, `numpy.random.seed(42)`, etc.) to ensure reproducibility. Minor floating-point variations may occur across different hardware or library versions.

4. **Matplotlib backend**: Figure generation scripts set the matplotlib backend to `Agg` for headless (non-interactive) rendering. No display server is required.
