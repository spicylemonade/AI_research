# PhysDiffuse: Masked Diffusion Transformers for Autonomous Derivation of Newtonian Physics Equations

A novel masked-diffusion transformer architecture for deriving Newtonian physics equations from numerical observation data. Combines recursive soft-masking refinement (inspired by ARC2025 ARChitects), physics-informed dimensional analysis constraints, and per-equation test-time training with LoRA adapters.

## Project Structure

```
.
├── data/                      # Data pipeline
│   ├── tokenizer.py           # Prefix-notation tokenizer (73-token vocab)
│   ├── data_generator.py      # Physics-informed equation generator
│   ├── feynman_loader.py      # Feynman/SRSD benchmark loader
│   └── augmentation.py        # Data augmentation (4 types)
├── model/                     # Model implementations
│   ├── phys_diffuse.py        # PhysDiffuse masked-diffusion model
│   ├── dim_analysis.py        # Dimensional analysis constraint
│   ├── ttt.py                 # Test-time training with LoRA
│   └── postprocess.py         # SymPy simplification + BFGS
├── baselines/                 # Baseline models
│   ├── autoregressive.py      # Encoder-decoder transformer
│   └── train_baseline.py      # Baseline training script
├── evaluation/                # Metrics
│   └── metrics.py             # 6 evaluation metrics
├── configs/                   # Experiment configurations
│   ├── baseline_training.yaml
│   ├── phys_diffuse_training.yaml
│   ├── ttt.yaml
│   └── ablations.yaml
├── results/                   # Experiment results (JSON)
│   ├── checkpoints/           # Model checkpoints
│   └── *.json                 # Result files
├── figures/                   # Publication-quality figures (PNG + PDF)
├── train_phys_diffuse.py      # PhysDiffuse training script
├── run_ablations.py           # Ablation study (10 configs)
├── run_ttt_eval.py            # TTT evaluation
├── run_derivation.py          # Derivation from scratch
├── run_generalization.py      # OOD generalization
├── run_sota_comparison.py     # SOTA comparison
├── run_error_analysis.py      # Error analysis
├── generate_figures.py        # Figure generation
├── run_all.sh                 # Master experiment script
├── report.md                  # Technical report
├── sources.bib                # BibTeX references (21 entries)
└── requirements.txt           # Dependencies
```

## Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt
```

### Hardware Requirements
- NVIDIA A100-SXM4-40GB (or equivalent)
- CUDA 12.x
- ~4GB GPU memory for training, ~2GB for inference

## Quick Start

```bash
# Run all experiments end-to-end (~1.5 hours)
bash run_all.sh
```

## Reproducing Individual Experiments

### 1. Train Autoregressive Baseline (~9 min)
```bash
python3 baselines/train_baseline.py
```
Output: `results/baseline_results.json`, `results/checkpoints/baseline.pt`

### 2. Train PhysDiffuse (~13 min)
```bash
python3 train_phys_diffuse.py
```
Output: `results/phys_diffuse_results.json`, `results/checkpoints/phys_diffuse.pt`

### 3. Ablation Study (~30 min)
```bash
python3 run_ablations.py
```
Output: `results/ablation_study.json`, `results/ablation_summary.md`

### 4. TTT Evaluation (~5 min)
```bash
python3 run_ttt_eval.py
```
Output: `results/phys_diffuse_ttt_results.json`

### 5. Derivation from Scratch (~2 min)
```bash
python3 run_derivation.py
```
Output: `results/derivation_from_scratch.json`

### 6. Generalization Experiments (~5 min)
```bash
python3 run_generalization.py
```
Output: `results/generalization.json`

### 7. Generate Figures
```bash
python3 generate_figures.py
```
Output: `figures/fig1_training_curves.{png,pdf}` through `figures/fig8_dim_consistency.{png,pdf}`

## Key Results

| Method | Exact Match | NED | Training Data |
|--------|-------------|-----|--------------|
| Autoregressive Baseline | 6.2% | 0.847 | 20K |
| PhysDiffuse | 0.0% | 0.847 | 20K |
| PhysDiffuse + TTT | 0.0% | 0.893 | 20K |

**Key finding:** PhysDiffuse demonstrates strong masked reconstruction (lower training loss than baseline) but generation from fully masked sequences remains challenging with limited training data (20K vs 100M+ used by published methods).

## Model Architecture

- **Encoder:** Set-Transformer with 4 ISAB layers, 32 inducing points, PMA pooling
- **Decoder:** 8-layer bidirectional transformer with masked-diffusion objective
- **Parameters:** 62.2M (encoder ~7M, decoder ~55M)
- **Inference:** Recursive soft-masking with cold restarts, logit normalization, geometric temperature annealing

## Random Seeds

All experiments use `seed=42` for reproducibility. Seeds are documented in `configs/`.

## Citation

```bibtex
@article{physdiffuse2026,
  title={PhysDiffuse: Masked Diffusion Transformers for Autonomous Derivation of Newtonian Physics Equations},
  year={2026}
}
```

## License

Research use only.
