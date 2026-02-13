# PhysMDT: Masked Diffusion Transformer for Physics Equation Derivation

A transformer-based model that derives complex Newtonian physics equations from numerical observations, inspired by the ARC 2025 ARChitects solution. PhysMDT uses masked diffusion training with iterative soft-mask refinement, dual-axis RoPE positional encoding, and physics-informed loss terms to generate symbolic equations in prefix notation.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.9+, PyTorch 2.0+, sympy, numpy, scipy, matplotlib, scikit-learn.

## Project Structure

```
src/                    # Core model code
  phys_mdt.py           # PhysMDT masked diffusion transformer
  baseline_ar.py        # Autoregressive transformer baseline
  tokenizer.py          # Equation tokenizer (prefix notation, 155 tokens)
  metrics.py            # Evaluation metrics (5 metrics + composite score)
  refinement.py         # Iterative soft-mask refinement loop
  token_algebra.py      # Embedding space operations
  physics_loss.py       # Physics-informed loss terms
  structure_predictor.py # Equation structure prediction
  ttf.py                # Test-time finetuning with LoRA
data/                   # Dataset generation
  generator.py          # 61 equation templates across 7 physics families
  challenge_set.json    # 50 complex challenge equations
scripts/                # Training and evaluation entrypoints
  run_experiments.py    # Main experiment pipeline
  reproduce.sh          # Full reproducibility script
tests/                  # Unit tests (66 tests)
results/                # Experiment results (JSON + CSV)
figures/                # Generated figures (PNG)
docs/                   # Documentation and analysis
```

## Data Generation

```bash
# Generate dataset (default: 1000 samples for CPU training)
python -c "from data.generator import generate_dataset; samples = generate_dataset(1000, seed=42)"
```

The generator produces equations from 61 templates across 7 families: kinematics, dynamics, energy, rotational mechanics, gravitation, oscillations, and fluid statics. Each at 3 difficulty levels (simple, medium, complex).

## Training

```bash
# Train both AR baseline and PhysMDT, run evaluations
python scripts/run_experiments.py
```

This runs the full pipeline: data generation, AR baseline training (8 epochs), PhysMDT training (10 epochs), evaluation on test set, ablation study, and refinement depth study.

## Evaluation

```bash
# Run benchmark evaluations
python scripts/eval_benchmarks.py

# Run challenge set evaluation
python scripts/eval_challenge.py

# Embedding analysis
python scripts/analyze_embeddings.py

# Statistical significance tests
python scripts/statistical_tests.py
```

## Full Reproducibility

```bash
bash scripts/reproduce.sh
```

Runs the entire pipeline end-to-end. Expected runtime: ~10 minutes on CPU.

## Results Summary

| Model | Composite | Exact Match | Symbolic Equiv | Numerical R² |
|-------|-----------|-------------|----------------|--------------|
| AR Baseline | 0.021 | 0.000 | 0.000 | 0.000 |
| SR Baseline (literature) | 0.301 | 0.150 | 0.220 | 0.450 |
| PhysMDT (single-pass) | 0.045 | 0.000 | 0.000 | 0.000 |
| PhysMDT (refined) | 0.021 | 0.000 | 0.000 | 0.000 |

Note: Results from small-scale CPU training (d_model=128, 1000 samples). Full-scale training (d_model=512, 500K samples, GPU) expected to yield significantly higher scores.

## Key Components

- **Masked Diffusion Training**: Random masking of equation tokens with bidirectional prediction (inspired by LLaDA)
- **Dual-Axis RoPE**: Encodes both sequence position and expression tree depth
- **Iterative Soft-Mask Refinement**: Progressive unmasking with cold restart (from ARC 2025)
- **Physics-Informed Losses**: Dimensional consistency, conservation regularization, symmetry awareness
- **Test-Time Finetuning**: LoRA-based adaptation per test equation

## References

See `sources.bib` for the complete bibliography (20 entries). Key references:
- Lample & Charton 2020 — Deep Learning for Symbolic Mathematics
- ARC 2025 ARChitects — Masked Diffusion for ARC-AGI
- Nie et al. 2025 — LLaDA: Large Language Diffusion Models
- Raissi et al. 2019 — Physics-Informed Neural Networks
