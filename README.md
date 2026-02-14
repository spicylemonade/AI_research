# PhysDiffuser+: Masked Discrete Diffusion for Physics Equation Derivation

PhysDiffuser+ is a neural symbolic regression system that derives closed-form
physics equations from numerical observations.  It combines a masked discrete
diffusion backbone (inspired by LLaDA) with physics-informed priors,
test-time LoRA adaptation, derivation-chain compositionality training, and
BFGS constant fitting.

This repository contains the full reproducibility package (item 024).


## Installation

Requires Python 3.10+ and a recent pip.

```bash
# Clone the repository
git clone <repo-url> && cd <repo-dir>

# Install pinned dependencies
pip install -r requirements.txt
```

All pinned versions are recorded in `requirements.txt`.  The project runs on
CPU -- no GPU is required.


## Training

Train the PhysDiffuser+ model from scratch on synthetic data:

```bash
python scripts/train.py \
    --num_steps 5000 \
    --batch_size 2 \
    --lr 1e-4 \
    --timeout 600
```

Key options (see `python scripts/train.py --help` for the full list):

| Flag                  | Default | Description                          |
|-----------------------|---------|--------------------------------------|
| `--num_steps`         | 5000    | Total training iterations            |
| `--batch_size`        | 2       | Equations per training step          |
| `--lr`                | 1e-4    | Adam learning rate                   |
| `--embed_dim`         | 256     | Model embedding dimension            |
| `--diffuser_layers`   | 4       | Diffusion transformer depth          |
| `--timeout`           | 600     | Wall-clock budget in seconds         |
| `--no_diffusion`      | -       | Ablation: AR-only mode               |
| `--no_physics_priors` | -       | Ablation: disable physics priors     |
| `--no_tta`            | -       | Ablation: disable test-time adapt.   |

The checkpoint is saved to `models/physdiffuser_plus_checkpoint.pt` and a
training-loss figure is saved to `figures/`.


## Evaluation

Evaluate a trained checkpoint on the Feynman or OOD benchmark:

```bash
# All 120 Feynman equations
python scripts/evaluate.py --benchmark feynman \
    --checkpoint models/physdiffuser_plus_checkpoint.pt

# 20 out-of-distribution equations
python scripts/evaluate.py --benchmark ood \
    --checkpoint models/physdiffuser_plus_checkpoint.pt
```

Results are written to `results/eval_feynman.json` (or `eval_ood.json`) and
printed to stdout with per-tier breakdowns.


## Demo

Run the interactive demo to derive an equation from custom data:

```bash
# Single-variable example: y = sin(x)
python scripts/demo.py \
    --x "0.1,0.5,1.0,1.5,2.0,2.5,3.0" \
    --y "0.0998,0.4794,0.8415,0.9975,0.9093,0.5985,0.1411"

# Multi-variable example (semicolons separate variables): y = x1 + x2
python scripts/demo.py \
    --x "1,2,3,4;10,20,30,40" \
    --y "11,22,33,44"

# From a CSV file (last column = y)
python scripts/demo.py --file observations.csv

# Using the baseline checkpoint
python scripts/demo.py --baseline \
    --checkpoint models/baseline_checkpoint.pt \
    --x "1,2,3,4" --y "1,4,9,16"
```


## Expected Results

Performance on the Feynman Symbolic Regression Benchmark (120 equations) with
a CPU-only 10-minute training budget:

| Model                   | Exact Match | R^2 > 0.9 | Mean R^2 |
|-------------------------|-------------|------------|----------|
| AR Baseline             |   ~5%       |   ~8%      |  -0.20   |
| PhysDiffuser+ (full)    |   ~8%       |  ~12%      |  -0.05   |
| Published: NeSymReS     |   72%       |    --      |    --    |
| Published: ODEFormer    |   85%       |    --      |    --    |
| Published: AI Feynman   |  100%       |    --      |    --    |

Note: the low absolute numbers are expected given the minimal training budget
(5000 steps, batch size 2, single CPU).  The purpose of this package is to
demonstrate the full pipeline and verify that the architecture works correctly.
Production-quality results require multi-GPU training for 100k+ steps.


## Project Structure

```
.
|-- README.md                   This file
|-- requirements.txt            Pinned Python dependencies
|-- benchmarks/
|   |-- feynman_equations.json  120 Feynman benchmark equations
|   |-- ood_equations.json      20 out-of-distribution equations
|-- figures/                    Generated plots
|-- models/                     Saved checkpoints (.gitkeep for git)
|   |-- .gitkeep
|-- paper/                      LaTeX source for the paper
|-- results/                    Evaluation outputs (JSON, CSV)
|-- scripts/
|   |-- train.py                PhysDiffuser+ training pipeline
|   |-- evaluate.py             Benchmark evaluation script
|   |-- demo.py                 Interactive equation derivation demo
|   |-- train_baseline.py       Autoregressive baseline training
|   |-- run_experiments.py      Full experiment runner
|   |-- run_noise_ood.py        Noise robustness / OOD experiments
|   |-- profile_cpu.py          CPU performance profiling
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- generator.py        Synthetic equation generator
|   |   |-- derivation_chains.py Multi-step derivation chains
|   |-- eval/
|   |   |-- metrics.py          R^2, symbolic equivalence, TED
|   |-- model/
|   |   |-- encoder.py          Set Transformer encoder (IEEE-754)
|   |   |-- decoder.py          Autoregressive decoder with KV-cache
|   |   |-- phys_diffuser.py    Masked diffusion backbone
|   |   |-- phys_diffuser_plus.py  Full combined model
|   |   |-- physics_priors.py   Dimensional analysis, arity masks
|   |   |-- test_time_adapt.py  LoRA-based test-time adaptation
```


## Citation

If you use this code in your research, please cite:

```
@article{physdiffuser2025,
  title={PhysDiffuser+: Masked Discrete Diffusion for Physics Equation Derivation},
  year={2025},
}
```


## License

See the repository license file for details.
