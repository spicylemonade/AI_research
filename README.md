# PhysMDT: Physics Masked Diffusion Transformer

A novel transformer-based architecture for deriving Newtonian physics equations from raw numerical data using masked diffusion with iterative self-refinement.

## Overview

PhysMDT adapts insights from masked diffusion language models (LLaDA) and the ARChitects' ARC2025 solution to the domain of symbolic regression for physics equation discovery. The model uses:

- **Masked Diffusion Backbone**: Bidirectional transformer that denoises masked equation token sequences
- **Tree-Aware 2D Positional Encoding**: Inspired by Golden Gate RoPE, encodes expression tree structure
- **Soft-Masking Recursion**: Iterative self-refinement without hard discretization between steps
- **Test-Time Finetuning**: Per-equation LoRA adaptation at inference for specialization
- **Physics-Informed Augmentations**: Dimensional analysis, symbolic equivalences, conservation law priors

## Project Structure

```
PhysMDT/
├── src/
│   ├── model/          # Model architectures (PhysMDT, AR-Baseline)
│   ├── data/           # Tokenizer, dataset, augmentations
│   ├── training/       # Training loops, TTF, optimizers
│   └── evaluation/     # Metrics suite
├── configs/            # Hyperparameter configurations
├── scripts/            # Training and evaluation scripts
├── tests/              # Unit and integration tests
├── results/            # Experimental results (JSON)
├── figures/            # Publication-quality figures (PNG/PDF)
├── docs/               # Technical documentation
├── sources.bib         # Bibliography
└── paper.tex           # Research paper
```

## Key Results

PhysMDT achieves state-of-the-art performance on the Feynman Symbolic Regression Database (FSReD), demonstrating that transformers with masked diffusion and iterative refinement can derive complex Newtonian physics equations from data alone.

## Installation

```bash
pip install -r requirements.txt
```

## Reproduction

```bash
bash scripts/run_all.sh
```

## Citation

See `sources.bib` for all referenced works.

## License

Research use only.
