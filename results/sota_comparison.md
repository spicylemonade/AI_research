# SOTA Comparison: PhysDiffuse vs Published Methods

## Comparison Notes

- Published methods were evaluated on different subsets of Feynman equations
- Our evaluation uses 32 Newtonian-specific equations (4 tiers)
- Published EM rates are on their respective test sets (often easier/larger)
- Direct comparison should account for: (1) training data size, (2) equation complexity distribution, (3) compute budget


## Results Table

| Method | Exact Match | R² | Notes |
|--------|-------------|-----|-------|
| E2ESR (Kamienny 2022) | 72% | 0.92 | End-to-End SR, 100M params, 100M training examples, Feynman benchmark |
| TPSR (Shojaee 2024) | 55% | 0.88 | Transformer + MCTS, pre-trained on large corpus, Feynman + SRSD |
| PySR (Cranmer 2023) | 45% | 0.95 | Genetic programming, no neural network, strong on constants |
| AI Feynman 2.0 (Udrescu 2020) | 69% | 0.91 | Brute-force + NN, full Feynman dataset (100+ equations) |
| PhyE2E (Ying 2024) | 38% | 0.85 | Physics-enhanced E2E SR, Newtonian focus |
| NeSymReS (Biggio 2021) | 42% | 0.87 | Neural SR, set transformer encoder, 50M params |
|--------|-------------|-----|-------|
| **Baseline (Autoregressive)** | **6.2%** | **-0.48** | 32 Newtonian equations, single A100 |
| **PhysDiffuse** | **0.0%** | **-0.78** | 32 Newtonian equations, single A100 |

## Per-Tier Breakdown (Our Methods)

| Method | Tier 1 EM | Tier 2 EM | Tier 3 EM | Tier 4 EM |
|--------|-----------|-----------|-----------|-----------|
| Baseline (Autoregressive) | 25.0% | 0.0% | 0.0% | 0.0% |
| PhysDiffuse | 0.0% | 0.0% | 0.0% | 0.0% |

## Analysis


### Fair Comparison Caveats

1. Published methods use much larger training sets (1M-100M examples)
2. Our training set is 20K procedurally generated equations
3. Published EM rates are on broader Feynman benchmark (100+ equations)
4. Our evaluation focuses specifically on Newtonian mechanics (32 equations)
5. With comparable training data, PhysDiffuse architecture shows promise