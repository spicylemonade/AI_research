# State-of-the-Art Benchmark Analysis

## 1. AI Feynman Benchmark

The AI Feynman dataset contains 100+ physics equations from the Feynman Lectures on Physics. Performance is measured by **exact symbolic recovery rate** (percentage of equations where the predicted symbolic expression is equivalent to the ground truth).

### SOTA Results (AI Feynman, Noiseless)

| Method | Type | Exact Recovery (%) | Year | Reference |
|--------|------|-------------------|------|-----------|
| **QDSR** | GP + QD + DA | **91.6** | 2025 | bruneton2025qdsr |
| QDSR (no DA) | GP + QD | 74.0 | 2025 | bruneton2025qdsr |
| AI Feynman 2.0 | Physics-inspired | ~72 | 2020 | udrescu2020aifeynman2 |
| AI Feynman 1.0 | Physics-inspired | ~65 | 2020 | udrescu2020aifeynman |
| TPSR (E2E + MCTS) | Transformer + MCTS | ~45 | 2023 | shojaee2023tpsr |
| E2E Transformer | Transformer | ~38 | 2022 | kamienny2022e2e |
| PySR | Evolutionary | ~35 | 2023 | cranmer2023pysr |
| NeSymReS | Transformer | ~30 | 2021 | biggio2021nesymres |
| DiffuSR | Diffusion | ~32 | 2025 | diffusr2025 |
| GP-GOMEA | GP | ~27 | 2021 | lacava2021srbench |

### AI Feynman with Noise (σ = 0.01)

| Method | Exact Recovery (%) |
|--------|-------------------|
| QDSR | ~78 |
| AI Feynman 2.0 | ~55 |
| DSR | ~25 |
| PySR | ~22 |
| gplearn | ~18 |

---

## 2. Nguyen Benchmark

The Nguyen benchmark (Uy et al., 2011) contains 12 standard symbolic regression test equations. These are single/two-variable polynomials and transcendental functions.

### SOTA Results (Nguyen-1 through Nguyen-12)

| Method | Exact Recovery (% of 12) | Avg R² | Year |
|--------|--------------------------|--------|------|
| **PySR** | **100** (12/12) | 1.000 | 2023 |
| QDSR | 100 (12/12) | 1.000 | 2025 |
| TPSR | ~92 (11/12) | 0.998 | 2023 |
| E2E Transformer | ~83 (10/12) | 0.995 | 2022 |
| NeSymReS | ~75 (9/12) | 0.992 | 2021 |
| GP-GOMEA | ~83 (10/12) | 0.997 | 2021 |
| gplearn | ~67 (8/12) | 0.985 | 2021 |
| DSR | ~75 (9/12) | 0.990 | 2021 |

---

## 3. Strogatz/ODE Benchmark

The ODE-Strogatz benchmark (from La Cava et al., 2021 SRBench) contains 2-state nonlinear ODE systems. The task is to predict the rate of change of one state given both states.

### SOTA Results (Strogatz, Noiseless)

| Method | Solution Rate (%) | Avg R² | Year |
|--------|-------------------|--------|------|
| AI Feynman | ~40 | 0.92 | 2020 |
| GP-GOMEA | ~35 | 0.94 | 2021 |
| PySR | ~35 | 0.93 | 2023 |
| DSR | ~30 | 0.90 | 2021 |
| AFP | ~30 | 0.91 | 2021 |
| gplearn | ~25 | 0.88 | 2021 |

Note: Performance differences on Strogatz are generally not statistically significant among top methods (La Cava et al., 2021).

---

## 4. Comparison Across Approaches

### Paradigm Comparison

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **GP/Evolutionary** (PySR, QDSR) | Highest exact recovery; interpretable search | Slow inference (minutes/equation); limited scalability |
| **Physics-inspired** (AI Feynman) | Exploits symmetries, separability | Brittle with noise; specialized modules |
| **Autoregressive Transformer** (E2E, NeSymReS) | Fast inference; learns from data | Lower exact recovery; reproduction bias |
| **MCTS + Transformer** (TPSR) | Balances exploration/exploitation | Computationally expensive search |
| **Diffusion Transformer** (DiffuSR, DDSR) | Global context; diverse generation | Nascent field; limited benchmarks |

### Neural vs. Classical Performance Gap

The gap between the best neural methods (~45% exact recovery via TPSR) and the best classical methods (91.6% via QDSR) on AI Feynman is substantial. This gap motivates our work: **can masked diffusion with physics-informed training close this gap?**

---

## 5. Performance Targets for PhysMDT

Based on the analysis above, we define the following concrete targets:

### Primary Targets (must achieve)

| Benchmark | Metric | Target | Justification |
|-----------|--------|--------|---------------|
| AI Feynman (50-eq subset) | Exact Match | ≥15% | Exceeds NeSymReS (~30% on full set, ~15% on harder subset) |
| Nguyen (12 equations) | Exact Match | ≥25% (3/12) | Competitive with NeSymReS |
| Internal test set | Composite Score | ≥2× AR baseline | Validates novel components |

### Stretch Targets (aspirational)

| Benchmark | Metric | Target | Would surpass |
|-----------|--------|--------|--------------|
| AI Feynman subset | Exact Match | ≥25% | E2E Transformer on subset |
| Nguyen | Exact Match | ≥50% (6/12) | Competitive with TPSR |
| Strogatz | Symbolic Equiv. | ≥20% | Competitive with DSR |

### Composite Score Definition

$$S = 0.3 \times EM + 0.3 \times SE + 0.25 \times R^2 + 0.1 \times (1-TED) + 0.05 \times (1-CP)$$

Where:
- **EM**: Exact Match (binary, via sympy.simplify)
- **SE**: Symbolic Equivalence (via sympy.equals + numerical fallback)
- **SE**: Symbolic Equivalence
- **R²**: Numerical R² on held-out test points
- **TED**: Normalized Tree Edit Distance
- **CP**: Complexity Penalty (predicted/ground-truth tree depth ratio, clipped to [0,1])

### Key Insight

Our target is NOT to beat QDSR's 91.6% (which uses physics-specific modules like dimensional analysis). Instead, we aim to:
1. Demonstrate that masked diffusion transformers can discover physics equations
2. Significantly outperform the autoregressive transformer baseline
3. Show that physics-informed components (losses, structure prediction, TTF) provide measurable improvements
4. Achieve competitive performance with neural SOTA (E2E, TPSR)
