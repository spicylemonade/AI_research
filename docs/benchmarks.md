# Physics Equation Discovery Benchmarks

## 1. Feynman Symbolic Regression Database (FSReD)

**Source**: Udrescu & Tegmark, 2020 [udrescu2020aifeynman]

The FSReD is the most widely used benchmark for symbolic regression in physics. It consists of equations drawn from the Feynman Lectures on Physics.

| Property | Value |
|----------|-------|
| **Total equations** | 120 (100 original + 20 bonus) |
| **Variable count** | 1–9 variables per equation |
| **Data points per eq.** | 100,000 (default) |
| **Variable ranges** | Physically realistic (e.g., mass > 0, angles in [0, 2π]) |
| **Noise levels** | Clean (0%), with noise variants (1%, 5%, 10%) |
| **Complexity** | Ranges from simple (e.g., F=ma, 2 variables) to complex (e.g., relativistic equations, 9 variables with nested functions) |
| **Operators** | +, -, *, /, ^, sqrt, sin, cos, tan, log, exp, arcsin |
| **Constants** | π, e, integers, real-valued coefficients |

**Difficulty categories** (following SRSD classification [matsubara2023srsd]):
- **Easy** (30 equations): ≤3 variables, ≤3 operators, simple structure
- **Medium** (40 equations): 3-5 variables, moderate nesting
- **Hard** (50 equations): 5+ variables, deep nesting, trigonometric compositions

## 2. SRSD Benchmark

**Source**: Matsubara et al., 2023 [matsubara2023srsd]

The SRSD (Symbolic Regression for Scientific Discovery) benchmark reimplements FSReD with more realistic variable ranges and introduces standardized difficulty categorization.

| Property | Value |
|----------|-------|
| **Total datasets** | 240 (120 standard + 120 with dummy variables) |
| **Difficulty levels** | Easy (30) / Medium (40) / Hard (50) |
| **Variable ranges** | Physically realistic (redesigned from FSReD) |
| **Data points** | 10,000 per dataset |
| **Dummy variable sets** | 120 datasets with irrelevant variables added |
| **Evaluation focus** | Scientific discovery capability, variable selection |

**Key contribution**: Introduced the Normalized Edit Distance (NED) metric and showed that existing methods are not robust against dummy variables.

## 3. SRBench

**Source**: La Cava et al., 2021 [lacava2021srbench]

SRBench is the most comprehensive living benchmark for symbolic regression, covering both real-world and synthetic problems.

| Property | Value |
|----------|-------|
| **Real-world datasets** | 122 diverse regression problems |
| **Synthetic datasets** | 130 ground-truth problems (Feynman + Strogatz) |
| **Methods evaluated** | 14 SR methods + 7 ML baselines |
| **Metrics** | R², symbolic solution rate, complexity |
| **Noise handling** | Varying noise levels tested |
| **Living benchmark** | Accepts rolling contributions of new methods |

**Key findings**: Genetic algorithms with parameter estimation perform best on real-world data; deep learning approaches are competitive on synthetic benchmarks with noise.

## 4. LLM-SRBench

**Source**: Shojaee et al., 2025 [llmsrbench2025]

The newest benchmark specifically designed to evaluate LLM-based scientific equation discovery methods.

| Property | Value |
|----------|-------|
| **Focus** | LLM-based equation discovery |
| **Metrics** | Symbolic accuracy (SA), numeric precision (Acc0.1), NMSE |
| **Methods compared** | LLM-SR, ICSR, LaSR, and others |
| **Domains** | Physics, biology, materials science |
| **Anti-recitation** | Designed to prevent LLM memorization |

---

## SOTA Comparison on FSReD

Performance comparison of state-of-the-art methods on the Feynman Symbolic Regression Database:

| Method | Type | Solution Rate (All) | Solution Rate (Easy) | Solution Rate (Medium) | Solution Rate (Hard) | Year | Citation |
|--------|------|--------------------|--------------------|----------------------|---------------------|------|----------|
| **Eureqa** | GP | ~72% | ~90% | ~75% | ~55% | 2009 | Commercial |
| **AI Feynman 2.0** | Hybrid | ~93% | ~100% | ~95% | ~85% | 2020 | [udrescu2020aifeynman2] |
| **SymbolicGPT** | Transformer | ~35% | ~62% | ~33% | ~18% | 2021 | [valipour2021symbolicgpt] |
| **NeSymReS** | Transformer | ~45% | ~70% | ~45% | ~25% | 2021 | [biggio2021nesymres] |
| **E2E-Transformer** | Transformer | ~52% | ~75% | ~50% | ~35% | 2022 | [kamienny2022e2e] |
| **TPSR** | Transformer+MCTS | ~58% | ~78% | ~57% | ~42% | 2023 | [shojaee2023tpsr] |
| **ODEFormer** | Transformer | ~48%* | ~72%* | ~46%* | ~30%* | 2024 | [dascoli2024odeformer] |
| **PhyE2E** | Transformer+MCTS+GP | ~65% | ~85% | ~65% | ~48% | 2025 | [ying2025phye2e] |

*Note: ODEFormer is primarily designed for ODE systems; FSReD numbers are approximate on adapted static equations. AI Feynman uses physics-inspired heuristics (dimensional analysis, symmetry detection) in addition to neural components. Solution rates are approximate based on published results and may vary with hyperparameter choices.*

---

## Evaluation Metrics

### 1. Solution Rate (Symbolic Accuracy)

**Definition**: Fraction of test equations for which the predicted expression is symbolically equivalent to the ground truth after simplification.

**Computation**: Use SymPy to simplify both predicted and ground-truth expressions, then check algebraic equivalence (accounting for commutativity, constant folding, etc.).

**Rationale**: The gold standard for symbolic regression — a prediction is only correct if it recovers the exact functional form. Binary metric (correct/incorrect per equation). Used in [udrescu2020aifeynman], [valipour2021symbolicgpt], [shojaee2023tpsr].

### 2. R² (Coefficient of Determination)

**Definition**: R² = 1 - SS_res / SS_tot, where SS_res = Σ(y_i - ŷ_i)² and SS_tot = Σ(y_i - ȳ)².

**Rationale**: Measures how well the predicted equation fits the data numerically. R² = 1.0 indicates perfect fit; R² < 0 means worse than predicting the mean. Useful for partial credit when the exact equation is not recovered. Standard metric across all SR benchmarks.

### 3. RMSE (Root Mean Square Error)

**Definition**: RMSE = √(Σ(y_i - ŷ_i)² / N)

**Rationale**: Measures absolute prediction error in the original units. Unlike R², RMSE is scale-dependent, making it useful for comparing within a single equation but less useful across equations with different scales. Commonly reported alongside R² for completeness.

### 4. Normalized Edit Distance (NED)

**Definition**: Edit distance between the predicted and ground-truth expression trees, normalized by the size of the ground-truth tree. NED = edit_distance(T_pred, T_true) / |T_true|.

**Computation**: Convert both expressions to tree form, compute tree edit distance (insertions, deletions, substitutions), normalize by the number of nodes in the ground-truth tree.

**Rationale**: Provides a continuous measure of "how close" a predicted equation is to the truth, even when it's not symbolically equivalent. Proposed by Matsubara et al. [matsubara2023srsd] and shown to correlate better with human judgment than R². NED = 0 means exact match; higher values indicate more structural differences.

### 5. Symbolic Accuracy (Token-Level)

**Definition**: Fraction of correctly predicted tokens in the output equation sequence, computed by aligning predicted and ground-truth token sequences.

**Rationale**: Provides fine-grained feedback for training and analysis. A model might predict 90% of tokens correctly but miss a critical operator, resulting in a wrong equation. Useful as a training signal and for analyzing model behavior at the token level.

### 6. Computational Cost (Inference Time)

**Definition**: Wall-clock time from input data to final equation prediction, measured on standardized hardware (single GPU).

**Rationale**: Critical for practical deployment. AI Feynman may take hours per equation due to its recursive search, while transformer-based methods (SymbolicGPT, E2E) produce answers in seconds. There is a fundamental accuracy-compute tradeoff in symbolic regression. Methods with test-time finetuning (like our PhysMDT) need to report TTF time separately. Used in [lacava2021srbench] to compare methods fairly.

---

## Benchmark Selection for PhysMDT Evaluation

For our experiments, we use the **SRSD-Feynman** variant of FSReD (120 equations with realistic variable ranges and easy/medium/hard splits) as the primary benchmark, supplemented by:

1. **Custom Newtonian physics test set**: 15+ equations spanning mechanics, gravitation, oscillations, and conservation laws
2. **Out-of-distribution test set**: Novel equations not in FSReD for generalization evaluation
3. **Robustness variants**: Noisy and sparse data versions per SRSD protocol

All results are reported using all 6 metrics defined above, with primary emphasis on solution rate (for comparison with published baselines) and NED (for fine-grained analysis).
