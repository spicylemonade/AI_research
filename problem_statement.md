# PhysDiffuse: Problem Statement

## Research Question

**Can a masked-diffusion transformer autonomously derive complex Newtonian physics equations from numerical observation data, and can recursive self-refinement (inspired by ARC2025 LLaDA soft-masking) improve symbolic recovery over autoregressive baselines?**

---

## 1. Input Format

The model receives **numerical observation tables** as input. Each observation table consists of:

- A matrix of shape `(N, D+1)` where `N` is the number of observation points (default: 200) and `D` is the number of independent physical variables (1-9 variables).
- The first `D` columns represent independent variables (e.g., mass, velocity, distance, time, charge), sampled from physically meaningful ranges.
- The last column represents the dependent variable (the quantity to be predicted by the discovered equation).
- Variables are optionally annotated with SI base-unit exponents `[M, L, T]` (mass, length, time) for dimensional analysis.
- Observation values are normalized to the range `[-10, 10]` during preprocessing, with the original scale factors recorded for post-processing constant recovery.

**Example Input (Newton's Second Law: F = m * a):**
```
m (kg)  | a (m/s^2) | F (N)
--------|-----------|------
1.5     | 3.2       | 4.8
0.8     | 9.8       | 7.84
2.3     | 1.1       | 2.53
...     | ...       | ...
```

## 2. Output Format

The model outputs **symbolic equations in prefix (Polish) notation**, represented as sequences of tokens from a fixed vocabulary.

**Token Vocabulary (≤128 tokens):**
- **Operators:** `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `sin`, `cos`, `exp`, `log`, `neg`, `abs`, `inv`
- **Variables:** `x_0`, `x_1`, ..., `x_9` (up to 10 input variables)
- **Constants:** `c_0`, `c_1`, ..., `c_9` (learnable numerical constants, optimized via BFGS post-processing)
- **Special constants:** `pi`, `e`, `one`, `two`, `half`
- **Control tokens:** `<SOS>`, `<EOS>`, `<PAD>`, `<MASK>`

**Example Output (F = m * a):**
```
<SOS> mul x_0 x_1 <EOS>
```

**Example Output (Gravitational Force: F = G * m1 * m2 / r^2):**
```
<SOS> mul c_0 div mul x_0 x_1 pow x_2 two <EOS>
```

## 3. Complexity Tiers of Target Equations

Target equations are stratified into four tiers of increasing complexity:

### Tier 1: Single-Variable Kinematics (8 equations)
- Simple linear and quadratic relationships
- 1-2 variables, 1-3 operators, nesting depth ≤ 2
- Examples: `s = v*t`, `F = m*a`, `p = m*v`, `E_k = 0.5*m*v^2`

### Tier 2: Multi-Variable Dynamics (10 equations)
- Polynomial and rational expressions with 2-4 variables
- 2-4 variables, 3-6 operators, nesting depth ≤ 3
- Examples: `F = G*m1*m2/r^2`, `F = -k*x`, `T = 2*pi*sqrt(L/g)`, projectile range

### Tier 3: Energy/Momentum Conservation Laws (8 equations)
- Composite expressions involving conservation principles
- 3-6 variables, 5-8 operators, nesting depth ≤ 4
- Examples: `E_total = 0.5*m*v^2 + m*g*h`, elastic collision formulas, orbital velocity, escape velocity

### Tier 4: Coupled Multi-Body and Transcendental Equations (6 equations)
- Complex equations involving trigonometric functions, nested compositions, and multi-body interactions
- 4-9 variables, 7+ operators, nesting depth ≤ 6
- Examples: Kepler's third law, damped harmonic oscillator, Lagrangian-derived equations, relativistic corrections

**Total: 32 benchmark equations** spanning the full range of Newtonian mechanics complexity.

## 4. Key Hypothesis

**Primary Hypothesis:** Recursive masked-diffusion refinement outperforms single-pass autoregressive decoding for physics equation discovery from numerical data.

**Rationale:** Autoregressive (left-to-right) decoding commits to early tokens in the symbolic expression and cannot revise them. This is fundamentally mismatched with symbolic equations, where:
1. **Global structure matters:** The overall form of an equation (e.g., ratio vs. product) determines the meaning of every sub-expression.
2. **Operator-operand dependencies are bidirectional:** A `div` token early in the prefix sequence depends on operands that haven't been generated yet.
3. **Physical consistency is holistic:** Dimensional analysis requires checking the entire expression, not just a prefix.

Masked diffusion models address these issues by:
- Generating all tokens simultaneously and refining iteratively (soft-masking refinement)
- Allowing bidirectional attention during generation, capturing global expression structure
- Enabling test-time "re-thinking" via cold restarts and re-masking

**Secondary Hypotheses:**
- H2: Physics-informed dimensional analysis constraints reduce the search space and improve Exact Match rates.
- H3: Per-equation test-time training (TTT) with LoRA adapters provides the largest improvements on complex (Tier 3-4) equations.
- H4: The combination of all components (masked diffusion + refinement + dimensional analysis + TTT) achieves state-of-the-art performance on Newtonian physics equation recovery from the Feynman/SRSD benchmarks.

## 5. Scope Constraints

### Hardware
- **Single NVIDIA A100-SXM4-40GB GPU** for all training and inference
- Target model size: 100M-300M parameters (fits comfortably in 40GB with mixed precision)
- Maximum training time: 48 hours for the main model

### Domain Focus
- **Newtonian mechanics only:** kinematics, dynamics, energy, momentum, gravitation, harmonic motion
- Excludes: electromagnetism, thermodynamics, quantum mechanics, relativity (beyond simple E=mc^2)
- Focus on real-valued functions (no complex numbers, no PDEs)

### Data
- Synthetic training data generated from a physics-informed grammar (500K+ equations)
- Evaluation on 32 curated Newtonian equations from Feynman/SRSD benchmarks
- Up to 9 input variables, expressions up to 30 tokens in prefix notation

### Baselines
- Primary baseline: autoregressive encoder-decoder transformer (E2ESR-style, ~50M parameters)
- Literature comparisons: E2ESR, TPSR, PySR, AI Feynman, PhyE2E (reported numbers)

---

## Architecture Overview (PhysDiffuse)

PhysDiffuse consists of:
1. **Set-Transformer Encoder:** Processes unordered numerical observation points into a fixed-length latent representation (permutation-invariant).
2. **Masked-Diffusion Decoder:** Generates symbolic expressions using the masked diffusion language modeling objective, with iterative soft-masking refinement.
3. **Dimensional Analysis Module:** Auxiliary loss and inference-time filter ensuring physical dimensional consistency.
4. **Test-Time Training (TTT):** Per-equation LoRA adaptation using augmented test data.
5. **Post-Processing:** SymPy simplification + BFGS constant optimization.

The full system is designed to be trained end-to-end on a single A100 GPU and to demonstrate that transformers, when equipped with the right inductive biases and inference strategies, can autonomously derive physics equations from raw numerical data.
