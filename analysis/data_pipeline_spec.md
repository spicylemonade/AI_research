# Synthetic Data Generation Pipeline Specification

## 1. Overview

The data pipeline generates physics equation-observation pairs for training a transformer model to derive Newtonian physics equations from numerical data. Each sample consists of:
- A symbolic equation (target)
- A set of numerical observations consistent with that equation (input)

## 2. Procedural Equation Generation Grammar

### 2.1 Grammar Definition
Equations are generated from parameterized templates organized by complexity tier. Each template is a symbolic expression tree with placeholder coefficients.

### 2.2 Tier 1: Kinematic Equations (12 templates)
```
T1.01: y = c1*x1 + c2*x1^2                    # s = ut + 0.5at²
T1.02: y = c1 + c2*x1                          # v = u + at
T1.03: y = c1^2 + c2*x1                        # v² = u² + 2as
T1.04: y = c1*x1                                # s = vt
T1.05: y = (c1+c2)*x1/2                        # s = 0.5(u+v)t
T1.06: y = c1*x1 + c2*x1^2                    # θ = ω₀t + 0.5αt²
T1.07: y = c1 + c2*x1                          # ω = ω₀ + αt
T1.08: y = c1*x1 - c2*x1^2                    # h = v₀t - 0.5gt²
T1.09: y = c1*sin(c2*x1)                       # x = A*sin(ωt)
T1.10: y = c1*c2*cos(c2*x1)                   # v = Aω*cos(ωt)
T1.11: y = c1*x1^2                             # s = 0.5*a*t² (from rest)
T1.12: y = c1*cos(c2*x1 + c3)                 # x = A*cos(ωt + φ)
```

### 2.3 Tier 2: Force Laws (14 templates)
```
T2.01: y = x1*x2                                # F = ma
T2.02: y = -c1*x1                               # F = -kx
T2.03: y = c1*x1*x2/x3^2                       # F = Gm₁m₂/r²
T2.04: y = c1*x1*c2                             # F = μmg
T2.05: y = x1*x2*sin(x3)                       # τ = rF*sin(θ)
T2.06: y = x1*x2^2/x3                          # F = mv²/r
T2.07: y = x1*x2                                # p = mv
T2.08: y = x1*x2                                # L = Iω
T2.09: y = -c1*x1                               # F = -bv
T2.10: y = -c1*x1^2                             # F = -cv²
T2.11: y = x1*x2*cos(x3)                       # W = Fd*cos(θ)
T2.12: y = x1*x2                                # P = Fv
T2.13: y = c1*x1^2                              # KE = 0.5*mv²
T2.14: y = c1*x1*x2                             # PE = mgh
```

### 2.4 Tier 3: Conservation Laws (12 templates)
```
T3.01: y = c1*x1^2 + c2*x2                     # E = 0.5mv² + mgh
T3.02: y = (x1*x2 + x3*x4)                    # p_total = m₁v₁ + m₂v₂
T3.03: y = c1*x1^2 + c2*x2 - c1*x3^2 - c2*x4 # ΔE = 0.5m(v₁²-v₂²) + mg(h₁-h₂)
T3.04: y = sqrt(c1*x1/x2)                      # v = sqrt(kx²/m) spring-KE
T3.05: y = x1*x2 - x3*x4                       # I₁ω₁ = I₂ω₂ → Δ
T3.06: y = c1*sqrt(x1/c2)                      # T = 2π√(L/g)
T3.07: y = c1*sqrt(x1/x2)                      # T = 2π√(m/k)
T3.08: y = c1*x1^2 - c2*x2                    # KE_loss = 0.5mv₁² - μmgd
T3.09: y = c1*x1*x2^2                          # KE_rot = 0.5Iω²
T3.10: y = (x1*x2 + x3*x4)/(x1+x3)           # v_cm = (m₁v₁+m₂v₂)/(m₁+m₂)
T3.11: y = c1*x1^2 + c2*x2^2                  # E = 0.5kx² + 0.5mv²
T3.12: y = x1*x2 + c1*x3^2                    # E = mgh + 0.5mv²
```

### 2.5 Tier 4: Coupled/Composite Systems (14 templates)
```
T4.01: y = x1*cos(x2)*x3                       # x_proj = v₀cos(θ)t
T4.02: y = x1*sin(x2)*x3 - c1*x3^2           # y_proj = v₀sin(θ)t - 0.5gt²
T4.03: y = (x1*cos(x2)/c1)*(1-exp(-c1*x3))   # x with drag
T4.04: y = c1*sin(x1) - c2*cos(x1)            # a = g*sin(θ) - μg*cos(θ)
T4.05: y = (x1-x2)*c1/(x1+x2)                # a_atwood = (m₁-m₂)g/(m₁+m₂)
T4.06: y = sqrt(2*c1*x1/(1+x2/(x3*x4^2)))    # rolling body on incline
T4.07: y = x1*c1 - c2*x2                       # F_net = mg - bv
T4.08: y = x1*c1/c2                            # v_terminal = mg/b
T4.09: y = c1*sqrt(x1/(x2*c2*x3))             # T_physical_pendulum
T4.10: y = -c1*x1*x2/(2*x3)                   # E_orbital = -Gm₁m₂/(2a)
T4.11: y = exp(-c1*x1)*(c2*cos(c3*x1)+c4*sin(c3*x1)) # damped oscillator
T4.12: y = c1*cos(c2*x1) + c3*cos(c4*x1)     # coupled oscillator modes
T4.13: y = x1*x2*sin(x3)/(x4*x5)             # precession
T4.14: y = c1*x1^2*sin(2*x2)/c2              # projectile range
```

**Total: 52 distinct equation templates**

## 3. Numerical Observation Sampling Strategy

### 3.1 Variable Ranges
For each equation instance, coefficient values and variable ranges are sampled:

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| Coefficients (c1-c4) | Log-uniform | [0.1, 100] with random sign |
| Physical constants (g, G) | Fixed or narrow range | g ∈ [9.7, 10.0], G ∈ [6.6e-11, 6.8e-11] |
| Variables (x1-x6) | Uniform | [-10, 10] base range, scaled per equation |
| Angles | Uniform | [0, 2π] |
| Time variables | Uniform positive | [0.01, 10] |
| Mass/length variables | Uniform positive | [0.1, 100] |

### 3.2 Observation Count
- N = 50 observation points per equation instance (default)
- Variable during training: N ∈ {20, 30, 50, 75, 100} for data augmentation

### 3.3 Noise Injection Levels
Gaussian noise added to output variable y:
- Level 0: σ = 0 (clean data)
- Level 1: σ = 0.01 × std(y)
- Level 2: σ = 0.05 × std(y)
- Level 3: σ = 0.10 × std(y)

Training data distribution: 40% clean, 25% level-1, 20% level-2, 15% level-3

### 3.4 Filtering
- Discard samples where y contains NaN, Inf, or values outside [-1e6, 1e6]
- Discard samples where std(y) < 1e-8 (degenerate/constant equations)
- Ensure at least 2 distinct y values per sample

## 4. Data Augmentation

### 4.1 Variable Permutation
For equations with interchangeable variables (e.g., F = m₁*m₂ symmetric in masses), randomly permute variable assignment to input columns.

### 4.2 Unit Scaling
Multiply all values by a random scale factor s ∈ [0.01, 100] (simulating different unit systems). The equation structure is preserved; only coefficients change.

### 4.3 Coordinate Transformations
- **Sign flip**: Negate selected variables (equation must be adjusted accordingly)
- **Offset**: Add random offset to independent variables (adjusts constant terms in equation)
- **Resampling**: Generate fresh observation points from the same equation with different random seeds

### 4.4 Input Perturbation
Add small perturbations to input variables (not just outputs) to simulate measurement uncertainty:
- σ_input = 0.01 × std(xᵢ) for each input variable

## 5. Train/Val/Test Split Strategy

### 5.1 Template-Level Splitting
- All 52 templates are available in all splits (since the model sees numerical data, not templates)
- For each template, coefficient instances are split: 90% train, 5% val, 5% test
- Random seed 42 used for reproducible splits

### 5.2 Held-Out Configurations (Tier 4)
For Tier 4 coupled systems, 20% of specific parameter configurations are held out entirely for the test set to evaluate true generalization.

### 5.3 No-Leakage Verification
- Hash each (template_id, rounded_coefficients) tuple
- Verify zero overlap between train and test hash sets
- Automated check runs after dataset generation

## 6. Dataset Sizes and Storage

### 6.1 Target Sizes
| Split | Samples | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|-------|---------|--------|--------|--------|--------|
| Train | 1,000,000 | 300K | 300K | 250K | 150K |
| Val | 50,000 | 15K | 15K | 12.5K | 7.5K |
| Test | 50,000 | 15K | 15K | 12.5K | 7.5K |

### 6.2 Storage Format
- **Format**: Memory-mapped NumPy arrays (.npy) for numerical data + JSON for metadata
- **Observation arrays**: float32, shape (N_samples, max_obs_points, max_variables+1)
- **Equation token sequences**: int32, shape (N_samples, max_eq_length)
- **Metadata**: JSON with template_id, tier, coefficients, noise_level per sample

### 6.3 Estimated Storage
- Observations: 1M × 50 × 7 × 4 bytes ≈ 1.4 GB
- Equations: 1M × 128 × 4 bytes ≈ 0.5 GB
- Metadata: ~500 MB
- **Total: ~2.5 GB** (well within 50GB constraint)

### 6.4 Memory-Mapped Loading
Dataset uses np.memmap for training, allowing random access without loading entire dataset into RAM. DataLoader with num_workers=4 prefetches batches.

## 7. Generation Pipeline Architecture

```
┌─────────────────────┐
│ Template Sampler     │ → Selects equation template + tier
├─────────────────────┤
│ Coefficient Sampler  │ → Samples random coefficients
├─────────────────────┤
│ Variable Sampler     │ → Samples input variable values
├─────────────────────┤
│ Equation Evaluator   │ → Computes y = f(x1,...,xk; c1,...,cm)
├─────────────────────┤
│ Noise Injector       │ → Adds Gaussian noise at selected level
├─────────────────────┤
│ Augmentation Engine  │ → Variable permutation, scaling, etc.
├─────────────────────┤
│ Tokenizer           │ → Converts equation to prefix notation tokens
├─────────────────────┤
│ Quality Filter       │ → Removes degenerate samples
├─────────────────────┤
│ Serializer          │ → Saves to memory-mapped arrays
└─────────────────────┘
```

### 7.1 Parallelization
Generation uses multiprocessing (8 workers) with per-tier work distribution. Expected generation time: ~30 minutes for 1.1M samples on single A100.

## 8. Comparison with Prior Work Data Strategies

### 8.1 AI-Newton (Fang et al. 2025)
AI-Newton uses 46 manually designed physics experiments with physical simulators. Our approach differs:
- **Scale**: 1M synthetic pairs vs. 46 experiments
- **Breadth**: 52 equation templates covering full Newtonian mechanics
- **Controllability**: Arbitrary noise levels and coefficient ranges
- **Trade-off**: Our data is more synthetic; AI-Newton's is more physically grounded

### 8.2 TPSR (Shojaee et al. 2023)
TPSR uses randomly generated symbolic expressions without physics constraints. Our approach differs:
- **Domain specificity**: Our templates are grounded in physics, not random expressions
- **Structure**: Our equations respect physical dimensions and constraints
- **Noise model**: We add physics-realistic observation noise, not just random perturbation

### 8.3 SymbolicGPT (Valipour et al. 2021)
SymbolicGPT generates random polynomial expressions. Our approach:
- **Richer function space**: Includes sin, cos, exp, sqrt, not just polynomials
- **Multi-variable**: Up to 6 input variables vs. typically 1-2
- **Tiered complexity**: Structured difficulty progression for curriculum learning
