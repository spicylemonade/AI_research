# Research Problem Statement and Hypotheses

## 1. Problem Definition

**Given** a set of observational data pairs {(x_i, y_i)}_{i=1}^{N} sampled from an unknown physical system, and optionally partial symbolic hints (e.g., known variable names, dimensional units, or structural constraints), **derive** the governing Newtonian physics equation in symbolic form that explains the observed relationship.

Formally, let f* : R^d → R be the unknown ground-truth function implemented by a physics equation. Given observations {(x_i, f*(x_i) + ε_i)}_{i=1}^{N} where ε_i ~ N(0, σ²) represents measurement noise, the goal is to recover a symbolic expression f_hat such that:

1. **Symbolic equivalence**: f_hat is algebraically equivalent to f* (up to simplification)
2. **Numerical accuracy**: R²(f_hat, f*) > 0.99 on held-out test points
3. **Parsimony**: f_hat has minimal complexity (tree depth) among equivalent expressions

This differs from standard symbolic regression in that we specifically target physics equations with known structural properties: dimensional consistency, conservation laws, symmetries, and compositional structure.

## 2. Testable Hypotheses

### H1: Masked Diffusion for Competitive Equation Derivation

**Statement**: A masked-diffusion transformer (PhysMDT) trained on a large corpus of physics equations can derive symbolic equations with accuracy competitive with or exceeding state-of-the-art symbolic regression baselines (PySR, genetic programming) and autoregressive neural baselines.

**Operationalization**: PhysMDT achieves a composite score ≥ 0.65 on our test set (vs. expected ~0.50 for AR baseline and ~0.45 for SR baselines on complex equations). The composite score is defined as:

S = 0.3 × exact_match + 0.3 × symbolic_equivalence + 0.25 × numerical_R² + 0.1 × (1 - tree_edit_distance) + 0.05 × (1 - complexity_penalty)

**Falsification criterion**: If PhysMDT composite score < AR baseline composite score, H1 is rejected.

### H2: Iterative Refinement Improves Derivation Accuracy

**Statement**: Iterative refinement via the soft-masking procedure (inspired by ARC 2025) improves equation derivation accuracy over single-pass decoding. The improvement follows a logarithmic curve with diminishing returns, analogous to the ARC finding that ~100 refinement steps was optimal.

**Operationalization**: PhysMDT with K refinement steps achieves higher composite score than PhysMDT with 1 step (single-pass), for K in {5, 10, 25, 50, 100}. The improvement from K=1 to K=50 is at least 10 composite points.

**Falsification criterion**: If single-pass PhysMDT equals or exceeds iterative PhysMDT on the test set, H2 is rejected.

### H3: Test-Time Finetuning Enables Generalization

**Statement**: Test-time finetuning (TTF) with LoRA on the specific observation pairs of a test equation enables PhysMDT to generalize to equation families not seen during training, including complex multi-variable Newtonian systems.

**Operationalization**: On a held-out challenge set of 50 complex equations: PhysMDT + TTF achieves ≥ 40% symbolic equivalence, while PhysMDT without TTF achieves < 25% and baselines achieve < 10%.

**Falsification criterion**: If TTF does not improve composite score by at least 5 percentage points on the challenge set, H3 is rejected.

## 3. Scope Definition

### Equations In Scope

The research covers Newtonian mechanics equations organized by increasing complexity:

**Level 1 — Simple (single-variable, algebraic)**
- Kinematics: v = v₀ + at, s = v₀t + ½at², v² = v₀² + 2as
- Newton's second law: F = ma
- Hooke's law: F = -kx
- Gravitational force: F = GMm/r²

**Level 2 — Medium (multi-variable, trigonometric/transcendental)**
- Projectile motion: y = x·tan(θ) - gx²/(2v₀²cos²(θ))
- Simple harmonic motion: x(t) = A·cos(ωt + φ)
- Energy conservation: ½mv² + mgh = E
- Orbital velocity: v = √(GM/r)
- Kepler's third law: T² = (4π²/GM)a³

**Level 3 — Complex (coupled, multi-variable systems)**
- Damped driven oscillator: x(t) = (F₀/m)/√((ω₀²-ω²)² + (2γω)²) · cos(ωt - δ)
- Lagrangian mechanics: L = T - V with generalized coordinates
- Hamiltonian: H = Σ(pᵢq̇ᵢ) - L
- Coupled spring-mass systems: m₁ẍ₁ = -k₁x₁ + k₂(x₂-x₁)
- Two-body gravitational problem with perturbations
- Moment of inertia: I = Σmᵢrᵢ²
- Fluid statics: P = P₀ + ρgh, F_buoyancy = ρ_fluid · V · g

### Equations Out of Scope

- Relativistic mechanics (special/general relativity)
- Quantum mechanics (Schrödinger equation, quantum operators)
- Statistical mechanics (partition functions, entropy formulations)
- Electrodynamics (Maxwell's equations)
- Continuum mechanics (stress tensors, Navier-Stokes)

### Data Representation

- **Input**: N observation pairs (x_i, y_i) where x_i ∈ R^d and y_i ∈ R, with d ∈ {1, 2, ..., 6}
- **Output**: Symbolic equation in prefix notation (e.g., `+ * m a * G / m r ^ r 2` for F = ma + GMm/r²)
- **Variables**: Physical variables with semantic meaning (m, g, F, E, v, a, t, r, θ, ω, etc.)
- **Constants**: Both symbolic (π, e, G) and numeric (integers, rational approximations)
