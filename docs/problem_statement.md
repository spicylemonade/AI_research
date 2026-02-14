# Problem Statement: Physics Equation Discovery via Masked Diffusion Transformers

## 1. Formal Problem Definition

### Symbolic Regression from Numerical Observations

Given a set of N numerical observation pairs:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}, \quad \mathbf{x}_i \in \mathbb{R}^d, \quad y_i \in \mathbb{R}$$

find a symbolic expression $f^*$ from a grammar $\mathcal{G}$ of mathematical operations such that:

$$f^* = \arg\min_{f \in \mathcal{G}} \mathcal{L}(f, \mathcal{D}) + \lambda \cdot C(f)$$

where:
- $\mathcal{L}(f, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} (f(\mathbf{x}_i) - y_i)^2$ is the data-fitting loss
- $C(f)$ is a complexity measure (e.g., expression tree depth)
- $\lambda$ is a regularization coefficient

### Masked Diffusion Formulation

We reformulate this as a conditional masked diffusion problem. Let $\mathbf{s} = (s_1, s_2, \ldots, s_L)$ be the tokenized prefix-notation representation of $f$, where each $s_j \in \mathcal{V}$ (vocabulary).

**Forward process**: At diffusion time $t \in [0, 1]$, each token is independently masked with probability $t$:

$$q(\mathbf{s}^t | \mathbf{s}^0) = \prod_{j=1}^{L} \left[ t \cdot \delta_{s_j^t, \texttt{[MASK]}} + (1-t) \cdot \delta_{s_j^t, s_j^0} \right]$$

**Reverse process**: A transformer $p_\theta$ predicts masked tokens conditioned on unmasked tokens and numerical observations:

$$p_\theta(\mathbf{s}^0 | \mathbf{s}^t, \mathcal{D}) = \prod_{j: s_j^t = \texttt{[MASK]}} p_\theta(s_j^0 | \mathbf{s}^t_{\text{unmasked}}, \mathcal{D})$$

**Training objective**: Minimize the masked token prediction loss (ELBO):

$$\mathcal{L}_{\text{MDT}} = \mathbb{E}_{t \sim U(0,1)} \mathbb{E}_{\mathbf{s}^t \sim q} \left[ -\sum_{j: s_j^t = \texttt{[MASK]}} \log p_\theta(s_j^0 | \mathbf{s}^t, \mathcal{D}) \right]$$

---

## 2. Testable Hypotheses

### H1: Masked Diffusion Superiority
**Statement**: A masked diffusion transformer (PhysMDT) will achieve a composite score at least **2× higher** than an autoregressive encoder-decoder baseline of equivalent parameter count on a generated physics equation test set.

**Quantitative criterion**: Composite score ratio $S_{\text{PhysMDT}} / S_{\text{AR}} \geq 2.0$

**Rationale**: Masked diffusion provides bidirectional context at all positions during generation (unlike left-to-right AR models), which should be especially beneficial for structured symbolic expressions where later tokens constrain earlier ones. Supported by LLaDA's advantages over AR models on reversal tasks (Nie et al., 2025).

### H2: Physics-Informed Training Advantage
**Statement**: Adding physics-informed loss components (dimensional consistency, conservation regularization, symmetry enforcement) will improve the composite score by at least **5 percentage points** over the base PhysMDT model without physics losses.

**Quantitative criterion**: $S_{\text{PhysMDT+physics}} - S_{\text{PhysMDT-physics}} \geq 5.0$ on the composite scale [0, 100]

**Rationale**: Physics constraints provide inductive bias that reduces the search space and penalizes physically implausible equations. PINNs (Raissi et al., 2019) demonstrated that embedding physical laws into training improves neural network solutions for PDEs.

### H3: Iterative Refinement Improvement
**Statement**: K-step iterative soft-mask refinement will improve the composite score by at least **5 percentage points** over single-pass decoding, with diminishing returns beyond an optimal K.

**Quantitative criterion**: $S_{K=\text{optimal}} - S_{K=1} \geq 5.0$ and monotonic improvement from K=1 to K=optimal.

**Rationale**: The ARChitects' ARC 2025 solution showed recursive soft-mask refinement was critical for solving tasks requiring global structure understanding. Equation derivation similarly requires global consistency between operands and operators.

---

## 3. Scope Definition

### In-Scope: Newtonian Mechanics Equation Families

| Family | Examples | Variables |
|--------|----------|-----------|
| **1. Kinematics** | $x = x_0 + v_0 t + \frac{1}{2}a t^2$, $v = v_0 + at$ | x, v, a, t |
| **2. Dynamics (Newton's Laws)** | $F = ma$, $F = \mu m g$, $F_{spring} = -kx$ | F, m, a, g, k, μ |
| **3. Energy** | $KE = \frac{1}{2}mv^2$, $PE = mgh$, $W = Fd\cos\theta$ | E, m, v, h, g, d, θ |
| **4. Rotational Mechanics** | $\tau = I\alpha$, $L = I\omega$, $KE_{rot} = \frac{1}{2}I\omega^2$ | τ, I, α, ω, L |
| **5. Gravitation** | $F = \frac{Gm_1 m_2}{r^2}$, $v_{orbit} = \sqrt{\frac{GM}{r}}$ | G, M, m, r |
| **6. Oscillations** | $x = A\sin(\omega t + \phi)$, $T = 2\pi\sqrt{\frac{l}{g}}$ | A, ω, t, φ, l, g |
| **7. Fluid Statics** | $P = \rho g h$, $F_{buoy} = \rho V g$ | P, ρ, g, h, V |

### Difficulty Levels

- **Simple**: Single operator, 1-2 variables (e.g., $F = ma$, $KE = \frac{1}{2}mv^2$)
- **Medium**: 2-3 operators, 2-3 variables (e.g., $x = v_0 t + \frac{1}{2}at^2$)
- **Complex**: 3+ operators, nested functions, 3+ variables (e.g., $T = 2\pi\sqrt{l/g}$, $F = \frac{Gm_1 m_2}{r^2}$)

---

## 4. Explicit Exclusions

The following are **out of scope** for this work:

1. **Quantum mechanics** equations (Schrödinger, Dirac, etc.)
2. **Electromagnetism** (Maxwell's equations, electromagnetic waves)
3. **Thermodynamics** (entropy, heat transfer, gas laws)
4. **Relativistic mechanics** (Lorentz transformations, E=mc²)
5. **Partial differential equations** (wave equation, heat equation)
6. **Stochastic/statistical** mechanics (Boltzmann distribution, etc.)
7. **Multi-step derivations** (we discover final-form equations, not derivation chains)
8. **Vector equations** (we work with scalar projections)
9. **Equations requiring more than 5 independent variables**
10. **Equations with special functions** (Bessel, Legendre, etc.)
