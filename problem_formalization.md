# Problem Formalization: Transformer-Based Physics Equation Derivation

## 1. Problem Statement

**Objective.** Given a set of numerical observations $\{(x_i, y_i)\}_{i=1}^{N}$, where each $x_i \in \mathbb{R}^d$ is a vector of independent physical variables and $y_i \in \mathbb{R}$ is the corresponding measured response, the model must output a symbolic mathematical expression $f$ such that $y \approx f(x)$ for all observations.

This task is formally distinct from standard regression in two critical ways:

1. **Symbolic output.** The model produces a discrete, variable-length token sequence that encodes a closed-form mathematical expression, not a continuous parameter vector. The search space is combinatorial over the grammar of valid expressions.

2. **Interpretability requirement.** The recovered expression must be human-readable and physically meaningful. A neural network approximator that achieves low MSE but offers no interpretable formula does not satisfy the objective. The model must discover the *structure* of the underlying physical law, not merely interpolate the data.

Concretely, "deriving a physics equation" means:

- **Input:** A support set $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{N}$, with $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$.
- **Output:** A prefix-notation token sequence $\hat{T} = (t_1, t_2, \ldots, t_L)$ representing a symbolic expression $\hat{f}$.
- **Success criterion:** The predicted expression $\hat{f}$ is algebraically equivalent to the ground-truth expression $f^*$ after SymPy simplification, **or** achieves $R^2 > 0.999$ on held-out test points with constants fitted via BFGS optimization.

This goes beyond curve fitting -- the model must discover interpretable, closed-form symbolic expressions that capture underlying physical laws. The challenge lies in bridging the gap between continuous numerical input and discrete symbolic output within a single neural architecture.

---

## 2. Input Representation

### 2.1 Numerical Observations

Each input instance consists of a set of $N$ support points, where each point is a tuple of independent variable values and a single dependent variable value:

| Property | Specification |
|---|---|
| Support set size | $N = 200$ points (default) |
| Independent variables | $d \in \{1, 2, \ldots, 9\}$ dimensions |
| Point format | $(x_1, x_2, \ldots, x_d, \; y)$ where $x_j, y \in \mathbb{R}$ |
| Sampling domain | $x_j \sim \text{Uniform}[-5, 5]$ for each dimension $j$ |
| Output range filter | Discard points where $|y| > 10^6$ or $y \in \{\text{NaN}, \pm\infty\}$ |

The full input for a single equation instance is therefore a matrix of shape $(N, d+1)$, where each row is a support point and columns correspond to the $d$ independent variables followed by the response $y$.

**Variable number of dimensions.** The model must handle a variable number of input dimensions ($d = 1$ through $d = 9$) without architectural changes. This is achieved by padding unused variable columns with zeros and providing a dimension indicator to the encoder.

### 2.2 IEEE-754 Half-Precision Multi-Hot Encoding (Primary)

Following NeSymReS (Biggio et al., 2021), each scalar value is converted to an IEEE-754 half-precision (16-bit) binary representation and used directly as a multi-hot feature vector.

**Encoding procedure:**

1. Cast each scalar value $v \in \mathbb{R}$ to `float16` (IEEE-754 half-precision).
2. Extract the 16-bit binary representation: $b = (b_{15}, b_{14}, \ldots, b_0)$.
3. Partition into three fields:

| Field | Bits | Position | Interpretation |
|---|---|---|---|
| Sign | 1 bit | $b_{15}$ | 0 = positive, 1 = negative |
| Exponent | 5 bits | $b_{14} \ldots b_{10}$ | Biased exponent (bias = 15) |
| Mantissa | 10 bits | $b_9 \ldots b_0$ | Fractional significand |

4. The resulting 16-dimensional binary vector $[b_{15}, b_{14}, \ldots, b_0]$ is used as the multi-hot encoding for that scalar.

**Tensor shapes:**

- Per scalar: $\mathbb{R}^{16}$ (16-bit binary vector)
- Per support point: $(d+1) \times 16$ bits (one 16-bit vector per variable and the output)
- Full input tensor: $[N, \; d+1, \; 16]$
- After flattening per point: $[N, \; 16(d+1)]$

This tensor is consumed by the set encoder (Section 6.1), which produces a permutation-invariant latent representation.

**Representable range:**

| Property | Value |
|---|---|
| Smallest positive subnormal | $\approx 5.96 \times 10^{-8}$ |
| Smallest positive normal | $\approx 6.10 \times 10^{-5}$ |
| Largest representable value | $65504$ |
| Precision (significand) | $\approx 3.31$ decimal digits |

### 2.3 Justification for IEEE-754 over Learned Embeddings

The choice of IEEE-754 half-precision multi-hot encoding over alternative representations (raw floating-point input, tokenized decimal strings, or learned embeddings) is motivated by the following considerations:

1. **Numerical precision with fixed size.** Unlike tokenized decimal strings (e.g., "-3.14159" requiring 8+ character tokens of variable length), the IEEE-754 encoding always produces exactly 16 bits per scalar. This eliminates variable-length input handling and provides consistent precision across the representable range ($\sim 6 \times 10^{-8}$ to $65504$).

2. **Structured magnitude information.** The exponent and mantissa fields explicitly encode the order of magnitude and significant digits, respectively. This structure allows the encoder to learn scale-aware features without needing to discover magnitude representation from raw floats.

3. **Proven empirical effectiveness.** Biggio et al. (2021) demonstrate that the IEEE-754 multi-hot encoding improves encoder accuracy by approximately 15% over raw float input on the NeSymReS benchmark. The encoding's binary structure provides useful inductive bias for the set transformer's attention mechanism.

4. **Compatibility with binary neural operations.** The multi-hot binary format is naturally suited to dot-product attention: each bit position can be attended to independently, enabling the model to separately process sign, magnitude, and precision information.

5. **No learned parameters in encoding.** The encoding is deterministic and parameter-free, reducing the model's total parameter count and eliminating a potential source of training instability for the input representation.

**Trade-offs acknowledged:**

- Limited dynamic range compared to `float32` (max $65504$ vs $\sim 3.4 \times 10^{38}$). Mitigated by normalizing observation values to $[-5, 5]$ before encoding.
- Loss of precision for values near zero. Mitigated by the subnormal representation regime of IEEE-754.

---

## 3. Output Representation

### 3.1 Prefix Notation Token Sequence

Equations are represented as prefix-notation (Polish notation) token sequences. In prefix notation, each operator precedes its operands, eliminating the need for parentheses or precedence rules.

**Example conversions:**

| Infix (standard) | Prefix (model output) | Token count |
|---|---|---|
| $x^2 + \sin(x)$ | `add pow x1 2 sin x1` | 6 |
| $\frac{m \cdot v^2}{2}$ | `div mul x1 pow x2 2 2` | 8 |
| $G \cdot \frac{m_1 \cdot m_2}{r^2}$ | `mul C div mul x1 x2 pow x3 2` | 10 |
| $e^{-x^2}$ | `exp neg pow x1 2` | 5 |
| $\sqrt{x_1^2 + x_2^2}$ | `sqrt add pow x1 2 pow x2 2` | 9 |

**Advantages of prefix notation:**

- **Unambiguous without parentheses.** Every valid prefix sequence has exactly one parse tree. There is no operator precedence ambiguity ($2 + 3 \times 4$ is unambiguous as either `add 2 mul 3 4` or `mul add 2 3 4`).
- **Natural fit for tree-structured expressions.** Prefix notation directly mirrors a pre-order traversal of the expression tree, making tree reconstruction trivial.
- **Fixed arity per operator.** Each operator has a known arity (unary or binary), so the parser always knows how many operands to consume. This simplifies both validation and generation.
- **Compatible with autoregressive and masked generation.** The left-to-right structure of prefix notation is naturally suited to autoregressive decoding (generate tokens sequentially) and also compatible with masked diffusion (mask and predict any subset of positions).

### 3.2 Token Vocabulary

The complete vocabulary consists of 43 tokens organized into five categories:

#### 3.2.1 Operators (16 tokens)

**Binary operators (5):**

| Token | Operation | Arity | Semantics |
|---|---|---|---|
| `add` | Addition | 2 | $a + b$ |
| `sub` | Subtraction | 2 | $a - b$ |
| `mul` | Multiplication | 2 | $a \times b$ |
| `div` | Division | 2 | $a \;/\; b$ |
| `pow` | Exponentiation | 2 | $a^b$ |

**Unary operators (11):**

| Token | Operation | Arity | Semantics |
|---|---|---|---|
| `sin` | Sine | 1 | $\sin(a)$ |
| `cos` | Cosine | 1 | $\cos(a)$ |
| `tan` | Tangent | 1 | $\tan(a)$ |
| `exp` | Exponential | 1 | $e^a$ |
| `log` | Natural logarithm | 1 | $\ln(a)$ |
| `sqrt` | Square root | 1 | $\sqrt{a}$ |
| `neg` | Negation | 1 | $-a$ |
| `abs` | Absolute value | 1 | $|a|$ |
| `asin` | Inverse sine | 1 | $\arcsin(a)$ |
| `acos` | Inverse cosine | 1 | $\arccos(a)$ |
| `atan` | Inverse tangent | 1 | $\arctan(a)$ |

#### 3.2.2 Variables (9 tokens)

| Token | Interpretation |
|---|---|
| `x1` through `x9` | Independent variable dimensions $x_1, x_2, \ldots, x_9$ |

Variables are indexed starting from 1. For a $d$-dimensional problem, only `x1` through `x{d}` appear in valid expressions. The model must learn to use the appropriate variable subset.

#### 3.2.3 Constants (12 tokens)

| Token | Value | Usage |
|---|---|---|
| `C` | Fitted via BFGS | Generic constant placeholder |
| `0` | $0$ | Integer literal |
| `1` | $1$ | Integer literal |
| `2` | $2$ | Integer literal |
| `3` | $3$ | Integer literal |
| `4` | $4$ | Integer literal |
| `5` | $5$ | Integer literal |
| `-1` | $-1$ | Negative integer literal |
| `-2` | $-2$ | Negative integer literal |
| `pi` | $\pi \approx 3.14159$ | Mathematical constant |
| `e` | $e \approx 2.71828$ | Euler's number |

The `C` token serves as a placeholder for real-valued constants that are determined after decoding via BFGS optimization. Multiple `C` tokens in a single expression are treated as independent constants, each fitted separately.

#### 3.2.4 Control Tokens (4 tokens)

| Token | Purpose |
|---|---|
| `PAD` | Padding token for batching variable-length sequences |
| `BOS` | Beginning-of-sequence marker |
| `EOS` | End-of-sequence marker |
| `MASK` | Mask token for the masked diffusion model (PhysDiffuser) |

#### 3.2.5 Derivation Tokens (2 tokens)

| Token | Purpose |
|---|---|
| `STEP` | Separates intermediate derivation steps in chain-of-thought mode |
| `END_STEP` | Marks the end of a derivation step |

These tokens enable multi-step derivation, where the model produces a sequence of intermediate expressions leading to the final equation.

**Vocabulary summary:**

| Category | Count | Tokens |
|---|---|---|
| Binary operators | 5 | `add`, `sub`, `mul`, `div`, `pow` |
| Unary operators | 11 | `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `neg`, `abs`, `asin`, `acos`, `atan` |
| Variables | 9 | `x1` -- `x9` |
| Constants | 12 | `C`, `0`--`5`, `-1`, `-2`, `pi`, `e` |
| Control tokens | 4 | `PAD`, `BOS`, `EOS`, `MASK` |
| Derivation tokens | 2 | `STEP`, `END_STEP` |
| **Total** | **43** | |

### 3.3 Maximum Sequence Length

| Mode | Max tokens | Rationale |
|---|---|---|
| Single equation | 64 | Sufficient for most physics equations; covers expressions with up to ~30 operators |
| Derivation chain | 256 | Up to 4 intermediate steps $\times$ 64 tokens each |

The 64-token limit for single equations is derived from analysis of the Feynman benchmark: the most complex equations (e.g., relativistic energy, Compton scattering formulas) require at most 45--55 tokens in prefix notation. The 256-token budget for derivation chains allows for multi-step compositional reasoning while keeping memory requirements bounded.

Sequences shorter than the maximum are right-padded with `PAD` tokens. The `EOS` token marks the true end of the expression.

---

## 4. Evaluation Metrics

### 4.1 Primary Metrics

#### 4.1.1 Exact Symbolic Match Rate

The exact symbolic match rate measures whether the predicted expression is algebraically equivalent to the ground-truth expression, accounting for mathematical identities and simplification rules.

**Procedure:**

1. Parse the predicted token sequence $\hat{T}$ and ground-truth token sequence $T^*$ into SymPy expression objects $\hat{f}$ and $f^*$.
2. Compute the difference: $\Delta = \hat{f} - f^*$.
3. Apply SymPy's `simplify()` function with a 5-second timeout.
4. If $\text{simplify}(\Delta) = 0$, the prediction is an exact match.

**Equivalences handled:**

- **Commutativity:** $a + b = b + a$, $a \cdot b = b \cdot a$
- **Constant folding:** $2 \cdot 3 = 6$, $\sin(0) = 0$
- **Trigonometric identities:** $\sin^2(x) + \cos^2(x) = 1$
- **Algebraic simplification:** $x + x = 2x$, $\frac{x^2}{x} = x$
- **Associativity:** $(a + b) + c = a + (b + c)$

**Scoring:** Binary -- 1 if equivalent, 0 otherwise. Reported as a percentage over the benchmark equation set, with per-tier breakdowns.

**Timeout rationale:** The 5-second timeout prevents SymPy from spending excessive time on pathological expressions. If simplification does not complete within 5 seconds, the match is scored as 0 (non-equivalent). In practice, most equivalence checks resolve in under 100ms.

#### 4.1.2 R-Squared Fit Score ($R^2$)

The $R^2$ metric quantifies the goodness of fit of the predicted expression on held-out test data, after optimizing any free constants.

**Procedure:**

1. Replace all `C` tokens in the predicted expression with independent real-valued parameters $\{c_1, c_2, \ldots, c_k\}$.
2. Sample 1000 held-out test points from the same distribution as the training support points: $x_j \sim \text{Uniform}[-5, 5]^d$.
3. Optimize the constants via L-BFGS to minimize $\sum_{j=1}^{1000} (\hat{f}(x_j; c_1, \ldots, c_k) - y_j)^2$.
4. Compute:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{j=1}^{1000}(y_j - \hat{f}(x_j))^2}{\sum_{j=1}^{1000}(y_j - \bar{y})^2}$$

**Interpretation thresholds:**

| $R^2$ Range | Interpretation |
|---|---|
| $R^2 > 0.999$ | Near-perfect fit (likely correct equation) |
| $0.99 < R^2 \leq 0.999$ | Excellent fit (correct structure, minor constant error) |
| $0.9 < R^2 \leq 0.99$ | Good fit (partially correct structure) |
| $R^2 \leq 0.9$ | Poor fit (wrong structure) |
| $R^2 < 0$ | Prediction worse than predicting the mean |

**Note:** $R^2$ alone is insufficient for evaluation because many structurally incorrect expressions can achieve high $R^2$ through overfitting with free constants. It must be reported alongside the exact symbolic match rate.

#### 4.1.3 Normalized Tree-Edit Distance (NTED)

The normalized tree-edit distance provides a continuous measure of structural similarity between predicted and ground-truth expressions, offering partial credit for near-miss predictions.

**Procedure:**

1. Parse both the predicted expression $\hat{T}$ and ground-truth expression $T^*$ into expression trees $\hat{\mathcal{T}}$ and $\mathcal{T}^*$.
2. Compute the minimum tree-edit distance $\text{TED}(\hat{\mathcal{T}}, \mathcal{T}^*)$ using the Zhang-Shasha algorithm. Allowed operations: node insertion, node deletion, node substitution (each with unit cost).
3. Normalize by the size of the larger tree:

$$\text{NTED} = \frac{\text{TED}(\hat{\mathcal{T}}, \mathcal{T}^*)}{\max(|\hat{\mathcal{T}}|, |\mathcal{T}^*|)}$$

**Properties:**

| Property | Value |
|---|---|
| Range | $[0, 1]$ |
| $\text{NTED} = 0$ | Identical expression trees |
| $\text{NTED} = 1$ | Completely different trees |
| Sensitivity | Captures structural similarity even when symbolic equivalence fails |

**Use case:** NTED is particularly valuable for analyzing failure modes. A prediction with $\text{NTED} = 0.1$ but no exact match likely has the correct overall structure with a minor substitution error (e.g., `sin` vs `cos`), whereas $\text{NTED} = 0.8$ indicates a fundamentally different expression.

### 4.2 Secondary Metrics

#### 4.2.1 Equation Complexity (Node Count)

**Definition:** The total number of nodes (operators, variables, and constants) in the predicted expression tree.

**Reporting:** The complexity ratio $\rho = |\hat{\mathcal{T}}| \;/\; |\mathcal{T}^*|$ is reported alongside each prediction.

| $\rho$ Value | Interpretation |
|---|---|
| $\rho \approx 1.0$ | Predicted and ground-truth have similar complexity |
| $\rho > 1.5$ | Prediction is significantly more complex (overfitting risk) |
| $\rho < 0.7$ | Prediction is oversimplified (underfitting risk) |

This metric operationalizes Occam's razor: among expressions with similar $R^2$ scores, the simpler one is preferred.

#### 4.2.2 Inference Time

**Definition:** Wall-clock time from receiving the input observations to producing the final predicted equation, measured in seconds.

**Components included:**

1. Encoding: support points to latent vector $z$
2. Decoding: autoregressive generation or diffusion refinement
3. Test-time adaptation: LoRA fine-tuning (if enabled)
4. Constant fitting: BFGS optimization of `C` placeholders
5. Candidate selection: scoring and ranking multiple candidates

**Budget:** $\leq 60$ seconds per equation on a single CPU core. This budget is allocated approximately as:

| Component | Time budget |
|---|---|
| Encoding | $\leq 1$s |
| Diffusion refinement (50 steps) | $\leq 15$s |
| Test-time adaptation (32 steps) | $\leq 20$s |
| Constant fitting (BFGS) | $\leq 5$s |
| Candidate selection | $\leq 5$s |
| Overhead | $\leq 14$s |

#### 4.2.3 Derivation Depth

**Definition:** For chain-of-thought mode, the number of intermediate derivation steps generated before the final equation.

| Property | Specification |
|---|---|
| Expected range | 2--5 intermediate steps |
| Measurement | Count of `STEP` tokens in the output sequence |
| Purpose | Quantifies the model's compositional reasoning depth |
| Correlation | Higher derivation depth expected for complex/multi-step tier equations |

---

## 5. Data Generation Protocol

### 5.1 Synthetic Training Data (following NeSymReS)

The training dataset is generated synthetically by sampling random expression trees and evaluating them on random input points. This follows the protocol established by Biggio et al. (2021) in NeSymReS.

**Expression tree generation:**

| Parameter | Range | Default |
|---|---|---|
| Tree depth | 1--8 levels | Uniform sampling |
| Variable count | 1--9 independent variables | Distribution skewed toward 1--3 |
| Operator set | `{add, sub, mul, div, sin, cos, exp, log, sqrt, pow}` | All operators |
| Leaf distribution | 60% variables, 30% integer constants, 10% `C` placeholders | Configurable |

**Support point generation:**

- For each generated equation, sample $N = 200$ support points.
- Each input variable $x_j \sim \text{Uniform}[-5, 5]$ independently.
- Evaluate the equation to produce corresponding $y$ values.

**Filtering criteria (discard equation if any of the following hold):**

| Condition | Rationale |
|---|---|
| Any $y_i = \text{NaN}$ | Undefined operations (e.g., $\log(-1)$) |
| Any $y_i = \pm\infty$ | Overflow or division by zero |
| Any $|y_i| > 10^6$ | Numerical instability |
| Equation is trivially constant | No dependence on input variables |
| Duplicate (up to algebraic equivalence) | Avoid redundant training data |

**Equation skeleton generation:**

- Physical constants in generated equations are replaced with the `C` placeholder token.
- This forces the model to learn structural patterns rather than memorizing specific constant values.
- Constants are recovered at inference time via BFGS optimization.

**Dataset scale:**

| Property | Target |
|---|---|
| Unique equations | $\geq 500{,}000$ |
| Generation throughput | $\geq 100{,}000$ equations/hour on CPU |
| Storage format | Prefix-notation tokens + support point arrays (HDF5 or memory-mapped) |
| Train/validation split | 95% / 5% |

### 5.2 Benchmark Evaluation Data

#### 5.2.1 Feynman Benchmark

The primary evaluation benchmark consists of 120 equations drawn from the Feynman Lectures on Physics, stored in `benchmarks/feynman_equations.json`.

**Difficulty tier distribution:**

| Tier | Count | Variables | Operators | Example |
|---|---|---|---|---|
| Trivial | 20 | 1--2 | $\leq 3$ | $y = x_1 \cdot x_2$ |
| Simple | 25 | 2--3 | $\leq 5$ | $y = \frac{1}{2} m v^2$ |
| Moderate | 30 | 3--5 | $\leq 8$ | $y = G \frac{m_1 m_2}{r^2}$ |
| Complex | 25 | 4--7 | $\leq 12$ | $y = q E + q v B \sin(\theta)$ |
| Multi-step | 20 | 5--9 | $\geq 10$ | Relativistic energy, Compton scattering |

**Per-equation metadata:**

- Symbolic expression in prefix notation
- LaTeX rendering for human readability
- Variable names and physical units
- Number of independent variables ($d$)
- Number of operators (tree internal nodes)
- Difficulty tier with assignment rationale

**Support point generation for evaluation:**

- For each benchmark equation, generate 200 support points using ground-truth parameter values.
- An additional 1000 held-out test points are generated for $R^2$ computation.
- All points sampled from $\text{Uniform}[-5, 5]^d$.

#### 5.2.2 Out-of-Distribution Benchmark

A secondary benchmark of 20 equations **not** present in the Feynman database, stored in `benchmarks/ood_equations.json`. These test generalization to unseen physical domains:

- Simplified Navier-Stokes forms
- 1D Schrodinger equation solutions
- Maxwell's thermodynamic relations
- Statistical mechanics identities
- Quantum harmonic oscillator energy levels

For each OOD equation: 200 support points are generated with ground-truth parameters, following the same protocol as the Feynman benchmark.

---

## 6. Model Architecture Overview

### 6.1 Encoder

The encoder maps a variable-size set of numerical observations to a fixed-dimensional latent vector, using a Set Transformer architecture with Induced Set Attention Blocks (ISAB) for permutation-invariant processing.

**Architecture:**

| Component | Specification |
|---|---|
| Input | $[N, \; d+1, \; 16]$ IEEE-754 multi-hot representations |
| Architecture | Set Transformer with ISAB blocks |
| Number of layers | 4 |
| Attention heads | 8 |
| Embedding dimension | 256 |
| Inducing points (ISAB) | 32 |
| Output | Latent vector $z \in \mathbb{R}^{256}$ |
| Parameter count | $\leq 15$M |

**Processing pipeline:**

1. **Input projection.** Each support point's multi-hot encoding $[d+1, 16]$ is flattened to a vector of dimension $16(d+1)$ and linearly projected to the embedding dimension (256).
2. **ISAB layers.** Four Induced Set Attention Block layers process the set of $N$ projected point embeddings. The ISAB mechanism uses 32 inducing points to reduce the attention complexity from $O(N^2)$ to $O(N \cdot M)$ where $M = 32$.
3. **Pooling.** A Pooling by Multihead Attention (PMA) layer aggregates the $N$ point representations into a single latent vector $z \in \mathbb{R}^{256}$.

**Permutation invariance.** The Set Transformer architecture guarantees that the encoder output $z$ is invariant to the ordering of support points in $\mathcal{S}$. This is essential because the observation set has no natural ordering.

### 6.2 Decoder Variants

#### 6.2.1 Baseline: Autoregressive Transformer Decoder

A standard transformer decoder that generates prefix-notation equation tokens left-to-right, conditioned on the encoder latent vector $z$.

| Component | Specification |
|---|---|
| Architecture | Standard transformer decoder |
| Number of layers | 8 |
| Attention heads | 8 |
| Embedding dimension | 256 |
| Feed-forward dimension | 1024 |
| Cross-attention | To encoder output $z$ |
| Positional encoding | Learned |
| Parameter count | $\leq 30$M |

**Training:** Teacher forcing with cross-entropy loss on the ground-truth token sequence.

**Inference modes:**

- **Greedy decoding:** Select the highest-probability token at each step. Fastest but lowest quality.
- **Beam search:** Maintain top-$k$ candidates ($k \in \{1, \ldots, 10\}$) and select the highest-scoring complete sequence. Default beam width: 5.

#### 6.2.2 PhysDiffuser: Masked Diffusion Decoder

A novel masked discrete diffusion transformer that iteratively refines equation token sequences, inspired by LLaDA (Nie et al., 2024) and the ARChitects ARC 2025 solution.

**Forward process (training):**

1. Given a ground-truth token sequence $T = (t_1, \ldots, t_L)$.
2. Sample a masking ratio $t \sim \text{Uniform}[0, 1]$.
3. Independently mask each token with probability $t$, replacing it with the `MASK` token.
4. The model predicts the original tokens at all masked positions, conditioned on the encoder latent $z$ and the unmasked context.

**Reverse process (inference):**

1. Initialize: all positions set to `MASK`.
2. For each refinement step $s = 1, \ldots, S$ (default $S = 50$):
   a. The model predicts token probabilities at all masked positions.
   b. Tokens with prediction confidence above a threshold (scheduled by step $s$) are unmasked.
   c. Remaining positions retain the `MASK` token or receive soft-mask embeddings for iterative refinement.
3. After $S$ steps, all positions are unmasked.

**Token algebra and soft-masking (from ARChitects):**

- Rather than hard masking, add learnable mask embeddings to all token positions.
- This enables continuous refinement: tokens can be "partially masked" with varying confidence.
- At each step, the model can revise previously committed tokens, providing self-correction capability.

**Most-visited-candidate selection:**

- Run $K$ independent refinement trajectories (default $K = 8$).
- Track which token was predicted at each position across all trajectories.
- Select the most frequently predicted token at each position (majority voting).
- This ensemble technique reduces variance and improves reliability.

### 6.3 Inference Pipeline

The complete inference pipeline for a single equation derivation proceeds as follows:

```
Input: Support set S = {(x_i, y_i)}_{i=1}^{N}

Step 1: ENCODE
    z = SetTransformer(IEEE754_encode(S))
    z ∈ R^256

Step 2: GENERATE CANDIDATES
    Option A (Autoregressive):
        candidates = BeamSearch(Decoder, z, beam_width=5)
    Option B (PhysDiffuser):
        candidates = MaskedDiffusion(PhysDiffuser, z, steps=50, trajectories=8)
        Apply most-visited-candidate selection

Step 3: TEST-TIME ADAPTATION (optional)
    For top candidate c:
        Initialize LoRA adapters (rank=8) on query/value projections
        For 32 adaptation steps:
            Mask random subset of c
            Train to reconstruct masked tokens
            Augment: add noise to observations, retrain
        Re-generate candidates with adapted model

Step 4: FIT CONSTANTS
    For each candidate expression:
        Replace C tokens with free parameters
        Optimize via L-BFGS on support points
        Compute R² on held-out test points

Step 5: SELECT BEST
    Return candidate with highest R² score

Step 6: DERIVATION CHAIN (optional, for complex equations)
    If derivation mode enabled:
        Generate intermediate steps separated by STEP tokens
        Validate each intermediate step
        Return full derivation chain

Output: Symbolic expression f_hat (and optional derivation chain)
```

**End-to-end parameter budget:**

| Component | Parameters |
|---|---|
| Encoder (Set Transformer) | $\leq 15$M |
| Decoder (AR or PhysDiffuser) | $\leq 35$M |
| Physics priors module | $\leq 5$M |
| LoRA adapters (TTA) | $\leq 0.5$M |
| **Total** | $\leq 80$M (base) + $0.5$M (adapters) |

**Inference time budget:** $\leq 60$ seconds per equation on a single CPU core.

---

## References

- Biggio, L., Bendinelli, T., Neitz, A., Lucchi, A., & Parascandolo, G. (2021). Neural Symbolic Regression that Scales. *ICML 2021*.
- Nie, S., et al. (2024). LLaDA: Large Language Diffusion with mAsking. *arXiv preprint*.
- Udrescu, S.-M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, 6(16).
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2019). Set Transformer: A framework for attention-based permutation-invariant input. *ICML 2019*.
- Zhang, K., & Shasha, D. (1989). Simple fast algorithms for the editing distance between trees and related problems. *SIAM Journal on Computing*, 18(6).
