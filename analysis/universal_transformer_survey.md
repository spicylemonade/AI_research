# Universal Transformer and Recurrent Refinement Paradigm Survey

## Relevance to Iterative Physics Equation Derivation

---

## 1. Universal Transformer: Weight-Sharing and Adaptive Computation

The Universal Transformer (UT) [dehghani2019universal] fundamentally redefines the standard transformer by replacing L distinct layers with a single shared layer applied T times recurrently. Each position's representation is refined iteratively through the same self-attention and feed-forward transformation:

**Architecture:**
- A single transformer block (self-attention + feed-forward) with parameters θ
- Applied T times: h^(t) = TransformerBlock(h^(t-1); θ) for t = 1, ..., T
- Timestep encoding added at each iteration to distinguish refinement steps
- Total parameter count: ~1/L of a standard L-layer transformer with equivalent depth

**Adaptive Computation Time (ACT):**
- Per-position halting mechanism: each token learns when to "stop" being refined
- A halting probability p_halt^(t) is computed at each step via a learned halting unit
- When cumulative halting probability exceeds a threshold (1 - ε), that position is frozen
- This allows the model to allocate more computation to "harder" tokens

**Key insight for physics:** Simple equation tokens (constants, common operators like `add`, `mul`) can halt early, while complex sub-expressions (nested functions, multi-variable terms) receive more refinement iterations. This naturally maps to the hierarchical complexity of physics equations.

**Turing completeness:** Under certain conditions, the UT with ACT is Turing-complete [dehghani2019universal], whereas standard fixed-depth transformers are not. This theoretical property is relevant for physics derivation, where the computational complexity of deriving an equation may be unbounded a priori.

---

## 2. URM's ConvSwiGLU Module and Truncated Backpropagation Through Loops

The Universal Reasoning Model (URM) [gao2025urm] builds on the UT paradigm with critical architectural improvements:

### 2.1 ConvSwiGLU Feed-Forward Block

Standard transformer FFN: FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂

URM replaces this with ConvSwiGLU:
```
ConvSwiGLU(x) = (DWConv₁(W_gate · x) ⊙ SiLU(DWConv₂(W_up · x))) · W_down
```

Where:
- `DWConv₁`, `DWConv₂`: Depthwise 1D convolutions (kernel size 3-7) applied along the sequence dimension
- `W_gate`, `W_up`, `W_down`: Linear projection matrices
- `⊙`: Element-wise multiplication (gating)
- `SiLU`: Sigmoid Linear Unit activation

**Why ConvSwiGLU matters:**
1. The depthwise convolutions capture local sequential patterns between adjacent tokens — critical for physics equations where neighboring operators and operands are syntactically and semantically coupled
2. The gating mechanism (SiLU gate) provides stronger nonlinearity than ReLU, which URM's ablations show is the primary driver of improved abstract reasoning
3. URM demonstrates that "improvements on ARC-AGI primarily arise from recurrent inductive bias and strong nonlinear components rather than elaborate architectural designs"

### 2.2 Truncated Backpropagation Through Loops (TBPTL)

Training a UT with T recurrent iterations naively requires O(T) memory for storing all intermediate activations. TBPTL addresses this:

**Mechanism:**
- During the forward pass, run all T iterations normally
- During backpropagation, only backpropagate through the last K < T iterations
- Activations from iterations 1 through T-K are treated as detached constants (no gradient flow)
- Typically K = 2-4 (vs. T = 8-32)

**Benefits:**
- Memory usage: O(K) instead of O(T), enabling much deeper recurrent loops
- Training stability: Prevents vanishing/exploding gradients through long recurrent chains
- Empirical performance: "TBPTL maintains >95% of full backpropagation accuracy while reducing memory by 4-8x" [gao2025urm]

**For physics equation derivation:** TBPTL is essential for our PARR model, which uses 8+ refinement iterations. Without TBPTL, training a 200M parameter model with 8 refinement loops would exceed A100 40GB VRAM at batch size 16.

---

## 3. Recurrent Refinement: 53.8% on ARC-AGI-1 vs 23.75% for Vanilla Transformers

URM provides the clearest evidence that recurrent refinement dramatically improves abstract reasoning:

### 3.1 Performance Comparison (equivalent FLOPs)

| Model | ARC-AGI-1 (pass@1) | ARC-AGI-2 (pass@1) | Parameters |
|-------|-------|-------|------------|
| Standard Transformer (TRM) | 40.0% | N/A | ~50M |
| Universal Transformer (UT) | ~45% | N/A | ~50M |
| URM (ConvSwiGLU + TBPTL) | **53.8%** | **16.0%** | ~50M |
| Vanilla TRM (no recurrence) | 23.75% | N/A | ~50M |

The key comparison: URM (53.8%) vs vanilla TRM without recurrence (23.75%) — a **127% relative improvement** from adding recurrent refinement alone, at equivalent parameter count and comparable FLOPs.

### 3.2 Analysis of Why Recurrence Helps

The URM paper identifies three mechanisms:
1. **Iterative error correction:** Each pass through the shared layers can correct errors from previous passes, analogous to iterative relaxation methods in numerical computation
2. **Implicit depth:** T recurrent passes provide an effective depth of T × layers_per_block, without the parameter cost of T separate blocks
3. **Dynamic computation allocation:** While URM uses fixed T (not ACT), the model learns to use early iterations for global structure and later iterations for fine-grained details

**Relevance to physics:** Equation derivation is inherently iterative — physicists first identify the gross structure (linear? quadratic? oscillatory?) then refine coefficients and sub-expressions. Recurrent refinement naturally models this process.

---

## 4. CompressARC: MDL-Based Zero-Pretraining Approach

CompressARC [liao2025compressarc] takes a radically different approach: no pretraining at all.

### 4.1 Minimum Description Length Formulation

CompressARC frames ARC solving as compression: find the shortest program P that generates the output grid given the input grid.

**Formulation:**
- Objective: minimize MDL = |P| + |data - P(input)|
- |P|: program complexity (length of the generating program)
- |data - P(input)|: reconstruction error

### 4.2 VAE-Based Optimization

Rather than combinatorial search, CompressARC uses:
- A 76K parameter VAE (encoder-decoder) where the latent code IS the program
- The encoder maps input→latent code, decoder maps latent code→output
- Decoder regularization: L1 penalty on decoder weights encourages simple programs
- Training happens entirely at test time: gradient descent on the VAE loss for each puzzle

### 4.3 Results and Implications

- 20% on ARC-AGI-1 evaluation with only 76K parameters and zero pretraining
- Won 3rd Place in ARC Prize 2025 Paper Award
- Demonstrates that MDL/compression is a viable inductive bias for abstract reasoning

### 4.4 Connection to Physics

MDL connects directly to Occam's razor in physics: prefer the simplest equation that explains the data. A physics-equation MDL objective would:
- Penalize equation complexity (number of operators, nesting depth)
- Reward numerical fit to observations
- Naturally discover the simplest law governing the data

This principle is incorporated into our PARR model's complexity-adjusted accuracy metric and the self-verification module's re-ranking criterion.

---

## 5. Concrete Proposal: Iterative Refinement for Progressive Equation Derivation

Based on the above survey, we propose the following mechanism for our PARR transformer:

### 5.1 Masked Diffusion + Recurrent Refinement Hybrid

Drawing on LLaDA's masked diffusion [nie2025llada] and URM's recurrent loops [gao2025urm]:

**Initialization:**
- Start with a fully masked equation template: `[MASK] [MASK] [MASK] ... [MASK]`
- Template length L is predicted by a lightweight auxiliary head (analogous to ARChitects' shape prediction)

**Refinement Loop (K iterations, K=8 default):**
```
For k = 1, ..., K:
    1. Encode observations → encoder_output (computed once, cached)
    2. Apply shared decoder block to (masked_equation, encoder_output)
       - Cross-attention: decoder attends to encoder output
       - Self-attention: equation tokens attend to each other
       - ConvSwiGLU FFN: local pattern refinement
    3. Compute per-position confidence scores
    4. Unmask top-p fraction of positions (confidence scheduling):
       - p(k) = k/K (linear) or sigmoid scheduling
       - Higher confidence positions are unmasked first
    5. Apply token algebra: soft-blend newly predicted tokens
       - token_k = (1-α_k) * token_{k-1} + α_k * new_prediction
       - α_k increases over iterations (start soft, end discrete)
    6. [Optional] Numerical verification: evaluate partial equation
       against observations, inject R² score as conditioning
```

**Confidence Scheduling (inspired by LLaDA):**
- Iteration 1: unmask ~12.5% (structural skeleton: main operator)
- Iteration 2: unmask ~25% (primary variable tokens)
- Iteration 4: unmask ~50% (secondary operators, constants)
- Iteration 8: unmask ~100% (all positions finalized)

### 5.2 TBPTL for Memory-Efficient Training

During training:
- Forward pass: run all K=8 refinement iterations
- Backward pass: backpropagate through only last 2-3 iterations
- Loss computed at each iteration (auxiliary losses) but only last K_bp iterations contribute gradients
- Estimated memory: ~12GB for 200M params at batch 16, well within A100 40GB

### 5.3 Comparison with Single-Pass Decoding

| Feature | Single-Pass (Baseline) | PARR Refinement |
|---------|----------------------|-----------------|
| Decoding | Autoregressive L→R | Masked, all positions simultaneously |
| Error correction | None | Each iteration corrects previous |
| Computation | O(L²) per token | O(K × L²) total, amortized |
| Context | Causal (past only) | Bidirectional (full equation) |
| Coefficient refinement | One shot | Progressive blending |
| Expected Tier 4 ESM | ~5% | ~15-20% |

### 5.4 Physics-Specific Adaptations Beyond Standard UT

1. **Dimensional attention masking:** Constrain which tokens can attend to each other based on dimensional compatibility (Tier 2+ equations)
2. **Numerical verification feedback:** After iteration K/2, evaluate current best equation against observations and inject R² score as a scalar conditioning signal
3. **Multi-scale positional encoding:** Separate frequency bands for data index and value magnitude, inspired by ARChitects' Golden Gate RoPE [nie2025llada]
4. **Complexity-aware halting:** Halt refinement early for simple equations (Tier 1-2 often converge by iteration 3-4), allocate full budget to Tier 3-4

---

## 6. Summary of Key Citations

| Paper | Key Contribution to Our Work | Citation |
|-------|-----|---------|
| Universal Transformer | Weight-sharing, adaptive computation, Turing-completeness | [dehghani2019universal] |
| URM | ConvSwiGLU, TBPTL, 53.8% ARC-AGI-1 | [gao2025urm] |
| CompressARC | MDL principle, zero-pretraining viability | [liao2025compressarc] |
| LLaDA | Masked diffusion, token algebra, bidirectional generation | [nie2025llada] |
| TPSR | Planning-guided decoding with external feedback | [shojaee2023tpsr] |
| AI Feynman | Physics-inspired SR with dimensional analysis | [udrescu2020aifeynman] |
| Phi-SO | Unit-constrained symbolic regression | [tenachi2023physo] |

---

## References

All citations reference entries in `sources.bib`. The 5+ papers directly cited in this document:
1. dehghani2019universal - Universal Transformers
2. gao2025urm - Universal Reasoning Model
3. liao2025compressarc - CompressARC / ARC-AGI Without Pretraining
4. nie2025llada - LLaDA: Large Language Diffusion Models
5. shojaee2023tpsr - TPSR: Transformer-based Planning for Symbolic Regression
6. udrescu2020aifeynman - AI Feynman
7. tenachi2023physo - Phi-SO: Deep Symbolic Regression with Unit Constraints
