# ARC 2025 ARChitects Solution Analysis

## Overview

The ARChitects placed 2nd in the ARC Prize 2025 competition (16.53% score) using a 2D-aware masked-diffusion language model (modified LLaDA-8B) with recursive self-refinement and perspective-based scoring. This document analyzes their five key innovations and proposes adaptations for physics symbolic regression.

**Citation**: architects2025arc

---

## 1. Masked Diffusion Training Objective

### Technique
The ARChitects adapted LLaDA's (Nie et al., 2025) masked diffusion objective: during training, all output tokens are replaced by `[MASK]` tokens and the model learns to predict them simultaneously. Unlike BERT's fixed 15% masking, the masking rate $t$ is sampled uniformly from $[0, 1]$, yielding a proper generative model with an ELBO bound.

The loss is:
$$\mathcal{L} = \mathbb{E}_{t \sim U(0,1)} \left[ -\sum_{j: s_j^t = \texttt{[MASK]}} \log p_\theta(s_j^0 | \mathbf{s}^t) \right]$$

### Relevance to Physics SR
Physics equations have strong internal structure: operators constrain their operands and vice versa. Masked diffusion allows the model to reason bidirectionally, e.g., knowing that a `div` operator appears makes it more likely that `pow ... INT_2` appears in the denominator (as in $F = Gm_1m_2/r^2$).

### Proposed Adaptation
- **Direct adoption**: Use the same masked diffusion objective for PhysMDT
- **Conditioning**: Add cross-attention to numerical observation pairs (unlike ARC which conditions on input grids)
- **Variable masking schedule**: During training, sample $t \sim \text{Beta}(2, 5)$ to bias toward higher masking rates, since full denoising is harder for symbolic expressions

---

## 2. Dual-Axis RoPE

### Technique
The ARChitects modified LLaDA's 1D Rotary Position Embedding (RoPE) to a 2D scheme inspired by Golden Gate RoPE. Standard RoPE encodes a single position index; their version encodes both horizontal (column) and vertical (row) positions, incorporating multiple directional encodings. This helps the model understand 2D spatial relationships in ARC grids.

### Relevance to Physics SR
Mathematical expressions have an inherent tree structure. A token's position in the sequence (left-to-right index) and its depth in the expression tree are both critical for understanding its role. For example, an `add` at depth 0 is the root operation, while `add` at depth 3 is a deeply nested sub-expression.

### Proposed Adaptation
- **Axis 1 — Sequence Position**: Standard RoPE encoding the left-to-right token index (0, 1, 2, ...)
- **Axis 2 — Tree Depth**: RoPE encoding the expression tree depth (0 for root, increasing for deeper nodes)
- **Implementation**: Split the d_model embedding dimensions in half. First half encodes sequence position, second half encodes tree depth. Compute RoPE rotations independently for each axis and concatenate.
- **Depth computation**: Pre-compute tree depth for each token position based on the operator arity prefix-notation rules. During masked diffusion, depth of masked positions can be inferred from the structure predictor or set to the average depth.

---

## 3. Recursive Soft-Mask Refinement

### Technique
Instead of single-pass decoding, the ARChitects perform iterative refinement:
1. Initial forward pass: predict all masked tokens, producing logit distributions
2. Soft-mask injection: replace hard tokens with soft (probabilistic) representations
3. Iterate for K steps: each step refines predictions using the previous step's soft outputs
4. Cold-restart: After K/2 steps, reinitialize from scratch but keep top candidates
5. Select the most frequently generated candidate across iterations

### Relevance to Physics SR
Single-pass generation often produces locally plausible but globally inconsistent equations (e.g., correct operators but wrong variable assignments). Iterative refinement allows the model to:
- Fix early mistakes that propagate through the expression tree
- Enforce global consistency between operands
- Explore multiple candidates and select the most stable one

### Proposed Adaptation
- **K=50 refinement steps** (default, tunable)
- **Cold-restart**: 2 rounds of 25 steps each. After round 1, reset to fully masked but condition on round 1's best candidate via soft logits
- **Convergence detection**: Stop early if >95% of positions have confidence >0.95 for 3 consecutive steps
- **Candidate tracking**: Maintain top-2 most frequently generated complete equations across all iterations; final output is the candidate with highest composite score
- **Ablation flags**: Each component (soft-masking, cold-restart, convergence detection, candidate tracking) can be independently disabled

---

## 4. Test-Time Finetuning (TTF)

### Technique
The ARChitects perform per-task finetuning at inference time: for each test problem, they create a small training set from the given examples and finetune the model for a few steps. Single-task TTF consistently outperforms multi-task approaches. They use LoRA (Hu et al., 2022) for parameter-efficient adaptation.

### Relevance to Physics SR
Each physics equation generates a unique (x, y) data signature. By finetuning on the specific observation pairs for a given equation, the model can specialize its predictions. This is analogous to how physicists would focus on the specific experimental data when trying to derive an equation.

### Proposed Adaptation
- **LoRA rank-32**: Apply low-rank adapters to attention projections (Q, V)
- **64-128 finetuning steps**: On the given (x, y) observation pairs
- **Data augmentation**: (1) Gaussian noise injection (σ=0.01), (2) Variable renaming (permute variable names), (3) Coefficient scaling (multiply all y by a random scale, predict the equation with scaled coefficients)
- **Weight restoration**: Save and restore base model weights after each test equation
- **Objective**: Same masked diffusion loss, but computed only on the test equation's observations

---

## 5. Token Algebra

### Technique
The ARChitects manipulate tokens in continuous embedding space for symbolic reasoning. This includes linear interpolation between embeddings, vector arithmetic (analogies), and nearest-neighbor projection back to the discrete vocabulary.

### Relevance to Physics SR
Physics has rich analogical structure: Force and mass×acceleration are related the same way Energy and mass×velocity² are related. If the model learns these relationships in embedding space, token algebra can:
- Suggest alternative tokens during refinement (e.g., if a position is uncertain between `m` and `F`, the algebra can check if `F ≈ m + a` in embedding space)
- Validate predicted equations by checking embedding-space consistency
- Enable discovery of new relationships by analogy

### Proposed Adaptation
- **Interpolation**: `embed(α * tok_a + (1-α) * tok_b)` for exploring intermediate tokens
- **Analogy**: `embed(F) - embed(mul m a) + embed(E)` should approximate `embed(mul m pow v INT_2)` (since F=ma and KE=½mv²)
- **Nearest-neighbor projection**: After arithmetic, find the closest vocabulary token
- **Integration into refinement**: At each refinement step, for low-confidence positions, use token algebra to suggest alternatives based on high-confidence neighboring tokens
- **Validation**: Post-generation, check that token relationships are consistent with known physics analogies (cosine similarity > 0.6 for known-related pairs)

---

## Summary of Adaptations

| ARC Innovation | ARC Context | PhysMDT Adaptation |
|---------------|-------------|-------------------|
| Masked diffusion | 2D grid completion | Equation token prediction |
| Dual-axis RoPE | Row + column position | Sequence position + tree depth |
| Soft-mask refinement | Grid refinement | Equation refinement with convergence detection |
| Test-time finetuning | Per-task LoRA | Per-equation LoRA on observation pairs |
| Token algebra | Grid token manipulation | Physics embedding analogies |

All five innovations from the ARChitects solution are directly applicable to physics symbolic regression with domain-specific modifications. The key insight is that both tasks require discovering structured outputs (2D grids / symbolic expressions) where local and global consistency are equally important — exactly the setting where masked diffusion excels over autoregressive generation.
