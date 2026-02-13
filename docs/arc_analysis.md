# ARC 2025 ARChitects Solution Analysis

## Overview

The ARChitects achieved 1st place in the ARC-AGI-2 competition using a masked diffusion approach built on LLaDA-8B-Base, a discrete masked diffusion language model. This document analyzes the five key technical innovations and proposes concrete adaptations for physics equation derivation.

---

## 1. Masked Diffusion Architecture

### What They Did

The team used **LLaDA-8B-Base** (Nie et al. 2025), a masked diffusion language model that replaces autoregressive next-token prediction with a mask-and-predict objective. During training, output tokens are randomly masked with varying probabilities, and the model learns to reconstruct masked positions conditioned on visible context. Cross-entropy loss is applied only to masked positions.

Key training specifications:
- **LoRA rank-512** for pretraining on combined datasets (ReARC, ARC-GEN-100K, ARC1/ARC2, ARC-Heavy, ConceptARC)
- 175,000 training steps at effective batch size 8
- 39 hours on 8x H100 GPUs
- Random masking ratio sampled uniformly from [0, 1] per training step

### Why It Outperformed Autoregressive

1. **Bidirectional context**: Unlike autoregressive models that can only attend to preceding tokens, the masked diffusion model attends to the entire sequence, enabling holistic reasoning about the output structure.
2. **Non-sequential generation**: Equations have tree structure, not linear structure. Masked diffusion can fill in any part of the equation in any order, naturally matching the non-sequential nature of mathematical expressions.
3. **Error correction**: Autoregressive models commit to each token irreversibly. Masked diffusion can revise any position during iterative refinement.

### Adaptation for Physics Equations

**Proposal**: Build PhysMDT as a masked diffusion transformer where:
- Input: numerical observation pairs (x_i, y_i) encoded through a learned encoder
- Output: equation tokens in prefix notation, generated via iterative unmasking
- Training: randomly mask 0-100% of output equation tokens, train to predict masked positions
- Architecture: 8-layer transformer with 8 heads, d_model=512 (scaled down from 8B for tractability)
- Use variable masking ratios during training (not fixed 15% like BERT) — this is critical for the diffusion formulation

---

## 2. Recursive Soft-Masking Refinement Loop

### What They Did

Instead of single-pass decoding, the ARChitects iterate:

1. **Initial pass**: Feed fully masked output; model produces logit distribution over all positions
2. **Soft-mask injection**: Add the `<MASK>` embedding to every position's representation, signaling "refine this"
3. **Iterate**: Run the model again on its own soft output; repeat for N steps
4. **Cold restart**: Two rounds of N/2 steps (51+51 = 102 total), resetting to hard-masked input between rounds
5. **Candidate tracking**: Count visit frequency of distinct output candidates across iterations; select top-2 most-visited

The critical insight is operating in **continuous embedding space** — the model processes soft logit distributions, not hard token choices, enabling gradient-like information flow across refinement steps.

### Analogy to Iterative Equation Refinement

This process mirrors how physicists derive equations:
1. Start with an initial guess (all masked → uniform prior over equation tokens)
2. Use context (observed data) to fill in the most confident parts first
3. Iteratively refine: adjust coefficients, correct operator choices, simplify structure
4. "Sleep on it" (cold restart) — reset and re-derive to avoid local optima
5. Converge on the most frequently visited solution (robust to noise)

### Adaptation for Physics Equations

**Proposal**: Implement an `IterativeRefinement` module with:
- Default 50 refinement steps (tunable)
- Cold restart after 25 steps
- Convergence detection: stop early when >95% of positions have confidence >0.9
- Candidate tracking with frequency-based selection
- Ablation flags for: soft-masking (vs hard), cold restart, convergence detection
- Normalize logits between steps for numerical stability

---

## 3. 2D Positional Encoding Innovations

### What They Did

The ARChitects replaced standard 1D RoPE (Rotary Position Embedding) with a **Golden Gate RoPE** variant that encodes positions along multiple directional axes, not just left-to-right. For the ARC grid domain:
- Grids normalized to 32x32 with delimiter symbols
- Position IDs encode both row and column information
- Additional "diagonal" directions beyond horizontal/vertical
- Padding tokens removed post-tokenization while preserving position IDs

### Mapping to Equation Tree Structures

Equations in prefix notation have both:
- **Sequential position**: left-to-right order in the token sequence
- **Tree depth**: how deep in the expression tree a token sits (root operator at depth 0, leaves at maximum depth)

Standard 1D positional encodings miss the tree structure entirely.

### Adaptation for Physics Equations

**Proposal**: Implement **Dual-Axis Positional Encoding**:
- **Axis 1 (sequence position)**: Standard RoPE encoding the left-to-right position
- **Axis 2 (tree depth)**: Additional RoPE encoding the depth of each token in the prefix-notation expression tree
- Computation: For prefix notation, tree depth can be computed in O(n) by tracking operator arity
- Implementation: Concatenate two sets of RoPE dimensions (d_model/2 for each axis) in the query/key projections
- This gives the model explicit awareness of both "where am I in the sequence?" and "how deep am I in the expression?"

---

## 4. Test-Time Finetuning (TTF) Protocol

### What They Did

Two-stage LoRA approach:
1. **Pretraining**: LoRA rank-512, large combined dataset, 175K steps
2. **Test-time finetuning**: For each test task individually:
   - LoRA rank-32 (much smaller)
   - 128 training steps
   - Batch size 1
   - Distinct random augmentations per step
   - Applied on Kaggle L4 GPUs

After TTF, run the refinement loop. Then restore base weights for the next task.

### Adaptation for Physics Equations

**Proposal**: For each test equation:
1. Take the numerical observation pairs as a "task"
2. Apply LoRA rank-32 finetuning for 64-128 steps
3. Data augmentation strategies:
   - **Noise injection**: Add Gaussian noise to observations (σ = 0.01-0.1)
   - **Variable renaming**: Permute variable names (x↔y, t↔s)
   - **Coefficient scaling**: Multiply coefficients by random factors
   - **Sub-sampling**: Use random subsets of observation pairs
4. Run iterative refinement post-TTF
5. Restore base LoRA weights after evaluation

Expected benefit: TTF allows the model to "specialize" its internal representations to the specific equation family at test time, improving accuracy on equations structurally different from training data.

---

## 5. Token Algebra in Continuous Embedding Space

### What They Did

The fundamental insight: "Modern LLMs don't actually operate on discrete symbols except in the first embedding layer. Inside the model, every token is just a point in continuous embedding space."

This enables:
- **Token blending**: `0.5 * embed(token_A) + 0.5 * embed(token_B)` creates a "superposition" state
- **Soft masking**: Combine token embeddings with mask embeddings at varying ratios
- **Normalization**: Re-normalize blended embeddings to maintain numerical stability

The team observed that operating in continuous space allows "creative exploration" — the model can smoothly interpolate between candidate tokens rather than making hard discrete choices.

### Applicability to Symbolic Manipulation

Physics equations exhibit rich algebraic structure that maps naturally to embedding space operations:

1. **Operator interpolation**: Blend sin and cos embeddings to explore trigonometric alternatives
2. **Physical analogies**: If the model learns that `F = ma`, then in embedding space: `embed(F) - embed(m) + embed(v) ≈ embed(p)` (momentum)
3. **Dimensional consistency**: Embeddings of dimensionally compatible quantities should cluster
4. **Coefficient exploration**: Interpolate between numeric constant embeddings to find optimal coefficients

### Adaptation for Physics Equations

**Proposal**: Implement `TokenAlgebra` module supporting:
1. **Linear interpolation**: `lerp(embed_A, embed_B, alpha)` for smooth exploration
2. **Vector arithmetic**: Analogy-based symbolic manipulation
3. **Nearest-neighbor projection**: Project continuous vectors back to discrete vocabulary
4. **Integration with refinement**: During soft-mask refinement, optionally apply token algebra operations to explore the equation space more efficiently
5. **Embedding analysis**: After training, test whether physically meaningful analogies emerge (F-m+v≈p, E-K≈U, etc.)

---

## Summary: Key Transferable Principles

| ARC Innovation | Physics Adaptation | Expected Impact |
|---|---|---|
| Masked diffusion (LLaDA) | PhysMDT with mask-and-predict objective | Bidirectional context for holistic equation reasoning |
| Iterative soft-mask refinement | 50-step refinement with cold restart | Progressive equation derivation, error correction |
| 2D positional encoding | Dual-axis RoPE (sequence + tree depth) | Structure-aware equation generation |
| Test-time finetuning | Per-equation LoRA rank-32 with augmentation | Adaptation to unseen equation families |
| Token algebra | Embedding-space symbolic manipulation | Physics-aware exploration of equation space |
