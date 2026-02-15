# Technical Analysis: Transferring ARChitects ARC 2025 Innovations to Physics Equation Discovery

## Overview

The ARChitects' ARC 2025 solution achieved 21.67% on the ARC public leaderboard using a fine-tuned LLaDA-8B masked diffusion language model with several key innovations. This document maps each innovation to the physics equation discovery domain, analyzing expected benefits and risks.

## 1. Masked Diffusion Training → Masked Symbolic Expression Generation

### ARChitects Approach
- Used LLaDA-8B (masked diffusion LLM) as the base model
- Fully random masking strategy with cross-entropy loss on masked positions
- Variable masking ratio sampled from U[0,1] during training

### Transfer to PhysMDT
- **Input**: Numerical observation pairs (x_i, y_i) encoded by a set-transformer encoder
- **Output**: Symbolic expression tokens in prefix notation, initialized as fully masked
- **Training**: Randomly mask 0–100% of expression tokens; model predicts all masked tokens simultaneously given observations + unmasked context
- **Loss**: Cross-entropy on masked positions only, matching MDLM/LLaDA formulation

### Expected Benefits
- **Parallel token prediction**: Unlike autoregressive models, all tokens predicted simultaneously → captures global expression structure
- **Bidirectional context**: Model sees unmasked tokens on both sides → better for recognizing algebraic patterns
- **Natural training curriculum**: Variable mask ratio means model learns both easy (low mask) and hard (high mask) reconstruction

### Risks
- **Expression coherence**: Mathematical expressions have strict syntactic constraints; random masking may produce invalid expressions during training. Mitigation: constrained decoding + validity checking.
- **Smaller model**: ARChitects used 8B params; we target 50-80M. Need to verify masked diffusion works at small scale.

## 2. Recursive Soft-Masking Refinement → Iterative Equation Refinement

### ARChitects Approach
- Initialize with fully masked output
- Run model to get predictions, then add <mask> embedding to all positions (soft-masking)
- Feed soft predictions back as input; repeat for 102 steps (2 rounds of 51)
- Select candidates via most-visited voting

### Transfer to PhysMDT
- **Step 1**: Start with fully masked expression sequence [MASK, MASK, ..., MASK]
- **Step 2**: Run PhysMDT conditioned on observations → get token probability distributions P(token|position)
- **Step 3**: Compute soft embeddings: e_soft = Σ_t P(t) * E(t) (probability-weighted mixture of token embeddings)
- **Step 4**: Add residual <MASK> embedding: e_input = e_soft + α * E(<MASK>) where α decays over refinement steps
- **Step 5**: Repeat for 64 steps with cosine unmasking schedule (unmask highest-confidence tokens first)
- **Step 6**: Generate K=8 candidates, select via most-visited voting

### Expected Benefits
- **Progressive refinement**: Model can correct early mistakes by reconsidering in context of emerging solution
- **Token algebra**: Soft embeddings enable "between" states (e.g., 0.6*sin + 0.4*cos) that resolve as refinement progresses
- **Better for complex equations**: Tier 3-5 equations benefit most from iterative refinement vs. single-pass decoding
- **Uncertainty quantification**: Confidence scores at each position provide natural uncertainty estimates

### Risks
- **Convergence**: May oscillate between candidates without settling. Mitigation: cosine annealing schedule + cold restart.
- **Computational cost**: 64 forward passes per equation. Mitigation: small model (50-80M params) makes this feasible.

## 3. 2D Golden Gate RoPE → Tree-Positional Encoding for Expression Trees

### ARChitects Approach
- Replaced standard 1D RoPE with 2D variant for ARC's grid structure
- Incorporated multiple directional axes beyond horizontal/vertical
- Called "Golden Gate RoPE" — captures 2D spatial relationships

### Transfer to PhysMDT
- **Observation**: Symbolic expressions in prefix notation have inherent tree structure (operator at root, operands as children)
- **Adaptation**: Map 2D grid coordinates → (tree_depth, sibling_index) for each token
  - `tree_depth`: distance from root in the expression tree (0 for top-level operator)
  - `sibling_index`: position among siblings (0 for left child, 1 for right child, etc.)
- **Implementation**: 2D RoPE where one frequency dimension encodes depth and the other encodes sibling position
- **Example**: In `+ * m a 0` (F = m*a):
  - `+`: depth=0, sibling=0
  - `*`: depth=1, sibling=0
  - `m`: depth=2, sibling=0
  - `a`: depth=2, sibling=1
  - `0`: depth=1, sibling=1

### Expected Benefits
- **Structural awareness**: Model knows that tokens at same tree depth are at same "level" of mathematical complexity
- **Compositional reasoning**: RoPE's relative position encoding means the model learns patterns like "operator followed by two operands at next depth"
- **Better generalization**: New equations with familiar sub-tree structures should be recognized

### Risks
- **Prefix notation ambiguity**: Tree structure must be inferred from prefix sequence (operator arity determines subtree boundaries). Mitigation: precompute tree positions during tokenization.
- **Variable-arity operators**: Functions like `sum` with variable operands complicate tree construction. Mitigation: fix all operators to binary (use nested binary operators).

## 4. Test-Time Fine-Tuning with LoRA → Per-Problem Adaptation

### ARChitects Approach
- 128 steps per task with rank-32 LoRA adapters
- Applied to model attention layers only
- Used augmented examples for data efficiency

### Transfer to PhysMDT
- **Setup**: Inject LoRA adapters (rank 16-32) into PhysMDT attention layers (Q, K, V, O projections)
- **Self-consistency loss**: Given observations {(x_i, y_i)}, decode expression f_pred, evaluate f_pred(x_i), compute MSE against y_i
- **Procedure**:
  1. Freeze base PhysMDT weights
  2. Initialize LoRA adapters
  3. For 64-128 steps: decode expression → evaluate on observations → backprop self-consistency loss → update LoRA
  4. Re-run refinement with adapted model
- **Memory overhead**: rank-32 LoRA on 50M param model ≈ 2.5M additional params (<5% overhead)

### Expected Benefits
- **Per-problem specialization**: Adapts to specific physical system's characteristics (scale, noise profile, variable ranges)
- **Improved constants**: Self-consistency loss directly optimizes for numerical fit, improving constant prediction
- **Works on held-out equations**: Since loss is observation-based (not expression-based), works for equations never seen in training

### Risks
- **Overfitting to noise**: With limited observations, LoRA may fit noise instead of true relationship. Mitigation: early stopping + regularization.
- **Computational cost**: 64-128 forward+backward passes per problem. At <60s target, requires efficient implementation.
- **Local optima**: Self-consistency loss is non-convex. Mitigation: run multiple random initializations.

## 5. Token Algebra in Continuous Embedding Space → Symbolic Token Interpolation

### ARChitects Approach
- Discovered that tokens as continuous embeddings enable meaningful interpolation
- E.g., 0.5 * E(color_0) + 0.5 * E(color_1) produces a "between" state
- This soft representation enabled smooth refinement trajectories

### Transfer to PhysMDT
- **Mathematical operators**: Interpolation between sin and cos (e.g., phase-shifted trigonometric function)
- **Variables**: Soft mixture of x0 and x1 embeddings may represent a linear combination
- **Constants**: Interpolation between integer embeddings could represent non-integer values
- **During refinement**: Soft embeddings allow the model to "hedge" between candidate tokens before committing

### Expected Benefits
- **Smooth optimization landscape**: Refinement trajectory through continuous space avoids discrete jumps
- **Exploration**: Soft embeddings implicitly explore the space of nearby expressions
- **Better for novel equations**: Equations not in training may require token combinations the model hasn't seen discretely

### Risks
- **Meaningless interpolations**: Not all embedding interpolations are semantically meaningful (e.g., 0.5*"+" + 0.5*"*" has no clear interpretation). Mitigation: constrain final output to valid tokens.
- **Normalization**: Soft embeddings may drift in magnitude. Mitigation: layer normalization after mixing.

## Summary Table

| ARChitects Innovation | PhysMDT Adaptation | Expected Impact | Risk Level |
|----------------------|-------------------|-----------------|------------|
| Masked Diffusion Training | Masked expression generation | High (core architecture) | Medium |
| Recursive Soft-Masking | Iterative equation refinement | Very High (key differentiator) | Medium |
| 2D Golden Gate RoPE | Tree-Positional Encoding | Medium (structural bias) | Low |
| Test-Time Fine-Tuning | Per-problem LoRA adaptation | High (zero-shot discovery) | Medium |
| Token Algebra | Symbolic token interpolation | Medium (refinement quality) | Low |
