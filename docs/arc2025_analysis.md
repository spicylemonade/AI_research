# Analysis of the ARChitects' ARC2025 Solution

## Overview

The ARChitects team achieved a score of 21.67% on the ARC Prize 2025 public leaderboard using a masked diffusion transformer approach that fundamentally differs from the autoregressive paradigm dominant in language modeling. Their solution is built around LLaDA-8B, a Large Language Diffusion model, enhanced with several key innovations: soft-masking recursion, 2D Golden Gate RoPE positional encoding, and per-task test-time finetuning. This document provides a comprehensive technical analysis of each component and discusses how these innovations can be adapted for physics equation discovery.

## 1. LLaDA-8B Masked Diffusion Backbone

The foundation of the ARChitects' approach is LLaDA-8B-Base (Nie et al., 2025), a discrete token masked diffusion language model with 8 billion parameters, pre-trained from scratch on 2.3 trillion tokens. Unlike autoregressive models that generate tokens left-to-right, LLaDA employs a bidirectional transformer that operates on masked token sequences.

### Forward Process (Masking)
The forward diffusion process randomly replaces tokens with a special `<mask>` token according to a drawn masking probability. At training time, a masking rate is sampled uniformly, and each token is independently masked with that probability. The model is then trained to predict the original tokens at all masked positions using cross-entropy loss computed only on masked positions. This is analogous to the forward noising process in continuous diffusion models, but operates in discrete token space.

### Reverse Process (Demasking)
At inference time, the model starts from a fully masked sequence and iteratively replaces `<mask>` tokens with predicted tokens. The standard approach uses a schedule that unmasks tokens in order of prediction confidence across multiple steps. However, the ARChitects discovered a much more powerful inference strategy: soft-masking recursion (described below).

### Key Design Choices
- The model uses a standard transformer architecture (likely based on LLaMA) but with bidirectional attention (no causal mask)
- Training uses fully random masking with uniform masking rate distribution
- Loss is computed only on masked positions, focusing the model on the denoising task
- Multiple output grids can be masked simultaneously during training without performance degradation

## 2. Soft-Masking Recursion Mechanism

Soft-masking recursion is the ARChitects' most impactful innovation, transforming the masked diffusion model into a continuous iterative solver.

### Core Insight
The key observation is that the `<mask>` embedding acts as a signal meaning "this position needs improvement." By adding the mask embedding to every input position (not just truly masked ones), the model is prompted to refine its predictions at every position, even those it has already filled in.

### Technical Details
The recursion operates in continuous embedding space rather than discrete token space:

1. **Initialization**: Start with logits initialized to the mask token distribution (all probability mass on `<mask>`).
2. **Forward Pass**: Feed these continuous logits through the model, which outputs new logits for each position.
3. **Normalization**: Normalize the output logits (softmax or temperature scaling).
4. **Mask Embedding Addition**: Add the `<mask>` embedding to the normalized logits, signaling that all positions should be reconsidered.
5. **Repeat**: Feed the modified logits back as input. Repeat for N steps (the ARChitects used 102 steps: two rounds of 51 with cold restarts).

### Token Algebra
A critical technical insight is that since tokens exist as continuous vectors in embedding space, they can be blended: e.g., `0.5 * token_A + 0.5 * token_B` produces a meaningful intermediate representation. This is impossible in discrete token space and is what enables the soft recursion to gradually converge on the correct answer without premature commitment to specific tokens.

### Candidate Selection
Rather than simply taking the final output, the ARChitects implement a "most-visited-candidate selection" strategy: they track which discrete token sequences appear most frequently across all refinement steps, then select the most commonly visited candidate as the final answer. This provides robustness against oscillation in the refinement trajectory.

### Performance
The soft-masking recursion approach significantly outperformed standard unmasking schedules and was the single largest contributor to their final score. Without it, the system would have relied on conventional masked diffusion sampling, which is far less effective for ARC tasks.

## 3. 2D Golden Gate RoPE Positional Encoding

The ARChitects replaced LLaDA's standard 1D Rotary Position Embedding (RoPE) with a 2D variant inspired by "Golden Gate RoPE."

### Standard vs. Golden Gate RoPE
- **Standard 2D RoPE**: Encodes only horizontal and vertical grid positions, splitting the embedding dimension equally between x and y coordinates.
- **Golden Gate RoPE**: Incorporates additional directional components beyond just horizontal and vertical. The frequency bases include rotations at various angles (not just 0° and 90°), providing the model with multi-directional positional awareness.

### Implementation for ARC
- Input grids are expanded to a uniform 32×32 size by adding delimiter symbols along right and bottom edges, then padding tokens to fill the target dimensions.
- After tokenization and 2D position ID assignment, padding tokens are removed to save computation while preserving the positional IDs of meaningful tokens.
- This ensures each token retains its correct 2D positional information relative to the grid structure.

### Rationale
ARC tasks frequently require the model to recognize patterns across diagonals, rotations, and other non-axis-aligned directions. The broader directional coverage of Golden Gate RoPE provides the attention mechanism with richer positional signals, which the team reports "significantly boosted performance" compared to standard 1D or simple 2D RoPE variants.

## 4. Test-Time Finetuning Pipeline

The ARChitects employ a two-stage training approach with per-task specialization at inference time.

### Stage 1: General Pre-Training
- Fine-tune LLaDA-8B on aggregated ARC datasets (ReARC, ARC-GEN-100k, ARC1/2, ARC-Heavy, ConceptARC)
- 175,000 training steps with rank-512 LoRA
- Batch size 8 across 8×H100 GPUs
- Total training time: ~39 hours
- This produces a general-purpose ARC solver

### Stage 2: Per-Task Test-Time Finetuning
- For each evaluation task, finetune the model specifically on that task's input-output examples
- 128 training steps per task
- Rank-32 LoRA (reduced from pre-training rank for efficiency)
- Batch size 1
- Each training step uses a distinct random augmentation of the task examples
- Runs on a single GPU (Kaggle L4)

### Key Design Decisions
- Per-task finetuning replaced the 2024 approach of simultaneous multi-task fine-tuning, proving more effective
- The rank reduction from 512 to 32 between stages is intentional: it provides enough capacity for task specialization while preventing overfitting to the few available examples
- Random augmentations at each step are critical for preventing memorization and encouraging generalization

## 5. Relevance to Physics Equation Discovery

Several elements of the ARChitects' approach are directly applicable to symbolic regression for physics:

1. **Masked Diffusion for Sequences**: Instead of 2D grids, we apply the masking/demasking paradigm to 1D equation token sequences. The bidirectional transformer can naturally attend to all parts of an equation simultaneously.

2. **Soft-Masking Recursion for Equation Refinement**: The iterative refinement mechanism is particularly powerful for equation discovery, where the model can progressively correct mathematical structure — first getting the rough functional form, then refining operators and constants.

3. **Tree-Aware Positional Encoding**: We adapt Golden Gate RoPE from 2D grid positions to expression tree positions, encoding both depth and horizontal position in the abstract syntax tree. This gives the model structural awareness of mathematical expressions.

4. **Test-Time Finetuning for Per-Equation Specialization**: Each physics equation has unique numerical patterns. TTF allows the model to adapt its representations to the specific data distribution of each target equation at inference time.

5. **Token Algebra for Continuous Refinement**: The ability to blend token embeddings is especially valuable for symbolic regression, where the model might be uncertain between sin and cos, and the soft blending allows gradual resolution of such ambiguities.

## 6. What Didn't Work for the ARChitects

Important negative results to inform our approach:
- Tiny recursive models: Small models could not capture the complexity needed
- Canon layers and H-Net-style architectures: Did not improve over standard transformers
- Intermediate reasoning tokens: Chain-of-thought style tokens did not help
- Alternative grid representations: The standard tokenization was most effective
- Hybrid positional encoding schemes: Simple Golden Gate RoPE outperformed more complex schemes

These failures suggest that the power comes from the training paradigm (masked diffusion + soft-masking recursion) rather than from exotic architectural modifications, an insight that supports our approach of adapting the proven paradigm to a new domain rather than inventing entirely new architectures.
