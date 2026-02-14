# ARC2025 ARChitects Solution Architecture Review

## Technical Analysis of the LLaDA-8B Masked Diffusion Approach and Transferable Techniques

---

## 1. Overview and Competition Context

The ARChitects team, sponsored by Lambda Labs, placed 2nd in the ARC Prize 2025 competition out of 1,455 teams. Their final submission achieved 21.67% on the public ARC-AGI-2 leaderboard and 16.53% on private evaluation. The team's approach represented a fundamental architectural shift from their 2024 winning solution (autoregressive Mistral-NeMo-Minitron-8B with depth-first search sampling) to a masked diffusion language model (LLaDA-8B) with recursive latent sampling. The 2025 solution is described as "a 2D-aware masked-diffusion LLM with recursive self-refinement and perspective-based scoring."

The team was provided with 1-3 GH200 machines throughout the competition for development, plus two weeks of 16x NVIDIA H100 GPUs for final optimization. The Kaggle submission environment used L4 GPUs for test-time fine-tuning.

---

## 2. LLaDA-8B: The Masked Diffusion Language Model Foundation

### 2.1 What is LLaDA?

LLaDA (Large Language Diffusion with mAsking) is an 8-billion parameter masked diffusion language model developed by GSAI-ML, trained entirely from scratch (not fine-tuned from an autoregressive base). It was pre-trained on 2.3 trillion tokens using 130,000 H800 GPU hours, followed by supervised fine-tuning on 4.5 million prompt-response pairs.

### 2.2 Core Mechanism: Masked Diffusion

Unlike autoregressive models that generate tokens left-to-right, LLaDA operates through a forward masking process and a reverse denoising process:

- **Forward process (training):** Tokens in a sequence are independently masked at a random rate t sampled from U[0,1]. At t=0 the sequence is clean; at t=1 it is fully masked. This variable masking rate (unlike BERT's fixed ~15%) makes the training objective an upper bound on the negative log-likelihood, establishing LLaDA as a proper generative model.

- **Reverse process (inference):** Starting from a fully masked output sequence (t=1), the model iteratively predicts all masked positions simultaneously using a bidirectional transformer (no causal mask). At each step, some predictions are accepted and others are "remasked" based on confidence scores, progressively denoising from t=1 to t=0 over S steps.

### 2.3 Architectural Distinctions from Standard LLMs

- Uses vanilla multi-head attention (not grouped-query attention) since LLaDA is incompatible with KV caching.
- The attention layer has more parameters than a comparably-sized autoregressive model; the FFN dimension is reduced to compensate.
- No causal mask: the transformer sees the entire input bidirectionally.
- The timestep t is not explicitly provided as input; the model infers it from the ratio of masked tokens.

### 2.4 Why LLaDA for ARC?

The bidirectional nature of masked diffusion is critical for ARC tasks. ARC grids require understanding spatial relationships in all directions simultaneously. Autoregressive models suffer from the "reversal curse" (inability to reason backward), while LLaDA captures bidirectional context naturally. LLaDA has been shown to outperform GPT-4o on reversal poem completion tasks, demonstrating its superior bidirectional reasoning.

---

## 3. Token Algebra and Soft-Masking Innovation

### 3.1 The Core Insight

The ARChitects' breakthrough was recognizing that tokens in LLaDA exist as continuous vectors in embedding space, enabling linear algebraic operations. The critical discovery: adding a `<mask>` embedding to every token embedding in the input instructs the model to refine its predictions on those positions.

The operation is:

```
soft_masked_token = token_embedding * 1.0 + mask_embedding * 1.0
```

This produces a "soft-masked" representation -- the token is not fully replaced by the mask, but is blended with it in continuous space. The model learned during training that masked positions "need improvement," and this soft-masking signal exploits that learned behavior to trigger refinement of already-predicted tokens.

### 3.2 Why This Works

During LLaDA's training, the model learns that whenever it encounters a mask token at a position, it must predict what token should be there. By adding mask signal to an already-predicted token, the model receives a signal that says "this position has a prediction, but it may need improvement." The model then outputs a refined prediction that takes into account all surrounding context -- including other predictions from the current iteration.

### 3.3 Distinction from Standard Remasking

Standard masked diffusion inference uses discrete remasking: select low-confidence predictions, replace them entirely with `<mask>`, and re-predict. The ARChitects' token algebra operates entirely in continuous space, avoiding the information loss that comes with discretizing intermediate predictions. This is a fundamental departure from the standard discrete sampling pipeline.

---

## 4. Recursive Latent Sampling and Iterative Refinement

### 4.1 The Algorithm

The recursive latent sampling procedure is the ARChitects' most impactful innovation. The core loop operates as follows:

```
logits = np.zeros(shape)
logits[..., mask_position] = 1  # Initialize as fully masked

for step in range(iterations):
    logits = model(logits)          # Forward pass in continuous space
    logits = normalize(logits)      # Normalize predictions
    logits[..., mask_position] = 1  # Re-inject mask signal
```

### 4.2 Key Properties

1. **Continuous-space operation:** The entire refinement loop operates on continuous logit vectors, never discretizing to hard token selections between iterations. This preserves uncertainty information and allows smooth gradient-like refinement.

2. **Mask re-injection:** After each forward pass, the mask signal is added back to all output positions, creating the soft-masking effect that signals "keep refining." This creates a feedback loop where each iteration improves upon the previous one.

3. **No remasking strategy needed:** Unlike standard masked diffusion inference that must decide which tokens to remask and which to keep, the recursive approach refines all positions simultaneously in every iteration.

4. **Cold restart:** The team found that running two rounds of 51 refinement steps with a cold restart (resetting between rounds) worked better than a single run of 102 steps. This suggests the refinement can get trapped in local optima and benefits from restarts.

### 4.3 Performance Configuration

The final submission used 102 recursive refinement iterations, executed as two 51-step rounds with cold restart. With known output shape, this achieved approximately 30.5% +/- 1% accuracy on evaluation data.

---

## 5. 2D Positional Encoding: Golden Gate RoPE

### 5.1 The Problem with Standard Positional Encoding

LLaDA was originally trained with standard 1D RoPE (Rotary Position Embedding), designed for sequential text. ARC tasks involve 2D grids where spatial relationships in all directions (horizontal, vertical, diagonal) are critical for reasoning.

### 5.2 Classical 2D Axial RoPE Limitation

The naive approach of 2D axial RoPE partitions embedding dimensions into two halves: one encoding x-position (column), one encoding y-position (row). This creates a fundamental limitation: the first half of a query contributes equally to attention regardless of the key's y-position, and vice versa. The model cannot attend to specific (row, column) positions -- it can only attend to entire rows or columns. Diagonal and other non-axis-aligned spatial relationships are poorly captured.

### 5.3 Golden Gate RoPE (GGRoPE)

The ARChitects adopted Golden Gate RoPE, which addresses this limitation by encoding positions along arbitrary 2D directions rather than just horizontal and vertical axes. The key idea:

- Instead of rotating dimension pairs based on only x-position or only y-position, rotations are based on positions measured along arbitrary directions in 2D space.
- Direction vectors are sampled quasi-randomly using the generalized golden ratio, mapped to Gaussian samples via inverse CDF, and normalized to unit length.
- The golden ratio quasi-random sampling ensures uniform coverage of all angular directions with minimal clustering.

This gives the model the ability to "look" across the grid from multiple angles -- not just left/right and up/down, but along diagonals and other orientations. This capability was identified as significantly beneficial for ARC tasks.

### 5.4 Grid Formatting for Positional Encoding

All input/output grids were expanded to a uniform 32x32 size:
- Delimiter symbols were added along grid edges.
- Remaining space was filled with special padding tokens.
- During tokenization, padding tokens were removed after position-ID assignment, preserving 2D position information while reducing computation.

---

## 6. Test-Time Fine-Tuning Strategy

### 6.1 Two-Stage Training Pipeline

**Stage 1: Initial Pretraining (offline)**
- Rank-512 LoRA fine-tuning of LLaDA-8B-Base
- Training data: ReARC, ARC-GEN-100K, official ARC1/ARC2 sets, ARC-Heavy, ConceptARC
- 175,000 training steps, effective batch size of 8
- Duration: 39 hours on 8x H100 cluster
- Masking strategy: "fully random masking" -- random masking probability drawn per output grid, then each token independently replaced with `<mask>`. Loss computed only on masked positions.

**Stage 2: Test-Time Fine-Tuning (per-task, online)**
- Per-task specialization on Kaggle L4 GPUs during inference
- Rank-32 LoRA (much smaller than pretraining LoRA)
- 128 training steps per task
- Batch size of 1
- Random augmentations applied during TTF (rotations, reflections, transpositions, color permutations, example reordering)

### 6.2 Key Design Decision: Per-Task vs. Multi-Task TTF

A critical improvement over their 2024 approach was switching from multi-task fine-tuning (training on multiple test tasks simultaneously) to per-task fine-tuning (training a separate LoRA adapter for each individual test task). This allowed the model to fully specialize its representations for each task's specific transformation pattern.

### 6.3 Candidate Selection: Most-Visited-Candidate

During the soft-masking refinement loop, the team tracked which solution candidates appeared most frequently. The two most-visited candidates were selected as the final two guesses (ARC allows two guesses per task). This stateful tracking approach proved more reliable than stateless confidence-based scoring methods.

---

## 7. Shape Prediction Model

A separate model was needed to predict output grid dimensions, since the demasking model assumes known output shape.

- The shape prediction model was fine-tuned from the same LLaDA base using a modified data format: pixel-color tokens were replaced with padding tokens, leaving only delimiter tokens that define grid boundaries.
- Additional loss was applied specifically to delimiter token placement.
- Shape prediction accuracy: approximately 85% +/- 2% on evaluation data.
- The combined system (shape prediction + demasking) achieved the final 21.67% public leaderboard score.

---

## 8. Performance Numbers Summary

| Metric | Value |
|--------|-------|
| Public leaderboard score | 21.67% |
| Private evaluation score | 16.53% |
| Accuracy with known shape | ~30.5% +/- 1% |
| Shape prediction accuracy | ~85% +/- 2% |
| Expected combined score | ~26% (overfitting gap) |
| Refinement iterations | 102 (2 rounds of 51) |
| Pretraining duration | 39 hours on 8x H100 |
| TTF steps per task | 128 |
| Pretraining LoRA rank | 512 |
| TTF LoRA rank | 32 |
| Grid size (uniform) | 32x32 |

---

## 9. Evolution from 2024 Autoregressive Approach

The team's 2024 solution used nvidia/Mistral-NeMo-Minitron-8B-Base with depth-first search (DFS) sampling. In the first half of 2025, they optimized this approach with:

- **Speculative decoding:** Heuristically guessing 16-32 tokens per step instead of single-token generation (4.7x speedup).
- **Prefix caching:** Reusing KV cache across augmentation passes (5.8x speedup on scoring).
- **DFS threshold reduction:** Lowering probability cutoff from 17% to 7% to generate more candidates.
- **Per-task fine-tuning:** Switching from multi-task to single-task TTF.

This optimized autoregressive approach reached 16.94% on the public leaderboard by August 11, but the team recognized it was insufficient for the 2025 competition. They pivoted to the masked diffusion approach after attending ICML 2025.

---

## 10. Failed Approaches

The team documented several approaches that did not yield improvements:

1. **Synthetic data generation:** Used Qwen2.5-Coder-32B with GRPO training to generate Atari-like game screens. Only 1-2% of generated tasks were meaningful. A semi-manually curated 150-task synthetic dataset produced insufficient gains. Early exploration used 8x A100 GPUs for approximately three weeks.

2. **Architectural experiments:** Canon layers, H-Net-style architectures, intermediate reasoning tokens, alternative grid representations, and tiny language models all produced minimal improvements.

3. **Alternative positional encodings:** Multiple variants beyond Golden Gate RoPE were tested without significant benefits.

---

## 11. Transferable Techniques for Physics Equation Derivation

### 11.1 Directly Transferable

1. **Masked diffusion for symbolic expressions:** The concept of starting with a fully masked equation template and progressively demasking tokens maps directly to physics equation derivation. Instead of grid cells, the masked positions represent equation tokens (variables, operators, constants). The bidirectional context is crucial -- knowing that a term involves "acceleration" constrains both the left-hand side (likely force-related) and right-hand side (likely involving mass).

2. **Token algebra / soft-masking refinement:** The continuous-space refinement loop is highly transferable. For equation derivation, this allows the model to iteratively refine a partially-correct equation in continuous space, adjusting coefficients and operators without committing to discrete choices prematurely. This is analogous to how physicists refine hypothesized equations.

3. **Recursive latent sampling:** The iterative refinement loop with cold restarts directly applies to equation search -- generating candidate equations, refining them, and restarting to escape local optima. The continuous-space operation is particularly valuable for numerical coefficients in physics equations.

4. **Test-time fine-tuning with augmentations:** Per-problem LoRA fine-tuning is directly applicable. For physics, augmentations would include: variable renaming, unit scaling, coordinate transformations, and noise perturbation of observations. The most-visited-candidate selection provides a natural way to identify robust equation candidates.

5. **Most-visited-candidate selection:** Tracking which equation forms appear most frequently across multiple refinement runs and augmentation perspectives provides a robust selection mechanism superior to single-pass confidence scoring.

### 11.2 Requiring Adaptation

1. **2D Positional encoding:** ARC's 2D grid structure does not directly map to physics data (which may be 1D time series, multi-dimensional phase spaces, or tabular observations). However, the principle of encoding structural relationships via multi-directional RoPE variants can be adapted: separate frequency bands for data index (time step) and value magnitude, or encoding both the input variable identity and its position in the observation sequence.

2. **Shape prediction:** In ARC, output dimensions are unknown. For physics, the "shape" analog is equation complexity (number of terms, nesting depth). A separate model predicting equation template structure before filling in specific tokens could be valuable.

3. **Grid expansion to uniform size:** For physics, this translates to normalizing input observation sets to fixed-length sequences, potentially with padding and positional encoding to handle variable-length experimental data.

### 11.3 Novel Adaptations Needed

1. **Dimensional analysis constraints:** Physics equations must satisfy dimensional consistency, which has no analog in ARC. Adding dimensional attention masks that prevent dimensionally invalid sub-expressions is a physics-specific innovation.

2. **Numerical verification feedback:** Physics equations can be numerically evaluated against observations, providing a quantitative fitness signal. Integrating this as a feedback signal into the refinement loop (conditioning the next iteration on R-squared or residual error) goes beyond ARC's pattern-matching paradigm.

3. **Hierarchical equation structure:** Physics equations have explicit tree structure (operator precedence, nested functions) that grid puzzles do not. The masked diffusion approach may need to operate on expression trees rather than flat token sequences, or use tree-positional encodings.

---

## 12. Key Takeaways

1. **Masked diffusion outperforms autoregressive for structured spatial reasoning.** The shift from Mistral-NeMo (AR) to LLaDA (diffusion) produced substantial gains on ARC, and the bidirectional context is equally important for equation derivation where all terms are mutually constraining.

2. **Continuous-space refinement is more powerful than discrete resampling.** The token algebra innovation -- operating in embedding space rather than token space -- enables smoother optimization landscapes for iterative refinement. This is directly applicable to refining symbolic expressions.

3. **Test-time specialization is critical.** Per-task LoRA fine-tuning with only 128 steps provides significant accuracy gains. For physics, per-equation-type or per-experiment fine-tuning could similarly boost performance.

4. **The combination of pretraining breadth and test-time depth is the winning formula.** Broad pretraining on diverse data provides general capability; test-time fine-tuning provides problem-specific adaptation. This two-stage paradigm should be adopted for physics equation derivation.

---

## Sources

- [The ARChitects - Technical Report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)
- [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [LLaDA: Large Language Diffusion Models (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992)
- [LLaDA Official Repository (ML-GSAI)](https://github.com/ML-GSAI/LLaDA)
- [On N-dimensional Rotary Positional Embeddings (Golden Gate RoPE)](https://jerryxio.ng/posts/nd-rope/)
- [Spiral RoPE: Multi-directional Positional Encoding](https://arxiv.org/html/2602.03227v1)
- [ARC Prize 2025 Competition Details](https://arcprize.org/competitions/2025/)
- [The (LLM) Architects - GTC 2025 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtc25-s74252/)
- [ARC-AGI 2025: A Research Review](https://lewish.io/posts/arc-agi-2025-research-review)
- [ARC Prize 2025: Technical Report (arXiv:2601.10904)](https://arxiv.org/abs/2601.10904)
