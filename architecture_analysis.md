# Architecture Analysis: ARChitects ARC 2025 Solution and Transfer to Physics Equation Derivation

## Research Context

The ARChitects team achieved a breakthrough in the ARC (Abstraction and Reasoning Corpus) 2025 competition by adapting LLaDA-8B, a masked diffusion language model, for abstract visual reasoning tasks. Their solution combined several innovative techniques: variable-ratio masked diffusion, 2D rotary positional embeddings, a novel "token algebra" soft-masking iterative refinement strategy, test-time finetuning with LoRA adapters, and a most-visited-candidate selection heuristic. Each of these techniques addresses a distinct challenge in reasoning under uncertainty, and together they form a coherent system that dramatically outperforms standard autoregressive approaches on grid-based reasoning tasks.

Our project, PhysDiffuser, aims to derive physics equations from numerical observations using a masked discrete diffusion transformer. The core hypothesis is that the same inductive biases that allow a masked diffusion model to reason about abstract grid patterns can be repurposed for symbolic regression -- the task of recovering an analytic equation from input-output data. Physics equation derivation shares several structural similarities with ARC tasks: both require identifying latent structure from examples, both benefit from iterative refinement rather than single-pass prediction, and both demand robustness to ambiguity (multiple valid solutions may exist). This document provides a detailed analysis of each ARChitects technique and maps it to a concrete adaptation strategy for PhysDiffuser.

---

## 1. LLaDA Masked Diffusion Mechanism

### Background

LLaDA (Large Language Diffusion with mAsking), introduced by Nie et al. (2025, arXiv:2502.09992), is a large-scale masked diffusion language model that reimagines text generation as a discrete denoising process. Unlike autoregressive models that generate tokens left-to-right, LLaDA generates all tokens simultaneously by learning to reverse a masking corruption process.

### Forward Process

The forward diffusion process in LLaDA is straightforward: given a sequence of tokens, each token is independently replaced by a special `[MASK]` token with probability `t`, where `t` is sampled uniformly from `[0, 1]` during training. This means the model sees sequences at every corruption level -- from nearly intact (low `t`) to almost fully masked (high `t`). The continuous uniform distribution over masking ratios is a critical design choice that distinguishes LLaDA from BERT-style masked language models. BERT uses a fixed masking ratio of approximately 15%, which makes it effective for representation learning but unsuitable as a generative model because it never learns to generate from scratch (i.e., from a fully masked sequence).

### Reverse Process and Training

The reverse process is a neural network (a transformer) that takes a partially masked sequence and predicts the identity of each masked token. Training uses standard cross-entropy loss computed only on the masked positions. The variable masking ratio is what transforms this seemingly simple setup into a proper generative model: by integrating over all possible masking ratios, the training objective provides a variational lower bound on the log-likelihood of the data. This theoretical grounding means that LLaDA can be used for unconditional generation, conditional generation, and likelihood evaluation -- all capabilities that BERT lacks.

### Scale and Performance

LLaDA was scaled to 8 billion parameters and trained on 2.3 trillion tokens. At this scale, it achieves competitive performance with autoregressive models of similar size on standard language modeling benchmarks, demonstrating that masked diffusion is a viable alternative to autoregressive generation even for open-ended text.

### Transfer to Physics Equation Derivation

For PhysDiffuser, we adapt the masked diffusion mechanism to operate on equation token sequences in prefix notation. Instead of masking natural language tokens, we mask equation tokens drawn from our vocabulary of mathematical operators (`+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `sqrt`, `pow`), variables (`x1` through `x9`), and constant placeholders (`C`).

The variable masking ratio is particularly valuable for physics equation derivation. At high masking ratios, the model learns to propose plausible equation structures given only the observation encoding and a few revealed tokens. At low masking ratios, it learns to make fine-grained corrections to nearly-complete equations. This spectrum of corruption levels maps naturally to the different stages of equation derivation: early stages require broad structural hypotheses (e.g., "this looks like an exponential decay"), while later stages require precise refinement (e.g., "the exponent should be negative, not positive").

Furthermore, the variational lower bound property ensures that our model can serve as a proper generative model over the space of equations, enabling principled sampling and likelihood-based ranking of candidate equations. This is a significant advantage over deterministic regression approaches that produce a single point estimate.

The training procedure remains simple: sample an equation from the training set, mask tokens at a random ratio, and train the transformer to reconstruct the masked positions given the observation encoding as conditioning. The cross-entropy loss on masked positions only means the model never wastes capacity learning to copy already-known tokens, focusing entirely on the generative task.

---

## 2. 2D RoPE Adaptation Rationale

### Background

Rotary Position Embeddings (RoPE) encode positional information by rotating token embeddings in complex space, providing a natural way to capture relative position information. Standard RoPE operates in one dimension, encoding the sequential position of each token.

### ARChitects' 2D Adaptation

ARC tasks are inherently two-dimensional: inputs and outputs are colored grids. The ARChitects recognized that 1D positional encoding destroys critical spatial information when grids are flattened into sequences. Their solution, inspired by "Golden Gate RoPE," replaced 1D RoPE with a 2D variant that encodes both row and column positions separately. They standardized all input and output grids to a uniform 32x32 size using delimiter symbols, and assigned position IDs that preserved 2D spatial awareness. This allowed the transformer to reason about spatial relationships (adjacency, symmetry, containment) that are fundamental to ARC tasks.

### Transfer to Physics Equation Derivation

Physics equations are inherently one-dimensional sequences, so a direct 2D RoPE adaptation is unnecessary for the equation decoder. However, the deeper insight from the ARChitects' approach is that positional encoding should be matched to the inherent structure of the data. This principle leads us to a heterogeneous positional encoding strategy in PhysDiffuser:

**Observation Encoder**: The input to our system is a set of observation points `{(x_i, y_i)}`. This set is fundamentally unordered -- permuting the observation points does not change the underlying equation. Therefore, we use a set transformer architecture with induced set attention blocks (ISAB) that is naturally permutation-invariant. No positional encoding is applied to observation points, as doing so would introduce a spurious ordering bias.

**Equation Decoder**: The output equation in prefix notation is an ordered sequence where position matters (e.g., `[* x1 x2]` means `x1 * x2`, while `[* x2 x1]` means `x2 * x1`, which is semantically equivalent but syntactically distinct). We use standard 1D sinusoidal positional encodings for the equation token positions.

**Hybrid Conditioning**: The cross-attention mechanism bridges these two domains -- the decoder attends to the set-encoded observation representations when predicting each equation token. This design ensures that the geometric structure of the input data is respected (unordered set) while the sequential structure of the output is properly encoded (ordered sequence).

The key takeaway from the ARChitects' 2D RoPE adaptation is not the specific dimensionality, but the principle that mismatched positional encoding introduces inductive bias misalignment that degrades performance. By carefully matching our positional encoding to each component's data structure, we avoid this pitfall.

---

## 3. Token Algebra / Soft-Masking Iterative Refinement

### Background

This is arguably the most innovative technique in the ARChitects solution and the one with the most profound implications for PhysDiffuser. The key insight is deceptively simple: tokens are not discrete symbols but points in a continuous embedding space, and linear combinations of embeddings are meaningful.

### The Mechanism

In standard masked diffusion inference, the model predicts token identities for masked positions, then samples discrete tokens, and re-encodes them for the next refinement step. This hard discretization creates information bottlenecks -- once a token is sampled, the model's uncertainty about that position is lost. The ARChitects bypassed this by adding mask embeddings to all positions simultaneously:

```
embedding_refined = embedding_color_ANY * 1.0 + embedding_MASK * 1.0
```

Instead of replacing predicted tokens with their discrete embeddings, the model maintains a superposition of the predicted token embedding and the mask embedding at every position. This creates what the ARChitects described as a "natural form of recursive self-improvement." The model iteratively refines its predictions, with the mask embedding component acting as a persistent signal that says "this position may still need revision."

### Inference Procedure

The ARChitects used 102 inference steps organized as two rounds of 51 steps with a restart between them. The restart mechanism resets the refinement trajectory, providing an implicit form of diversity that prevents the model from getting stuck in local optima. This two-round structure with restart is reminiscent of simulated annealing, where periodic temperature increases allow the system to escape shallow energy minima.

### Transfer to Physics Equation Derivation (Core Innovation)

This technique forms the backbone of PhysDiffuser's inference procedure. We adapt it as follows:

**Soft-Masking During Refinement**: At each refinement step, we compute the predicted equation token embeddings from the transformer's output logits, but instead of taking the argmax or sampling, we add a learnable mask embedding vector to every position:

```
h_t[i] = softmax(logits[i]) @ E + alpha_t * e_mask
```

where `E` is the token embedding matrix, `e_mask` is a learned mask embedding, and `alpha_t` is a schedule parameter that decreases over refinement steps. Early in refinement, `alpha_t` is large, signaling high uncertainty; late in refinement, `alpha_t` approaches zero, allowing the model to commit to its predictions.

**Advantages Over Hard Re-Masking**: The soft-masking approach preserves gradient flow through the refinement process, enabling end-to-end training of the refinement dynamics if desired. It also allows the model to "hedge" on uncertain positions -- for example, if a position could be either `sin` or `cos`, the soft embedding will lie between these two embeddings in the continuous space, and subsequent refinement steps can resolve the ambiguity based on the global context.

**Physical Analogy**: This mirrors the actual process of physics equation derivation. A physicist examining data might initially hypothesize a generic oscillatory behavior (uncertain between sin and cos), then refine this hypothesis as they examine boundary conditions and asymptotic behavior. The soft-masking process encodes exactly this kind of graduated certainty.

**Refinement Schedule**: We use 50 refinement steps with a linear decay schedule for `alpha_t`, from 1.0 to 0.0. Unlike the ARChitects' two-round restart, we instead sample multiple independent trajectories (see Section 5) for diversity.

---

## 4. Test-Time Finetuning with LoRA

### Background

Low-Rank Adaptation (LoRA) adds small trainable rank-decomposition matrices to frozen transformer layers, enabling efficient task-specific adaptation with minimal parameter overhead. The ARChitects applied LoRA at test time to adapt the pretrained LLaDA model to each individual ARC task.

### ARChitects' Implementation

The ARChitects used rank-32 LoRA adapters and ran 128 training steps with batch size 1 per task. They applied random augmentations at each step to prevent overfitting to the small number of examples (typically 2-3 input-output pairs per ARC task). The finetuning was executed on Kaggle L4 GPUs and needed to complete within the competition's inference time budget. This test-time finetuning was one of the key differentiators in their solution, allowing the model to specialize its representations for each specific task rather than relying solely on the general-purpose pretrained weights.

### Transfer to Physics Equation Derivation

We adapt test-time finetuning for PhysDiffuser with modifications appropriate to our CPU-constrained setting and physics domain:

**LoRA Configuration**: We apply rank-8 LoRA adapters to query and value projection matrices in all transformer layers. The choice of rank 8 (rather than 32) is driven by CPU memory constraints -- rank-8 adds approximately 500K parameters total, which is well within our 16GB RAM budget even when combined with the base model.

**Adaptation Loop**: For each test equation, we run 32 adaptation steps (reduced from 128 to meet CPU time budgets) with the following self-supervised procedure:
1. Take the current best-guess equation from the most recent refinement trajectory
2. Mask random subsets of the equation tokens at variable ratios
3. Train the LoRA adapters to reconstruct the masked tokens, using the observation encoding as conditioning
4. Additionally augment by adding Gaussian noise to the observation input points and repeating the masking-reconstruction task

**Intuition**: This adaptation loop specializes the model to the specific observation-equation relationship at hand. The base model provides a strong prior over the space of equations, while the LoRA adapters fine-tune this prior based on the specific data distribution. Observation augmentation with noise ensures that the adapted model is robust to the particular sampling of observation points, rather than overfitting to the exact numerical values.

**CPU Feasibility**: At rank 8, the LoRA parameters are small enough that the adaptation loop completes well within our 20-second per-equation time budget on CPU. The key optimization is that we only backpropagate through the LoRA parameters, not the full model, reducing the computational cost of each gradient step by approximately one to two orders of magnitude.

---

## 5. Most-Visited-Candidate Selection

### Background

Rather than relying on a single deterministic inference pass, the ARChitects tracked which solutions appeared most frequently during the iterative refinement process and selected the top candidates based on visit count. This is conceptually related to Monte Carlo Tree Search (MCTS) in game playing, where moves visited most frequently by the search process are considered most promising.

### ARChitects' Implementation

During the 102-step refinement process (and across multiple independent runs), the ARChitects recorded the predicted grid at each step. They then counted how many times each unique solution appeared and selected the top two candidates as their final submissions (the ARC competition allowed two guesses per task). This provided an implicit ensemble effect over the refinement trajectory, where solutions that the model converges to repeatedly from different starting points are more likely to be correct.

### Transfer to Physics Equation Derivation

We extend this concept for PhysDiffuser with additional physics-specific selection criteria:

**Multi-Trajectory Sampling**: We run 8 independent refinement trajectories, each initialized with a different random seed for the initial masking pattern. Each trajectory consists of 50 refinement steps. At every fifth step in each trajectory, we decode the current soft embeddings to a discrete equation by taking the argmax at each position.

**Visit Counting**: We canonicalize each decoded equation using SymPy simplification (to merge algebraically equivalent forms such as `x + x` and `2*x`) and maintain a frequency count across all trajectories and checkpointed steps. This gives us approximately 80 candidate equations (8 trajectories times 10 checkpoints), many of which will be duplicates after canonicalization.

**Physics-Informed Ranking**: Unlike ARC tasks where the output is a grid that can be checked for exact correctness, physics equations admit continuous parameters. For each unique candidate equation, we:
1. Fit any constant placeholders (`C`) using BFGS optimization against the observation data
2. Compute the R-squared score on a held-out validation subset of the observation points
3. Compute equation complexity as the total node count in the expression tree

The final selection criterion combines visit frequency, R-squared score, and a complexity penalty (Occam's razor):

```
score = log(visit_count + 1) + lambda_fit * R2 - lambda_complexity * node_count
```

This scoring function balances three desiderata: the model's internal confidence (visit count), empirical fit quality (R-squared), and parsimony (complexity penalty). The weighting parameters `lambda_fit` and `lambda_complexity` are tuned on a validation set of known equations.

**Confidence Estimation**: A natural byproduct of the visit-counting approach is a confidence estimate. If 7 out of 8 trajectories converge to the same equation, we have high confidence in the prediction. If the trajectories produce 6 distinct equations, we know the model is uncertain. This confidence estimate can be reported to the user as a reliability indicator, which is valuable in scientific applications where incorrect equations could lead to flawed physical predictions.

---

## Summary Table

The following table maps each ARChitects technique to our PhysDiffuser adaptation, with a feasibility assessment for CPU-only deployment.

| # | ARChitects Technique | PhysDiffuser Adaptation | Key Modification | CPU Feasibility |
|---|---------------------|------------------------|-------------------|-----------------|
| 1 | **LLaDA Masked Diffusion** (variable-ratio token masking, cross-entropy on masked positions) | Masked diffusion over prefix-notation equation tokens conditioned on observation encodings | Conditioning on set-encoded observations instead of grid context; smaller vocabulary (~50 tokens vs 32K) | High -- smaller model (35-80M vs 8B params) makes training and inference tractable on CPU |
| 2 | **2D RoPE** (spatial positional encoding for grids, 32x32 uniform size) | Heterogeneous positional encoding: no positional encoding for observation set (permutation-invariant set transformer), 1D sinusoidal for equation sequence | Match positional encoding to data structure rather than forcing spatial encoding | High -- standard sinusoidal encoding has zero computational overhead; set transformer adds minimal cost |
| 3 | **Token Algebra / Soft-Masking** (add mask embeddings to all positions, 102 iterative refinement steps with restart) | Soft-mask refinement with decaying mask embedding weight `alpha_t` over 50 steps; multiple trajectories replace restart mechanism | Decaying schedule replaces fixed superposition; fewer steps offset by multiple trajectories | High -- each step is a single forward pass through a small transformer; 50 steps x 8 trajectories = 400 forward passes, feasible in ~15s on CPU |
| 4 | **Test-Time LoRA Finetuning** (rank-32, 128 steps, GPU-accelerated) | Rank-8 LoRA on Q/V projections, 32 adaptation steps with observation noise augmentation | Reduced rank (8 vs 32) and fewer steps (32 vs 128) for CPU budget; self-supervised masking-reconstruction objective | Moderate -- 32 backward passes through LoRA params (~500K) takes ~10-15s on CPU; within budget but is the tightest constraint |
| 5 | **Most-Visited-Candidate Selection** (frequency counting over refinement trajectory, top-2 selection) | Multi-trajectory visit counting + BFGS constant fitting + R-squared ranking + complexity penalty | Physics-informed ranking beyond pure frequency; constant fitting step; Occam's razor penalty | High -- SymPy canonicalization and BFGS fitting add ~2-5s total; negligible compared to refinement |

---

## Conclusion

The ARChitects ARC 2025 solution demonstrates that masked diffusion models with iterative refinement, test-time adaptation, and ensemble-based selection can solve abstract reasoning tasks that defeat standard autoregressive approaches. Each of these techniques has a natural and well-motivated adaptation for physics equation derivation:

- **Masked diffusion** provides a principled generative model over the discrete space of symbolic equations, with the variable masking ratio enabling learning at all corruption levels.
- **Structure-matched positional encoding** (the principle behind 2D RoPE) motivates our heterogeneous encoding strategy that respects the unordered nature of observations and the sequential nature of equations.
- **Soft-masking iterative refinement** (token algebra) is the most impactful technique for our setting, enabling the model to express and gradually resolve uncertainty about equation structure in a differentiable manner.
- **Test-time LoRA adaptation** allows the model to specialize to each specific equation's data distribution, compensating for the limited expressiveness of a CPU-sized model.
- **Most-visited-candidate selection** combined with physics-informed ranking (BFGS constant fitting, R-squared scoring, complexity penalties) provides robust final predictions with natural confidence estimates.

Together, these adaptations form a coherent system -- PhysDiffuser -- that applies the lessons of the ARChitects' competition-winning approach to the scientifically significant problem of autonomous physics equation derivation from numerical data. The feasibility analysis confirms that all techniques can be adapted to operate within CPU-only constraints (16GB RAM, less than 60 seconds per equation), making PhysDiffuser practical for deployment in resource-constrained research environments.
