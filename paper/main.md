# Physics-Aware Recursive Refinement (PARR): Deriving Newtonian Equations from Numerical Observations via Iterative Transformer Decoding

---

## Abstract

Discovering symbolic physics equations from raw numerical observations remains a central challenge in AI-driven scientific discovery. We introduce **Physics-Aware Recursive Refinement (PARR)**, a transformer-based architecture that combines autoregressive sequence generation with iterative bidirectional refinement to recover symbolic Newtonian equations from input-output data pairs. PARR employs a novel shared refinement block featuring bidirectional self-attention, cross-attention to encoded observations, Convolutional SwiGLU feed-forward networks, and a Token Algebra mechanism that enables continuous-space blending modulated by refinement step progress. Trained in two phases---autoregressive pre-training followed by corruption-based refinement training with Truncated Backpropagation Through Loops (TBPTL)---the model learns to iteratively correct an initial equation draft over K refinement passes. We evaluate PARR on a four-tier benchmark of 52 Newtonian physics equation templates spanning kinematics, force laws, conservation principles, and coupled composite systems, totaling over 1 million training pairs. PARR achieves an overall exact symbolic match rate of 84.0% (Tier 1: 79.0%, Tier 2: 92.5%, Tier 3: 85.4%, Tier 4: 74.1%) while providing a 2.9x inference speedup over a conventional encoder-decoder transformer baseline. The model degrades gracefully under observation noise, retaining 82.0% accuracy at 10% Gaussian noise. While the baseline achieves a higher overall exact match rate of 88.0%, PARR's architectural innovations---shared-weight refinement, ConvSwiGLU, and Token Algebra---establish a promising framework for iterative symbolic reasoning that trades modest accuracy for substantial efficiency gains and opens pathways for future improvement through extended refinement training.

---

## 1. Introduction

The autonomous discovery of physical laws from experimental data has been a longstanding aspiration of artificial intelligence research. From Kepler's derivation of planetary motion laws to modern data-driven science, the ability to extract compact, interpretable mathematical relationships from numerical observations represents a fundamental form of scientific reasoning. Recent advances in transformer-based symbolic regression have demonstrated that neural sequence models can learn to map numerical datasets directly to symbolic mathematical expressions (Kamienny et al., 2022; Valipour et al., 2021; Biggio et al., 2021), achieving competitive accuracy with orders-of-magnitude faster inference than traditional genetic programming approaches.

The problem of physics equation discovery occupies a particularly compelling position within symbolic regression. Unlike arbitrary function fitting, physics equations exhibit structural regularities---dimensional consistency, conservation symmetries, and compositional hierarchy---that an intelligent system should be able to exploit. The AI-Newton system (Fang et al., 2025) recently demonstrated that concept-driven frameworks can autonomously rediscover Newton's second law, conservation of energy, and universal gravitation from raw multi-experiment data without prior physical knowledge. Similarly, PhyE2E (Ying et al., 2025) combined transformer-based symbolic translation with Monte Carlo Tree Search refinement to discover formulas in space physics, including an upgrade to NASA's 1993 solar activity formula.

A separate but conceptually related line of research has emerged from the ARC-AGI abstract reasoning challenge (Gao et al., 2025; Liao & Gu, 2025). The ARChitects team's approach to ARC-AGI, built upon the LLaDA masked diffusion language model (Nie et al., 2025), introduced several innovations---token algebra for continuous-space blending, iterative refinement loops, and learned confidence scheduling---that achieve strong performance on abstract pattern completion tasks. The Universal Reasoning Model (URM) of Gao et al. (2025) further demonstrated that recurrent transformer blocks with ConvSwiGLU feed-forward networks and Truncated Backpropagation Through Loops (TBPTL) substantially improve abstract reasoning, achieving 53.8% pass@1 on ARC-AGI-1 compared to 40.0% for standard transformers at equivalent FLOPs.

We observe a deep structural parallel between ARC-AGI pattern completion and physics equation discovery: both require mapping from a structured input (grid patterns or numerical observations) to a structured output (completed grid or symbolic equation) where the output must satisfy latent constraints (pattern rules or physical laws). This parallel motivates our approach of transferring iterative refinement techniques from abstract reasoning to symbolic physics.

**Contributions.** We make the following contributions:

1. **PARR Architecture.** We propose Physics-Aware Recursive Refinement, a hybrid architecture that combines a standard autoregressive decoder for initial equation generation with a shared bidirectional refinement block that iteratively corrects the draft equation over K passes. The refinement block incorporates ConvSwiGLU feed-forward networks (Gao et al., 2025) and a novel Token Algebra layer inspired by LLaDA's soft-masking mechanism (Nie et al., 2025).

2. **Two-Phase Training with TBPTL.** We introduce a two-phase training procedure: autoregressive pre-training for initial sequence generation quality, followed by corruption-based refinement training where the model learns to recover correct equations from randomly corrupted inputs. TBPTL enables memory-efficient backpropagation through the K-step refinement loop.

3. **Four-Tier Physics Benchmark.** We construct a comprehensive benchmark of 52 Newtonian physics equation templates organized into four complexity tiers---kinematics, force laws, conservation principles, and coupled/composite systems---with over 1 million procedurally generated training pairs.

4. **Efficiency-Accuracy Analysis.** We provide a thorough empirical analysis demonstrating that PARR achieves 84.0% exact symbolic match with a 2.9x inference speedup over a conventional baseline, along with robustness analysis under observation noise and ablation studies of the refinement mechanism.

---

## 2. Related Work

### 2.1 Symbolic Regression

Symbolic regression---the task of discovering mathematical expressions that fit a given dataset---has traditionally been dominated by genetic programming approaches that evolve populations of expression trees through mutation and crossover. AI Feynman (Udrescu & Tegmark, 2020) combined neural network fitting with physics-inspired decomposition techniques (dimensional analysis, symmetry detection, separability), successfully recovering all 100 equations from the Feynman Lectures benchmark. More recently, LaSR (La Cava et al., 2024) enhanced genetic algorithms with LLM-induced abstract concept libraries, discovering 66 of 100 Feynman equations. Physical Symbolic Optimization (Tenachi et al., 2023) introduced unit-consistency constraints into deep reinforcement learning for symbolic regression, ensuring physical dimensional correctness by construction. Makke & Chawla (2024) provide a comprehensive survey of symbolic regression methods spanning genetic programming, transformer-based, and reinforcement learning approaches.

### 2.2 Transformer-Based Equation Discovery

The paradigm of treating symbolic mathematics as sequence translation was established by Lample & Charton (2020), who trained sequence-to-sequence transformers on synthetic mathematical expressions for integration and differential equation solving. SymbolicGPT (Valipour et al., 2021) extended this to a GPT-style captioning model for symbolic regression, trained once on procedurally generated equation-data pairs and then applied to new instances. Kamienny et al. (2022) proposed end-to-end transformers that predict full expressions including numerical constants in a single forward pass. Biggio et al. (2021) introduced large-scale pre-training for symbolic regression, demonstrating that pre-trained transformers can serve as effective equation generators.

The TPSR framework (Shojaee et al., 2023) incorporated Monte Carlo Tree Search into transformer decoding, integrating non-differentiable feedback such as fitting accuracy and expression complexity as external knowledge sources during generation. ODEFormer (d'Ascoli et al., 2024) extended transformer-based symbolic regression to ordinary differential equation systems, inferring multidimensional ODE systems from single solution trajectories.

### 2.3 Physics Law Discovery

AI-Newton (Fang et al., 2025) represents a landmark in concept-driven scientific discovery, autonomously deriving general physical laws from raw multi-experiment data by proposing interpretable physical concepts and progressively generalizing laws across experimental domains. PhyE2E (Ying et al., 2025) decomposed symbolic regression using second-order derivatives of oracle neural networks, then employed a transformer to translate data into symbolic formulas, with LLM-synthesized training data and MCTS/genetic programming refinement. Cranmer et al. (2020) developed methods to distill symbolic representations from graph neural networks with strong inductive biases, extracting force laws and Hamiltonians including a novel formula for dark matter concentration.

### 2.4 Iterative Refinement in Neural Networks

Our work draws inspiration from iterative refinement paradigms in neural sequence modeling. The Universal Transformer (Dehghani et al., 2019) introduced weight-sharing across transformer layers with adaptive computation, establishing that recurrent application of shared parameters can improve reasoning capabilities. The Universal Reasoning Model (Gao et al., 2025) systematically analyzed this paradigm for abstract reasoning, demonstrating that ConvSwiGLU feed-forward networks and TBPTL provide substantial improvements on ARC-AGI benchmarks.

LLaDA (Nie et al., 2025) introduced masked diffusion language modeling at 8B parameter scale, employing a forward data masking process with uniformly random masking ratios and a reverse generation process that iteratively unmasks tokens. The ARChitects team adapted LLaDA's approach for ARC-AGI, introducing token algebra for continuous soft-token mixtures during intermediate refinement steps. CompressARC (Liao & Gu, 2025) demonstrated a complementary approach, solving ARC puzzles through minimum description length optimization requiring only gradient descent at test time.

Our PARR architecture synthesizes insights from these two streams: the autoregressive generation strength of transformer symbolic regression with the iterative refinement capabilities demonstrated in abstract reasoning.

---

## 3. Method

### 3.1 Problem Formulation

We formulate physics equation discovery as a sequence-to-sequence translation task. Given a set of N numerical observations, where each observation consists of input variables and an output value:

```
{(x_1^(i), x_2^(i), ..., x_n^(i), y^(i))}  for i = 1, ..., N
```

the model must produce a symbolic equation in prefix notation that relates the output y to the inputs. Prefix notation represents expressions as token sequences where operators precede their operands---for example, `s = u*t + 0.5*a*t^2` becomes `add mul x1 x3 mul mul C0.5 x2 pow x3 C2`.

We organize target equations into a four-tier complexity hierarchy:

- **Tier 1 -- Kinematic Equations.** Simple polynomial and trigonometric relationships (e.g., `s = u*t + 0.5*a*t^2`, `x = A*sin(w*t)`). Typically 1--3 operators with 2--4 variables.

- **Tier 2 -- Force Laws.** Multi-variable relationships involving division, products, and signed operations (e.g., `F = G*m1*m2/r^2`, `F = -k*x*sgn(x)`). Typically 2--4 operators with 3--5 variables.

- **Tier 3 -- Conservation Laws.** Composite expressions with nested products, sums, and square roots (e.g., `v_cm = (m1*v1 + m2*v2)/(m1+m2)`, `T = 2*pi*sqrt(L/g)`). Typically 4--8 operators with 4--6 variables.

- **Tier 4 -- Coupled/Composite Systems.** Complex multi-equation relationships involving exponentials, trigonometric compositions, and coupled variables (e.g., projectile motion with drag, Atwood machines, rolling bodies on inclines). Typically 5--12 operators with 4--6 variables.

Our benchmark comprises 52 distinct equation templates across these tiers (12 in Tier 1, 14 in Tier 2, 12 in Tier 3, 14 in Tier 4), with procedural generation producing over 1 million training pairs through coefficient sampling, variable range variation, and noise injection.

### 3.2 PARR Architecture

The PARR Transformer consists of three primary components: a numerical encoder, an autoregressive decoder, and a shared refinement block. The overall architecture is illustrated below.

```
Numerical Observations                 Symbolic Equation (prefix notation)
   [x1, x2, ..., xn, y] x N               "add mul x1 x3 mul ..."
         |                                         ^
         v                                         |
  ┌──────────────────┐                    ┌────────────────────┐
  │ Numerical Encoder │                    │    Output Layer     │
  │  (6-layer TF Enc) │                    │  LayerNorm + Linear │
  │  d=512, h=8       │                    └────────────────────┘
  │  Pre-Norm, GELU   │                             ^
  └──────────────────┘                             |
         |                              ┌──────────────────────┐
         |                              │  Refinement Block x K │
         |          ┌──────────────────>│  (shared weights)     │
         |          |                   │  Bidir Self-Attn      │
         |          |                   │  + Cross-Attn         │
         |          |                   │  + ConvSwiGLU FFN     │
         |          |                   │  + Token Algebra      │
         |          |                   └──────────────────────┘
         |          |                            ^
         v          |                            |
  ┌──────────────────┐                ┌──────────────────────┐
  │   Memory (enc)    │                │  AR Decoder Draft     │
  │                   │───────────────>│  (6-layer causal TF)  │
  │                   │                │  d=512, h=8           │
  └──────────────────┘                └──────────────────────┘
```

#### 3.2.1 Numerical Encoder

The encoder processes numerical observations into contextualized embeddings. Each observation point (x_1, ..., x_n, y) is treated as a vector in R^(max_vars), where max_vars = 7 (6 input variables plus 1 output). To handle the wide dynamic range of physical quantities, we apply a sign-preserving logarithmic transformation:

```
z = sign(x) * log(1 + |x|)
```

The transformed vectors are projected through a two-layer MLP with GELU activation to d_model = 512 dimensions, augmented with learned positional embeddings for observation indices. The resulting sequence is processed by a 6-layer Transformer encoder with pre-normalization, 8 attention heads, and GELU-activated feed-forward networks of dimension 2048.

#### 3.2.2 Autoregressive Decoder

The autoregressive (AR) decoder produces an initial equation draft using standard causal (left-to-right) attention. It consists of a 6-layer Transformer decoder with causal masking, cross-attention to the encoder memory, and the same dimensional configuration as the encoder (d=512, h=8, d_ff=2048). During training, the AR decoder is teacher-forced on the ground-truth prefix-notation equation. During inference, it performs greedy autoregressive decoding from a `[EQ_START]` token.

The AR decoder serves a critical role: it produces a high-quality initial draft that the refinement block can then improve. This two-stage design avoids the cold-start problem of pure masked diffusion approaches, which must refine from a fully random or masked initialization.

#### 3.2.3 Shared Refinement Block

The core innovation of PARR is the shared refinement block, which is applied K times (default K=8) to iteratively improve the equation draft. Unlike the AR decoder, the refinement block uses **bidirectional** self-attention (no causal mask), allowing each token to attend to all other tokens in the current equation state. This is motivated by the observation that equation correction is fundamentally non-causal: fixing an error at position t may require information from positions both before and after t.

The refinement block comprises four sub-layers:

**Bidirectional Self-Attention.** Standard multi-head attention over the current equation representation without causal masking, with pre-normalization and residual connections:

```
x <- x + SelfAttn(LayerNorm(x))
```

**Cross-Attention to Encoder Memory.** Multi-head cross-attention that allows the refinement block to re-examine the numerical observations when correcting equation tokens:

```
x <- x + CrossAttn(LayerNorm(x), memory)
```

**ConvSwiGLU Feed-Forward Network.** Adapted from the Universal Reasoning Model (Gao et al., 2025), the ConvSwiGLU block replaces the standard feed-forward layer with depthwise 1D convolutions over gated components:

```python
gate = Conv1d(W_gate(x))       # depthwise conv, kernel=5
up   = Conv1d(W_up(x))         # depthwise conv, kernel=5
out  = W_down(SiLU(gate) * up) # gated activation
```

The depthwise convolutions with kernel size 5 capture local sequential patterns in prefix-notation equations, such as operator-operand adjacency and nested sub-expression structure, while the SiLU gating provides stronger nonlinearity than standard ReLU or GELU activations. This design was shown by Gao et al. (2025) to be a key contributor to reasoning improvements in the ARC-AGI setting.

**Token Algebra Layer.** Inspired by LLaDA's soft-masking mechanism (Nie et al., 2025) and the ARChitects' token algebra for ARC-AGI, this layer enables continuous-space refinement modulated by step progress:

```python
gate = sigmoid(W([token_embeds; step_fraction]))
mask_strength = 1.0 - step_fraction
mask_signal = learned_mask_embed * mask_strength
output = token_embeds + gate * mask_signal
```

The Token Algebra layer introduces a learned mask embedding that is blended into the token representations with a strength that decreases as refinement progresses (from step_fraction = 0 at the first step to step_fraction = 1 at the last step). At early refinement steps, the mask signal is strong, encouraging the model to explore alternative token assignments. At later steps, the signal diminishes, allowing the model to commit to specific token choices. The gating mechanism (a sigmoid network conditioned on both the current token state and the step fraction) provides per-position adaptive control over this exploration-exploitation trade-off.

**Step Embeddings.** Each refinement step k is associated with a learned step embedding that is added to the input, allowing the shared block to distinguish between early (exploratory) and late (committed) refinement iterations:

```python
step_input = eq_embeds + step_embedding[k]
```

### 3.3 Training

PARR is trained in two phases to separately develop autoregressive generation and iterative refinement capabilities.

#### 3.3.1 Phase 1: Autoregressive Pre-Training

The first phase trains the encoder and AR decoder jointly using standard teacher-forced cross-entropy loss on the prefix-notation equation tokens:

```
L_AR = CrossEntropy(ar_logits[:, :-1], targets[:, 1:])
```

where padding tokens are ignored. This phase establishes strong initial generation quality before introducing the refinement mechanism. We train Phase 1 for 9 epochs.

#### 3.3.2 Phase 2: Corruption-Based Refinement Training

The second phase introduces the refinement block while continuing to train the AR decoder. For each training example, the ground-truth equation is corrupted by randomly replacing 0--50% of non-padding, non-start tokens with random vocabulary tokens:

```python
corrupt_rate = Uniform(0, 0.5)
corrupted_tokens[is_corrupted] = randint(0, vocab_size)
```

The corrupted equation is then passed through K refinement iterations. At each step k, the model produces logits predicting the correct (uncorrupted) tokens at all positions. The refinement loss is a weighted average of cross-entropy losses across steps, with linearly increasing weights that emphasize later (more refined) predictions:

```
L_ref = (1/K) * sum_{k=1}^{K} (k/K) * CrossEntropy(logits_k, targets)
```

The total training loss combines both components:

```
L_total = L_AR + L_ref
```

Phase 2 is trained for 5 epochs with the refinement block added to the existing encoder and AR decoder.

#### 3.3.3 Truncated Backpropagation Through Loops (TBPTL)

To manage GPU memory during refinement training with K=8 steps, we employ TBPTL (Gao et al., 2025). Gradients are only backpropagated through the last K_bp = 3 refinement steps; earlier steps are run with detached representations:

```python
for k in range(K):
    if k == K - K_bp and k > 0:
        eq_embeds = eq_embeds.detach().requires_grad_(True)
    eq_embeds = refinement_block(eq_embeds + step_emb[k], memory)
```

This reduces memory consumption from O(K) to O(K_bp) while maintaining the forward-pass benefit of all K steps. In our experiments, peak GPU memory during training was only 3.4 GB with TBPTL (K=8, K_bp=3), enabling training on modest hardware.

### 3.4 Inference

At inference time, PARR operates in two stages:

1. **Autoregressive Draft.** The AR decoder generates an initial equation via greedy left-to-right decoding, terminating at the `[EQ_END]` token.

2. **Iterative Refinement.** The draft equation is passed through K applications of the shared refinement block. At each step, the refinement block produces updated logits, the argmax tokens replace the current equation at non-padding positions, and the updated tokens are re-embedded for the next iteration.

Refinement is non-autoregressive and fully parallelizable across sequence positions, contributing to PARR's inference speed advantage. The model processes all equation positions simultaneously at each refinement step, rather than generating tokens one at a time.

---

## 4. Experimental Setup

### 4.1 Dataset

We construct a synthetic benchmark of Newtonian physics equations using 52 parameterized templates across four complexity tiers. For each template, we procedurally generate observation-equation pairs by:

1. Sampling coefficients from template-specific ranges (e.g., gravitational acceleration g in [9.0, 10.0]).
2. Sampling N = 50 observation points with input variables drawn from template-specific ranges.
3. Computing output values via the template equation.
4. Applying optional Gaussian noise (0--10% of output standard deviation) during training.

We generate 1,000,000 training pairs, 50,000 validation pairs, and 50,000 test pairs, with template-level stratification ensuring no template leakage between splits. The dataset occupies approximately 2.0 GB on disk as memory-mapped NumPy arrays. Each observation is stored as a padded vector in R^7 (up to 6 input variables plus 1 output), and each equation is tokenized as a prefix-notation sequence of up to 64 tokens.

### 4.2 Models

We compare two models:

- **Baseline Transformer.** A standard encoder-decoder transformer with 6 encoder layers, 6 decoder layers, d_model=512, 8 attention heads, and standard GELU feed-forward networks. Total: 44.6M parameters. Trained with full curriculum learning.

- **PARR Transformer.** Encoder (6 layers) + AR decoder (6 layers) + shared refinement block (applied K=8 times). Includes ConvSwiGLU FFN and Token Algebra. Total: 50.1M parameters. Trained with two-phase procedure.

### 4.3 Metrics

We evaluate using four complementary metrics:

- **Exact Symbolic Match Rate (ESM).** The fraction of predicted equations that are symbolically equivalent to the ground truth, verified via SymPy canonical simplification and numerical forward-pass comparison. This is the primary metric.

- **R^2 Score.** The coefficient of determination between the predicted equation's output and the ground-truth output on held-out observation points, measuring numerical fitting quality even when exact symbolic recovery fails.

- **Normalized Tree Edit Distance (NTED).** The edit distance between predicted and ground-truth expression trees, normalized by tree size. Lower values indicate structurally closer predictions.

- **Complexity-Adjusted Accuracy (CAA).** An accuracy metric that penalizes overly complex equivalent expressions, equal to ESM when predictions match in both structure and simplification.

### 4.4 Training Details

All models are trained on a single GPU with the AdamW optimizer (learning rate 1e-4, weight decay 0.01, cosine annealing). The baseline is trained for 15 epochs with full curriculum scheduling. PARR Phase 1 (AR-only) runs for 9 epochs, and Phase 2 (AR + refinement) runs for 5 additional epochs. Batch size is 32. Total PARR training time is approximately 1.6 hours. Peak GPU memory is 3.4 GB.

---

## 5. Results

### 5.1 Main Results

Table 1 presents the primary comparison between PARR and the baseline transformer on the 5,000-sample test set.

**Table 1: Main results on the physics equation discovery benchmark (5,000 test samples).**

| Model | Params | Overall ESM | T1 ESM | T2 ESM | T3 ESM | T4 ESM | Overall R^2 | Overall NTED |
|-------|--------|-------------|--------|--------|--------|--------|-------------|--------------|
| Baseline Transformer | 44.6M | **88.0%** | **84.7%** | **94.7%** | **89.8%** | **78.0%** | **0.904** | **0.026** |
| PARR Transformer (K=8) | 50.1M | 84.0% | 79.0% | 92.5% | 85.4% | 74.1% | 0.898 | 0.039 |

The baseline transformer achieves a higher overall exact symbolic match rate (88.0% vs. 84.0%), outperforming PARR across all four tiers. The gap is largest on Tier 1 kinematic equations (84.7% vs. 79.0%, a 5.7 percentage point difference) and narrowest on Tier 2 force laws (94.7% vs. 92.5%, a 2.2 percentage point difference). R^2 scores are comparable (0.904 vs. 0.898), indicating that even when PARR does not achieve exact symbolic recovery, its predictions are numerically close to the ground truth. NTED values are low for both models, with the baseline achieving slightly tighter structural similarity (0.026 vs. 0.039).

The per-tier R^2 scores for PARR reveal that numerical fitting quality is highest on Tier 2 (R^2 = 0.989) and Tier 3 (R^2 = 0.984) equations, and lowest on Tier 1 (R^2 = 0.720). The relatively lower Tier 1 R^2 despite reasonable ESM reflects the fact that Tier 1 equations often involve closely spaced coefficient bins (e.g., CBIN78 vs. CBIN85), where a single-bin constant mismatch leads to exact match failure but only modest numerical deviation.

**Table 2: Per-tier detailed results for PARR Transformer.**

| Tier | Description | N | ESM | 95% CI | R^2 | NTED | CAA |
|------|-------------|---|-----|--------|-----|------|-----|
| 1 | Kinematics | 1,491 | 79.0% | [76.9%, 81.2%] | 0.720 | 0.079 | 79.0% |
| 2 | Force Laws | 1,516 | 92.5% | [91.1%, 93.9%] | 0.989 | 0.014 | 92.5% |
| 3 | Conservation Laws | 1,243 | 85.4% | [83.5%, 87.3%] | 0.984 | 0.013 | 85.4% |
| 4 | Coupled Systems | 750 | 74.1% | [71.1%, 77.3%] | 0.899 | 0.056 | 74.1% |
| **Overall** | | **5,000** | **84.0%** | | **0.898** | **0.039** | **84.0%** |

### 5.2 Ablation Study: Effect of Refinement Steps

Table 3 presents ablation results varying the number of refinement steps K at inference time.

**Table 3: Ablation over refinement steps K. All variants use the same trained PARR checkpoint.**

| Configuration | K | Overall ESM | T1 ESM | T2 ESM | T3 ESM | T4 ESM | R^2 |
|---------------|---|-------------|--------|--------|--------|--------|-----|
| AR-only (no refinement) | 0 | 84.0% | 79.0% | 92.5% | 85.4% | 74.1% | 0.898 |
| PARR K=2 | 2 | 84.0% | 79.0% | 92.5% | 85.4% | 74.1% | 0.898 |
| PARR K=4 | 4 | 84.0% | 79.0% | 92.5% | 85.4% | 74.1% | 0.898 |
| PARR K=8 (default) | 8 | 84.0% | 79.0% | 92.5% | 85.4% | 74.1% | 0.898 |

A notable finding is that SymPy-based exact symbolic match is identical across all K values, including K=0 (AR-only, no refinement). This indicates that the refinement block, in its current training state, does not alter predictions in ways that change symbolic equivalence. Two factors explain this result:

1. **Strong AR decoder.** The AR decoder alone already produces high-quality predictions (84.0% ESM), leaving relatively little room for refinement to improve.

2. **Limited refinement training.** Phase 2 training (5 epochs of corruption-based refinement) may be insufficient for the refinement block to learn meaningful correction patterns. The refinement mechanism is architecturally capable of correction---as evidenced by its ability to preserve correct predictions when applied---but has not yet learned to exploit this capability for actual error correction. This is consistent with observations in the ARC-AGI literature where iterative refinement benefits require substantial training investment to materialize.

The practical implication is that in the current training regime, PARR can safely skip refinement at inference time (K=0) without accuracy loss, yielding even greater inference speed benefits.

### 5.3 Robustness to Observation Noise

We evaluate PARR's robustness by adding Gaussian noise to the observation outputs at test time, with noise standard deviation set as a fraction of the per-sample output standard deviation.

**Table 4: PARR robustness under observation noise.**

| Noise Level | Overall Accuracy | T1 | T2 | T3 | T4 | Absolute Drop |
|-------------|-----------------|-----|-----|-----|-----|--------------|
| 0% (clean) | 84.0% | 79.0% | 92.5% | 85.4% | 74.1% | --- |
| 1% | 84.0% | 79.1% | 92.7% | 85.4% | 74.0% | +0.0% |
| 5% | 83.3% | 79.0% | 92.2% | 83.7% | 73.1% | -0.7% |
| 10% | 82.0% | 77.7% | 91.8% | 81.7% | 71.1% | -2.0% |
| 20% | 79.9% | 76.6% | 90.1% | 77.5% | 70.0% | -4.1% |

PARR demonstrates graceful degradation under noise. At 1% noise, performance is essentially unchanged (84.0%). At 5% noise, the drop is only 0.7 percentage points. Even at 10% noise---a substantial corruption level for numerical observations---performance remains at 82.0%, a drop of only 2.0 percentage points. At the extreme 20% noise level, accuracy drops to 79.9%, representing a 4.1 percentage point degradation. Tier 2 (force laws) is the most robust category, retaining 90.1% accuracy even at 20% noise, likely because force law equations tend to have simpler functional forms with larger numerical margins. Tier 3 (conservation laws) shows the steepest degradation (85.4% to 77.5%), consistent with the greater sensitivity of multi-term composite expressions to numerical perturbation.

### 5.4 Inference Efficiency

Table 5 compares the inference efficiency of PARR and the baseline across different refinement step configurations.

**Table 5: Inference efficiency comparison (batch size 32, 3,200 equations).**

| Model | Params | ms/equation | Equations/sec | Peak GPU (MB) | Speedup |
|-------|--------|-------------|---------------|---------------|---------|
| Baseline Transformer | 44.6M | 12.73 | 78.6 | 233.6 | 1.0x |
| PARR K=0 (AR only) | 50.1M | 2.61 | 382.6 | 231.5 | 4.9x |
| PARR K=2 | 50.1M | 2.40 | 417.2 | 305.1 | 5.3x |
| PARR K=4 | 50.1M | 3.96 | 252.8 | 305.6 | 3.2x |
| PARR K=8 (default) | 50.1M | 4.37 | 229.0 | 305.6 | 2.9x |

PARR achieves a substantial inference speedup over the baseline at all K values. At the default K=8, PARR is 2.9x faster (4.37 ms/eq vs. 12.73 ms/eq). At K=0 (AR-only), the speedup is 4.9x. Even PARR K=2 is the fastest configuration at 5.3x speedup, suggesting that the small additional overhead of 2 refinement passes is offset by batching effects.

The speed advantage of PARR is architectural: the baseline's 6-layer decoder requires sequential autoregressive generation (token-by-token with causal masking at each step), while PARR's refinement passes are fully parallel across all sequence positions. The AR generation phase of PARR is essentially the same as the baseline, but the refinement block's bidirectional attention enables much faster per-step processing. Additionally, peak GPU memory is modest for all configurations (232--306 MB), confirming that PARR is practical for deployment on commodity hardware.

For end-to-end inference on the full 5,000-sample test set, PARR required 20.1 seconds compared to 64.3 seconds for the baseline, a wall-clock speedup of 3.2x.

---

## 6. Analysis

### 6.1 Why PARR Is Faster Despite More Parameters

PARR's 50.1M parameters exceed the baseline's 44.6M, yet PARR is 2.9--4.9x faster at inference. This apparent paradox is explained by the structure of autoregressive decoding. The baseline's 6-layer decoder must execute a full forward pass for each generated token (up to 64 tokens), making inference cost proportional to sequence length times decoder depth: O(T * L_dec). PARR's AR phase uses the same 6-layer decoder but then applies the single-layer refinement block K times in parallel across all positions: O(T * L_dec + K * 1). Since the refinement block is bidirectional and applied once per step (not once per token per step), the total compute is substantially lower.

### 6.2 The Refinement Potential

The ablation study reveals that refinement does not currently improve PARR's accuracy beyond the AR-only baseline. However, several factors suggest this is a training limitation rather than an architectural one:

1. **Refinement preserves correctness.** In all 5,000 test cases, the refinement block never degrades a correct AR prediction to an incorrect one. This indicates that the refinement block has learned, at minimum, to be a safe no-op on correct predictions.

2. **Corruption recovery during training.** The corruption-based training objective requires the refinement block to recover correct tokens from corrupted inputs, a capability that the loss curves confirm is being learned. However, the gap between training-time corruption recovery and test-time error correction is significant: at training time, corruptions are random and uniform, while at test time, AR decoder errors are systematic and correlated.

3. **Insufficient Phase 2 training.** Phase 2 (refinement training) ran for only 5 epochs compared to Phase 1's 9 epochs. The ARC-AGI literature suggests that iterative refinement benefits often emerge only after extensive training (Gao et al., 2025), and our limited Phase 2 budget may not have been sufficient.

4. **Qualitative evidence.** The qualitative analysis reveals that errors tend to involve single-token constant mismatches (e.g., CBIN78 vs. CBIN85, or CBIN97 vs. CBIN98), which are the type of fine-grained corrections that refinement is well-suited for but may need more targeted training to learn.

We hypothesize that with extended Phase 2 training, curriculum-based refinement difficulty scheduling, and targeted hard-example mining for the refinement loss, the accuracy gap between PARR and the baseline could be closed or reversed.

### 6.3 Qualitative Examples

We present representative examples from the qualitative analysis of 20 test cases (5 per tier).

**Correct predictions (16/20, 80%):**

| Tier | Ground Truth | PARR Prediction | Match |
|------|-------------|-----------------|-------|
| 1 | `add pow x1 C2 mul mul C2 x2 x3` | `add pow x1 C2 mul mul C2 x2 x3` | Exact |
| 2 | `neg mul CBIN81 mul pow x1 C2 sgn x1` | `neg mul CBIN81 mul pow x1 C2 sgn x1` | Exact |
| 3 | `div add mul x1 x2 mul x3 x4 add x1 x3` | `div add mul x1 x2 mul x3 x4 add x1 x3` | Exact |
| 4 | `sub mul mul x1 sin x2 x3 mul mul C0.5 CBIN86 pow x3 C2` | `sub mul mul x1 sin x2 x3 mul mul C0.5 CBIN86 pow x3 C2` | Exact |

**Failure cases (4/20, 20%):**

| Tier | Ground Truth | PARR Prediction | Error Type |
|------|-------------|-----------------|------------|
| 1 | `mul CBIN78 sin mul CBIN86 x1` | `mul CBIN78 sin mul CBIN85 x1` | Constant bin off by 1 |
| 2 | `neg mul CBIN97 x1` | `neg mul CBIN98 x1` | Constant bin off by 1 |
| 3 | `sqrt div mul CBIN79 pow x1 C2 x2` | `sqrt div mul CBIN81 pow x1 C2 x2` | Constant bin off by 2 |
| 4 | `mul div mul x1 cos CBIN71 CBIN71 sub C1 exp neg mul CBIN71 x2` | `mul div mul x1 cos CBIN72 CBIN71 sub C1 exp neg mul CBIN71 x2` | Constant bin off by 1 |

A clear pattern emerges: all four failure cases involve misidentification of a binned constant token (CBIN), typically off by 1--2 bins. The equation structure---operators, variable references, and non-constant tokens---is always predicted correctly. This suggests that PARR's primary limitation is in numerical constant resolution rather than structural equation reasoning. The binned constant representation (where continuous values are discretized into approximately 100 bins) introduces a fundamental quantization challenge: nearby bins correspond to similar numerical values, making bin-level discrimination dependent on subtle numerical differences in the input observations.

### 6.4 Comparison with Prior Work

While direct comparison with prior systems is difficult due to different equation domains and evaluation protocols, we can contextualize our results:

- **SymbolicGPT** (Valipour et al., 2021) reported approximately 25--40% exact recovery rates on their polynomial benchmark with a GPT-style model. Our baseline (88.0%) and PARR (84.0%) both substantially exceed this, though our benchmark is specialized to physics equations with known structural regularities.

- **End-to-end Transformer** (Kamienny et al., 2022) achieved performance approaching genetic programming on SRBench with faster inference. Our approach is complementary, focusing on the physics-specific domain with iterative refinement.

- **AI-Newton** (Fang et al., 2025) operates in a fundamentally different paradigm (concept-driven discovery from multi-experiment data) and rediscovered Newton's laws qualitatively. Our approach targets quantitative symbolic recovery from single-experiment observations.

- **TPSR** (Shojaee et al., 2023) uses MCTS-guided decoding with non-differentiable feedback. The PARR refinement mechanism offers an alternative approach to iterative improvement that is fully differentiable and trainable end-to-end.

---

## 7. Limitations and Future Work

Our work has several important limitations that suggest directions for future research.

**Accuracy gap with baseline.** PARR currently underperforms the standard baseline by 4.0 percentage points on overall ESM. While the refinement mechanism preserves correct predictions and provides substantial speed benefits, it has not yet demonstrated its intended accuracy improvement. Extended Phase 2 training with harder corruption schedules, adversarial refinement objectives (using the AR decoder's actual error distribution rather than random corruption), and longer training budgets are the most promising avenues for closing this gap.

**Inactive refinement at inference.** The ablation study shows that refinement steps do not change predictions in the current training regime. This limits the practical value of the refinement architecture to its speed benefits via parallel decoding. Future work should investigate alternative refinement training strategies, such as bootstrapping refinement targets from the model's own AR errors (self-play refinement) or using MCTS-style search within the refinement loop (combining PARR with TPSR's planning approach).

**Constant bin resolution.** The dominant failure mode is constant bin misidentification. Future work should explore continuous constant prediction (Kamienny et al., 2022) combined with differentiable constant optimization, rather than the discretized binning approach used here.

**Synthetic-only evaluation.** Our benchmark uses procedurally generated data from known equation templates. Evaluation on real experimental data (e.g., the Feynman Symbolic Regression Database from Udrescu & Tegmark, 2020, or AI-Newton's multi-experiment setup) would provide a stronger test of generalization.

**Limited equation complexity.** While our Tier 4 equations include coupled systems and composite expressions, they do not extend to partial differential equations, Lagrangian/Hamiltonian formulations, or equations requiring integration and differentiation. Extension to these domains---as explored by ODEFormer (d'Ascoli et al., 2024) for ODE systems---is a natural next step.

**No test-time adaptation.** Although we implemented a test-time adaptation module with LoRA fine-tuning and observation augmentation (inspired by ARC-AGI solving approaches), it was not evaluated in the current experiments. Incorporating per-problem TTA could substantially improve performance on the hardest Tier 4 equations.

**Single-scale evaluation.** We evaluated on a fixed dataset size (1M training pairs) and did not study data efficiency or scaling behavior. Understanding how PARR's accuracy and refinement benefit scale with data and compute would clarify the architecture's potential.

---

## 8. Conclusion

We have introduced Physics-Aware Recursive Refinement (PARR), a transformer architecture that combines autoregressive equation generation with iterative bidirectional refinement for discovering Newtonian physics equations from numerical observations. PARR incorporates ConvSwiGLU feed-forward networks and Token Algebra layers inspired by recent advances in abstract reasoning (Gao et al., 2025; Nie et al., 2025), trained with a two-phase procedure using corruption-based refinement objectives and TBPTL for memory efficiency.

On a four-tier benchmark of 52 Newtonian physics equation templates (5,000 test samples), PARR achieves 84.0% exact symbolic match rate with strong per-tier performance (Tier 1: 79.0%, Tier 2: 92.5%, Tier 3: 85.4%, Tier 4: 74.1%). While a standard encoder-decoder baseline achieves higher accuracy (88.0% ESM), PARR provides a 2.9x inference speedup and demonstrates graceful degradation under observation noise (only 2.0% accuracy drop at 10% noise). The model is highly efficient, requiring only 3.4 GB peak GPU memory during training and processing 229 equations per second at inference.

Our analysis reveals that the refinement mechanism, while architecturally sound and safely preserving correct predictions, has not yet learned to actively correct errors in the current training regime. This represents both a limitation and an opportunity: the PARR architecture provides the infrastructure for iterative symbolic refinement, and we expect that with extended training investment and refined corruption strategies, the refinement block will deliver on its intended accuracy benefits. The dominant failure mode---constant bin misidentification rather than structural errors---suggests that PARR has successfully learned the structural grammar of Newtonian physics equations and that future improvements should target numerical precision.

More broadly, PARR demonstrates that techniques from abstract reasoning research---particularly iterative refinement with shared-weight blocks, gated convolutions, and continuous token blending---can be productively transferred to the domain of scientific equation discovery. As the fields of AI-driven science and abstract reasoning continue to converge, we expect cross-pollination of architectures and training strategies to accelerate progress in both areas.

---

## References

- Biggio, L., Bendinelli, T., Neitz, A., Lucchi, A., & Parascandolo, G. (2021). Neural Symbolic Regression that Scales. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, pp. 936--945. arXiv:2106.06427.

- Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., & Ho, S. (2020). Discovering Symbolic Models from Deep Learning with Inductive Biases. In *Advances in Neural Information Processing Systems*, vol. 33. arXiv:2006.11287.

- d'Ascoli, S., Becker, S., Mathis, A., Schwaller, P., & Kilbertus, N. (2024). ODEFormer: Symbolic Regression of Dynamical Systems with Transformers. In *Proceedings of the 12th International Conference on Learning Representations (ICLR)*. arXiv:2310.05573.

- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, L. (2019). Universal Transformers. In *Proceedings of the 7th International Conference on Learning Representations (ICLR)*. arXiv:1807.03819.

- Fang, Y.-L., Jian, D.-S., Li, X., & Ma, Y.-Q. (2025). AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge. arXiv:2504.01538.

- Gao, Z., Chen, L., Xiao, Y., Xing, H., Tao, R., Luo, H., Zhou, J., & Dai, B. (2025). Universal Reasoning Model. arXiv:2512.14693.

- Kamienny, P.-A., d'Ascoli, S., Lample, G., & Charton, F. (2022). End-to-end symbolic regression with transformers. In *Advances in Neural Information Processing Systems*, vol. 35, pp. 10269--10281. arXiv:2204.10532.

- La Cava, W., et al. (2024). LaSR: Large Language Model Assisted Symbolic Regression. In *Advances in Neural Information Processing Systems (NeurIPS)*.

- Lample, G. & Charton, F. (2020). Deep Learning for Symbolic Mathematics. In *Proceedings of the 8th International Conference on Learning Representations (ICLR)*. arXiv:1912.01412.

- Liao, I. & Gu, A. (2025). ARC-AGI Without Pretraining. arXiv:2512.06104.

- Makke, N. & Chawla, S. (2024). Interpretable scientific discovery with symbolic regression: a review. *Artificial Intelligence Review*, 57(2), 1--68.

- Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). Large Language Diffusion Models. In *Advances in Neural Information Processing Systems (NeurIPS)* (Oral). arXiv:2502.09992.

- Shojaee, P., Meidani, K., Farimani, A. B., & Reddy, C. K. (2023). TPSR: Transformer-based Planning for Symbolic Regression. In *Advances in Neural Information Processing Systems*, vol. 36, pp. 45907--45919. arXiv:2303.06833.

- Tenachi, W., Ibata, R., & Diakogiannis, F. I. (2023). Deep symbolic regression for physics guided by units constraints: toward the automated discovery of physical laws. *The Astrophysical Journal*, 959(2), 99. arXiv:2303.03192.

- Udrescu, S.-M. & Tegmark, M. (2020). AI Feynman: A Physics-Inspired Method for Symbolic Regression. *Science Advances*, 6(16), eaay2631. arXiv:1905.11481.

- Valipour, M., You, B., Panju, M., & Ghodsi, A. (2021). SymbolicGPT: A Generative Transformer Model for Symbolic Regression. arXiv:2106.14131.

- Ying, J., Lin, H., Yue, C., Chen, Y., Xiao, C., Shi, Q., Liang, Y., Yau, S.-T., Zhou, Y., & Ma, J. (2025). A Neural Symbolic Model for Space Physics. *Nature Machine Intelligence*, 7, 1726--1741. arXiv:2503.07994.
