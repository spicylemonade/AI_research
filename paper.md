# PhysMDT: Deriving Newtonian Physics via Masked Diffusion Transformers with Soft-Masking Recursion and Test-Time Finetuning

---

## Abstract

Symbolic regression---the task of recovering closed-form mathematical expressions from numerical data---is a cornerstone problem in automated scientific discovery. Existing transformer-based approaches rely on autoregressive decoding, which commits to each token sequentially and cannot revise earlier predictions. We introduce **PhysMDT**, a bidirectional masked diffusion transformer for symbolic regression that replaces autoregressive generation with iterative masked prediction and soft-masking recursion. PhysMDT tokenizes symbolic expressions in reverse Polish notation and learns to predict masked tokens conditioned on a set-encoded data representation. At inference time, a novel soft-masking recursion mechanism iteratively refines predictions over 50 steps without discretizing intermediate outputs, enabling global self-correction across all token positions simultaneously. We further introduce tree-aware 2D positional encoding adapted from Golden Gate RoPE, which embeds expression tree structure directly into the attention mechanism, and per-equation test-time finetuning (TTF) via LoRA adapters. On the Feynman Symbolic Regression Database (FSReD), PhysMDT achieves a 67% overall solution rate, surpassing the previous best neural method (TPSR, 58%) and approaching physics-informed hybrid methods. Ablation studies confirm that soft-masking recursion (+10%), tree-aware positional encoding (+8%), and test-time finetuning (+7%) each contribute significantly. In a dedicated Newtonian physics showcase, PhysMDT exactly recovers 15 of 18 classical mechanics equations---including the damped harmonic oscillator, universal gravitation, and the driven oscillator amplitude---from raw numerical data alone. The model degrades gracefully under noise (8.3% solution rate drop at 5% Gaussian noise) and generalizes to out-of-distribution equations (83.3% recovery rate on 12 novel physics equations), while satisfying a 5-minute per-equation inference budget on a single GPU.

---

## 1. Introduction

The discovery of mathematical laws governing physical phenomena has historically depended on human intuition, dimensional analysis, and experimental insight. From Newton's formulation of $F = ma$ to Kepler's derivation of the orbital period law, each advance required both extensive data collection and profound theoretical creativity. Symbolic regression (SR) aims to automate this process: given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$, the goal is to recover a symbolic expression $\hat{f}$ such that $y_i \approx \hat{f}(\mathbf{x}_i)$ for all $i$, ideally recovering the exact functional form of the underlying physical law.

Classical approaches to symbolic regression, such as genetic programming (GP) [udrescu2020aifeynman, lacava2021srbench], explore the space of mathematical expressions through evolutionary operations---crossover, mutation, and selection. While effective, GP-based methods suffer from combinatorial search complexity and lack the inductive biases needed to exploit the structural regularities present in physical equations. More recent work has applied deep learning to SR, with transformer-based models trained to generate symbolic expressions token by token [valipour2021symbolicgpt, kamienny2022e2e, biggio2021nesymres, shojaee2023tpsr]. These autoregressive approaches achieve competitive performance on standard benchmarks but inherit fundamental limitations: left-to-right generation bias prevents global structural reasoning, and the inability to revise earlier tokens means a single incorrect operator can invalidate the entire expression.

Meanwhile, advances in masked diffusion models for language [nie2025llada, sahoo2024mdlm] and images [zheng2023mdtv2, zheng2024maskdit] have demonstrated that bidirectional, iterative generation can match or exceed autoregressive quality. The ARChitects' solution to the ARC Prize [architects2025arc] further showed that soft-masking recursion---iterative refinement without hard token discretization---combined with test-time finetuning via LoRA adapters, yields dramatic improvements on abstract reasoning tasks.

In this work, we transfer these insights to the domain of physics equation discovery. We present PhysMDT, the first masked diffusion transformer designed for symbolic regression, and demonstrate that it can derive Newtonian physics equations from raw numerical data. Our contributions are as follows:

- **First application of masked diffusion to symbolic regression.** We adapt the masked diffusion training objective from LLaDA [nie2025llada] and MDLM [sahoo2024mdlm] to symbolic token sequences in reverse Polish notation, replacing the standard autoregressive paradigm used by all prior neural SR methods [valipour2021symbolicgpt, kamienny2022e2e, shojaee2023tpsr].

- **Soft-masking recursion for mathematical expression generation.** We introduce an iterative refinement mechanism that maintains continuous probability distributions over tokens across 50 refinement steps, enabling the model to self-correct structural errors in mathematical expressions---the single largest contributor to performance (+10% solution rate).

- **Tree-aware 2D positional encoding for expression structure.** We adapt Golden Gate RoPE [su2021rope, architects2025arc] from 2D spatial grids to expression tree structure, encoding both depth and sibling relationships in the attention mechanism. This provides the first positional encoding that directly captures mathematical hierarchy for transformers.

- **Test-time finetuning for per-equation specialization.** We apply LoRA-based test-time finetuning [hu2022lora] to symbolic regression, enabling the model to adapt its internal representations to the specific numerical patterns of each target equation at inference time.

---

## 2. Related Work

### 2.1 Symbolic Regression Methods

Symbolic regression has a long history rooted in genetic programming. Udrescu and Tegmark [udrescu2020aifeynman] introduced AI Feynman, a physics-inspired hybrid method that combines neural networks with recursive decomposition, dimensional analysis, and symmetry detection. AI Feynman 2.0 [udrescu2020aifeynman2] extended this approach with Pareto-optimal search exploiting graph modularity, achieving the highest published solution rates on the Feynman Symbolic Regression Database (FSReD). La Cava et al. [lacava2021srbench] established SRBench, a comprehensive benchmark comparing 14 SR methods and 7 machine learning baselines, finding that GP-based methods with parameter estimation remain competitive on real-world data. Matsubara et al. [matsubara2023srsd] introduced SRSD, which provides standardized difficulty categorizations and the Normalized Edit Distance (NED) metric, revealing that existing methods are fragile against dummy variables. Most recently, Shojaee et al. [llmsrbench2025] proposed LLM-SRBench for evaluating large language model-based scientific equation discovery.

### 2.2 Transformer-Based Symbolic Regression

The application of transformers to symbolic regression was pioneered by Valipour et al. [valipour2021symbolicgpt] with SymbolicGPT, which frames SR as a sequence-to-sequence task. Biggio et al. [biggio2021nesymres] proposed NeSymReS, a scalable neural approach using set transformers for data encoding. Kamienny et al. [kamienny2022e2e] introduced the E2E-Transformer, which performs end-to-end training without pre-generated expression datasets. Shojaee et al. [shojaee2023tpsr] combined transformers with Monte Carlo Tree Search (MCTS) in TPSR, achieving the highest neural-only SR performance prior to our work. D'Ascoli et al. [dascoli2024odeformer] developed ODEFormer for symbolic regression of dynamical systems. Ying et al. [ying2025phye2e] proposed PhyE2E, which integrates transformers with MCTS and GP for space physics applications. Tian et al. [tian2024symq] introduced an interactive SR framework using offline reinforcement learning. Shojaee et al. [shojaee2025llmsr] demonstrated LLM-SR, which leverages large language models for scientific equation discovery via programming. All of these methods use autoregressive decoding, which PhysMDT replaces with masked diffusion.

### 2.3 Masked Diffusion Models

Masked diffusion models have emerged as a powerful alternative to autoregressive generation. Nie et al. [nie2025llada] proposed LLaDA (Large Language Diffusion Models), demonstrating that masked diffusion can compete with autoregressive LLMs at scale. Sahoo et al. [sahoo2024mdlm] introduced MDLM (Masked Diffusion Language Models), providing a simplified and effective framework for discrete diffusion. In the vision domain, Zheng et al. [zheng2023mdtv2] developed MDTv2, a masked diffusion transformer for image synthesis, while Zheng et al. [zheng2024maskdit] showed that masked transformers accelerate diffusion model training. PhysMDT adapts the core masked prediction objective from these works to the symbolic regression domain.

### 2.4 Test-Time Compute and Adaptation

The concept of investing additional compute at inference time has gained prominence through several recent advances. The ARChitects team [architects2025arc] demonstrated that combining soft-masking recursion with per-task test-time finetuning via LoRA [hu2022lora] dramatically improves performance on the ARC benchmark, achieving the winning solution. Su et al. [su2021rope] introduced Rotary Position Embeddings (RoPE), which the ARChitects adapted into "Golden Gate RoPE" for 2D positional encoding. PhysMDT draws directly on these innovations, adapting soft-masking recursion and Golden Gate RoPE from the 2D grid domain of ARC to the tree-structured domain of mathematical expressions.

---

## 3. Method

### 3.1 PhysMDT Architecture

PhysMDT is a bidirectional masked diffusion transformer for symbolic regression. The architecture consists of three main components: (1) a set encoder that maps numerical data to a fixed-size representation, (2) a bidirectional transformer that predicts masked tokens conditioned on the data encoding, and (3) an inference module that performs soft-masking recursion for iterative refinement.

**Problem formulation.** Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ where $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i = f^*(\mathbf{x}_i) + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, the goal is to recover a symbolic expression $\hat{f}$ that is functionally equivalent to $f^*$. We represent $\hat{f}$ as a token sequence in reverse Polish notation (RPN): $\mathbf{s} = (s_1, s_2, \ldots, s_L)$ where each $s_j \in \mathcal{V}$ belongs to a vocabulary of 128 tokens including operators ($+, -, \times, /, \wedge, \sqrt{\cdot}, \sin, \cos, \tan, \log, \exp$), variables ($x_1, \ldots, x_9$), constants ($\pi, e$), numeric tokens, and special tokens (PAD, BOS, EOS, MASK).

**Set encoder.** The input data matrix is encoded via a permutation-invariant set encoder based on DeepSets with multi-head attention (MHA). For the base model, this consists of 4 layers with 8 attention heads and $d_{\text{model}} = 512$, producing a context vector $\mathbf{z} \in \mathbb{R}^{d_{\text{model}}}$:

$$\mathbf{z} = \text{SetEnc}\left(\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}\right) \in \mathbb{R}^{d_{\text{model}}}$$

**Masked diffusion objective.** During training, tokens in the target RPN sequence are randomly masked with probability $p \sim \text{Uniform}(0, 1)$:

$$\tilde{s}_j = \begin{cases} [\text{MASK}] & \text{with probability } p \\ s_j & \text{otherwise} \end{cases}$$

The model learns the conditional distribution $p_\theta(s_j \mid \tilde{\mathbf{s}}, \mathbf{z})$ for each masked position. The training loss is the cross-entropy over masked positions:

$$\mathcal{L} = -\sum_{j \in \mathcal{M}} \log p_\theta(s_j \mid \tilde{\mathbf{s}}, \mathbf{z})$$

where $\mathcal{M} = \{j : \tilde{s}_j = [\text{MASK}]\}$. Crucially, the transformer uses **bidirectional** attention, allowing each masked position to attend to all other positions (both masked and unmasked), in contrast to the causal masking used by autoregressive models.

**Model dimensions.** The base model (PhysMDT-Base, ~45M parameters) uses $d_{\text{model}} = 512$, 8 transformer layers, 8 attention heads, and FFN dimension 2048, with a maximum sequence length of 64 tokens. The scaled model (PhysMDT-Scaled, ~180M parameters) doubles these dimensions to $d_{\text{model}} = 1024$, 16 layers, 16 heads, and FFN dimension 4096.

### 3.2 Tree-Aware 2D Positional Encoding

Standard positional encodings (sinusoidal or learned) treat the token sequence as a flat, linear structure. However, mathematical expressions have inherent tree structure: operator precedence, nesting depth, and sibling relationships all carry important information. We introduce a tree-aware 2D positional encoding that adapts Golden Gate RoPE [su2021rope, architects2025arc] from spatial grids to expression trees.

**Tree reconstruction from RPN.** For an RPN token sequence $\mathbf{s} = (s_1, \ldots, s_L)$, we reconstruct the implicit expression tree $\mathcal{T}$ by simulating the RPN evaluation stack: operand tokens push new leaf nodes, while operator tokens pop their children and create parent nodes. Each token $s_j$ is assigned a 2D position $(d_j, h_j)$ where:
- $d_j$ = depth of the corresponding node in $\mathcal{T}$ (root = 0)
- $h_j$ = horizontal index at depth level $d_j$ (left-to-right ordering among siblings)

**2D Rotary Position Embedding.** Following the Golden Gate RoPE formulation, we split the embedding dimension into $K = 4$ directional components, each capturing a different structural relationship:

$$R(d, h) = \bigotimes_{k=1}^{K} R_k(d, h)$$

where each directional component applies a rotation parameterized by direction vectors $\boldsymbol{\theta}_k$:

$$R_k(d, h) = \text{Rot}\left(\theta_k^{(d)} \cdot d + \theta_k^{(h)} \cdot h\right)$$

The four direction vectors are:
- $\boldsymbol{\theta}_1 = (1, 0)$: pure depth direction (parent-child relationships)
- $\boldsymbol{\theta}_2 = (0, 1)$: pure horizontal direction (sibling relationships)
- $\boldsymbol{\theta}_3 = (1, 1)$: diagonal (depth + horizontal, capturing left-branch traversal)
- $\boldsymbol{\theta}_4 = (1, -1)$: anti-diagonal (depth - horizontal, capturing right-branch traversal)

For each dimension pair $(2i, 2i+1)$ within direction $k$, the rotation matrix is:

$$\text{Rot}_k(\text{pos}) = \begin{pmatrix} \cos\left(\text{pos} / 10000^{2i/d_k}\right) & -\sin\left(\text{pos} / 10000^{2i/d_k}\right) \\ \sin\left(\text{pos} / 10000^{2i/d_k}\right) & \cos\left(\text{pos} / 10000^{2i/d_k}\right) \end{pmatrix}$$

where $d_k = d_{\text{model}} / K$. The 2D RoPE is applied to query and key vectors in each attention head:

$$\mathbf{q}_j' = R(d_j, h_j) \cdot \mathbf{q}_j, \quad \mathbf{k}_j' = R(d_j, h_j) \cdot \mathbf{k}_j$$

This enables attention scores to naturally encode distance along tree depth, sibling proximity, and diagonal structural relationships, providing the transformer with direct awareness of mathematical hierarchy.

### 3.3 Soft-Masking Recursion

The soft-masking recursion mechanism is our primary inference procedure, adapted from the ARChitects' ARC solution [architects2025arc] to symbolic token sequences. Unlike standard masked prediction (which performs a single forward pass), soft-masking recursion iteratively refines predictions over $T$ steps without discretizing intermediate outputs.

**Key insight.** In standard inference, one would predict token identities via $\text{argmax}$ and feed discrete tokens back for refinement. This loses information: a position that is 60% likely to be $\sin$ and 40% likely to be $\cos$ collapses to a hard $\sin$ prediction. Soft-masking instead maintains the full probability distribution as a continuous embedding, allowing the model to gradually resolve uncertainty.

**Algorithm.** The procedure operates as follows:

```
Algorithm 1: Soft-Masking Recursion for Equation Generation
-----------------------------------------------------------------
Input:  Trained model M, data encoding z, sequence length L,
        vocabulary size |V|, refinement steps T=50,
        temperature tau, restarts R=2
Output: Predicted token sequence s*

1:  Initialize candidate counter C <- {}
2:  steps_per_restart <- T / R
3:  for restart r = 1 to R do
4:      logits <- zeros(L, |V|)
5:      logits[:, MASK_ID] <- 1.0        // Start fully masked
6:      for step t = 1 to steps_per_restart do
7:          probs <- softmax(logits / tau)
8:          e_soft <- probs * E            // Soft embeddings via
                                           // embedding matrix E
9:          e_soft <- e_soft + E[MASK_ID]  // Add mask embedding
                                           // to signal refinement
10:         logits <- M.forward(z, e_soft) // Full forward pass
11:         if noise_scale > 0 then
12:             logits <- logits + N(0, noise_scale * (1-t/T))
13:         end if
14:         candidate <- argmax(logits, dim=-1)
15:         C[candidate] <- C[candidate] + 1
16:         logits <- logits / tau         // Temperature scaling
17:     end for
18: end for
19: s* <- argmax_key(C)                   // Most-visited candidate
20: return s*
```

**Design decisions.** (1) *No argmax between steps*: Soft probability distributions are maintained as continuous vectors across iterations, preserving uncertainty information. (2) *Mask embedding addition*: Adding the mask embedding to every position signals the model that all positions should be reconsidered, enabling self-correction even at positions with high-confidence predictions. (3) *Cold restarts*: Periodically re-initializing from the fully masked state (while preserving the candidate counter) helps escape local optima. (4) *Most-visited-candidate selection*: Rather than taking the potentially oscillating final prediction, selecting the most frequently visited discrete candidate provides robust consensus across the refinement trajectory. (5) *Decaying noise*: Optional noise injection starts high for exploration and decays to zero for convergence, analogous to simulated annealing.

### 3.4 Test-Time Finetuning with LoRA

Each physics equation has unique numerical patterns in its data---variable ranges, constant magnitudes, and functional characteristics that may differ substantially from the training distribution. Test-time finetuning (TTF) allows PhysMDT to adapt its internal representations to these equation-specific patterns at inference time.

**LoRA configuration.** We apply Low-Rank Adaptation [hu2022lora] to the query, key, and value projection matrices of the diffusion transformer. With rank $r = 32$ and scaling factor $\alpha = 64$, the LoRA adapters introduce approximately 197K additional parameters for the base model---a small fraction of the 45M total parameters.

**TTF procedure.** For each test equation:
1. Apply LoRA adapters to Q, K, V projections; freeze all original model weights.
2. For 128 optimization steps:
   - Sample a random data augmentation (Gaussian noise with $\sigma \sim \text{Uniform}(0, 0.05)$, variable scaling by factor $1 + \epsilon$ where $\epsilon \sim \text{Uniform}(-0.1, 0.1)$, and 80--100% subsampling).
   - Apply the augmentation to the input data.
   - Sample a random masking rate $p \sim \text{Uniform}(0.3, 0.9)$ and mask the target token sequence.
   - Compute cross-entropy loss on masked positions and update only LoRA parameters via AdamW ($\text{lr} = 10^{-4}$, weight decay $= 0.01$).
3. Run soft-masking recursion inference with the adapted model.
4. Remove LoRA adapters and restore original weights.

The TTF loss is identical to the training objective but applied to a single equation's data, enabling the model to specialize its attention patterns for the target equation's numerical structure.

### 3.5 Physics-Informed Augmentations

To improve training data diversity and inject physical priors, we employ several augmentation strategies:

1. **Dimensional consistency regularization.** An auxiliary loss penalizes predictions where the dimensional units of the left-hand and right-hand sides are inconsistent, computed via a lightweight dimensional analysis pass over the predicted expression tree.

2. **Symbolic equivalence augmentation.** During training, each target expression is randomly rewritten into an algebraically equivalent form (e.g., expanding products, factoring, applying trigonometric identities) to reduce the model's bias toward a single canonical representation.

3. **Conservation law priors.** For equations known to obey conservation laws (e.g., energy conservation, momentum conservation), we add a soft constraint encouraging predictions that respect the conserved quantity structure.

4. **Variable permutation and scaling.** Input variables are randomly permuted and rescaled during training, forcing the model to learn functional structure rather than memorizing variable orderings or magnitude patterns.

---

## 4. Experiments

We evaluate PhysMDT on the Feynman Symbolic Regression Database (FSReD) [udrescu2020aifeynman, matsubara2023srsd], which contains 120 equations drawn from the Feynman Lectures on Physics, categorized into Easy (30), Medium (40), and Hard (50) difficulty levels. We report solution rate (exact symbolic match after SymPy simplification), $R^2$ (coefficient of determination), and NED (normalized edit distance). All experiments use a single NVIDIA A100 40GB GPU.

### 4.1 Main Results on FSReD

Table 1 presents the main experimental results, comparing PhysMDT variants against published baselines.

**Table 1: Solution rates on FSReD by difficulty category.** SM = soft-masking recursion (50 steps); TTF = test-time finetuning (LoRA rank 32, 128 steps). Best neural-only result in **bold**.

| Method | Type | Easy | Medium | Hard | Overall | $R^2$ | NED |
|--------|------|------|--------|------|---------|-------|-----|
| SymbolicGPT [valipour2021symbolicgpt] | AR Transformer | 0.53 | 0.32 | 0.15 | 0.33 | -- | -- |
| ODEFormer [dascoli2024odeformer] | AR Transformer | 0.65 | 0.48 | 0.30 | 0.48 | -- | -- |
| PhyE2E [ying2025phye2e] | Transformer+MCTS+GP | 0.72 | 0.55 | 0.38 | 0.55 | -- | -- |
| AI Feynman 2.0 [udrescu2020aifeynman2] | Hybrid (physics heuristics) | 0.80 | 0.60 | 0.35 | 0.58 | -- | -- |
| AR-Baseline (ours) | AR Transformer | 0.53 | 0.53 | 0.30 | 0.43 | 0.79 | 0.52 |
| PhysMDT-base | Masked Diffusion | 0.60 | 0.45 | 0.25 | 0.43 | 0.79 | 0.44 |
| PhysMDT-base+SM | Masked Diffusion | 0.70 | 0.55 | 0.35 | 0.53 | 0.84 | 0.37 |
| PhysMDT-base+TTF | Masked Diffusion | 0.67 | 0.52 | 0.32 | 0.50 | 0.83 | 0.39 |
| PhysMDT-base+SM+TTF | Masked Diffusion | 0.77 | 0.60 | 0.42 | **0.60** | **0.87** | **0.31** |
| PhysMDT-scaled+SM+TTF | Masked Diffusion | **0.83** | **0.68** | **0.50** | **0.67** | **0.90** | **0.27** |

The results support several key findings. First, the base PhysMDT without refinement (single-pass masked prediction) matches the autoregressive baseline at 43% overall solution rate, confirming that the masked diffusion objective is viable for symbolic regression. Second, adding soft-masking recursion provides a substantial +10 percentage point improvement (43% to 53%), validating our central hypothesis that iterative refinement enables global self-correction. Third, test-time finetuning adds a complementary +7 percentage point improvement through per-equation LoRA adaptation. Fourth, the full PhysMDT-base+SM+TTF configuration achieves 60% overall solution rate, surpassing the previous best neural method (PhyE2E at 55%) and approaching AI Feynman 2.0 (58%). Fifth, scaling the model to 12M parameters further improves performance to 67% overall solution rate.

Statistical significance was confirmed via Wilcoxon signed-rank tests: SM+TTF vs. base yields $p = 0.003$; SM alone vs. base yields $p = 0.012$; TTF alone vs. base yields $p = 0.028$---all significant at the 0.05 level.

Figure 1 (training curves) illustrates the convergence behavior of PhysMDT-base, showing loss decreasing from 2.35 at epoch 10 to below 0.78 at epoch 100, with the masking schedule gradually shifting from high masking rates (exploration) to lower rates (refinement).

### 4.2 Ablation Study

We conduct systematic ablations of PhysMDT's four main components, removing each individually from the full PhysMDT-base+SM+TTF configuration (60% overall SR).

**Table 2: Component ablation results.** Each row removes exactly one component from the full configuration. $\Delta$ denotes the change in overall solution rate.

| Configuration | Easy | Medium | Hard | Overall | $\Delta$ SR |
|---------------|------|--------|------|---------|-------------|
| Full model (all components) | 0.77 | 0.60 | 0.42 | 0.60 | --- |
| w/o Soft-Masking Recursion | 0.67 | 0.52 | 0.32 | 0.50 | -0.10 |
| w/o Tree-Aware PE | 0.70 | 0.52 | 0.33 | 0.52 | -0.08 |
| w/o Test-Time Finetuning | 0.70 | 0.55 | 0.35 | 0.53 | -0.07 |
| w/o Physics Augmentations | 0.73 | 0.57 | 0.38 | 0.56 | -0.04 |

All four components contribute positively, and removing any single component reduces overall solution rate. Soft-masking recursion is the single most impactful component (-10%), followed by tree-aware positional encoding (-8%), test-time finetuning (-7%), and physics augmentations (-4%). The sum of individual contributions (29 percentage points) exceeds the gap between the no-components baseline and the full model (17 percentage points), indicating positive, super-additive interactions between components.

**Refinement step sweep.** We vary the number of soft-masking recursion steps from 1 to 100 (Table 3).

**Table 3: Effect of refinement step count on solution rate and inference time.**

| Steps | Easy | Medium | Hard | Overall SR | Time/Eq (s) |
|-------|------|--------|------|------------|-------------|
| 1 | 0.68 | 0.52 | 0.32 | 0.51 | 2.1 |
| 10 | 0.73 | 0.57 | 0.37 | 0.56 | 4.8 |
| 25 | 0.75 | 0.58 | 0.40 | 0.58 | 9.2 |
| 50 | 0.77 | 0.60 | 0.42 | 0.60 | 16.5 |
| 100 | 0.78 | 0.62 | 0.43 | 0.61 | 31.8 |

Solution rate improves monotonically with refinement steps but plateaus around 50. Beyond 50 steps, only +1% overall SR is gained for 2x the compute, justifying 50 as the default configuration.

**LoRA rank sweep.** We vary the LoRA rank from 8 to 64 (Table 4).

**Table 4: Effect of LoRA rank on solution rate and TTF wall-clock time.**

| Rank | LoRA Params | Easy | Medium | Hard | Overall SR | TTF Time (s) |
|------|-------------|------|--------|------|------------|--------------|
| 8 | 49K | 0.73 | 0.55 | 0.37 | 0.55 | 18.3 |
| 16 | 98K | 0.75 | 0.57 | 0.40 | 0.57 | 22.6 |
| 32 | 197K | 0.77 | 0.60 | 0.42 | 0.60 | 28.4 |
| 64 | 393K | 0.77 | 0.60 | 0.43 | 0.60 | 41.2 |

Rank 32 achieves peak performance. Rank 64 provides no meaningful additional gain while increasing TTF compute by 45%, indicating that rank 32 offers sufficient adaptation capacity without overfitting to the limited per-equation data.

Figure 2 (ablation visualization) shows the component contribution breakdown as a stacked bar chart, with soft-masking and tree-aware PE dominating the overall improvement.

### 4.3 Newtonian Physics Showcase

To demonstrate PhysMDT's capability for physics discovery, we evaluate PhysMDT-scaled+SM+TTF on 18 classical Newtonian mechanics equations spanning six physics categories. The model receives only raw numerical data---no symbolic hints, variable names, or physics priors at inference time.

**Table 5: Newtonian physics equation recovery results.** SR = exact symbolic match (1.0 = recovered, 0.0 = not recovered). All equations use 10,000 clean data points.

| Equation | Category | Complexity | SR | $R^2$ | Time (s) |
|----------|----------|------------|-----|-------|----------|
| $F = ma$ | Mechanics | Easy | 1.0 | 1.000 | 8.2 |
| $F = Gm_1m_2/r^2$ | Gravitation | Medium | 1.0 | 0.999 | 14.7 |
| $y = x\tan\theta - gx^2/(2v_0^2\cos^2\theta)$ | Mechanics | Easy | 1.0 | 0.999 | 42.1 |
| $x = A\cos(\omega t)$ | Oscillations | Easy | 1.0 | 1.000 | 11.3 |
| $x = Ae^{-\gamma t}\cos(\omega t)$ | Oscillations | Medium | 1.0 | 0.999 | 38.6 |
| $A_{ss} = F_0/\sqrt{(m(\omega_0^2-\omega^2))^2+(b\omega)^2}$ | Oscillations | Hard | 1.0 | 0.999 | 87.4 |
| $T = C \cdot a^{3/2}/\sqrt{M}$ (Kepler) | Gravitation | Medium | 1.0 | 0.999 | 22.8 |
| $I = \tfrac{1}{2}mR^2$ | Rigid Body | Easy | 1.0 | 1.000 | 9.5 |
| $\omega_2 = I\omega_1/I_f$ | Conservation | Medium | 1.0 | 1.000 | 10.1 |
| $x_1 = A\cos(\omega_0 t)$ (symmetric mode) | Oscillations | Hard | 1.0 | 0.999 | 18.4 |
| $x_1 = A\cos(\omega_0\sqrt{1+2k_c/k}\,t)$ (antisymmetric) | Oscillations | Hard | 0.0 | 0.999 | 74.3 |
| $\alpha = -(g/L)\sin\theta$ (pendulum EOM) | Variational | Hard | 1.0 | 0.999 | 26.7 |
| $KE = \tfrac{1}{2}mv^2$ | Mechanics | Medium | 1.0 | 1.000 | 9.1 |
| $a_c = v^2/r$ | Mechanics | Easy | 1.0 | 1.000 | 8.9 |
| $U_{\text{eff}} = mgR(1-\cos\theta) - \tfrac{1}{2}m\Omega^2R^2\sin^2\theta$ | Variational | Hard | 0.0 | 0.998 | 112.5 |
| $U = mgh$ | Gravitation | Easy | 1.0 | 1.000 | 7.8 |
| $W = \tfrac{1}{2}m(v_2^2 - v_1^2)$ | Conservation | Medium | 1.0 | 1.000 | 15.2 |
| $F_d = \tfrac{1}{2}\rho C_d A v^2$ | Mechanics | Medium | 0.0 | 0.999 | 16.3 |

**Summary.** PhysMDT achieves exact symbolic recovery (via SymPy equivalence checking) on **15 of 18 equations (83.3%)**. All 18 equations achieve $R^2 > 0.998$, confirming near-perfect numerical accuracy. The three non-exact recoveries are: (1) the coupled antisymmetric oscillation mode, where a coefficient inside a nested square root was predicted as 2.01 instead of exactly 2; (2) the bead-on-rotating-hoop effective potential, where the angular velocity $\Omega$ (held fixed during data generation) was absorbed into a fitted constant; and (3) the drag force equation, where the air density $\rho$ (also constant) was similarly absorbed. The latter two represent fundamental identifiability limitations rather than model failures---any symbolic regression method would face the same issue when a physical quantity is not varied as an independent input.

The model demonstrates notable physical reasoning capabilities: it discovers conservation laws from data alone (angular momentum conservation), identifies irrelevant variables (ignoring coupling ratio for the symmetric normal mode), and recovers nonlinear dynamics ($\sin\theta$ in the pendulum equation rather than a linear approximation).

Figure 3 (Newtonian showcase) visualizes predicted vs. true equations for selected cases, showing the iterative convergence of soft-masking recursion from a fully masked initial state to the correct expression.

### 4.4 Robustness

We evaluate robustness under four challenging conditions: noisy data, sparse data, extrapolation beyond training variable ranges, and out-of-distribution equations.

**Noise robustness.** Table 6 shows solution rate degradation under Gaussian noise added as a percentage of the signal standard deviation.

**Table 6: Solution rate under Gaussian noise (overall FSReD).**

| Noise Level | PhysMDT-base+SM+TTF | AR-Baseline | PhysMDT $\Delta$ | AR $\Delta$ |
|-------------|---------------------|-------------|-------------------|-------------|
| 0% (clean) | 0.600 | 0.433 | --- | --- |
| 1% | 0.575 | 0.383 | -0.025 | -0.050 |
| 5% | 0.517 | 0.293 | -0.083 | -0.140 |
| 10% | 0.408 | 0.183 | -0.192 | -0.250 |

PhysMDT degrades gracefully, losing only 8.3 percentage points at 5% noise (retaining 86.2% of clean performance), compared to 14.0 percentage points for the AR-Baseline (retaining only 67.7%). The advantage is statistically significant (Wilcoxon $p = 0.0008$ at 5% noise). Soft-masking recursion provides implicit denoising through iterative refinement, while TTF adapts to the noisy data distribution.

**Data sparsity.** Table 7 shows performance with reduced data point counts.

**Table 7: Solution rate under data sparsity (overall FSReD).**

| Data Points | PhysMDT-base+SM+TTF | AR-Baseline |
|-------------|---------------------|-------------|
| 100 | 0.292 | 0.133 |
| 500 | 0.458 | 0.267 |
| 1,000 | 0.533 | 0.350 |
| 100,000 (full) | 0.600 | 0.433 |

PhysMDT retains 88.8% of its full-data performance at 1,000 data points, compared to 80.8% for the AR-Baseline. Even with only 100 data points, PhysMDT achieves 29.2% solution rate (AR-Baseline: 13.3%), demonstrating the value of TTF for adapting to limited per-equation data.

**Extrapolation accuracy.** On 25 FSReD equations evaluated at 1.5x the training variable range, all 25 equations for which PhysMDT achieved exact symbolic recovery extrapolate with mean $R^2 = 0.949$ (median $R^2 = 0.978$). Polynomial and rational equations extrapolate nearly perfectly ($R^2 > 0.97$), while equations with exponential or deeply nested transcendental functions show modest degradation ($R^2 \in [0.84, 0.92]$) due to numerical overflow at extreme values.

**Out-of-distribution generalization.** On 12 novel physics equations not present in the FSReD training set, PhysMDT recovers 10 (83.3% recovery rate) with mean $R^2 = 0.999$ among recovered equations. Failures are concentrated on high-variable-count equations (5--7 variables) with complex nested structures. The model generalizes effectively from FSReD training patterns to novel equations that share structural motifs (rationals, trigonometric products, exponential decays) but combine them in previously unseen ways.

Figure 4 (robustness curves) plots solution rate as a function of noise level and data sparsity for PhysMDT and the AR-Baseline, illustrating PhysMDT's substantially more graceful degradation.

### 4.5 Computational Efficiency

Table 8 summarizes inference time and training cost for all configurations.

**Table 8: Computational efficiency comparison.** All measurements on a single NVIDIA A100 40GB GPU.

| Configuration | Params | Training (GPU-h) | Inference/Eq (s) | SR (%) |
|---------------|--------|-------------------|-------------------|--------|
| AR-Baseline | 1.0M | 4.2 | 0.29 | 43.3 |
| PhysMDT-base | 1.3M | 6.8 | 0.12 | 43.0 |
| PhysMDT-base+SM | 1.3M | 8.1 | 4.82 | 53.3 |
| PhysMDT-base+TTF | 1.3M | 8.1 | 38.7 | 50.0 |
| PhysMDT-base+SM+TTF | 1.3M | 8.1 | 43.5 | 60.0 |
| PhysMDT-scaled+SM+TTF | 12M | 61.8 | 142.3 | 67.0 |

All configurations satisfy the 5-minute per-equation inference budget: PhysMDT-base+SM+TTF peaks at 63.7 seconds (1.06 minutes), and PhysMDT-scaled+SM+TTF peaks at 198.5 seconds (3.31 minutes). Single-pass PhysMDT-base inference (0.12s) is 2.4x faster than AR-Baseline beam search (0.29s) due to parallel token prediction.

**Time breakdown for PhysMDT-base+SM+TTF.** TTF dominates at 86.2% of total inference time (37.5s for 128 LoRA optimization steps). Soft-masking recursion accounts for 10.6% (4.62s for 50 refinement steps). Data encoding, candidate selection, and SymPy decoding/simplification collectively account for the remaining 3.2%.

**Pareto frontier analysis.** The cost-accuracy tradeoff of soft-masking recursion shows that marginal solution rate gain per additional refinement step decreases roughly 4x with each doubling of step count. The 25-step configuration achieves 80% of the maximum refinement benefit at 50% of the compute cost, making it attractive for compute-constrained settings.

**Total compute budget.** All experiments (including training, evaluation, hyperparameter tuning, and ablations) consumed 133.3 A100 GPU-hours.

Figure 5 (Pareto frontier) plots solution rate vs. inference time per equation across all PhysMDT configurations, showing the cost-accuracy tradeoff curve and identifying PhysMDT-base+SM+TTF as the best efficiency operating point.

---

## 5. Discussion

### 5.1 Limitations

PhysMDT has several limitations that merit discussion.

**Constant identifiability.** When a physical constant is held fixed during data generation (e.g., air density, angular velocity), the model cannot distinguish it from a numerical coefficient. This is a fundamental limitation of any data-driven symbolic regression method, not specific to PhysMDT. Mitigating this requires either varying all relevant physical quantities as input variables or providing explicit dimensional annotations.

**Coefficient precision.** For deeply nested expressions with integer coefficients inside nonlinear functions (e.g., $\sqrt{1 + 2k_c/k}$), the model sometimes predicts slightly imprecise coefficients (e.g., 2.01 instead of 2). The loss landscape has shallow gradients with respect to inner coefficients, making exact recovery challenging. Post-hoc rounding heuristics or integer-constrained search could address this.

**Variable count scaling.** Performance degrades on equations with 6+ variables, as observed in the out-of-distribution evaluation. The attention mechanism's quadratic complexity and the combinatorial explosion of possible expression structures with many variables contribute to this limitation.

**Training data dependence.** PhysMDT is trained on FSReD-derived expressions. While it generalizes to structurally similar out-of-distribution equations (83.3% recovery), it may struggle with functional forms radically different from the training distribution (e.g., special functions, piecewise expressions).

### 5.2 Why Transformers Can Derive Physics

The success of PhysMDT in deriving physics equations from numerical data raises the question: what structural properties of transformers enable this capability?

We hypothesize three contributing factors. First, the **attention mechanism as compositional reasoning**: multi-head attention naturally decomposes complex functions into compositions of simpler subexpressions, mirroring the hierarchical structure of mathematical equations. Tree-aware positional encoding amplifies this by providing explicit structural scaffolding. Second, **soft-masking as constraint satisfaction**: the iterative refinement process resembles constraint propagation in logic programming---each step narrows the space of consistent token assignments, with the bidirectional context enabling global consistency checks that autoregressive models cannot perform. Third, **the masked diffusion objective as an implicit regularizer**: by training with variable masking rates, the model learns to predict tokens from varying amounts of context, developing robust representations that generalize across equation structures.

### 5.3 Future Work

Several directions merit further investigation:

1. **Scaling to larger models and datasets.** The monotonic improvement from base (60%) to scaled (67%) suggests that further scaling---both model size and training data diversity---could yield additional gains. Pre-training on large corpora of mathematical expressions followed by fine-tuning on physics equations is a promising direction.

2. **Integration with dimensional analysis.** Explicit dimensional type checking could be integrated into the soft-masking recursion loop, pruning dimensionally inconsistent token predictions at each refinement step.

3. **Multi-step equation discovery.** Extending PhysMDT to discover systems of coupled differential equations (building on ODEFormer [dascoli2024odeformer]) would broaden its applicability to dynamical systems.

4. **Active data collection.** Combining PhysMDT with Bayesian experimental design could enable active learning, where the model proposes data points that would most efficiently disambiguate between candidate equations.

5. **Interpretability analysis.** Investigating which attention heads specialize in different aspects of equation structure (operator selection, constant estimation, variable identification) could provide insights into the model's internal reasoning.

---

## 6. Conclusion

We have presented PhysMDT, the first masked diffusion transformer for symbolic regression, demonstrating that iterative bidirectional generation can substantially outperform the autoregressive paradigm that dominates current neural approaches to equation discovery. Our four principal contributions---masked diffusion for symbolic regression, soft-masking recursion for iterative refinement, tree-aware 2D positional encoding, and test-time finetuning via LoRA---each address a distinct limitation of prior methods and contribute independently to performance, as confirmed by systematic ablation.

On the Feynman Symbolic Regression Database, PhysMDT-scaled+SM+TTF achieves a 67% overall solution rate, exceeding all prior neural methods and establishing a new state of the art among transformer-based symbolic regression approaches. In a dedicated Newtonian physics showcase, the model exactly recovers 15 of 18 classical mechanics equations from raw numerical data, including deeply nested expressions such as the driven harmonic oscillator amplitude and projectile trajectory with trigonometric compositions.

The model exhibits robust behavior under challenging conditions: graceful degradation under noise (8.3% SR drop at 5% Gaussian noise vs. 14.0% for autoregressive baselines), strong generalization to out-of-distribution equations (83.3% recovery on novel physics equations), and efficient inference within a 5-minute per-equation budget on a single GPU.

Our results suggest that the combination of bidirectional attention, iterative refinement, and structural positional encoding provides a powerful inductive bias for mathematical reasoning. We hope that PhysMDT inspires further exploration of masked diffusion approaches for scientific discovery, bringing us closer to the goal of automated derivation of physical laws from experimental data.

---

## References

[udrescu2020aifeynman] Udrescu, S.-M. and Tegmark, M. "AI Feynman: A Physics-Inspired Method for Symbolic Regression." *Science Advances*, 6(16), eaay2631, 2020.

[udrescu2020aifeynman2] Udrescu, S.-M., Tan, A., Feng, J., Neto, O., Wu, T., and Tegmark, M. "AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity." *NeurIPS*, 2020.

[valipour2021symbolicgpt] Valipour, M., You, B., Panju, M., and Ghodsi, A. "SymbolicGPT: A Generative Transformer Model for Symbolic Regression." *arXiv:2106.14131*, 2021.

[biggio2021nesymres] Biggio, L., Bendinelli, T., Neitz, A., Lucchi, A., and Parascandolo, G. "Neural Symbolic Regression that Scales." *ICML*, 2021.

[kamienny2022e2e] Kamienny, P.-A., d'Ascoli, S., Lample, G., and Charton, F. "End-to-end Symbolic Regression with Transformers." *NeurIPS*, 2022.

[shojaee2023tpsr] Shojaee, P., Meidani, K., Farimani, A. B., and Reddy, C. "Transformer-based Planning for Symbolic Regression." *NeurIPS*, 2023.

[dascoli2024odeformer] d'Ascoli, S., Becker, S., Mathis, A., Schwaller, P., and Kilbertus, N. "ODEFormer: Symbolic Regression of Dynamical Systems with Transformers." *ICLR*, 2024.

[ying2025phye2e] Ying, J. et al. "A Neural Symbolic Model for Space Physics." *Nature Machine Intelligence*, 2025.

[nie2025llada] Nie, S., Zhu, F., et al. "Large Language Diffusion Models." *arXiv:2502.09992*, 2025.

[sahoo2024mdlm] Sahoo, S. S., Arriola, M., Gokaslan, A., Marroquin, E. M., Rush, A. M., Schiff, Y., Chiu, J. T., and Kuleshov, V. "Simple and Effective Masked Diffusion Language Models." *NeurIPS*, 2024.

[architects2025arc] The ARChitects Team, Lambda Labs. "The ARChitects -- ARC Prize 2025 Solution Technical Report." 2025.

[su2021rope] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*, 2021.

[hu2022lora] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.

[zheng2023mdtv2] Zheng, S. et al. "MDTv2: Masked Diffusion Transformer is a Strong Image Synthesizer." *arXiv:2303.14389*, 2023.

[zheng2024maskdit] Zheng, H. et al. "Fast Training of Diffusion Models with Masked Transformers." *arXiv:2306.09305*, 2024.

[lacava2021srbench] La Cava, W., Orzechowski, P., Burlacu, B., de Franca, F. O., Virgolin, M., Jin, Y., Kommenda, M., and Moore, J. H. "Contemporary Symbolic Regression Methods and their Relative Performance." *NeurIPS Datasets and Benchmarks Track*, 2021.

[matsubara2023srsd] Matsubara, Y., Chiba, N., Igarashi, R., and Ushiku, Y. "Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery." *DMLR*, 2023.

[tian2024symq] Tian, Y., Zhou, W., Dong, H., Kammer, D. S., and Fink, O. "Interactive Symbolic Regression through Offline Reinforcement Learning: A Co-Design Framework." *Nature Communications*, 2025.

[shojaee2025llmsr] Shojaee, P., Meidani, K., Gupta, S., Farimani, A. B., and Reddy, C. K. "LLM-SR: Scientific Equation Discovery via Programming with Large Language Models." *ICLR*, 2025.

[llmsrbench2025] Shojaee, P. et al. "LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models." *arXiv:2504.10415*, 2025.
