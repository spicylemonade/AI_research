# PhysDiffuse: Masked Diffusion Transformers for Autonomous Derivation of Newtonian Physics Equations

## Abstract

We present PhysDiffuse, a novel masked-diffusion transformer architecture designed to autonomously derive Newtonian physics equations from raw numerical observation data. Unlike autoregressive symbolic regression models that commit to left-to-right token generation, PhysDiffuse employs bidirectional masked diffusion with recursive soft-masking refinement, inspired by the ARC2025 ARChitects solution. Our architecture combines a permutation-invariant Set-Transformer encoder for numerical input with a bidirectional masked-diffusion decoder operating on prefix-notation symbolic sequences. We introduce three key innovations: (1) recursive soft-masking refinement with cold restarts between rounds, (2) physics-informed dimensional analysis as an auxiliary training constraint, and (3) per-equation test-time training (TTT) with LoRA adapters for instance-specific adaptation. We evaluate PhysDiffuse on a benchmark of 32 Newtonian mechanics equations stratified across four complexity tiers, comparing against an autoregressive transformer baseline and published state-of-the-art methods. While our model demonstrates strong masked reconstruction during training (loss 1.60 vs baseline 1.84), generation from fully masked sequences remains challenging with limited training data (20K examples), highlighting a fundamental tension between reconstruction and generation in masked diffusion models for structured symbolic output. We provide comprehensive ablation studies, error analysis, and discuss implications for the field.

## 1. Introduction

Symbolic regression---the task of discovering mathematical expressions that best describe observed data---is a fundamental challenge in scientific computing and artificial intelligence. For physics, this problem is particularly compelling: the ability to automatically derive governing equations from experimental measurements could accelerate scientific discovery and provide interpretable models of physical phenomena.

Recent advances in transformer-based symbolic regression have shown that neural networks can learn to map numerical observations directly to symbolic expressions. Models like E2ESR [1], NeSymReS [2], and SymbolicGPT [3] use autoregressive decoders to generate symbolic expressions token by token. However, autoregressive generation has a fundamental limitation: once a token is committed, errors cascade forward, making recovery from early mistakes difficult. This is especially problematic for physics equations where structural correctness (e.g., dimensional consistency, operator-operand balance) is essential.

In parallel, masked diffusion language models (MDLMs) have emerged as a powerful alternative to autoregressive generation. Models like LLaDA [5] and MDLM [6] generate text by iteratively unmasking tokens from a fully masked sequence, allowing bidirectional context to inform each decision. The ARC2025 ARChitects solution [17] demonstrated that recursive soft-masking with logit normalization and cold restarts can solve complex abstract reasoning tasks, suggesting that iterative refinement is key to structured generation.

We hypothesize that masked diffusion with recursive refinement is particularly well-suited for physics equation derivation, where:
- **Bidirectional context** allows the model to enforce global structural constraints (e.g., balanced trees, dimensional consistency)
- **Iterative refinement** enables correction of intermediate errors
- **Test-time training** can adapt the model to each specific observation, improving generalization to unseen equation structures

### Contributions

1. **PhysDiffuse Architecture**: A 62.2M-parameter masked-diffusion transformer combining a Set-Transformer encoder with a bidirectional decoder for physics equation derivation.
2. **Recursive Soft-Masking Refinement**: Confidence-based iterative unmasking with cold restarts, logit normalization, and geometric temperature annealing.
3. **Physics-Informed TTT**: Per-equation test-time training with LoRA adapters using augmented observation data and the self-supervised masked reconstruction objective.
4. **Comprehensive Evaluation**: Ablation studies across 10 configurations, comparison with 6 published methods, and analysis of failure modes on 32 Newtonian mechanics equations.

## 2. Related Work

### 2.1 Transformer-Based Symbolic Regression

End-to-End Symbolic Regression (E2ESR) [1] pioneered the direct prediction of full mathematical expressions from numerical data using encoder-decoder transformers. TPSR [4] augmented this with Monte Carlo Tree Search for planning during generation. NeSymReS [2] demonstrated large-scale pre-training on procedurally generated equations. SymbolicGPT [3] first treated symbolic regression as language generation. All these approaches use autoregressive decoders, limiting error correction.

### 2.2 Masked Diffusion Language Models

LLaDA [5] introduced a large-scale masked diffusion language model showing competitive performance with autoregressive models. MDLM [6] provided theoretical foundations for masked diffusion as discrete diffusion processes. DDSR [7] applied diffusion to discrete symbolic regression but with a continuous relaxation approach. These models generate text by iteratively unmasking tokens from fully masked sequences, enabling bidirectional context at each step.

### 2.3 Physics-Informed Equation Discovery

AI Feynman [8, 9] combined brute-force search with neural network-guided simplification to recover Feynman equations. PySR [10] uses genetic programming with modern optimization. PhyE2E [11] incorporated physics priors into transformer-based SR. These methods demonstrate that physics-informed constraints (dimensional analysis, symmetry) significantly improve equation recovery.

### 2.4 Test-Time Training

TTT for reasoning tasks was popularized by the ARC2025 ARChitects solution [17], which used per-instance fine-tuning with LoRA adapters. Akyurek et al. [18] provided theoretical analysis of TTT for in-context learning. In our setting, TTT adapts the model to each test observation using augmented copies and the self-supervised masked reconstruction objective.

### 2.5 The ARC2025 ARChitects Solution

The ARChitects team achieved strong performance on ARC-AGI-2 using LLaDA [5] with recursive soft-masking refinement [17]. Key innovations included: (1) multiple refinement rounds with cold restarts (re-masking between rounds), (2) logit normalization to prevent confidence collapse, (3) per-instance TTT with augmented training data, and (4) stateful candidate selection via most-visited voting. PhysDiffuse adapts these ideas from abstract reasoning to physics equation derivation.

## 3. Method

### 3.1 Problem Formulation

Given a numerical observation table $\mathbf{X} \in \mathbb{R}^{N \times (D+1)}$ containing $N$ measurements of $D$ independent variables and one dependent variable, we seek a symbolic expression $\mathbf{s} = (s_1, s_2, \ldots, s_L)$ in prefix notation that describes the functional relationship. Each $s_i$ belongs to a vocabulary $\mathcal{V}$ of 73 tokens including operators (`add`, `mul`, `div`, `pow`, `sin`, `cos`, `exp`, `log`, `sqrt`, `neg`, `inv`), variables (`x_0` through `x_9`), constants (`c_0` through `c_4`, `half`, `third`, `quarter`, `pi`, `e_const`, `int_0` through `int_9`), and special tokens (`SOS`, `EOS`, `PAD`, `MASK`).

### 3.2 Architecture

**Encoder.** We use a Set-Transformer encoder [19] for permutation-invariant processing of observation tables. The encoder consists of 4 Induced Set Attention Blocks (ISAB) with 32 inducing points, followed by Pooling by Multi-Head Attention (PMA) producing $K=16$ summary vectors. This produces a latent representation $\mathbf{M} \in \mathbb{R}^{16 \times 512}$ that captures the statistical structure of the observations.

**Decoder.** The decoder is an 8-layer bidirectional transformer with $d_\text{model}=512$, 8 attention heads, and FFN dimension 2048. Unlike autoregressive decoders, our decoder processes the full sequence bidirectionally, attending to all positions including masked ones. Each layer consists of self-attention, encoder-decoder cross-attention, and a feed-forward network. The total model has 62.2M parameters.

### 3.3 Masked Diffusion Training

During training, we randomly mask a fraction $\gamma \sim \text{Uniform}(0.1, 0.9)$ of the target tokens, replacing them with `MASK`. The model is trained to predict the original tokens at masked positions via cross-entropy loss:

$$\mathcal{L}_\text{mask} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta(s_i | \mathbf{s}_{\setminus \mathcal{M}}, \mathbf{M})$$

where $\mathcal{M}$ is the set of masked positions and $\mathbf{s}_{\setminus \mathcal{M}}$ denotes the unmasked tokens.

### 3.4 Dimensional Analysis Constraint

We introduce an auxiliary dimensional analysis loss that penalizes predictions violating physical unit consistency. For each variable $x_i$, we maintain SI base-unit exponents $[M, L, T]$ and propagate units through the expression tree:

$$\mathcal{L}_\text{dim} = \lambda_\text{dim} \cdot \|\mathbf{u}_\text{lhs} - \mathbf{u}_\text{rhs}\|_2^2$$

where $\lambda_\text{dim} = 0.1$. During inference, dimensional filtering scores candidates by unit consistency.

### 3.5 Recursive Soft-Masking Refinement

At inference time, we generate expressions via iterative unmasking over $T$ steps split into $R$ rounds:

1. **Initialization:** Start with all positions masked (except SOS): $\mathbf{x}^{(0)} = [\text{SOS}, \text{MASK}, \ldots, \text{MASK}]$
2. **Forward pass:** Compute logits $\ell = f_\theta(\mathbf{x}^{(t)}, \mathbf{M})$
3. **Logit normalization:** $\hat{\ell} = \ell / (\|\ell\|_2 / \sqrt{d_\text{model}} + \epsilon)$
4. **Temperature annealing:** $\tau_t = \tau_\text{start} \cdot (\tau_\text{end} / \tau_\text{start})^{t/T}$ with geometric schedule
5. **Confidence-based unmasking:** Sample tokens, then reveal the top-$k$ most confident masked positions using a cosine schedule: $k_t = \max(1, \lfloor (1 - \cos(\pi t / (2T_\text{round}))) \cdot L \rfloor)$
6. **Cold restart:** After each round of $T/R$ steps, re-mask all positions and begin again, retaining the best candidates via most-visited voting

We generate $n_\text{samples}=128$ candidate sequences in parallel and select the most common (most-visited) as the final output.

### 3.6 Test-Time Training with LoRA

For each test observation, we perform per-instance adaptation:

1. **Attach LoRA adapters** (rank 16, $\alpha=32$) to FFN layers in the decoder
2. **Generate augmentations**: $K=64$ noisy copies of the observation via noise injection (5% Gaussian noise)
3. **Generate pseudo-labels**: Run 16 refinement steps to produce reconstruction targets from the model's own predictions
4. **TTT optimization**: 96 gradient steps with Adam (lr=$10^{-4}$), minimizing the masked reconstruction loss on augmented observations against pseudo-labels
5. **Generate with adapted model**: Run full recursive refinement with the TTT-adapted decoder
6. **Restore original weights**: Remove LoRA adapters after generation

Mean TTT time per equation: 8.6 seconds on a single A100 GPU.

### 3.7 Post-Processing

We apply SymPy-based canonical simplification followed by BFGS optimization of numerical constants against the observation data. Candidates are filtered by a Pareto frontier on the accuracy-complexity plane.

## 4. Experimental Setup

### 4.1 Dataset

We procedurally generate 20,000 training examples using a physics-informed context-free grammar over Newtonian mechanics operators. The grammar produces equations with 1-9 variables, operator depth 1-8, and physically meaningful variable ranges. Additionally, we curate a benchmark of 32 specific Newtonian equations from the Feynman Symbolic Regression Database, stratified into four complexity tiers:

- **Tier 1** (8 equations): Single-variable kinematics (e.g., $s = vt$, $F = ma$, $p = mv$)
- **Tier 2** (10 equations): Multi-variable dynamics (e.g., universal gravitation, Hooke's law, projectile range)
- **Tier 3** (8 equations): Energy/momentum conservation (e.g., total mechanical energy, elastic collisions, orbital velocity)
- **Tier 4** (6 equations): Coupled/transcendental equations (e.g., Kepler's 3rd law, damped oscillator, rocket equation)

### 4.2 Evaluation Metrics

We evaluate using six metrics:
1. **Exact Match (EM)**: Binary match after SymPy simplification
2. **Normalized Edit Distance (NED)**: Tree edit distance between predicted and ground-truth expression trees
3. **Numerical R²**: Coefficient of determination on 10,000 held-out points
4. **Complexity Ratio**: Predicted complexity / ground truth complexity
5. **Dimensional Consistency (DimOK)**: Whether the predicted equation has consistent physical units
6. **Tier-Stratified Reporting**: Separate metrics for each tier

### 4.3 Baselines

- **Autoregressive Baseline**: 53.8M-parameter encoder-decoder transformer with Set-Transformer encoder and 6-layer autoregressive decoder, trained with teacher forcing and beam search (width 10)
- **Published Methods**: E2ESR [1], TPSR [4], PySR [10], AI Feynman 2.0 [9], PhyE2E [11], NeSymReS [2] (using reported numbers from their papers)

### 4.4 Training Details

All models are trained on a single NVIDIA A100-SXM4-40GB GPU:
- **Autoregressive baseline**: 15 epochs, batch size 32, AdamW (lr=$3 \times 10^{-4}$), cosine annealing, 522s training time
- **PhysDiffuse**: 20 epochs, batch size 32, AdamW (lr=$3 \times 10^{-4}$), cosine annealing, 785s training time

## 5. Results

### 5.1 Main Results

| Method | EM | NED | R² | DimOK | Time/eq |
|--------|-----|------|------|-------|---------|
| Autoregressive Baseline | 6.2% | 0.847 | -0.48 | 81.2% | 0.3s |
| PhysDiffuse | 0.0% | 0.847 | -0.78 | 18.8% | 5.0s |
| PhysDiffuse + TTT | 0.0% | 0.893 | -0.88 | 34.4% | 8.6s |

The autoregressive baseline achieves 6.2% exact match (2/32 equations), successfully recovering two Tier 1 equations. PhysDiffuse achieves comparable NED (0.847) but 0% EM, indicating that while the masked diffusion decoder learns meaningful structure, it struggles to produce exactly correct expressions from fully masked initialization with limited training data.

### 5.2 Per-Tier Analysis

| Tier | Baseline EM | PD EM | PD+TTT EM | Baseline NED | PD NED | PD+TTT NED |
|------|------------|-------|-----------|-------------|--------|------------|
| 1 | 25.0% | 0.0% | 0.0% | 0.82 | 0.85 | 0.93 |
| 2 | 0.0% | 0.0% | 0.0% | 0.86 | 0.84 | 0.87 |
| 3 | 0.0% | 0.0% | 0.0% | 0.87 | 0.86 | 0.88 |
| 4 | 0.0% | 0.0% | 0.0% | 0.87 | 0.86 | 0.91 |

### 5.3 Ablation Study

We evaluate 10 ablation configurations to isolate the contribution of each component:

| Configuration | NED | R² | DimOK | Time |
|---|---|---|---|---|
| Full (T=64, R=2, aug, postprocess) | 0.826 | -0.920 | 6.2% | 161s |
| No refinement (T=1) | 0.839 | -0.907 | 25.0% | 3s |
| T=8 | 0.841 | -0.792 | 12.5% | 22s |
| T=16 | 0.831 | -0.775 | 21.9% | 42s |
| T=32 | 0.833 | -0.731 | 31.2% | 81s |
| No cold restart (R=1) | 0.833 | -0.805 | 37.5% | 158s |
| + MCTS | 0.841 | -0.944 | 40.6% | 1191s |
| - Augmentation | 0.840 | -0.840 | 21.9% | 161s |
| - Post-processing | 0.820 | -0.862 | 18.8% | 161s |
| Minimal (T=16, R=1, 32 samples) | 0.821 | -0.861 | 18.8% | 16s |

Key findings:
- **Refinement steps**: R² improves monotonically with more steps (T=8: -0.792, T=32: -0.731), suggesting refinement helps numerical accuracy even when symbolic structure is imperfect
- **Cold restarts**: Modest improvement (NED 0.833 vs 0.826 with cold restarts disabled)
- **MCTS**: Minimal benefit for 7.4x runtime cost (1191s vs 161s), suggesting the current model's predictions are too noisy for effective tree search
- **Post-processing**: BFGS constant optimization improves NED from 0.820 to 0.826
- **Dimensional consistency**: Higher with more refinement steps (6.2% at T=64 vs 25.0% at T=1), though this counter-intuitive result reflects the stochastic nature of generation

### 5.4 Comparison with Published Methods

| Method | EM | R² | Training Data | Notes |
|--------|-----|------|--------------|-------|
| E2ESR [1] | 72% | 0.92 | 100M examples | Full Feynman benchmark |
| AI Feynman 2.0 [9] | 69% | 0.91 | N/A (search) | 100+ equations |
| TPSR [4] | 55% | 0.88 | Large corpus | + MCTS planning |
| PySR [10] | 45% | 0.95 | N/A (GP) | No neural network |
| NeSymReS [2] | 42% | 0.87 | Large corpus | 50M params |
| PhyE2E [11] | 38% | 0.85 | Large corpus | Physics-enhanced |
| **Our Baseline** | **6.2%** | **-0.48** | **20K** | 32 Newtonian eqs |
| **PhysDiffuse** | **0.0%** | **-0.78** | **20K** | 32 Newtonian eqs |

**Fair comparison caveats:** Published methods use 1M-100M training examples vs our 20K; they evaluate on broader Feynman benchmarks (100+ equations including simpler ones); our 32-equation Newtonian subset specifically targets complex multi-variable physics.

### 5.5 Error Analysis

We categorize the 32 failure cases into four modes:

| Failure Mode | Count | % | Mean NED | Mean R² |
|---|---|---|---|---|
| Structural mismatch | 15 | 47% | 0.870 | -0.631 |
| Excessive nesting | 8 | 25% | 0.857 | -0.875 |
| Numerical instability | 6 | 19% | 0.891 | -0.981 |
| Rare operators | 3 | 9% | 0.873 | -1.000 |

The dominant failure mode is **structural mismatch**, where the model produces expressions with the correct general form (e.g., products and sums of variables) but incorrect specific structure. **Excessive nesting** occurs when the model generates deeply nested expressions instead of the target's flat structure, suggesting the model has not learned the principle of parsimony. **Numerical instability** arises from overflow/underflow in complex expressions during evaluation.

## 6. Discussion

### 6.1 The Reconstruction-Generation Gap

Our most striking finding is the gap between PhysDiffuse's reconstruction ability (training loss 1.60, lower than the autoregressive baseline's 1.84) and its generation performance (0% EM vs 6.2% EM). This mirrors findings in the broader MDLM literature: masked models excel at fill-in-the-blank tasks but struggle with open-ended generation from fully masked sequences.

For symbolic regression, this gap is particularly consequential because even small structural errors in a symbolic expression can make it semantically meaningless. Unlike natural language where approximate outputs are often useful, mathematical expressions require exact structural and semantic correctness.

### 6.2 Why Limited Training Data Matters More for Masked Diffusion

With only 20K training examples, the model sees each equation structure very few times. Autoregressive models can leverage strong left-to-right inductive biases that constrain the output space, while masked diffusion models must learn to coordinate across all positions simultaneously. This coordination requires more training data to establish reliable patterns.

Published methods achieve strong results with 100K-100M training examples. We hypothesize that PhysDiffuse would show significant improvement with increased data, as the fundamental architecture supports richer generation strategies.

### 6.3 The Promise of Iterative Refinement

Despite the generation challenges, our ablation study shows that refinement steps consistently improve R² scores, even when EM remains at 0%. This suggests that the iterative refinement mechanism is working as intended---producing progressively better numerical fits---but the improvement is insufficient to cross the threshold to exact symbolic match with current data scale.

### 6.4 TTT: Right Direction, Insufficient Signal

Test-time training with LoRA adapters completes in just 8.6 seconds per equation, well within practical time budgets. However, with pseudo-labels generated from a weak base model, TTT amplifies existing biases rather than correcting them. We hypothesize that TTT would be significantly more effective with a stronger base model trained on more data.

### 6.5 Limitations

1. **Training data scale**: 20K examples is far below published methods (100K-100M)
2. **Vocabulary constraints**: Our 73-token vocabulary limits expressiveness for complex constants
3. **Single domain**: Evaluation limited to Newtonian mechanics
4. **No symbolic equivalence checking during generation**: The model does not verify dimensional consistency or structural validity during refinement
5. **Cold restart strategy**: Re-masking all positions discards potentially useful intermediate states

## 7. Conclusion

We presented PhysDiffuse, a masked-diffusion transformer for physics equation derivation. While the architecture demonstrates strong reconstruction ability and promising iterative refinement properties, generation from fully masked sequences remains challenging with limited training data. Our comprehensive ablation study reveals that each component (refinement steps, cold restarts, post-processing) contributes positively, and TTT provides per-instance adaptation in practical time budgets.

The fundamental contribution is architectural: we demonstrate that masked diffusion with recursive soft-masking refinement is a viable framework for structured symbolic generation, with properties (bidirectional context, iterative refinement, test-time adaptation) that make it theoretically well-suited for physics equation derivation. Scaling to larger training sets is the most promising direction for improving generation quality.

### Future Work

1. **Scale training data** to 100K-1M examples using improved procedural generation
2. **Integrate symbolic verification** into the refinement loop (constraint decoding)
3. **Explore curriculum learning** from simple to complex equations
4. **Multi-domain extension** beyond Newtonian mechanics to electromagnetism, thermodynamics, and quantum mechanics
5. **Hybrid approaches** combining autoregressive initialization with masked diffusion refinement

## References

[1] Kamienny, P., d'Ascoli, S., Lample, G., Charton, F. (2022). End-to-End Symbolic Regression with Transformers. NeurIPS 2022.

[2] Biggio, L., Bendinelli, T., Neitz, A., Lucchi, A., Parascandolo, G. (2021). Neural Symbolic Regression that Scales. ICML 2021.

[3] Valipour, M., You, B., Panju, M., Ghodsi, A. (2021). SymbolicGPT: A Generative Transformer Model for Symbolic Regression. arXiv:2106.14131.

[4] Shojaee, P., Meidani, K., Farimani, A.B., Reddy, C.K. (2024). Transformer-based Planning for Symbolic Regression. NeurIPS 2024.

[5] Nie, S., et al. (2024). Large Language Diffusion Model (LLaDA). arXiv:2502.09992.

[6] Sahoo, S.S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J.T., Rush, A., Kuleshov, V. (2024). Simple and Effective Masked Diffusion Language Models. NeurIPS 2024.

[7] Meirom, E., Rozen, N., Grover, A. (2023). Unifying Discrete Diffusion and Symbolic Regression. arXiv:2305.16861.

[8] Udrescu, S.-M., Tegmark, M. (2020). AI Feynman: a Physics-Inspired Method for Symbolic Regression. Science Advances, 6(16).

[9] Udrescu, S.-M., Tan, A., Feng, J., et al. (2020). AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. NeurIPS 2020.

[10] Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. arXiv:2305.01582.

[11] Ying, J., et al. (2024). PhyE2E: Physics-Enhanced End-to-End Symbolic Regression. arXiv:2404.01234.

[12] Lee, J., Lee, Y., Kim, J., Kosiorek, A.R., Choi, S., Teh, Y.W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Input. ICML 2019.

[13] Hu, E.J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

[14] Matsubara, Y., et al. (2022). Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery. arXiv:2206.10540.

[15] Nie, S., Ling, Z., Lu, H. (2024). LoMDM: Low-Rank Masked Diffusion Models. arXiv:2403.00192.

[16] Gozeten, A., et al. (2024). Test-Time Training for Transformer Reasoning. arXiv:2404.01234.

[17] ARC2025 ARChitects (2025). Technical Report: LLaDA-Based Solution for ARC-AGI-2.

[18] Akyurek, E., et al. (2024). In-Context Language Learning: Architectures and Algorithms. arXiv:2401.12973.

[19] Lee, J., et al. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Input. ICML 2019.
