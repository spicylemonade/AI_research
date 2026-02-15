# Literature Review: Symbolic Regression and Equation Discovery with Transformers

## 1. Symbolic Regression with Transformers

### 1.1 Neural Symbolic Regression that Scales (Biggio et al., ICML 2021)
**Key idea**: Pre-train a Set Transformer encoder + autoregressive Transformer decoder on procedurally generated equations. At test time, query the model on numerical observations and use its output to guide symbolic search.
- **Architecture**: Set Transformer encoder (~11M params) for permutation-invariant processing of observation pairs; standard Transformer decoder (~13M params) for autoregressive equation generation.
- **Strengths**: First scalable neural approach to SR; orders of magnitude faster than genetic programming; improves with more data and compute.
- **Limitations**: Two-step (skeleton then constants); limited to relatively simple equations; struggles with complex multi-variable physics.
- **Relevance**: Our observation encoder design directly builds on their Set Transformer approach. We extend with masked diffusion decoding instead of autoregressive.

### 1.2 End-to-End Symbolic Regression with Transformers (Kamienny et al., NeurIPS 2022)
**Key idea**: Single Transformer predicts full symbolic expressions including numerical constants end-to-end, eliminating the skeleton-then-fit pipeline.
- **Strengths**: End-to-end training simplifies the pipeline; competitive with genetic programming at much faster inference.
- **Limitations**: Still autoregressive; sequential token generation limits refinement capability.
- **Relevance**: Validates that transformers can handle constants directly. Our approach further improves by enabling parallel token prediction and iterative refinement.

### 1.3 SymbolicGPT (Valipour et al., 2021)
**Key idea**: GPT-style decoder-only transformer for symbolic regression with an order-invariant T-net for encoding input datasets.
- **Strengths**: Demonstrates pure language modeling approach works for SR; good data efficiency.
- **Relevance**: Establishes the autoregressive baseline paradigm we compare against.

### 1.4 Deep Symbolic Regression for Recurrent Sequences (d'Ascoli et al., ICML 2022)
**Key idea**: Train transformers to infer functions/recurrence relations underlying sequences, demonstrating discovery of out-of-vocabulary approximations.
- **Strengths**: Discovers approximate symbolic forms for functions not in training set.
- **Relevance**: Directly demonstrates zero-shot discovery capability of transformers, which is our core claim.

## 2. Physics Discovery Methods

### 2.1 AI Feynman (Udrescu & Tegmark, Science Advances 2020)
**Key idea**: Recursive physics-inspired symbolic regression combining neural network fitting with dimensional analysis, symmetry detection, and separability.
- **Strengths**: Discovers all 100 Feynman equations (vs 71 for Eureqa); leverages physics priors (dimensional analysis, separability).
- **Limitations**: Hand-crafted pipeline of physics heuristics; not end-to-end learnable; requires careful feature engineering.
- **Relevance**: Gold standard for physics SR benchmarks. Our dimensional analysis bias is inspired by their dimensional analysis module. We use their equation corpus as reference.

### 2.2 SINDy (Brunton et al., PNAS 2016)
**Key idea**: Sparse regression on a library of candidate nonlinear functions to identify governing dynamical equations.
- **Strengths**: Parsimonious models; works well for dynamical systems; theoretically grounded.
- **Limitations**: Requires pre-defined function library; struggles with complex compositions; needs time-series data.
- **Relevance**: Complementary approach focused on ODEs/PDEs rather than algebraic equations. Validates sparsity as inductive bias.

### 2.3 Discovering Symbolic Models from Deep Learning (Cranmer et al., NeurIPS 2020)
**Key idea**: Train GNNs with sparse latent representations, then apply symbolic regression to extract explicit formulas from learned message functions.
- **Strengths**: Discovered novel cosmological formula; better OOD generalization than GNN alone.
- **Limitations**: Two-stage pipeline (train GNN, then SR); requires graph-structured domain knowledge.
- **Relevance**: Demonstrates that neural networks can discover genuinely novel physics, supporting our hypothesis.

## 3. Physics-Informed Neural Networks

### 3.1 Hamiltonian Neural Networks (Greydanus et al., NeurIPS 2019)
**Key idea**: Parameterize the Hamiltonian with a neural network and enforce Hamilton's equations as an inductive bias, guaranteeing energy conservation.
- **Strengths**: Learns conservation laws from data; works from pixels.
- **Limitations**: Requires canonical coordinates; doesn't produce symbolic expressions.

### 3.2 Lagrangian Neural Networks (Cranmer et al., 2020)
**Key idea**: Parameterize the Lagrangian directly, avoiding the need for canonical coordinates.
- **Strengths**: Works with arbitrary coordinates; energy-conserving by construction.
- **Relevance**: Inspires our dimensional analysis bias — embedding physical structure into the architecture.

## 4. Masked Diffusion Language Models

### 4.1 MDLM (Sahoo et al., NeurIPS 2024)
**Key idea**: Simplified masked diffusion framework with Rao-Blackwellized objective; mixture of masked language modeling losses.
- **Strengths**: Closes gap with autoregressive models on language benchmarks; efficient sampling; robust to overfitting.
- **Key insight**: The training objective is an upper bound on negative log-likelihood, making it a proper generative model.
- **Relevance**: Core foundation for our PhysMDT training procedure. The masking schedule and denoising framework directly inform our approach.

### 4.2 LLaDA (Nie et al., 2025)
**Key idea**: Scale masked diffusion to 8B parameters with standard pre-training + SFT pipeline; demonstrates in-context learning and instruction following.
- **Architecture**: Transformer as mask predictor, no causal mask, sees entire input for predictions.
- **Key difference from BERT**: Variable masking ratio (U[0,1]) instead of fixed, making it a proper generative model.
- **Relevance**: The ARChitects' ARC 2025 solution fine-tuned LLaDA-8B. Validates masked diffusion at scale.

## 5. ARC 2025 ARChitects Solution

### 5.1 Architecture & Training
- Fine-tuned LLaDA-8B with rank-512 LoRA on 8×H100 GPUs
- Replaced standard 1D RoPE with 2D Golden Gate RoPE for grid structure
- Fully random masking with CE loss on masked positions only

### 5.2 Recursive Soft-Masking Refinement (Key Innovation)
- **Token algebra**: Tokens are points in continuous embedding space; enables blending (e.g., 0.5*color_0 + 0.5*color_1)
- **Soft-masking**: Add <mask> embedding to every position → model refines predictions iteratively
- **102 refinement steps** (two rounds of 51 with cold restart)
- **Candidate selection**: Most-visited-candidate voting across refinement trajectory

### 5.3 Test-Time Fine-Tuning
- 128 steps per task with rank-32 LoRA
- Augmented examples for data efficiency
- Critical for adapting to specific problem structure

### 5.4 Results
- 21.67% on public leaderboard (competitive for a first-generation approach)
- Shape prediction: ~85% accuracy
- Known-shape performance: ~30.5%

## 6. Key Insights for PhysMDT Design

1. **Masked diffusion > autoregressive for structured output**: Parallel prediction enables global structure capture and iterative refinement.
2. **Set Transformer for observations**: Permutation invariance is the right inductive bias for measurement data.
3. **Iterative refinement is critical**: The ARChitects' soft-masking shows that progressive denoising dramatically improves output quality.
4. **Test-time adaptation works**: LoRA-based fine-tuning on specific problems bridges the distribution gap efficiently.
5. **Tree structure matters**: Expressions have inherent tree structure; positional encoding should reflect this.
6. **Physical priors help**: Dimensional analysis (AI Feynman) and conservation laws (HNNs) show that physics-aware inductive biases improve discovery.
7. **Curriculum training**: Starting simple and increasing complexity mirrors successful pedagogical approaches.

## 7. Comparison with State of the Art

| Method | Type | Physics-Aware | Zero-Shot Discovery | End-to-End | Iterative Refinement |
|--------|------|--------------|-------------------|-----------|---------------------|
| AI Feynman | Pipeline | Yes (hand-crafted) | Limited | No | No |
| SINDy | Sparse regression | Library-based | No | No | No |
| NeSymReS (Biggio) | Transformer (AR) | No | Partial | Yes | No |
| E2E-SR (Kamienny) | Transformer (AR) | No | Partial | Yes | No |
| Cranmer et al. | GNN + SR | Inductive bias | Yes | No (2-stage) | No |
| **PhysMDT (Ours)** | **Transformer (MDM)** | **Yes (learned)** | **Yes** | **Yes** | **Yes (soft-masking)** |
