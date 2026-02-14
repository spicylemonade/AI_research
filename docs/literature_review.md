# Literature Review: Transformer-Based Physics Equation Discovery

## 1. Symbolic Regression Foundations

### 1.1 Genetic Programming & Classical Approaches

**PySR** (Cranmer, 2023) is the current gold-standard evolutionary SR tool. It uses a multi-population evolutionary algorithm with an evolve-simplify-optimize loop, backed by a high-performance Julia backend (SymbolicRegression.jl). PySR maintains a Pareto front balancing accuracy and complexity. It is widely used across physics, astrophysics, and engineering.

**AI Feynman** (Udrescu & Tegmark, 2020) introduced a physics-inspired recursive SR method that combines neural network fitting with symmetry detection, separability testing, and dimensional analysis. It discovered all 100 equations from the Feynman Lectures. **AI Feynman 2.0** (Udrescu et al., 2020) added Pareto-optimal search and graph modularity discovery, achieving orders-of-magnitude greater robustness to noise.

**QDSR** (Bruneton, 2025) achieves **91.6% exact recovery** on the AI Feynman noiseless benchmark — the current SOTA — by combining genetic programming with Quality-Diversity (MAP-Elites) search and dimensional analysis. This surpasses previous methods by >20 percentage points.

### 1.2 Standard Benchmarks

- **AI Feynman**: 100+ physics equations from the Feynman Lectures (Udrescu & Tegmark, 2020)
- **Nguyen**: 12 standard equations for SR evaluation (Uy et al., 2011)
- **Strogatz/ODE-Strogatz**: 2-state nonlinear ODE systems from Strogatz's textbook, used in SRBench (La Cava et al., 2021)
- **SRBench**: Comprehensive benchmark comparing 14 SR methods (La Cava et al., 2021)

---

## 2. Neural/Transformer Approaches to Symbolic Regression

### 2.1 Lample & Charton (2020) — Deep Learning for Symbolic Mathematics

**Key contribution**: First demonstration that seq2seq transformers can perform symbolic mathematics (integration, ODE solving), outperforming Mathematica and Matlab. Introduced **prefix (Polish) notation** for encoding mathematical expressions as token sequences. Architecture: 6-layer transformer, 8 heads, d_model=512.

**Relevance to our work**: We adopt their prefix notation scheme and encoder-decoder architecture as our autoregressive baseline.

### 2.2 NeSymReS (Biggio et al., 2021) — Neural Symbolic Regression that Scales

**Key contribution**: First large-scale pre-trained transformer for SR. Encoder compresses numerical (X, Y) pairs into a latent vector z, which conditions an autoregressive decoder via cross-attention. Pre-trained on procedurally generated equations. Orders of magnitude faster than GP at equivalent accuracy.

**Limitations**: Only supports d ≤ 3 input variables; strong reproduction bias (struggles with expressions outside training distribution).

### 2.3 E2E Symbolic Regression (Kamienny et al., 2022)

**Key contribution**: End-to-end transformer that directly predicts the full symbolic expression including numerical constants (no skeleton + constant fitting two-step process). Constants can be further refined via non-convex optimization using the transformer's prediction as initialization. Approaches GP performance with orders of magnitude faster inference.

### 2.4 TPSR (Shojaee et al., 2023) — Transformer-based Planning for SR

**Key contribution**: Integrates Monte Carlo Tree Search (MCTS) into transformer decoding, enabling non-differentiable feedback (fitting accuracy, complexity) to guide generation. Frames SR as a Markov Decision Process with UCT-based selection. Outperforms standard beam search and sampling on SRBench.

**Relevance**: Demonstrates value of iterative refinement during inference — our soft-mask refinement is conceptually related but operates in the masked diffusion paradigm.

---

## 3. Masked Diffusion Models for Discrete Sequences

### 3.1 LLaDA (Nie et al., 2025) — Large Language Diffusion Models

**Key contribution**: Scales masked diffusion to an 8B-parameter language model. Forward process masks tokens with probability t ∈ (0,1); reverse process iteratively predicts masked tokens. Unlike BERT (fixed mask ratio), LLaDA's variable masking yields a proper generative model (ELBO bound). Competitive with LLaMA3 8B on benchmarks, and resolves the "reversal curse."

**Critical relevance**: Our PhysMDT directly adapts LLaDA's masked diffusion training objective to the physics equation domain. We use random masking rates during training and iterative unmasking during inference.

### 3.2 MDLM (Sahoo et al., 2024) — Simple and Effective Masked Diffusion LMs

**Key contribution**: Shows masked discrete diffusion performs much better than previously thought with proper engineering. Derives a simplified Rao-Blackwellized objective (mixture of classical MLM losses). Achieves SOTA among diffusion LMs, within 14% of AR perplexity on LM1B.

**Relevance**: Validates the masked diffusion paradigm as viable for discrete token generation. Our training objective builds on their simplified loss formulation.

### 3.3 DiffuSR (2025) — Diffusion Language Model for SR

**Key contribution**: Pre-training framework using continuous-state diffusion. Maps discrete mathematical symbols to continuous latent space via trainable embeddings. Cross-attention conditions on numerical data. Logit priors can seed genetic programming search.

### 3.4 DDSR (Bastiani et al., 2025) — Diffusion-Based SR

**Key contribution**: Discrete masking diffusion combined with reinforcement learning (token-wise GRPO). Masks one token at a time during forward process. Outperforms 4/5 ML methods on SRBench while maintaining interpretability.

### 3.5 Symbolic-Diffusion (2025) — D3PM for SR

**Key contribution**: Uses D3PM discrete state-space diffusion to generate all tokens simultaneously. Unlike autoregressive models, transformer blocks have global context to all positions. Shows comparable/improved performance over AR generation.

---

## 4. ARC 2025 Architecture Innovations

### 4.1 The ARChitects Solution (2025)

**Key innovations adapted for our work**:

1. **Masked diffusion training**: Modified LLaDA-8B with task-specific adaptations. All output tokens are masked and must be predicted, including structural delimiters.

2. **Dual-axis RoPE**: Modified standard 1D RoPE to 2D positional encoding (inspired by Golden Gate RoPE), incorporating multiple directional encodings. We adapt this concept to encode both **sequence position** and **expression tree depth**.

3. **Recursive soft-mask refinement**: Iterative inference where the model refines its predictions over multiple passes, with soft masks allowing partial confidence to propagate. We implement this as K-step iterative refinement with convergence detection.

4. **Test-time finetuning (TTF)**: Per-task LoRA adaptation at inference time. Single-task TTF outperforms multi-task approaches.

5. **Token algebra**: Manipulation of tokens in continuous embedding space for symbolic reasoning.

---

## 5. Physics-Informed Machine Learning

### 5.1 PINNs (Raissi et al., 2019)

**Key contribution**: Embeds physical laws (PDEs) directly into neural network training via loss function regularization. Demonstrated on Navier-Stokes, Schrödinger, and other PDEs.

**Relevance**: We adapt the physics-informed loss concept to symbolic regression: dimensional consistency, conservation laws, and symmetry enforcement as additional training signals.

---

## 6. Identified Gaps and Our Contributions

| Gap | Our Solution |
|-----|-------------|
| No masked diffusion model specifically for physics SR | PhysMDT: first masked diffusion transformer for physics equation discovery |
| Autoregressive models lack global context during generation | Masked diffusion provides bidirectional context at all positions |
| No physics-aware positional encodings for equation trees | Dual-axis RoPE encoding sequence position + tree depth |
| Single-pass inference limits equation quality | K-step iterative soft-mask refinement with cold-restart |
| No physics constraints in neural SR training | Dimensional consistency, conservation, and symmetry losses |
| No test-time adaptation for neural SR | Per-equation LoRA TTF with data augmentation |
| No structure-guided generation for neural SR | Dual-model skeleton predictor constraining diffusion |

---

## References

All 21 papers are cited in `sources.bib`. Key references by topic:

- **Masked Diffusion**: nie2025llada, sahoo2024mdlm
- **Transformer SR**: lample2020deep, biggio2021nesymres, kamienny2022e2e, shojaee2023tpsr
- **Diffusion SR**: diffusr2025, ddsr2025, symbolicdiffusion2025
- **Classical SR**: bruneton2025qdsr, cranmer2023pysr, udrescu2020aifeynman, udrescu2020aifeynman2
- **Benchmarks**: lacava2021srbench, uy2011nguyen, strogatz2015nonlinear
- **Architecture**: vaswani2017attention, su2024rope, hu2022lora, architects2025arc
- **Physics-Informed ML**: raissi2019pinns
