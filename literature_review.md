# Literature Review: PhysDiffuse

## Overview

This literature review covers five key research areas that inform the design of PhysDiffuse: (a) transformer-based symbolic regression, (b) masked diffusion language models, (c) physics-informed equation discovery, (d) test-time training for reasoning, and (e) the ARC2025 ARChitects solution. We survey 18 papers/resources across these areas.

---

## (a) Transformer-Based Symbolic Regression

### 1. End-to-End Symbolic Regression with Transformers (E2ESR)
- **Authors:** Kamienny, d'Ascoli, Lample, Charton
- **Year:** 2022
- **Venue:** NeurIPS 2022
- **Summary:** E2ESR challenges the two-step approach of predicting a skeleton then fitting constants, instead training a transformer to directly predict the full mathematical expression (constants included). The encoder processes numerical data points and the decoder autoregressively generates prefix-notation expressions. BFGS refinement of predicted constants further improves accuracy. This is our primary autoregressive baseline architecture. E2ESR demonstrates that a single forward pass can recover many benchmark equations, but struggles with complex multi-variable expressions where left-to-right commitment is suboptimal.

### 2. Neural Symbolic Regression that Scales (NeSymReS)
- **Authors:** Biggio, Bendinelli, Neitz, Lucchi, Parascandolo
- **Year:** 2021
- **Venue:** ICML 2021
- **Summary:** NeSymReS pioneered large-scale pre-training for symbolic regression, using a transformer encoder-decoder trained on procedurally generated equations. The encoder produces a latent representation of the numerical data that conditions the autoregressive decoder. NeSymReS demonstrated three orders of magnitude speedup over genetic programming methods. However, it was limited to ≤3 input variables and exhibited strong reproduction bias (difficulty generating expressions outside the training distribution). These limitations motivate our approach of using diffusion-based generation with test-time training for better generalization.

### 3. SymbolicGPT
- **Authors:** Valipour, You, Panju, Ghodsi
- **Year:** 2021
- **Venue:** arXiv:2106.14131
- **Summary:** SymbolicGPT was the first model to treat symbolic regression as a language generation task using a GPT-style autoregressive transformer. Each symbol/character is treated as a token, with a data feature extractor as encoder and autoregressive decoder. BFGS optimization handles constant fitting. The work demonstrated competitive performance with fast inference, but the purely autoregressive approach limits error correction during generation.

### 4. Transformer-based Planning for Symbolic Regression (TPSR)
- **Authors:** Shojaee, Meidani, Farimani, Reddy
- **Year:** 2023
- **Venue:** NeurIPS 2023
- **Summary:** TPSR integrates Monte Carlo Tree Search (MCTS) into the transformer decoding process, enabling non-differentiable feedback (fitting accuracy, expression complexity) to guide generation. The MCTS lookahead plans multiple tokens ahead, balancing exploration and exploitation via UCT. This is a key inspiration for our optional MCTS-guided refinement during masked diffusion inference. TPSR showed significant improvements over greedy/beam search decoding but remains fundamentally autoregressive at the token level.

### 5. SNIP: Symbolic-Numeric Integrated Pre-training
- **Authors:** Meidani, Shojaee, Reddy, Farimani
- **Year:** 2023
- **Venue:** arXiv:2310.02227
- **Summary:** SNIP employs contrastive learning between symbolic equation representations and their numeric evaluations, creating a unified latent space that captures both modalities. This task-agnostic pre-training produces numeric-symbolic priors useful for downstream symbolic regression. The insight of aligning numeric and symbolic representations in a shared space informs our encoder design, where observation data must be mapped to representations useful for symbolic generation.

### 6. MDLformer: Minimum Description Length for Symbolic Regression
- **Authors:** Li et al.
- **Year:** 2025
- **Venue:** ICLR 2025
- **Summary:** MDLformer replaces prediction error with minimum description length (MDL) as the search objective, observing that MDL decreases monotonically as search approaches the target formula (unlike prediction error). A neural network estimates MDL for any dataset, guiding symbolic search. This achieves 43.92% improvement over state-of-the-art on benchmark datasets. The MDL principle complements our dimensional analysis constraint as a physics-informed inductive bias.

---

## (b) Masked Diffusion Language Models

### 7. LLaDA: Large Language Diffusion Models
- **Authors:** Nie, Zhu, You, Zhang, Ou, Hu, Zhou, Lin, Wen, Li
- **Year:** 2025
- **Venue:** arXiv:2502.09992
- **Summary:** LLaDA demonstrates that masked diffusion models (MDMs) can compete with autoregressive models at scale (8B parameters). The key innovation is a variable masking ratio (0 to 1) during training, making the model a proper generative model (optimizing a log-likelihood lower bound). LLaDA achieves performance comparable to LLaMA3 8B. The ARC2025 ARChitects solution built directly on LLaDA's architecture, using its soft-masking and iterative refinement capabilities. Our PhysDiffuse decoder directly adapts LLaDA's masked diffusion objective.

### 8. MDLM: Simple and Effective Masked Diffusion Language Models
- **Authors:** Sahoo, Arriola, Gokaslan, Marroquin, Rush, Schiff, Chiu, Kuleshov
- **Year:** 2024
- **Venue:** NeurIPS 2024
- **Summary:** MDLM simplifies masked discrete diffusion with a substitution-based parameterization and Rao-Blackwellized ELBO, achieving state-of-the-art diffusion model performance. The key insight is that a mixture of masked language modeling losses with different masking rates approximates the continuous-time diffusion objective. MDLM closes the gap with autoregressive models significantly. Our masked diffusion training objective follows MDLM's simplified formulation.

### 9. Diffusion-Based Symbolic Regression (DDSR)
- **Authors:** Anonymous
- **Year:** 2025
- **Venue:** arXiv:2505.24776
- **Summary:** DDSR applies discrete diffusion models directly to symbolic regression, generating expression trees through iterative denoising. It uses a hybrid training policy combining short-term and long-term risk-seeking strategies. Constants are optimized with the Levenberg-Marquardt algorithm. This is the most directly related work to PhysDiffuse, but our approach differs in using soft-masking refinement (vs. standard discrete diffusion), physics-informed constraints, and test-time training.

### 10. LoMDM: Locally Masked Diffusion Model
- **Authors:** Anonymous
- **Year:** 2025
- **Venue:** arXiv:2602.02112
- **Summary:** LoMDM introduces a learnable scheduler that determines the generation order in masked diffusion models, unifying various generation orderings within a single framework. The learned velocity field improves generation quality while retaining parallel decoding efficiency. This informs our refinement strategy where the model can learn which positions to unmask first (e.g., operators before constants).

---

## (c) Physics-Informed Equation Discovery

### 11. AI Feynman 1.0 & 2.0
- **Authors:** Udrescu, Tegmark
- **Year:** 2020
- **Venue:** Science Advances
- **Summary:** AI Feynman introduced physics-inspired heuristics (dimensional analysis, symmetry detection, separability testing) for symbolic regression, successfully recovering all 100 Feynman Lecture equations. The 6GB Feynman Symbolic Regression Database (FSReD) became the standard benchmark. AI Feynman 2.0 added Pareto-optimal search on the accuracy-complexity frontier. Our dimensional analysis constraint is directly inspired by this work's demonstration that physics priors dramatically reduce search space.

### 12. PySR
- **Authors:** Cranmer
- **Year:** 2023
- **Venue:** arXiv:2305.01582
- **Summary:** PySR is a high-performance symbolic regression library using multi-population evolutionary algorithms with an evolve-simplify-optimize loop. Built on Julia's SymbolicRegression.jl backend with SIMD kernel fusion, it represents the state of the art in GP-based symbolic regression. PySR serves as a strong non-neural baseline for comparison and introduces the EmpiricalBench benchmark for evaluating scientific discovery capabilities.

### 13. PhyE2E: A Neural Symbolic Model for Space Physics
- **Authors:** Ying et al.
- **Year:** 2025
- **Venue:** Nature Machine Intelligence
- **Summary:** PhyE2E uses a transformer to translate numerical observations into symbolic expressions end-to-end, with LLM-synthesized training data (264K formulas from fine-tuned OpenLLaMA). It decomposes problems via second-order neural network derivatives and refines with MCTS+GP. Published in Nature Machine Intelligence, it demonstrates practical scientific discovery (improving NASA's 1993 solar activity formula). PhyE2E's end-to-end paradigm and physical unit prediction directly inform our approach.

### 14. SRSD Benchmark
- **Authors:** Matsubara, Chiba, Igarashi, Ushiku
- **Year:** 2022
- **Venue:** DMLR (NeurIPS 2022 AI4Science Workshop)
- **Summary:** SRSD recreated 120 Feynman equation datasets with physically realistic sampling ranges and proposed the Normalized Edit Distance (NED) metric for evaluating symbolic similarity between expression trees. The benchmark includes dummy-variable versions to test robustness. We adopt NED as one of our primary metrics and use the SRSD dataset categorization to identify Newtonian physics equations.

---

## (d) Test-Time Training for Reasoning

### 15. TTT for Few-Shot Learning
- **Authors:** Akyurek et al.
- **Year:** 2024
- **Venue:** arXiv:2411.07279
- **Summary:** Demonstrates that temporarily updating model parameters during inference (test-time training) dramatically improves performance on structurally novel tasks. On ARC, TTT with LoRA adapters achieves 6x higher accuracy than fine-tuned baselines, reaching 53% (matching average human performance). Key ingredients: leave-one-out task creation, geometric data augmentation, and task-specific LoRA training. This directly inspires our per-equation TTT procedure.

### 16. TTT Provably Improves Transformers
- **Authors:** Various
- **Year:** 2025
- **Venue:** arXiv:2503.11842
- **Summary:** Provides theoretical foundations for test-time training, proving that gradient-based TTT with in-context demonstrations improves transformer generalization. Characterizes how TTT mitigates distribution shift between pre-training and test tasks. This theoretical grounding supports our use of TTT for adapting to individual physics equations at test time.

---

## (e) ARC2025 ARChitects Solution

### 17. ARC2025 Solution by the ARChitects
- **Authors:** Lambda Labs / ARChitects Team
- **Year:** 2025
- **Venue:** ARC Prize 2025 Competition
- **Summary:** The winning ARC2025 solution built on LLaDA's masked diffusion architecture with key innovations: (1) recursive soft-masking refinement where all positions start masked and are iteratively refined with logit normalization, (2) 2D positional encoding for grid-structured data, (3) stateful candidate selection tracking the most-visited outputs across multiple sampling rounds, (4) cold restarts between refinement rounds to escape local optima, (5) test-time training with data augmentation. This solution achieved state-of-the-art on ARC tasks and is the primary inspiration for PhysDiffuse's inference strategy.

### 18. Deep Generative Symbolic Regression (DGSR)
- **Authors:** Holt, Qian, van der Schaar
- **Year:** 2023
- **Venue:** ICLR 2023
- **Summary:** DGSR uses a deep generative model with invariant neural networks to learn equation representations, trained end-to-end with PPO and genetic programming for diversity. Achieves higher recovery rates with more input variables and is more computationally efficient than RL-based methods. The end-to-end generative approach and invariant architectures inform our choice of permutation-invariant encoder design.

---

## Comparison Table

| Method | Type | Architecture | Training Data | Key Innovation | Feynman Benchmark | Year |
|--------|------|-------------|---------------|----------------|-------------------|------|
| E2ESR | Autoregressive | Enc-Dec Transformer | Synthetic 100M | End-to-end with constants | ~60% recovery | 2022 |
| NeSymReS | Autoregressive | Enc-Dec Transformer | Synthetic | Large-scale pre-training | 3 vars only | 2021 |
| SymbolicGPT | Autoregressive | GPT-style | Synthetic | Language model approach | Limited | 2021 |
| TPSR | Autoregressive+MCTS | Enc-Dec + Tree Search | Synthetic | MCTS lookahead | Improved recovery | 2023 |
| SNIP | Contrastive | Dual Encoder | Synthetic | Numeric-symbolic alignment | Improved | 2023 |
| MDLformer | Search-guided | Enc + MLP | Synthetic | MDL objective | +43.9% SOTA | 2025 |
| LLaDA | Masked Diffusion | Transformer | 2.3T tokens | Variable mask ratio | N/A (LM) | 2025 |
| MDLM | Masked Diffusion | Encoder-only | Text corpora | Rao-Blackwellized ELBO | N/A (LM) | 2024 |
| DDSR | Discrete Diffusion | Diffusion Model | Synthetic | Diffusion for SR | New approach | 2025 |
| AI Feynman | Physics heuristics | Neural + heuristics | N/A | Dimensional analysis | 100/100 | 2020 |
| PySR | Evolutionary | Multi-pop GA | N/A | Evolve-simplify-optimize | Strong | 2023 |
| PhyE2E | End-to-end | Transformer + MCTS/GP | LLM-synthesized 264K | Unit prediction, LLM data | Nature MI | 2025 |
| ARC2025 | Masked Diffusion | LLaDA-based | Synthetic + TTT | Recursive soft-masking | N/A (ARC) | 2025 |
| **PhysDiffuse** | **Masked Diffusion** | **Set-Enc + MDM Dec** | **Synthetic 500K** | **Soft-mask + dim. analysis + TTT** | **This work** | **2026** |

---

## Key Insights for PhysDiffuse Design

1. **Masked diffusion > autoregressive for structured output:** LLaDA and the ARC2025 solution demonstrate that iterative refinement via masked diffusion allows the model to consider global structure, which is critical for symbolic expressions where operator-operand dependencies are bidirectional.

2. **Test-time training is transformative:** The ARC2025 solution and Akyurek et al. show that per-instance LoRA adaptation with data augmentation can dramatically improve performance on novel instances, especially when the test distribution differs from training.

3. **Physics priors matter:** AI Feynman's dimensional analysis, PhyE2E's unit prediction, and MDLformer's MDL objective all demonstrate that incorporating domain knowledge as inductive biases significantly improves symbolic recovery rates.

4. **Post-processing is essential:** E2ESR, SymbolicGPT, and PhyE2E all use BFGS or similar optimization to refine predicted constants. Combining neural prediction with classical optimization yields the best results.

5. **MCTS-guided search complements neural generation:** TPSR and PhyE2E both show that tree search methods can improve upon greedy/beam search decoding, motivating our optional MCTS refinement during masked diffusion sampling.
