# Internal Peer Review: PhysMDT

**Paper:** PhysMDT: Physics Masked Diffusion Transformer for Symbolic Regression
**Authors:** Research Lab
**Date of Review:** February 2026
**Venue Target:** ML conference (e.g., NeurIPS, ICML, ICLR)

---

## Criterion Scores

### 1. Novelty (Score: 5/7)

The combination of six components -- masked diffusion training over prefix-notation symbolic sequences, dual-axis RoPE encoding sequence position and expression tree depth, iterative soft-mask refinement, test-time LoRA finetuning, physics-informed losses (dimensional consistency, conservation, symmetry), and a structure predictor for skeleton-first generation -- is genuinely novel. No prior work has assembled all six elements into a single architecture for symbolic regression. However, each individual component draws heavily from existing work (LLaDA for masked diffusion, ARChitects for dual-axis RoPE and refinement, PINNs for physics losses, LoRA for adaptation), and the novelty is primarily in the integration rather than in the invention of fundamentally new techniques. The structure predictor decomposition and the adaptation of dual-axis RoPE from grid-based abstract reasoning to expression-tree depth are the most original contributions.

### 2. Significance (Score: 2/7)

The paper does not advance the empirical state of the art. PhysMDT achieves 0% exact match and 0% symbolic equivalence across all benchmarks, placing it below every published method in the field. The composite score of 1.52 (out of 100) is categorically non-competitive. The ablation study provides some architectural insights -- particularly that dual-axis RoPE and the structure predictor are the most impactful components -- but because the absolute performance is so low (all variants score between 1.25 and 1.52), these findings are difficult to generalize to practical settings. The embedding analysis showing meaningful structure (sin/cos grouping at 0.69 similarity, arithmetic analogies at 0.59--0.63) is a positive signal but does not by itself constitute a significant contribution. To achieve significance, the approach would need to demonstrate competitive or superior performance at adequate scale, or provide theoretical insights that transcend the specific experimental setting.

### 3. Clarity (Score: 5/7)

The paper is generally well-written and well-organized. The six components are clearly described with proper mathematical formulations (Equations 1--7). The experimental setup is transparent about constraints (CPU-only, 420K parameters, 4K samples, 15 epochs), and the paper is commendably honest about the negative results -- the abstract itself states "0% exact match" and "does not recover any test equation correctly." The discussion section provides a thoughtful analysis of why the model underperforms. However, clarity could be improved in two areas: (a) the ablation study methodology is insufficiently explained -- the distinction between "estimated" (projected) and "evaluated" (directly measured) variants needs more rigorous description of how projections were computed; and (b) the composite score formula and its component weights are introduced without justification for the specific coefficients (0.3, 0.3, 0.25, 0.1, 0.05).

### 4. Correctness (Score: 3/7)

Claims about the architecture and its components are sound and well-formulated. The mathematical descriptions of masked diffusion (Eq. 1--3), dual-axis RoPE (Eq. 4--5), and physics losses (Eq. 6--8) are technically correct. However, several correctness concerns arise. First, the ablation study is largely based on projected estimates rather than actual re-training -- six of seven ablation variants were estimated, not directly evaluated. This means the component rankings (e.g., dual-axis RoPE contributing -0.274 CS) are extrapolations, not measurements, undermining the paper's central architectural claims. Second, the statistical tests (paired bootstrap, Wilcoxon) confirm that PhysMDT is significantly worse than the AR baseline, but they do not address whether the ablation differences are statistically significant. Third, the comparison with the AR baseline is confounded by a 2.8x parameter disparity (420K vs. 1.18M), which the paper acknowledges but does not control for experimentally.

### 5. Reproducibility (Score: 4/7)

The paper provides sufficient detail for reproducing the architecture: model dimensions (d_model=64, 4 layers, 4 heads), vocabulary size (147 tokens), training configuration (15 epochs, AdamW, 4K samples), and the dataset generator covering 62 templates across 7 physics families. The codebase includes modular implementations for each component (src/phys_mdt.py, src/refinement.py, src/token_algebra.py, src/ttf.py, src/physics_loss.py, src/structure_predictor.py) with passing test suites. A reproducibility script (scripts/reproduce.sh) is planned. However, the projected ablation estimates are not reproducible because the projection methodology is not fully described. Additionally, the small scale of experiments means that meaningful replication would require scaling to GPU with larger models and datasets, which fundamentally changes the experimental conditions.

### 6. Related Work (Score: 5/7)

The related work section is comprehensive, covering three relevant threads: transformer-based symbolic regression (NeSymReS, E2E Transformer, TPSR, QDSR, PySR), masked diffusion models for discrete sequences (LLaDA, MDLM, DiffuSR, DDSR, Symbolic-Diffusion), and physics-informed machine learning (PINNs, AI Feynman, dimensional analysis in QDSR). The paper cites 21 references and provides quantitative comparisons with published results. Two areas could be strengthened: (a) the paper does not discuss concurrent work on program synthesis with diffusion models, which is structurally related; and (b) the distinction between this work and DiffuSR/DDSR -- the most directly comparable diffusion-based SR methods -- could be more explicitly articulated in terms of architectural differences and why the physics-informed additions are expected to help.

### 7. Presentation (Score: 4/7)

The paper includes 7 tables and references 5 figures (architecture diagram, ablation bar chart, refinement depth curve, embedding t-SNE, embedding similarity heatmap). Tables are well-formatted with clear headers, proper use of bold for best results, and informative captions. The ablation table (Table 4) helpfully distinguishes estimated from evaluated variants with dagger notation. However, several presentation issues exist: (a) the figures are referenced but their quality and effectiveness cannot be fully assessed from the LaTeX source alone; (b) the per-difficulty table (Table 7) presents a counter-intuitive result (higher CS for complex equations) that is explained only in the text rather than being visually clarified in the table itself; and (c) the paper would benefit from a summary figure or diagram showing the overall performance landscape -- how PhysMDT fits relative to all baselines and published methods -- rather than requiring the reader to mentally synthesize across Tables 1, 5, and 6.

---

## Overall Score: 28/49 mapped to scale => 20/35

| Criterion       | Score |
|-----------------|-------|
| Novelty         | 5/7   |
| Significance    | 2/7   |
| Clarity         | 5/7   |
| Correctness     | 3/7   |
| Reproducibility | 4/7   |
| Related Work    | 5/7   |
| Presentation    | 4/7   |
| **Total**       | **28/49 => 20/35** |

**Scaled to 35-point maximum:** Using the raw sum directly on the 1--7 scale, the total is **28/49**. Normalizing to a 35-point scale: **(28/49) x 35 = 20/35**.

---

## Strengths

1. **Honest and transparent reporting.** The paper does not overclaim. It states 0% exact match in the abstract, provides detailed failure mode analysis, and explicitly characterizes the results as testing the "floor imposed by resource constraints, not the ceiling of the architecture." This level of scientific honesty is rare and commendable.

2. **Novel architectural combination.** The integration of six components -- masked diffusion, dual-axis RoPE adapted for expression tree depth, soft-mask refinement, LoRA-based test-time finetuning, physics-informed losses, and skeleton-first structure prediction -- represents a genuinely new framework for symbolic regression that no prior work has explored.

3. **Structured ablation study with clear component ranking.** Despite performance limitations, the ablation study provides a useful ranking of component importance (dual-axis RoPE > structure predictor > physics losses > token algebra > soft masking > refinement > TTF), offering actionable guidance for future work on which inductive biases to prioritize.

4. **Meaningful embedding analysis.** The embedding space exhibits physically sensible structure: sin/cos grouping (midpoint similarity 0.69), arithmetic inverse-operation analogies (similarity 0.59--0.63), and kinematic derivative-chain relationships (velocity-acceleration analogy at 0.65). This demonstrates that the training objective produces representations with genuine semantic content even at minimal scale.

5. **Rigorous statistical methodology.** The paper employs paired bootstrap confidence intervals (1000 resamples, 95% CI), Wilcoxon signed-rank tests, and Cohen's d effect sizes for all comparisons. All six metrics show statistically significant differences (p < 0.05) between PhysMDT and the AR baseline, with medium to large effect sizes (|d| = 0.56--1.24).

---

## Weaknesses

1. **Empirical results are categorically non-competitive.** PhysMDT achieves 0% exact match and 0% symbolic equivalence on all benchmarks (AI Feynman, Nguyen, Strogatz, internal test set). The composite score of 1.52 out of 100 is orders of magnitude below the weakest published method (NeSymReS at 30% EM on AI Feynman). Even the internal AR baseline outperforms by 17.6x on composite score. Without any successful equation recovery, it is difficult to argue the architecture "works" in any practical sense.

2. **Ablation study is mostly projected, not measured.** Six of seven ablation variants are estimated projections rather than actual re-training experiments. Only the full model and the no-refinement variant were directly evaluated. The paper's central claim -- that dual-axis RoPE and the structure predictor are the most impactful components -- rests on projections whose methodology is not fully described, making this claim unverifiable.

3. **Unfair baseline comparison.** The AR baseline has 2.8x more parameters (1.18M vs. 420K) than PhysMDT. A fair comparison would require equal parameter budgets, equal training compute, and equal data. The paper acknowledges this but does not provide a controlled experiment. This confound weakens the conclusion that masked diffusion is inherently harder to train than autoregressive modeling at small scale; it might simply be a capacity difference.

4. **Ablation differences are not tested for statistical significance.** While the paper correctly applies statistical tests to the PhysMDT vs. AR baseline comparison, the ablation deltas (ranging from 0.061 to 0.274 CS on a base of 1.524) are not accompanied by confidence intervals or significance tests. Given that the full model's CS is only 1.52, these small absolute differences might not be statistically distinguishable from noise.

5. **No demonstration at adequate scale.** The paper argues that its "primary contribution is the architecture and approach, not raw performance," but this claim remains untestable because no experiment at adequate scale (d_model >= 256, 50K+ samples, GPU training) was conducted. Without even a partial scaling experiment, it is impossible to know whether the proposed components would help, hurt, or be irrelevant at competitive scale.

---

## Required Revisions

1. **Conduct at least one controlled ablation with full re-training.** The projected ablation estimates are the paper's weakest methodological element. At minimum, re-train and directly evaluate the top-2 most impactful ablations (no dual-axis RoPE, no structure predictor) to validate the projected rankings. Report statistical significance tests (bootstrap CIs or Wilcoxon) for the ablation differences, not just for the PhysMDT-vs-baseline comparison.

2. **Equalize the AR baseline comparison.** Either (a) train a 420K-parameter AR baseline with the same architecture dimensions (d_model=64, 4 layers) so that the only difference is the training objective (masked diffusion vs. autoregressive), or (b) train a 1.18M-parameter PhysMDT model. Without parameter-controlled baselines, the masked-diffusion-vs-autoregressive comparison is confounded and the hypothesis test (H1) is inconclusive rather than refuted.

3. **Describe the projection methodology for estimated ablations.** The paper must explain exactly how the six "estimated" ablation variants were computed. Were component-specific loss terms zeroed out? Were attention masks modified? Were pretrained weights re-used with specific modules disabled? Without this information, the ablation results are not reproducible and their validity cannot be assessed.

4. **Add a small-scale scaling experiment.** Even within CPU constraints, demonstrate how performance changes when increasing d_model from 64 to 128 or training data from 4K to 8K. A two-point scaling curve would provide evidence for or against the hypothesis that the architecture benefits from scale, and would substantially strengthen the "architecture contribution" framing.

5. **Clarify the composite score weighting rationale.** Justify the specific weights in the composite score formula (0.3 EM + 0.3 SE + 0.25 R2 + 0.1 (1-TED) + 0.05 (1-CP)). Since EM and SE receive the largest weights and PhysMDT scores 0 on both, the composite score is dominated by these terms. Discuss whether an alternative weighting (e.g., emphasizing structural similarity for models at the pre-competitive stage) would provide more informative evaluation.

---

## Detailed Assessment

### 1. Whether Claims Match Evidence

The paper's claims are carefully calibrated to the evidence, which is its greatest strength. The abstract explicitly states "0% exact match and 0% symbolic equivalence across all benchmarks" and "does not recover any test equation correctly." The paper frames its contribution as the "architecture and approach, not raw performance" -- a framing that is consistent with the results but difficult for reviewers to evaluate without a scaling experiment.

Three specific claim-evidence alignments deserve scrutiny:

- **Claim:** "Dual-axis RoPE is the most impactful component (-0.274 CS)." **Evidence:** This is based on a projected estimate, not a directly evaluated ablation. The claim should be softened to "estimated to be" rather than stated as fact.
- **Claim:** "Physics losses provide meaningful inductive bias even at minimal scale." **Evidence:** The physics loss ablation contributes 0.153 CS on a base of 1.524 (10% relative improvement). However, this is also a projected estimate, and given that 0% equations are recovered correctly, "meaningful" is subjective. The improvement manifests in reduced complexity penalty (more compact outputs), not in improved symbolic correctness.
- **Claim:** "The learned token embeddings exhibit meaningful structure." **Evidence:** This claim is well-supported. The sin/cos midpoint analysis (similarity 0.69), arithmetic analogies (0.59--0.63), and kinematic analogies (0.65) are directly computed and represent genuine semantic structure in the embedding space.

Overall assessment: Claims are conservative and mostly match evidence, but ablation-related claims should be qualified as projections.

### 2. Whether Ablation Results Are Statistically Significant

**The ablation results are not tested for statistical significance.** This is a critical gap. The paper reports:

| Component Removed | Delta CS |
|-------------------|----------|
| Dual-axis RoPE    | -0.274   |
| Structure Predictor| -0.229  |
| Physics Losses    | -0.153   |
| Token Algebra     | -0.107   |
| Soft Masking      | -0.076   |
| Refinement        | -0.067   |
| TTF               | -0.061   |

These deltas range from 0.061 to 0.274 on a base composite score of 1.524. Without confidence intervals or p-values, it is impossible to determine whether these differences are statistically distinguishable from zero or from each other. The refinement depth study (Table 5) reports zero standard deviation across 3 seeds, which suggests deterministic behavior but also raises the question of whether any seed-based variation was captured.

Furthermore, six of seven variants are projected rather than directly evaluated, meaning there are no per-equation paired scores to compute bootstrap CIs even if the authors wished to. Only the refinement ablation (CS = 1.457 vs. full CS = 1.524, delta = 0.067) was directly evaluated, and even this single comparison lacks a significance test.

**Verdict:** The ablation differences should be treated as indicative orderings, not as statistically validated findings. The paper partially acknowledges this ("component rankings should be treated as indicative orderings, not precise measurements") but the ablation table presents the numbers with a precision that suggests more confidence than is warranted.

### 3. Whether Comparison to SOTA Is Fair

The comparison to SOTA is presented honestly but is fundamentally unfair in resource terms, which the paper acknowledges:

| Factor | PhysMDT | Published SOTA (typical) |
|--------|---------|--------------------------|
| Parameters | 420K | 10M--100M+ |
| d_model | 64 | 256--512 |
| Layers | 4 | 6--12 |
| Training data | 4K samples | 50K--500K |
| Training compute | CPU, 15 epochs | GPU, 100s of epochs |
| Training time | Minutes | Hours to days |

The paper places PhysMDT (0% EM) alongside QDSR (91.6% EM) in Table 6 without controlling for any of these resource differences. While the paper acknowledges the disparity in the Limitations section, the benchmark table as presented may give the impression that masked diffusion is fundamentally incapable of symbolic regression, when in reality the comparison is between a minimally-resourced prototype and fully-optimized systems.

The internal baseline comparison is also confounded: the AR baseline has 2.8x more parameters (1.18M vs. 420K). The paper attributes the performance gap to the inherent difficulty of masked diffusion training, but the parameter gap is an equally plausible explanation.

**Verdict:** The comparison is transparently presented and the limitations are discussed, but it is not a fair evaluation of the masked diffusion approach. The paper should more prominently caveat that the benchmark comparison reflects resource constraints, not architectural limits.

### 4. Whether Limitations Are Honestly Discussed

The limitations section is one of the paper's strongest elements. It explicitly enumerates five limitations:

1. CPU-only constraints as the binding bottleneck
2. Most ablation variants are estimated, not measured
3. No out-of-distribution evaluation
4. No hyperparameter tuning
5. Unfair parameter budget in the AR comparison

This is a thorough and honest accounting. The paper correctly characterizes the results as testing the "floor imposed by resource constraints, not the ceiling of the architecture." The Discussion section provides detailed failure mode analysis (wrong structure 20/20, excessive length 20/20, operator bias 16/20, repetitive outputs ~15/20) that does not shy away from the severity of the failures.

Two additional limitations could be mentioned:

- The composite score formula itself may not be the most informative metric for a model at this performance level, since it is dominated by the 0-valued EM and SE terms.
- The training loss reduction (2.8 to 0.17) suggests memorization of the small training set, which is noted in the Discussion but could be explored more formally (e.g., by reporting train vs. test loss divergence).

**Verdict:** Limitations are discussed with exemplary honesty. The paper sets a good standard for transparent reporting of negative results.

---

## Recommendation

**Weak Reject** for a top ML venue in its current form. The architectural ideas are interesting and the honest framing is appreciated, but the lack of any successful equation recovery, the reliance on projected ablation estimates, and the absence of even a minimal scaling experiment make the empirical contribution insufficient for acceptance. The paper would be substantially strengthened by: (1) directly evaluating the top ablation variants, (2) conducting a controlled parameter-matched baseline comparison, and (3) providing a small scaling experiment showing performance trends as capacity increases.

If the authors can demonstrate competitive or meaningfully improved performance at moderate scale (even a fraction of published SOTA), the architectural contributions and ablation insights would become significantly more compelling.

---

## Summary Table

| Criterion       | Score | Key Issue |
|-----------------|-------|-----------|
| Novelty         | 5/7   | Integration is novel; individual components are adapted from prior work |
| Significance    | 2/7   | 0% EM/SE on all benchmarks; ablation insights limited by low absolute performance |
| Clarity         | 5/7   | Well-written and honest; ablation projection method needs explanation |
| Correctness     | 3/7   | Math is sound but ablation claims rest on unverified projections |
| Reproducibility | 4/7   | Architecture is reproducible; projected ablations are not |
| Related Work    | 5/7   | Comprehensive; could better differentiate from DiffuSR/DDSR |
| Presentation    | 4/7   | Good tables; needs performance landscape summary figure |
| **Total**       | **28/49 (20/35)** | |
