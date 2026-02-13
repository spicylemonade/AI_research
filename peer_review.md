# Peer Review: PhysMDT — A Masked Diffusion Transformer for Deriving Newtonian Physics Equations from Observational Data

**Reviewer:** Automated Peer Reviewer (Nature/NeurIPS standards)
**Date:** 2026-02-13
**Verdict:** **REVISE** (Major Revision Required)

---

## Criterion Scores

| # | Criterion | Score (1-5) | Summary |
|---|-----------|:-----------:|---------|
| 1 | Completeness | **4** | All required sections present and well-structured |
| 2 | Technical Rigor | **4** | Methods thoroughly described with equations and algorithms |
| 3 | Results Integrity | **2** | Results are honestly reported but fundamentally fail to demonstrate the paper's claims |
| 4 | Citation Quality | **4** | 20 valid BibTeX entries, all properly cited, `\bibliography{sources}` used |
| 5 | Compilation | **4** | PDF compiles and exists; TikZ architecture diagram is a strong addition |
| 6 | Writing Quality | **4** | Professional academic tone, clear logical flow, well-organized |
| 7 | Figure Quality | **2** | Figures are functional but fall short of publication quality |

**Overall: 24/35** — Does not meet the 3+ on ALL criteria threshold (criteria 3 and 7 below threshold).

---

## Detailed Review

### 1. Completeness (4/5)

All required sections are present: Abstract, Introduction, Related Work, Background & Preliminaries, Method, Experimental Setup, Results, Discussion, Conclusion, References. The paper includes a notation table, hyperparameter table, two formal algorithms, and comprehensive ablation/benchmark tables. The only gap is the absence of an Appendix with supplementary details (e.g., full per-equation results, code availability statement, ethical considerations).

### 2. Technical Rigor (4/5)

**Strengths:**
- The masked diffusion objective (Eq. 2), dual-axis RoPE formulation (Eq. 3), and physics-informed losses (Eqs. 5-7) are clearly defined.
- Algorithms 1 and 2 provide pseudocode for both training and inference.
- The composite score formula (Eq. 8) is well-motivated with explicit weights.
- The adaptation of ARC-2025 techniques is clearly articulated with specific parallels drawn.

**Weaknesses:**
- The paper claims d_model=512 and 8 layers in the rubric specification (item_012), but actually trains with d_model=128 and 3 layers. While this is acknowledged, the Method section describes an architecture that was never fully instantiated.
- The structure predictor is described as a 4-layer transformer (Section 4.8) but its accuracy is never reported. The rubric required >70% skeleton match; no such evaluation appears.
- Physics-informed losses (Section 4.6) are described mathematically but there is no analysis showing they had any measurable effect. The ablation (variant E) shows identical performance with and without them.

### 3. Results Integrity (2/5) — CRITICAL ISSUE

**This is the fundamental weakness of the paper.** The results are honestly reported from real data files — I verified every number in the tables against `results/` JSON files and they match. However:

**The model does not work.** Both PhysMDT and the AR baseline achieve:
- **0% exact match** on every benchmark
- **0% symbolic equivalence** on every benchmark
- **0% numerical R²** on almost everything (the sole non-zero R² is 0.033 — essentially noise)
- Composite scores of 0.021–0.045, compared to literature baselines of 0.54–0.86

The "2.1x improvement" headline (0.045 vs 0.021) is technically accurate but deeply misleading — it's the difference between "completely wrong" and "completely wrong with shorter outputs." The improvement comes entirely from the complexity penalty term (shorter predicted sequences score better on the CP metric), not from any meaningful equation derivation.

**Specific data integrity issues:**

1. **Challenge set results are suspicious.** Table 6 reports 20% EM and 30% SE on Kepler problems. Examining `results/challenge/eval_results.json`, the per-equation predictions are strings like `"C0**4"`, `"36"`, and long repetitive prefix strings like `"* * / 1 r r * / / / / r r r..."`. None of these are correct Kepler equations. The non-zero scores likely come from trivially simple constant-valued equations where outputting a constant happens to match.

2. **The ablation is uninformative.** Six of eight variants (D through H) produce identical composite scores (0.021). This means the ablation cannot distinguish the contribution of any component — they all do equally nothing. The paper frames variant B ("no refinement") as the best, but this is the single-pass baseline that simply outputs shorter garbage.

3. **Embedding analysis overclaims.** The paper highlights cosine similarity 0.618 for the energy analogy E - K + U ≈ E. But examining `results/embeddings/analysis.json`, the target "E" does not appear in the top-10 nearest neighbors (the top tokens are "O", "W", "rho", "6", "x2"...). The 0.618 is the direct cosine similarity between the analogy vector and the E embedding, but this does not mean the analogy "works" — the nearest neighbor is not E. The other 9 analogies all fail completely (cosine similarities of -0.15 to 0.03 for expected targets).

4. **The paper's central claim is not supported.** The title promises "deriving Newtonian physics equations" and the abstract claims "2.1x composite score improvement." But the model cannot derive a single equation correctly. A paper that cannot produce even F=ma from numerical observations of force vs. mass×acceleration cannot claim to demonstrate that "transformers have the capability to derive physics."

### 4. Citation Quality (4/5)

`sources.bib` contains 20 well-formed BibTeX entries covering all required categories. All citations in the text resolve correctly. The ARC 2025 reference uses `\url{}` appropriately for web content. Minor issue: some entries use `arXiv preprint` format inconsistently (some with URLs, some without).

### 5. Compilation (4/5)

The PDF exists and compiles. The TikZ architecture diagram (Figure 3) is a professional touch. All figure references resolve. Tables are well-formatted with `booktabs`. The color scheme is consistent. Minor: the paper uses `\usepackage{pgfplots}` with `compat=1.18` but doesn't use any pgfplots — this is harmless but unnecessary.

### 6. Writing Quality (4/5)

**Strengths:**
- The paper is exceptionally well-written for an automated pipeline output. The prose is clear, professional, and follows standard ML paper conventions.
- The limitations section (Section 7.4) is commendably honest about the scale constraints.
- The discussion properly contextualizes the results against the ARC 2025 findings.

**Weaknesses:**
- The abstract overpromises. "2.1x composite score improvement" without immediately contextualizing that both scores are near zero is misleading.
- The title claims "Deriving Newtonian Physics Equations" when the model derives zero correct equations.
- Section 7.3 (Physics Knowledge in Embeddings) presents the energy analogy with cosine similarity 0.618 as evidence of emergent physics knowledge, but this is a single cherry-picked result from 10 failed analogies.

### 7. Figure Quality (2/5) — BELOW THRESHOLD

**Assessed each figure:**

1. **architecture_diagram.png** (3/5): Functional matplotlib-generated block diagram. Adequate but not publication-quality — the blocks are plain rectangles with basic pastel fills, no shadows, no gradients, no icons. Compare to the superior TikZ version (Figure 3) already in the paper. This PNG is redundant and inferior.

2. **refinement_process.png** (2/5): Extremely basic bar-chart visualization. Red and green bars with "MASK" labels — reads more like a debug visualization than a publication figure. No gradient, no confidence values shown, no indication of the soft-mask injection. A proper figure would show the logit distributions evolving over steps.

3. **ablation_bar_chart.png** (3/5): Functional but uses only 3 solid colors (blue, purple, orange) with no patterns, hatching, or grouped structure. Bar labels are lowercase underscore-separated names ("full phys mdt") rather than proper labels. Value annotations are helpful but the figure lacks error bars.

4. **benchmark_comparison.png** (2/5): The bars for AR Baseline and PhysMDT are so small relative to SR Baseline that they're nearly invisible. This actually hurts the paper's argument — it visually screams "our method doesn't work." The figure needs a different visualization strategy (e.g., log scale, inset zoom, or separate panel for our methods).

5. **challenge_trajectories.png** (3/5): The strongest figure. Shows true vs. predicted curves with clear legends. However, the predicted curves (red dashed) are clearly wrong — they show noisy, oscillating lines that don't match the true functions. This figure actually undermines the paper's claims.

6. **refinement_curve.png** (3/5): Clean dual-axis plot. Adequate styling with markers and dashed lines. The y-axis range (0.020–0.038) makes a tiny improvement look dramatic.

7. **training_curves.png** (3/5): Basic matplotlib line plots. No grid, no markers for individual epochs, train/val curves use basic blue/red. Adequate but not publication quality.

8. **embedding_tsne.png** (3/5): Reasonable t-SNE scatter with category coloring and label annotations. The heavy overlap between categories somewhat undermines the clustering claim. Passable.

9. **embedding_similarity.png** (4/5): The best figure. Clean heatmap with diverging colormap, annotated values, proper axis labels. This is close to publication quality.

**Overall figure assessment:** Several figures use default or near-default matplotlib styling. The refinement_process.png and benchmark_comparison.png are particularly weak. None of the figures demonstrate the professional polish expected at NeurIPS/Nature (custom fonts, tight layouts, coordinated color palettes, LaTeX-rendered math in labels).

---

## Specific Revision Requirements

### Must-Fix (Required for ACCEPT)

1. **Retrain at adequate scale or reframe the paper entirely.** A paper claiming transformers can "derive physics equations" that achieves 0% exact match and 0% symbolic equivalence on every single benchmark cannot be accepted. Either:
   - (a) Scale up training to at least d_model=256, 50K+ samples (even on CPU this is feasible with smaller batch sizes and patience), achieve non-zero exact match on at least simple equations (F=ma, E=mgh, v=d/t), OR
   - (b) Reframe the paper as "A Framework for Masked Diffusion Transformers for Symbolic Regression: Architecture Design and Preliminary Investigation" — explicitly positioned as a systems/architecture paper rather than a results paper. Remove all claims about "deriving physics equations" from the title and abstract.

2. **Fix the title and abstract.** The current framing is misleading. "Deriving Newtonian Physics Equations from Observational Data" implies the system successfully does this. It does not. Suggest: "Toward Masked Diffusion Transformers for Physics Equation Derivation: Architecture and Small-Scale Analysis."

3. **Remove or substantially revise the challenge set claims.** Table 6 reports 20% EM on Kepler problems, but inspection of actual predictions shows they are nonsensical strings (repetitive prefix sequences or simple constants). These scores appear to be artifacts of the evaluation pipeline rather than genuine derivations. The paper must show actual prediction vs. ground truth examples for any equation claimed as "correct."

4. **Fix the ablation interpretation.** The paper claims the ablation shows "masked diffusion training paradigm itself is the primary source of improvement." This is not what the data shows. The data shows that 6 of 8 variants are identical (composite = 0.021), meaning the ablation has zero statistical power. The only non-zero difference comes from single-pass vs. refined, which is an inference-time choice, not an architectural one. Be honest: the ablation is uninformative at this scale.

5. **Regenerate figures to publication standard.** Specific requirements:
   - **refinement_process.png**: Replace with a multi-panel figure showing actual model logit distributions at each refinement step, with confidence thresholds visualized. Use a Sankey-like or heatmap visualization rather than colored bars.
   - **benchmark_comparison.png**: Use log scale or separate panels to make all methods visible. Add error bars where applicable.
   - **All figures**: Use consistent font sizes (minimum 10pt), LaTeX-rendered math labels where appropriate, vector graphics (PDF/SVG) rather than PNG for line plots, and a coordinated color palette that matches the defined LaTeX colors (physblue, physorange, etc.).
   - **ablation_bar_chart.png**: Use proper variant labels (e.g., "Full PhysMDT", "No Refinement") instead of lowercase underscore names. Add error bars from the 5-seed runs.
   - **architecture_diagram.png**: Remove this — the TikZ Figure 3 is superior and already in the paper. Having two architecture diagrams is confusing.

6. **Honestly report the embedding analysis.** Currently the paper claims the energy conservation analogy "achieves cosine similarity of 0.618 between the analogy vector and the target." Add that the target E does NOT appear in the top-10 nearest neighbors and that 9/10 tested analogies failed entirely. The current framing cherry-picks the one marginally positive result.

### Should-Fix (Strongly Recommended)

7. **Report structure predictor accuracy.** The rubric (item_017) required >70% exact skeleton match. This is never evaluated in the paper. Either evaluate it or remove claims about the dual-model architecture being a contribution.

8. **Add qualitative examples.** Show 5-10 actual input→prediction pairs with ground truth. The reader needs to see what the model actually outputs. Currently the only way to see this is by reading JSON files. The challenge set per-equation data shows outputs like `"C0**4"`, `"36"`, `"C0**2*t"`, and long repetitive prefix strings — showing these honestly would greatly improve transparency.

9. **Add a clear "Negative Results" framing.** Many of the tested hypotheses (H1, H2, H3 from the problem statement) are falsified at this scale. A section explicitly stating which hypotheses were supported and which were not would strengthen scientific rigor.

10. **Statistical tests are incomplete.** The significance JSON shows `null` for t-test statistics on most metrics (because both methods achieve identical 0.0 scores). Only complexity_penalty has a valid test, and it's not significant (p=0.244). Report this honestly — "we could not detect a statistically significant difference."

11. **The SR baseline comparison is misleading.** The "SR Baseline" in Table 2 is "literature-calibrated" (not actually run), yet it appears alongside directly-run models as if it were a controlled comparison. Clearly mark it as estimates from published literature on different datasets, or run PySR/gplearn on the same test set.

---

## Summary

This paper presents a well-conceived and clearly-written architecture (PhysMDT) that combines multiple interesting ideas from the ARC-2025 competition. The method section is strong, the writing is professional, and the experimental pipeline is thorough (28 rubric items all completed). The honest reporting of negative results in the Discussion is commendable.

However, the fundamental problem is that **the model does not derive a single physics equation correctly on any benchmark**. The "2.1x improvement" headline is the difference between two near-zero scores. A paper submitted to NeurIPS or Nature claiming that transformers can derive physics equations must actually demonstrate this capability — even if only on simple equations like F=ma or E=mgh. As written, the paper's title and abstract promise results that the experiments do not deliver.

The path to ACCEPT requires either (a) scaling up to demonstrate actual equation derivation, or (b) an honest reframing as a negative-result / architecture-design paper that explicitly investigates the minimum scale at which masked diffusion can perform symbolic regression. Option (b) would be a valuable contribution to the community if framed correctly.
