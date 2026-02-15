# Peer Review: Physics-Aware Recursive Refinement (PARR) Transformer

**Reviewer:** Automated Peer Review (Nature/NeurIPS standard)
**Date:** 2026-02-15
**Paper:** "Physics-Aware Recursive Refinement: A Transformer Architecture with Iterative Masked-Diffusion Decoding for Autonomous Newtonian Equation Discovery"

---

## Overall Verdict: REVISE

---

## Criterion Scores

| Criterion | Score (1-5) | Summary |
|---|---|---|
| 1. Completeness | 5 | All required sections present, well-structured |
| 2. Technical Rigor | 4 | Methods well-described with equations; training procedure clear |
| 3. Results Integrity | 3 | Numbers match data files, but critical concerns about ablation results |
| 4. Citation Accuracy | 2 | 1 citation has incorrect authors and title; BibTeX formatting errors |
| 5. Compilation | 4 | PDF compiles and is well-formatted |
| 6. Writing Quality | 5 | Excellent academic prose, honest discussion of limitations |
| 7. Figure Quality | 3 | Architecture figure is publication-quality; data plots need improvement |

---

## Detailed Review

### 1. Completeness (5/5)

All required sections are present and substantive:
- Abstract (clear, quantitative headline results)
- Introduction with gap analysis and contributions
- Related Work (3 well-organized paragraphs covering symbolic regression, physics-informed discovery, masked diffusion)
- Background & Preliminaries (problem formulation, notation table)
- Method (detailed architecture description with 5 subsections)
- Experimental Setup (dataset, baselines, metrics, hyperparameters)
- Results (main results, efficiency, ablation, robustness, training dynamics, comparison, qualitative analysis)
- Discussion (implications, "refinement paradox" analysis, limitations)
- Conclusion with future work

The paper is well above the 8-page minimum (approximately 10+ pages excluding references). The algorithm box (Algorithm 1) is a welcome addition for reproducibility.

### 2. Technical Rigor (4/5)

**Strengths:**
- The PARR architecture is described precisely with equations for each component (Eqs. 1-10).
- The training procedure (two-phase: AR-only then joint AR+Refinement) is well-motivated and clearly specified.
- The ConvSwiGLU FFN, token algebra, and TBPTL are all formally defined with equations.
- Hyperparameter table (Table 1) is comprehensive and enables reproducibility.
- Algorithm 1 provides a clear procedural description.

**Weaknesses:**
- The paper claims to be "inspired by masked diffusion" but the actual inference procedure (Section 4.6) is purely AR draft + argmax refinement. The connection to masked diffusion is tenuous -- LLaDA starts from fully masked sequences and progressively unmasks, while PARR starts from a complete AR draft and refines. This distinction should be made more explicit.
- The token algebra equation (Eq. 8) adds a mask embedding scaled by `(1-s)` which diminishes to zero as refinement progresses. The theoretical justification for this specific functional form is thin.
- No formal analysis of the TBPTL approximation quality (e.g., gradient norm comparison with full backprop).
- The paper specifies several novel components in the rubric (physics-informed positional encodings, dimensional attention masks, test-time adaptation, curriculum learning, self-verification) that are implemented in the codebase but **not actually used or evaluated** in the final model. The paper correctly does not claim these, but it means significant planned novelty was not delivered.

### 3. Results Integrity (3/5)

**Verified claims (numbers match `results/` data files):**

- **Table 2 (Main Results):** PARR ESM values (T1=79.0%, T2=92.5%, T3=85.4%, T4=74.1%, Overall=84.0%) match `parr_results.json` exactly. Baseline values (T1=84.7%, T2=94.7%, T3=89.8%, T4=78.0%, Overall=88.0%) match `comparison_results.json`. R-squared and NTED values also match. **VERIFIED.**

- **Table 3 (Efficiency):** Baseline 12.7ms/eq, 78.5 eq/s; PARR K=8: 4.4ms/eq, 229.0 eq/s. All match `efficiency_results.json`. **VERIFIED.**

- **Table 4 (Robustness):** All noise-level ESM values match `robustness_results.json` (after appropriate rounding). **VERIFIED.**

- **Table 5 (Qualitative Examples):** Examples match `qualitative_analysis.json`. **VERIFIED.**

**Critical concerns:**

1. **Ablation study is scientifically vacuous.** The `ablation_study.json` file shows that K=0 (AR-only), K=2, K=4, and K=8 produce **byte-for-byte identical results** across all metrics (ESM=83.96%, R^2=0.8976, NTED=0.0393 for every K value, every tier). This means the refinement mechanism has literally zero effect on predictions -- not even at the token level. The paper's framing of this as a "refinement paradox" and attributing it to SymPy normalization absorbing "surface-level token differences" is misleading. The actual data shows there ARE no token-level differences -- AR-only and refined outputs are identical. This is confirmed by `qualitative_analysis.json` where `refinement_helped: 0` out of 20 examples and every AR-only prediction is character-identical to the refined prediction.

2. **The "2.9x inference speedup" claim is the paper's central efficiency result, but its origin is unclear.** If refinement does nothing (K=0 and K=8 give identical outputs), then the speedup must come from PARR's architectural differences vs. the baseline (shared-weight decoder vs. 6 stacked decoder layers), not from "parallel refinement." The paper conflates architectural efficiency with refinement-based parallelism.

3. **The baseline outperforms PARR by 4.0 pp on ESM.** The paper attributes this to "unequal training compute" but the baseline trained for only 15 epochs (~1 hour) while PARR trained for 14 epochs (~1.6 hours). PARR actually used MORE wall-clock training time. The real issue is that PARR's Phase 2 reduced learning rate (1e-4 vs 3e-4) and the refinement loss may have interfered with continued AR improvement.

4. **Minor discrepancy:** `baseline_results.json` (from direct baseline evaluation) shows slightly different numbers (T1=83.6%, T2=95.4%, Overall=88.2%) than `comparison_results.json` (T1=84.7%, T2=94.7%, Overall=88.0%). The paper uses the comparison_results.json values. This suggests two different evaluation runs with slightly different results, which is concerning for reproducibility. The paper should clarify which evaluation run is canonical and why they differ.

5. **Training curves figure (Figure 5) has a plotting bug.** The "Val ESM (AR)" line during Phase 1 appears stuck at ~0.0 for all Phase 1 steps. According to the rubric notes, AR-only training reached 77.4% ESM by end of Phase 1, so the validation ESM should be plotted and visible during Phase 1 as well.

### 4. Citation Accuracy (2/5)

**Citation Verification Report (Web-search verified):**

| # | Citation Key | Verdict | Details |
|---|---|---|---|
| 1 | `kamienny2022end` | VERIFIED | Title, authors, NeurIPS 2022, arXiv:2204.10532 all correct |
| 2 | `shojaee2023tpsr` | VERIFIED | Title, authors, NeurIPS 2023, arXiv:2303.06833 all correct |
| 3 | `valipour2021symbolicgpt` | VERIFIED | Title, authors, arXiv:2106.14131 all correct |
| 4 | `biggio2021neural` | VERIFIED | Title, authors, ICML 2021, arXiv:2106.06427 all correct |
| 5 | `dascoli2024odeformer` | VERIFIED | Title, authors, ICLR 2024, arXiv:2310.05573 all correct |
| 6 | `fang2025ainewton` | VERIFIED | Title, authors, arXiv:2504.01538 all correct |
| 7 | `ying2025phye2e` | VERIFIED | Title, authors, Nature Machine Intelligence vol. 7, arXiv:2503.07994 all correct |
| 8 | `udrescu2020aifeynman` | VERIFIED | Title, authors, Science Advances 6(16), arXiv:1905.11481 all correct |
| 9 | `cranmer2020discovering` | VERIFIED | Title, authors, NeurIPS 2020, arXiv:2006.11287 all correct |
| 10 | `dehghani2019universal` | VERIFIED | Title, authors, ICLR 2019, arXiv:1807.03819 all correct |
| 11 | `gao2025urm` | VERIFIED | Title, authors, arXiv:2512.14693 all correct |
| 12 | `liao2025compressarc` | VERIFIED | Title, authors, arXiv:2512.06104 all correct |
| 13 | `nie2025llada` | VERIFIED | Title, authors, NeurIPS 2025 Oral, arXiv:2502.09992 all correct |
| 14 | `lample2020deep` | MINOR ISSUE | Paper is correct (ICLR 2020, arXiv:1912.01412), but BibTeX entry uses `@article` type with a `booktitle` field -- should be `@inproceedings`. May cause rendering issues with some bibliography styles. |
| 15 | `makke2024interpretable` | MINOR ISSUE | Paper is correct (AI Review, DOI matches). BibTeX lists `number={2}` but the paper is Volume 57, Issue 1, Article number 2. Minor metadata error. |
| 16 | `la2024lasr` | **INCORRECT -- MUST FIX** | **Wrong authors and wrong title.** The BibTeX lists author "La Cava, William and others" but William La Cava is NOT an author of this paper. The actual authors are **Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, and Swarat Chaudhuri**. The actual title is **"Symbolic Regression with a Learned Concept Library"** (not "LaSR: Large Language Model Assisted Symbolic Regression"). The paper does exist at NeurIPS 2024 (arXiv:2409.09359), but the citation metadata is fabricated/hallucinated. |
| 17 | `tenachi2023physo` | VERIFIED | Title (full title in bib is correct), authors, ApJ 959(2), arXiv:2303.03192 all correct |

**Summary:** 13 fully verified, 2 minor issues, **1 citation with fabricated authors and incorrect title** (`la2024lasr`). This is a disqualifying error for a venue requiring citation accuracy.

### 5. Compilation (4/5)

- The PDF exists (1.65 MB) and appears well-formatted.
- LaTeX compiles with standard pdflatex + bibtex toolchain.
- All figures are properly referenced and included.
- Tables are well-formatted with booktabs.
- Minor: The `@article` / `booktitle` mismatch in `lample2020deep` may produce a warning or incorrect rendering depending on the natbib/bibtex backend.

### 6. Writing Quality (5/5)

This is the strongest aspect of the paper. The writing is:
- Professional academic tone throughout
- Honestly acknowledges that the baseline outperforms PARR on ESM
- The "refinement paradox" discussion (Section 7.2) is intellectually honest and insightful
- Limitations section is thorough (5 specific limitations including closed-world evaluation, constant quantization, scale constraints)
- Related work is well-organized and substantive with meaningful comparisons
- The paper flows logically from motivation to method to results to analysis
- Notation is consistent and clearly defined

### 7. Figure Quality (3/5)

**Publication-quality figures:**
- `parr_architecture.png` (Figure 1): Excellent. Clear, well-designed architectural diagram with color-coded components, proper labels, and informative layout. Publication-ready.
- `tier_radar.png` (Figure 6): Good quality radar chart with proper legends, clear labels, and meaningful visual comparison.

**Figures needing improvement:**
- `comparison_bar.png` (Figure 4): Uses only 2 colors (blue/orange) with default-looking matplotlib styling. The chart is very wide and short (aspect ratio issue). Value annotations are crowded. Needs: better aspect ratio, grid lines, confidence interval error bars, and more professional color palette.
- `ablation_refinement.png` (Figure 3): All lines are near-identical shades of blue, making them very hard to distinguish. The monochrome blue palette is not colorblind-friendly. Since all K values give identical results, the plot is essentially 5 overlapping horizontal lines -- this is not informative and arguably misleading (it looks like there might be differences when there are none). Consider a different visualization.
- `robustness_curve.png` (Figure 2): Same monochrome blue palette issue. Lines for different tiers are hard to distinguish. Needs distinct colors/markers, error bars from multiple seeds, and a more distinctive color scheme.
- `training_curves.png` (Figure 5): Has the plotting bug described above (Val ESM stuck at 0 during Phase 1). The dual y-axis is acceptable but the loss scale and ESM scale are not clearly differentiated. Baseline training curves are mentioned in the caption but NOT shown in the figure -- only PARR training curves appear.

---

## Summary of Required Revisions

### Critical (must fix before acceptance):

1. **Fix `la2024lasr` citation.** Replace fabricated author "La Cava, William" with actual authors "Grayeli, Arya and Sehgal, Atharva and Costilla-Reyes, Omar and Cranmer, Miles and Chaudhuri, Swarat". Replace title with "Symbolic Regression with a Learned Concept Library". Add arXiv:2409.09359. Verify that all in-text references to LaSR (Section 2, paragraph 2) accurately describe the paper's actual method.

2. **Address the ablation study honestly.** The ablation data shows K=0 and K=8 produce byte-identical outputs (not just symbolically equivalent -- actually identical token sequences). The "refinement paradox" framing is partially misleading: refinement doesn't produce "surface-level token differences absorbed by SymPy" -- it produces no differences at all. The paper should:
   - Report token-level match rates between K=0 and K=8 outputs (which appear to be 100%)
   - Acknowledge that the refinement mechanism is not functioning as intended during inference
   - Reframe the contribution: PARR's value is in its shared-weight architecture (parameter efficiency + inference speed) rather than in iterative refinement per se
   - Investigate why refinement has no effect: is the argmax at each step selecting the same tokens the AR draft produced? Is the refinement block not learning meaningful corrections?

3. **Fix the training curves figure.** The Val ESM (AR) line should show non-zero values during Phase 1 (the model reaches 77.4% ESM by end of Phase 1 per training logs). Also add the baseline training curves as described in the caption.

### Important (strongly recommended):

4. **Clarify the speedup claim.** The 2.9x speedup comes from PARR's architectural differences (shared-weight decoder vs. 6 stacked layers), not from "parallel refinement replacing sequential decoding." Since K=0 gives identical results to K=8, the speedup at K=0 (4.9x) is the more honest comparison point. Reframe the efficiency discussion.

5. **Address the baseline/PARR compute comparison fairly.** The paper says PARR's lower accuracy is due to "unequal training compute" but PARR actually trained for more wall-clock time (1.6h vs 1.0h). The real explanation is the two-phase training schedule with reduced Phase 2 learning rate. Be precise about this.

6. **Fix minor BibTeX issues:**
   - `lample2020deep`: Change `@article` to `@inproceedings` and `booktitle` to proper format
   - `makke2024interpretable`: Fix `number={2}` to `number={1}` (it's Volume 57, Issue 1, Article 2)

7. **Improve data visualization figures:**
   - Use distinct color palettes (not all-blue monochrome) for robustness and ablation plots
   - Add error bars/confidence intervals where applicable
   - Fix comparison_bar.png aspect ratio
   - Make figures colorblind-friendly (use both color and marker/linestyle differentiation)

### Minor (nice to have):

8. **Reconcile the two baseline evaluation runs.** Explain why `baseline_results.json` (ESM=88.2%) and `comparison_results.json` baseline (ESM=88.0%) differ, and specify which is canonical.

9. **Table 6 (comparison with prior work):** The accuracy ranges for prior methods (SymbolicGPT "25-40%", E2E Transformer "15-45%") are vague. Either cite specific numbers from the original papers with the corresponding benchmark, or note that direct comparison is not possible and remove the accuracy column.

10. **The paper does not evaluate several implemented components:** Physics-informed positional encodings, dimensional attention masks, test-time adaptation, curriculum learning, and self-verification are all implemented in the codebase but not used in the final model or ablated. A brief mention in the Limitations or Future Work section would be appropriate.

---

## Verdict Justification

The paper demonstrates solid engineering work, strong writing quality, and an intellectually honest discussion of its own limitations. However, three issues prevent acceptance:

1. **A fabricated citation** (`la2024lasr` with wrong authors and title) is a disqualifying error.
2. **The core novelty (iterative refinement) demonstrably has zero effect** on model outputs, yet the paper frames this as a nuanced "paradox" rather than acknowledging it as a negative result. The contribution needs reframing around the architecture's efficiency properties rather than refinement capabilities.
3. **Several figures are below publication quality** (monochrome palettes, plotting bugs, missing data).

With these revisions, the paper would be a solid contribution to the symbolic regression literature, presenting an efficient shared-weight transformer architecture for physics equation discovery with honest analysis of what works and what doesn't.
