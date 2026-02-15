# Peer Review: PhysMDT — Physics Masked Diffusion Transformer for Autonomous Equation Discovery from Numerical Observations

**Reviewer:** Automated Peer Review Agent
**Date:** 2026-02-15
**Paper:** `research_paper.tex` / `research_paper.pdf`

---

## Criterion Scores (1–5)

| # | Criterion | Score | Summary |
|---|-----------|-------|---------|
| 1 | Completeness | **4** | All required sections present; some subsections could be expanded |
| 2 | Technical Rigor | **3** | Methods well-described with equations; reproducibility concerns around ablation and robustness evaluation |
| 3 | Results Integrity | **2** | Several mismatches between paper text and actual figures/data; misleading R² reporting; core contribution (refinement) underperforms ablated variant |
| 4 | Citation Accuracy | **4** | 15/16 citations verified correct; 1 minor author name error |
| 5 | Compilation | **4** | PDF compiles and is well-formatted; TikZ architecture diagram is a strong addition |
| 6 | Writing Quality | **4** | Professional academic tone; clear arguments; honest about limitations |
| 7 | Figure Quality | **3** | Most figures are publication-quality; refinement heatmaps are problematic (see details) |

**Overall Score: 3.4 / 5**

---

## Overall Verdict: **REVISE**

---

## Detailed Review

### 1. Completeness (4/5)

The paper includes all required sections: Abstract, Introduction, Related Work, Background & Preliminaries, Method, Experimental Setup, Results, Discussion (including limitations and comparison with prior art), Conclusion, References, and Appendix with qualitative visualizations.

**Strengths:**
- The Background section with a notation table is a welcome addition for readability.
- The Method section is well-structured with six subsections covering each component.
- The Discussion section honestly addresses limitations.

**Weaknesses:**
- Missing a dedicated "Ethics / Broader Impact" section beyond the single paragraph in the Conclusion.
- No statistical significance testing (bootstrap confidence intervals) despite the rubric requiring them (item_017). The paper states "500 test samples per tier" as a target but only uses 5 test samples per equation.

### 2. Technical Rigor (3/5)

**Strengths:**
- Mathematical formulations are clearly presented (Eqs. 1–5).
- Algorithm 1 for recursive soft-masking refinement is precise and reproducible.
- Hyperparameter tables are comprehensive.
- Training dynamics are well-documented.

**Weaknesses:**
- **Ablation methodology is flawed.** The "no_curriculum" ablation uses the same checkpoint trained WITH curriculum (acknowledged in the paper and data). This makes it impossible to measure curriculum contribution. This should either be removed or clearly labeled as a non-informative control.
- **Ablation was run in quick_mode** (n_test_samples=2, n_points=30, n_candidates=2, n_steps=8) rather than the full configuration (n_test_samples=5, n_points=100, n_candidates=8, n_steps=64). This severely limits the validity of the ablation conclusions, particularly the claim that "single-pass outperforms multi-step refinement." With only 8 refinement steps (vs. the 64 used in the main evaluation), the refinement procedure is not given a fair comparison.
- **Robustness evaluation uses only 2 test samples per equation** (quick_mode), making the noise robustness and data efficiency claims statistically weak.
- The rubric called for evaluation at noise levels {0%, 1%, 5%, 10%, 20%}, but only {0%, 5%, 20%} were tested.

### 3. Results Integrity (2/5)

**Critical Issues:**

1. **Figure–text mismatch (Appendix, attention patterns).** The paper states (line 1124–1128): *"Left: F = ma (Tier 1)—the model attends uniformly... Right: F = Gm₁m₂/r² (Tier 3)—the model shows more selective attention."* However, the actual figure `figures/attention_patterns.png` shows **"Kinetic energy (t2_01, Tier 2)"** on top and **"Simple pendulum period (t4_01, Tier 4)"** on the bottom. **The paper text does not match the figure content.** This is a significant misrepresentation.

2. **Refinement heatmap step count discrepancy.** The paper repeatedly claims "64 refinement steps" and Figure captions reference "64 refinement steps," but the actual heatmap figures (`refinement_heatmap_t2_01.png`, `refinement_heatmap_t3_01.png`) show only **16 refinement steps** (y-axis labeled 0–16). This is misleading — the reader is told they are seeing a 64-step process but the visualization shows a 16-step process.

3. **Misleading R² reporting in robustness analysis.** The robustness data reports `mean_r2 = 1.0` at all noise levels for Tier 3, but inspection of per-equation data reveals that many equations have `mean_r2 = -1.0` (failed decoding). The `mean_r2` is computed only over equations that produced valid outputs, silently dropping failures. The noise robustness figure shows R² flatlined at 1.0 across all noise levels, which is misleading — it should either report R² over all equations (including failures) or clearly state the denominator. The paper text does not disclose this filtering.

4. **Core contribution underperforms its own ablation.** The ablation data shows that removing recursive soft-masking refinement (the paper's primary methodological contribution transferred from ARChitects) IMPROVES accuracy from 21.2% to 61.5% on Tier 3–5. While the paper honestly discusses this (Section 7.2), the framing throughout the paper — title, abstract, contributions list, and algorithm description — all center recursive soft-masking refinement as the key innovation. A result where the core contribution degrades performance is fundamentally problematic for a publication. The paper should either: (a) reframe the contribution around tree-positional encoding (the actually indispensable component), or (b) demonstrate that refinement helps at larger scale before claiming it as a contribution.

5. **Numerical claims vs. data — mostly consistent.** The following claims were verified against `results/*.json`:
   - Overall 40% symbolic accuracy: ✅ matches `in_distribution_comparison.json` (0.4)
   - Mean R² = 0.911: ✅ matches (0.9108)
   - Tier 1 = 83.3%: ✅ matches (0.8333)
   - Zero-shot discovery rate 9.1% (1/11): ✅ matches
   - Lorentz force F=qvB discovered: ✅ matches (100% across all 5 samples)
   - Coriolis R² = 0.75: ✅ matches
   - Training time 0.62 hours: ✅ matches (0.615)
   - Peak GPU 3.17 GB: ✅ matches (3.174)

### 4. Citation Accuracy (4/5)

**Citation Verification Report:**

| # | Citation Key | Title | Authors | Year | Venue | Verified? | Notes |
|---|-------------|-------|---------|------|-------|-----------|-------|
| 1 | `vaswani2017attention` | Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | ✅ VERIFIED | Correct |
| 2 | `biggio2021neural` | Neural Symbolic Regression that Scales | Biggio et al. | 2021 | ICML | ✅ VERIFIED | Correct |
| 3 | `kamienny2022end` | End-to-End Symbolic Regression with Transformers | Kamienny et al. | 2022 | NeurIPS | ✅ VERIFIED | Correct |
| 4 | `dascoli2022deep` | Deep Symbolic Regression for Recurrent Sequences | d'Ascoli et al. | 2022 | ICML | ✅ VERIFIED | Correct |
| 5 | `udrescu2020ai` | AI Feynman: A Physics-Inspired Method for Symbolic Regression | Udrescu & Tegmark | 2020 | Science Advances | ✅ VERIFIED | Correct, DOI verified |
| 6 | `cranmer2020discovering` | Discovering Symbolic Models from Deep Learning with Inductive Biases | Cranmer et al. | 2020 | NeurIPS | ✅ VERIFIED | Correct |
| 7 | `brunton2016sindy` | Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems | Brunton et al. | 2016 | PNAS | ✅ VERIFIED | Correct, DOI verified |
| 8 | `sahoo2024mdlm` | Simple and Effective Masked Diffusion Language Models | Sahoo et al. | 2024 | NeurIPS | ✅ VERIFIED | Correct; author order in bib differs slightly from published but all names present |
| 9 | `nie2025llada` | Large Language Diffusion Models | Nie et al. | 2025 | arXiv preprint | ✅ VERIFIED | Correct |
| 10 | `architects2025arc` | ARC 2025 Solution by the ARChitects | Lambda Labs | 2025 | Technical report | ✅ VERIFIED | URL verified, content matches |
| 11 | `greydanus2019hamiltonian` | Hamiltonian Neural Networks | Greydanus et al. | 2019 | NeurIPS | ✅ VERIFIED | Correct |
| 12 | `cranmer2020lagrangian` | Lagrangian Neural Networks | Cranmer et al. | 2020 | arXiv preprint | ✅ VERIFIED | Published at ICLR 2020 Workshop; bib lists as arXiv preprint, acceptable |
| 13 | `valipour2021symbolicgpt` | SymbolicGPT: A Generative Transformer Model for Symbolic Regression | Valipour et al. | 2021 | arXiv preprint | ✅ VERIFIED | Correct |
| 14 | `lee2019set` | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | Lee et al. | 2019 | ICML | ✅ VERIFIED | Correct |
| 15 | `tenachi2023physo` | Deep Symbolic Regression for Physics Guided by Units Constraints | Tenachi et al. | 2023 | Astrophysical Journal | ✅ VERIFIED | Correct; bib missing volume/pages (ApJ 959, 99) |
| 16 | `landajuela2022unified` | A Unified Framework for Deep Symbolic Regression | Landajuela et al. | 2022 | NeurIPS | ⚠️ MINOR ERROR | Author "Mulber" in bib should be "Mulcahy" (Garrett Mulcahy). Otherwise correct. |

**Summary:** 15/16 fully correct, 1 with minor author name typo. No fabricated citations. All in-text `\cite` commands resolve to entries in `sources.bib`. The `vaswani2017attention` entry is present in the bib but not cited in the paper text — this is a minor inconsistency (unused reference) but not an error per se.

### 5. Compilation (4/5)

- PDF exists and is 1.74 MB, well-formatted.
- TikZ architecture diagram renders correctly and is a significant quality contribution.
- All figure references resolve to existing files.
- No compilation errors observed.

**Minor issue:** The paper uses `\documentclass[11pt]{article}` rather than the official NeurIPS or ICML style file. For a venue submission, the correct style file should be used.

### 6. Writing Quality (4/5)

**Strengths:**
- Professional, clear academic prose throughout.
- Contributions are clearly enumerated.
- Limitations section is unusually honest for an automated pipeline.
- The paper correctly frames the comparison: "The performance gap between PhysMDT and the AR baseline on in-distribution data reflects the inherently harder task of masked diffusion."
- Good use of notation table and consistent mathematical notation.

**Weaknesses:**
- The abstract claims the model "demonstrates that transformers can autonomously derive physics equations beyond their training distribution" — this is overstated for discovering a single three-variable product (F=qvB). A simple product of three inputs is arguably the simplest non-trivial function class. While technically "never seen during training," the claim of "autonomous derivation" should be tempered.
- The phrase "first application of masked diffusion language modelling to physics equation discovery" is a strong novelty claim that, while likely true, should be softened (e.g., "to our knowledge, the first...").

### 7. Figure Quality (3/5)

**Strengths:**
- `physmdt_training_loss.png`: Excellent three-panel figure with phase transitions clearly marked, color-coded phases, and log-scale. Publication-quality.
- `tier_accuracy_comparison.png`: Clean grouped bar chart with value annotations.
- `r2_distributions.png`: Good violin + box plot combination.
- `embedding_space.png`: t-SNE with dual encoding (color=tier, marker=family). Publication-quality.
- `attention_patterns.png`: Informative cross-attention heatmaps with proper labels and colorbars.
- `ablation_table.png`: Clean tabular visualization.
- `noise_robustness.png` and `data_efficiency.png`: Dual-axis plots with error bars. Reasonable quality.

**Weaknesses:**
- **Refinement heatmaps are problematic.** `refinement_heatmap_t3_01.png` (Newton's gravitation) is almost entirely uniform yellow (confidence ~1.0 everywhere from step 0), making it uninformative. The model appears to converge instantly, which contradicts the narrative of "progressive denoising over 64 steps." Similarly, `refinement_heatmap_t2_01.png` shows most variation only in the first 3 steps and the last 3 steps, with a large uniform region in between. These heatmaps actually demonstrate that the refinement procedure is NOT working as described. They should either be regenerated with more informative examples or the narrative should be adjusted.
- The `baseline_loss.png` uses a slightly different color palette than the PhysMDT figure, creating visual inconsistency when placed side by side.
- Missing: the rubric called for a radar chart (`figures/ablation_radar.png`) which was never generated.

---

## Specific Actionable Feedback for Revision

### Must Fix (Blocking)

1. **Fix the attention pattern figure–text mismatch.** Either regenerate `attention_patterns.png` with F=ma and F=Gm₁m₂/r² as stated in the paper text, OR update the paper text (lines 1124–1128 in the appendix) to correctly describe the actual figure content (Kinetic energy Tier 2 and Simple pendulum period Tier 4).

2. **Fix the refinement step count discrepancy.** The heatmap figures show 16 steps, not 64. Either regenerate the heatmaps with the full 64-step refinement, or update all references to match (paper text and figure captions).

3. **Fix the misleading R² reporting in robustness analysis.** Either: (a) compute R² over ALL equations including failures (using R²=-1 or 0 for failures), (b) clearly state in the paper that R² is computed only over equations with valid symbolic output and report the fraction of valid outputs, or (c) replace R² with a metric that handles failures gracefully.

4. **Rerun ablation study with full evaluation settings** (not quick_mode). The core claim that "single-pass outperforms refinement" cannot be made from an ablation with 8 refinement steps when the main evaluation uses 64. At minimum, the paper must prominently disclose the ablation used reduced settings.

5. **Fix the Landajuela citation:** Change "Mulber, Garrett" to "Mulcahy, Garrett" in `sources.bib`.

### Should Fix (Strongly Recommended)

6. **Reframe the paper's primary contribution.** The ablation shows tree-positional encoding is the indispensable component (0% accuracy without it), while the titular "masked diffusion" with refinement underperforms single-pass decoding. Consider repositioning the paper around: (a) first application of masked diffusion to symbolic regression, (b) the critical finding that tree-positional encoding is essential for masked diffusion on symbolic sequences, and (c) honest exploration of when refinement helps vs. hurts. This would be a more intellectually honest and arguably more interesting paper.

7. **Add statistical significance testing.** Bootstrap confidence intervals were promised in the rubric (item_017) but not delivered. With only 5 test samples per equation, the current results lack statistical power. Consider increasing to at least 20 test samples per equation or adding bootstrap CIs.

8. **Temper the abstract.** The claim "demonstrates that transformers can autonomously derive physics equations" is strong for discovering a single three-variable product. Consider: "provides preliminary evidence that masked diffusion transformers can recover simple physics equations outside their training distribution."

9. **Add the missing ablation radar chart** (`figures/ablation_radar.png`) or remove the reference from the rubric.

10. **Expand the noise robustness evaluation** to include the {1%, 10%} noise levels that were planned but not tested.

### Nice to Have

11. Add volume/page numbers to the `tenachi2023physo` bib entry (ApJ 959, 99).
12. Use the official NeurIPS/ICML style file instead of plain `article` class.
13. Add a Broader Impact section.
14. Remove unused `vaswani2017attention` from the bibliography or add an in-text citation.
15. Harmonize color palettes between baseline and PhysMDT training loss figures.

---

## Summary Assessment

This paper presents an interesting and novel idea — applying masked diffusion language models to physics equation discovery — and implements it competently with several thoughtful components (tree-positional encoding, dimensional analysis bias, curriculum learning). The writing is clear and the limitations discussion is commendably honest.

However, the paper has critical results integrity issues that prevent acceptance: figure–text mismatches, misleading R² reporting, refinement heatmaps that contradict the narrative, and an ablation study that undermines the paper's central contribution. The most significant conceptual problem is that the paper's titular innovation (recursive soft-masking refinement from ARChitects) actually degrades performance compared to the ablation without it. This needs to be addressed either by demonstrating that refinement helps at larger scale or by reframing the contribution.

The zero-shot discovery of F=qvB is genuinely interesting but should be presented with appropriate caveats (it is a simple three-variable product). The true gem of this paper may be the finding that tree-positional encoding is critical for masked diffusion on symbolic sequences — this is a novel, empirically grounded insight that could be the centerpiece of a revised submission.

**Verdict: REVISE** — address the 5 blocking issues and at least items 6–8 from the recommended fixes before resubmission.
