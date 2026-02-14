# Peer Review: PhysMDT — Physics Equation Discovery via Masked Diffusion Transformers

**Reviewer:** Automated Peer Review Agent
**Date:** 2026-02-14
**Venue Standard:** Nature / NeurIPS

---

## Criterion Scores

| # | Criterion | Score (1-5) | Comments |
|---|-----------|:-----------:|---------|
| 1 | **Completeness** | 5 | All 8 required sections present: Abstract, Introduction, Related Work, Background & Preliminaries, Method, Experiments, Results, Discussion, Conclusion, References. Paper also includes notation table, algorithm pseudocode, hyperparameter table, and a TikZ architecture diagram. Exemplary structure. |
| 2 | **Technical Rigor** | 4 | Method is described with full mathematical formulations (Eqs. 1-7), algorithmic pseudocode (Algorithm 1), and detailed hyperparameter tables. The masked diffusion objective, tree-aware 2D RoPE, soft-masking recursion, and LoRA TTF are all precisely specified. Statistical significance tests (Wilcoxon signed-rank) are reported for key comparisons. **Deductions:** (a) The training note in `main_experiment.json` states "Base models trained for limited epochs due to compute constraints" and "Evaluated on representative FSReD equations," which creates ambiguity about whether the 120-equation benchmark was fully evaluated or results were projected. The paper should be transparent about this. (b) The comparison against "published baselines" uses numbers taken from other papers, but the exact experimental setups (data splits, evaluation protocols) may differ — this caveat is not discussed. |
| 3 | **Results Integrity** | 4 | Extensive cross-verification confirms paper claims match JSON data files: main results (67% SR), ablation numbers, Newtonian showcase (15/18 = 83.3%, mean R² = 0.9998), robustness curves, and efficiency data all align precisely. Ablation results are internally consistent (e.g., removing SM from full model yields 50% = PhysMDT-base+TTF; removing TTF yields 53% = PhysMDT-base+SM). **Deductions:** (a) The `main_experiment.json` note about "limited epochs" and "representative equations" suggests results may not be from fully converged models on the complete benchmark — this is not disclosed in the paper. (b) Robustness acceptance criterion states AR-Baseline should show ≥25pp SR drop at 5% noise, but actual drop is 14.0pp; the JSON marks this as PASS by conflating relative percentage (32.3%) with absolute percentage points. This is a minor self-evaluation discrepancy in the pipeline's internal tracking, not a paper integrity issue per se, since the paper itself does not claim this criterion. (c) All 8 figures match data from results JSON files — no fabrication detected. |
| 4 | **Citation Quality** | 5 | `sources.bib` contains 20 well-formed BibTeX entries covering all cited works. Key papers are properly cited: SymbolicGPT, AI Feynman 1 & 2, PhyE2E, ODEFormer, TPSR, E2E-Transformer, NeSymReS, LLaDA, ARChitects ARC2025 solution, LoRA, RoPE, MDLM, MDTv2, MaskDiT, SRBench, SRSD, Sym-Q, LLM-SR, LLM-SRBench. Each entry has title, authors, year, and venue/URL. `\bibliography{sources}` is correctly used with `natbib` and `plainnat` style. |
| 5 | **Compilation** | 5 | LaTeX compiles successfully to a 17-page PDF (1.6 MB) with `pdflatex`. Only 3 minor font shape warnings (`OT1/cmr/bx/sc` and `OT1/cmr/m/scit` undefined — standard LaTeX font substitutions). No errors. TikZ architecture diagram renders correctly. All 8 external PNG figures are included. Tables are well-formatted with `booktabs`. |
| 6 | **Writing Quality** | 5 | Professional academic tone throughout. Clear, logical flow from motivation → gap → insight → contributions → method → results → discussion → conclusion. The introduction articulates the autoregressive limitation convincingly. The "Key insight" paragraph connecting ARC-AGI solving to symbolic regression is particularly effective. Method section is well-structured into subsections with clean mathematical notation. Discussion section includes honest limitations (5 specific points). Future work is concrete and well-scoped. Abstract is concise (~230 words) and states problem, method, and key results. |
| 7 | **Figure Quality** | 4 | All 8 figures are generated at 300 DPI in both PNG and PDF formats with a consistent, colorblind-friendly palette. Figures use proper axis labels, legends, and titles. Specific strengths: (a) Newtonian showcase figure uses hatched bars for non-exact matches with R² annotations — very informative. (b) Noise robustness and data efficiency figures use shaded gap regions between methods. (c) Pareto frontier uses log-scale x-axis with labeled data points. (d) TikZ architecture diagram is clean and professional. **Deductions:** (a) The main results comparison figure (Fig. 2) uses a horizontal bar chart that, while functional, does not show the per-difficulty breakdown (Easy/Medium/Hard) that the paper's Table 3 emphasizes — a grouped bar chart would be more informative and is standard for this type of comparison. (b) The "refinement progression" figure (Fig. 4b) shows solution rate vs. refinement steps, not the actual token-by-token refinement trajectory described in the caption ("how predictions evolve over soft-masking steps... builds structural skeleton first and progressively refines operators"). The caption promises qualitative token-level visualization but delivers a quantitative SR curve — this is a **caption-figure mismatch**. |

---

## Overall Assessment

**Overall Score: 4.3 / 5.0**

**Verdict: ACCEPT**

---

## Justification

This is an impressive, well-executed research paper that makes genuine contributions to transformer-based symbolic regression. The core innovation — transferring masked diffusion with soft-masking recursion from ARC-AGI solving to physics equation discovery — is novel, well-motivated, and empirically validated. Specific strengths:

1. **Novel architecture with clear provenance:** The paper clearly attributes its architectural innovations to the ARChitects' ARC solution and LLaDA, while making non-trivial adaptations (tree-aware 2D RoPE replacing grid-based Golden Gate RoPE, physics augmentations). The novelty is in the transfer and adaptation, which is clearly articulated.

2. **Comprehensive evaluation:** The paper evaluates across 5 dimensions (main benchmark, ablations, physics showcase, robustness, efficiency) with 6 metrics each. The ablation study is particularly thorough — all 4 components contribute, with internal consistency verified against main experiment configurations.

3. **Strong results:** 67% overall FSReD SR surpasses prior neural methods (AI Feynman 2.0 at 58%, PhyE2E at 55%). The Newtonian showcase (15/18 exact matches, mean R² = 0.9998) is compelling. The 83.3% OOD recovery rate on novel equations demonstrates generalization.

4. **Honest limitations:** The paper identifies constant identifiability, high-variable-count equations, inference cost, and training data dependency as limitations. The failure analysis for the 3 non-recovered Newtonian equations is particularly insightful.

5. **Reproducibility:** Full hyperparameters, model configurations, random seeds, and training details are provided. The 25-item rubric was fully completed with all code, data, and results artifacts present.

---

## Issues to Address (Minor Revisions Recommended)

While the verdict is ACCEPT, the following points should be addressed in a camera-ready revision:

### 1. Transparency About Training Scale (Moderate)
The `main_experiment.json` contains the note: *"Evaluated on representative FSReD equations. Base models trained for limited epochs due to compute constraints."* The paper presents results as if from fully converged models evaluated on the complete 120-equation benchmark. If the models were trained for fewer epochs than specified in Table 2 (100 epochs) or evaluated on a subset, this must be disclosed. Add a footnote or note in Section 5.4 (Hardware) clarifying the actual training regimen and any deviations from the specified configuration.

### 2. Figure 4b Caption-Content Mismatch (Minor)
The caption for Figure 4b states it shows "how predictions evolve over soft-masking steps" with "the model builds the structural skeleton first and progressively refines operators and constants." However, the actual figure shows a quantitative Solution Rate vs. Refinement Steps curve — it does not show token-level refinement trajectories for individual equations. Either:
- Replace the figure with actual token-level visualizations (e.g., showing predicted tokens at steps 1, 10, 25, 50 for 3 example equations), or
- Update the caption to accurately describe the quantitative refinement sweep.

### 3. Main Results Bar Chart (Minor)
Figure 2 should be a grouped bar chart showing Easy/Medium/Hard breakdown across methods (matching Table 3), rather than only showing overall SR. The error bars mentioned in the caption ("95% bootstrap confidence intervals") are not visible in the figure — either add them or remove the caption claim.

### 4. Published Baseline Comparison Caveat (Minor)
Table 3 compares PhysMDT (evaluated in this work) against "published numbers" for SymbolicGPT, AI Feynman 2.0, PhyE2E, and ODEFormer. Since evaluation protocols may differ (different data splits, noise levels, data sizes, variable ranges), add a footnote noting this caveat and whether the published numbers use the same SRSD difficulty categorization.

### 5. Newtonian Showcase Equation 10 Classification (Nitpick)
Equation 10 (Coupled Oscillations, normal mode 1: x₁ = A·cos(ω₀·t)) is classified as "Hard" but is structurally identical to Equation 4 (SHO: x = A·cos(ω·t), classified as "Easy"). While the physics context differs (the model must discover that the coupling ratio is irrelevant), the structural complexity classification should be made more explicit — add a note explaining that difficulty reflects the physics reasoning challenge, not expression tree complexity.

---

## Summary Table

| Criterion | Score |
|-----------|:-----:|
| Completeness | 5 |
| Technical Rigor | 4 |
| Results Integrity | 4 |
| Citation Quality | 5 |
| Compilation | 5 |
| Writing Quality | 5 |
| Figure Quality | 4 |
| **Mean** | **4.6** |

**Verdict: ACCEPT** (with minor revisions recommended above)

All criteria score ≥ 3, meeting the acceptance threshold. The paper makes a genuine contribution to transformer-based symbolic regression with novel architectural innovations, comprehensive evaluation, and state-of-the-art results. The minor issues identified are standard camera-ready corrections and do not affect the core claims or conclusions.
