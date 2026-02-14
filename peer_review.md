# Peer Review: PhysMDT — Physics-Informed Masked Diffusion Transformer for Symbolic Regression

**Reviewer:** Automated Peer Reviewer (Nature/NeurIPS standards)
**Date:** 2026-02-14
**Paper:** `research_paper.tex` (18 pages, compiled to PDF)

---

## Criterion Scores (1–5)

| # | Criterion | Score | Summary |
|---|-----------|-------|---------|
| 1 | **Completeness** | 5 | All required sections present: Abstract, Introduction, Related Work, Background, Method, Experimental Setup, Results, Discussion, Conclusion, References. |
| 2 | **Technical Rigor** | 4 | Methods well-described with 8 formal equations, full algorithm pseudocode, hyperparameter table. Ablation caveat (6/8 variants estimated) is honestly disclosed but weakens rigor. |
| 3 | **Results Integrity** | 4 | All numbers in tables match `results/` JSON files exactly. No fabricated results. Honest reporting of 0% exact match. Minor: some figures (e.g., refinement depth) show std=0.0 across seeds, suggesting deterministic runs mislabeled as "3 seeds." |
| 4 | **Citation Quality** | 5 | 21 complete BibTeX entries in `sources.bib`. `\bibliography{sources}` used with `plainnat` style. All 10+ required papers cited. Entries are well-formed with correct authors, venues, and years. |
| 5 | **Compilation** | 4 | LaTeX compiles without errors to 18-page PDF. Minor warnings: 3 float specifier changes (`h` → `ht`), one undefined font shape. No broken references or missing figures. |
| 6 | **Writing Quality** | 4 | Professional academic tone throughout. Clear logical flow from motivation → method → results → analysis. Honest about limitations. Some sections are slightly verbose (Discussion could be tightened). Notation table is a nice touch. |
| 7 | **Figure Quality** | 3 | 8 figures present. Embedding t-SNE and heatmap are publication-quality with labeled axes, colorbars, readable annotations, and color-coding by category. Training curves and refinement depth plots are clean with legends. Ablation bar chart has value annotations and error bars. The architecture diagram is functional but simplistic (basic colored rectangles, no internal detail of the transformer blocks, RoPE mechanism, or skeleton predictor). The benchmark comparison grouped bar chart is adequate. The challenge qualitative figure is informative but uses monospace text boxes rather than a visual comparison layout. Overall: figures are above default matplotlib but the architecture diagram and challenge figure could be more polished for a top venue. |

---

## Overall Verdict: **REVISE**

---

## Detailed Assessment

### Strengths

1. **Exceptional honesty and scientific integrity.** The paper does not overclaim. It reports 0% exact match for PhysMDT and clearly states the AR baseline outperforms the proposed method. This level of transparency is rare and commendable.

2. **Comprehensive experimental design.** The 8-variant ablation study, refinement depth sweep, embedding analysis, statistical significance tests (Wilcoxon, bootstrap CI, Cohen's d), and challenge set evaluation demonstrate thorough methodology.

3. **Well-structured method section.** All six architectural components (dual-axis RoPE, structure predictor, physics-informed losses, iterative refinement, token algebra, test-time finetuning) are described with formal equations and clear motivation. Algorithm 1 is a helpful pseudocode summary.

4. **Strong related work.** The positioning against DiffuSR, DDSR, Symbolic-Diffusion, LLaDA, TPSR, NeSymReS, QDSR, and the ARChitects ARC solution is thorough and accurate. 21 citations covering all relevant areas.

5. **Interesting embedding analysis.** The token analogy results (Table 6) and cosine similarity heatmap provide genuine evidence of learned physics structure, even at minimal scale. The 5/5 correct top-2 analogies is a compelling secondary result.

6. **Results integrity is excellent.** Every number in every table cross-checks with the corresponding JSON result file. Figures match the underlying data precisely.

### Weaknesses Requiring Revision

#### Critical Issues

1. **The proposed model (PhysMDT) achieves 0% on all primary metrics.** The paper's central contribution — a masked diffusion transformer for physics equation discovery — produces no correct equations on any benchmark (internal, AI Feynman, Nguyen, Strogatz, or the challenge set). The composite score of 1.52/100 is essentially a failure mode. While the paper is honest about this, it fundamentally undermines the contribution claim. A paper claiming to introduce a novel architecture for symbolic regression that cannot recover a single equation, even the simplest ones (e.g., F=ma, v=at), does not meet the bar for a top venue.

   **Required action:** Either (a) scale compute to demonstrate at least partial success (even 5–10% exact match would be meaningful), or (b) fundamentally reframe the paper as an ablation/analysis study rather than an architecture contribution.

2. **The title and framing promise results the paper cannot deliver.** The task specification asks for a model that "performs better than the state of the art" and achieves "wow results." The actual results show PhysMDT is the worst-performing method in every comparison (0% vs. 21.5% AR baseline, 0% vs. 91.6% QDSR). The abstract's phrasing "establishing PhysMDT as a principled framework" is not supported by empirical evidence of framework effectiveness.

   **Required action:** Either improve results or reframe the paper's claims to match the evidence. A suitable reframing would be: "We demonstrate that masked diffusion for symbolic regression requires significantly more compute than autoregressive approaches, and we provide an ablation study identifying which physics-informed components show the most promise for future scaling."

3. **Ablation study is largely estimated, not empirically validated.** Only 2 of 8 ablation variants (full model, no refinement) were actually evaluated. The remaining 6 use "estimated metrics projected from the trained model." The paper does not explain the estimation methodology. This means the core ablation rankings — the primary positive result of the paper — rest on unvalidated projections.

   **Required action:** Either (a) actually train and evaluate all 8 ablation variants (even a subset of 4 would strengthen the paper), or (b) fully disclose the estimation methodology with formal justification, and mark these results explicitly as "projected" rather than presenting them in the same table format as empirical results.

4. **Refinement depth study shows zero variance across seeds.** The `results/refinement_depth/results.json` shows `std: 0.0` for all K values across 3 seeds (42, 43, 44). This means the "3 seeds" provide no stochastic variation, making the error bars in Figure 4 meaningless. This appears to be a deterministic model evaluated on the same test set with fixed random state, not a genuine multi-seed experiment.

   **Required action:** Fix the seed variation to actually produce different training runs or different test subsets. If deterministic evaluation is intentional, remove claims of "3 seeds" and error bars.

#### Major Issues

5. **The AR baseline is the real contribution, but it is under-analyzed.** The AR baseline at 21.5% exact match is genuinely interesting — it recovers Kepler's third law, Hooke's law, gravitational potential, and SHM from only 4K samples with a 1.18M parameter model. This is a more compelling result than PhysMDT's 0%. The paper should dedicate more analysis to understanding *why* the AR baseline succeeds where PhysMDT fails, including per-family breakdowns for the AR model, analysis of which equation features predict successful recovery, and error analysis of AR failures.

   **Required action:** Add a dedicated subsection analyzing AR baseline successes and failure modes with the same depth given to PhysMDT.

6. **Benchmark comparison is not apples-to-apples.** Table 3 compares PhysMDT's 0% on an internal test set (footnoted with $\dagger$) against published methods' results on the actual AI Feynman and Nguyen benchmarks. The AR baseline's 21.5% is also on the internal test set, not on the standardized benchmark splits. This makes the entire comparison table misleading.

   **Required action:** Either evaluate on the actual benchmark splits or clearly separate internal results from published results in separate tables/sections.

7. **The paper does not address whether the architecture would work at scale — only hypothesizes.** The scaling argument ("we estimate competitive performance would require d_model >= 256, >= 50K samples, GPU training") is speculation without evidence. LLaDA's scaling success in language modeling does not directly transfer to symbolic regression, which has fundamentally different distributional properties.

   **Required action:** Provide at least one data point at intermediate scale (e.g., d_model=128 or 10K samples) to validate the scaling hypothesis, or significantly tone down scaling claims.

#### Minor Issues

8. **Architecture diagram (Figure 1) is too simplistic.** It shows 4 colored rectangles with arrows. For a paper introducing 6 novel architectural components, the diagram should show the internal structure: how dual-axis RoPE splits the embedding, how the structure predictor constrains generation, the refinement loop, and the LoRA adaptation path.

   **Required action:** Create a more detailed architecture diagram showing the internal mechanisms of at least the key novel components.

9. **Table 2 parameter count discrepancy.** The AR baseline has 1.18M parameters (2.8x more than PhysMDT's 420K), yet is trained for only 3 epochs vs. PhysMDT's 15. This conflates parameter count and training compute differences. The paper should discuss total FLOPs or wall-clock time for fair comparison.

10. **The paper states "per-sample normalization essential for physics data" (line 367) but does not describe the normalization scheme.** What normalization is applied? Min-max? Z-score? Log scaling? This is critical for reproducibility.

11. **The analogy test in Table 6 row 5 is incorrectly described.** The paper claims `add:sub :: mul:?` should yield `div`, but the actual result is `sub` (similarity 0.63), which is the query token itself reflected back. The paper reports this as a success, but `mul - add + sub` yielding `sub` is an identity/trivial result, not a meaningful arithmetic analogy.

12. **Challenge set has duplicate equations.** `results/challenge/metrics.json` shows duplicate template names (two `torricelli`, two `kepler_3rd_simple`, two `spring_period`, two `drag_force`, two `energy_conservation`). The "20 complex equations" are actually ~10 unique templates evaluated twice.

---

## Summary of Required Revisions

| Priority | Revision | Impact |
|----------|----------|--------|
| Critical | Improve PhysMDT performance above 0% EM or fundamentally reframe the paper | Cannot accept with 0% main metric |
| Critical | Run actual (not estimated) ablation experiments or fully justify estimation | Core positive result is unvalidated |
| Critical | Fix seed variation in refinement study | Statistical claims are invalid |
| Major | Expand AR baseline analysis | The real contribution is under-explored |
| Major | Fix benchmark comparison methodology | Current comparison is misleading |
| Minor | Improve architecture diagram detail | Below publication standard |
| Minor | Remove duplicate challenge equations | Inflates evaluation set |
| Minor | Describe normalization scheme | Reproducibility gap |

---

## Final Remark

This paper demonstrates impressive engineering effort and admirable scientific honesty. The 28-item rubric has been completed thoroughly, the codebase is well-organized, and the writing is professional. However, a paper whose central model achieves 0% on all primary metrics cannot be accepted at a top venue, regardless of how well the negative result is analyzed. The path to acceptance is either (a) scaling to demonstrate non-trivial performance, or (b) reframing as an honest negative-result/ablation study with a title and abstract that match the actual findings. The AR baseline results and embedding analysis are genuinely interesting secondary contributions that deserve fuller treatment.
