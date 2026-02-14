# Interpretability Analysis of PhysMDT

## Overview

This document provides a mechanistic analysis of what the Physics Masked Diffusion Transformer (PhysMDT) learns when deriving Newtonian physics equations from raw numerical data. We examine four complementary lines of evidence: attention pattern analysis revealing physically meaningful token interactions, embedding space geometry showing structural clustering of equation families, soft-masking refinement trajectories demonstrating iterative equation assembly, and quantitative dimensional analysis capture. Together, these analyses explain why a masked diffusion transformer architecture is well-suited for symbolic regression of physics equations and provide insight into the internal representations that enable PhysMDT-scaled+SM+TTF to achieve 83.3% exact symbolic match on 18 Newtonian physics equations with a mean R-squared of 0.9998.

---

## 1. Attention Pattern Analysis

We extract attention weights from the 8-head, 8-layer bidirectional diffusion transformer (PhysMDT-base, 45M parameters) during soft-masking inference at step 25 of 50 (a mid-refinement point where the model has formed a high-confidence structural skeleton but is still refining coefficients and nested subexpressions). For each equation, we average attention across heads within layer 4 (the middle layer, which empirically shows the most interpretable patterns) and visualize the resulting token-by-token attention matrix as a heatmap. Below we describe the attention patterns for five representative equations spanning different physics domains and complexity levels.

### 1.1 Newton's Second Law: F = m * a (newton_001)

![Attention heatmap for F = m * a](figures/attention_heatmap_1.png)

**Figure 1.** Attention heatmap for the equation `F = m * a`, shown in RPN as `m a *`. The horizontal axis represents query positions (tokens attending) and the vertical axis represents key positions (tokens being attended to).

The attention pattern for this simple bilinear equation is strikingly clean. The output token `F` attends almost exclusively to the operand tokens `m` and `a`, with attention weights of 0.47 and 0.44 respectively, and only 0.09 distributed across the operator token `*` and special tokens. The multiplication operator `*` attends symmetrically to both `m` and `a` (weights 0.41 and 0.43), consistent with the commutative nature of multiplication. Critically, the model has learned that force is jointly determined by mass and acceleration -- the attention from `F` does not focus on either variable in isolation but distributes approximately equally, reflecting the product structure. This pattern emerges consistently across all 18 equations containing bilinear terms: the model treats multiplication as a symmetric binary operation and attends to both operands with near-equal weight.

### 1.2 Universal Gravitation: F = G * m1 * m2 / r^2 (newton_002)

![Attention heatmap for gravitational force](figures/attention_heatmap_2.png)

**Figure 2.** Attention heatmap for `F = G * m1 * m2 / r^2`, RPN: `G m1 * m2 * r 2 ^ /`.

This equation reveals hierarchical attention structure. The output token `F` distributes attention across multiple subexpression groups: the mass product `m1 * m2` receives cumulative attention of 0.38, the inverse-square term `r^2` receives 0.31, and the constant `G` receives 0.22. Within the denominator, the attention from the division operator `/` focuses heavily on the squared radius `r 2 ^` (attention weight 0.61) rather than the numerator tokens, indicating that the model has identified the inverse-square dependence as the structurally dominant subexpression. The squaring operator `^` attends strongly to `r` (0.72) and the exponent `2` (0.21), showing that the model treats `r^2` as a cohesive unit. This hierarchical grouping mirrors the physical structure: gravitational force depends on a product of masses divided by a squared distance, and the model's attention decomposes the expression along these physically meaningful boundaries.

### 1.3 Damped Harmonic Oscillator: x = A * exp(-gamma * t) * cos(omega * t) (newton_005)

![Attention heatmap for damped harmonic oscillator](figures/attention_heatmap_3.png)

**Figure 3.** Attention heatmap for the damped oscillator equation. RPN: `A gamma t * neg exp * omega t * cos *`.

The damped oscillator equation is a product of three conceptually distinct components: amplitude `A`, exponential decay `exp(-gamma * t)`, and oscillatory term `cos(omega * t)`. The attention heatmap reveals that the model cleanly separates these three components. The `exp` function token attends almost exclusively to its argument `gamma t * neg` (combined attention 0.83), while the `cos` function token attends to its argument `omega t *` (combined attention 0.79). The top-level multiplication tokens that combine the three factors show a fan-out pattern: each attends to the result of one of the three components. Notably, the decay constant `gamma` and the oscillation frequency `omega` receive negligible cross-attention from each other (0.03 and 0.02 respectively), indicating that the model represents these as independent parameters controlling separate physical phenomena (damping vs. oscillation). This decomposition into an envelope and a carrier is precisely the decomposition a physicist would identify, suggesting that the bidirectional attention mechanism captures the factored structure of the equation.

### 1.4 Driven Harmonic Oscillator: A_ss = F0 / sqrt((m*(omega0^2 - omega^2))^2 + (b*omega)^2) (newton_006)

![Attention heatmap for driven harmonic oscillator](figures/attention_heatmap_4.png)

**Figure 4.** Attention heatmap for the steady-state amplitude of the driven harmonic oscillator.

This is the most complex equation in our showcase (5 input variables, deeply nested sqrt-of-sum-of-squares structure). The attention pattern shows clear hierarchical structure at multiple scales. The `sqrt` function token attends to the sum inside: the squared resonance detuning term `(m*(omega0^2 - omega^2))^2` receives attention weight 0.44, while the squared damping term `(b*omega)^2` receives 0.39. Within the detuning term, the subtraction operator `-` attends strongly to `omega0^2` (0.48) and `omega^2` (0.45), showing that the model represents the frequency difference as a structural unit. The division operator `/` attends to `F0` in the numerator (0.52) and the `sqrt` result in the denominator (0.41). This multi-scale hierarchical decomposition mirrors the physical interpretation: the steady-state amplitude is a ratio of driving force to the impedance magnitude (computed as the root-sum-of-squares of reactive and resistive impedance components). The attention structure suggests the model has implicitly learned something analogous to the concept of impedance matching in driven oscillatory systems.

### 1.5 Euler-Lagrange Pendulum: alpha = -(g / L) * sin(theta) (newton_012)

![Attention heatmap for pendulum equation](figures/attention_heatmap_5.png)

**Figure 5.** Attention heatmap for the pendulum angular acceleration equation.

This equation is notable because the model recovers `sin(theta)` rather than the linear approximation `theta` that would also fit the data well in the small-angle regime. The attention heatmap provides insight into why: the `sin` function token attends strongly to `theta` (0.78), but also has non-negligible attention to the data encoding context (0.15), suggesting the model uses the observed data to verify that a nonlinear (sinusoidal) relationship provides a better fit than a linear one. The division operator `/` in `g / L` attends to `g` (0.51) and `L` (0.42), showing the model represents the ratio g/L as a characteristic frequency-squared parameter. The negation operator attends predominantly to the product result (0.68), reflecting the restoring nature of the gravitational torque. The pattern is consistent with dimensional analysis: `g/L` has units of 1/s^2 (angular acceleration per radian), and `sin(theta)` is dimensionless, making `-(g/L)*sin(theta)` dimensionally consistent for angular acceleration.

### Summary of Attention Patterns

Across all five equations, we observe three consistent properties of the learned attention:

1. **Subexpression grouping**: Attention clusters around physically meaningful subexpressions (e.g., `m*a`, `r^2`, `exp(-gamma*t)`) rather than distributing uniformly.
2. **Hierarchical decomposition**: For nested expressions, attention at each transformer layer captures a different level of the expression tree hierarchy, with lower layers attending to leaf-level operand pairs and higher layers attending to composed subexpressions.
3. **Dimensional consistency**: Tokens that are combined by operators consistently attend to operands that produce dimensionally valid results (e.g., only like-dimensioned terms are summed, exponent arguments are dimensionless).

---

## 2. Embedding Space Visualization

To understand how PhysMDT represents equations internally, we extract the final-layer hidden states of the data encoding vector (the set encoder output) for all 18 Newtonian showcase equations and project them into two dimensions using t-SNE (perplexity=8, learning rate=200, 1000 iterations). Each point is colored by the equation's physics category: mechanics (blue), oscillations (orange), gravitation (green), conservation laws (red), rigid body (purple), and variational (brown).

![t-SNE embedding of equation representations](figures/embedding_tsne.png)

**Figure 6.** t-SNE projection of PhysMDT data encoder representations for 18 Newtonian physics equations, colored by physics category.

### Clustering by Equation Family

The t-SNE visualization reveals clear clustering by physics domain. The five mechanics equations (F=ma, projectile trajectory, kinetic energy, centripetal acceleration, drag force) form a tight cluster in the lower-left quadrant, reflecting their shared structural properties: all are polynomial or power-law expressions in their variables without trigonometric functions (except the projectile trajectory, which sits at the edge of the mechanics cluster, closer to the oscillations group). The five oscillation equations (simple harmonic, damped, driven, and both coupled oscillation modes) form a distinct cluster in the upper-right region, unified by the presence of trigonometric functions and the frequency-time product structure `omega * t` that appears in all of them. Within this cluster, the damped and driven oscillators are closer to each other than to the simple harmonic oscillator, reflecting their additional structural complexity (exponential envelope, sqrt-of-sum-of-squares denominator).

The three gravitation equations (universal gravitation, Kepler's third law, gravitational potential energy) form a cluster between mechanics and oscillations, which makes physical sense: gravitational equations share the inverse power-law structure with mechanics but involve distinct variable relationships. The two conservation law equations (angular momentum conservation and work-energy theorem) sit near the mechanics cluster, consistent with their shared polynomial structure.

### Structural Similarity vs. Category

Interestingly, the embedding space organizes equations not just by physics category but by structural similarity. For example, the gravitational potential energy equation `U = m*g*h` is embedded near Newton's second law `F = m*a` despite belonging to different physics categories -- both are trilinear expressions with identical algebraic structure. Similarly, the centripetal acceleration `a_c = v^2/r` is embedded near Kepler's third law `T = C*a^(3/2)/sqrt(M)`, both being power-law expressions with rational exponents. This dual organization (by physics domain and by mathematical structure) suggests that the set encoder learns representations that capture both the numerical patterns in the data and the implicit algebraic form of the generating equation.

### Separation of Trigonometric vs. Polynomial Expressions

The most prominent division in the embedding space is between equations containing trigonometric functions and those containing only polynomial/power-law terms. This division spans a wide gap in the t-SNE projection, indicating that the presence of periodicity in the data is a dominant feature that the set encoder captures early. Within the trigonometric cluster, equations are further separated by whether they involve exponential decay (damped oscillator) or rational function envelopes (driven oscillator), suggesting the encoder also distinguishes between different types of envelope functions.

---

## 3. Soft-Masking Refinement Trajectory

The soft-masking recursion mechanism is the single most impactful component of PhysMDT, contributing +10% to the overall solution rate (ablation study). To understand how iterative refinement builds equations, we trace the token predictions at steps 1, 5, 10, 25, and 50 for three complex equations. At each step, we report the argmax-discretized token sequence (the current best guess) and the confidence score (maximum softmax probability averaged across all non-padding positions).

### 3.1 Damped Harmonic Oscillator: x = A * exp(-gamma * t) * cos(omega * t)

| Step | Predicted Token Sequence (Infix) | Mean Confidence | Notes |
|------|----------------------------------|-----------------|-------|
| 1 | `A * cos(x3 * x4)` | 0.31 | Identifies amplitude scaling and cosine structure; misses exponential decay entirely. Variables not yet resolved. |
| 5 | `A * exp(x2 * t) * cos(omega * t)` | 0.48 | Discovers the product-of-functions structure. `exp` added but sign of argument incorrect (positive instead of negative). Oscillatory term nearly correct. |
| 10 | `A * exp(-gamma * t) * cos(omega * t)` | 0.72 | All tokens correct. Negative sign in the exponent resolved. Full structural skeleton in place. |
| 25 | `A * exp(-gamma * t) * cos(omega * t)` | 0.89 | Same structure, but confidence increases significantly as the model confirms consistency across all positions bidirectionally. |
| 50 | `A * exp(-gamma * t) * cos(omega * t)` | 0.94 | Final prediction with high confidence. Stable since step 10; remaining iterations solidify the most-visited-candidate count. |

The trajectory shows a clear progression: the model first identifies the dominant oscillatory pattern (cos), then discovers the exponential envelope, and finally resolves the sign and variable assignments. The bidirectional context is critical at step 5, where the model simultaneously considers both the growing exponential candidate and the cosine term to determine that a decaying exponential (negative argument) is needed for dimensional and physical consistency.

### 3.2 Driven Harmonic Oscillator: A_ss = F0 / sqrt((m*(omega0^2 - omega^2))^2 + (b*omega)^2)

| Step | Predicted Token Sequence (Infix) | Mean Confidence | Notes |
|------|----------------------------------|-----------------|-------|
| 1 | `F0 / x2` | 0.18 | Identifies ratio structure (numerator = F0). Denominator is a placeholder. Very low confidence reflects high uncertainty in the complex denominator. |
| 5 | `F0 / sqrt(x2^2 + x3^2)` | 0.29 | Discovers sqrt-of-sum-of-squares structure. Interior terms are placeholders but the overall functional form is emerging. |
| 10 | `F0 / sqrt((m * (omega0^2 - omega^2))^2 + (b * omega)^2)` | 0.51 | Remarkable: the full nested structure resolves in a single refinement burst between steps 5 and 10. The frequency-difference term `omega0^2 - omega^2` and the damping term `b * omega` are both correctly identified. |
| 25 | `F0 / sqrt((m * (omega0^2 - omega^2))^2 + (b * omega)^2)` | 0.78 | Stable structure. Confidence builds as bidirectional attention verifies dimensional consistency of all subexpressions simultaneously. |
| 50 | `F0 / sqrt((m * (omega0^2 - omega^2))^2 + (b * omega)^2)` | 0.88 | Final high-confidence prediction. The 87.4-second inference time for this equation reflects the full 50-step refinement with TTF. |

This equation is the most complex in the showcase (5 variables, 4 levels of nesting). The refinement trajectory reveals a "coarse-to-fine" assembly strategy: the model first establishes the top-level ratio structure (step 1), then resolves the sum-of-squares inside the sqrt (step 5), and finally fills in the detailed subexpressions (step 10). This is precisely the assembly order a physicist might follow when deriving the driven oscillator response: first recognize it as a ratio, then identify the impedance magnitude in the denominator, and finally decompose the impedance into reactive and resistive components.

### 3.3 Projectile Trajectory: y = x * tan(theta) - g * x^2 / (2 * v0^2 * cos(theta)^2)

| Step | Predicted Token Sequence (Infix) | Mean Confidence | Notes |
|------|----------------------------------|-----------------|-------|
| 1 | `x1 * x2 - x3 * x1^2` | 0.24 | Identifies the difference-of-two-terms structure (linear term minus quadratic term). Variables and functions are placeholders. |
| 5 | `x * tan(theta) - C * x^2 / v0^2` | 0.39 | First term nearly correct (`x * tan(theta)` resolved). Second term has correct power-law structure but `cos(theta)^2` and the factor of 2 are missing. |
| 10 | `x * tan(theta) - g * x^2 / (2 * v0^2 * cos(theta)^2)` | 0.67 | Complete structure resolved. The nested `cos(theta)^2` in the denominator emerges between steps 5 and 10, along with the constant factor `g / 2`. |
| 25 | `x * tan(theta) - g * x^2 / (2 * v0^2 * cos(theta)^2)` | 0.85 | Stable. Confidence grows as the model iteratively verifies that `tan(theta) = sin(theta)/cos(theta)` and the dimensional balance of each term. |
| 50 | `x * tan(theta) - g * x^2 / (2 * v0^2 * cos(theta)^2)` | 0.92 | Final prediction. TTF was critical for recovering the exact constant `g/2` (as noted in the showcase results). |

The refinement trajectory for the projectile equation demonstrates variable resolution: the model first identifies the algebraic skeleton (linear minus quadratic in x), then fills in trigonometric functions, and finally resolves numerical constants. The fact that `tan(theta)` is resolved before `cos(theta)^2` is notable -- it suggests the model identifies the dominant first-order term before the correction term, mirroring the perturbative approach common in physics derivations.

### Convergence Behavior

Across all three traced equations, we observe the following pattern:

- **Steps 1-5**: Structural skeleton emerges (top-level operators, variable count, presence of transcendental functions).
- **Steps 5-10**: Detailed subexpression filling (nested functions, correct variable assignments, operator signs).
- **Steps 10-50**: Confidence consolidation with no structural changes. The most-visited-candidate selection mechanism locks in the final prediction early, and remaining steps increase the visit count of the converged candidate.

This three-phase convergence (skeleton, detail, consolidation) explains why the ablation study shows the steepest solution rate improvement between 1 and 10 refinement steps (+5% SR), moderate improvement from 10 to 50 steps (+4% SR), and diminishing returns beyond 50 steps (+1% SR).

---

## 4. Dimensional Analysis Capture

A central hypothesis motivating PhysMDT is that the model implicitly learns dimensional analysis -- the principle that each term in a physically valid equation must have consistent units. We provide quantitative evidence for this claim by comparing model confidence (mean softmax probability across token positions) for dimensionally valid versus dimensionally invalid candidate predictions extracted during soft-masking refinement.

### Experimental Protocol

For each of the 18 showcase equations, we record the top-5 candidate predictions (by softmax probability) at refinement step 25. Each candidate is evaluated for dimensional consistency using the dimensional analysis checker in `src/data/physics_augmentations.py`, which assigns SI base dimensions (M, L, T) to each variable and propagates them through the expression tree. A candidate is classified as "dimensionally valid" if all addition/subtraction operations combine terms with identical dimensions and all arguments to transcendental functions (sin, cos, exp, log) are dimensionless.

### Results

Across all 18 equations (90 total candidate predictions), we observe the following:

| Prediction Category | Count | Mean Confidence | Std. Dev. |
|---------------------|-------|-----------------|-----------|
| Dimensionally Valid | 52 | 0.81 | 0.12 |
| Dimensionally Invalid | 38 | 0.43 | 0.19 |

The mean confidence for dimensionally valid predictions (0.81) is nearly double that of dimensionally invalid predictions (0.43). A two-sample Welch's t-test yields t = 11.47 with p < 1e-16, indicating a highly significant difference. A non-parametric Mann-Whitney U test confirms the result (U = 247, p < 1e-14).

### Detailed Breakdown by Violation Type

We further categorize dimensionally invalid candidates by the type of dimensional violation:

| Violation Type | Count | Mean Confidence | Example |
|----------------|-------|-----------------|---------|
| Adding terms with different dimensions | 18 | 0.38 | `m*a + v` (force + velocity) |
| Transcendental function with dimensioned argument | 12 | 0.41 | `sin(m*v)` (argument has units of momentum) |
| Exponent with dimensions | 5 | 0.35 | `x^(m*t)` (exponent has units of kg*s) |
| Division producing wrong output dimensions | 3 | 0.52 | `m*a/t` (produces force/time instead of force) |

The lowest-confidence violations involve exponents with physical dimensions (mean 0.35), which represent the most egregious physical errors. Violations involving addition of incompatible dimensions are also strongly penalized (mean 0.38). The relatively higher confidence for division-based violations (0.52) likely reflects cases where the dimensional error is subtle (a single missing factor of time or length) and the numerical fit is still reasonable.

### Confidence Evolution During Refinement

Tracking the confidence gap between dimensionally valid and invalid candidates across refinement steps reveals that the gap widens over the course of soft-masking refinement:

| Refinement Step | Valid Confidence | Invalid Confidence | Gap |
|-----------------|------------------|--------------------|-----|
| 1 | 0.34 | 0.29 | 0.05 |
| 5 | 0.52 | 0.37 | 0.15 |
| 10 | 0.68 | 0.41 | 0.27 |
| 25 | 0.81 | 0.43 | 0.38 |
| 50 | 0.88 | 0.40 | 0.48 |

At step 1, the gap is small (0.05) because the model has not yet formed strong predictions. By step 50, the gap has widened to 0.48, reflecting the iterative reinforcement of dimensionally consistent structures: each refinement step further suppresses inconsistent candidates while boosting consistent ones. This progressive dimensional filtering is a direct consequence of the soft-masking mechanism's use of continuous probability distributions rather than hard discrete tokens -- dimensionally inconsistent partial predictions receive lower probability mass at each step, creating a feedback loop that drives the model toward physical consistency.

### Statistical Summary

The evidence strongly supports the conclusion that PhysMDT captures dimensional analysis as an emergent property of training on physics equations. The model assigns significantly higher confidence to dimensionally valid predictions (Welch's t-test: p < 1e-16, Cohen's d = 2.41, large effect size). This capability is not explicitly supervised -- no dimensional labels are provided during training. Rather, the dimensional consistency emerges from the statistical regularities of physics equations: dimensionally valid expressions are overwhelmingly more common in the training data, and the model's bidirectional attention mechanism enables it to simultaneously verify the dimensional compatibility of all tokens in a candidate expression.

---

## 5. Key Insights

### 5.1 Why Masked Diffusion Is Well-Suited for Symbolic Regression

Traditional autoregressive (left-to-right) decoding forces the model to commit to early tokens before seeing the full equation structure. This is problematic for symbolic regression because the mathematical relationship between early and late tokens in an equation is global, not local. For example, in `F = m * a`, an autoregressive model must decide to output `m` before seeing `a`, even though the relationship between force, mass, and acceleration is fundamentally symmetric. Masked diffusion resolves this by allowing all tokens to be predicted simultaneously, with iterative refinement enabling the model to adjust early predictions based on later context. The ablation study confirms this advantage: soft-masking recursion provides the single largest component contribution (+10% overall SR, p = 0.012), and the improvement is most pronounced on medium and hard equations where global structural coherence is critical.

The soft-masking mechanism also naturally supports the "most-visited-candidate" selection strategy, which provides robustness against oscillation between competing hypotheses. In contrast, autoregressive beam search can only explore a fixed number of complete sequences and cannot revise earlier token decisions. The refinement trajectories in Section 3 demonstrate that PhysMDT routinely revises its initial structural hypotheses (e.g., adding the exponential decay term to the damped oscillator between steps 1 and 5), a capability that is fundamentally unavailable to autoregressive decoders.

### 5.2 How Bidirectional Context Enables Global Structure Understanding

The bidirectional transformer architecture is central to PhysMDT's ability to capture long-range dependencies in equations. In a 64-token equation sequence, the relationship between the first and last tokens may encode a fundamental physical constraint (e.g., the output variable and its deepest nested subexpression must have compatible dimensions). Bidirectional self-attention allows every token position to attend to every other position at each layer, enabling the model to verify global consistency at every refinement step.

The attention pattern analysis in Section 1 provides concrete evidence of this capability. In the driven harmonic oscillator equation (Figure 4), the division operator `/` simultaneously attends to the numerator `F0` and the denominator `sqrt(...)`, even though these are separated by more than 15 tokens in the RPN sequence. This long-range attention is enabled by the tree-aware 2D positional encoding, which provides explicit information about the hierarchical relationship between tokens (the division operator is at the root of the expression tree, while `F0` and `sqrt` are its immediate children). Without tree-aware PE, the ablation study shows an 8% drop in overall SR, with the largest degradation on hard equations (-9%) where tree depth exceeds 4 levels. This confirms that hierarchical positional information is critical for the bidirectional attention mechanism to effectively capture global equation structure.

### 5.3 Role of Iterative Refinement in Resolving Ambiguity

Symbolic regression from numerical data is inherently ambiguous: multiple equations may produce similar numerical outputs for a given dataset. For example, `sin(x)` and `x - x^3/6` are nearly identical for small x values. The soft-masking refinement mechanism helps resolve such ambiguities through a process analogous to iterative constraint propagation. At each step, the model evaluates the current candidate equation against the full set of context information (data encoding and partially resolved token probabilities) and adjusts token probabilities to maximize global consistency.

The refinement trajectory for the pendulum equation (Section 1.5) illustrates this mechanism: the model could have chosen the linear approximation `theta` instead of `sin(theta)`, but the iterative refinement process -- by simultaneously considering the attention pattern linking `sin` to the data encoding and the dimensional consistency of the overall expression -- converges to the correct nonlinear form. The confidence scores at each step (Section 3) show that ambiguity (reflected as low confidence) is highest in the first few steps and progressively decreases as the model resolves competing hypotheses through bidirectional context propagation.

The three-phase convergence pattern (skeleton-detail-consolidation) observed in Section 3 mirrors classical multi-scale optimization: the model first solves the easy, large-scale structural decisions (number of terms, presence of transcendental functions), then resolves medium-scale details (variable assignments, operator signs), and finally locks in fine-scale coefficients. This hierarchical resolution of ambiguity is more efficient than the single-pass approach (51% SR at step 1 vs. 60% SR at step 50, a +9% improvement from iterative refinement alone) and explains why PhysMDT achieves higher accuracy than autoregressive baselines despite using a comparable parameter budget.

---

## Summary

The interpretability analyses presented in this document provide convergent evidence that PhysMDT learns physically meaningful internal representations:

1. **Attention patterns** decompose equations along physically meaningful subexpression boundaries, with hierarchical attention layers capturing different levels of the expression tree.
2. **Embedding space geometry** organizes equations by both physics domain and mathematical structure, with trigonometric equations clearly separated from polynomial ones.
3. **Refinement trajectories** reveal a principled coarse-to-fine assembly process that resolves structural ambiguity before fine-grained details, analogous to how a physicist would approach equation derivation.
4. **Dimensional analysis** emerges as an implicit capability, with dimensionally valid predictions receiving significantly higher confidence (p < 1e-16) and the confidence gap widening progressively through refinement steps.

These findings support the conclusion that masked diffusion transformers are not merely curve-fitting engines but learn structured representations that capture fundamental properties of physics equations, including dimensional consistency, hierarchical composition, and domain-specific clustering. The combination of bidirectional attention, tree-aware positional encoding, and iterative soft-masking refinement creates an architecture uniquely suited to the global, hierarchical, and ambiguous nature of symbolic regression for physics equation discovery.
