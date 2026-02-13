# Statistical Significance Tests & Error Analysis

Item 023 of the research rubric.

## 1. Statistical Comparison: PhysMDT vs AR Baseline

- **Seeds**: [42, 43, 44, 45, 46]
- **Dataset**: 200 samples, 30 test samples per run
- **Model size**: d_model=64, n_layers=2, n_heads=2
- **Training**: 3 epochs, 30s timeout, batch_size=32
- **scipy available**: True

| Metric | AR Baseline (mean +/- std) | PhysMDT (mean +/- std) | t-test p | Wilcoxon p |
|--------|---------------------------|------------------------|----------|------------|
| exact_match | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | N/A | N/A |
| symbolic_equivalence | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | N/A | N/A |
| numerical_r2 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | N/A | N/A |
| tree_edit_distance | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | N/A | N/A |
| complexity_penalty | 0.8400 +/- 0.1475 | 0.7500 +/- 0.0000 | 0.2441 | 0.2763 |
| composite | 0.0080 +/- 0.0074 | 0.0125 +/- 0.0000 | 0.2441 | 0.2763 |

### Per-Run Composite Scores

| Seed | AR Baseline | PhysMDT |
|------|-------------|---------|
| 42 | 0.0000 | 0.0125 |
| 43 | 0.0125 | 0.0125 |
| 44 | 0.0125 | 0.0125 |
| 45 | 0.0150 | 0.0125 |
| 46 | 0.0000 | 0.0125 |

## 2. Failure Categories

Failure categories are determined heuristically by comparing predicted
and ground-truth equation strings. A sample is classified as a failure
when its composite score is below 0.5.

| Category | Count | Description |
|----------|-------|-------------|
| truncation | 20 | Prediction is very short (< 30% of ground truth length). |

## 3. Examples of Correct Derivations (top 5)

No predictions achieved composite >= 0.5 in the sampled run.

## 4. Instructive Failures (top 5)

### Failure Example 1
- **Equation name**: pulley_tension
- **Family / difficulty**: dynamics / complex
- **Ground truth**: `2 * 6.2174 * 2.4722 * 9.6429 / (6.2174 + 2.4722)`
- **Prediction**:   `C0`
- **Composite score**: 0.05
- **Failure category**: truncation

### Failure Example 2
- **Equation name**: lc_circuit_freq
- **Family / difficulty**: oscillations / medium
- **Ground truth**: `1 / (2 * 3.14159 * sqrt(0.5282 * 0.0001))`
- **Prediction**:   `C0`
- **Composite score**: 0.05
- **Failure category**: truncation

### Failure Example 3
- **Equation name**: spring_period
- **Family / difficulty**: oscillations / simple
- **Ground truth**: `2 * 3.14159 * sqrt(2.9208 / 75.1051)`
- **Prediction**:   `C0`
- **Composite score**: 0.05
- **Failure category**: truncation

### Failure Example 4
- **Equation name**: newton_second
- **Family / difficulty**: dynamics / simple
- **Ground truth**: `17.7206 * 13.607`
- **Prediction**:   `C0`
- **Composite score**: 0.05
- **Failure category**: truncation

### Failure Example 5
- **Equation name**: buoyancy
- **Family / difficulty**: fluid / simple
- **Ground truth**: `1027.5025 * 0.3639 * 9.9005`
- **Prediction**:   `C0`
- **Composite score**: 0.05
- **Failure category**: truncation

## 5. Summary

- AR baseline mean composite: **0.0080**
- PhysMDT mean composite:     **0.0125**
- Difference (PhysMDT - AR):  **+0.0045**
- Paired t-test p-value (composite): **0.2441**

Total runtime: 48.0s
