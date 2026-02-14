# Error Analysis

Model: phys_diffuse
Total: 32 equations, 0 correct (0.0%)

## Failure Mode Categories

### structural_mismatch (15 failures, 47%)

- Mean R²: -0.631
- Mean NED: 0.870

**Examples:**

- **Kinetic Energy** (E_k = 0.5 * m * v^2)
  - Predicted: `mul e_const div mul x_1 x_1 mul x_1 x_1 mul x_0 mul add mul x_0 x_1 x_1 add x_1 inv x_1 x_0 x_1`
  - Ground truth: `mul half mul x_0 pow x_1 int_2`
  - R²=-0.633, NED=0.83, DimOK=True
- **Weight** (W = m * g)
  - Predicted: `add add add x_0 x_0 x_0 mul x_0 x_0 add add mul mul x_0 x_0 x_0 mul x_0 x_0 x_1 x_1 quarter x_0`
  - Ground truth: `mul x_0 x_1`
  - R²=-1.000, NED=0.87, DimOK=False
- **Free-fall distance** (s = 0.5 * g * t^2)
  - Predicted: `sub add add x_1 x_1 mul x_0 mul x_0 x_1 add x_1 mul x_1 mul x_0 x_0 x_0 x_0 div int_3 x_0 c_0 x_1 x_0 x_0 x_1 x_0 x_1 x_1 x_0 x_1 x_1 x_0 x_0`
  - Ground truth: `mul half mul x_0 pow x_1 int_2`
  - R²=-0.973, NED=0.89, DimOK=False

### excessive_nesting (8 failures, 25%)

- Mean R²: -0.875
- Mean NED: 0.857

**Examples:**

- **Distance (uniform velocity)** (s = v * t)
  - Predicted: `mul add add x_1 mul x_1 x_1 mul x_1 x_0 mul x_1 sub mul mul x_1 x_1 x_0 x_1 c_2`
  - Ground truth: `mul x_0 x_1`
  - R²=-1.000, NED=0.85, DimOK=False
- **Newton's 2nd Law** (F = m * a)
  - Predicted: `mul mul mul add x_1 x_1 x_1 mul x_1 x_1 mul x_1 x_1 mul x_1 x_0 x_0 x_1 x_1`
  - Ground truth: `mul x_0 x_1`
  - R²=-1.000, NED=0.84, DimOK=True
- **Impulse** (J = F * t)
  - Predicted: `mul sqrt mul x_1 x_1 mul mul x_1 x_1 mul x_1 mul mul mul x_1 x_1 x_1 x_1 x_1 x_1 x_1`
  - Ground truth: `mul x_0 x_1`
  - R²=-1.000, NED=0.90, DimOK=True

### numerical_instability (6 failures, 19%)

- Mean R²: -0.981
- Mean NED: 0.749

**Examples:**

- **Gravitational PE (near surface)** (U = m * g * h)
  - Predicted: `mul mul add c_1 mul x_2 x_2 add x_2 neg x_2 mul mul x_0 x_0 c_2 quarter x_2 mul x_0`
  - Ground truth: `mul mul x_0 x_1 x_2`
  - R²=-0.884, NED=0.80, DimOK=False
- **Velocity under uniform accel** (v = v0 + a * t)
  - Predicted: `add sqrt x_1 c_0 add add int_1 sub x_0 add mul add mul add x_0 x_1 x_0 add x_2 x_2`
  - Ground truth: `add x_0 mul x_1 x_2`
  - R²=-1.000, NED=0.75, DimOK=False
- **Position under uniform accel** (s = v0*t + 0.5*a*t^2)
  - Predicted: `sub pow mul x_2 mul mul x_2 x_2 sub mul int_3 x_2 mul mul x_2 x_2 int_1 x_2`
  - Ground truth: `add mul x_0 x_2 mul half mul x_1 pow x_2 int_2`
  - R²=-1.000, NED=0.72, DimOK=False

### empty_prediction (2 failures, 6%)

- Mean R²: -1.000
- Mean NED: 1.000

**Examples:**

- **Momentum** (p = m * v)
  - Predicted: ``
  - Ground truth: `mul x_0 x_1`
  - R²=-1.000, NED=1.00, DimOK=False
- **Reduced Mass** (mu = m1*m2/(m1+m2))
  - Predicted: ``
  - Ground truth: `div mul x_0 x_1 add x_0 x_1`
  - R²=-1.000, NED=1.00, DimOK=False

### wrong_operator (1 failures, 3%)

- Mean R²: -0.543
- Mean NED: 0.714

**Examples:**

- **Relativistic KE approx** (E = 0.5*m*v^2 + (3/8)*m*v^4/c^2)
  - Predicted: `mul mul int_0 x_0 x_0 mul mul x_0 e_const x_1 x_0 add mul mul x_0 x_0 x_0 x_0 div x_0 x_0`
  - Ground truth: `add mul half mul x_0 pow x_1 int_2 div mul mul c_0 x_0 pow x_1 int_4 pow x_2 int_2`
  - R²=-0.543, NED=0.71, DimOK=True

## Proposed Mitigations

- **excessive_nesting**: Use tree depth penalty during generation; increase training examples with deep nesting
- **empty_prediction**: Requires further investigation
- **structural_mismatch**: Increase model capacity (more layers); longer training with curriculum learning
- **numerical_instability**: Better input normalization; add numerical stability loss term
- **wrong_operator**: Improve operator coverage in training data; use MCTS-guided selection for operator choices

## Complexity-Success Correlation

| Tier | Total | Success | Rate |
|------|-------|---------|------|
| 1 | 8 | 0 | 0% |
| 2 | 10 | 0 | 0% |
| 3 | 8 | 0 | 0% |
| 4 | 6 | 0 | 0% |