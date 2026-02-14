# Challenge Set: Qualitative Examples

## 5 Best Predictions

| # | Template | Family | Composite | EM | SE | Ground Truth | Prediction |
|---|----------|--------|-----------|----|----|--------------|------------|
| 1 | energy_conservation | energy | 7.13 | 0 | 0 | `add mul div INT_1 INT_2 mul m pow v INT_2 mul m mul g_accel ` | `m m div INT_1 INT_2 pow pow mul INT_1 INT_2 INT_2 mul m m ad` |
| 2 | kepler_3rd_simple | gravitation | 5.49 | 0 | 0 | `div pow r INT_3 pow mul div INT_1 mul INT_2 pi mul G_const m` | `G_const pow pow INT_3 INT_3 INT_3 pow pow INT_3 INT_3 pi mul` |
| 3 | kepler_3rd_simple | gravitation | 5.49 | 0 | 0 | `div pow r INT_3 pow mul div INT_1 mul INT_2 pi mul G_const m` | `G_const pow pow INT_3 INT_3 INT_3 pow pow INT_3 INT_3 pi mul` |
| 4 | orbital_period | gravitation | 4.81 | 0 | 0 | `mul mul INT_2 pi sqrt div pow r INT_3 mul G_const m` | `G_const pow pow INT_3 INT_3 INT_3 pow pow INT_3 INT_3 pi mul` |
| 5 | energy_conservation | energy | 4.63 | 0 | 0 | `add mul div INT_1 INT_2 mul m pow v INT_2 mul m mul g_accel ` | `m m div INT_1 pow pow pow pow INT_1 INT_2 INT_2 mul m m m di` |

## 5 Worst Predictions

| # | Template | Family | Composite | EM | SE | Ground Truth | Prediction |
|---|----------|--------|-----------|----|----|--------------|------------|
| 1 | spring_period | oscillations | 0.85 | 0 | 0 | `mul mul INT_2 pi sqrt div m k_spring` | `add add div INT_1 pow pow pow mul INT_1 INT_2 mul mul mul ad` |
| 2 | spring_period | oscillations | 0.85 | 0 | 0 | `mul mul INT_2 pi sqrt div m k_spring` | `add add div INT_1 pow pow pow mul INT_1 INT_2 mul mul mul ad` |
| 3 | torricelli | fluid_statics | 0.64 | 0 | 0 | `sqrt mul INT_2 mul g_accel h` | `add add add add h sqrt P_pressure add add pi pi mul add add ` |
| 4 | torricelli | fluid_statics | 0.64 | 0 | 0 | `sqrt mul INT_2 mul g_accel h` | `add add add add h sqrt sqrt add add pi pi mul add add add ad` |
| 5 | orbital_velocity | gravitation | 0.43 | 0 | 0 | `sqrt div mul G_const m r` | `add add div INT_1 pow pow pow mul INT_1 INT_2 mul mul mul ad` |

---
Total equations evaluated: 20
Symbolic equivalence rate: 0.0%
