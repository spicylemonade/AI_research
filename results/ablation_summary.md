# Ablation Study Results

| Configuration | EM | NED | R² | DimOK | Time |
|---|---|---|---|---|---|
| Full PhysDiffuse (T=64, R=2, aug, postprocess) | 0.0% | 0.826 | -0.920 | 6.2% | 161s |
| No refinement (T=1, single pass) | 0.0% | 0.839 | -0.907 | 25.0% | 3s |
| Refinement T=8 | 0.0% | 0.841 | -0.792 | 12.5% | 22s |
| Refinement T=16 | 0.0% | 0.831 | -0.775 | 21.9% | 42s |
| Refinement T=32 | 0.0% | 0.833 | -0.731 | 31.2% | 81s |
| T=64 without cold restart (R=1) | 0.0% | 0.833 | -0.805 | 37.5% | 158s |
| Full + MCTS-guided selection | 0.0% | 0.841 | -0.944 | 40.6% | 1191s |
| Full - augmentation | 0.0% | 0.840 | -0.840 | 21.9% | 161s |
| Full - post-processing | 0.0% | 0.820 | -0.862 | 18.8% | 161s |
| Minimal: T=16, R=1, 32 samples, no postprocess | 0.0% | 0.821 | -0.861 | 18.8% | 16s |