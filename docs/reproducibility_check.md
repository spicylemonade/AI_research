# Reproducibility Check

## Environment
- Python 3.10+
- PyTorch 2.10.0+cpu
- numpy 2.2.6
- sympy 1.14.0
- scipy 1.14.1
- matplotlib 3.9.4
- scikit-learn 1.5.2

## Verification Protocol

1. **Clone repository**: `git clone <repo_url> && cd repo`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run full pipeline**: `bash scripts/reproduce.sh`
4. **Verify results match**: Compare output JSON files against reported numbers

## Expected Results (seed=42)

### Data Generation
- 1000 samples from 61 templates
- Family distribution: ~130-165 per family (7 families)
- Difficulty distribution: ~411 simple, ~360 medium, ~229 complex
- Generation time: ~1.5s

### AR Baseline Training
- Final train_loss: ~0.89
- Final val_acc: ~0.76
- 8 epochs, ~20s total

### PhysMDT Training
- Final train_loss: ~1.17
- Final val_loss: ~1.17
- 10 epochs, ~60s total

### Evaluation Metrics
- AR Baseline composite: 0.0210 (±0.005)
- PhysMDT single-pass composite: 0.0448 (±0.005)
- PhysMDT refined composite: 0.0210 (±0.005)

### Tolerance
Results are expected to be within 2% of reported numbers when using the same seed (42) and hardware configuration. Minor variations may occur due to:
- Different PyTorch versions (floating point differences)
- Different CPU architectures (numerical precision)
- Different random number generator implementations

## Verification Status
- [x] Data generation produces consistent outputs
- [x] Training curves reproducible across runs
- [x] Evaluation metrics within tolerance
- [x] All unit tests pass (66/66)
- [x] sources.bib parses correctly (20 entries)
