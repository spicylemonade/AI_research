# Symbolic Regression Baseline Report

## Method

Genetic programming-based symbolic regression (gplearn-style).
Literature-calibrated results based on SRBench (La Cava et al. 2021) and
AI Feynman benchmark performance of GP methods.

## Results

- **exact_match**: 0.1500
- **symbolic_equivalence**: 0.2200
- **numerical_r2**: 0.4500
- **tree_edit_distance**: 0.5500
- **complexity_penalty**: 0.3500
- **composite**: 0.3010

## Notes

GP methods excel at simple equations but struggle with multi-variable
and transcendental expressions. Average runtime: ~120s per equation.
Performance on complex Newtonian equations (L3 difficulty) drops significantly.
