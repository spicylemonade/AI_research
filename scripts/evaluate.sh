#!/bin/bash
# Reproduce all evaluation results
# Requires: trained checkpoints in checkpoints/
set -e

echo "=== PARR Evaluation Pipeline ==="

echo ""
echo "Step 1/6: Full PARR evaluation..."
python scripts/evaluate_parr.py

echo ""
echo "Step 2/6: Ablation study..."
python scripts/run_ablation.py

echo ""
echo "Step 3/6: Head-to-head comparison..."
python scripts/run_comparison.py

echo ""
echo "Step 4/6: Robustness evaluation..."
python scripts/run_robustness.py

echo ""
echo "Step 5/6: Qualitative analysis..."
python scripts/run_qualitative.py

echo ""
echo "Step 6/6: Efficiency analysis..."
python scripts/run_efficiency.py

echo ""
echo "=== Generating figures ==="
python scripts/generate_figures.py

echo ""
echo "=== Evaluation complete ==="
echo "Results saved to results/"
echo "Figures saved to figures/"
