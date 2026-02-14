#!/bin/bash
# PhysDiffuse: Master experiment script
# Reproduces all experiments end-to-end
# Hardware: Single NVIDIA A100-SXM4-40GB
# Expected total runtime: ~1.5 hours

set -e

echo "============================================"
echo "PhysDiffuse: Full Experiment Pipeline"
echo "============================================"
echo ""

# Step 1: Train autoregressive baseline
echo "[1/8] Training autoregressive baseline..."
python3 -u baselines/train_baseline.py 2>&1 | tee results/baseline_training.log
echo "  Baseline training complete."
echo ""

# Step 2: Train PhysDiffuse
echo "[2/8] Training PhysDiffuse..."
python3 -u train_phys_diffuse.py 2>&1 | tee results/phys_diffuse_training.log
echo "  PhysDiffuse training complete."
echo ""

# Step 3: Run ablation study
echo "[3/8] Running ablation study (10 configs)..."
python3 -u run_ablations.py 2>&1 | tee results/ablation_study.log
echo "  Ablation study complete."
echo ""

# Step 4: Run TTT evaluation
echo "[4/8] Running TTT evaluation..."
python3 -u run_ttt_eval.py 2>&1 | tee results/ttt_eval.log
echo "  TTT evaluation complete."
echo ""

# Step 5: Run derivation from scratch
echo "[5/8] Running derivation experiments..."
python3 -u run_derivation.py 2>&1 | tee results/derivation.log
echo "  Derivation experiments complete."
echo ""

# Step 6: Run generalization experiments
echo "[6/8] Running generalization experiments..."
python3 -u run_generalization.py 2>&1 | tee results/generalization.log
echo "  Generalization experiments complete."
echo ""

# Step 7: Run SOTA comparison and error analysis
echo "[7/8] Running analysis..."
python3 -u run_sota_comparison.py 2>&1 | tee results/sota_comparison.log
python3 -u run_error_analysis.py 2>&1 | tee results/error_analysis.log
echo "  Analysis complete."
echo ""

# Step 8: Generate figures
echo "[8/8] Generating figures..."
python3 -u generate_figures.py 2>&1 | tee results/figures.log
echo "  Figures generated."
echo ""

echo "============================================"
echo "All experiments complete!"
echo "Results are in results/"
echo "Figures are in figures/"
echo "============================================"
