#!/bin/bash
# PhysMDT: Full experiment reproduction script
# Usage: bash scripts/run_all.sh
#
# This script reproduces all experiments from the paper.
# Expected runtime: ~2-4 hours on a single GPU (NVIDIA A100 or equivalent)
# All random seeds are fixed (seed=42) for reproducibility.

set -e

echo "=========================================="
echo "PhysMDT: Physics Masked Diffusion Transformer"
echo "Full Experiment Reproduction"
echo "=========================================="

# Step 0: Install dependencies
echo ""
echo "[Step 0] Installing dependencies..."
pip install -r requirements.txt

# Step 1: Run unit tests
echo ""
echo "[Step 1] Running unit tests..."
python -m pytest tests/ -v --timeout=60

# Step 2: Train AR-Baseline
echo ""
echo "[Step 2] Training AR-Baseline on FSReD..."
python scripts/train_baseline.py --epochs 50

# Step 3: Evaluate AR-Baseline
echo ""
echo "[Step 3] Evaluating AR-Baseline on FSReD..."
python scripts/eval_baseline.py

# Step 4: Train PhysMDT-base
echo ""
echo "[Step 4] Training PhysMDT-base..."
python scripts/train_physmdt.py --config configs/physmdt_base.yaml --max-steps 100000

# Step 5: Train PhysMDT-scaled
echo ""
echo "[Step 5] Training PhysMDT-scaled..."
python scripts/train_physmdt.py --config configs/physmdt_scaled.yaml --max-steps 500000

# Step 6: Main experiment evaluation
echo ""
echo "[Step 6] Running main FSReD evaluation..."
python scripts/eval_physmdt.py --epochs 100

# Step 7: Generate figures
echo ""
echo "[Step 7] Generating figures..."
python scripts/generate_figures.py

echo ""
echo "=========================================="
echo "All experiments completed successfully!"
echo "Results saved in results/"
echo "Figures saved in figures/"
echo "=========================================="
