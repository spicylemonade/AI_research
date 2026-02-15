#!/bin/bash
# Reproduce PARR training from scratch
# Requires: GPU with >=4GB memory, ~2 hours total
set -e

echo "=== PARR Training Pipeline ==="
echo "Step 1/3: Generate training data..."
python -m src.data.generator

echo ""
echo "Step 2/3: Train baseline model..."
python -m src.training.train_baseline

echo ""
echo "Step 3/3: Train PARR model..."
echo "  Phase 1: AR-only training (9 epochs)..."
python -m src.training.train_parr
echo "  Phase 2: Refinement fine-tuning (5 epochs)..."
python scripts/resume_parr_refinement.py

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to checkpoints/"
echo "  - baseline_best.pt"
echo "  - parr_best.pt"
echo "  - parr_final.pt"
