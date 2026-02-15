#!/usr/bin/env bash
# run_all.sh - Full PhysMDT pipeline: data generation -> training -> evaluation -> figures
#
# Usage:
#   ./run_all.sh           # Full mode (50K samples/phase, 10 epochs, ~1 hour)
#   ./run_all.sh --quick   # Smoke-test mode (~5 minutes)
#
# Requirements:
#   - Python 3.8+ with packages from requirements.txt
#   - NVIDIA GPU (A100 recommended, any CUDA GPU works)
#   - ~4 GB GPU memory
#
# Random seed: 42 (configurable via SEED environment variable)

set -euo pipefail

SEED="${SEED:-42}"
QUICK=""
QUICK_FLAG=""

if [[ "${1:-}" == "--quick" ]]; then
    QUICK="1"
    QUICK_FLAG="--quick"
    echo "=== Running in QUICK (smoke-test) mode ==="
else
    echo "=== Running in FULL mode ==="
fi

echo "Seed: $SEED"
echo ""

# Create output directories
mkdir -p checkpoints results figures

# ============================================================
# Step 1: Verify data generation
# ============================================================
echo "Step 1: Verify data generation and tokenizer..."
python -c "
import sys; sys.path.insert(0, '.')
from data.equations import get_training_equations, get_held_out_equations
from data.tokenizer import ExprTokenizer
from data.physics_generator import PhysicsDataset

eqs = get_training_equations()
held = get_held_out_equations()
tok = ExprTokenizer()
print(f'  Training equations: {len(eqs)}')
print(f'  Held-out equations: {len(held)}')
print(f'  Vocabulary size: {len(tok.vocab)}')

# Quick validation
ds = PhysicsDataset(equations=eqs[:5], n_samples=10, seed=$SEED)
print(f'  Sample dataset size: {len(ds)}')
print('  Data generation: OK')
"
echo ""

# ============================================================
# Step 2: Train AR Baseline
# ============================================================
echo "Step 2: Train AR Baseline..."
if [[ -n "$QUICK" ]]; then
    python training/train_baseline.py --quick --seed "$SEED" 2>&1
else
    python training/train_baseline.py --seed "$SEED" 2>&1
fi
echo ""

# ============================================================
# Step 3: Evaluate AR Baseline
# ============================================================
echo "Step 3: Evaluate AR Baseline..."
python training/evaluate_baseline.py 2>&1
echo ""

# ============================================================
# Step 4: Train PhysMDT (curriculum training)
# ============================================================
echo "Step 4: Train PhysMDT..."
if [[ -n "$QUICK" ]]; then
    python training/train_physmdt.py --quick --seed "$SEED" 2>&1
else
    python training/train_physmdt.py \
        --n_samples_per_phase 50000 --n_val 3000 \
        --batch_size 64 --lr 2e-4 --warmup_steps 500 \
        --phase1_epochs 10 --phase2_epochs 10 --phase3_epochs 10 \
        --max_hours 2.0 --log_interval 100 --seed "$SEED" 2>&1
fi
echo ""

# ============================================================
# Step 5: Generate training loss figure
# ============================================================
echo "Step 5: Generate training figures..."
python training/plot_training.py 2>&1
echo ""

# ============================================================
# Step 6: Evaluate PhysMDT (in-distribution + zero-shot)
# ============================================================
echo "Step 6: Evaluate PhysMDT..."
if [[ -n "$QUICK" ]]; then
    python training/evaluate_physmdt.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" --quick 2>&1
else
    python training/evaluate_physmdt.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" 2>&1
fi
echo ""

# ============================================================
# Step 7: Generate comparison figures
# ============================================================
echo "Step 7: Generate comparison figures..."
python training/plot_comparison.py 2>&1
echo ""

# ============================================================
# Step 8: Ablation study
# ============================================================
echo "Step 8: Run ablation study..."
if [[ -n "$QUICK" ]]; then
    python training/run_ablation.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" --quick 2>&1
else
    python training/run_ablation.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" 2>&1
fi
echo ""

# ============================================================
# Step 9: Robustness evaluation
# ============================================================
echo "Step 9: Run robustness evaluation..."
if [[ -n "$QUICK" ]]; then
    python training/run_robustness.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" --quick 2>&1
else
    python training/run_robustness.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" 2>&1
fi
echo ""

# ============================================================
# Step 10: Visualizations
# ============================================================
echo "Step 10: Generate visualizations..."
if [[ -n "$QUICK" ]]; then
    python training/run_visualizations.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" --quick 2>&1
else
    python training/run_visualizations.py --checkpoint checkpoints/physmdt_best.pt --seed "$SEED" 2>&1
fi
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
ls -la results/*.json 2>/dev/null || echo "  (no JSON results)"
echo ""
echo "Figures:"
ls -la figures/*.png 2>/dev/null || echo "  (no PNG figures)"
echo ""
echo "Checkpoints (not tracked by git):"
ls -la checkpoints/*.pt 2>/dev/null || echo "  (no checkpoints)"
echo ""
echo "Done!"
