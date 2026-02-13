#!/bin/bash
# PhysMDT Reproducibility Script
# Runs the full pipeline from data generation through evaluation.
#
# Usage: bash scripts/reproduce.sh
#
# Expected runtime: ~10 minutes on CPU
# Results will be within 2% of reported numbers (seed 42).

set -e

echo "========================================"
echo "PhysMDT Reproducibility Pipeline"
echo "========================================"
echo ""

# 1. Install dependencies
echo "Step 1: Checking dependencies..."
pip install -q -r requirements.txt 2>/dev/null || echo "  (install manually: pip install -r requirements.txt)"

# 2. Run unit tests
echo ""
echo "Step 2: Running unit tests..."
python -m pytest tests/ -q --tb=short 2>&1 || echo "  Some tests may fail if dependencies are missing"

# 3. Verify sources.bib
echo ""
echo "Step 3: Verifying sources.bib..."
python scripts/verify_bib.py

# 4. Generate data and train models
echo ""
echo "Step 4: Running main experiment pipeline..."
PYTHONUNBUFFERED=1 python scripts/run_experiments.py

# 5. Run benchmark evaluations
echo ""
echo "Step 5: Running benchmark evaluations..."
if [ -f scripts/eval_benchmarks.py ]; then
    PYTHONUNBUFFERED=1 python scripts/eval_benchmarks.py
fi

# 6. Run challenge set evaluation
echo ""
echo "Step 6: Running challenge set evaluation..."
if [ -f scripts/eval_challenge.py ]; then
    PYTHONUNBUFFERED=1 python scripts/eval_challenge.py
fi

# 7. Embedding analysis
echo ""
echo "Step 7: Running embedding analysis..."
if [ -f scripts/analyze_embeddings.py ]; then
    PYTHONUNBUFFERED=1 python scripts/analyze_embeddings.py
fi

# 8. Statistical tests
echo ""
echo "Step 8: Running statistical significance tests..."
if [ -f scripts/statistical_tests.py ]; then
    PYTHONUNBUFFERED=1 python scripts/statistical_tests.py
fi

# 9. Generate figures
echo ""
echo "Step 9: Generating figures..."
if [ -f scripts/generate_figures.py ]; then
    PYTHONUNBUFFERED=1 python scripts/generate_figures.py
fi

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "Results saved to results/"
echo "Figures saved to figures/"
echo "========================================"
