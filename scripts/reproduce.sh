#!/bin/bash
set -e
echo "=== PhysMDT Reproducibility Script ==="

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
python -m pytest tests/ -v

# 3. Generate dataset and train baseline
python scripts/train_baseline.py

# 4. Evaluate SR baselines
python scripts/eval_sr_baseline.py

# 5. Train PhysMDT
python scripts/train_phys_mdt.py

# 6. Run ablation evaluation
python scripts/eval_quick.py

# 7. Benchmark evaluation
python scripts/eval_benchmarks.py

# 8. Challenge set evaluation
python scripts/eval_challenge.py

# 9. Refinement depth study
python scripts/refinement_depth_study.py

# 10. Embedding analysis
python scripts/analyze_embeddings.py

# 11. Statistical tests
python scripts/statistical_tests.py

# 12. Generate figures
python scripts/generate_figures.py

echo "=== All experiments complete ==="
echo "Results in: results/"
echo "Figures in: figures/"
