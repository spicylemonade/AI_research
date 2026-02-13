# Repository Analysis & Project Scaffold

## Existing Files and Directories

| Path | Purpose |
|------|---------|
| `README.md` | Project overview (to be expanded) |
| `.gitignore` | Git ignore rules for secrets, logs, task files |
| `.gitattributes` | Git attributes configuration |
| `research_rubric.json` | Research plan with 28 items across 5 phases — tracks progress |
| `sources.bib` | Bibliography file for all cited references |
| `TASK_researcher_attempt_1.md` | Task instructions for the researcher agent |
| `.archivara/` | Orchestration logs (gitignored) |

## Project Directory Scaffold

| Directory | Purpose |
|-----------|---------|
| `src/` | Core model code — PhysMDT architecture, baseline AR transformer, metrics, refinement loop, token algebra, physics losses, structure predictor, TTF |
| `data/` | Dataset generators, generated datasets, challenge sets, benchmark data |
| `configs/` | Hyperparameter configuration files (YAML/JSON) for training runs |
| `scripts/` | Training, evaluation, and utility entrypoints — `train_baseline.py`, `train_physmdt.py`, `evaluate.py`, `generate_figures.py`, etc. |
| `tests/` | Unit tests for all modules — generator, metrics, tokenizer, model components |
| `notebooks/` | Jupyter notebooks for exploratory analysis and visualization |
| `docs/` | Documentation — literature review, architecture design, analysis reports |
| `results/` | Experimental results as JSON — organized by experiment (baseline_ar/, sr_baseline/, ablations/, etc.) |
| `figures/` | Publication-quality figures as PNG and PDF |

## Directory Creation Status

All directories created:
- `src/` ✓
- `data/` ✓
- `configs/` ✓
- `scripts/` ✓
- `tests/` ✓
- `notebooks/` ✓
- `docs/` ✓
- `results/` ✓
- `figures/` ✓
