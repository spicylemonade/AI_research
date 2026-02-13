# State-of-the-Art Benchmarks for Physics Equation Discovery

## 1. AI Feynman Dataset (FSRD)

- **Source**: Udrescu & Tegmark (2020), Science Advances
- **Number of equations**: 100 equations from the Feynman Lectures on Physics + 20 additional equations (120 total in v2.0)
- **Complexity distribution**: Ranges from simple (1-2 variables, e.g., Coulomb's law) to complex (up to 9 variables with nested transcendental functions). Covers classical mechanics, electromagnetism, quantum mechanics, and thermodynamics.
- **Input/output format**: Tabular CSV data with randomly sampled variable values and corresponding function outputs. 6 GB database (FSReD).
- **Best published accuracy**: AI Feynman 2.0 solves all 100 original equations and achieves 90% on the extended difficulty set. NeSymReS (Biggio et al. 2021) matches within 3 orders of magnitude less time on the original 100. PySR achieves ~85% on the original set.
- **Citation**: `udrescu2020ai`, `udrescu2020ai2`

## 2. SRBench

- **Source**: La Cava et al. (2021), NeurIPS Datasets & Benchmarks Track
- **Number of equations**: 252 regression problems (130 ground-truth synthetic + 122 real-world datasets)
- **Complexity distribution**: Synthetic problems range from simple polynomial to complex compositions of transcendental functions. Real-world datasets span biology, physics, economics, and engineering.
- **Input/output format**: Scikit-learn compatible tabular data with train/test splits
- **Best published accuracy**: On ground-truth problems, Operon (GP-based) and PySR achieve highest exact recovery rates (~60-70%). On real-world problems, GP methods (Operon, FEAT, EPLEX) achieve best Pareto front of accuracy vs. complexity. Neural methods (DSR, NeSymReS) perform comparably for large compute budgets.
- **Citation**: `lacava2021srbench`

## 3. Strogatz Dataset (ODE-Strogatz)

- **Source**: La Cava et al. (2016), adapted from Strogatz "Nonlinear Dynamics and Chaos"
- **Number of equations**: 7 two-dimensional ODE systems
- **Complexity distribution**: All systems are 2-state first-order ODEs exhibiting chaotic and/or nonlinear behavior. Includes Lotka-Volterra, Van der Pol oscillator, glycolytic oscillator, etc.
- **Input/output format**: Time-series data from MATLAB/Simulink simulations with specified initial conditions
- **Best published accuracy**: ODEFormer (2023) achieves highest symbolic equivalence rate, consistently outperforming SINDy and GP-based methods. ODEFormer demonstrates improved robustness to noise and irregular sampling.
- **Citation**: `lacava2016strogatz`

## 4. Nguyen Benchmark

- **Source**: Nguyen Quang Uy et al. (2011), Genetic Programming and Evolvable Machines
- **Number of equations**: 12 equations (Nguyen-1 through Nguyen-12)
- **Complexity distribution**: 8 single-variable equations (polynomials, trigonometric, logarithmic, square root) and 4 two-variable equations. Difficulty ranges from simple polynomial (x³+x²+x) to moderately complex (sin(x²)·cos(x)−1, x^y).
- **Input/output format**: Tabular data with variables sampled from uniform distributions over small ranges (e.g., U[-1,1] or U[0,2])
- **Best published accuracy**: Most modern methods (PySR, NeSymReS, DSR, TPSR) solve all 12 equations. NeSymReS achieves this in ~4.4 minutes per equation. TPSR with MCTS guidance achieves 100% on all 12 benchmarks with improved sample efficiency.
- **Citation**: `nguyen2011benchmark`

## 5. ODEBench

- **Source**: d'Ascoli et al. (2023), introduced with ODEFormer (ICLR 2024)
- **Number of equations**: 63 ODE systems, ranging from 1 to 4 state variables
- **Complexity distribution**: Curated from textbooks and literature, covers ecology (Lotka-Volterra), chemistry (reaction kinetics), physics (oscillators, pendulums), biology (population dynamics), and epidemiology (SIR models). More comprehensive than the original Strogatz dataset.
- **Input/output format**: Single solution trajectories with configurable noise levels and sampling irregularity
- **Best published accuracy**: ODEFormer achieves state-of-the-art symbolic recovery on this benchmark. MDBench (2025) further evaluated 12 algorithms, finding that linear methods and GP methods achieve lowest prediction error for ODEs respectively.
- **Citation**: `dascoli2023odeformer`

## 6. Recent Developments (2024-2026)

- **MDBench (2025)**: Comprehensive open-source benchmarking framework evaluating 12 model discovery algorithms on 14 PDEs and 63 ODEs under varying noise levels.
- **TPSR (NeurIPS 2023)**: Transformer-based Planning for SR using MCTS, evaluated on AI Feynman and Nguyen benchmarks.
- **Controllable NSR (2023)**: Neural SR with hypothesis integration, enabling prior-knowledge-guided equation discovery.
- **LLM-based equation discovery (2024)**: Using natural language prompts to guide LLMs in extracting governing equations from data.

## Evaluation Landscape Summary

| Benchmark | # Equations | Complexity | Best Method | Our Target |
|-----------|-------------|-----------|-------------|------------|
| AI Feynman | 120 | Low-High | AI Feynman 2.0 (100%) | Match or exceed NeSymReS |
| Nguyen | 12 | Low-Medium | All modern methods (~100%) | Must solve all 12 |
| Strogatz | 7 ODE systems | Medium | ODEFormer | Competitive recovery |
| SRBench | 252 | Variable | Operon/PySR (GP) | Top-3 on synthetic |
| ODEBench | 63 ODE systems | Medium-High | ODEFormer | Competitive on 1-2D |

Our PhysMDT model must demonstrate clear advantages on physics-specific equations (AI Feynman) and novel Newtonian systems beyond what existing methods handle.
