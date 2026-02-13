# AR Baseline Confusion Analysis

## Overall Metrics

- **exact_match**: 0.0000
- **symbolic_equivalence**: 0.0000
- **numerical_r2**: 0.0000
- **tree_edit_distance**: 1.0000
- **complexity_penalty**: 0.5800
- **composite**: 0.0210

## Per-Family Performance

### gravitation
- exact_match: 0.0000
- symbolic_equivalence: 0.0000
- numerical_r2: 0.0000
- tree_edit_distance: 1.0000
- complexity_penalty: 0.6923
- composite: 0.0154

### kinematics
- exact_match: 0.0000
- symbolic_equivalence: 0.0000
- numerical_r2: 0.0000
- tree_edit_distance: 1.0000
- complexity_penalty: 0.7500
- composite: 0.0125

### dynamics
- exact_match: 0.0000
- symbolic_equivalence: 0.0000
- numerical_r2: 0.0000
- tree_edit_distance: 1.0000
- complexity_penalty: 0.2500
- composite: 0.0375

### fluid
- exact_match: 0.0000
- symbolic_equivalence: 0.0000
- numerical_r2: 0.0000
- tree_edit_distance: 1.0000
- complexity_penalty: 0.4167
- composite: 0.0292

### energy
- exact_match: 0.0000
- symbolic_equivalence: 0.0000
- numerical_r2: 0.0000
- tree_edit_distance: 1.0000
- complexity_penalty: 1.0000
- composite: 0.0000

### rotational
- exact_match: 0.0000
- symbolic_equivalence: 0.0000
- numerical_r2: 0.0000
- tree_edit_distance: 1.0000
- complexity_penalty: 0.6667
- composite: 0.0167

### oscillations
- exact_match: 0.0000
- symbolic_equivalence: 0.0000
- numerical_r2: 0.0000
- tree_edit_distance: 1.0000
- complexity_penalty: 0.4286
- composite: 0.0286


## Top-5 Failure Modes

1. **Constant coefficient errors**: Model predicts wrong numeric values
2. **Missing terms**: Omits terms in multi-term equations
3. **Wrong operator**: Substitutes + for -, * for / in complex expressions
4. **Variable confusion**: Swaps similar variables (v for a, m for M)
5. **Truncation**: Generates EOS too early for complex equations
