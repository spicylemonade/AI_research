"""
Physics equation dataset generator for PhysMDT.

Procedurally generates training data with 50+ distinct Newtonian physics
equation templates spanning 7 families: kinematics, dynamics, energy,
rotational mechanics, gravitation, oscillations, and fluid statics.

Each sample consists of:
    - Symbolic equation in prefix notation
    - Infix equation string
    - Equation family and difficulty level
    - Optional numerical (x, y) data pairs

References:
    - lample2020deep: Deep Learning for Symbolic Mathematics
    - udrescu2020ai: AI Feynman benchmark equations
"""

import json
import os
import sys
import random
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tokenizer import encode, infix_to_prefix, VOCAB_SIZE, MAX_SEQ_LEN

SEED = 42

# ─── Equation Templates ──────────────────────────────────────────────────────
# Each template: (name, family, difficulty, infix_template, variables, coeff_ranges)
# Variables dict maps variable names to sampling ranges for data generation
# coeff_ranges: dict of coefficient names to (low, high) for randomization

EQUATION_TEMPLATES = [
    # ═══ KINEMATICS (family: kinematics) ═══
    # Simple (L1)
    ("uniform_velocity", "kinematics", "simple",
     "{v0} * t", {"t": (0, 10)}, {"v0": (0.5, 20)}),
    ("uniform_accel_displacement", "kinematics", "simple",
     "{v0} * t + 0.5 * {a_val} * t**2", {"t": (0, 10)}, {"v0": (0, 15), "a_val": (0.5, 10)}),
    ("velocity_accel", "kinematics", "simple",
     "{v0} + {a_val} * t", {"t": (0, 10)}, {"v0": (0, 20), "a_val": (0.5, 10)}),
    ("velocity_squared", "kinematics", "simple",
     "{v0}**2 + 2 * {a_val} * x", {"x": (0, 20)}, {"v0": (1, 15), "a_val": (0.5, 10)}),
    ("free_fall", "kinematics", "simple",
     "0.5 * {g_val} * t**2", {"t": (0, 5)}, {"g_val": (9.0, 10.5)}),
    # Medium (L2)
    ("projectile_range", "kinematics", "medium",
     "{v0}**2 * sin(2 * {theta_val}) / {g_val}",
     {}, {"v0": (5, 30), "theta_val": (0.3, 1.2), "g_val": (9.5, 10.0)}),
    ("projectile_height", "kinematics", "medium",
     "{v0} * sin({theta_val}) * t - 0.5 * {g_val} * t**2",
     {"t": (0, 5)}, {"v0": (10, 30), "theta_val": (0.3, 1.2), "g_val": (9.5, 10.0)}),
    ("projectile_x", "kinematics", "medium",
     "{v0} * cos({theta_val}) * t", {"t": (0, 10)}, {"v0": (5, 30), "theta_val": (0.3, 1.2)}),
    # Complex (L3)
    ("projectile_trajectory", "kinematics", "complex",
     "x * tan({theta_val}) - {g_val} * x**2 / (2 * {v0}**2 * cos({theta_val})**2)",
     {"x": (0, 50)}, {"v0": (10, 40), "theta_val": (0.3, 1.2), "g_val": (9.5, 10.0)}),

    # ═══ DYNAMICS (family: dynamics) ═══
    # Simple
    ("newton_second", "dynamics", "simple",
     "{m_val} * {a_val}", {}, {"m_val": (0.5, 20), "a_val": (0.5, 15)}),
    ("weight", "dynamics", "simple",
     "{m_val} * {g_val}", {}, {"m_val": (0.5, 100), "g_val": (9.5, 10.0)}),
    ("hooke_law", "dynamics", "simple",
     "{k_val} * x", {"x": (-5, 5)}, {"k_val": (1, 50)}),
    ("friction", "dynamics", "simple",
     "{mu_val} * {m_val} * {g_val}", {}, {"mu_val": (0.1, 0.9), "m_val": (1, 50), "g_val": (9.5, 10.0)}),
    # Medium
    ("inclined_plane_accel", "dynamics", "medium",
     "{g_val} * (sin({theta_val}) - {mu_val} * cos({theta_val}))",
     {}, {"g_val": (9.5, 10.0), "theta_val": (0.1, 1.3), "mu_val": (0.1, 0.5)}),
    ("spring_mass_accel", "dynamics", "medium",
     "-{k_val} * x / {m_val}", {"x": (-5, 5)}, {"k_val": (1, 50), "m_val": (0.5, 10)}),
    ("drag_force", "dynamics", "medium",
     "0.5 * {rho_val} * {Cd_val} * {A_val} * v**2", {"v": (0, 30)},
     {"rho_val": (1.0, 1.3), "Cd_val": (0.1, 1.0), "A_val": (0.1, 2.0)}),
    ("net_force_two_body", "dynamics", "medium",
     "{m_val} * {a_val} - {mu_val} * {m_val} * {g_val}",
     {}, {"m_val": (1, 20), "a_val": (1, 10), "mu_val": (0.1, 0.5), "g_val": (9.5, 10.0)}),
    # Complex
    ("atwood_accel", "dynamics", "complex",
     "({m1_val} - {m2_val}) * {g_val} / ({m1_val} + {m2_val})",
     {}, {"m1_val": (2, 20), "m2_val": (1, 15), "g_val": (9.5, 10.0)}),
    ("pulley_tension", "dynamics", "complex",
     "2 * {m1_val} * {m2_val} * {g_val} / ({m1_val} + {m2_val})",
     {}, {"m1_val": (1, 20), "m2_val": (1, 20), "g_val": (9.5, 10.0)}),

    # ═══ ENERGY (family: energy) ═══
    # Simple
    ("kinetic_energy", "energy", "simple",
     "0.5 * {m_val} * v**2", {"v": (0, 30)}, {"m_val": (0.5, 50)}),
    ("potential_energy_grav", "energy", "simple",
     "{m_val} * {g_val} * h", {"h": (0, 100)}, {"m_val": (0.5, 50), "g_val": (9.5, 10.0)}),
    ("spring_pe", "energy", "simple",
     "0.5 * {k_val} * x**2", {"x": (-5, 5)}, {"k_val": (1, 100)}),
    ("work", "energy", "simple",
     "{F_val} * x * cos({theta_val})", {"x": (0, 20)}, {"F_val": (1, 50), "theta_val": (0, 1.5)}),
    # Medium
    ("energy_conservation_fall", "energy", "medium",
     "sqrt(2 * {g_val} * h)", {"h": (0, 50)}, {"g_val": (9.5, 10.0)}),
    ("ke_plus_pe", "energy", "medium",
     "0.5 * {m_val} * v**2 + {m_val} * {g_val} * h",
     {"v": (0, 20), "h": (0, 50)}, {"m_val": (0.5, 20), "g_val": (9.5, 10.0)}),
    ("power", "energy", "medium",
     "{F_val} * v", {"v": (0, 30)}, {"F_val": (1, 100)}),
    # Complex
    ("elastic_collision_v1", "energy", "complex",
     "({m1_val} - {m2_val}) * v / ({m1_val} + {m2_val})",
     {"v": (0, 20)}, {"m1_val": (1, 20), "m2_val": (1, 20)}),
    ("elastic_collision_v2", "energy", "complex",
     "2 * {m1_val} * v / ({m1_val} + {m2_val})",
     {"v": (0, 20)}, {"m1_val": (1, 20), "m2_val": (1, 20)}),

    # ═══ ROTATIONAL MECHANICS (family: rotational) ═══
    # Simple
    ("torque", "rotational", "simple",
     "{F_val} * r * sin({theta_val})", {"r": (0.1, 5)},
     {"F_val": (1, 50), "theta_val": (0.3, 1.5)}),
    ("angular_velocity", "rotational", "simple",
     "{omega0_val} + {alpha_val} * t", {"t": (0, 10)},
     {"omega0_val": (0, 10), "alpha_val": (0.5, 5)}),
    ("rotational_ke", "rotational", "simple",
     "0.5 * {I_val} * omega**2", {"omega": (0, 20)}, {"I_val": (0.1, 10)}),
    # Medium
    ("angular_momentum", "rotational", "medium",
     "{I_val} * omega", {"omega": (0, 20)}, {"I_val": (0.1, 10)}),
    ("moment_of_inertia_rod", "rotational", "medium",
     "{m_val} * {L_val}**2 / 12", {}, {"m_val": (0.5, 20), "L_val": (0.5, 5)}),
    ("rolling_ke", "rotational", "medium",
     "0.5 * {m_val} * v**2 + 0.5 * {I_val} * v**2 / {R_val}**2",
     {"v": (0, 15)}, {"m_val": (0.5, 10), "I_val": (0.1, 5), "R_val": (0.1, 2)}),
    # Complex
    ("precession_rate", "rotational", "complex",
     "{m_val} * {g_val} * {r_val} / ({I_val} * omega)",
     {"omega": (1, 20)}, {"m_val": (0.5, 5), "g_val": (9.5, 10.0), "r_val": (0.1, 1), "I_val": (0.01, 1)}),
    ("parallel_axis", "rotational", "complex",
     "{I_cm_val} + {m_val} * {d_val}**2",
     {}, {"I_cm_val": (0.01, 5), "m_val": (0.5, 20), "d_val": (0.1, 3)}),

    # ═══ GRAVITATION (family: gravitation) ═══
    # Simple
    ("gravitational_force", "gravitation", "simple",
     "{G_val} * {m1_val} * {m2_val} / r**2", {"r": (1, 100)},
     {"G_val": (6.67, 6.67), "m1_val": (1e3, 1e6), "m2_val": (1, 1e3)}),
    ("gravitational_pe", "gravitation", "simple",
     "-{G_val} * {m1_val} * {m2_val} / r", {"r": (1, 100)},
     {"G_val": (6.67, 6.67), "m1_val": (1e3, 1e6), "m2_val": (1, 1e3)}),
    # Medium
    ("orbital_velocity", "gravitation", "medium",
     "sqrt({G_val} * {M_val} / r)", {"r": (1, 100)},
     {"G_val": (6.67, 6.67), "M_val": (1e5, 1e8)}),
    ("escape_velocity", "gravitation", "medium",
     "sqrt(2 * {G_val} * {M_val} / r)", {"r": (1, 100)},
     {"G_val": (6.67, 6.67), "M_val": (1e5, 1e8)}),
    ("kepler_third", "gravitation", "medium",
     "(4 * 3.14159**2 / ({G_val} * {M_val}))**(1.0/3.0) * {T_val}**(2.0/3.0)",
     {}, {"G_val": (6.67, 6.67), "M_val": (1e5, 1e8), "T_val": (100, 1e4)}),
    # Complex
    ("grav_field_strength", "gravitation", "complex",
     "{G_val} * {M_val} / r**2", {"r": (1, 100)}, {"G_val": (6.67, 6.67), "M_val": (1e5, 1e8)}),
    ("gravitational_potential", "gravitation", "complex",
     "-{G_val} * {M_val} / r", {"r": (1, 100)}, {"G_val": (6.67, 6.67), "M_val": (1e5, 1e8)}),
    ("two_body_reduced_mass", "gravitation", "complex",
     "{m1_val} * {m2_val} / ({m1_val} + {m2_val})",
     {}, {"m1_val": (1, 100), "m2_val": (1, 100)}),

    # ═══ OSCILLATIONS (family: oscillations) ═══
    # Simple
    ("shm_displacement", "oscillations", "simple",
     "{A_val} * cos({omega_val} * t + {phi_val})", {"t": (0, 10)},
     {"A_val": (0.1, 5), "omega_val": (0.5, 10), "phi_val": (0, 6.28)}),
    ("shm_velocity", "oscillations", "simple",
     "-{A_val} * {omega_val} * sin({omega_val} * t + {phi_val})", {"t": (0, 10)},
     {"A_val": (0.1, 5), "omega_val": (0.5, 10), "phi_val": (0, 6.28)}),
    ("spring_period", "oscillations", "simple",
     "2 * 3.14159 * sqrt({m_val} / {k_val})", {}, {"m_val": (0.1, 10), "k_val": (1, 100)}),
    ("pendulum_period", "oscillations", "simple",
     "2 * 3.14159 * sqrt({L_val} / {g_val})", {}, {"L_val": (0.1, 5), "g_val": (9.5, 10.0)}),
    # Medium
    ("damped_oscillation", "oscillations", "medium",
     "{A_val} * exp(-{gamma_val} * t) * cos({omega_val} * t + {phi_val})",
     {"t": (0, 10)}, {"A_val": (1, 5), "gamma_val": (0.05, 0.5), "omega_val": (1, 10), "phi_val": (0, 6.28)}),
    ("shm_energy", "oscillations", "medium",
     "0.5 * {k_val} * {A_val}**2", {}, {"k_val": (1, 100), "A_val": (0.1, 5)}),
    ("lc_circuit_freq", "oscillations", "medium",
     "1 / (2 * 3.14159 * sqrt({L_val} * {C_val}))",
     {}, {"L_val": (0.001, 1), "C_val": (1e-6, 0.001)}),
    # Complex
    ("driven_oscillator_amplitude", "oscillations", "complex",
     "{F0_val} / ({m_val} * sqrt(({omega0_val}**2 - {omega_val}**2)**2 + (2 * {gamma_val} * {omega_val})**2))",
     {}, {"F0_val": (1, 10), "m_val": (0.5, 5), "omega0_val": (2, 10),
          "omega_val": (1, 12), "gamma_val": (0.1, 1)}),
    ("beat_frequency", "oscillations", "complex",
     "{A_val} * cos(({omega1_val} - {omega2_val}) * t / 2) * cos(({omega1_val} + {omega2_val}) * t / 2)",
     {"t": (0, 20)}, {"A_val": (1, 5), "omega1_val": (5, 15), "omega2_val": (5, 15)}),

    # ═══ FLUID STATICS (family: fluid) ═══
    # Simple
    ("pressure_depth", "fluid", "simple",
     "{P0_val} + {rho_val} * {g_val} * h", {"h": (0, 50)},
     {"P0_val": (1e5, 1.1e5), "rho_val": (900, 1100), "g_val": (9.5, 10.0)}),
    ("buoyancy", "fluid", "simple",
     "{rho_val} * {V_val} * {g_val}", {},
     {"rho_val": (900, 1100), "V_val": (0.001, 1), "g_val": (9.5, 10.0)}),
    ("pascal_law", "fluid", "simple",
     "{F1_val} * {A2_val} / {A1_val}", {},
     {"F1_val": (10, 1000), "A2_val": (0.01, 1), "A1_val": (0.001, 0.5)}),
    # Medium
    ("bernoulli_velocity", "fluid", "medium",
     "sqrt(2 * ({P1_val} - {P2_val}) / {rho_val})", {},
     {"P1_val": (1e5, 2e5), "P2_val": (0.5e5, 1.5e5), "rho_val": (900, 1100)}),
    ("flow_rate", "fluid", "medium",
     "{A_val} * v", {"v": (0, 10)}, {"A_val": (0.001, 0.1)}),
    ("viscous_drag", "fluid", "medium",
     "6 * 3.14159 * {mu_val} * {r_val} * v", {"v": (0, 5)},
     {"mu_val": (0.001, 1), "r_val": (0.001, 0.1)}),
    # Complex
    ("bernoulli_full", "fluid", "complex",
     "{P_val} + 0.5 * {rho_val} * v**2 + {rho_val} * {g_val} * h",
     {"v": (0, 20), "h": (0, 50)},
     {"P_val": (1e5, 1.1e5), "rho_val": (900, 1100), "g_val": (9.5, 10.0)}),
    ("terminal_velocity", "fluid", "complex",
     "sqrt(2 * {m_val} * {g_val} / ({rho_val} * {Cd_val} * {A_val}))",
     {}, {"m_val": (0.01, 10), "g_val": (9.5, 10.0), "rho_val": (1.0, 1.3),
          "Cd_val": (0.1, 1.0), "A_val": (0.01, 1)}),
]

NUM_TEMPLATES = len(EQUATION_TEMPLATES)


def instantiate_equation(template_tuple, rng=None) -> Dict[str, Any]:
    """Instantiate an equation template with random coefficients.

    Returns dict with keys: name, family, difficulty, infix, prefix, coeffs
    """
    if rng is None:
        rng = random.Random(SEED)

    name, family, difficulty, template, variables, coeff_ranges = template_tuple

    # Sample random coefficients
    coeffs = {}
    for cname, (lo, hi) in coeff_ranges.items():
        if lo == hi:
            coeffs[cname] = lo
        else:
            coeffs[cname] = round(rng.uniform(lo, hi), 4)

    # Substitute coefficients into template
    infix = template.format(**coeffs)

    # Convert to prefix notation
    try:
        prefix = infix_to_prefix(infix)
    except Exception:
        prefix = infix.split()

    return {
        "name": name,
        "family": family,
        "difficulty": difficulty,
        "infix": infix,
        "prefix": prefix,
        "coefficients": coeffs,
        "variable_ranges": variables,
    }


def generate_numerical_data(equation_dict: Dict, n_points: int = 20,
                            rng=None) -> Optional[Dict[str, List]]:
    """Generate numerical (x, y) pairs for a given equation.

    Returns dict with 'x' (list of input dicts) and 'y' (list of floats),
    or None if the equation has no free variables.
    """
    if rng is None:
        rng = random.Random(SEED)

    variables = equation_dict["variable_ranges"]
    if not variables:
        return None

    infix = equation_dict["infix"]
    coeffs = equation_dict["coefficients"]

    x_data = []
    y_data = []

    for _ in range(n_points):
        point = {}
        for var, (lo, hi) in variables.items():
            point[var] = rng.uniform(lo, hi)

        # Evaluate equation
        try:
            local_vars = {**point}
            # Replace math functions for eval
            eval_expr = infix
            eval_expr = eval_expr.replace('sin(', 'math.sin(')
            eval_expr = eval_expr.replace('cos(', 'math.cos(')
            eval_expr = eval_expr.replace('tan(', 'math.tan(')
            eval_expr = eval_expr.replace('exp(', 'math.exp(')
            eval_expr = eval_expr.replace('log(', 'math.log(')
            eval_expr = eval_expr.replace('sqrt(', 'math.sqrt(')
            eval_expr = eval_expr.replace('abs(', 'abs(')
            y_val = eval(eval_expr, {"__builtins__": {}, "math": math}, local_vars)
            if math.isfinite(y_val):
                x_data.append(point)
                y_data.append(round(float(y_val), 6))
        except Exception:
            continue

    if len(x_data) < 5:
        return None

    return {"x": x_data, "y": y_data}


def generate_dataset(n_samples: int = 500000, seed: int = SEED,
                     include_numerical: bool = True,
                     n_points: int = 20) -> List[Dict]:
    """Generate a full dataset of physics equations.

    Args:
        n_samples: Total number of samples to generate
        seed: Random seed for reproducibility
        include_numerical: Whether to include numerical data pairs
        n_points: Number of numerical data points per equation

    Returns:
        List of sample dicts
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    samples = []

    samples_per_template = max(1, n_samples // NUM_TEMPLATES)

    for template in EQUATION_TEMPLATES:
        for i in range(samples_per_template):
            sample = instantiate_equation(template, rng)

            if include_numerical:
                num_data = generate_numerical_data(sample, n_points, rng)
                if num_data:
                    sample["numerical_data"] = num_data
                else:
                    sample["numerical_data"] = None

            # Encode for model input
            try:
                sample["token_ids"] = encode(sample["infix"])
            except Exception:
                sample["token_ids"] = None

            samples.append(sample)

            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    # If we haven't hit n_samples yet, keep generating from random templates
    while len(samples) < n_samples:
        template = rng.choice(EQUATION_TEMPLATES)
        sample = instantiate_equation(template, rng)
        if include_numerical:
            num_data = generate_numerical_data(sample, n_points, rng)
            sample["numerical_data"] = num_data
        try:
            sample["token_ids"] = encode(sample["infix"])
        except Exception:
            sample["token_ids"] = None
        samples.append(sample)

    return samples


def split_dataset(samples: List[Dict], train_ratio=0.8, val_ratio=0.1,
                  seed: int = SEED) -> Tuple[List, List, List]:
    """Split dataset into train/val/test with 80/10/10."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def get_family_distribution(samples: List[Dict]) -> Dict[str, int]:
    """Count samples per equation family."""
    counts = {}
    for s in samples:
        fam = s["family"]
        counts[fam] = counts.get(fam, 0) + 1
    return counts


def get_difficulty_distribution(samples: List[Dict]) -> Dict[str, int]:
    """Count samples per difficulty level."""
    counts = {}
    for s in samples:
        d = s["difficulty"]
        counts[d] = counts.get(d, 0) + 1
    return counts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate physics equation dataset")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of samples (default 10000, use 500000 for full)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--include_numerical", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Number of equation templates: {NUM_TEMPLATES}")
    print(f"Generating {args.n_samples} samples with seed {args.seed}...")

    samples = generate_dataset(args.n_samples, args.seed, args.include_numerical)

    print(f"Generated {len(samples)} samples")
    print(f"Family distribution: {get_family_distribution(samples)}")
    print(f"Difficulty distribution: {get_difficulty_distribution(samples)}")

    # Split
    train, val, test = split_dataset(samples, seed=args.seed)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metadata (without full numerical data for space)
    metadata = {
        "n_samples": len(samples),
        "n_templates": NUM_TEMPLATES,
        "family_distribution": get_family_distribution(samples),
        "difficulty_distribution": get_difficulty_distribution(samples),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {args.output_dir}/metadata.json")

    # Save a small preview
    preview = samples[:5]
    for s in preview:
        if s.get("token_ids"):
            s["token_ids"] = s["token_ids"][:20]  # Truncate for preview
    with open(os.path.join(args.output_dir, "preview.json"), 'w') as f:
        json.dump(preview, f, indent=2, default=str)

    print("Preview saved. Done!")
