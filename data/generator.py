#!/usr/bin/env python3
"""Physics equation dataset generator for PhysMDT.

Generates equations in prefix notation with random coefficients and paired
numerical observations across 7 Newtonian mechanics families at 3 difficulty levels.
"""

import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np


# ---------------------------------------------------------------------------
# Equation template dataclass
# ---------------------------------------------------------------------------

@dataclass
class EquationTemplate:
    """A single equation template."""
    name: str
    family: str  # kinematics, dynamics, energy, rotational, gravitation, oscillations, fluid_statics
    difficulty: str  # simple, medium, complex
    prefix_notation: str  # prefix tokens with {coeff} placeholders
    infix_readable: str  # human-readable form
    variables: List[str]  # list of variable token names used
    coeff_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    eval_func: Optional[str] = None  # Python expression for evaluation

    def generate_coefficients(self, rng: random.Random) -> Dict[str, float]:
        """Generate random coefficients within defined ranges."""
        coeffs = {}
        for name, (lo, hi) in self.coeff_ranges.items():
            coeffs[name] = rng.uniform(lo, hi)
        return coeffs

    def instantiate_prefix(self, coeffs: Dict[str, float]) -> str:
        """Fill in coefficient placeholders in prefix notation."""
        result = self.prefix_notation
        for name, val in coeffs.items():
            result = result.replace(f'{{{name}}}', format_constant(val))
        return result

    def evaluate(self, coeffs: Dict[str, float], var_values: Dict[str, float]) -> float:
        """Evaluate the equation given coefficients and variable values."""
        namespace = {**coeffs, **var_values, 'math': math, 'np': np,
                     'pi': math.pi, 'g': 9.81, 'G': 6.674e-11}
        try:
            return float(eval(self.eval_func, {"__builtins__": {}}, namespace))
        except (ZeroDivisionError, ValueError, OverflowError):
            return float('nan')


def format_constant(val: float) -> str:
    """Format a float as a constant token sequence."""
    if val == int(val) and 0 <= val <= 9:
        return f"INT_{int(val)}"
    if abs(val - math.pi) < 1e-10:
        return "pi"
    if abs(val - math.e) < 1e-10:
        return "euler"
    if abs(val - 9.81) < 1e-6:
        return "g_accel"

    sign = "C_+" if val >= 0 else "C_-"
    aval = abs(val)
    if aval == 0:
        return "INT_0"

    exp_val = int(math.floor(math.log10(aval))) if aval > 0 else 0
    mantissa = aval / (10 ** exp_val) if exp_val != 0 else aval

    mant_str = f"{mantissa:.4f}".rstrip('0').rstrip('.')
    digits = []
    for ch in mant_str:
        if ch == '.':
            digits.append("DOT")
        else:
            digits.append(f"D_{ch}")

    exp_sign = "E_+" if exp_val >= 0 else "E_-"
    exp_digits = [f"E_{d}" for d in str(abs(exp_val))]

    return " ".join(["CONST_START", sign] + digits + [exp_sign] + exp_digits + ["CONST_END"])


# ---------------------------------------------------------------------------
# Define 60+ equation templates across 7 families
# ---------------------------------------------------------------------------

def build_templates() -> List[EquationTemplate]:
    """Build the complete set of equation templates."""
    templates = []

    # ===== FAMILY 1: KINEMATICS =====
    templates.extend([
        EquationTemplate("uniform_velocity", "kinematics", "simple",
                         "mul v t", "x = v*t", ["v", "t"],
                         {"c_v": (0.5, 10.0)},
                         "c_v * var_t"),
        EquationTemplate("uniform_accel_v", "kinematics", "simple",
                         "add v0 mul a t", "v = v0 + a*t", ["v0", "a", "t"],
                         {"c_v0": (0.0, 5.0), "c_a": (0.5, 10.0)},
                         "c_v0 + c_a * var_t"),
        EquationTemplate("uniform_accel_x", "kinematics", "medium",
                         "add mul v0 t mul mul div INT_1 INT_2 a pow t INT_2",
                         "x = v0*t + 0.5*a*t^2", ["v0", "a", "t"],
                         {"c_v0": (0.0, 5.0), "c_a": (0.5, 10.0)},
                         "c_v0 * var_t + 0.5 * c_a * var_t**2"),
        EquationTemplate("free_fall_dist", "kinematics", "simple",
                         "mul mul div INT_1 INT_2 g_accel pow t INT_2",
                         "d = 0.5*g*t^2", ["t"],
                         {},
                         "0.5 * 9.81 * var_t**2"),
        EquationTemplate("free_fall_vel", "kinematics", "simple",
                         "mul g_accel t", "v = g*t", ["t"],
                         {},
                         "9.81 * var_t"),
        EquationTemplate("range_projectile", "kinematics", "complex",
                         "div mul pow v INT_2 sin mul INT_2 theta g_accel",
                         "R = v^2 * sin(2*theta) / g", ["v", "theta"],
                         {"c_v": (5.0, 20.0)},
                         "c_v**2 * math.sin(2 * var_theta) / 9.81"),
        EquationTemplate("max_height_proj", "kinematics", "medium",
                         "div mul pow v INT_2 pow sin theta INT_2 mul INT_2 g_accel",
                         "h = v^2*sin^2(theta)/(2g)", ["v", "theta"],
                         {"c_v": (5.0, 20.0)},
                         "c_v**2 * math.sin(var_theta)**2 / (2 * 9.81)"),
        EquationTemplate("time_of_flight", "kinematics", "medium",
                         "div mul INT_2 mul v sin theta g_accel",
                         "T = 2*v*sin(theta)/g", ["v", "theta"],
                         {"c_v": (5.0, 20.0)},
                         "2 * c_v * math.sin(var_theta) / 9.81"),
        EquationTemplate("vel_squared", "kinematics", "medium",
                         "add pow v0 INT_2 mul mul INT_2 a d",
                         "v^2 = v0^2 + 2*a*d", ["v0", "a", "d"],
                         {"c_v0": (1.0, 5.0), "c_a": (0.5, 5.0)},
                         "c_v0**2 + 2 * c_a * var_d"),
        EquationTemplate("avg_velocity", "kinematics", "simple",
                         "div add v0 v INT_2",
                         "v_avg = (v0+v)/2", ["v0", "v"],
                         {"c_v0": (1.0, 5.0), "c_v": (1.0, 10.0)},
                         "(c_v0 + c_v) / 2"),
    ])

    # ===== FAMILY 2: DYNAMICS (Newton's Laws) =====
    templates.extend([
        EquationTemplate("newton_2nd", "dynamics", "simple",
                         "mul m a", "F = m*a", ["m", "a"],
                         {"c_m": (0.5, 10.0), "c_a": (0.5, 10.0)},
                         "c_m * c_a"),
        EquationTemplate("weight", "dynamics", "simple",
                         "mul m g_accel", "W = m*g", ["m"],
                         {"c_m": (0.5, 50.0)},
                         "c_m * 9.81"),
        EquationTemplate("friction", "dynamics", "simple",
                         "mul mu mul m g_accel", "F_f = mu*m*g", ["mu", "m"],
                         {"c_mu": (0.1, 0.9), "c_m": (1.0, 20.0)},
                         "c_mu * c_m * 9.81"),
        EquationTemplate("spring_force", "dynamics", "simple",
                         "mul neg k_spring x", "F = -k*x", ["k_spring", "x"],
                         {"c_k": (1.0, 100.0)},
                         "-c_k * var_x"),
        EquationTemplate("normal_incline", "dynamics", "medium",
                         "mul m mul g_accel cos theta",
                         "N = m*g*cos(theta)", ["m", "theta"],
                         {"c_m": (1.0, 20.0)},
                         "c_m * 9.81 * math.cos(var_theta)"),
        EquationTemplate("net_force_incline", "dynamics", "complex",
                         "sub mul m mul g_accel sin theta mul mu mul m mul g_accel cos theta",
                         "F_net = mg*sin(theta) - mu*mg*cos(theta)", ["m", "theta", "mu"],
                         {"c_m": (1.0, 20.0), "c_mu": (0.1, 0.5)},
                         "c_m * 9.81 * math.sin(var_theta) - c_mu * c_m * 9.81 * math.cos(var_theta)"),
        EquationTemplate("centripetal_force", "dynamics", "medium",
                         "div mul m pow v INT_2 r",
                         "F_c = m*v^2/r", ["m", "v", "r"],
                         {"c_m": (1.0, 10.0), "c_v": (1.0, 10.0)},
                         "c_m * c_v**2 / var_r"),
        EquationTemplate("drag_force", "dynamics", "complex",
                         "mul mul div INT_1 INT_2 mul rho mul A_area pow v INT_2 mul INT_1 INT_1",
                         "F_d = 0.5*rho*A*v^2*Cd", ["rho", "A_area", "v"],
                         {"c_rho": (1.0, 1.3), "c_A": (0.1, 2.0), "c_v": (1.0, 20.0)},
                         "0.5 * c_rho * c_A * c_v**2"),
        EquationTemplate("impulse", "dynamics", "simple",
                         "mul F t", "J = F*t", ["F", "t"],
                         {"c_F": (1.0, 50.0)},
                         "c_F * var_t"),
        EquationTemplate("momentum", "dynamics", "simple",
                         "mul m v", "p = m*v", ["m", "v"],
                         {"c_m": (0.5, 10.0), "c_v": (0.5, 10.0)},
                         "c_m * c_v"),
    ])

    # ===== FAMILY 3: ENERGY =====
    templates.extend([
        EquationTemplate("kinetic_energy", "energy", "simple",
                         "mul div INT_1 INT_2 mul m pow v INT_2",
                         "KE = 0.5*m*v^2", ["m", "v"],
                         {"c_m": (0.5, 10.0), "c_v": (0.5, 10.0)},
                         "0.5 * c_m * c_v**2"),
        EquationTemplate("potential_energy", "energy", "simple",
                         "mul m mul g_accel h",
                         "PE = m*g*h", ["m", "h"],
                         {"c_m": (0.5, 10.0)},
                         "c_m * 9.81 * var_h"),
        EquationTemplate("elastic_pe", "energy", "simple",
                         "mul div INT_1 INT_2 mul k_spring pow x INT_2",
                         "PE = 0.5*k*x^2", ["k_spring", "x"],
                         {"c_k": (1.0, 50.0)},
                         "0.5 * c_k * var_x**2"),
        EquationTemplate("work_force_dist", "energy", "simple",
                         "mul F mul d cos theta",
                         "W = F*d*cos(theta)", ["F", "d", "theta"],
                         {"c_F": (1.0, 50.0)},
                         "c_F * var_d * math.cos(var_theta)"),
        EquationTemplate("power", "energy", "simple",
                         "mul F v", "P = F*v", ["F", "v"],
                         {"c_F": (1.0, 50.0), "c_v": (0.5, 10.0)},
                         "c_F * c_v"),
        EquationTemplate("energy_conservation", "energy", "complex",
                         "add mul div INT_1 INT_2 mul m pow v INT_2 mul m mul g_accel h",
                         "E = 0.5*m*v^2 + m*g*h", ["m", "v", "h"],
                         {"c_m": (1.0, 10.0), "c_v": (1.0, 5.0)},
                         "0.5 * c_m * c_v**2 + c_m * 9.81 * var_h"),
        EquationTemplate("work_energy_theorem", "energy", "medium",
                         "sub mul div INT_1 INT_2 mul m pow v INT_2 mul div INT_1 INT_2 mul m pow v0 INT_2",
                         "W = 0.5*m*v^2 - 0.5*m*v0^2", ["m", "v", "v0"],
                         {"c_m": (1.0, 10.0), "c_v": (1.0, 10.0), "c_v0": (0.5, 5.0)},
                         "0.5 * c_m * c_v**2 - 0.5 * c_m * c_v0**2"),
        EquationTemplate("gravitational_pe", "energy", "medium",
                         "neg div mul G_const mul m1 m2 r",
                         "U = -G*m1*m2/r", ["m1", "m2", "r"],
                         {"c_m1": (1e10, 1e12), "c_m2": (1e10, 1e12)},
                         "-6.674e-11 * c_m1 * c_m2 / var_r"),
        EquationTemplate("escape_velocity", "energy", "complex",
                         "sqrt div mul INT_2 mul G_const m r",
                         "v_esc = sqrt(2*G*M/r)", ["m", "r"],
                         {"c_m": (1e20, 1e25)},
                         "math.sqrt(2 * 6.674e-11 * c_m / var_r)"),
    ])

    # ===== FAMILY 4: ROTATIONAL MECHANICS =====
    templates.extend([
        EquationTemplate("torque", "rotational", "simple",
                         "mul I_inertia alpha",
                         "tau = I*alpha", ["I_inertia", "alpha"],
                         {"c_I": (0.5, 10.0), "c_alpha": (0.5, 10.0)},
                         "c_I * c_alpha"),
        EquationTemplate("angular_momentum", "rotational", "simple",
                         "mul I_inertia omega",
                         "L = I*omega", ["I_inertia", "omega"],
                         {"c_I": (0.5, 10.0), "c_omega": (0.5, 10.0)},
                         "c_I * c_omega"),
        EquationTemplate("rotational_ke", "rotational", "simple",
                         "mul div INT_1 INT_2 mul I_inertia pow omega INT_2",
                         "KE_rot = 0.5*I*omega^2", ["I_inertia", "omega"],
                         {"c_I": (0.5, 10.0), "c_omega": (0.5, 10.0)},
                         "0.5 * c_I * c_omega**2"),
        EquationTemplate("torque_cross", "rotational", "medium",
                         "mul r mul F sin theta",
                         "tau = r*F*sin(theta)", ["r", "F", "theta"],
                         {"c_r": (0.1, 2.0), "c_F": (1.0, 50.0)},
                         "c_r * c_F * math.sin(var_theta)"),
        EquationTemplate("moment_inertia_rod", "rotational", "medium",
                         "mul div INT_1 INT_3 mul m pow l_length INT_2",
                         "I = (1/3)*m*L^2", ["m", "l_length"],
                         {"c_m": (0.5, 10.0)},
                         "(1.0/3.0) * c_m * var_l_length**2"),
        EquationTemplate("moment_inertia_disk", "rotational", "medium",
                         "mul div INT_1 INT_2 mul m pow r INT_2",
                         "I = 0.5*m*R^2", ["m", "r"],
                         {"c_m": (0.5, 10.0)},
                         "0.5 * c_m * var_r**2"),
        EquationTemplate("rolling_ke", "rotational", "complex",
                         "add mul div INT_1 INT_2 mul m pow v INT_2 mul div INT_1 INT_2 mul I_inertia pow omega INT_2",
                         "KE = 0.5*m*v^2 + 0.5*I*omega^2", ["m", "v", "I_inertia", "omega"],
                         {"c_m": (1.0, 10.0), "c_v": (1.0, 5.0), "c_I": (0.5, 5.0), "c_omega": (1.0, 10.0)},
                         "0.5 * c_m * c_v**2 + 0.5 * c_I * c_omega**2"),
        EquationTemplate("angular_accel", "rotational", "simple",
                         "div tau I_inertia",
                         "alpha = tau/I", ["tau", "I_inertia"],
                         {"c_tau": (1.0, 20.0), "c_I": (0.5, 10.0)},
                         "c_tau / c_I"),
        EquationTemplate("angular_velocity_const", "rotational", "simple",
                         "add omega mul alpha t",
                         "omega_f = omega_i + alpha*t", ["omega", "alpha", "t"],
                         {"c_omega": (1.0, 10.0), "c_alpha": (0.5, 5.0)},
                         "c_omega + c_alpha * var_t"),
    ])

    # ===== FAMILY 5: GRAVITATION =====
    templates.extend([
        EquationTemplate("gravity_force", "gravitation", "medium",
                         "div mul G_const mul m1 m2 pow r INT_2",
                         "F = G*m1*m2/r^2", ["m1", "m2", "r"],
                         {"c_m1": (1e10, 1e15), "c_m2": (1e10, 1e15)},
                         "6.674e-11 * c_m1 * c_m2 / var_r**2"),
        EquationTemplate("orbital_velocity", "gravitation", "complex",
                         "sqrt div mul G_const m r",
                         "v = sqrt(G*M/r)", ["m", "r"],
                         {"c_m": (1e20, 1e25)},
                         "math.sqrt(6.674e-11 * c_m / var_r)"),
        EquationTemplate("orbital_period", "gravitation", "complex",
                         "mul mul INT_2 pi sqrt div pow r INT_3 mul G_const m",
                         "T = 2*pi*sqrt(r^3/(G*M))", ["r", "m"],
                         {"c_m": (1e25, 1e30)},
                         "2 * math.pi * math.sqrt(var_r**3 / (6.674e-11 * c_m))"),
        EquationTemplate("grav_accel_surface", "gravitation", "medium",
                         "div mul G_const m pow r INT_2",
                         "g = G*M/R^2", ["m", "r"],
                         {"c_m": (1e20, 1e25)},
                         "6.674e-11 * c_m / var_r**2"),
        EquationTemplate("grav_potential", "gravitation", "medium",
                         "neg div mul G_const m r",
                         "phi = -G*M/r", ["m", "r"],
                         {"c_m": (1e20, 1e25)},
                         "-6.674e-11 * c_m / var_r"),
        EquationTemplate("kepler_3rd_simple", "gravitation", "complex",
                         "div pow r INT_3 pow mul div INT_1 mul INT_2 pi mul G_const m INT_1 INT_2",
                         "T^2 = (4pi^2/(GM))*r^3", ["r", "m"],
                         {"c_m": (1e25, 1e30)},
                         "4 * math.pi**2 / (6.674e-11 * c_m) * var_r**3"),
        EquationTemplate("tidal_force", "gravitation", "complex",
                         "mul INT_2 div mul G_const mul m d pow r INT_3",
                         "F_tidal = 2*G*M*d/r^3", ["m", "d", "r"],
                         {"c_m": (1e20, 1e25)},
                         "2 * 6.674e-11 * c_m * var_d / var_r**3"),
    ])

    # ===== FAMILY 6: OSCILLATIONS =====
    templates.extend([
        EquationTemplate("shm_position", "oscillations", "medium",
                         "mul A_area sin mul omega t",
                         "x = A*sin(omega*t)", ["A_area", "omega", "t"],
                         {"c_A": (0.5, 5.0), "c_omega": (1.0, 10.0)},
                         "c_A * math.sin(c_omega * var_t)"),
        EquationTemplate("shm_velocity", "oscillations", "medium",
                         "mul mul A_area omega cos mul omega t",
                         "v = A*omega*cos(omega*t)", ["A_area", "omega", "t"],
                         {"c_A": (0.5, 5.0), "c_omega": (1.0, 10.0)},
                         "c_A * c_omega * math.cos(c_omega * var_t)"),
        EquationTemplate("shm_accel", "oscillations", "complex",
                         "neg mul mul A_area pow omega INT_2 sin mul omega t",
                         "a = -A*omega^2*sin(omega*t)", ["A_area", "omega", "t"],
                         {"c_A": (0.5, 5.0), "c_omega": (1.0, 10.0)},
                         "-c_A * c_omega**2 * math.sin(c_omega * var_t)"),
        EquationTemplate("pendulum_period", "oscillations", "complex",
                         "mul mul INT_2 pi sqrt div l_length g_accel",
                         "T = 2*pi*sqrt(l/g)", ["l_length"],
                         {},
                         "2 * math.pi * math.sqrt(var_l_length / 9.81)"),
        EquationTemplate("spring_period", "oscillations", "complex",
                         "mul mul INT_2 pi sqrt div m k_spring",
                         "T = 2*pi*sqrt(m/k)", ["m", "k_spring"],
                         {"c_m": (0.5, 10.0), "c_k": (1.0, 50.0)},
                         "2 * math.pi * math.sqrt(c_m / c_k)"),
        EquationTemplate("spring_frequency", "oscillations", "medium",
                         "div INT_1 mul INT_2 pi sqrt div m k_spring",
                         "f = 1/(2*pi*sqrt(m/k))", ["m", "k_spring"],
                         {"c_m": (0.5, 10.0), "c_k": (1.0, 50.0)},
                         "1.0 / (2 * math.pi * math.sqrt(c_m / c_k))"),
        EquationTemplate("shm_energy", "oscillations", "medium",
                         "mul div INT_1 INT_2 mul k_spring pow A_area INT_2",
                         "E = 0.5*k*A^2", ["k_spring", "A_area"],
                         {"c_k": (1.0, 50.0), "c_A": (0.1, 2.0)},
                         "0.5 * c_k * c_A**2"),
        EquationTemplate("shm_phase", "oscillations", "complex",
                         "mul A_area sin add mul omega t phi",
                         "x = A*sin(omega*t + phi)", ["A_area", "omega", "t", "phi"],
                         {"c_A": (0.5, 5.0), "c_omega": (1.0, 10.0)},
                         "c_A * math.sin(c_omega * var_t + var_phi)"),
        EquationTemplate("damped_oscillation", "oscillations", "complex",
                         "mul mul A_area exp neg mul mul div INT_1 INT_2 mul mu div t m sin mul omega t",
                         "x = A*exp(-mu*t/(2m))*sin(omega*t)", ["A_area", "omega", "t", "mu", "m"],
                         {"c_A": (1.0, 5.0), "c_omega": (2.0, 10.0), "c_mu": (0.1, 1.0), "c_m": (1.0, 5.0)},
                         "c_A * math.exp(-c_mu * var_t / (2 * c_m)) * math.sin(c_omega * var_t)"),
    ])

    # ===== FAMILY 7: FLUID STATICS =====
    templates.extend([
        EquationTemplate("pressure_depth", "fluid_statics", "simple",
                         "mul rho mul g_accel h",
                         "P = rho*g*h", ["rho", "h"],
                         {"c_rho": (500.0, 1500.0)},
                         "c_rho * 9.81 * var_h"),
        EquationTemplate("buoyancy", "fluid_statics", "simple",
                         "mul rho mul g_accel V_volume",
                         "F_b = rho*g*V", ["rho", "V_volume"],
                         {"c_rho": (500.0, 1500.0)},
                         "c_rho * 9.81 * var_V_volume"),
        EquationTemplate("absolute_pressure", "fluid_statics", "medium",
                         "add P_pressure mul rho mul g_accel h",
                         "P_abs = P0 + rho*g*h", ["P_pressure", "rho", "h"],
                         {"c_P0": (1e5, 1.1e5), "c_rho": (500.0, 1500.0)},
                         "c_P0 + c_rho * 9.81 * var_h"),
        EquationTemplate("bernoulli_simple", "fluid_statics", "complex",
                         "add P_pressure add mul div INT_1 INT_2 mul rho pow v INT_2 mul rho mul g_accel h",
                         "P + 0.5*rho*v^2 + rho*g*h", ["P_pressure", "rho", "v", "h"],
                         {"c_P": (1e4, 1e5), "c_rho": (500.0, 1500.0), "c_v": (0.5, 5.0)},
                         "c_P + 0.5 * c_rho * c_v**2 + c_rho * 9.81 * var_h"),
        EquationTemplate("flow_rate", "fluid_statics", "simple",
                         "mul A_area v",
                         "Q = A*v", ["A_area", "v"],
                         {"c_A": (0.01, 1.0), "c_v": (0.1, 5.0)},
                         "c_A * c_v"),
        EquationTemplate("pascal_law", "fluid_statics", "simple",
                         "div F A_area",
                         "P = F/A", ["F", "A_area"],
                         {"c_F": (10.0, 1000.0), "c_A": (0.01, 1.0)},
                         "c_F / c_A"),
        EquationTemplate("hydraulic_press", "fluid_statics", "medium",
                         "mul F div A_area r",
                         "F2 = F1 * A2/A1", ["F", "A_area", "r"],
                         {"c_F": (10.0, 500.0), "c_A": (0.5, 5.0)},
                         "c_F * c_A / var_r"),
        EquationTemplate("torricelli", "fluid_statics", "complex",
                         "sqrt mul INT_2 mul g_accel h",
                         "v = sqrt(2*g*h)", ["h"],
                         {},
                         "math.sqrt(2 * 9.81 * var_h)"),
    ])

    return templates


# ---------------------------------------------------------------------------
# Variable sampling ranges
# ---------------------------------------------------------------------------

VARIABLE_RANGES = {
    'x': (0.1, 10.0), 'y': (0.1, 10.0), 'z': (0.1, 10.0),
    't': (0.1, 10.0), 'v': (0.5, 20.0), 'v0': (0.0, 10.0),
    'vx': (0.5, 10.0), 'vy': (0.5, 10.0), 'vz': (0.5, 10.0),
    'a': (0.1, 10.0), 'ax': (0.1, 5.0), 'ay': (0.1, 5.0), 'az': (0.1, 5.0),
    'F': (1.0, 100.0), 'Fx': (1.0, 50.0), 'Fy': (1.0, 50.0), 'Fz': (1.0, 50.0),
    'm': (0.5, 50.0), 'm1': (1.0, 100.0), 'm2': (1.0, 100.0),
    'r': (0.1, 10.0), 'R': (0.1, 10.0),
    'theta': (0.1, 1.5), 'phi': (0.0, 6.28),
    'omega': (0.5, 20.0), 'alpha': (0.1, 10.0),
    'tau': (1.0, 50.0), 'I_inertia': (0.1, 10.0),
    'L_angular': (0.1, 10.0), 'E_energy': (1.0, 100.0),
    'KE': (1.0, 100.0), 'PE': (1.0, 100.0), 'W_work': (1.0, 100.0),
    'P_power': (1.0, 100.0), 'p_momentum': (1.0, 100.0),
    'rho': (500.0, 1500.0), 'P_pressure': (1e3, 1e5),
    'V_volume': (0.001, 1.0), 'A_area': (0.01, 2.0),
    'h': (0.1, 20.0), 'l_length': (0.1, 5.0),
    'd': (0.1, 20.0), 'k_spring': (1.0, 100.0), 'mu': (0.1, 0.9),
    'x0': (0.0, 5.0),
}


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class PhysicsDatasetGenerator:
    """Generate physics equation datasets for training and evaluation."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.templates = build_templates()

    @property
    def num_templates(self) -> int:
        return len(self.templates)

    def get_family_counts(self) -> Dict[str, int]:
        counts = {}
        for t in self.templates:
            counts[t.family] = counts.get(t.family, 0) + 1
        return counts

    def get_difficulty_counts(self) -> Dict[str, int]:
        counts = {}
        for t in self.templates:
            counts[t.difficulty] = counts.get(t.difficulty, 0) + 1
        return counts

    def sample_observations(self, template: EquationTemplate, coeffs: Dict[str, float],
                            n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) observation pairs for a given equation instance."""
        variables = template.variables
        n_vars = len(variables)
        X = np.zeros((n_points, n_vars))
        Y = np.zeros(n_points)

        for i in range(n_points):
            var_values = {}
            for j, var in enumerate(variables):
                var_key = var
                if var_key in VARIABLE_RANGES:
                    lo, hi = VARIABLE_RANGES[var_key]
                else:
                    lo, hi = 0.1, 10.0
                val = self.np_rng.uniform(lo, hi)
                var_values[f'var_{var}'] = val
                X[i, j] = val

            # Merge coefficients (as c_xxx) with variable values (as var_xxx)
            namespace = {**coeffs, **var_values, 'math': math, 'np': np,
                         'pi': math.pi, 'g': 9.81, 'G': 6.674e-11}
            try:
                y_val = float(eval(template.eval_func, {"__builtins__": {}}, namespace))
                if math.isfinite(y_val):
                    Y[i] = y_val
                else:
                    Y[i] = 0.0
            except Exception:
                Y[i] = 0.0

        return X, Y

    def generate_sample(self, template: Optional[EquationTemplate] = None,
                        n_points: int = 50) -> Dict:
        """Generate a single training sample."""
        if template is None:
            template = self.rng.choice(self.templates)

        coeffs = template.generate_coefficients(self.rng)
        prefix = template.instantiate_prefix(coeffs)
        X, Y = self.sample_observations(template, coeffs, n_points)

        return {
            'template_name': template.name,
            'family': template.family,
            'difficulty': template.difficulty,
            'prefix_notation': prefix,
            'infix_readable': template.infix_readable,
            'variables': template.variables,
            'observations_x': X.tolist(),
            'observations_y': Y.tolist(),
            'n_points': n_points,
        }

    def generate_dataset(self, n_samples: int, n_points: int = 50,
                         difficulty: Optional[str] = None,
                         family: Optional[str] = None) -> List[Dict]:
        """Generate a full dataset of n_samples equation-observation pairs."""
        # Filter templates if needed
        pool = self.templates
        if difficulty:
            pool = [t for t in pool if t.difficulty == difficulty]
        if family:
            pool = [t for t in pool if t.family == family]

        if not pool:
            raise ValueError(f"No templates match difficulty={difficulty}, family={family}")

        dataset = []
        for _ in range(n_samples):
            template = self.rng.choice(pool)
            sample = self.generate_sample(template, n_points)
            dataset.append(sample)

        return dataset

    def generate_and_save(self, n_samples: int, output_path: str,
                          n_points: int = 50, **kwargs):
        """Generate and save dataset to a JSON file."""
        dataset = self.generate_dataset(n_samples, n_points, **kwargs)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f)
        print(f"Generated {len(dataset)} samples -> {output_path}")
        return dataset

    def generate_split(self, n_total: int, train_frac: float = 0.8,
                       val_frac: float = 0.1, test_frac: float = 0.1,
                       output_dir: str = 'data/', n_points: int = 50):
        """Generate and save train/val/test splits."""
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        n_test = n_total - n_train - n_val

        os.makedirs(output_dir, exist_ok=True)

        train = self.generate_dataset(n_train, n_points)
        val = self.generate_dataset(n_val, n_points)
        test = self.generate_dataset(n_test, n_points)

        for name, data in [('train', train), ('val', val), ('test', test)]:
            path = os.path.join(output_dir, f'{name}.json')
            with open(path, 'w') as f:
                json.dump(data, f)
            print(f"  {name}: {len(data)} samples -> {path}")

        return train, val, test


if __name__ == '__main__':
    gen = PhysicsDatasetGenerator(seed=42)
    print(f"Total templates: {gen.num_templates}")
    print(f"Family counts: {gen.get_family_counts()}")
    print(f"Difficulty counts: {gen.get_difficulty_counts()}")

    # Generate a small test
    sample = gen.generate_sample()
    print(f"\nSample: {sample['template_name']} ({sample['family']}/{sample['difficulty']})")
    print(f"  Prefix: {sample['prefix_notation']}")
    print(f"  Infix: {sample['infix_readable']}")
    print(f"  Variables: {sample['variables']}")
    print(f"  X shape: ({len(sample['observations_x'])}, {len(sample['observations_x'][0])})")
    print(f"  Y sample: {sample['observations_y'][:5]}")
