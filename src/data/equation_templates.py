"""
Equation templates for synthetic physics data generation.
Each template defines a symbolic equation, its variables, and sampling constraints.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Tuple

# Dataset constants
MAX_OBS_POINTS = 50
MAX_INPUT_VARS = 6
MAX_EQ_LENGTH = 64


@dataclass
class EquationTemplate:
    """A parameterized physics equation template."""
    template_id: str
    tier: int  # 1-4
    description: str
    n_input_vars: int  # Number of input variables (x1, ..., xk)
    # Function: (x_array [N, n_input_vars], coeffs) -> y_array [N]
    eval_fn: Callable
    # Function: (coeffs) -> prefix notation token list
    to_prefix: Callable
    # Coefficient sampling: list of (low, high) tuples
    coeff_ranges: List[Tuple[float, float]]
    # Variable sampling: list of (low, high) tuples for each input var
    var_ranges: List[Tuple[float, float]]
    # Whether variables must be positive
    positive_vars: List[bool] = field(default_factory=list)


def _safe_eval(y: np.ndarray) -> np.ndarray:
    """Replace inf/nan with large finite values, then clip."""
    y = np.where(np.isfinite(y), y, np.nan)
    return y


def build_tier1_templates() -> List[EquationTemplate]:
    """Tier 1: Kinematic equations (simple polynomial/trig)."""
    templates = []

    # K1: s = u*t + 0.5*a*t^2
    templates.append(EquationTemplate(
        template_id="T1.01", tier=1,
        description="s = u*t + 0.5*a*t^2",
        n_input_vars=3,  # u, a, t
        eval_fn=lambda x, c: x[:, 0] * x[:, 2] + 0.5 * x[:, 1] * x[:, 2] ** 2,
        to_prefix=lambda c: ["add", "mul", "x1", "x3", "mul", "mul", "C0.5", "x2", "pow", "x3", "C2"],
        coeff_ranges=[],
        var_ranges=[(-10, 10), (-10, 10), (0.01, 10)],
        positive_vars=[False, False, True],
    ))

    # K2: v = u + a*t
    templates.append(EquationTemplate(
        template_id="T1.02", tier=1,
        description="v = u + a*t",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] + x[:, 1] * x[:, 2],
        to_prefix=lambda c: ["add", "x1", "mul", "x2", "x3"],
        coeff_ranges=[],
        var_ranges=[(-10, 10), (-10, 10), (0.01, 10)],
        positive_vars=[False, False, True],
    ))

    # K3: v^2 = u^2 + 2*a*s
    templates.append(EquationTemplate(
        template_id="T1.03", tier=1,
        description="v^2 = u^2 + 2*a*s",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] ** 2 + 2 * x[:, 1] * x[:, 2],
        to_prefix=lambda c: ["add", "pow", "x1", "C2", "mul", "mul", "C2", "x2", "x3"],
        coeff_ranges=[],
        var_ranges=[(-10, 10), (0.1, 10), (0.1, 10)],
        positive_vars=[False, True, True],
    ))

    # K4: s = v*t
    templates.append(EquationTemplate(
        template_id="T1.04", tier=1,
        description="s = v*t",
        n_input_vars=2,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1],
        to_prefix=lambda c: ["mul", "x1", "x2"],
        coeff_ranges=[],
        var_ranges=[(-10, 10), (0.01, 10)],
        positive_vars=[False, True],
    ))

    # K5: s = 0.5*(u+v)*t
    templates.append(EquationTemplate(
        template_id="T1.05", tier=1,
        description="s = 0.5*(u+v)*t",
        n_input_vars=3,
        eval_fn=lambda x, c: 0.5 * (x[:, 0] + x[:, 1]) * x[:, 2],
        to_prefix=lambda c: ["mul", "mul", "C0.5", "add", "x1", "x2", "x3"],
        coeff_ranges=[],
        var_ranges=[(-10, 10), (-10, 10), (0.01, 10)],
        positive_vars=[False, False, True],
    ))

    # K6: theta = w0*t + 0.5*alpha*t^2
    templates.append(EquationTemplate(
        template_id="T1.06", tier=1,
        description="theta = w0*t + 0.5*alpha*t^2",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] * x[:, 2] + 0.5 * x[:, 1] * x[:, 2] ** 2,
        to_prefix=lambda c: ["add", "mul", "x1", "x3", "mul", "mul", "C0.5", "x2", "pow", "x3", "C2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 10), (-5, 5), (0.01, 5)],
        positive_vars=[True, False, True],
    ))

    # K7: omega = omega0 + alpha*t
    templates.append(EquationTemplate(
        template_id="T1.07", tier=1,
        description="omega = omega0 + alpha*t",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] + x[:, 1] * x[:, 2],
        to_prefix=lambda c: ["add", "x1", "mul", "x2", "x3"],
        coeff_ranges=[],
        var_ranges=[(0.1, 10), (-5, 5), (0.01, 5)],
        positive_vars=[True, False, True],
    ))

    # K8: h = v0*t - 0.5*g*t^2 (g is a coefficient ~9.8)
    templates.append(EquationTemplate(
        template_id="T1.08", tier=1,
        description="h = v0*t - 0.5*g*t^2",
        n_input_vars=2,  # v0, t; g is coefficient
        eval_fn=lambda x, c: x[:, 0] * x[:, 1] - 0.5 * c[0] * x[:, 1] ** 2,
        to_prefix=lambda c: ["sub", "mul", "x1", "x2", "mul", "mul", "C0.5", f"C{c[0]:.4f}", "pow", "x2", "C2"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(1, 50), (0.01, 5)],
        positive_vars=[True, True],
    ))

    # K9: x = A*sin(w*t)
    templates.append(EquationTemplate(
        template_id="T1.09", tier=1,
        description="x = A*sin(w*t)",
        n_input_vars=1,  # t; A, w are coefficients
        eval_fn=lambda x, c: c[0] * np.sin(c[1] * x[:, 0]),
        to_prefix=lambda c: ["mul", f"C{c[0]:.4f}", "sin", "mul", f"C{c[1]:.4f}", "x1"],
        coeff_ranges=[(0.5, 10), (0.5, 10)],
        var_ranges=[(0, 10)],
        positive_vars=[True],
    ))

    # K10: v = A*w*cos(w*t)
    templates.append(EquationTemplate(
        template_id="T1.10", tier=1,
        description="v = A*w*cos(w*t)",
        n_input_vars=1,
        eval_fn=lambda x, c: c[0] * c[1] * np.cos(c[1] * x[:, 0]),
        to_prefix=lambda c: ["mul", "mul", f"C{c[0]:.4f}", f"C{c[1]:.4f}", "cos", "mul", f"C{c[1]:.4f}", "x1"],
        coeff_ranges=[(0.5, 10), (0.5, 10)],
        var_ranges=[(0, 10)],
        positive_vars=[True],
    ))

    # K11: s = 0.5*a*t^2
    templates.append(EquationTemplate(
        template_id="T1.11", tier=1,
        description="s = 0.5*a*t^2",
        n_input_vars=2,
        eval_fn=lambda x, c: 0.5 * x[:, 0] * x[:, 1] ** 2,
        to_prefix=lambda c: ["mul", "mul", "C0.5", "x1", "pow", "x2", "C2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 10), (0.01, 10)],
        positive_vars=[True, True],
    ))

    # K12: x = A*cos(w*t + phi)
    templates.append(EquationTemplate(
        template_id="T1.12", tier=1,
        description="x = A*cos(w*t + phi)",
        n_input_vars=1,
        eval_fn=lambda x, c: c[0] * np.cos(c[1] * x[:, 0] + c[2]),
        to_prefix=lambda c: ["mul", f"C{c[0]:.4f}", "cos", "add", "mul", f"C{c[1]:.4f}", "x1", f"C{c[2]:.4f}"],
        coeff_ranges=[(0.5, 10), (0.5, 10), (0, 6.28)],
        var_ranges=[(0, 10)],
        positive_vars=[True],
    ))

    return templates


def build_tier2_templates() -> List[EquationTemplate]:
    """Tier 2: Force laws (multi-variable, products, division)."""
    templates = []

    # F1: F = m*a
    templates.append(EquationTemplate(
        template_id="T2.01", tier=2,
        description="F = m*a",
        n_input_vars=2,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1],
        to_prefix=lambda c: ["mul", "x1", "x2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 100), (-10, 10)],
        positive_vars=[True, False],
    ))

    # F2: F = -k*x
    templates.append(EquationTemplate(
        template_id="T2.02", tier=2,
        description="F = -k*x",
        n_input_vars=1,
        eval_fn=lambda x, c: -c[0] * x[:, 0],
        to_prefix=lambda c: ["neg", "mul", f"C{c[0]:.4f}", "x1"],
        coeff_ranges=[(0.1, 100)],
        var_ranges=[(-5, 5)],
    ))

    # F3: F = G*m1*m2/r^2
    templates.append(EquationTemplate(
        template_id="T2.03", tier=2,
        description="F = G*m1*m2/r^2",
        n_input_vars=3,
        eval_fn=lambda x, c: c[0] * x[:, 0] * x[:, 1] / (x[:, 2] ** 2),
        to_prefix=lambda c: ["div", "mul", "mul", f"C{c[0]:.4f}", "x1", "x2", "pow", "x3", "C2"],
        coeff_ranges=[(0.1, 10)],
        var_ranges=[(0.1, 100), (0.1, 100), (0.5, 50)],
        positive_vars=[True, True, True],
    ))

    # F4: F = mu*m*g
    templates.append(EquationTemplate(
        template_id="T2.04", tier=2,
        description="F = mu*m*g",
        n_input_vars=2,
        eval_fn=lambda x, c: c[0] * x[:, 0] * x[:, 1],
        to_prefix=lambda c: ["mul", "mul", f"C{c[0]:.4f}", "x1", "x2"],
        coeff_ranges=[(0.05, 1.0)],  # mu coefficient of friction
        var_ranges=[(0.1, 100), (9.0, 10.0)],
        positive_vars=[True, True],
    ))

    # F5: tau = r*F*sin(theta)
    templates.append(EquationTemplate(
        template_id="T2.05", tier=2,
        description="tau = r*F*sin(theta)",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1] * np.sin(x[:, 2]),
        to_prefix=lambda c: ["mul", "mul", "x1", "x2", "sin", "x3"],
        coeff_ranges=[],
        var_ranges=[(0.1, 10), (0.1, 100), (0.01, 3.14)],
        positive_vars=[True, True, True],
    ))

    # F6: F = m*v^2/r
    templates.append(EquationTemplate(
        template_id="T2.06", tier=2,
        description="F = m*v^2/r",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1] ** 2 / x[:, 2],
        to_prefix=lambda c: ["div", "mul", "x1", "pow", "x2", "C2", "x3"],
        coeff_ranges=[],
        var_ranges=[(0.1, 100), (0.1, 20), (0.5, 50)],
        positive_vars=[True, True, True],
    ))

    # F7: p = m*v
    templates.append(EquationTemplate(
        template_id="T2.07", tier=2,
        description="p = m*v",
        n_input_vars=2,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1],
        to_prefix=lambda c: ["mul", "x1", "x2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 100), (-20, 20)],
        positive_vars=[True, False],
    ))

    # F8: L = I*omega
    templates.append(EquationTemplate(
        template_id="T2.08", tier=2,
        description="L = I*omega",
        n_input_vars=2,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1],
        to_prefix=lambda c: ["mul", "x1", "x2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 50), (0.1, 20)],
        positive_vars=[True, True],
    ))

    # F9: F = -b*v (linear drag)
    templates.append(EquationTemplate(
        template_id="T2.09", tier=2,
        description="F = -b*v",
        n_input_vars=1,
        eval_fn=lambda x, c: -c[0] * x[:, 0],
        to_prefix=lambda c: ["neg", "mul", f"C{c[0]:.4f}", "x1"],
        coeff_ranges=[(0.01, 10)],
        var_ranges=[(-20, 20)],
    ))

    # F10: F = -c*v^2
    templates.append(EquationTemplate(
        template_id="T2.10", tier=2,
        description="F = -c*v^2",
        n_input_vars=1,
        eval_fn=lambda x, c: -c[0] * x[:, 0] ** 2 * np.sign(x[:, 0]),
        to_prefix=lambda c: ["neg", "mul", f"C{c[0]:.4f}", "mul", "pow", "x1", "C2", "sgn", "x1"],
        coeff_ranges=[(0.01, 5)],
        var_ranges=[(-20, 20)],
    ))

    # F11: W = F*d*cos(theta)
    templates.append(EquationTemplate(
        template_id="T2.11", tier=2,
        description="W = F*d*cos(theta)",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1] * np.cos(x[:, 2]),
        to_prefix=lambda c: ["mul", "mul", "x1", "x2", "cos", "x3"],
        coeff_ranges=[],
        var_ranges=[(0.1, 100), (0.1, 50), (0, 3.14)],
        positive_vars=[True, True, True],
    ))

    # F12: P = F*v
    templates.append(EquationTemplate(
        template_id="T2.12", tier=2,
        description="P = F*v",
        n_input_vars=2,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1],
        to_prefix=lambda c: ["mul", "x1", "x2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 100), (0.1, 20)],
        positive_vars=[True, True],
    ))

    # F13: KE = 0.5*m*v^2
    templates.append(EquationTemplate(
        template_id="T2.13", tier=2,
        description="KE = 0.5*m*v^2",
        n_input_vars=2,
        eval_fn=lambda x, c: 0.5 * x[:, 0] * x[:, 1] ** 2,
        to_prefix=lambda c: ["mul", "mul", "C0.5", "x1", "pow", "x2", "C2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 100), (-20, 20)],
        positive_vars=[True, False],
    ))

    # F14: PE = m*g*h
    templates.append(EquationTemplate(
        template_id="T2.14", tier=2,
        description="PE = m*g*h",
        n_input_vars=2,
        eval_fn=lambda x, c: x[:, 0] * c[0] * x[:, 1],
        to_prefix=lambda c: ["mul", "mul", "x1", f"C{c[0]:.4f}", "x2"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 100), (0.1, 50)],
        positive_vars=[True, True],
    ))

    return templates


def build_tier3_templates() -> List[EquationTemplate]:
    """Tier 3: Conservation laws (composite expressions)."""
    templates = []

    # C1: E = 0.5*m*v^2 + m*g*h
    templates.append(EquationTemplate(
        template_id="T3.01", tier=3,
        description="E = 0.5*m*v^2 + m*g*h",
        n_input_vars=3,  # m, v, h; g is coeff
        eval_fn=lambda x, c: 0.5 * x[:, 0] * x[:, 1] ** 2 + x[:, 0] * c[0] * x[:, 2],
        to_prefix=lambda c: ["add", "mul", "mul", "C0.5", "x1", "pow", "x2", "C2",
                              "mul", "mul", "x1", f"C{c[0]:.4f}", "x3"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 50), (-20, 20), (0, 50)],
        positive_vars=[True, False, True],
    ))

    # C2: p_total = m1*v1 + m2*v2
    templates.append(EquationTemplate(
        template_id="T3.02", tier=3,
        description="p_total = m1*v1 + m2*v2",
        n_input_vars=4,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3],
        to_prefix=lambda c: ["add", "mul", "x1", "x2", "mul", "x3", "x4"],
        coeff_ranges=[],
        var_ranges=[(0.1, 50), (-20, 20), (0.1, 50), (-20, 20)],
        positive_vars=[True, False, True, False],
    ))

    # C3: v = sqrt(k*x^2/m) (spring PE->KE)
    templates.append(EquationTemplate(
        template_id="T3.03", tier=3,
        description="v = sqrt(k*x^2/m)",
        n_input_vars=2,  # x, m; k is coeff
        eval_fn=lambda x, c: np.sqrt(np.abs(c[0] * x[:, 0] ** 2 / x[:, 1])),
        to_prefix=lambda c: ["sqrt", "div", "mul", f"C{c[0]:.4f}", "pow", "x1", "C2", "x2"],
        coeff_ranges=[(0.1, 50)],
        var_ranges=[(-5, 5), (0.1, 50)],
        positive_vars=[False, True],
    ))

    # C4: T = 2*pi*sqrt(L/g)
    templates.append(EquationTemplate(
        template_id="T3.04", tier=3,
        description="T = 2*pi*sqrt(L/g)",
        n_input_vars=1,  # L; g is coeff
        eval_fn=lambda x, c: 2 * np.pi * np.sqrt(x[:, 0] / c[0]),
        to_prefix=lambda c: ["mul", "mul", "C2", "Cpi", "sqrt", "div", "x1", f"C{c[0]:.4f}"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 20)],
        positive_vars=[True],
    ))

    # C5: T = 2*pi*sqrt(m/k)
    templates.append(EquationTemplate(
        template_id="T3.05", tier=3,
        description="T = 2*pi*sqrt(m/k)",
        n_input_vars=1,  # m; k is coeff
        eval_fn=lambda x, c: 2 * np.pi * np.sqrt(x[:, 0] / c[0]),
        to_prefix=lambda c: ["mul", "mul", "C2", "Cpi", "sqrt", "div", "x1", f"C{c[0]:.4f}"],
        coeff_ranges=[(0.1, 100)],
        var_ranges=[(0.1, 50)],
        positive_vars=[True],
    ))

    # C6: KE_rot = 0.5*I*omega^2
    templates.append(EquationTemplate(
        template_id="T3.06", tier=3,
        description="KE_rot = 0.5*I*omega^2",
        n_input_vars=2,
        eval_fn=lambda x, c: 0.5 * x[:, 0] * x[:, 1] ** 2,
        to_prefix=lambda c: ["mul", "mul", "C0.5", "x1", "pow", "x2", "C2"],
        coeff_ranges=[],
        var_ranges=[(0.1, 50), (0.1, 20)],
        positive_vars=[True, True],
    ))

    # C7: v_cm = (m1*v1 + m2*v2)/(m1+m2)
    templates.append(EquationTemplate(
        template_id="T3.07", tier=3,
        description="v_cm = (m1*v1 + m2*v2)/(m1+m2)",
        n_input_vars=4,
        eval_fn=lambda x, c: (x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]) / (x[:, 0] + x[:, 2]),
        to_prefix=lambda c: ["div", "add", "mul", "x1", "x2", "mul", "x3", "x4",
                              "add", "x1", "x3"],
        coeff_ranges=[],
        var_ranges=[(0.1, 50), (-20, 20), (0.1, 50), (-20, 20)],
        positive_vars=[True, False, True, False],
    ))

    # C8: E = 0.5*k*x^2 + 0.5*m*v^2 (total energy SHM)
    templates.append(EquationTemplate(
        template_id="T3.08", tier=3,
        description="E = 0.5*k*x^2 + 0.5*m*v^2",
        n_input_vars=3,  # x, m, v; k is coeff
        eval_fn=lambda x, c: 0.5 * c[0] * x[:, 0] ** 2 + 0.5 * x[:, 1] * x[:, 2] ** 2,
        to_prefix=lambda c: ["add", "mul", "mul", "C0.5", f"C{c[0]:.4f}", "pow", "x1", "C2",
                              "mul", "mul", "C0.5", "x2", "pow", "x3", "C2"],
        coeff_ranges=[(0.1, 100)],
        var_ranges=[(-5, 5), (0.1, 50), (-20, 20)],
        positive_vars=[False, True, False],
    ))

    # C9: E = mgh + 0.5*m*v^2
    templates.append(EquationTemplate(
        template_id="T3.09", tier=3,
        description="E = m*g*h + 0.5*m*v^2",
        n_input_vars=3,  # m, h, v; g is coeff
        eval_fn=lambda x, c: x[:, 0] * c[0] * x[:, 1] + 0.5 * x[:, 0] * x[:, 2] ** 2,
        to_prefix=lambda c: ["add", "mul", "mul", "x1", f"C{c[0]:.4f}", "x2",
                              "mul", "mul", "C0.5", "x1", "pow", "x3", "C2"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 50), (0, 50), (-20, 20)],
        positive_vars=[True, True, False],
    ))

    # C10: KE_loss = 0.5*m*v1^2 - mu*m*g*d
    templates.append(EquationTemplate(
        template_id="T3.10", tier=3,
        description="v2^2 = v1^2 - 2*mu*g*d",
        n_input_vars=2,  # v1, d; mu, g coeff
        eval_fn=lambda x, c: x[:, 0] ** 2 - 2 * c[0] * c[1] * x[:, 1],
        to_prefix=lambda c: ["sub", "pow", "x1", "C2", "mul", "mul", "C2",
                              f"C{c[0]:.4f}", "mul", f"C{c[1]:.4f}", "x2"],
        coeff_ranges=[(0.05, 0.8), (9.0, 10.0)],
        var_ranges=[(1, 20), (0.1, 50)],
        positive_vars=[True, True],
    ))

    # C11: I1*w1 = I2*w2 -> w2 = I1*w1/I2
    templates.append(EquationTemplate(
        template_id="T3.11", tier=3,
        description="w2 = I1*w1/I2",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] * x[:, 1] / x[:, 2],
        to_prefix=lambda c: ["div", "mul", "x1", "x2", "x3"],
        coeff_ranges=[],
        var_ranges=[(0.1, 50), (0.1, 20), (0.1, 50)],
        positive_vars=[True, True, True],
    ))

    # C12: elastic collision: v1_f = (m1-m2)/(m1+m2)*v1_i + 2*m2/(m1+m2)*v2_i
    templates.append(EquationTemplate(
        template_id="T3.12", tier=3,
        description="v1_f = (m1-m2)/(m1+m2)*v1i + 2*m2/(m1+m2)*v2i",
        n_input_vars=4,  # m1, m2, v1i, v2i
        eval_fn=lambda x, c: ((x[:, 0] - x[:, 1]) / (x[:, 0] + x[:, 1])) * x[:, 2] + \
                               (2 * x[:, 1] / (x[:, 0] + x[:, 1])) * x[:, 3],
        to_prefix=lambda c: ["add",
                              "mul", "div", "sub", "x1", "x2", "add", "x1", "x2", "x3",
                              "mul", "div", "mul", "C2", "x2", "add", "x1", "x2", "x4"],
        coeff_ranges=[],
        var_ranges=[(0.1, 50), (0.1, 50), (-20, 20), (-20, 20)],
        positive_vars=[True, True, False, False],
    ))

    return templates


def build_tier4_templates() -> List[EquationTemplate]:
    """Tier 4: Coupled/composite systems (complex, multi-equation)."""
    templates = []

    # D1: x_proj = v0*cos(theta)*t
    templates.append(EquationTemplate(
        template_id="T4.01", tier=4,
        description="x_proj = v0*cos(theta)*t",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] * np.cos(x[:, 1]) * x[:, 2],
        to_prefix=lambda c: ["mul", "mul", "x1", "cos", "x2", "x3"],
        coeff_ranges=[],
        var_ranges=[(1, 50), (0.1, 1.4), (0.01, 10)],
        positive_vars=[True, True, True],
    ))

    # D2: y_proj = v0*sin(theta)*t - 0.5*g*t^2
    templates.append(EquationTemplate(
        template_id="T4.02", tier=4,
        description="y_proj = v0*sin(theta)*t - 0.5*g*t^2",
        n_input_vars=3,
        eval_fn=lambda x, c: x[:, 0] * np.sin(x[:, 1]) * x[:, 2] - 0.5 * c[0] * x[:, 2] ** 2,
        to_prefix=lambda c: ["sub", "mul", "mul", "x1", "sin", "x2", "x3",
                              "mul", "mul", "C0.5", f"C{c[0]:.4f}", "pow", "x3", "C2"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(1, 50), (0.1, 1.4), (0.01, 5)],
        positive_vars=[True, True, True],
    ))

    # D3: a = g*sin(theta) - mu*g*cos(theta)
    templates.append(EquationTemplate(
        template_id="T4.03", tier=4,
        description="a = g*sin(theta) - mu*g*cos(theta)",
        n_input_vars=1,  # theta; g, mu are coeffs
        eval_fn=lambda x, c: c[0] * np.sin(x[:, 0]) - c[1] * c[0] * np.cos(x[:, 0]),
        to_prefix=lambda c: ["sub", "mul", f"C{c[0]:.4f}", "sin", "x1",
                              "mul", "mul", f"C{c[1]:.4f}", f"C{c[0]:.4f}", "cos", "x1"],
        coeff_ranges=[(9.0, 10.0), (0.05, 0.8)],
        var_ranges=[(0.05, 1.4)],
        positive_vars=[True],
    ))

    # D4: a_atwood = (m1-m2)*g/(m1+m2)
    templates.append(EquationTemplate(
        template_id="T4.04", tier=4,
        description="a = (m1-m2)*g/(m1+m2)",
        n_input_vars=2,  # m1, m2; g is coeff
        eval_fn=lambda x, c: (x[:, 0] - x[:, 1]) * c[0] / (x[:, 0] + x[:, 1]),
        to_prefix=lambda c: ["div", "mul", "sub", "x1", "x2", f"C{c[0]:.4f}",
                              "add", "x1", "x2"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 50), (0.1, 50)],
        positive_vars=[True, True],
    ))

    # D5: v = sqrt(2*g*h/(1 + I/(m*r^2)))
    templates.append(EquationTemplate(
        template_id="T4.05", tier=4,
        description="v = sqrt(2*g*h/(1 + I/(m*r^2)))",
        n_input_vars=4,  # h, I, m, r; g is coeff
        eval_fn=lambda x, c: np.sqrt(np.abs(2 * c[0] * x[:, 0] / (1 + x[:, 1] / (x[:, 2] * x[:, 3] ** 2)))),
        to_prefix=lambda c: ["sqrt", "div", "mul", "mul", "C2", f"C{c[0]:.4f}", "x1",
                              "add", "C1", "div", "x2", "mul", "x3", "pow", "x4", "C2"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 20), (0.01, 5), (0.1, 50), (0.05, 1)],
        positive_vars=[True, True, True, True],
    ))

    # D6: v_terminal = m*g/b
    templates.append(EquationTemplate(
        template_id="T4.06", tier=4,
        description="v_terminal = m*g/b",
        n_input_vars=2,  # m, b; g is coeff
        eval_fn=lambda x, c: x[:, 0] * c[0] / x[:, 1],
        to_prefix=lambda c: ["div", "mul", "x1", f"C{c[0]:.4f}", "x2"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 100), (0.01, 10)],
        positive_vars=[True, True],
    ))

    # D7: T = 2*pi*sqrt(I/(m*g*d))
    templates.append(EquationTemplate(
        template_id="T4.07", tier=4,
        description="T = 2*pi*sqrt(I/(m*g*d))",
        n_input_vars=3,  # I, m, d; g is coeff
        eval_fn=lambda x, c: 2 * np.pi * np.sqrt(x[:, 0] / (x[:, 1] * c[0] * x[:, 2])),
        to_prefix=lambda c: ["mul", "mul", "C2", "Cpi", "sqrt", "div", "x1",
                              "mul", "mul", "x2", f"C{c[0]:.4f}", "x3"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.01, 5), (0.1, 50), (0.05, 2)],
        positive_vars=[True, True, True],
    ))

    # D8: E_orbital = -G*m1*m2/(2*a)
    templates.append(EquationTemplate(
        template_id="T4.08", tier=4,
        description="E = -G*m1*m2/(2*a)",
        n_input_vars=3,  # m1, m2, a; G is coeff
        eval_fn=lambda x, c: -c[0] * x[:, 0] * x[:, 1] / (2 * x[:, 2]),
        to_prefix=lambda c: ["neg", "div", "mul", "mul", f"C{c[0]:.4f}", "x1", "x2",
                              "mul", "C2", "x3"],
        coeff_ranges=[(0.1, 10)],
        var_ranges=[(0.1, 100), (0.1, 100), (0.5, 50)],
        positive_vars=[True, True, True],
    ))

    # D9: x(t) = exp(-gamma*t)*(A*cos(wd*t) + B*sin(wd*t))
    templates.append(EquationTemplate(
        template_id="T4.09", tier=4,
        description="x(t) = exp(-gamma*t)*(A*cos(wd*t) + B*sin(wd*t))",
        n_input_vars=1,  # t; gamma, A, B, wd are coeffs
        eval_fn=lambda x, c: np.exp(-c[0] * x[:, 0]) * (c[1] * np.cos(c[3] * x[:, 0]) + c[2] * np.sin(c[3] * x[:, 0])),
        to_prefix=lambda c: ["mul", "exp", "neg", "mul", f"C{c[0]:.4f}", "x1",
                              "add", "mul", f"C{c[1]:.4f}", "cos", "mul", f"C{c[3]:.4f}", "x1",
                              "mul", f"C{c[2]:.4f}", "sin", "mul", f"C{c[3]:.4f}", "x1"],
        coeff_ranges=[(0.1, 2), (0.5, 5), (0.5, 5), (1, 10)],
        var_ranges=[(0, 5)],
        positive_vars=[True],
    ))

    # D10: coupled oscillator: x = A*cos(w1*t) + B*cos(w2*t)
    templates.append(EquationTemplate(
        template_id="T4.10", tier=4,
        description="x = A*cos(w1*t) + B*cos(w2*t)",
        n_input_vars=1,
        eval_fn=lambda x, c: c[0] * np.cos(c[2] * x[:, 0]) + c[1] * np.cos(c[3] * x[:, 0]),
        to_prefix=lambda c: ["add", "mul", f"C{c[0]:.4f}", "cos", "mul", f"C{c[2]:.4f}", "x1",
                              "mul", f"C{c[1]:.4f}", "cos", "mul", f"C{c[3]:.4f}", "x1"],
        coeff_ranges=[(0.5, 5), (0.5, 5), (1, 10), (1, 10)],
        var_ranges=[(0, 10)],
        positive_vars=[True],
    ))

    # D11: projectile range: R = v0^2*sin(2*theta)/g
    templates.append(EquationTemplate(
        template_id="T4.11", tier=4,
        description="R = v0^2*sin(2*theta)/g",
        n_input_vars=2,  # v0, theta; g is coeff
        eval_fn=lambda x, c: x[:, 0] ** 2 * np.sin(2 * x[:, 1]) / c[0],
        to_prefix=lambda c: ["div", "mul", "pow", "x1", "C2", "sin", "mul", "C2", "x2", f"C{c[0]:.4f}"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(1, 50), (0.05, 1.5)],
        positive_vars=[True, True],
    ))

    # D12: F_net = m*g - b*v (falling with drag)
    templates.append(EquationTemplate(
        template_id="T4.12", tier=4,
        description="F_net = m*g - b*v",
        n_input_vars=3,  # m, b, v; g is coeff
        eval_fn=lambda x, c: x[:, 0] * c[0] - x[:, 1] * x[:, 2],
        to_prefix=lambda c: ["sub", "mul", "x1", f"C{c[0]:.4f}", "mul", "x2", "x3"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 50), (0.01, 10), (0, 30)],
        positive_vars=[True, True, True],
    ))

    # D13: precession: wp = tau/(I*ws) = m*g*d/(I*ws)
    templates.append(EquationTemplate(
        template_id="T4.13", tier=4,
        description="wp = m*g*d/(I*ws)",
        n_input_vars=4,  # m, d, I, ws; g is coeff
        eval_fn=lambda x, c: x[:, 0] * c[0] * x[:, 1] / (x[:, 2] * x[:, 3]),
        to_prefix=lambda c: ["div", "mul", "mul", "x1", f"C{c[0]:.4f}", "x2", "mul", "x3", "x4"],
        coeff_ranges=[(9.0, 10.0)],
        var_ranges=[(0.1, 50), (0.01, 2), (0.01, 5), (1, 50)],
        positive_vars=[True, True, True, True],
    ))

    # D14: drag projectile x = (v0*cos(theta)/b)*(1-exp(-b*t))
    templates.append(EquationTemplate(
        template_id="T4.14", tier=4,
        description="x = (v0*cos(theta)/b)*(1-exp(-b*t))",
        n_input_vars=2,  # v0, t; theta, b are coeffs
        eval_fn=lambda x, c: (x[:, 0] * np.cos(c[0]) / c[1]) * (1 - np.exp(-c[1] * x[:, 1])),
        to_prefix=lambda c: ["mul", "div", "mul", "x1", "cos", f"C{c[0]:.4f}", f"C{c[1]:.4f}",
                              "sub", "C1", "exp", "neg", "mul", f"C{c[1]:.4f}", "x2"],
        coeff_ranges=[(0.1, 1.4), (0.1, 2)],
        var_ranges=[(1, 50), (0.01, 10)],
        positive_vars=[True, True],
    ))

    return templates


def get_all_templates() -> List[EquationTemplate]:
    """Return all equation templates across all tiers."""
    return (build_tier1_templates() + build_tier2_templates() +
            build_tier3_templates() + build_tier4_templates())


# Vocabulary for equation tokens
EQUATION_VOCAB = {
    # Operators
    "add": 0, "sub": 1, "mul": 2, "div": 3, "pow": 4,
    "sqrt": 5, "sin": 6, "cos": 7, "exp": 8, "log": 9,
    "neg": 10, "abs": 11, "sgn": 12,
    # Variables
    "x1": 13, "x2": 14, "x3": 15, "x4": 16, "x5": 17, "x6": 18,
    # Special constants
    "C0.5": 19, "C1": 20, "C2": 21, "C3": 22, "Cpi": 23,
    # Special tokens
    "[PAD]": 24, "[EQ_START]": 25, "[EQ_END]": 26, "[MASK]": 27,
    # Numerical constant slots (for arbitrary constants)
    # We reserve indices 28-127 for discretized constant tokens
}

# Build inverse vocab
INV_EQUATION_VOCAB = {v: k for k, v in EQUATION_VOCAB.items()}

# Add discretized constant bins
NUM_CONST_BINS = 100
CONST_BIN_START = 28
for i in range(NUM_CONST_BINS):
    key = f"CBIN{i}"
    EQUATION_VOCAB[key] = CONST_BIN_START + i
    INV_EQUATION_VOCAB[CONST_BIN_START + i] = key

VOCAB_SIZE = CONST_BIN_START + NUM_CONST_BINS  # 128


def constant_to_bin(value: float) -> int:
    """Map a constant value to a discretized bin index."""
    # Map values in [-100, 100] to bins 0-99 using log-like scaling
    sign = 1 if value >= 0 else -1
    abs_val = abs(value)
    if abs_val < 0.01:
        bin_idx = 50  # Near-zero bin
    else:
        # Log scale mapping: [0.01, 100] -> [0, 49]
        log_val = np.log10(abs_val)  # Range: [-2, 2]
        normalized = (log_val + 2) / 4  # Range: [0, 1]
        normalized = np.clip(normalized, 0, 1)
        half_bin = int(normalized * 49)
        if sign > 0:
            bin_idx = 50 + half_bin  # 50-99 for positive
        else:
            bin_idx = 49 - half_bin  # 0-49 for negative
    return np.clip(bin_idx, 0, NUM_CONST_BINS - 1)


def bin_to_constant(bin_idx: int) -> float:
    """Map a bin index back to an approximate constant value."""
    if bin_idx == 50:
        return 0.0
    elif bin_idx > 50:
        normalized = (bin_idx - 50) / 49  # [0, 1]
        log_val = normalized * 4 - 2  # [-2, 2]
        return 10 ** log_val
    else:
        normalized = (49 - bin_idx) / 49  # [0, 1]
        log_val = normalized * 4 - 2
        return -(10 ** log_val)


def tokenize_prefix(prefix_tokens: List[str]) -> List[int]:
    """Convert prefix notation token strings to integer indices."""
    result = [EQUATION_VOCAB["[EQ_START]"]]
    for tok in prefix_tokens:
        if tok in EQUATION_VOCAB:
            result.append(EQUATION_VOCAB[tok])
        elif tok.startswith("C") and tok not in EQUATION_VOCAB:
            # Parse numerical constant
            try:
                val = float(tok[1:])
                bin_idx = constant_to_bin(val)
                result.append(CONST_BIN_START + bin_idx)
            except ValueError:
                result.append(EQUATION_VOCAB["[PAD]"])
        else:
            result.append(EQUATION_VOCAB["[PAD]"])
    result.append(EQUATION_VOCAB["[EQ_END]"])
    return result
