"""
Physics Equation Corpus for PhysMDT
====================================

Defines 55+ physics equations across 5 complexity tiers, plus 11 held-out
equations for zero-shot discovery evaluation. All symbolic expressions are
built with SymPy. Variables use generic names (x0, x1, ...) and constants
use named placeholders (C0, C1, ...) with default physical values.

Tier 1 (12 eqs): Single-variable linear relationships
Tier 2 (12 eqs): Multi-variable polynomial / quadratic
Tier 3 (12 eqs): Inverse-square / rational / sqrt
Tier 4 (10 eqs): Trigonometric and oscillatory
Tier 5 (4 training + 11 held-out): Multi-step derivations and advanced
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sympy
from sympy import (
    Abs,
    Rational,
    Symbol,
    cos,
    exp,
    log,
    pi,
    sin,
    sqrt,
    tan,
)

# ---------------------------------------------------------------------------
# Generic variable symbols  (x0 .. x9)
# ---------------------------------------------------------------------------
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = sympy.symbols(
    "x0 x1 x2 x3 x4 x5 x6 x7 x8 x9", positive=True
)

_ALL_VARS = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]

# ---------------------------------------------------------------------------
# Physical constants (used as named constants C0, C1, ... in equations)
# ---------------------------------------------------------------------------
GRAVITATIONAL_CONSTANT = 6.67430e-11     # G  [m^3 kg^-1 s^-2]
COULOMB_CONSTANT = 8.9875517873681764e9  # k_e [N m^2 C^-2]
SPEED_OF_LIGHT = 2.99792458e8            # c  [m/s]
PLANCK_CONSTANT = 6.62607015e-34         # h  [J s]
BOLTZMANN_CONSTANT = 1.380649e-23        # k_B [J/K]
STEFAN_BOLTZMANN = 5.670374419e-8        # sigma [W m^-2 K^-4]


# ---------------------------------------------------------------------------
# Equation data-class
# ---------------------------------------------------------------------------
@dataclass
class Equation:
    """A single physics equation in the corpus."""

    id: str                              # Unique identifier, e.g. "t1_01"
    name: str                            # Human-readable name
    symbolic_expr: sympy.Expr            # SymPy expression (RHS of equation)
    variables: List[sympy.Symbol]        # Ordered list of SymPy symbols used
    constants: Dict[str, float]          # Named constants -> default value
    tier: int                            # Complexity tier 1-5
    held_out: bool                       # True => never seen in training
    prefix_notation: str                 # Prefix (Polish) notation string
    var_ranges: Dict[str, Tuple[float, float]]  # {var_name: (min, max)}
    description: str                     # One-line description


# ===================================================================
# Helper to build variable ranges that avoid singularities
# ===================================================================
def _default_range(
    variables: List[sympy.Symbol],
    lo: float = 0.1,
    hi: float = 10.0,
) -> Dict[str, Tuple[float, float]]:
    """Return {str(var): (lo, hi)} for every variable."""
    return {str(v): (lo, hi) for v in variables}


def _custom_ranges(
    variables: List[sympy.Symbol],
    overrides: Optional[Dict[str, Tuple[float, float]]] = None,
    lo: float = 0.1,
    hi: float = 10.0,
) -> Dict[str, Tuple[float, float]]:
    """Default range for all variables with optional per-variable overrides."""
    ranges = _default_range(variables, lo, hi)
    if overrides:
        ranges.update(overrides)
    return ranges


# ===================================================================
# Tier 1 — Single-variable linear relationships (12 equations)
# ===================================================================

def _tier1_equations() -> List[Equation]:
    eqs: List[Equation] = []

    # T1-01  F = m * a   (Newton's second law)
    eqs.append(Equation(
        id="t1_01",
        name="Newton's second law (F=ma)",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Force equals mass times acceleration",
    ))

    # T1-02  v = v0 + a*t
    eqs.append(Equation(
        id="t1_02",
        name="Velocity with constant acceleration",
        symbolic_expr=x0 + x1 * x2,
        variables=[x0, x1, x2],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="+ x0 * x1 x2",
        var_ranges=_default_range([x0, x1, x2]),
        description="v = v0 + a*t",
    ))

    # T1-03  p = m * v   (momentum)
    eqs.append(Equation(
        id="t1_03",
        name="Linear momentum",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Momentum equals mass times velocity",
    ))

    # T1-04  W = F * d   (work)
    eqs.append(Equation(
        id="t1_04",
        name="Work (W=Fd)",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Work equals force times displacement",
    ))

    # T1-05  P = W / t   (power)
    eqs.append(Equation(
        id="t1_05",
        name="Power (P=W/t)",
        symbolic_expr=x0 / x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="/ x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.1, 10.0)}),
        description="Power equals work divided by time",
    ))

    # T1-06  I = F * t   (impulse)
    eqs.append(Equation(
        id="t1_06",
        name="Impulse (I=Ft)",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Impulse equals force times time",
    ))

    # T1-07  rho = m / V   (density)
    eqs.append(Equation(
        id="t1_07",
        name="Density (rho=m/V)",
        symbolic_expr=x0 / x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="/ x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.1, 10.0)}),
        description="Density equals mass divided by volume",
    ))

    # T1-08  P = F / A   (pressure)
    eqs.append(Equation(
        id="t1_08",
        name="Pressure (P=F/A)",
        symbolic_expr=x0 / x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="/ x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.1, 10.0)}),
        description="Pressure equals force divided by area",
    ))

    # T1-09  v = d / t   (speed)
    eqs.append(Equation(
        id="t1_09",
        name="Speed (v=d/t)",
        symbolic_expr=x0 / x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="/ x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.1, 10.0)}),
        description="Speed equals distance divided by time",
    ))

    # T1-10  a = F / m   (acceleration from force)
    eqs.append(Equation(
        id="t1_10",
        name="Acceleration (a=F/m)",
        symbolic_expr=x0 / x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="/ x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.1, 10.0)}),
        description="Acceleration equals force divided by mass",
    ))

    # T1-11  tau = F * r   (torque)
    eqs.append(Equation(
        id="t1_11",
        name="Torque (tau=Fr)",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Torque equals force times lever arm",
    ))

    # T1-12  f = 1 / T   (frequency)
    eqs.append(Equation(
        id="t1_12",
        name="Frequency (f=1/T)",
        symbolic_expr=1 / x0,
        variables=[x0],
        constants={},
        tier=1,
        held_out=False,
        prefix_notation="/ 1 x0",
        var_ranges=_custom_ranges([x0], {"x0": (0.1, 10.0)}),
        description="Frequency equals one over period",
    ))

    return eqs


# ===================================================================
# Tier 2 — Multi-variable polynomial / quadratic (12 equations)
# ===================================================================

def _tier2_equations() -> List[Equation]:
    eqs: List[Equation] = []

    # T2-01  KE = 0.5 * m * v^2
    eqs.append(Equation(
        id="t2_01",
        name="Kinetic energy",
        symbolic_expr=Rational(1, 2) * x0 * x1**2,
        variables=[x0, x1],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* * 0.5 x0 ^ x1 2",
        var_ranges=_default_range([x0, x1]),
        description="KE = 0.5 * m * v^2",
    ))

    # T2-02  PE = m * g * h   (gravitational PE near surface)
    # g stored as constant C0 = 9.81
    eqs.append(Equation(
        id="t2_02",
        name="Gravitational potential energy (surface)",
        symbolic_expr=x0 * x1 * x2,
        variables=[x0, x1, x2],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* * x0 x1 x2",
        var_ranges=_default_range([x0, x1, x2]),
        description="PE = m * g * h (g as variable for generality)",
    ))

    # T2-03  s = v0*t + 0.5*a*t^2
    eqs.append(Equation(
        id="t2_03",
        name="Displacement (constant acceleration)",
        symbolic_expr=x0 * x2 + Rational(1, 2) * x1 * x2**2,
        variables=[x0, x1, x2],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="+ * x0 x2 * * 0.5 x1 ^ x2 2",
        var_ranges=_default_range([x0, x1, x2]),
        description="s = v0*t + 0.5*a*t^2",
    ))

    # T2-04  v^2 = v0^2 + 2*a*s  => v_final = sqrt(v0^2 + 2*a*s)
    eqs.append(Equation(
        id="t2_04",
        name="Final velocity squared",
        symbolic_expr=x0**2 + 2 * x1 * x2,
        variables=[x0, x1, x2],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="+ ^ x0 2 * * 2 x1 x2",
        var_ranges=_default_range([x0, x1, x2]),
        description="v^2 = v0^2 + 2*a*s",
    ))

    # T2-05  E = 0.5*m*v^2 + m*g*h  (total mechanical energy)
    eqs.append(Equation(
        id="t2_05",
        name="Total mechanical energy",
        symbolic_expr=Rational(1, 2) * x0 * x1**2 + x0 * x2 * x3,
        variables=[x0, x1, x2, x3],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="+ * * 0.5 x0 ^ x1 2 * * x0 x2 x3",
        var_ranges=_default_range([x0, x1, x2, x3]),
        description="E = 0.5*m*v^2 + m*g*h",
    ))

    # T2-06  F_spring = k * x  (Hooke's law)
    eqs.append(Equation(
        id="t2_06",
        name="Hooke's law (F=kx)",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Spring force F = k * x",
    ))

    # T2-07  PE_spring = 0.5 * k * x^2
    eqs.append(Equation(
        id="t2_07",
        name="Spring potential energy",
        symbolic_expr=Rational(1, 2) * x0 * x1**2,
        variables=[x0, x1],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* * 0.5 x0 ^ x1 2",
        var_ranges=_default_range([x0, x1]),
        description="PE_spring = 0.5 * k * x^2",
    ))

    # T2-08  Q = m * c * dT  (heat transfer)
    eqs.append(Equation(
        id="t2_08",
        name="Heat transfer (Q=mcDT)",
        symbolic_expr=x0 * x1 * x2,
        variables=[x0, x1, x2],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* * x0 x1 x2",
        var_ranges=_default_range([x0, x1, x2]),
        description="Q = m * c * delta_T",
    ))

    # T2-09  L = I * omega  (angular momentum)
    eqs.append(Equation(
        id="t2_09",
        name="Angular momentum (L=Iw)",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Angular momentum L = I * omega",
    ))

    # T2-10  KE_rot = 0.5 * I * omega^2
    eqs.append(Equation(
        id="t2_10",
        name="Rotational kinetic energy",
        symbolic_expr=Rational(1, 2) * x0 * x1**2,
        variables=[x0, x1],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* * 0.5 x0 ^ x1 2",
        var_ranges=_default_range([x0, x1]),
        description="KE_rot = 0.5 * I * omega^2",
    ))

    # T2-11  F_friction = mu * N
    eqs.append(Equation(
        id="t2_11",
        name="Friction force (F=muN)",
        symbolic_expr=x0 * x1,
        variables=[x0, x1],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="* x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="Friction force F = mu * N",
    ))

    # T2-12  v_cm = (m1*v1 + m2*v2) / (m1 + m2)
    eqs.append(Equation(
        id="t2_12",
        name="Center-of-mass velocity",
        symbolic_expr=(x0 * x1 + x2 * x3) / (x0 + x2),
        variables=[x0, x1, x2, x3],
        constants={},
        tier=2,
        held_out=False,
        prefix_notation="/ + * x0 x1 * x2 x3 + x0 x2",
        var_ranges=_default_range([x0, x1, x2, x3]),
        description="v_cm = (m1*v1 + m2*v2) / (m1 + m2)",
    ))

    return eqs


# ===================================================================
# Tier 3 — Inverse-square / rational / sqrt (12 equations)
# ===================================================================

def _tier3_equations() -> List[Equation]:
    eqs: List[Equation] = []

    # T3-01  F_grav = G*m1*m2 / r^2
    eqs.append(Equation(
        id="t3_01",
        name="Newton's law of gravitation",
        symbolic_expr=x0 * x1 * x2 / x3**2,
        variables=[x0, x1, x2, x3],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=3,
        held_out=False,
        prefix_notation="/ * * x0 x1 x2 ^ x3 2",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3],
            {"x0": (1.0, 10.0), "x3": (0.5, 10.0)},
        ),
        description="F = G*m1*m2/r^2 (x0=G*scale, x1=m1, x2=m2, x3=r)",
    ))

    # T3-02  F_coulomb = k*q1*q2 / r^2
    eqs.append(Equation(
        id="t3_02",
        name="Coulomb's law",
        symbolic_expr=x0 * x1 * x2 / x3**2,
        variables=[x0, x1, x2, x3],
        constants={"C0_ke": COULOMB_CONSTANT},
        tier=3,
        held_out=False,
        prefix_notation="/ * * x0 x1 x2 ^ x3 2",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3],
            {"x0": (1.0, 10.0), "x3": (0.5, 10.0)},
        ),
        description="F = k_e*q1*q2/r^2 (x0=k*scale, x1=q1, x2=q2, x3=r)",
    ))

    # T3-03  g = G*M / R^2  (gravitational field)
    eqs.append(Equation(
        id="t3_03",
        name="Gravitational field strength",
        symbolic_expr=x0 * x1 / x2**2,
        variables=[x0, x1, x2],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=3,
        held_out=False,
        prefix_notation="/ * x0 x1 ^ x2 2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.5, 10.0)},
        ),
        description="g = G*M/R^2",
    ))

    # T3-04  v_esc = sqrt(2*G*M / R)
    eqs.append(Equation(
        id="t3_04",
        name="Escape velocity",
        symbolic_expr=sqrt(2 * x0 * x1 / x2),
        variables=[x0, x1, x2],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=3,
        held_out=False,
        prefix_notation="sqrt / * * 2 x0 x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.5, 10.0)},
        ),
        description="v_esc = sqrt(2*G*M/R)",
    ))

    # T3-05  PE_grav = -G*m1*m2 / r  (stored as positive: G*m1*m2/r)
    # We keep the negative sign in the expression.
    eqs.append(Equation(
        id="t3_05",
        name="Gravitational potential energy",
        symbolic_expr=-x0 * x1 * x2 / x3,
        variables=[x0, x1, x2, x3],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=3,
        held_out=False,
        prefix_notation="neg / * * x0 x1 x2 x3",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3],
            {"x3": (0.5, 10.0)},
        ),
        description="PE = -G*m1*m2/r",
    ))

    # T3-06  a_cent = v^2 / r  (centripetal acceleration)
    eqs.append(Equation(
        id="t3_06",
        name="Centripetal acceleration",
        symbolic_expr=x0**2 / x1,
        variables=[x0, x1],
        constants={},
        tier=3,
        held_out=False,
        prefix_notation="/ ^ x0 2 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.5, 10.0)}),
        description="a_cent = v^2 / r",
    ))

    # T3-07  F_cent = m*v^2 / r
    eqs.append(Equation(
        id="t3_07",
        name="Centripetal force",
        symbolic_expr=x0 * x1**2 / x2,
        variables=[x0, x1, x2],
        constants={},
        tier=3,
        held_out=False,
        prefix_notation="/ * x0 ^ x1 2 x2",
        var_ranges=_custom_ranges([x0, x1, x2], {"x2": (0.5, 10.0)}),
        description="F_cent = m*v^2/r",
    ))

    # T3-08  v_orb = sqrt(G*M / r)
    eqs.append(Equation(
        id="t3_08",
        name="Orbital velocity",
        symbolic_expr=sqrt(x0 * x1 / x2),
        variables=[x0, x1, x2],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=3,
        held_out=False,
        prefix_notation="sqrt / * x0 x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.5, 10.0)},
        ),
        description="v_orb = sqrt(G*M/r)",
    ))

    # T3-09  T_orb = 2*pi*sqrt(r^3 / (G*M))
    eqs.append(Equation(
        id="t3_09",
        name="Orbital period",
        symbolic_expr=2 * pi * sqrt(x2**3 / (x0 * x1)),
        variables=[x0, x1, x2],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=3,
        held_out=False,
        prefix_notation="* * 2 pi sqrt / ^ x2 3 * x0 x1",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x0": (0.5, 10.0), "x1": (0.5, 10.0)},
        ),
        description="T_orb = 2*pi*sqrt(r^3/(G*M))",
    ))

    # T3-10  I_sphere = (2/5)*m*r^2
    eqs.append(Equation(
        id="t3_10",
        name="Moment of inertia (solid sphere)",
        symbolic_expr=Rational(2, 5) * x0 * x1**2,
        variables=[x0, x1],
        constants={},
        tier=3,
        held_out=False,
        prefix_notation="* * 0.4 x0 ^ x1 2",
        var_ranges=_default_range([x0, x1]),
        description="I = (2/5)*m*r^2",
    ))

    # T3-11  F_drag = 0.5*rho*v^2*C*A
    eqs.append(Equation(
        id="t3_11",
        name="Drag force",
        symbolic_expr=Rational(1, 2) * x0 * x1**2 * x2 * x3,
        variables=[x0, x1, x2, x3],
        constants={},
        tier=3,
        held_out=False,
        prefix_notation="* * * * 0.5 x0 ^ x1 2 x2 x3",
        var_ranges=_default_range([x0, x1, x2, x3]),
        description="F_drag = 0.5*rho*v^2*C_d*A",
    ))

    # T3-12  v_term = sqrt(2*m*g / (rho*C*A))
    eqs.append(Equation(
        id="t3_12",
        name="Terminal velocity",
        symbolic_expr=sqrt(2 * x0 * x1 / (x2 * x3 * x4)),
        variables=[x0, x1, x2, x3, x4],
        constants={},
        tier=3,
        held_out=False,
        prefix_notation="sqrt / * * 2 x0 x1 * * x2 x3 x4",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3, x4],
            {"x2": (0.5, 10.0), "x3": (0.5, 10.0), "x4": (0.5, 10.0)},
        ),
        description="v_term = sqrt(2*m*g/(rho*C_d*A))",
    ))

    return eqs


# ===================================================================
# Tier 4 — Trigonometric / oscillatory (10 equations)
# ===================================================================

def _tier4_equations() -> List[Equation]:
    eqs: List[Equation] = []

    # T4-01  T_pend = 2*pi*sqrt(L/g)
    eqs.append(Equation(
        id="t4_01",
        name="Simple pendulum period",
        symbolic_expr=2 * pi * sqrt(x0 / x1),
        variables=[x0, x1],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="* * 2 pi sqrt / x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.5, 10.0)}),
        description="T = 2*pi*sqrt(L/g)",
    ))

    # T4-02  R_proj = v0^2 * sin(2*theta) / g
    eqs.append(Equation(
        id="t4_02",
        name="Projectile range",
        symbolic_expr=x0**2 * sin(2 * x1) / x2,
        variables=[x0, x1, x2],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="/ * ^ x0 2 sin * 2 x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x1": (0.1, 1.5), "x2": (0.5, 10.0)},  # theta in radians
        ),
        description="R = v0^2*sin(2*theta)/g",
    ))

    # T4-03  y_proj = v0*sin(theta)*t - 0.5*g*t^2
    eqs.append(Equation(
        id="t4_03",
        name="Projectile vertical position",
        symbolic_expr=x0 * sin(x1) * x2 - Rational(1, 2) * x3 * x2**2,
        variables=[x0, x1, x2, x3],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="- * * x0 sin x1 x2 * * 0.5 x3 ^ x2 2",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3],
            {"x1": (0.1, 1.5), "x2": (0.1, 3.0)},
        ),
        description="y = v0*sin(theta)*t - 0.5*g*t^2",
    ))

    # T4-04  x_proj = v0*cos(theta)*t
    eqs.append(Equation(
        id="t4_04",
        name="Projectile horizontal position",
        symbolic_expr=x0 * cos(x1) * x2,
        variables=[x0, x1, x2],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="* * x0 cos x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x1": (0.1, 1.5)},
        ),
        description="x = v0*cos(theta)*t",
    ))

    # T4-05  F_incline = m*g*sin(theta)
    eqs.append(Equation(
        id="t4_05",
        name="Force along incline",
        symbolic_expr=x0 * x1 * sin(x2),
        variables=[x0, x1, x2],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="* * x0 x1 sin x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.1, 1.5)},
        ),
        description="F_parallel = m*g*sin(theta)",
    ))

    # T4-06  N_incline = m*g*cos(theta)
    eqs.append(Equation(
        id="t4_06",
        name="Normal force on incline",
        symbolic_expr=x0 * x1 * cos(x2),
        variables=[x0, x1, x2],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="* * x0 x1 cos x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.1, 1.5)},
        ),
        description="N = m*g*cos(theta)",
    ))

    # T4-07  v_wave = sqrt(T / mu)  (wave speed on string)
    eqs.append(Equation(
        id="t4_07",
        name="Wave speed on string",
        symbolic_expr=sqrt(x0 / x1),
        variables=[x0, x1],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="sqrt / x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.5, 10.0)}),
        description="v_wave = sqrt(Tension/linear_mass_density)",
    ))

    # T4-08  omega = sqrt(k / m)  (angular frequency SHM)
    eqs.append(Equation(
        id="t4_08",
        name="Angular frequency (SHM)",
        symbolic_expr=sqrt(x0 / x1),
        variables=[x0, x1],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="sqrt / x0 x1",
        var_ranges=_custom_ranges([x0, x1], {"x1": (0.5, 10.0)}),
        description="omega = sqrt(k/m)",
    ))

    # T4-09  x_shm = A * sin(omega*t + phi)
    eqs.append(Equation(
        id="t4_09",
        name="SHM displacement",
        symbolic_expr=x0 * sin(x1 * x2 + x3),
        variables=[x0, x1, x2, x3],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="* x0 sin + * x1 x2 x3",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3],
            {"x3": (0.0, 6.28)},
        ),
        description="x = A*sin(omega*t + phi)",
    ))

    # T4-10  E_shm = 0.5 * k * A^2  (total SHM energy)
    eqs.append(Equation(
        id="t4_10",
        name="Total SHM energy",
        symbolic_expr=Rational(1, 2) * x0 * x1**2,
        variables=[x0, x1],
        constants={},
        tier=4,
        held_out=False,
        prefix_notation="* * 0.5 x0 ^ x1 2",
        var_ranges=_default_range([x0, x1]),
        description="E = 0.5*k*A^2",
    ))

    return eqs


# ===================================================================
# Tier 5 — Advanced / multi-step (4 training equations)
# ===================================================================

def _tier5_training_equations() -> List[Equation]:
    eqs: List[Equation] = []

    # T5-01  T_kepler = 2*pi*sqrt(a^3 / (G*M))   (same form as orbital period
    #         but semantically: Kepler's 3rd law with semi-major axis)
    eqs.append(Equation(
        id="t5_01",
        name="Kepler's third law",
        symbolic_expr=2 * pi * sqrt(x0**3 / (x1 * x2)),
        variables=[x0, x1, x2],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=5,
        held_out=False,
        prefix_notation="* * 2 pi sqrt / ^ x0 3 * x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x1": (0.5, 10.0), "x2": (0.5, 10.0)},
        ),
        description="T = 2*pi*sqrt(a^3/(G*M))  Kepler's third law",
    ))

    # T5-02  v_rocket = v_e * ln(m0 / mf)  (Tsiolkovsky rocket equation)
    eqs.append(Equation(
        id="t5_02",
        name="Tsiolkovsky rocket equation",
        symbolic_expr=x0 * log(x1 / x2),
        variables=[x0, x1, x2],
        constants={},
        tier=5,
        held_out=False,
        prefix_notation="* x0 log / x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x1": (1.0, 10.0), "x2": (0.1, 5.0)},  # m0 > mf
        ),
        description="delta_v = v_e * ln(m0/mf)",
    ))

    # T5-03  E_orbit = -G*M*m / (2*a)
    eqs.append(Equation(
        id="t5_03",
        name="Orbital energy",
        symbolic_expr=-x0 * x1 * x2 / (2 * x3),
        variables=[x0, x1, x2, x3],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=5,
        held_out=False,
        prefix_notation="neg / * * x0 x1 x2 * 2 x3",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3],
            {"x3": (0.5, 10.0)},
        ),
        description="E = -G*M*m/(2*a)",
    ))

    # T5-04  R_schwarz = 2*G*M / c^2   (Schwarzschild radius)
    eqs.append(Equation(
        id="t5_04",
        name="Schwarzschild radius",
        symbolic_expr=2 * x0 * x1 / x2**2,
        variables=[x0, x1, x2],
        constants={"C0_G": GRAVITATIONAL_CONSTANT, "C1_c": SPEED_OF_LIGHT},
        tier=5,
        held_out=False,
        prefix_notation="/ * * 2 x0 x1 ^ x2 2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.5, 10.0)},
        ),
        description="R_s = 2*G*M/c^2",
    ))

    return eqs


# ===================================================================
# Held-out equations (11) — NEVER used in training
# ===================================================================

def _held_out_equations() -> List[Equation]:
    eqs: List[Equation] = []

    # HO-01  Lorentz force = q * v * B
    eqs.append(Equation(
        id="ho_01",
        name="Lorentz force (magnetic)",
        symbolic_expr=x0 * x1 * x2,
        variables=[x0, x1, x2],
        constants={},
        tier=3,
        held_out=True,
        prefix_notation="* * x0 x1 x2",
        var_ranges=_default_range([x0, x1, x2]),
        description="F = q*v*B (magnitude of magnetic Lorentz force)",
    ))

    # HO-02  Wave power = 0.5 * rho * v * omega^2 * A^2
    eqs.append(Equation(
        id="ho_02",
        name="Wave power",
        symbolic_expr=Rational(1, 2) * x0 * x1 * x2**2 * x3**2,
        variables=[x0, x1, x2, x3],
        constants={},
        tier=4,
        held_out=True,
        prefix_notation="* * * * 0.5 x0 x1 ^ x2 2 ^ x3 2",
        var_ranges=_default_range([x0, x1, x2, x3]),
        description="P = 0.5*rho*v*omega^2*A^2",
    ))

    # HO-03  Doppler effect = v_s * f / (v_s - v_src)
    eqs.append(Equation(
        id="ho_03",
        name="Doppler effect (approaching source)",
        symbolic_expr=x0 * x1 / (x0 - x2),
        variables=[x0, x1, x2],
        constants={},
        tier=4,
        held_out=True,
        prefix_notation="/ * x0 x1 - x0 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x0": (5.0, 10.0), "x2": (0.1, 4.0)},  # v_s > v_src
        ),
        description="f_obs = v_s*f/(v_s - v_src)",
    ))

    # HO-04  Gravitational lensing deflection = 4*G*M / (c^2 * r)
    eqs.append(Equation(
        id="ho_04",
        name="Gravitational lensing deflection",
        symbolic_expr=4 * x0 * x1 / (x2**2 * x3),
        variables=[x0, x1, x2, x3],
        constants={"C0_G": GRAVITATIONAL_CONSTANT, "C1_c": SPEED_OF_LIGHT},
        tier=5,
        held_out=True,
        prefix_notation="/ * * 4 x0 x1 * ^ x2 2 x3",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3],
            {"x2": (0.5, 10.0), "x3": (0.5, 10.0)},
        ),
        description="theta = 4*G*M/(c^2*r)",
    ))

    # HO-05  LC circuit period = 2*pi*sqrt(L*C)
    eqs.append(Equation(
        id="ho_05",
        name="LC circuit resonant period",
        symbolic_expr=2 * pi * sqrt(x0 * x1),
        variables=[x0, x1],
        constants={},
        tier=4,
        held_out=True,
        prefix_notation="* * 2 pi sqrt * x0 x1",
        var_ranges=_default_range([x0, x1]),
        description="T = 2*pi*sqrt(L*C)",
    ))

    # HO-06  Coriolis acceleration = 2 * omega * v * sin(phi)
    eqs.append(Equation(
        id="ho_06",
        name="Coriolis acceleration",
        symbolic_expr=2 * x0 * x1 * sin(x2),
        variables=[x0, x1, x2],
        constants={},
        tier=4,
        held_out=True,
        prefix_notation="* * * 2 x0 x1 sin x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.1, 1.5)},  # latitude in radians
        ),
        description="a_cor = 2*omega*v*sin(phi)",
    ))

    # HO-07  Tidal force = 2*G*M*m*r / d^3
    eqs.append(Equation(
        id="ho_07",
        name="Tidal force",
        symbolic_expr=2 * x0 * x1 * x2 * x3 / x4**3,
        variables=[x0, x1, x2, x3, x4],
        constants={"C0_G": GRAVITATIONAL_CONSTANT},
        tier=5,
        held_out=True,
        prefix_notation="/ * * * * 2 x0 x1 x2 x3 ^ x4 3",
        var_ranges=_custom_ranges(
            [x0, x1, x2, x3, x4],
            {"x4": (0.5, 10.0)},
        ),
        description="F_tidal = 2*G*M*m*r/d^3",
    ))

    # HO-08  Time dilation = t0 / sqrt(1 - v^2/c^2)
    #   Using variables: x0=t0, x1=v, x2=c  with constraint x1 < x2
    eqs.append(Equation(
        id="ho_08",
        name="Time dilation (special relativity)",
        symbolic_expr=x0 / sqrt(1 - x1**2 / x2**2),
        variables=[x0, x1, x2],
        constants={"C0_c": SPEED_OF_LIGHT},
        tier=5,
        held_out=True,
        prefix_notation="/ x0 sqrt - 1 / ^ x1 2 ^ x2 2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x0": (0.1, 10.0), "x1": (0.1, 5.0), "x2": (6.0, 10.0)},
        ),
        description="t = t0 / sqrt(1 - v^2/c^2)  (x1 < x2 to stay subluminal)",
    ))

    # HO-09  de Broglie wavelength = h / (m * v)
    eqs.append(Equation(
        id="ho_09",
        name="de Broglie wavelength",
        symbolic_expr=x0 / (x1 * x2),
        variables=[x0, x1, x2],
        constants={"C0_h": PLANCK_CONSTANT},
        tier=3,
        held_out=True,
        prefix_notation="/ x0 * x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x1": (0.5, 10.0), "x2": (0.5, 10.0)},
        ),
        description="lambda = h/(m*v)",
    ))

    # HO-10  Stefan-Boltzmann law = sigma * A * T^4
    eqs.append(Equation(
        id="ho_10",
        name="Stefan-Boltzmann law",
        symbolic_expr=x0 * x1 * x2**4,
        variables=[x0, x1, x2],
        constants={"C0_sigma": STEFAN_BOLTZMANN},
        tier=5,
        held_out=True,
        prefix_notation="* * x0 x1 ^ x2 4",
        var_ranges=_default_range([x0, x1, x2]),
        description="P = sigma*A*T^4",
    ))

    # HO-11  RMS speed = sqrt(3*k_B*T / m)
    eqs.append(Equation(
        id="ho_11",
        name="RMS speed of gas molecules",
        symbolic_expr=sqrt(3 * x0 * x1 / x2),
        variables=[x0, x1, x2],
        constants={"C0_kB": BOLTZMANN_CONSTANT},
        tier=5,
        held_out=True,
        prefix_notation="sqrt / * * 3 x0 x1 x2",
        var_ranges=_custom_ranges(
            [x0, x1, x2],
            {"x2": (0.5, 10.0)},
        ),
        description="v_rms = sqrt(3*k_B*T/m)",
    ))

    return eqs


# ===================================================================
# Public API
# ===================================================================

def get_all_equations() -> List[Equation]:
    """Return every equation in the corpus (training + held-out)."""
    return (
        _tier1_equations()
        + _tier2_equations()
        + _tier3_equations()
        + _tier4_equations()
        + _tier5_training_equations()
        + _held_out_equations()
    )


def get_training_equations() -> List[Equation]:
    """Return only equations used for training (held_out=False)."""
    return [eq for eq in get_all_equations() if not eq.held_out]


def get_held_out_equations() -> List[Equation]:
    """Return only held-out equations (held_out=True), never seen during training."""
    return [eq for eq in get_all_equations() if eq.held_out]


def get_equations_by_tier(tier: int) -> List[Equation]:
    """Return all equations (training + held-out) belonging to a specific tier."""
    if tier not in {1, 2, 3, 4, 5}:
        raise ValueError(f"Tier must be 1-5, got {tier}")
    return [eq for eq in get_all_equations() if eq.tier == tier]


# ===================================================================
# CLI summary
# ===================================================================

def _print_summary() -> None:
    """Print a human-readable summary of the equation corpus."""
    all_eqs = get_all_equations()
    training = get_training_equations()
    held_out = get_held_out_equations()

    print("=" * 78)
    print("PhysMDT Equation Corpus Summary")
    print("=" * 78)
    print(f"  Total equations:    {len(all_eqs)}")
    print(f"  Training equations: {len(training)}")
    print(f"  Held-out equations: {len(held_out)}")
    print()

    for tier in range(1, 6):
        tier_eqs = get_equations_by_tier(tier)
        tier_train = [e for e in tier_eqs if not e.held_out]
        tier_held = [e for e in tier_eqs if e.held_out]
        print(f"--- Tier {tier}  ({len(tier_train)} training, {len(tier_held)} held-out) ---")
        for eq in tier_eqs:
            tag = "[HELD-OUT]" if eq.held_out else "[TRAIN]   "
            n_vars = len(eq.variables)
            n_const = len(eq.constants)
            print(
                f"  {tag}  {eq.id:8s}  {eq.name:44s}  "
                f"vars={n_vars}  consts={n_const}"
            )
            print(f"             expr: {eq.symbolic_expr}")
            print(f"             prefix: {eq.prefix_notation}")
        print()

    # Verify counts
    print("-" * 78)
    print("Verification:")
    print(f"  Tier 1: {len(get_equations_by_tier(1))} equations")
    print(f"  Tier 2: {len(get_equations_by_tier(2))} equations")
    print(f"  Tier 3: {len(get_equations_by_tier(3))} equations")
    print(f"  Tier 4: {len(get_equations_by_tier(4))} equations")
    print(f"  Tier 5: {len(get_equations_by_tier(5))} equations")
    print(f"  Training total:  {len(training)}")
    print(f"  Held-out total:  {len(held_out)}")
    print(f"  Grand total:     {len(all_eqs)}")
    print("=" * 78)


if __name__ == "__main__":
    _print_summary()
