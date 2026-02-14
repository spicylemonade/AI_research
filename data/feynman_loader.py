"""Loader for Feynman/SRSD benchmark equations (Newtonian mechanics subset)."""

import numpy as np
from typing import List, Dict, Tuple
from data.tokenizer import ExprNode, encode


# All 34 benchmark equations + 8 OOD equations defined as prefix token lists
# with their variable ranges and constant values

BENCHMARK_EQUATIONS = {
    # === Tier 1: Single-Variable Kinematics ===
    'T1.1': {
        'name': 'Distance (uniform velocity)',
        'formula': 's = v * t',
        'prefix': ['mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'v (m/s)', 'x_1': 't (s)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.01, 10.0)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [0, 1, -1], 'x_1': [0, 0, 1], 'y': [0, 1, 0]},
    },
    'T1.2': {
        'name': "Newton's 2nd Law",
        'formula': 'F = m * a',
        'prefix': ['mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'a (m/s^2)'},
        'var_ranges': {0: (0.1, 100.0), 1: (0.1, 20.0)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -2], 'y': [1, 1, -2]},
    },
    'T1.3': {
        'name': 'Momentum',
        'formula': 'p = m * v',
        'prefix': ['mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'v (m/s)'},
        'var_ranges': {0: (0.1, 100.0), 1: (0.1, 50.0)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'y': [1, 1, -1]},
    },
    'T1.4': {
        'name': 'Kinetic Energy',
        'formula': 'E_k = 0.5 * m * v^2',
        'prefix': ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'v (m/s)'},
        'var_ranges': {0: (0.1, 100.0), 1: (0.1, 50.0)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'y': [1, 2, -2]},
    },
    'T1.5': {
        'name': 'Gravitational PE (near surface)',
        'formula': 'U = m * g * h',
        'prefix': ['mul', 'mul', 'x_0', 'x_1', 'x_2'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'g (m/s^2)', 'x_2': 'h (m)'},
        'var_ranges': {0: (0.1, 100.0), 1: (9.7, 9.9), 2: (0.1, 100.0)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -2], 'x_2': [0, 1, 0], 'y': [1, 2, -2]},
    },
    'T1.6': {
        'name': 'Weight',
        'formula': 'W = m * g',
        'prefix': ['mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'g (m/s^2)'},
        'var_ranges': {0: (0.1, 100.0), 1: (9.7, 9.9)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -2], 'y': [1, 1, -2]},
    },
    'T1.7': {
        'name': 'Impulse',
        'formula': 'J = F * t',
        'prefix': ['mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'F (N)', 'x_1': 't (s)'},
        'var_ranges': {0: (0.1, 100.0), 1: (0.01, 10.0)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [1, 1, -2], 'x_1': [0, 0, 1], 'y': [1, 1, -1]},
    },
    'T1.8': {
        'name': 'Free-fall distance',
        'formula': 's = 0.5 * g * t^2',
        'prefix': ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],
        'variables': {'x_0': 'g (m/s^2)', 'x_1': 't (s)'},
        'var_ranges': {0: (9.7, 9.9), 1: (0.01, 10.0)},
        'constants': {},
        'tier': 1,
        'units': {'x_0': [0, 1, -2], 'x_1': [0, 0, 1], 'y': [0, 1, 0]},
    },

    # === Tier 2: Multi-Variable Dynamics ===
    'T2.1': {
        'name': 'Universal Gravitation',
        'formula': 'F = G * m1 * m2 / r^2',
        'prefix': ['div', 'mul', 'mul', 'c_0', 'x_0', 'x_1', 'pow', 'x_2', 'int_2'],
        'variables': {'x_0': 'm1 (kg)', 'x_1': 'm2 (kg)', 'x_2': 'r (m)'},
        'var_ranges': {0: (1.0, 100.0), 1: (1.0, 100.0), 2: (0.5, 50.0)},
        'constants': {'c_0': 6.674e-11},
        'tier': 2,
        'units': {'x_0': [1, 0, 0], 'x_1': [1, 0, 0], 'x_2': [0, 1, 0], 'y': [1, 1, -2]},
    },
    'T2.2': {
        'name': "Hooke's Law",
        'formula': 'F = -k * x',
        'prefix': ['neg', 'mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'k (N/m)', 'x_1': 'x (m)'},
        'var_ranges': {0: (1.0, 100.0), 1: (-5.0, 5.0)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [1, 0, -2], 'x_1': [0, 1, 0], 'y': [1, 1, -2]},
    },
    'T2.3': {
        'name': 'Pendulum Period',
        'formula': 'T = 2*pi*sqrt(L/g)',
        'prefix': ['mul', 'mul', 'int_2', 'pi', 'sqrt', 'div', 'x_0', 'x_1'],
        'variables': {'x_0': 'L (m)', 'x_1': 'g (m/s^2)'},
        'var_ranges': {0: (0.1, 10.0), 1: (9.7, 9.9)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [0, 1, 0], 'x_1': [0, 1, -2], 'y': [0, 0, 1]},
    },
    'T2.4': {
        'name': 'Projectile Range',
        'formula': 'R = v^2 * sin(2*theta) / g',
        'prefix': ['div', 'mul', 'pow', 'x_0', 'int_2', 'sin', 'mul', 'int_2', 'x_1', 'x_2'],
        'variables': {'x_0': 'v (m/s)', 'x_1': 'theta (rad)', 'x_2': 'g (m/s^2)'},
        'var_ranges': {0: (1.0, 50.0), 1: (0.1, 1.5), 2: (9.7, 9.9)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [0, 1, -1], 'x_1': [0, 0, 0], 'x_2': [0, 1, -2], 'y': [0, 1, 0]},
    },
    'T2.5': {
        'name': 'Centripetal Force',
        'formula': 'F = m * v^2 / r',
        'prefix': ['div', 'mul', 'x_0', 'pow', 'x_1', 'int_2', 'x_2'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'v (m/s)', 'x_2': 'r (m)'},
        'var_ranges': {0: (0.1, 100.0), 1: (0.1, 50.0), 2: (0.1, 50.0)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'x_2': [0, 1, 0], 'y': [1, 1, -2]},
    },
    'T2.6': {
        'name': 'Gravitational PE',
        'formula': 'U = -G * m1 * m2 / r',
        'prefix': ['neg', 'div', 'mul', 'mul', 'c_0', 'x_0', 'x_1', 'x_2'],
        'variables': {'x_0': 'm1 (kg)', 'x_1': 'm2 (kg)', 'x_2': 'r (m)'},
        'var_ranges': {0: (1.0, 100.0), 1: (1.0, 100.0), 2: (0.5, 50.0)},
        'constants': {'c_0': 6.674e-11},
        'tier': 2,
        'units': {'x_0': [1, 0, 0], 'x_1': [1, 0, 0], 'x_2': [0, 1, 0], 'y': [1, 2, -2]},
    },
    'T2.7': {
        'name': 'Spring PE',
        'formula': 'U = 0.5 * k * x^2',
        'prefix': ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],
        'variables': {'x_0': 'k (N/m)', 'x_1': 'x (m)'},
        'var_ranges': {0: (1.0, 100.0), 1: (-5.0, 5.0)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [1, 0, -2], 'x_1': [0, 1, 0], 'y': [1, 2, -2]},
    },
    'T2.8': {
        'name': 'Velocity under uniform accel',
        'formula': 'v = v0 + a * t',
        'prefix': ['add', 'x_0', 'mul', 'x_1', 'x_2'],
        'variables': {'x_0': 'v0 (m/s)', 'x_1': 'a (m/s^2)', 'x_2': 't (s)'},
        'var_ranges': {0: (0.0, 30.0), 1: (0.1, 10.0), 2: (0.01, 10.0)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [0, 1, -1], 'x_1': [0, 1, -2], 'x_2': [0, 0, 1], 'y': [0, 1, -1]},
    },
    'T2.9': {
        'name': 'Position under uniform accel',
        'formula': 's = v0*t + 0.5*a*t^2',
        'prefix': ['add', 'mul', 'x_0', 'x_2', 'mul', 'half', 'mul', 'x_1', 'pow', 'x_2', 'int_2'],
        'variables': {'x_0': 'v0 (m/s)', 'x_1': 'a (m/s^2)', 'x_2': 't (s)'},
        'var_ranges': {0: (0.0, 30.0), 1: (0.1, 10.0), 2: (0.01, 10.0)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [0, 1, -1], 'x_1': [0, 1, -2], 'x_2': [0, 0, 1], 'y': [0, 1, 0]},
    },
    'T2.10': {
        'name': 'Torque',
        'formula': 'tau = r * F * sin(theta)',
        'prefix': ['mul', 'mul', 'x_0', 'x_1', 'sin', 'x_2'],
        'variables': {'x_0': 'r (m)', 'x_1': 'F (N)', 'x_2': 'theta (rad)'},
        'var_ranges': {0: (0.1, 10.0), 1: (0.1, 100.0), 2: (0.0, np.pi)},
        'constants': {},
        'tier': 2,
        'units': {'x_0': [0, 1, 0], 'x_1': [1, 1, -2], 'x_2': [0, 0, 0], 'y': [1, 2, -2]},
    },

    # === Tier 3: Energy/Momentum Conservation ===
    'T3.1': {
        'name': 'Total Mechanical Energy',
        'formula': 'E = 0.5*m*v^2 + m*g*h',
        'prefix': ['add', 'mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2', 'mul', 'mul', 'x_0', 'x_2', 'x_3'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'v (m/s)', 'x_2': 'g (m/s^2)', 'x_3': 'h (m)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.1, 30.0), 2: (9.7, 9.9), 3: (0.1, 50.0)},
        'constants': {},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'x_2': [0, 1, -2], 'x_3': [0, 1, 0], 'y': [1, 2, -2]},
    },
    'T3.2': {
        'name': 'Elastic Collision v1 final',
        'formula': "v1' = (m1-m2)/(m1+m2) * v1",
        'prefix': ['mul', 'div', 'sub', 'x_0', 'x_1', 'add', 'x_0', 'x_1', 'x_2'],
        'variables': {'x_0': 'm1 (kg)', 'x_1': 'm2 (kg)', 'x_2': 'v1 (m/s)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.1, 50.0), 2: (0.1, 30.0)},
        'constants': {},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [1, 0, 0], 'x_2': [0, 1, -1], 'y': [0, 1, -1]},
    },
    'T3.3': {
        'name': 'Orbital Velocity',
        'formula': 'v = sqrt(G*M/r)',
        'prefix': ['sqrt', 'div', 'mul', 'c_0', 'x_0', 'x_1'],
        'variables': {'x_0': 'M (kg)', 'x_1': 'r (m)'},
        'var_ranges': {0: (1.0, 1e6), 1: (1.0, 1e4)},
        'constants': {'c_0': 6.674e-11},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, 0], 'y': [0, 1, -1]},
    },
    'T3.4': {
        'name': 'Escape Velocity',
        'formula': 'v_esc = sqrt(2*G*M/r)',
        'prefix': ['sqrt', 'div', 'mul', 'int_2', 'mul', 'c_0', 'x_0', 'x_1'],
        'variables': {'x_0': 'M (kg)', 'x_1': 'r (m)'},
        'var_ranges': {0: (1.0, 1e6), 1: (1.0, 1e4)},
        'constants': {'c_0': 6.674e-11},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, 0], 'y': [0, 1, -1]},
    },
    'T3.5': {
        'name': 'Reduced Mass',
        'formula': 'mu = m1*m2/(m1+m2)',
        'prefix': ['div', 'mul', 'x_0', 'x_1', 'add', 'x_0', 'x_1'],
        'variables': {'x_0': 'm1 (kg)', 'x_1': 'm2 (kg)'},
        'var_ranges': {0: (0.1, 100.0), 1: (0.1, 100.0)},
        'constants': {},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [1, 0, 0], 'y': [1, 0, 0]},
    },
    'T3.6': {
        'name': 'Relativistic KE approx',
        'formula': 'E = 0.5*m*v^2 + (3/8)*m*v^4/c^2',
        'prefix': ['add', 'mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2',
                    'div', 'mul', 'mul', 'c_0', 'x_0', 'pow', 'x_1', 'int_4', 'pow', 'x_2', 'int_2'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'v (m/s)', 'x_2': 'c (m/s)'},
        'var_ranges': {0: (0.1, 10.0), 1: (0.1, 1000.0), 2: (2.99e8, 3.01e8)},
        'constants': {'c_0': 0.375},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'x_2': [0, 1, -1], 'y': [1, 2, -2]},
    },
    'T3.7': {
        'name': 'Work-Energy Theorem',
        'formula': 'W = 0.5*m*(v2^2 - v1^2)',
        'prefix': ['mul', 'half', 'mul', 'x_0', 'sub', 'pow', 'x_1', 'int_2', 'pow', 'x_2', 'int_2'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'v2 (m/s)', 'x_2': 'v1 (m/s)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.1, 30.0), 2: (0.1, 30.0)},
        'constants': {},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'x_2': [0, 1, -1], 'y': [1, 2, -2]},
    },
    'T3.8': {
        'name': 'Gravitational Binding Energy',
        'formula': 'U = -3*G*M^2/(5*r)',
        'prefix': ['neg', 'div', 'mul', 'int_3', 'mul', 'c_0', 'pow', 'x_0', 'int_2', 'mul', 'int_5', 'x_1'],
        'variables': {'x_0': 'M (kg)', 'x_1': 'r (m)'},
        'var_ranges': {0: (1.0, 1e6), 1: (1.0, 1e4)},
        'constants': {'c_0': 6.674e-11},
        'tier': 3,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, 0], 'y': [1, 2, -2]},
    },

    # === Tier 4: Coupled Multi-Body & Transcendental ===
    'T4.1': {
        'name': "Kepler's 3rd Law",
        'formula': 'T^2 = 4*pi^2*a^3/(G*M)',
        'prefix': ['div', 'mul', 'int_4', 'mul', 'pow', 'pi', 'int_2', 'pow', 'x_0', 'int_3', 'mul', 'c_0', 'x_1'],
        'variables': {'x_0': 'a (m)', 'x_1': 'M (kg)'},
        'var_ranges': {0: (1.0, 1e6), 1: (1.0, 1e6)},
        'constants': {'c_0': 6.674e-11},
        'tier': 4,
        'units': {'x_0': [0, 1, 0], 'x_1': [1, 0, 0], 'y': [0, 0, 2]},
    },
    'T4.2': {
        'name': 'Damped Oscillator',
        'formula': 'x = A*exp(-b*t/(2*m))*cos(w_d*t)',
        'prefix': ['mul', 'mul', 'x_0', 'exp', 'neg', 'div', 'mul', 'x_1', 'x_2', 'mul', 'int_2', 'x_3',
                    'cos', 'mul', 'x_4', 'x_2'],
        'variables': {'x_0': 'A (m)', 'x_1': 'b (kg/s)', 'x_2': 't (s)', 'x_3': 'm (kg)', 'x_4': 'w_d (rad/s)'},
        'var_ranges': {0: (0.1, 5.0), 1: (0.01, 2.0), 2: (0.0, 10.0), 3: (0.1, 10.0), 4: (0.5, 10.0)},
        'constants': {},
        'tier': 4,
        'units': {'x_0': [0, 1, 0], 'x_1': [1, 0, -1], 'x_2': [0, 0, 1], 'x_3': [1, 0, 0], 'x_4': [0, 0, -1], 'y': [0, 1, 0]},
    },
    'T4.3': {
        'name': 'Rocket Equation (Tsiolkovsky)',
        'formula': 'dv = v_e * ln(m0/mf)',
        'prefix': ['mul', 'x_0', 'log', 'div', 'x_1', 'x_2'],
        'variables': {'x_0': 'v_e (m/s)', 'x_1': 'm0 (kg)', 'x_2': 'mf (kg)'},
        'var_ranges': {0: (100.0, 5000.0), 1: (10.0, 1000.0), 2: (1.0, 500.0)},
        'constants': {},
        'tier': 4,
        'units': {'x_0': [0, 1, -1], 'x_1': [1, 0, 0], 'x_2': [1, 0, 0], 'y': [0, 1, -1]},
    },
    'T4.4': {
        'name': 'Two-body Energy',
        'formula': 'E = 0.5*mu*v^2 - G*m1*m2/r',
        'prefix': ['sub', 'mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2',
                    'div', 'mul', 'mul', 'c_0', 'x_2', 'x_3', 'x_4'],
        'variables': {'x_0': 'mu (kg)', 'x_1': 'v (m/s)', 'x_2': 'm1 (kg)', 'x_3': 'm2 (kg)', 'x_4': 'r (m)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.1, 30.0), 2: (1.0, 100.0), 3: (1.0, 100.0), 4: (0.5, 50.0)},
        'constants': {'c_0': 6.674e-11},
        'tier': 4,
        'units': {'x_0': [1, 0, 0], 'x_1': [0, 1, -1], 'x_2': [1, 0, 0], 'x_3': [1, 0, 0], 'x_4': [0, 1, 0], 'y': [1, 2, -2]},
    },
    'T4.5': {
        'name': 'Forced Oscillator Amplitude',
        'formula': 'A = F0/sqrt((k-m*w^2)^2 + (b*w)^2)',
        'prefix': ['div', 'x_0', 'sqrt', 'add', 'pow', 'sub', 'x_1', 'mul', 'x_2', 'pow', 'x_3', 'int_2',
                    'int_2', 'pow', 'mul', 'x_4', 'x_3', 'int_2'],
        'variables': {'x_0': 'F0 (N)', 'x_1': 'k (N/m)', 'x_2': 'm (kg)', 'x_3': 'w (rad/s)', 'x_4': 'b (kg/s)'},
        'var_ranges': {0: (0.1, 50.0), 1: (1.0, 100.0), 2: (0.1, 10.0), 3: (0.1, 20.0), 4: (0.01, 5.0)},
        'constants': {},
        'tier': 4,
        'units': {'x_0': [1, 1, -2], 'x_1': [1, 0, -2], 'x_2': [1, 0, 0], 'x_3': [0, 0, -1], 'x_4': [1, 0, -1], 'y': [0, 1, 0]},
    },
    'T4.6': {
        'name': 'Orbit Equation (conic)',
        'formula': 'r = a*(1-e^2)/(1+e*cos(theta))',
        'prefix': ['div', 'mul', 'x_0', 'sub', 'int_1', 'pow', 'x_1', 'int_2',
                    'add', 'int_1', 'mul', 'x_1', 'cos', 'x_2'],
        'variables': {'x_0': 'a (m)', 'x_1': 'e (dimensionless)', 'x_2': 'theta (rad)'},
        'var_ranges': {0: (1.0, 1e6), 1: (0.01, 0.99), 2: (0.0, 2*np.pi)},
        'constants': {},
        'tier': 4,
        'units': {'x_0': [0, 1, 0], 'x_1': [0, 0, 0], 'x_2': [0, 0, 0], 'y': [0, 1, 0]},
    },
}

# OOD equations for generalization testing
OOD_EQUATIONS = {
    'OOD.1': {
        'name': 'Rotational KE',
        'formula': 'E = 0.5 * I * omega^2',
        'prefix': ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],
        'variables': {'x_0': 'I (kg*m^2)', 'x_1': 'omega (rad/s)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.1, 20.0)},
        'constants': {},
        'tier': 'ood',
    },
    'OOD.2': {
        'name': 'Angular Momentum',
        'formula': 'L = I * omega',
        'prefix': ['mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'I (kg*m^2)', 'x_1': 'omega (rad/s)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.1, 20.0)},
        'constants': {},
        'tier': 'ood',
    },
    'OOD.3': {
        'name': 'Fluid Pressure',
        'formula': 'P = rho * g * h',
        'prefix': ['mul', 'mul', 'x_0', 'x_1', 'x_2'],
        'variables': {'x_0': 'rho (kg/m^3)', 'x_1': 'g (m/s^2)', 'x_2': 'h (m)'},
        'var_ranges': {0: (100.0, 2000.0), 1: (9.7, 9.9), 2: (0.1, 100.0)},
        'constants': {},
        'tier': 'ood',
    },
    'OOD.4': {
        'name': 'Wave Speed',
        'formula': 'v = f * lambda',
        'prefix': ['mul', 'x_0', 'x_1'],
        'variables': {'x_0': 'f (Hz)', 'x_1': 'lambda (m)'},
        'var_ranges': {0: (0.1, 1000.0), 1: (0.01, 100.0)},
        'constants': {},
        'tier': 'ood',
    },
    'OOD.5': {
        'name': 'Wave Equation',
        'formula': 'y = A * sin(k*x - omega*t)',
        'prefix': ['mul', 'x_0', 'sin', 'sub', 'mul', 'x_1', 'x_2', 'mul', 'x_3', 'x_4'],
        'variables': {'x_0': 'A (m)', 'x_1': 'k (1/m)', 'x_2': 'x (m)', 'x_3': 'omega (rad/s)', 'x_4': 't (s)'},
        'var_ranges': {0: (0.1, 5.0), 1: (0.1, 10.0), 2: (0.0, 10.0), 3: (0.1, 10.0), 4: (0.0, 10.0)},
        'constants': {},
        'tier': 'ood',
    },
    'OOD.6': {
        'name': 'Drag Force',
        'formula': 'F_d = 0.5 * C_d * rho * A * v^2',
        'prefix': ['mul', 'half', 'mul', 'mul', 'mul', 'x_0', 'x_1', 'x_2', 'pow', 'x_3', 'int_2'],
        'variables': {'x_0': 'C_d', 'x_1': 'rho (kg/m^3)', 'x_2': 'A (m^2)', 'x_3': 'v (m/s)'},
        'var_ranges': {0: (0.1, 2.0), 1: (0.5, 2.0), 2: (0.01, 10.0), 3: (0.1, 50.0)},
        'constants': {},
        'tier': 'ood',
    },
    'OOD.7': {
        'name': 'Moment of Inertia (rod)',
        'formula': 'I = (1/12) * m * L^2',
        'prefix': ['mul', 'c_0', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],
        'variables': {'x_0': 'm (kg)', 'x_1': 'L (m)'},
        'var_ranges': {0: (0.1, 50.0), 1: (0.1, 10.0)},
        'constants': {'c_0': 1.0/12.0},
        'tier': 'ood',
    },
    'OOD.8': {
        'name': 'Bernoulli simplified (dynamic pressure)',
        'formula': 'P_dyn = 0.5 * rho * v^2',
        'prefix': ['mul', 'half', 'mul', 'x_0', 'pow', 'x_1', 'int_2'],
        'variables': {'x_0': 'rho (kg/m^3)', 'x_1': 'v (m/s)'},
        'var_ranges': {0: (0.5, 2.0), 1: (0.1, 100.0)},
        'constants': {},
        'tier': 'ood',
    },
}


def generate_benchmark_data(n_points: int = 200, seed: int = 42) -> List[Dict]:
    """Generate observation tables for all benchmark equations."""
    from data.data_generator import tree_to_numpy_func
    from data.tokenizer import _parse_prefix

    rng = np.random.default_rng(seed)
    benchmark_data = []

    for eq_id, eq_info in BENCHMARK_EQUATIONS.items():
        # Parse prefix to tree
        tree, _ = _parse_prefix(eq_info['prefix'], 0)

        # Generate observation table
        n_vars = len(eq_info['variables'])
        x_dict = {}
        x_cols = []
        for i in range(n_vars):
            lo, hi = eq_info['var_ranges'].get(i, (0.1, 10.0))
            x_i = rng.uniform(lo, hi, size=n_points)
            x_dict[i] = x_i
            x_cols.append(x_i)

        eval_func = tree_to_numpy_func(tree, eq_info['constants'])
        try:
            y = eval_func(x_dict)
            y = np.array(y, dtype=np.float64)
        except Exception as e:
            print(f"Error evaluating {eq_id}: {e}")
            continue

        valid = np.isfinite(y)
        if valid.sum() < n_points * 0.5:
            print(f"Warning: {eq_id} has {valid.sum()}/{n_points} valid points")
            continue

        table = np.column_stack(x_cols + [y])
        table = table[valid]

        if len(table) > n_points:
            idx = rng.choice(len(table), n_points, replace=False)
            table = table[idx]

        token_ids = encode(eq_info['prefix'])

        benchmark_data.append({
            'id': eq_id,
            'name': eq_info['name'],
            'formula': eq_info['formula'],
            'prefix': eq_info['prefix'],
            'token_ids': token_ids,
            'table': table.astype(np.float32),
            'tier': eq_info['tier'],
            'n_vars': n_vars,
            'constants': eq_info['constants'],
            'units': eq_info.get('units', {}),
        })

    return benchmark_data


def generate_ood_data(n_points: int = 200, seed: int = 42) -> List[Dict]:
    """Generate observation tables for OOD test equations."""
    from data.data_generator import tree_to_numpy_func
    from data.tokenizer import _parse_prefix

    rng = np.random.default_rng(seed)
    ood_data = []

    for eq_id, eq_info in OOD_EQUATIONS.items():
        tree, _ = _parse_prefix(eq_info['prefix'], 0)
        n_vars = len(eq_info['variables'])
        x_dict = {}
        x_cols = []
        for i in range(n_vars):
            lo, hi = eq_info['var_ranges'].get(i, (0.1, 10.0))
            x_i = rng.uniform(lo, hi, size=n_points)
            x_dict[i] = x_i
            x_cols.append(x_i)

        eval_func = tree_to_numpy_func(tree, eq_info.get('constants', {}))
        try:
            y = eval_func(x_dict)
            y = np.array(y, dtype=np.float64)
        except Exception as e:
            print(f"Error evaluating {eq_id}: {e}")
            continue

        valid = np.isfinite(y)
        table = np.column_stack(x_cols + [y])
        table = table[valid]

        if len(table) > n_points:
            idx = rng.choice(len(table), n_points, replace=False)
            table = table[idx]

        token_ids = encode(eq_info['prefix'])

        ood_data.append({
            'id': eq_id,
            'name': eq_info['name'],
            'formula': eq_info['formula'],
            'prefix': eq_info['prefix'],
            'token_ids': token_ids,
            'table': table.astype(np.float32),
            'tier': 'ood',
            'n_vars': n_vars,
            'constants': eq_info.get('constants', {}),
        })

    return ood_data


if __name__ == '__main__':
    print("Generating benchmark data...")
    data = generate_benchmark_data()
    print(f"Generated {len(data)} benchmark equations")
    for d in data[:3]:
        print(f"  {d['id']}: {d['name']} - {d['formula']} (table: {d['table'].shape})")

    print("\nGenerating OOD data...")
    ood = generate_ood_data()
    print(f"Generated {len(ood)} OOD equations")
