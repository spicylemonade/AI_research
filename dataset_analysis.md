# Dataset Analysis: Newtonian Mechanics Equations for PhysDiffuse

## Source Datasets
- **Feynman Symbolic Regression Database (FSReD):** 100 equations from the Feynman Lectures on Physics (Udrescu & Tegmark, 2020)
- **SRSD Benchmark:** 120 recreated Feynman datasets with realistic sampling ranges (Matsubara et al., 2022)
- **Additional canonical equations:** Standard Newtonian mechanics formulas from physics textbooks

## Equation Selection Criteria
1. Domain: Classical (Newtonian) mechanics — kinematics, dynamics, energy, momentum, gravitation, oscillations
2. Variables: Real-valued physical quantities with well-defined SI units
3. Complexity: Graded from simple single-variable to complex multi-body/transcendental

---

## Tier 1: Single-Variable Kinematics (8 equations)
*Characteristics: 1-2 variables, 1-3 operators, nesting depth ≤ 2*

| ID | Equation | Formula | Variables | Operators | Depth | Source |
|----|----------|---------|-----------|-----------|-------|--------|
| T1.1 | Distance (uniform velocity) | `s = v * t` | v, t | 1 (mul) | 1 | Feynman I.6.20a |
| T1.2 | Newton's 2nd Law | `F = m * a` | m, a | 1 (mul) | 1 | Feynman I.12.1 |
| T1.3 | Momentum | `p = m * v` | m, v | 1 (mul) | 1 | Feynman I.10.7 |
| T1.4 | Kinetic Energy | `E_k = 0.5 * m * v^2` | m, v | 3 (mul, mul, pow) | 2 | Feynman I.13.12 |
| T1.5 | Gravitational PE | `U = m * g * h` | m, g, h | 2 (mul, mul) | 2 | Feynman I.13.4 |
| T1.6 | Weight | `W = m * g` | m, g | 1 (mul) | 1 | Standard |
| T1.7 | Impulse | `J = F * t` | F, t | 1 (mul) | 1 | Standard |
| T1.8 | Free-fall distance | `s = 0.5 * g * t^2` | g, t | 3 (mul, mul, pow) | 2 | Feynman I.6.20b |

---

## Tier 2: Multi-Variable Dynamics (10 equations)
*Characteristics: 2-4 variables, 3-6 operators, nesting depth ≤ 3*

| ID | Equation | Formula | Variables | Operators | Depth | Source |
|----|----------|---------|-----------|-----------|-------|--------|
| T2.1 | Universal Gravitation | `F = G*m1*m2/r^2` | G, m1, m2, r | 4 (mul, mul, div, pow) | 3 | Feynman I.12.2 |
| T2.2 | Hooke's Law (spring) | `F = -k * x` | k, x | 2 (neg, mul) | 2 | Feynman I.12.4 |
| T2.3 | Pendulum Period | `T = 2*pi*sqrt(L/g)` | L, g | 4 (mul, mul, sqrt, div) | 3 | Feynman I.15.10 |
| T2.4 | Projectile Range | `R = v^2*sin(2*theta)/g` | v, theta, g | 5 (div, mul, pow, sin, mul) | 3 | Standard |
| T2.5 | Centripetal Force | `F = m*v^2/r` | m, v, r | 3 (div, mul, pow) | 3 | Feynman I.16.6 |
| T2.6 | Gravitational PE | `U = -G*m1*m2/r` | G, m1, m2, r | 4 (neg, div, mul, mul) | 3 | Feynman I.12.11 |
| T2.7 | Spring PE | `U = 0.5*k*x^2` | k, x | 3 (mul, mul, pow) | 2 | Feynman I.13.8 |
| T2.8 | Velocity under uniform accel | `v = v0 + a*t` | v0, a, t | 2 (add, mul) | 2 | Feynman I.6.20 |
| T2.9 | Position under uniform accel | `s = v0*t + 0.5*a*t^2` | v0, a, t | 5 (add, mul, mul, mul, pow) | 3 | Feynman I.6.20 |
| T2.10 | Torque | `tau = r * F * sin(theta)` | r, F, theta | 3 (mul, mul, sin) | 2 | Standard |

---

## Tier 3: Energy/Momentum Conservation Laws (8 equations)
*Characteristics: 3-6 variables, 5-8 operators, nesting depth ≤ 4*

| ID | Equation | Formula | Variables | Operators | Depth | Source |
|----|----------|---------|-----------|-----------|-------|--------|
| T3.1 | Total Mechanical Energy | `E = 0.5*m*v^2 + m*g*h` | m, v, g, h | 6 (add, mul, mul, pow, mul, mul) | 3 | Feynman I.13.12 |
| T3.2 | Elastic Collision (1D, v1') | `v1' = ((m1-m2)/(m1+m2))*v1` | m1, m2, v1 | 5 (mul, div, sub, add) | 3 | Standard |
| T3.3 | Orbital Velocity | `v = sqrt(G*M/r)` | G, M, r | 3 (sqrt, div, mul) | 3 | Feynman I.15.3t |
| T3.4 | Escape Velocity | `v_esc = sqrt(2*G*M/r)` | G, M, r | 4 (sqrt, div, mul, mul) | 3 | Standard |
| T3.5 | Reduced Mass | `mu = m1*m2/(m1+m2)` | m1, m2 | 3 (div, mul, add) | 2 | Feynman I.11.17 |
| T3.6 | Relativistic KE approx | `E = 0.5*m*v^2 + (3/8)*m*v^4/c^2` | m, v, c | 8 (add, mul, mul, pow, div, mul, pow, mul) | 4 | Feynman I.16.7 |
| T3.7 | Work-Energy Theorem | `W = 0.5*m*(v2^2 - v1^2)` | m, v1, v2 | 5 (mul, mul, sub, pow, pow) | 3 | Standard |
| T3.8 | Gravitational binding energy | `U = -3*G*M^2/(5*r)` | G, M, r | 6 (neg, div, mul, mul, pow, mul) | 4 | Feynman I.14.1 |

---

## Tier 4: Coupled Multi-Body and Transcendental Equations (8 equations)
*Characteristics: 4-9 variables, 7+ operators, nesting depth ≤ 6*

| ID | Equation | Formula | Variables | Operators | Depth | Source |
|----|----------|---------|-----------|-----------|-------|--------|
| T4.1 | Kepler's 3rd Law | `T^2 = 4*pi^2*a^3/(G*M)` | a, G, M | 6 (div, mul, mul, pow, pow, mul) | 4 | Feynman I.15.3 |
| T4.2 | Damped Oscillator | `x = A*exp(-b*t/(2*m))*cos(w_d*t)` | A, b, t, m, w_d | 7 (mul, mul, exp, neg, div, mul, cos, mul) | 5 | Feynman I.15.12 |
| T4.3 | Gravitational force (2-body, components) | `Fx = G*m1*m2*(x2-x1)/((x2-x1)^2+(y2-y1)^2)^(3/2)` | G,m1,m2,x1,x2,y1,y2 | 10+ | 6 | Standard |
| T4.4 | Rocket equation (Tsiolkovsky) | `dv = v_e * ln(m0/mf)` | v_e, m0, mf | 3 (mul, log, div) | 3 | Standard |
| T4.5 | Two-body energy | `E = 0.5*mu*v^2 - G*m1*m2/r` | mu, v, G, m1, m2, r | 7 (sub, mul, mul, pow, div, mul, mul) | 4 | Feynman I.12.12 |
| T4.6 | Forced oscillator amplitude | `A = F0/sqrt((k-m*w^2)^2+(b*w)^2)` | F0, k, m, w, b | 9 (div, sqrt, add, pow, sub, mul, pow, pow, mul) | 5 | Feynman I.15.14 |
| T4.7 | Orbit equation (conic) | `r = a*(1-e^2)/(1+e*cos(theta))` | a, e, theta | 7 (div, mul, sub, pow, add, mul, cos) | 4 | Feynman I.15.4 |
| T4.8 | Lagrangian (simple pendulum) | `L = 0.5*m*l^2*theta_dot^2 + m*g*l*cos(theta)` | m, l, theta_dot, g, theta | 9 (add, mul, mul, mul, pow, pow, mul, mul, cos) | 4 | Standard |

---

## Summary Statistics

| Tier | Count | Avg Variables | Avg Operators | Avg Depth | Description |
|------|-------|--------------|---------------|-----------|-------------|
| 1    | 8     | 1.8          | 1.5           | 1.4       | Single-variable kinematics |
| 2    | 10    | 2.8          | 3.3           | 2.5       | Multi-variable dynamics |
| 3    | 8     | 3.5          | 5.1           | 3.1       | Energy/momentum conservation |
| 4    | 8     | 5.0          | 7.4           | 4.4       | Coupled multi-body & transcendental |
| **Total** | **34** | **3.2** | **4.2** | **2.8** | **Full Newtonian benchmark** |

## Train/Test Split

- **Test set:** All 34 equations above (used for evaluation only)
- **Training set:** 500K+ procedurally generated equations from a physics-informed grammar, covering the same operator vocabulary and variable ranges but with different structural compositions
- **Validation set:** 10% holdout from training data for hyperparameter tuning

## Equation Representation

All equations are stored in prefix (Polish) notation using the token vocabulary defined in the problem statement. Constants are represented as learnable tokens (`c_0`, `c_1`, etc.) whose values are optimized via BFGS post-processing.

## Held-Out OOD Equations (for Generalization Testing, Phase 4)

These equations from adjacent physics domains are explicitly excluded from training:
1. Rotational KE: `E = 0.5 * I * omega^2`
2. Angular momentum: `L = I * omega`
3. Fluid pressure: `P = rho * g * h`
4. Bernoulli's equation: `P + 0.5*rho*v^2 + rho*g*h = const`
5. Wave speed: `v = f * lambda`
6. Wave equation: `y = A * sin(k*x - omega*t)`
7. Drag force: `F_d = 0.5 * C_d * rho * A * v^2`
8. Moment of inertia (rod): `I = (1/12) * m * L^2`
