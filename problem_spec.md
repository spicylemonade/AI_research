# Problem Specification: Physics Equation Discovery via Masked Diffusion Transformers

## 1. Problem Definition
Formal definition: Given N observation pairs {(x_i, y_i)} where x_i ∈ R^d and y_i ∈ R, discover the symbolic expression f such that y_i = f(x_i) + ε_i where ε is measurement noise.

## 2. Input Representation
- Numerical observations: matrix of shape (N, d+1) where N = number of observations, d = number of input variables
- Each row: [x_{i,1}, x_{i,2}, ..., x_{i,d}, y_i]
- Variable count d ranges from 1 to 5
- Observation count N ranges from 5 to 200
- Values normalized to [-10, 10] range with optional Gaussian noise (σ = 0%, 1%, 5%, 10%, 20%)
- Constants embedded as numeric tokens with fixed precision

## 3. Output Representation
- Symbolic expressions encoded as token sequences in prefix (Polish) notation
- Expression tree traversed in pre-order: operator first, then left subtree, then right subtree
- Example: F = m*a → prefix: "* x0 x1"
- Example: KE = 0.5*m*v^2 → prefix: "* 0.5 * x0 ^ x1 2"
- Example: F = G*m1*m2/r^2 → prefix: "/ * * C0 x0 x1 ^ x2 2"

## 4. Token Vocabulary
Operators: +, -, *, /, ^, sqrt, sin, cos, tan, log, exp, neg
Constants: pi, integers -10..10, float tokens, C0..C4 (learned constants)
Variables: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9
Special: SOS, EOS, PAD, MASK, SEP

## 5. Equation Corpus (55 equations across 5 tiers)

### Tier 1: Single-variable linear/simple (12 equations) — TRAINING
1. F = m*a (Newton's 2nd law)
2. v = v0 + a*t (velocity under constant acceleration)
3. p = m*v (momentum)
4. W = F*d (work)
5. P = W/t (power)
6. I = F*t (impulse)
7. ρ = m/V (density)
8. P = F/A (pressure)
9. v = d/t (velocity)
10. a = F/m (acceleration)
11. τ = F*r (torque, simplified)
12. f = 1/T (frequency)

### Tier 2: Multi-variable polynomial (12 equations) — TRAINING
13. KE = 0.5*m*v^2 (kinetic energy)
14. PE = m*g*h (gravitational PE)
15. s = v0*t + 0.5*a*t^2 (displacement)
16. v^2 = v0^2 + 2*a*s (velocity-displacement)
17. E = KE + PE = 0.5*m*v^2 + m*g*h (mechanical energy)
18. F_spring = k*x (Hooke's law)
19. PE_spring = 0.5*k*x^2 (spring PE)
20. Q = m*c*ΔT (heat energy)
21. L = I*ω (angular momentum)
22. KE_rot = 0.5*I*ω^2 (rotational KE)
23. F_friction = μ*N (friction force)
24. v_cm = (m1*v1 + m2*v2)/(m1 + m2) (center of mass velocity)

### Tier 3: Inverse-square/rational (12 equations) — TRAINING
25. F_grav = G*m1*m2/r^2 (gravitational force)
26. F_coulomb = k*q1*q2/r^2 (Coulomb's law)
27. g_surface = G*M/R^2 (surface gravity)
28. v_escape = sqrt(2*G*M/R) (escape velocity)
29. PE_grav = -G*m1*m2/r (gravitational PE)
30. a_centripetal = v^2/r (centripetal acceleration)
31. F_centripetal = m*v^2/r (centripetal force)
32. v_orbital = sqrt(G*M/r) (orbital velocity)
33. T_orbital = 2*pi*sqrt(r^3/(G*M)) (orbital period)
34. I_sphere = 2/5*m*r^2 (moment of inertia, solid sphere)
35. F_drag = 0.5*ρ*v^2*C*A (drag force)
36. v_terminal = sqrt(2*m*g/(ρ*C*A)) (terminal velocity)

### Tier 4: Compositions with trig/sqrt (10 equations) — TRAINING
37. T_pendulum = 2*pi*sqrt(L/g) (pendulum period)
38. R_projectile = v0^2*sin(2*θ)/g (projectile range)
39. y_projectile = v0*sin(θ)*t - 0.5*g*t^2 (projectile height)
40. x_projectile = v0*cos(θ)*t (projectile horizontal)
41. F_incline = m*g*sin(θ) (force on incline)
42. N_incline = m*g*cos(θ) (normal force on incline)
43. v_wave = sqrt(T_tension/μ) (wave speed on string)
44. ω = sqrt(k/m) (angular frequency, spring-mass)
45. x_shm = A*sin(ω*t + φ) (simple harmonic motion)
46. E_shm = 0.5*k*A^2 (total energy SHM)

### Tier 5: Multi-step derivations (9 equations) — TRAINING (4) + HELD-OUT (5)
**Training:**
47. T_kepler = 2*pi*sqrt(a^3/(G*M)) (Kepler's 3rd law)
48. v_rocket = v_e*log(m0/mf) (Tsiolkovsky rocket equation)
49. E_orbit = -G*M*m/(2*a) (orbital energy)
50. R_schwarzschild = 2*G*M/c^2 (Schwarzschild radius, simplified)

## 6. Held-Out Equations for Zero-Shot Discovery (11 equations, NEVER in training)

These equations span Tiers 3-5 and are NEVER seen during training. The model must discover them from numerical observations alone.

**Held-out Tier 3:**
H1. F_magnetic = q*v*B (Lorentz force, simplified) — prefix: * * x0 x1 x2
H2. P_wave = 0.5*ρ*v*ω^2*A^2 (wave power) — prefix: * 0.5 * * * x0 x1 ^ x2 2 ^ x3 2

**Held-out Tier 4:**
H3. v_doppler = v_sound * f / (v_sound - v_source) → prefix: / * x0 x1 - x0 x2
H4. θ_deflection = 4*G*M/(c^2*r) (gravitational lensing) — prefix: / * 4 * C0 x0 * ^ x1 2 x2
H5. T_lc = 2*pi*sqrt(L*C) (LC circuit period) — prefix: * * 2 pi sqrt * x0 x1
H6. a_coriolis = 2*ω*v*sin(φ) (Coriolis acceleration) — prefix: * * 2 * x0 x1 sin x2

**Held-out Tier 5:**
H7. F_tidal = 2*G*M*m*r/d^3 (tidal force) — prefix: / * * * 2 * C0 x0 * x1 x2 ^ x3 3
H8. Δt = t0/sqrt(1 - v^2/c^2) (time dilation, simplified) — prefix: / x0 sqrt - 1 / ^ x1 2 ^ C0 2
H9. λ_deBroglie = h/(m*v) (de Broglie wavelength) — prefix: / C0 * x0 x1
H10. P_radiation = σ*A*T^4 (Stefan-Boltzmann law) — prefix: * * C0 x0 ^ x1 4
H11. v_rms = sqrt(3*k*T/m) (RMS speed of gas) — prefix: sqrt / * 3 * C0 x0 x1

## 7. Evaluation Protocol
- **In-distribution**: Evaluate on training equations with new random variable instantiations
- **Zero-shot discovery**: Evaluate on held-out equations with only numerical observations
- **Metrics**: Symbolic equivalence accuracy, numeric R², token edit distance, complexity-weighted score, novel discovery rate

## 8. Data Generation Parameters
- Training set: ~500K samples (balanced across tiers with emphasis on harder tiers)
- Validation set: 10K samples
- Test set: 10K samples (in-distribution) + held-out set
- Variable ranges: uniform random from [0.1, 10.0] (avoiding singularities)
- Constants: sampled from predefined ranges per equation
- Noise levels: 0%, 1%, 5%, 10% (training), 0-20% (evaluation)
- Observation count: 50 points per sample (default), varied during robustness evaluation
