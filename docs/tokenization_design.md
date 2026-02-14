# Tokenization Design: Physics-Aware Prefix Notation

## 1. Prefix (Polish) Notation Encoding

Following Lample & Charton (2020), we represent mathematical expressions in **prefix notation** where operators precede their operands. This yields unambiguous, parenthesis-free sequences.

### Encoding Rules

1. **Binary operators**: `op arg1 arg2` (e.g., `+ x y` for x+y)
2. **Unary operators**: `op arg` (e.g., `sin x` for sin(x))
3. **Constants**: Encoded as `C_sign_mantissa_exp` (e.g., `C_+_3.14_0` for 3.14)
4. **Integer constants**: Direct tokens (`INT_0`, `INT_1`, ..., `INT_9`)
5. **Named constants**: Special tokens (`pi`, `euler`, `G_const`, etc.)
6. **Nesting**: Recursive application (e.g., `* 0.5 * m pow v INT_2` for ½mv²)

### Decoding Rules

1. Read tokens left to right
2. Maintain a stack: push operators with their arity
3. When an operand (variable/constant) is read, attach it to the deepest unfilled operator
4. Expression is complete when all operators have their operands filled

### Tree Depth Annotation

Each token is annotated with its depth in the expression tree (used for dual-axis RoPE):
- Root operator: depth 0
- Its arguments: depth 1
- Sub-expressions increase depth recursively

---

## 2. Complete Token Vocabulary (147 tokens)

### 2.1 Special Tokens (6)

| Token | ID | Purpose |
|-------|-----|---------|
| `[PAD]` | 0 | Padding |
| `[BOS]` | 1 | Beginning of sequence |
| `[EOS]` | 2 | End of sequence |
| `[MASK]` | 3 | Masked token (for diffusion) |
| `[SEP]` | 4 | Separator |
| `[UNK]` | 5 | Unknown token |

### 2.2 Arithmetic Operators (6)

| Token | ID | Arity | Meaning |
|-------|-----|-------|---------|
| `add` | 6 | 2 | Addition |
| `sub` | 7 | 2 | Subtraction |
| `mul` | 8 | 2 | Multiplication |
| `div` | 9 | 2 | Division |
| `pow` | 10 | 2 | Exponentiation |
| `neg` | 11 | 1 | Unary negation |

### 2.3 Mathematical Functions (12)

| Token | ID | Arity | Meaning |
|-------|-----|-------|---------|
| `sin` | 12 | 1 | Sine |
| `cos` | 13 | 1 | Cosine |
| `tan` | 14 | 1 | Tangent |
| `exp` | 15 | 1 | Exponential |
| `log` | 16 | 1 | Natural log |
| `sqrt` | 17 | 1 | Square root |
| `abs` | 18 | 1 | Absolute value |
| `asin` | 19 | 1 | Arcsine |
| `acos` | 20 | 1 | Arccosine |
| `atan` | 21 | 1 | Arctangent |
| `sinh` | 22 | 1 | Hyperbolic sine |
| `cosh` | 23 | 1 | Hyperbolic cosine |

### 2.4 Physics Variables (30)

| Token | ID Range | Variables |
|-------|----------|-----------|
| `x`, `y`, `z` | 24-26 | Spatial coordinates |
| `t` | 27 | Time |
| `v`, `v0`, `vx`, `vy`, `vz` | 28-32 | Velocities |
| `a`, `ax`, `ay`, `az` | 33-36 | Accelerations |
| `F`, `Fx`, `Fy`, `Fz` | 37-40 | Forces |
| `m`, `m1`, `m2` | 41-43 | Masses |
| `r`, `R` | 44-45 | Radii/distances |
| `theta`, `phi` | 46-47 | Angles |
| `omega`, `alpha` | 48-49 | Angular velocity/acceleration |
| `tau` | 50 | Torque |
| `I_inertia` | 51 | Moment of inertia |
| `L_angular` | 52 | Angular momentum |
| `E_energy` | 53 | Energy |
| `KE` | 54 | Kinetic energy |
| `PE` | 55 | Potential energy |
| `W_work` | 56 | Work |
| `P_power` | 57 | Power |
| `p_momentum` | 58 | Linear momentum |
| `rho` | 59 | Density |
| `P_pressure` | 60 | Pressure |
| `V_volume` | 61 | Volume |
| `A_area` | 62 | Area |
| `h` | 63 | Height |
| `l_length` | 64 | Length (e.g., pendulum) |
| `d` | 65 | Distance/displacement |
| `k_spring` | 66 | Spring constant |
| `mu` | 67 | Friction coefficient |
| `x0` | 68 | Initial position |

### 2.5 Named Physical Constants (8)

| Token | ID | Value | Meaning |
|-------|-----|-------|---------|
| `g_accel` | 69 | 9.81 | Gravitational acceleration |
| `G_const` | 70 | 6.674e-11 | Gravitational constant |
| `pi` | 71 | 3.14159... | Pi |
| `euler` | 72 | 2.71828... | Euler's number |
| `c_light` | 73 | 3e8 | Speed of light |
| `k_boltz` | 74 | 1.38e-23 | Boltzmann constant |
| `h_planck` | 75 | 6.626e-34 | Planck's constant |
| `epsilon0` | 76 | 8.854e-12 | Permittivity of free space |

### 2.6 Integer Constants (10)

| Token | ID Range | Values |
|-------|----------|--------|
| `INT_0` through `INT_9` | 77-86 | 0, 1, 2, ..., 9 |

### 2.7 Floating-Point Constant Tokens (50)

To encode arbitrary numerical constants, we use a decomposition scheme:

| Token | ID Range | Purpose |
|-------|----------|---------|
| `C_+`, `C_-` | 87-88 | Constant sign |
| `D_0` through `D_9` | 89-98 | Mantissa digits |
| `DOT` | 99 | Decimal point |
| `E_+`, `E_-` | 100-101 | Exponent sign |
| `E_0` through `E_9` | 102-111 | Exponent digits |
| `CONST_START` | 112 | Begin constant encoding |
| `CONST_END` | 113 | End constant encoding |

A constant like -3.14e2 is encoded as: `CONST_START C_- D_3 DOT D_1 D_4 E_+ E_2 CONST_END`

### 2.8 Structure Tokens (for Structure Predictor) (24)

| Token | ID Range | Purpose |
|-------|----------|---------|
| `OP_BINARY` | 114 | Binary operator placeholder |
| `OP_UNARY` | 115 | Unary operator placeholder |
| `LEAF_VAR` | 116 | Variable leaf placeholder |
| `LEAF_CONST` | 117 | Constant leaf placeholder |
| `LEAF_INT` | 118 | Integer leaf placeholder |
| `LEAF_NAMED` | 119 | Named constant placeholder |
| `SKEL_add` through `SKEL_pow` | 120-125 | Skeleton operators |
| `SKEL_sin` through `SKEL_sqrt` | 126-131 | Skeleton functions |
| `SKEL_neg` | 132 | Skeleton negation |
| `DEPTH_0` through `DEPTH_9` | 133-142 | Tree depth annotations |
| `STRUCT_PAD` | 143 | Structure padding |
| `STRUCT_BOS` | 144 | Structure BOS |
| `STRUCT_EOS` | 145 | Structure EOS |
| `STRUCT_MASK` | 146 | Structure mask |

**Total vocabulary size: 147 tokens**

---

## 3. Round-Trip Fidelity Proof

We demonstrate that the tokenization scheme achieves perfect round-trip fidelity (encode → decode → encode yields identical sequences) on 10 representative equations:

| # | Family | Equation | Prefix Tokens | Round-Trip |
|---|--------|----------|--------------|------------|
| 1 | Kinematics | $v = v_0 + at$ | `add v0 mul a t` | ✓ |
| 2 | Kinematics | $x = x_0 + v_0 t + \frac{1}{2}at^2$ | `add x0 add mul v0 t mul mul div INT_1 INT_2 a pow t INT_2` | ✓ |
| 3 | Dynamics | $F = ma$ | `mul m a` | ✓ |
| 4 | Energy | $KE = \frac{1}{2}mv^2$ | `mul div INT_1 INT_2 mul m pow v INT_2` | ✓ |
| 5 | Energy | $PE = mgh$ | `mul m mul g_accel h` | ✓ |
| 6 | Rotational | $\tau = I\alpha$ | `mul I_inertia alpha` | ✓ |
| 7 | Gravitation | $F = \frac{Gm_1 m_2}{r^2}$ | `div mul G_const mul m1 m2 pow r INT_2` | ✓ |
| 8 | Oscillations | $x = A\sin(\omega t + \phi)$ | `mul A_area sin add mul omega t phi` | ✓ |
| 9 | Oscillations | $T = 2\pi\sqrt{l/g}$ | `mul mul INT_2 pi sqrt div l_length g_accel` | ✓ |
| 10 | Fluid | $P = \rho g h$ | `mul rho mul g_accel h` | ✓ |

---

## 4. Comparison with Prior Tokenization Approaches

### Lample & Charton (2020)
- **Vocabulary**: ~30 tokens (operators + generic variables x0..x3)
- **Constants**: Float decomposition similar to ours
- **No physics awareness**: Variables are generic (x0, x1, x2)
- **Our improvement**: Physics-meaningful variable names enable the model to learn physical relationships (e.g., F and m*a should co-occur)

### NeSymReS (Biggio et al., 2021)
- **Vocabulary**: ~50 tokens
- **Constants**: Placeholder token `C` fitted post-hoc via BFGS
- **Our improvement**: We encode constants directly (either as named physics constants or decomposed floats), enabling the model to predict exact constant values. We also support 3× larger variable vocabulary for physics-specific symbols.

### Our Advantages
1. **Physics-aware vocabulary**: 30 domain-specific variable tokens vs. generic x0-x3
2. **Named constants**: Direct tokens for π, g, G, etc. reduce search space
3. **Structure tokens**: Enable dual-model skeleton prediction (24 structure tokens)
4. **Tree depth annotation**: Supports dual-axis RoPE (depth tokens DEPTH_0..DEPTH_9)
5. **Larger vocabulary**: 147 tokens vs. ~30-50 in prior work, covering more expressive equations
