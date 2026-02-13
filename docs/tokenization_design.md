# Symbolic Equation Tokenization and Representation Design

## 1. Prefix-Notation Encoding

Following Lample & Charton (2020), we encode equations as token sequences in **prefix (Polish) notation**. In prefix notation, operators precede their operands, which eliminates the need for parentheses and creates a unique, unambiguous representation for each expression tree.

**Example**: The equation `F = m*a + G*M*m/r²` becomes:
```
+ * a m * * G_const M * m / 1 ^ r 2
```

This maps directly to the expression tree:
```
        +
       / \
      *    *
     / \  /|\
    a   m G M  *
              / \
             m   /
                / \
               1   ^
                  / \
                 r   2
```

**Advantages of prefix notation**:
- Bijective mapping to expression trees (no ambiguity)
- No parentheses needed (reduces sequence length)
- Natural left-to-right parsing for transformers
- Easy computation of tree depth for positional encoding

## 2. Vocabulary (155 tokens)

| Category | Count | Tokens |
|----------|-------|--------|
| Structural | 5 | `<BOS>`, `<EOS>`, `<PAD>`, `<MASK>`, `<SEP>` |
| Arithmetic | 6 | `+`, `-`, `*`, `/`, `^`, `neg` |
| Trigonometric | 9 | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh` |
| Transcendental | 5 | `exp`, `log`, `ln`, `sqrt`, `abs` |
| Physics Variables | 81 | `m`, `M`, `g`, `F`, `E`, `L`, `H`, `K`, `U`, `W`, `P`, `T`, `V`, `omega`, `theta`, `phi`, `alpha`, `beta`, `gamma`, `delta`, `r`, `R`, `v`, `a`, `t`, `x`, `y`, `z`, `s`, `p`, `q`, `h`, `l`, `n`, `w`, `f`, `d`, `b`, `u`, `c`, `I`, `J`, `N`, `k`, `A`, `B`, `C`, `D`, `S`, `Q`, `O`, `rho`, `mu`, `sigma`, `tau`, `lambda_`, `x1`-`x3`, `y1`-`y3`, `v0`-`v2`, `a0`-`a2`, `r1`-`r2`, `m1`-`m3`, `k1`-`k2`, `F0`, `omega0`, `omega_d`, `theta0`, `phi0`, `P0` |
| Numeric Constants | 29 | `0`-`9`, `10`, `12`, `16`, `24`, `32`, `64`, `100`, `pi`, `e_const`, `G_const`, `c_const`, `0.5`, `1.5`, `2.0`, `3.0`, `4.0`, `-1`, `-2`, `-0.5` |
| Float Placeholders | 20 | `C0`-`C19` (for arbitrary constants) |
| **Total** | **155** | |

## 3. Bidirectional Conversion Functions

Implemented in `src/tokenizer.py`:

- **`infix_to_prefix(expr_str) -> List[str]`**: Parses infix equation string via sympy, converts to prefix tokens
- **`prefix_to_infix(tokens) -> str`**: Reconstructs infix string from prefix tokens via recursive descent
- **`encode(equation_str, max_len=128) -> List[int]`**: Full pipeline: infix → prefix → token indices (with BOS/EOS/PAD)
- **`decode(indices) -> str`**: Reverse pipeline: token indices → prefix tokens → infix string
- **`get_tree_depths(tokens) -> List[int]`**: Computes expression tree depth for each prefix token (for dual-axis positional encoding)
- **`round_trip_test(equation_str) -> bool`**: Verifies symbolic equivalence after encode→decode

### Round-Trip Fidelity Tests

All 6 test equations pass round-trip fidelity:
```
m * a                          → PASS
0.5 * m * v**2                 → PASS  (kinetic energy)
G_const * m * M / r**2         → PASS  (gravitational force)
sin(omega * t + phi)           → PASS  (SHM)
sqrt(2 * g * h)                → PASS  (Torricelli)
m * g * h + 0.5 * m * v**2    → PASS  (energy conservation)
```

## 4. Maximum Sequence Length Analysis

For the target equation corpus (50+ Newtonian physics templates at 3 difficulty levels):

| Difficulty | Avg Tokens | 95th %ile | Max Tokens |
|-----------|-----------|-----------|------------|
| Simple (L1) | 5-8 | 12 | 15 |
| Medium (L2) | 10-18 | 25 | 35 |
| Complex (L3) | 15-30 | 45 | 60 |

**Chosen context window: 128 tokens**

- With BOS/EOS tokens, 126 positions available for equation tokens
- The 95th percentile of even complex equations (45 tokens) fits comfortably within 128
- This matches the typical context lengths used in Lample & Charton (2020) for mathematical expressions
- The remaining capacity accommodates edge cases and observation pair encoding (when concatenated)

## Implementation Details

The tokenizer is fully implemented in `src/tokenizer.py` with:
- Sympy-based parsing for robust handling of mathematical expressions
- Fallback tokenization for expressions sympy cannot parse
- Sorted argument ordering for canonical prefix representation
- Special handling of: negation (neg operator), square root (sqrt), inverse (1/x → / 1 x)
