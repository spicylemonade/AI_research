"""
Symbolic equation tokenizer for PhysMDT.

Handles bidirectional conversion between equation strings (infix notation)
and token sequences (prefix notation). Designed for Newtonian physics
equations following Lample & Charton (2020) prefix-notation encoding.

References:
    - lample2020deep: Deep Learning for Symbolic Mathematics
    - vaswani2017attention: Attention is All You Need
"""

import re
from typing import List, Optional, Tuple, Dict
import sympy as sp
from sympy import symbols, sympify, sqrt, sin, cos, tan, exp, log, pi, oo
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ─── Vocabulary Definition ───────────────────────────────────────────────────

# Structural tokens
STRUCTURAL_TOKENS = ['<BOS>', '<EOS>', '<PAD>', '<MASK>', '<SEP>']

# Arithmetic operators
ARITHMETIC_OPS = ['+', '-', '*', '/', '^', 'neg']

# Trigonometric and transcendental functions
TRIG_FUNCTIONS = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']
TRANSCENDENTAL = ['exp', 'log', 'ln', 'sqrt', 'abs']

# Physics-specific variable symbols
PHYSICS_VARS = [
    'm', 'M', 'g', 'F', 'E', 'L', 'H', 'K', 'U', 'W', 'P', 'T', 'V',
    'omega', 'theta', 'phi', 'alpha', 'beta', 'gamma', 'delta',
    'r', 'R', 'v', 'a', 't', 'x', 'y', 'z', 's', 'p', 'q',
    'h', 'l', 'n', 'w', 'f', 'd', 'b', 'u', 'c',
    'I', 'J', 'N', 'k', 'A', 'B', 'C', 'D', 'S', 'Q', 'O',
    'rho', 'mu', 'sigma', 'tau', 'lambda_',
    'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
    'v0', 'v1', 'v2', 'a0', 'a1', 'a2',
    'r1', 'r2', 'm1', 'm2', 'm3',
    'k1', 'k2', 'F0', 'omega0', 'omega_d',
    'theta0', 'phi0', 'P0',
]

# Numeric constants
NUMERIC_CONSTANTS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '10', '12', '16', '24', '32', '64', '100',
    'pi', 'e_const', 'G_const', 'c_const',
    '0.5', '1.5', '2.0', '3.0', '4.0',
    '-1', '-2', '-0.5',
]

# Float placeholder tokens for arbitrary constants
FLOAT_TOKENS = [f'C{i}' for i in range(20)]  # C0, C1, ..., C19

ALL_TOKENS = (STRUCTURAL_TOKENS + ARITHMETIC_OPS + TRIG_FUNCTIONS +
              TRANSCENDENTAL + PHYSICS_VARS + NUMERIC_CONSTANTS + FLOAT_TOKENS)

# Build token-to-index and index-to-token mappings
TOKEN_TO_IDX: Dict[str, int] = {tok: idx for idx, tok in enumerate(ALL_TOKENS)}
IDX_TO_TOKEN: Dict[int, str] = {idx: tok for idx, tok in enumerate(ALL_TOKENS)}

VOCAB_SIZE = len(ALL_TOKENS)

# Special token indices
PAD_IDX = TOKEN_TO_IDX['<PAD>']
BOS_IDX = TOKEN_TO_IDX['<BOS>']
EOS_IDX = TOKEN_TO_IDX['<EOS>']
MASK_IDX = TOKEN_TO_IDX['<MASK>']
SEP_IDX = TOKEN_TO_IDX['<SEP>']

# Operator arities for prefix notation parsing
OPERATOR_ARITY = {
    '+': 2, '-': 2, '*': 2, '/': 2, '^': 2,
    'neg': 1,
    'sin': 1, 'cos': 1, 'tan': 1,
    'asin': 1, 'acos': 1, 'atan': 1,
    'sinh': 1, 'cosh': 1, 'tanh': 1,
    'exp': 1, 'log': 1, 'ln': 1, 'sqrt': 1, 'abs': 1,
}

MAX_SEQ_LEN = 128  # Maximum sequence length for equations


def infix_to_prefix(expr_str: str) -> List[str]:
    """Convert an infix equation string to prefix notation token list.

    Args:
        expr_str: Infix equation string, e.g., "m * a + G * M * m / r^2"

    Returns:
        List of tokens in prefix notation, e.g., ['+', '*', 'm', 'a', ...]
    """
    try:
        # Parse with sympy (keep ** for Python power operator; ^ is XOR)
        local_dict = _build_local_dict()
        expr = parse_expr(expr_str,
                          local_dict=local_dict,
                          transformations=standard_transformations + (implicit_multiplication_application,))
        return _sympy_to_prefix(expr)
    except Exception:
        # Fallback: try direct tokenization
        return _direct_tokenize(expr_str)


def prefix_to_infix(tokens: List[str]) -> str:
    """Convert prefix notation token list back to infix equation string.

    Args:
        tokens: List of tokens in prefix notation

    Returns:
        Infix equation string
    """
    expr, _ = _prefix_to_sympy(tokens, 0)
    if expr is not None:
        return str(expr)
    return ' '.join(tokens)


def encode(equation_str: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
    """Encode an equation string into a padded token index sequence.

    Args:
        equation_str: Infix equation string
        max_len: Maximum sequence length (with BOS/EOS)

    Returns:
        List of token indices, padded to max_len
    """
    prefix_tokens = infix_to_prefix(equation_str)
    indices = [BOS_IDX]
    for tok in prefix_tokens:
        if tok in TOKEN_TO_IDX:
            indices.append(TOKEN_TO_IDX[tok])
        else:
            # Try to map numeric constants
            idx = _map_numeric(tok)
            indices.append(idx)
    indices.append(EOS_IDX)

    # Truncate if too long
    if len(indices) > max_len:
        indices = indices[:max_len - 1] + [EOS_IDX]

    # Pad
    while len(indices) < max_len:
        indices.append(PAD_IDX)

    return indices


def decode(indices: List[int]) -> str:
    """Decode token indices back to an equation string.

    Args:
        indices: List of token indices

    Returns:
        Infix equation string
    """
    tokens = []
    for idx in indices:
        if idx == BOS_IDX:
            continue
        if idx == EOS_IDX:
            break
        if idx == PAD_IDX:
            continue
        tok = IDX_TO_TOKEN.get(idx, '?')
        tokens.append(tok)
    return prefix_to_infix(tokens)


def get_tree_depths(tokens: List[str]) -> List[int]:
    """Compute the expression tree depth for each token in prefix notation.

    This is used for the dual-axis positional encoding (sequence position + tree depth).

    Args:
        tokens: List of tokens in prefix notation

    Returns:
        List of tree depths (0-indexed from root)
    """
    depths = []
    stack = []  # Stack of (remaining_children, depth) pairs

    for tok in tokens:
        if stack:
            current_depth = stack[-1][1] + 1
        else:
            current_depth = 0

        depths.append(current_depth)

        arity = OPERATOR_ARITY.get(tok, 0)
        if arity > 0:
            # Operator: push expected children count
            stack.append([arity, current_depth])
        else:
            # Leaf: consume from parent chain
            while stack:
                stack[-1][0] -= 1
                if stack[-1][0] == 0:
                    stack.pop()
                else:
                    break

    return depths


def round_trip_test(equation_str: str) -> bool:
    """Test that encode -> decode produces a symbolically equivalent equation.

    Args:
        equation_str: Original infix equation string

    Returns:
        True if round-trip preserves symbolic equivalence
    """
    try:
        indices = encode(equation_str)
        recovered = decode(indices)
        local_dict = _build_local_dict()
        orig_expr = parse_expr(equation_str, local_dict=local_dict,
                               transformations=standard_transformations + (implicit_multiplication_application,))
        rec_expr = parse_expr(recovered, local_dict=local_dict,
                              transformations=standard_transformations + (implicit_multiplication_application,))
        diff = sp.simplify(orig_expr - rec_expr)
        return diff == 0
    except Exception:
        return False


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _build_local_dict():
    """Build a local dictionary for sympy parsing with physics symbols."""
    d = {}
    for v in PHYSICS_VARS:
        clean = v.rstrip('_')
        d[clean] = sp.Symbol(clean)
    d['pi'] = sp.pi
    d['e_const'] = sp.E
    d['G_const'] = sp.Symbol('G_const')
    d['c_const'] = sp.Symbol('c_const')
    return d


def _sympy_to_prefix(expr) -> List[str]:
    """Convert a sympy expression to prefix notation tokens."""
    if expr is sp.pi:
        return ['pi']
    if expr is sp.E:
        return ['e_const']
    if isinstance(expr, sp.Number):
        return [_number_to_token(expr)]
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        return [name]
    if isinstance(expr, sp.Add):
        args = sorted(expr.args, key=lambda a: str(a))
        if len(args) == 0:
            return ['0']
        result = _sympy_to_prefix(args[0])
        for arg in args[1:]:
            right = _sympy_to_prefix(arg)
            result = ['+'] + result + right
        return result
    if isinstance(expr, sp.Mul):
        # Separate numeric coefficient from symbolic factors
        coeff, rest = expr.as_coeff_Mul()
        if coeff == -1 and rest != sp.S.One:
            return ['neg'] + _sympy_to_prefix(rest)
        args = sorted(expr.args, key=lambda a: str(a))
        if len(args) == 0:
            return ['1']
        result = _sympy_to_prefix(args[0])
        for arg in args[1:]:
            right = _sympy_to_prefix(arg)
            result = ['*'] + result + right
        return result
    if isinstance(expr, sp.Pow):
        base_expr, exp_expr = expr.args
        # Handle sqrt: x^(1/2)
        if exp_expr == sp.Rational(1, 2):
            return ['sqrt'] + _sympy_to_prefix(base_expr)
        # Handle 1/x: x^(-1) -> / 1 x
        if exp_expr == sp.Integer(-1):
            return ['/'] + ['1'] + _sympy_to_prefix(base_expr)
        # Handle x^(-n) -> / 1 (^ x n)
        if isinstance(exp_expr, sp.Number) and exp_expr < 0:
            pos_exp = -exp_expr
            return ['/'] + ['1'] + ['^'] + _sympy_to_prefix(base_expr) + _sympy_to_prefix(pos_exp)
        return ['^'] + _sympy_to_prefix(base_expr) + _sympy_to_prefix(exp_expr)
    if isinstance(expr, sp.sin):
        return ['sin'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.cos):
        return ['cos'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.tan):
        return ['tan'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.exp):
        return ['exp'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.log):
        return ['log'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.Abs):
        return ['abs'] + _sympy_to_prefix(expr.args[0])
    # Inverse trig
    if isinstance(expr, sp.asin):
        return ['asin'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.acos):
        return ['acos'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.atan):
        return ['atan'] + _sympy_to_prefix(expr.args[0])
    # Hyperbolic
    if isinstance(expr, sp.sinh):
        return ['sinh'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.cosh):
        return ['cosh'] + _sympy_to_prefix(expr.args[0])
    if isinstance(expr, sp.tanh):
        return ['tanh'] + _sympy_to_prefix(expr.args[0])
    # Fallback for other functions
    func_name = str(type(expr).__name__).lower()
    if func_name in OPERATOR_ARITY:
        result = [func_name]
        for arg in expr.args:
            result.extend(_sympy_to_prefix(arg))
        return result
    # Last resort
    return [str(expr)]


def _number_to_token(num) -> str:
    """Convert a sympy number to the closest vocabulary token."""
    if num == sp.pi:
        return 'pi'
    if num is sp.E:
        return 'e_const'

    # Handle Rational
    if isinstance(num, sp.Rational) and not isinstance(num, sp.Integer):
        p, q = int(num.p), int(num.q)
        # Common fractions
        fval = float(num)
        for tok in NUMERIC_CONSTANTS:
            try:
                if tok in ('pi', 'e_const', 'G_const', 'c_const'):
                    continue
                if abs(float(tok) - fval) < 1e-12:
                    return tok
            except ValueError:
                continue
        # Encode as division: p/q
        return str(fval)

    val = float(num)
    # Check exact matches in vocabulary
    for tok in NUMERIC_CONSTANTS:
        try:
            if tok in ('pi', 'e_const', 'G_const', 'c_const'):
                continue
            if abs(float(tok) - val) < 1e-12:
                return tok
        except ValueError:
            continue
    # Use float representation
    if val == int(val) and abs(val) < 1000:
        return str(int(val))
    return str(val) if abs(val) < 1000 else 'C0'


def _prefix_to_sympy(tokens: List[str], pos: int) -> Tuple[Optional[sp.Expr], int]:
    """Parse prefix notation tokens into a sympy expression.

    Returns (expression, next_position).
    """
    if pos >= len(tokens):
        return None, pos

    tok = tokens[pos]

    if tok in OPERATOR_ARITY:
        arity = OPERATOR_ARITY[tok]
        if arity == 1:
            arg, next_pos = _prefix_to_sympy(tokens, pos + 1)
            if arg is None:
                return None, next_pos
            return _apply_unary(tok, arg), next_pos
        elif arity == 2:
            left, next_pos = _prefix_to_sympy(tokens, pos + 1)
            if left is None:
                return None, next_pos
            right, next_pos = _prefix_to_sympy(tokens, next_pos)
            if right is None:
                return None, next_pos
            return _apply_binary(tok, left, right), next_pos
    else:
        # It's a leaf (variable or constant)
        return _token_to_sympy(tok), pos + 1


def _apply_unary(op: str, arg):
    """Apply a unary operator."""
    ops = {
        'neg': lambda x: -x,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
        'sqrt': sp.sqrt, 'abs': sp.Abs,
    }
    return ops.get(op, lambda x: x)(arg)


def _apply_binary(op: str, left, right):
    """Apply a binary operator."""
    if op == '+':
        return left + right
    elif op == '-':
        return left - right
    elif op == '*':
        return left * right
    elif op == '/':
        return left / right
    elif op == '^':
        return left ** right
    return left


def _token_to_sympy(tok: str):
    """Convert a token string to a sympy expression."""
    if tok == 'pi':
        return sp.pi
    if tok == 'e_const':
        return sp.E
    if tok in ('G_const', 'c_const'):
        return sp.Symbol(tok)
    try:
        val = float(tok)
        if val == int(val) and abs(val) < 1000:
            return sp.Integer(int(val))
        return sp.Rational(val).limit_denominator(1000)
    except ValueError:
        pass
    return sp.Symbol(tok)


def _direct_tokenize(expr_str: str) -> List[str]:
    """Fallback: directly tokenize an expression string."""
    # Split on operators and whitespace
    pattern = r'(\+|\-|\*|\/|\^|\(|\)|\s+)'
    parts = re.split(pattern, expr_str)
    return [p.strip() for p in parts if p.strip() and p.strip() not in ('(', ')')]


def _map_numeric(tok: str) -> int:
    """Map a numeric string to the closest token index."""
    try:
        val = float(tok)
        for const_tok in NUMERIC_CONSTANTS:
            try:
                if const_tok in ('pi', 'e_const', 'G_const', 'c_const'):
                    continue
                if float(const_tok) == val:
                    return TOKEN_TO_IDX[const_tok]
            except ValueError:
                continue
        # Map to nearest float token
        return TOKEN_TO_IDX.get('C0', PAD_IDX)
    except ValueError:
        return TOKEN_TO_IDX.get(tok, PAD_IDX)


# Print vocab stats
if __name__ == '__main__':
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"  Structural tokens: {len(STRUCTURAL_TOKENS)}")
    print(f"  Arithmetic operators: {len(ARITHMETIC_OPS)}")
    print(f"  Trig functions: {len(TRIG_FUNCTIONS)}")
    print(f"  Transcendental: {len(TRANSCENDENTAL)}")
    print(f"  Physics variables: {len(PHYSICS_VARS)}")
    print(f"  Numeric constants: {len(NUMERIC_CONSTANTS)}")
    print(f"  Float placeholders: {len(FLOAT_TOKENS)}")

    # Run round-trip tests
    test_equations = [
        "m * a",
        "0.5 * m * v**2",
        "G_const * m * M / r**2",
        "sin(omega * t + phi)",
        "sqrt(2 * g * h)",
        "m * g * h + 0.5 * m * v**2",
    ]
    print(f"\nRound-trip tests:")
    for eq in test_equations:
        prefix = infix_to_prefix(eq)
        result = round_trip_test(eq)
        print(f"  {eq} -> {prefix} -> {'PASS' if result else 'FAIL'}")
