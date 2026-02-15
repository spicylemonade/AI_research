"""Symbolic expression tokenizer with prefix (Polish) notation.

Encodes SymPy expressions into token sequences and decodes them back.
Supports operators, variables x0..x9, numeric constants, and special tokens.
"""

import re
import sympy
from sympy import (Symbol, Add, Mul, Pow, sin, cos, tan, log, exp, sqrt, pi,
                   Rational, Integer, Float, S, Number)
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Vocabulary Definition
# ---------------------------------------------------------------------------

# Special tokens
SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<MASK>', '<SEP>']

# Operators (binary and unary)
BINARY_OPS = ['+', '-', '*', '/', '^']
UNARY_OPS = ['sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'neg']

# Variables
VARIABLES = [f'x{i}' for i in range(10)]

# Named constants
NAMED_CONSTANTS = ['pi']

# Placeholder constants (for physical constants like G, k, etc.)
CONSTANT_PLACEHOLDERS = [f'C{i}' for i in range(5)]

# Integer constants: -10 to 10
INT_CONSTANTS = [str(i) for i in range(-10, 11)]

# Float constants: common values
FLOAT_CONSTANTS = ['0.5', '1.5', '2.5', '0.25', '0.1', '0.01', '0.33', '0.67']

# Build full vocabulary
VOCAB = (SPECIAL_TOKENS + BINARY_OPS + UNARY_OPS + VARIABLES +
         NAMED_CONSTANTS + CONSTANT_PLACEHOLDERS + INT_CONSTANTS + FLOAT_CONSTANTS)

# Token to index and index to token mappings
TOKEN2IDX: Dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB)}
IDX2TOKEN: Dict[int, str] = {idx: tok for idx, tok in enumerate(VOCAB)}

# Special token indices
PAD_IDX = TOKEN2IDX['<PAD>']
SOS_IDX = TOKEN2IDX['<SOS>']
EOS_IDX = TOKEN2IDX['<EOS>']
MASK_IDX = TOKEN2IDX['<MASK>']
SEP_IDX = TOKEN2IDX['<SEP>']

VOCAB_SIZE = len(VOCAB)


# ---------------------------------------------------------------------------
# SymPy Expression → Prefix Token List
# ---------------------------------------------------------------------------

def _is_negative_one(expr):
    """Check if expression is -1."""
    return expr == S.NegativeOne


def _expr_to_prefix_tokens(expr) -> List[str]:
    """Convert a SymPy expression to prefix notation token list."""
    if expr is None:
        return []

    # Handle numeric constants
    if isinstance(expr, (Integer, int)):
        val = int(expr)
        if -10 <= val <= 10:
            return [str(val)]
        # For larger integers, decompose
        return [str(val)]

    if isinstance(expr, Rational) and not isinstance(expr, Integer):
        p, q = expr.p, expr.q
        if p == 1 and q == 2:
            return ['0.5']
        if p == 1 and q == 3:
            return ['0.33']
        if p == 2 and q == 3:
            return ['0.67']
        if p == 1 and q == 4:
            return ['0.25']
        if p == 3 and q == 2:
            return ['1.5']
        if p == 5 and q == 2:
            return ['2.5']
        # General rational: p/q
        return ['/', str(p) if -10 <= p <= 10 else str(p),
                str(q) if -10 <= q <= 10 else str(q)]

    if isinstance(expr, Float):
        val = float(expr)
        # Check common floats
        for fstr in FLOAT_CONSTANTS:
            if abs(val - float(fstr)) < 1e-10:
                return [fstr]
        if abs(val - round(val)) < 1e-10 and -10 <= round(val) <= 10:
            return [str(int(round(val)))]
        # Approximate
        return [_float_to_token(val)]

    if isinstance(expr, Number):
        val = float(expr)
        if abs(val - round(val)) < 1e-10 and -10 <= round(val) <= 10:
            return [str(int(round(val)))]
        for fstr in FLOAT_CONSTANTS:
            if abs(val - float(fstr)) < 1e-10:
                return [fstr]
        return [_float_to_token(val)]

    # pi
    if expr == pi:
        return ['pi']

    # Symbol
    if isinstance(expr, Symbol):
        name = str(expr)
        if name in TOKEN2IDX:
            return [name]
        # Try matching x0..x9 or C0..C4
        return [name]

    # Negation: -expr
    if isinstance(expr, Mul) and len(expr.args) == 2 and expr.args[0] == S.NegativeOne:
        return ['neg'] + _expr_to_prefix_tokens(expr.args[1])

    # Addition
    if isinstance(expr, Add):
        args = list(expr.args)
        if len(args) == 1:
            return _expr_to_prefix_tokens(args[0])
        # Build binary tree left-to-right
        result = _expr_to_prefix_tokens(args[0])
        for a in args[1:]:
            # Check if this term is negative (subtraction)
            if isinstance(a, Mul) and len(a.args) >= 1 and a.args[0] == S.NegativeOne:
                pos_part = Mul(*a.args[1:]) if len(a.args) > 2 else a.args[1]
                result = ['-'] + result + _expr_to_prefix_tokens(pos_part)
            else:
                result = ['+'] + result + _expr_to_prefix_tokens(a)
        return result

    # Multiplication
    if isinstance(expr, Mul):
        args = list(expr.args)
        # Handle leading -1
        if args[0] == S.NegativeOne:
            if len(args) == 2:
                return ['neg'] + _expr_to_prefix_tokens(args[1])
            rest = Mul(*args[1:])
            return ['neg'] + _expr_to_prefix_tokens(rest)
        # Filter out numeric coefficient
        result = _expr_to_prefix_tokens(args[0])
        for a in args[1:]:
            result = ['*'] + result + _expr_to_prefix_tokens(a)
        return result

    # Power
    if isinstance(expr, Pow):
        base, exponent = expr.args
        # sqrt
        if exponent == Rational(1, 2):
            return ['sqrt'] + _expr_to_prefix_tokens(base)
        # 1/x = x^(-1)
        if exponent == S.NegativeOne:
            return ['/', '1'] + _expr_to_prefix_tokens(base)
        # x^(-n)
        if isinstance(exponent, (Integer, Rational)) and exponent < 0:
            return ['/', '1'] + _expr_to_prefix_tokens(Pow(base, -exponent))
        return ['^'] + _expr_to_prefix_tokens(base) + _expr_to_prefix_tokens(exponent)

    # Unary functions
    func_map = {
        sympy.sin: 'sin',
        sympy.cos: 'cos',
        sympy.tan: 'tan',
        sympy.log: 'log',
        sympy.exp: 'exp',
        sympy.sqrt: 'sqrt',
    }
    for func_class, tok in func_map.items():
        if isinstance(expr, func_class.__class__) or (hasattr(expr, 'func') and expr.func == func_class):
            return [tok] + _expr_to_prefix_tokens(expr.args[0])

    # Fallback: try string representation
    s = str(expr)
    if s in TOKEN2IDX:
        return [s]

    raise ValueError(f"Cannot tokenize expression: {expr} (type: {type(expr)})")


def _float_to_token(val: float) -> str:
    """Convert a float to the nearest token in vocabulary."""
    # Check integers
    if abs(val - round(val)) < 1e-10 and -10 <= round(val) <= 10:
        return str(int(round(val)))
    # Check float constants
    best_tok = None
    best_dist = float('inf')
    for fstr in FLOAT_CONSTANTS:
        d = abs(val - float(fstr))
        if d < best_dist:
            best_dist = d
            best_tok = fstr
    if best_dist < 0.05:
        return best_tok
    # Round to nearest 0.5
    rounded = round(val * 2) / 2
    if abs(rounded - round(rounded)) < 1e-10 and -10 <= round(rounded) <= 10:
        return str(int(round(rounded)))
    return str(int(round(val))) if -10 <= round(val) <= 10 else '1'


# ---------------------------------------------------------------------------
# Prefix Token List → SymPy Expression
# ---------------------------------------------------------------------------

def _prefix_tokens_to_expr(tokens: List[str]) -> Tuple[sympy.Expr, int]:
    """Parse prefix token list into SymPy expression. Returns (expr, next_index)."""
    if not tokens:
        raise ValueError("Empty token list")

    tok = tokens[0]

    # Binary operators
    if tok in BINARY_OPS:
        left, idx1 = _prefix_tokens_to_expr(tokens[1:])
        right, idx2 = _prefix_tokens_to_expr(tokens[1 + idx1:])
        op_map = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '^': lambda a, b: a ** b,
        }
        return op_map[tok](left, right), 1 + idx1 + idx2

    # Unary operators
    if tok in UNARY_OPS:
        arg, idx = _prefix_tokens_to_expr(tokens[1:])
        unary_map = {
            'sqrt': lambda a: sympy.sqrt(a),
            'sin': lambda a: sympy.sin(a),
            'cos': lambda a: sympy.cos(a),
            'tan': lambda a: sympy.tan(a),
            'log': lambda a: sympy.log(a),
            'exp': lambda a: sympy.exp(a),
            'neg': lambda a: -a,
        }
        return unary_map[tok](arg), 1 + idx

    # Constants
    if tok == 'pi':
        return sympy.pi, 1

    # Variables
    if tok.startswith('x') and tok[1:].isdigit():
        return Symbol(tok, positive=True), 1

    # Constant placeholders
    if tok.startswith('C') and tok[1:].isdigit():
        return Symbol(tok, positive=True), 1

    # Numbers
    try:
        if '.' in tok:
            return sympy.Rational(tok).limit_denominator(1000), 1
        else:
            return sympy.Integer(int(tok)), 1
    except (ValueError, TypeError):
        pass

    raise ValueError(f"Unknown token: {tok}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ExprTokenizer:
    """Tokenizer for symbolic expressions in prefix notation."""

    def __init__(self):
        self.vocab = VOCAB
        self.token2idx = TOKEN2IDX
        self.idx2token = IDX2TOKEN
        self.vocab_size = VOCAB_SIZE
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.mask_idx = MASK_IDX
        self.sep_idx = SEP_IDX

    def encode(self, expr, add_sos_eos: bool = True) -> List[int]:
        """Encode a SymPy expression to a list of token indices.

        Args:
            expr: SymPy expression
            add_sos_eos: whether to wrap with <SOS> and <EOS>

        Returns:
            List of integer token indices
        """
        tokens = _expr_to_prefix_tokens(expr)
        indices = []
        if add_sos_eos:
            indices.append(SOS_IDX)
        for tok in tokens:
            if tok in self.token2idx:
                indices.append(self.token2idx[tok])
            else:
                # Try to handle unknown tokens gracefully
                indices.append(self.token2idx.get('1', SOS_IDX))
        if add_sos_eos:
            indices.append(EOS_IDX)
        return indices

    def decode(self, indices: List[int], strip_special: bool = True) -> sympy.Expr:
        """Decode a list of token indices back to a SymPy expression.

        Args:
            indices: List of integer token indices
            strip_special: whether to strip <SOS>, <EOS>, <PAD> tokens

        Returns:
            SymPy expression
        """
        tokens = [self.idx2token[idx] for idx in indices]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        if not tokens:
            return sympy.Integer(0)
        expr, consumed = _prefix_tokens_to_expr(tokens)
        return expr

    def encode_to_tokens(self, expr) -> List[str]:
        """Encode a SymPy expression to a list of token strings."""
        return _expr_to_prefix_tokens(expr)

    def decode_from_tokens(self, tokens: List[str]) -> sympy.Expr:
        """Decode a list of token strings to a SymPy expression."""
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        if not tokens:
            return sympy.Integer(0)
        expr, _ = _prefix_tokens_to_expr(tokens)
        return expr

    def pad_sequence(self, indices: List[int], max_len: int) -> List[int]:
        """Pad or truncate a token sequence to max_len."""
        if len(indices) >= max_len:
            return indices[:max_len]
        return indices + [PAD_IDX] * (max_len - len(indices))

    def get_token_str(self, idx: int) -> str:
        """Get the string representation of a token index."""
        return self.idx2token.get(idx, '<UNK>')


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def encode(expr, add_sos_eos: bool = True) -> List[int]:
    """Encode a SymPy expression to token indices."""
    return ExprTokenizer().encode(expr, add_sos_eos)


def decode(indices: List[int]) -> sympy.Expr:
    """Decode token indices to a SymPy expression."""
    return ExprTokenizer().decode(indices)


if __name__ == '__main__':
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Vocabulary: {VOCAB}")
    print()

    # Test round-trip on some expressions
    from data.equations import get_all_equations
    tokenizer = ExprTokenizer()
    passed = 0
    failed = 0
    for eq in get_all_equations():
        try:
            tokens = tokenizer.encode_to_tokens(eq.symbolic_expr)
            indices = tokenizer.encode(eq.symbolic_expr)
            decoded = tokenizer.decode(indices)
            # Check symbolic equivalence
            diff = sympy.simplify(decoded - eq.symbolic_expr)
            if diff == 0 or diff == sympy.S.Zero:
                passed += 1
            else:
                # Try numerical check — substitute all free symbols
                import random
                random.seed(42)
                all_syms = eq.symbolic_expr.free_symbols | decoded.free_symbols
                vars_dict = {s: random.uniform(0.5, 5.0) for s in all_syms}
                orig_val = complex(eq.symbolic_expr.subs(vars_dict))
                dec_val = complex(decoded.subs(vars_dict))
                if abs(orig_val - dec_val) < 1e-4 * (abs(orig_val) + 1):
                    passed += 1
                else:
                    failed += 1
                    print(f"FAIL: {eq.name}")
                    print(f"  Original: {eq.symbolic_expr}")
                    print(f"  Tokens: {tokens}")
                    print(f"  Decoded: {decoded}")
                    print(f"  orig_val={orig_val}, dec_val={dec_val}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {eq.name}: {e}")
    print(f"\nRound-trip test: {passed} passed, {failed} failed out of {passed + failed}")
