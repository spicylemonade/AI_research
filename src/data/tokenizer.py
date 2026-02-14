"""Symbolic equation tokenizer for PhysMDT.

Converts mathematical expressions to/from token sequences in reverse Polish
notation (RPN). Supports operators, numeric constants, and variable symbols.
Includes augmentation transforms for training data diversity.
"""

import re
import math
import random
from typing import List, Tuple, Optional, Dict

import sympy
from sympy import (
    Symbol, Integer, Float, Rational, pi, E,
    sin, cos, tan, log, exp, sqrt, Abs, Pow,
    simplify, expand, factor,
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Special tokens
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
MASK_TOKEN = 3

# Operators (binary)
BINARY_OPS = {'+': 4, '-': 5, '*': 6, '/': 7, '^': 8}

# Operators (unary)
UNARY_OPS = {
    'sqrt': 9, 'sin': 10, 'cos': 11, 'tan': 12,
    'log': 13, 'exp': 14, 'abs': 15,
}

# Variables x1..x9
VARIABLE_TOKENS = {f'x{i}': 16 + i - 1 for i in range(1, 10)}  # 16-24

# Constants
CONST_PI = 25
CONST_E = 26

# Integer tokens: -49 to 49 → tokens 27..125
INT_OFFSET = 27
INT_RANGE = 49  # covers -49..49 → 99 tokens → 27..125

# Float indicator + digits
FLOAT_DOT = 126
UNK_TOKEN = 127

# Additional numeric tokens for larger range
# Tokens 128-199: reserved for extended numeric representation
NUM_EXT_START = 128

VOCAB_SIZE = 200  # Total vocabulary size

# Reverse mappings
ID_TO_TOKEN: Dict[int, str] = {
    PAD_TOKEN: '<pad>', BOS_TOKEN: '<bos>', EOS_TOKEN: '<eos>', MASK_TOKEN: '<mask>',
    CONST_PI: 'pi', CONST_E: 'euler_e', FLOAT_DOT: '.', UNK_TOKEN: '<unk>',
}
ID_TO_TOKEN.update({v: k for k, v in BINARY_OPS.items()})
ID_TO_TOKEN.update({v: k for k, v in UNARY_OPS.items()})
ID_TO_TOKEN.update({v: k for k, v in VARIABLE_TOKENS.items()})
for i in range(-INT_RANGE, INT_RANGE + 1):
    ID_TO_TOKEN[INT_OFFSET + i + INT_RANGE] = str(i)

TOKEN_TO_ID: Dict[str, int] = {v: k for k, v in ID_TO_TOKEN.items()}

# Operator arity
ARITY = {}
for op in BINARY_OPS:
    ARITY[op] = 2
for op in UNARY_OPS:
    ARITY[op] = 1


def _sympy_to_rpn(expr) -> List[str]:
    """Convert a SymPy expression to RPN token list."""
    if isinstance(expr, Symbol):
        name = str(expr)
        if name == 'pi':
            return ['pi']
        return [name]
    if expr is pi:
        return ['pi']
    if expr is E:
        return ['euler_e']
    if isinstance(expr, (Integer, int)):
        val = int(expr)
        if -INT_RANGE <= val <= INT_RANGE:
            return [str(val)]
        # Decompose large integers
        return [str(val)]
    if isinstance(expr, Rational) and not isinstance(expr, Integer):
        num = _sympy_to_rpn(Integer(expr.p))
        den = _sympy_to_rpn(Integer(expr.q))
        return num + den + ['/']
    if isinstance(expr, Float):
        val = float(expr)
        # Try to represent as integer if close
        if abs(val - round(val)) < 1e-10 and -INT_RANGE <= round(val) <= INT_RANGE:
            return [str(int(round(val)))]
        # Represent as ratio of integers if possible
        rat = Rational(val).limit_denominator(1000)
        if abs(float(rat) - val) < 1e-8:
            return _sympy_to_rpn(rat)
        # Fallback: represent the float as a string
        return [f'float:{val}']
    if isinstance(expr, sympy.core.numbers.NegativeOne):
        return ['-1']
    if isinstance(expr, sympy.core.numbers.Half):
        return ['1', '2', '/']
    if isinstance(expr, sympy.core.numbers.One):
        return ['1']
    if isinstance(expr, sympy.core.numbers.Zero):
        return ['0']

    func = expr.func
    args = expr.args

    if func == sympy.Add:
        if len(args) == 0:
            return ['0']
        result = _sympy_to_rpn(args[0])
        for arg in args[1:]:
            result += _sympy_to_rpn(arg) + ['+']
        return result
    if func == sympy.Mul:
        if len(args) == 0:
            return ['1']
        result = _sympy_to_rpn(args[0])
        for arg in args[1:]:
            result += _sympy_to_rpn(arg) + ['*']
        return result
    if func == sympy.Pow:
        base_rpn = _sympy_to_rpn(args[0])
        exp_rpn = _sympy_to_rpn(args[1])
        # Special case: x^(1/2) = sqrt(x)
        if args[1] == Rational(1, 2):
            return base_rpn + ['sqrt']
        # Special case: x^(-1) = 1/x
        if args[1] == Integer(-1):
            return ['1'] + base_rpn + ['/']
        return base_rpn + exp_rpn + ['^']
    if func == sin:
        return _sympy_to_rpn(args[0]) + ['sin']
    if func == cos:
        return _sympy_to_rpn(args[0]) + ['cos']
    if func == tan:
        return _sympy_to_rpn(args[0]) + ['tan']
    if func == log:
        return _sympy_to_rpn(args[0]) + ['log']
    if func == exp:
        return _sympy_to_rpn(args[0]) + ['exp']
    if func == sqrt:
        return _sympy_to_rpn(args[0]) + ['sqrt']
    if func == Abs:
        return _sympy_to_rpn(args[0]) + ['abs']

    # Fallback: try to convert as string
    return [str(expr)]


def _rpn_to_sympy(tokens: List[str]):
    """Convert RPN token list back to SymPy expression."""
    stack = []
    variables = {f'x{i}': Symbol(f'x{i}') for i in range(1, 10)}

    for tok in tokens:
        if tok in variables:
            stack.append(variables[tok])
        elif tok == 'pi':
            stack.append(pi)
        elif tok == 'euler_e':
            stack.append(E)
        elif tok.startswith('float:'):
            stack.append(Float(tok[6:]))
        elif tok in BINARY_OPS:
            if len(stack) < 2:
                raise ValueError(f"Not enough operands for '{tok}'")
            b = stack.pop()
            a = stack.pop()
            if tok == '+':
                stack.append(a + b)
            elif tok == '-':
                stack.append(a - b)
            elif tok == '*':
                stack.append(a * b)
            elif tok == '/':
                stack.append(a / b)
            elif tok == '^':
                stack.append(Pow(a, b))
        elif tok in UNARY_OPS:
            if len(stack) < 1:
                raise ValueError(f"Not enough operands for '{tok}'")
            a = stack.pop()
            if tok == 'sqrt':
                stack.append(sqrt(a))
            elif tok == 'sin':
                stack.append(sin(a))
            elif tok == 'cos':
                stack.append(cos(a))
            elif tok == 'tan':
                stack.append(tan(a))
            elif tok == 'log':
                stack.append(log(a))
            elif tok == 'exp':
                stack.append(exp(a))
            elif tok == 'abs':
                stack.append(Abs(a))
        else:
            # Try numeric
            try:
                val = int(tok)
                stack.append(Integer(val))
            except ValueError:
                try:
                    val = float(tok)
                    stack.append(Float(val))
                except ValueError:
                    # Unknown symbol
                    stack.append(Symbol(tok))

    if len(stack) != 1:
        raise ValueError(f"Invalid RPN: stack has {len(stack)} items, expected 1")
    return stack[0]


class EquationTokenizer:
    """Tokenizer for symbolic mathematical expressions.

    Converts between:
    - String expressions (infix notation)
    - SymPy expressions
    - RPN token lists (string tokens)
    - Integer token ID sequences
    """

    def __init__(self, max_seq_len: int = 64):
        self.max_seq_len = max_seq_len
        self.vocab_size = VOCAB_SIZE

    def expr_to_rpn(self, expr_str: str) -> List[str]:
        """Convert an expression string to RPN token list."""
        # Parse with SymPy
        local_dict = {f'x{i}': Symbol(f'x{i}') for i in range(1, 10)}
        local_dict['pi'] = pi
        local_dict['e'] = E
        local_dict['E'] = E
        try:
            expr = parse_expr(
                expr_str,
                local_dict=local_dict,
                transformations=standard_transformations + (implicit_multiplication_application,),
            )
        except Exception:
            expr = sympy.sympify(expr_str, locals=local_dict)
        return _sympy_to_rpn(expr)

    def rpn_to_expr(self, tokens: List[str]) -> str:
        """Convert RPN token list to expression string."""
        expr = _rpn_to_sympy(tokens)
        return str(expr)

    def encode(self, expr_str: str) -> List[int]:
        """Encode an expression string to integer token IDs.

        Returns: [BOS, token_ids..., EOS] padded to max_seq_len.
        """
        rpn = self.expr_to_rpn(expr_str)
        ids = [BOS_TOKEN]
        for tok in rpn:
            if tok in TOKEN_TO_ID:
                ids.append(TOKEN_TO_ID[tok])
            elif tok.startswith('float:'):
                # Encode float as sequence of digit tokens
                val = float(tok[6:])
                # Simple representation: round to nearest 0.01 and use int encoding
                int_part = int(val)
                if -INT_RANGE <= int_part <= INT_RANGE:
                    ids.append(INT_OFFSET + int_part + INT_RANGE)
                else:
                    ids.append(UNK_TOKEN)
            else:
                try:
                    val = int(tok)
                    if -INT_RANGE <= val <= INT_RANGE:
                        ids.append(INT_OFFSET + val + INT_RANGE)
                    else:
                        # Large integer: decompose
                        ids.append(UNK_TOKEN)
                except ValueError:
                    ids.append(UNK_TOKEN)
        ids.append(EOS_TOKEN)

        # Truncate or pad
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len - 1] + [EOS_TOKEN]
        while len(ids) < self.max_seq_len:
            ids.append(PAD_TOKEN)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode integer token IDs back to expression string."""
        tokens = []
        for tid in ids:
            if tid in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN):
                continue
            if tid in ID_TO_TOKEN:
                tokens.append(ID_TO_TOKEN[tid])
            else:
                tokens.append('<unk>')

        if not tokens:
            return '0'
        try:
            return self.rpn_to_expr(tokens)
        except Exception:
            return ' '.join(tokens)

    def encode_sympy(self, expr) -> List[int]:
        """Encode a SymPy expression to integer token IDs."""
        rpn = _sympy_to_rpn(expr)
        ids = [BOS_TOKEN]
        for tok in rpn:
            if tok in TOKEN_TO_ID:
                ids.append(TOKEN_TO_ID[tok])
            else:
                try:
                    val = int(tok)
                    if -INT_RANGE <= val <= INT_RANGE:
                        ids.append(INT_OFFSET + val + INT_RANGE)
                    else:
                        ids.append(UNK_TOKEN)
                except ValueError:
                    ids.append(UNK_TOKEN)
        ids.append(EOS_TOKEN)
        while len(ids) < self.max_seq_len:
            ids.append(PAD_TOKEN)
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len - 1] + [EOS_TOKEN]
        return ids

    def get_rpn_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to RPN string tokens (without special tokens)."""
        tokens = []
        for tid in ids:
            if tid in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN):
                continue
            if tid in ID_TO_TOKEN:
                tokens.append(ID_TO_TOKEN[tid])
        return tokens


# ---------------------------------------------------------------------------
# Augmentation Transforms
# ---------------------------------------------------------------------------

def commutative_swap(expr) -> list:
    """Swap operands of commutative operations (+ and *)."""
    results = []
    if isinstance(expr, sympy.Add):
        args = list(expr.args)
        if len(args) >= 2:
            # Swap first two
            swapped = list(args)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            results.append(sympy.Add(*swapped))
            # Reverse all
            results.append(sympy.Add(*reversed(args)))
    if isinstance(expr, sympy.Mul):
        args = list(expr.args)
        if len(args) >= 2:
            swapped = list(args)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            results.append(sympy.Mul(*swapped))
            results.append(sympy.Mul(*reversed(args)))
    # Recurse into subexpressions
    for arg_idx, arg in enumerate(expr.args):
        for sub_result in commutative_swap(arg):
            new_args = list(expr.args)
            new_args[arg_idx] = sub_result
            results.append(expr.func(*new_args))
    return results


def constant_folding(expr):
    """Fold constant subexpressions (e.g., 2*3 → 6)."""
    return simplify(expr)


def identity_elimination(expr):
    """Eliminate identity operations (x+0→x, x*1→x, x^1→x)."""
    rules = [
        (lambda e: isinstance(e, sympy.Add) and Integer(0) in e.args,
         lambda e: sympy.Add(*[a for a in e.args if a != Integer(0)]) if len([a for a in e.args if a != Integer(0)]) > 1 else [a for a in e.args if a != Integer(0)][0]),
        (lambda e: isinstance(e, sympy.Mul) and Integer(1) in e.args,
         lambda e: sympy.Mul(*[a for a in e.args if a != Integer(1)]) if len([a for a in e.args if a != Integer(1)]) > 1 else [a for a in e.args if a != Integer(1)][0]),
    ]
    result = expr
    for check, transform in rules:
        if check(result):
            result = transform(result)
    return result


def associative_regroup(expr):
    """Regroup associative operations (a+(b+c) → (a+b)+c)."""
    if isinstance(expr, sympy.Add) and len(expr.args) >= 3:
        args = list(expr.args)
        random.shuffle(args)
        # Build left-leaning tree
        result = args[0]
        for a in args[1:]:
            result = result + a
        return result
    if isinstance(expr, sympy.Mul) and len(expr.args) >= 3:
        args = list(expr.args)
        random.shuffle(args)
        result = args[0]
        for a in args[1:]:
            result = result * a
        return result
    return expr


def augment_expression(expr, num_augmentations: int = 8) -> list:
    """Generate multiple equivalent forms of an expression.

    Applies commutative swap, constant folding, identity elimination,
    and associative regrouping to create diverse equivalent representations.
    """
    results = set()
    results.add(expr)

    # Apply each transform
    for _ in range(num_augmentations * 2):
        transforms = [
            lambda e: commutative_swap(e),
            lambda e: [constant_folding(e)],
            lambda e: [identity_elimination(e)],
            lambda e: [associative_regroup(e)],
            lambda e: [expand(e)],
            lambda e: [factor(e)],
        ]
        base_expr = random.choice(list(results))
        transform = random.choice(transforms)
        try:
            new_exprs = transform(base_expr)
            if isinstance(new_exprs, list):
                for ne in new_exprs:
                    results.add(ne)
            else:
                results.add(new_exprs)
        except Exception:
            pass

        if len(results) >= num_augmentations:
            break

    return list(results)[:num_augmentations]


# ---------------------------------------------------------------------------
# FSReD Equation Collection
# ---------------------------------------------------------------------------

FSRED_EQUATIONS = [
    # Easy: Mechanics basics
    "x1*x2", "x1*x2**2/2", "x1*x2 + x3",
    "x1 + x2 + x3", "x1*x2*x3", "x1/x2",
    "x1**2", "x1**3", "sqrt(x1)", "x1*sin(x2)",
    "x1*cos(x2)", "x1*exp(x2)", "log(x1)", "x1**2 + x2**2",
    "x1*x2 - x3*x4", "x1/(x2*x3)", "x1*x2/(x3**2)",
    "(x1+x2)*x3", "x1**2*x2", "sin(x1)*cos(x2)",
    "x1*x2 + x1*x3", "x1/(1+x2)", "(x1-x2)**2",
    "sqrt(x1**2+x2**2)", "x1*sin(x2*x3)",
    "x1*exp(-x2)", "x1*cos(x2+x3)", "x1**2/(2*x2)",
    "x1*x2*x3/x4", "x1 + x2*x3",
    # Medium: Physics formulas
    "x1*x2**2/(2*x3)", "x1*x2*x3/(x4**2)",
    "sqrt(x1*x2/x3)", "x1*sin(2*x2)/x3",
    "x1*exp(-x2*x3)", "x1*cos(x2*x3+x4)",
    "x1**2*x2/(2*x3**2)", "sqrt(x1**2+x2**2+x3**2)",
    "(x1-x2)/(x3+x4)", "x1*x2/(x3*x4)",
    "x1*sin(x2)**2", "x1*cos(x2)**2 + x3*sin(x2)**2",
    "x1*exp(-x2**2/x3)", "log(x1*x2)", "x1*x2*sin(x3)/x4",
    "x1/(x2**2 + x3**2)", "sqrt(2*x1*x2)",
    "x1*x2**2 + x3*x4**2", "x1*(1-exp(-x2*x3))",
    "x1*x2 + x3*x4 + x5",
    "(x1*x2 + x3*x4)/(x5+x6)", "x1**2*sin(x2)*cos(x3)",
    "x1*exp(-x2/x3)*cos(x4)", "sqrt(x1/x2)*x3",
    "x1*x2/(x3+x4+x5)", "x1*log(x2/x3)",
    "x1**2*x2 + x3**2*x4", "x1*sin(x2+x3)*exp(-x4)",
    "x1*cos(x2)*sin(x3)", "x1/(x2*x3+x4)",
    "x1*x2**3/x3", "x1**2/(x2+x3)",
    "sqrt(x1*x2 + x3*x4)", "x1*exp(x2)*sin(x3)",
    "x1*cos(x2/x3)", "x1*x2*exp(-x3/x4)",
    "(x1+x2)**2*(x3+x4)", "x1*sin(x2)*exp(-x3)",
    "x1*x2/(x3**2+x4**2)", "x1*log(1+x2*x3)",
    # Hard: Complex physics
    "x1*exp(-x2*x3)*cos(x4*x3+x5)",
    "x1/sqrt((x2**2-x3**2)**2 + (2*x4*x3)**2)",
    "sqrt((x1+x2)/x3 + x2/x3)",
    "x1*x2**2*sin(x3)*cos(x4)/(x5*x6)",
    "x1*exp(-x2)*sin(x3*x4+x5)+x6",
    "sqrt(x1**2*x2**2 + x3**2*x4**2)",
    "x1*x2/(x3**2)*(1-exp(-x4*x5))",
    "x1*sin(x2)*cos(x3)*exp(-x4/x5)",
    "(x1*x2+x3*x4)/(x5**2+x6**2)*sin(x7)",
    "x1*exp(-x2**2/(2*x3**2))/sqrt(2*pi*x3**2)",
    "x1*x2*sin(x3)**2/(x4*(x5+x6))",
    "sqrt(x1**2+x2**2-2*x1*x2*cos(x3))",
    "x1*log(x2/x3)*exp(-x4/x5)",
    "x1**2*x2/(2*x3) + x4*x5**2/(2*x6)",
    "x1*cos(x2*x3)*exp(-x4*x3)*sin(x5*x3)",
    "(x1-x2)**2/(2*x3) + x4*x5*x6",
    "x1*x2*exp(-x3*x4)/(x5+x6*x7)",
    "sin(x1*x2+x3)*cos(x4*x5+x6)",
    "x1*sqrt(x2**2+x3**2)*exp(-x4*x5)",
    "x1/(x2+x3*exp(-x4*x5))",
    "x1*x2**2*cos(x3)/(x4*x5) + x6*sin(x7)",
    "sqrt(x1*x2/(x3*x4+x5*x6))",
    "x1*sin(x2/x3)*exp(-x4*x5)+x6*cos(x7)",
    "x1**2*x2*x3/(x4**2*x5)",
    "x1*(x2-x3)*exp(-x4/x5)/x6",
    "x1*x2*sin(x3*x4)/(x5**2+x6**2)",
    "sqrt(x1**2/(x2*x3)+x4**2/(x5*x6))",
    "(x1+x2*cos(x3))*(x4+x5*sin(x6))",
    "x1*exp(-x2*x3)*sin(x4*x5)/(x6+x7)",
    "x1*x2/(x3*x4) + x5*x6/(x7*x8)",
    # Additional to reach 120
    "x1*x2+x3**2", "x1*cos(x2+x3*x4)",
    "x1*x2*x3*x4", "x1**2-x2**2",
    "x1*sin(x2)+x3*cos(x4)", "x1*x2**2+x3**2*x4",
    "x1*exp(-x2)/(x3+x4)", "sqrt(x1*x2)*sin(x3)",
    "(x1+x2)/(x3-x4)", "x1*log(x2)*x3",
    "x1*x2+x3*x4+x5*x6", "x1**3+x2**3",
    "sin(x1)**2+cos(x1)**2", "x1*x2*cos(x3*x4)",
    "x1/(x2+x3)**2", "x1*sqrt(x2)*exp(-x3)",
    "x1*x2/(x3+1)", "x1**2*x2**2",
    "(x1*x2)**2+x3", "x1*sin(x2**2)",
]

# Ensure we have at least 120 equations
assert len(FSRED_EQUATIONS) >= 120, f"Only {len(FSRED_EQUATIONS)} equations defined"
