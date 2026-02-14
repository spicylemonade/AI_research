"""Prefix-notation tokenizer for symbolic expressions."""

import re
from typing import List, Optional, Tuple

# Token vocabulary (73 tokens, well under 128 limit)
OPERATORS = ['add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'sin', 'cos',
             'exp', 'log', 'neg', 'abs', 'inv']
VARIABLES = [f'x_{i}' for i in range(10)]
CONSTANTS = [f'c_{i}' for i in range(10)]
INT_CONSTANTS = [f'int_{i}' for i in range(6)]
SPECIAL_CONSTANTS = ['pi', 'e_const', 'half', 'third', 'quarter']
CONTROL = ['<SOS>', '<EOS>', '<PAD>', '<MASK>']

ALL_TOKENS = CONTROL + OPERATORS + VARIABLES + CONSTANTS + INT_CONSTANTS + SPECIAL_CONSTANTS

TOKEN_TO_ID = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {i: tok for i, tok in enumerate(ALL_TOKENS)}

PAD_ID = TOKEN_TO_ID['<PAD>']
SOS_ID = TOKEN_TO_ID['<SOS>']
EOS_ID = TOKEN_TO_ID['<EOS>']
MASK_ID = TOKEN_TO_ID['<MASK>']

VOCAB_SIZE = len(ALL_TOKENS)
MAX_SEQ_LEN = 64

# Arity of each operator
ARITY = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2,
    'sqrt': 1, 'sin': 1, 'cos': 1, 'exp': 1, 'log': 1,
    'neg': 1, 'abs': 1, 'inv': 1,
}


class ExprNode:
    """Expression tree node."""
    def __init__(self, token: str, children: Optional[List['ExprNode']] = None):
        self.token = token
        self.children = children or []

    def to_prefix(self) -> List[str]:
        """Convert tree to prefix (Polish) notation token list."""
        result = [self.token]
        for child in self.children:
            result.extend(child.to_prefix())
        return result

    def to_infix(self) -> str:
        """Convert tree to human-readable infix notation."""
        if not self.children:
            return self.token
        if len(self.children) == 1:
            child_str = self.children[0].to_infix()
            return f'{self.token}({child_str})'
        left = self.children[0].to_infix()
        right = self.children[1].to_infix()
        op_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
        op_sym = op_map.get(self.token, self.token)
        return f'({left} {op_sym} {right})'

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def num_nodes(self) -> int:
        return 1 + sum(c.num_nodes() for c in self.children)

    def num_operators(self) -> int:
        count = 1 if self.token in ARITY else 0
        return count + sum(c.num_operators() for c in self.children)

    def get_variables(self) -> set:
        if self.token in VARIABLES:
            return {self.token}
        result = set()
        for c in self.children:
            result.update(c.get_variables())
        return result


def prefix_to_tree(tokens: List[str]) -> Tuple[ExprNode, int]:
    """Parse prefix token list into an expression tree. Returns (node, next_index)."""
    if not tokens:
        raise ValueError("Empty token list")

    token = tokens[0]

    if token in ARITY:
        arity = ARITY[token]
        children = []
        idx = 1
        for _ in range(arity):
            child, idx = prefix_to_tree(tokens[idx:])
            children.append(child)
            # idx is relative to the slice, need to accumulate
        # Re-parse properly with absolute indexing
        node = ExprNode(token)
        pos = 1
        for _ in range(arity):
            child, consumed = _parse_prefix(tokens, pos)
            node.children.append(child)
            pos = consumed
        return node, pos
    else:
        # Leaf node (variable, constant, etc.)
        return ExprNode(token), 1


def _parse_prefix(tokens: List[str], pos: int) -> Tuple[ExprNode, int]:
    """Parse prefix notation starting at position pos. Returns (node, next_pos)."""
    if pos >= len(tokens):
        raise ValueError(f"Unexpected end of tokens at position {pos}")

    token = tokens[pos]

    if token in ARITY:
        arity = ARITY[token]
        node = ExprNode(token)
        current_pos = pos + 1
        for _ in range(arity):
            child, current_pos = _parse_prefix(tokens, current_pos)
            node.children.append(child)
        return node, current_pos
    else:
        return ExprNode(token), pos + 1


def encode(prefix_tokens: List[str], max_len: int = MAX_SEQ_LEN) -> List[int]:
    """Encode prefix token list to integer IDs with SOS/EOS/PAD."""
    ids = [SOS_ID]
    for tok in prefix_tokens:
        if tok in TOKEN_TO_ID:
            ids.append(TOKEN_TO_ID[tok])
        else:
            raise ValueError(f"Unknown token: {tok}")
    ids.append(EOS_ID)

    # Pad to max_len
    if len(ids) < max_len:
        ids.extend([PAD_ID] * (max_len - len(ids)))
    elif len(ids) > max_len:
        ids = ids[:max_len - 1] + [EOS_ID]

    return ids


def decode(ids: List[int]) -> List[str]:
    """Decode integer IDs back to prefix token list (stripping SOS/EOS/PAD)."""
    tokens = []
    for i in ids:
        tok = ID_TO_TOKEN.get(i, '<UNK>')
        if tok == '<SOS>':
            continue
        if tok == '<EOS>':
            break
        if tok == '<PAD>':
            continue
        if tok == '<MASK>':
            continue
        tokens.append(tok)
    return tokens


def tree_to_sympy(node: ExprNode):
    """Convert expression tree to SymPy expression."""
    import sympy as sp

    if node.token in VARIABLES:
        idx = int(node.token.split('_')[1])
        return sp.Symbol(f'x{idx}')
    elif node.token in CONSTANTS:
        idx = int(node.token.split('_')[1])
        return sp.Symbol(f'C{idx}')
    elif node.token.startswith('int_'):
        return sp.Integer(int(node.token.split('_')[1]))
    elif node.token == 'pi':
        return sp.pi
    elif node.token == 'e_const':
        return sp.E
    elif node.token == 'half':
        return sp.Rational(1, 2)
    elif node.token == 'third':
        return sp.Rational(1, 3)
    elif node.token == 'quarter':
        return sp.Rational(1, 4)

    # Operators
    children_sp = [tree_to_sympy(c) for c in node.children]

    op_map = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b,
        'pow': lambda a, b: a ** b,
        'sqrt': lambda a: sp.sqrt(a),
        'sin': lambda a: sp.sin(a),
        'cos': lambda a: sp.cos(a),
        'exp': lambda a: sp.exp(a),
        'log': lambda a: sp.log(a),
        'neg': lambda a: -a,
        'abs': lambda a: sp.Abs(a),
        'inv': lambda a: 1 / a,
    }

    if node.token in op_map:
        return op_map[node.token](*children_sp)
    raise ValueError(f"Unknown operator: {node.token}")
