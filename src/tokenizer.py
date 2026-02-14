#!/usr/bin/env python3
"""Physics-aware prefix-notation tokenizer for PhysMDT.

Bidirectional conversion between equation strings (prefix notation) and
token ID sequences, with a 147-token vocabulary covering operators,
functions, physics variables, constants, and structural tokens.
"""

from typing import List, Dict, Optional, Tuple
import re


# ---------------------------------------------------------------------------
# Vocabulary definition
# ---------------------------------------------------------------------------

# Build the full vocabulary list (order defines token IDs)
_VOCAB_LIST = [
    # Special tokens (0-5)
    '[PAD]', '[BOS]', '[EOS]', '[MASK]', '[SEP]', '[UNK]',
    # Arithmetic operators (6-11)
    'add', 'sub', 'mul', 'div', 'pow', 'neg',
    # Mathematical functions (12-23)
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs',
    'asin', 'acos', 'atan', 'sinh', 'cosh',
    # Physics variables (24-68)
    'x', 'y', 'z', 't',
    'v', 'v0', 'vx', 'vy', 'vz',
    'a', 'ax', 'ay', 'az',
    'F', 'Fx', 'Fy', 'Fz',
    'm', 'm1', 'm2',
    'r', 'R',
    'theta', 'phi',
    'omega', 'alpha',
    'tau', 'I_inertia', 'L_angular',
    'E_energy', 'KE', 'PE', 'W_work', 'P_power', 'p_momentum',
    'rho', 'P_pressure', 'V_volume', 'A_area',
    'h', 'l_length', 'd', 'k_spring', 'mu', 'x0',
    # Named physical constants (69-76)
    'g_accel', 'G_const', 'pi', 'euler', 'c_light', 'k_boltz', 'h_planck', 'epsilon0',
    # Integer constants (77-86)
    'INT_0', 'INT_1', 'INT_2', 'INT_3', 'INT_4',
    'INT_5', 'INT_6', 'INT_7', 'INT_8', 'INT_9',
    # Float constant tokens (87-113)
    'C_+', 'C_-',
    'D_0', 'D_1', 'D_2', 'D_3', 'D_4', 'D_5', 'D_6', 'D_7', 'D_8', 'D_9',
    'DOT',
    'E_+', 'E_-',
    'E_0', 'E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'E_6', 'E_7', 'E_8', 'E_9',
    'CONST_START', 'CONST_END',
    # Structure tokens (114-146)
    'OP_BINARY', 'OP_UNARY', 'LEAF_VAR', 'LEAF_CONST', 'LEAF_INT', 'LEAF_NAMED',
    'SKEL_add', 'SKEL_sub', 'SKEL_mul', 'SKEL_div', 'SKEL_pow', 'SKEL_neg',
    'SKEL_sin', 'SKEL_cos', 'SKEL_tan', 'SKEL_exp', 'SKEL_log', 'SKEL_sqrt',
    'DEPTH_0', 'DEPTH_1', 'DEPTH_2', 'DEPTH_3', 'DEPTH_4',
    'DEPTH_5', 'DEPTH_6', 'DEPTH_7', 'DEPTH_8', 'DEPTH_9',
    'STRUCT_PAD', 'STRUCT_BOS', 'STRUCT_EOS', 'STRUCT_MASK',
]

# Operator arities
OPERATOR_ARITIES = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2,
    'neg': 1,
    'sin': 1, 'cos': 1, 'tan': 1, 'exp': 1, 'log': 1, 'sqrt': 1, 'abs': 1,
    'asin': 1, 'acos': 1, 'atan': 1, 'sinh': 1, 'cosh': 1,
}


class PhysicsTokenizer:
    """Tokenizer for physics equations in prefix notation."""

    def __init__(self):
        self.vocab = _VOCAB_LIST
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # Special token IDs
        self.pad_id = self.token_to_id['[PAD]']
        self.bos_id = self.token_to_id['[BOS]']
        self.eos_id = self.token_to_id['[EOS]']
        self.mask_id = self.token_to_id['[MASK]']
        self.unk_id = self.token_to_id['[UNK]']

    def tokenize(self, prefix_str: str) -> List[str]:
        """Split a prefix notation string into tokens."""
        return prefix_str.strip().split()

    def encode(self, prefix_str: str, add_special: bool = True,
               max_length: Optional[int] = None) -> List[int]:
        """Convert prefix notation string to token IDs.

        Args:
            prefix_str: Space-separated prefix notation string
            add_special: If True, prepend [BOS] and append [EOS]
            max_length: If set, pad/truncate to this length

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(prefix_str)
        ids = []
        if add_special:
            ids.append(self.bos_id)

        for tok in tokens:
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])
            else:
                ids.append(self.unk_id)

        if add_special:
            ids.append(self.eos_id)

        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length - 1] + [self.eos_id]
            else:
                ids.extend([self.pad_id] * (max_length - len(ids)))

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token IDs back to prefix notation string.

        Args:
            ids: List of token IDs
            skip_special: If True, skip [PAD], [BOS], [EOS], [MASK]

        Returns:
            Space-separated prefix notation string
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        tokens = []
        for id_val in ids:
            if id_val in self.id_to_token:
                tok = self.id_to_token[id_val]
                if skip_special and id_val in special_ids:
                    continue
                tokens.append(tok)
            else:
                tokens.append('[UNK]')
        return ' '.join(tokens)

    def get_tree_depths(self, prefix_str: str) -> List[int]:
        """Compute tree depth for each token in a prefix notation string.

        Returns list of depths (0 = root), same length as tokenized sequence.
        """
        tokens = self.tokenize(prefix_str)
        depths = []
        stack = []  # stack of (remaining_args, depth)

        for tok in tokens:
            if stack:
                current_depth = stack[-1][1]
            else:
                current_depth = 0

            if tok in OPERATOR_ARITIES:
                depths.append(current_depth)
                arity = OPERATOR_ARITIES[tok]
                stack.append([arity, current_depth + 1])
            else:
                depths.append(current_depth)
                # This is a leaf; pop completed operators
                while stack:
                    stack[-1][0] -= 1
                    if stack[-1][0] == 0:
                        stack.pop()
                    else:
                        break

        return depths

    def prefix_to_infix(self, prefix_str: str) -> str:
        """Convert prefix notation to human-readable infix notation."""
        tokens = self.tokenize(prefix_str)
        stack = []
        pos = len(tokens) - 1

        def _parse(idx):
            if idx < 0:
                return "?", idx
            tok = tokens[idx]
            if tok not in OPERATOR_ARITIES:
                return tok, idx - 1
            arity = OPERATOR_ARITIES[tok]
            if arity == 1:
                arg, new_idx = _parse(idx - 1)
                return f"{tok}({arg})", new_idx
            elif arity == 2:
                # In prefix: op arg1 arg2
                # We need to parse forward, not backward
                pass
            return tok, idx - 1

        # Forward recursive parser
        self._parse_idx = 0
        self._tokens = tokens

        def parse_forward():
            if self._parse_idx >= len(self._tokens):
                return "?"
            tok = self._tokens[self._parse_idx]
            self._parse_idx += 1
            if tok not in OPERATOR_ARITIES:
                return tok
            arity = OPERATOR_ARITIES[tok]
            if arity == 1:
                arg = parse_forward()
                return f"{tok}({arg})"
            elif arity == 2:
                left = parse_forward()
                right = parse_forward()
                op_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
                op_sym = op_map.get(tok, tok)
                return f"({left} {op_sym} {right})"
            return tok

        result = parse_forward()
        return result

    def encode_skeleton(self, prefix_str: str) -> List[int]:
        """Encode just the operator skeleton (replacing leaves with placeholders)."""
        tokens = self.tokenize(prefix_str)
        skel_ids = [self.token_to_id['STRUCT_BOS']]

        for tok in tokens:
            if tok in OPERATOR_ARITIES:
                skel_key = f'SKEL_{tok}'
                if skel_key in self.token_to_id:
                    skel_ids.append(self.token_to_id[skel_key])
                elif OPERATOR_ARITIES[tok] == 2:
                    skel_ids.append(self.token_to_id['OP_BINARY'])
                else:
                    skel_ids.append(self.token_to_id['OP_UNARY'])
            elif tok.startswith('INT_'):
                skel_ids.append(self.token_to_id['LEAF_INT'])
            elif tok in ('pi', 'euler', 'g_accel', 'G_const', 'c_light',
                         'k_boltz', 'h_planck', 'epsilon0'):
                skel_ids.append(self.token_to_id['LEAF_NAMED'])
            elif tok in ('CONST_START', 'CONST_END', 'DOT') or tok.startswith(('C_', 'D_', 'E_')):
                # Skip float constant internal tokens in skeleton
                continue
            else:
                skel_ids.append(self.token_to_id['LEAF_VAR'])

        skel_ids.append(self.token_to_id['STRUCT_EOS'])
        return skel_ids

    def batch_encode(self, prefix_strs: List[str], max_length: int = 128) -> List[List[int]]:
        """Encode a batch of prefix strings with padding."""
        return [self.encode(s, max_length=max_length) for s in prefix_strs]

    def batch_decode(self, id_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(ids) for ids in id_batch]
