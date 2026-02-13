"""Unit tests for the symbolic equation tokenizer."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.tokenizer import (
    infix_to_prefix, prefix_to_infix, encode, decode, get_tree_depths,
    round_trip_test, VOCAB_SIZE, ALL_TOKENS, TOKEN_TO_IDX, IDX_TO_TOKEN,
    PAD_IDX, BOS_IDX, EOS_IDX, MASK_IDX, SEP_IDX,
    OPERATOR_ARITY, MAX_SEQ_LEN,
)


class TestVocabulary:
    def test_vocab_size_minimum(self):
        assert VOCAB_SIZE >= 150, f"Vocab size {VOCAB_SIZE} < 150 minimum"

    def test_token_index_bijection(self):
        for tok, idx in TOKEN_TO_IDX.items():
            assert IDX_TO_TOKEN[idx] == tok

    def test_special_tokens_exist(self):
        for tok in ['<BOS>', '<EOS>', '<PAD>', '<MASK>', '<SEP>']:
            assert tok in TOKEN_TO_IDX

    def test_arithmetic_operators_exist(self):
        for op in ['+', '-', '*', '/', '^', 'neg']:
            assert op in TOKEN_TO_IDX

    def test_trig_functions_exist(self):
        for fn in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            assert fn in TOKEN_TO_IDX

    def test_physics_vars_exist(self):
        for var in ['m', 'g', 'F', 'E', 'v', 'a', 't', 'r', 'omega', 'theta']:
            assert var in TOKEN_TO_IDX

    def test_numeric_constants_exist(self):
        for c in ['0', '1', '2', 'pi', '0.5']:
            assert c in TOKEN_TO_IDX


class TestInfixToPrefix:
    def test_simple_product(self):
        tokens = infix_to_prefix("m * a")
        assert set(tokens) == {'*', 'a', 'm'}
        assert tokens[0] == '*'

    def test_power(self):
        tokens = infix_to_prefix("v**2")
        assert tokens == ['^', 'v', '2']

    def test_sin(self):
        tokens = infix_to_prefix("sin(x)")
        assert tokens == ['sin', 'x']

    def test_nested(self):
        tokens = infix_to_prefix("sin(omega * t)")
        assert tokens[0] == 'sin'
        assert '*' in tokens


class TestPrefixToInfix:
    def test_simple(self):
        result = prefix_to_infix(['*', 'a', 'm'])
        assert 'a' in result and 'm' in result

    def test_nested_ops(self):
        result = prefix_to_infix(['+', '*', 'a', 'b', 'c'])
        assert result  # Should not be empty


class TestEncodeDecode:
    def test_encode_length(self):
        indices = encode("m * a")
        assert len(indices) == MAX_SEQ_LEN

    def test_encode_starts_with_bos(self):
        indices = encode("m * a")
        assert indices[0] == BOS_IDX

    def test_encode_has_eos(self):
        indices = encode("m * a")
        assert EOS_IDX in indices

    def test_decode_recovers_vars(self):
        decoded = decode(encode("m * a"))
        assert 'a' in decoded and 'm' in decoded

    def test_padding(self):
        indices = encode("x")
        # Most of the sequence should be padding
        pad_count = indices.count(PAD_IDX)
        assert pad_count > MAX_SEQ_LEN - 10


class TestRoundTrip:
    @pytest.mark.parametrize("equation", [
        "m * a",
        "0.5 * m * v**2",
        "G_const * m * M / r**2",
        "sin(omega * t + phi)",
        "sqrt(2 * g * h)",
        "m * g * h + 0.5 * m * v**2",
    ])
    def test_round_trip(self, equation):
        assert round_trip_test(equation), f"Round-trip failed for: {equation}"


class TestTreeDepths:
    def test_single_leaf(self):
        depths = get_tree_depths(['x'])
        assert depths == [0]

    def test_binary_op(self):
        # * a b -> depths: *, a, b = 0, 1, 1
        depths = get_tree_depths(['*', 'a', 'b'])
        assert depths == [0, 1, 1]

    def test_unary_op(self):
        # sin x -> depths: sin, x = 0, 1
        depths = get_tree_depths(['sin', 'x'])
        assert depths == [0, 1]

    def test_nested(self):
        # + * a b c -> depths: +, *, a, b, c = 0, 1, 2, 2, 1
        depths = get_tree_depths(['+', '*', 'a', 'b', 'c'])
        assert depths == [0, 1, 2, 2, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
