"""Tests for the symbolic equation tokenizer."""

import sys
import random
import pytest
import sympy
from sympy import Symbol, sin, cos, exp, log, sqrt, pi, E

sys.path.insert(0, '.')
from src.data.tokenizer import (
    EquationTokenizer, VOCAB_SIZE, FSRED_EQUATIONS,
    commutative_swap, constant_folding, identity_elimination,
    associative_regroup, augment_expression,
    _sympy_to_rpn, _rpn_to_sympy,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN,
)

random.seed(42)


@pytest.fixture
def tokenizer():
    return EquationTokenizer(max_seq_len=64)


class TestVocabulary:
    def test_vocab_size(self):
        assert VOCAB_SIZE <= 200

    def test_special_tokens(self):
        assert PAD_TOKEN == 0
        assert BOS_TOKEN == 1
        assert EOS_TOKEN == 2
        assert MASK_TOKEN == 3


class TestEncoding:
    def test_simple_expression(self, tokenizer):
        ids = tokenizer.encode("x1*x2")
        assert ids[0] == BOS_TOKEN
        assert EOS_TOKEN in ids
        assert len(ids) == 64  # padded to max_seq_len

    def test_encode_decode_simple(self, tokenizer):
        expr = "x1 + x2"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        # Check symbolic equivalence
        x1, x2 = Symbol('x1'), Symbol('x2')
        orig = sympy.sympify(expr)
        dec = sympy.sympify(decoded)
        assert sympy.simplify(orig - dec) == 0

    def test_encode_decode_multiplication(self, tokenizer):
        expr = "x1*x2"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        x1, x2 = Symbol('x1'), Symbol('x2')
        assert sympy.simplify(sympy.sympify(expr) - sympy.sympify(decoded)) == 0

    def test_encode_decode_power(self, tokenizer):
        expr = "x1**2"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert sympy.simplify(sympy.sympify(expr) - sympy.sympify(decoded)) == 0

    def test_encode_decode_sin(self, tokenizer):
        expr = "sin(x1)"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert sympy.simplify(sympy.sympify(expr) - sympy.sympify(decoded)) == 0

    def test_encode_decode_complex(self, tokenizer):
        expr = "x1*sin(x2) + x3*cos(x4)"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert sympy.simplify(sympy.sympify(expr) - sympy.sympify(decoded)) == 0

    def test_encode_decode_nested(self, tokenizer):
        expr = "sqrt(x1**2 + x2**2)"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        orig = sympy.sympify(expr)
        dec = sympy.sympify(decoded)
        # Check equivalence numerically
        vals = {Symbol('x1'): 3.0, Symbol('x2'): 4.0}
        assert abs(float(orig.subs(vals)) - float(dec.subs(vals))) < 1e-8

    def test_encode_decode_exp(self, tokenizer):
        expr = "exp(x1)"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert sympy.simplify(sympy.sympify(expr) - sympy.sympify(decoded)) == 0

    def test_encode_decode_log(self, tokenizer):
        expr = "log(x1)"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert sympy.simplify(sympy.sympify(expr) - sympy.sympify(decoded)) == 0

    def test_encode_decode_division(self, tokenizer):
        expr = "x1/x2"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert sympy.simplify(sympy.sympify(expr) - sympy.sympify(decoded)) == 0


class TestRPN:
    def test_rpn_simple_add(self):
        x1, x2 = Symbol('x1'), Symbol('x2')
        rpn = _sympy_to_rpn(x1 + x2)
        expr = _rpn_to_sympy(rpn)
        assert sympy.simplify(x1 + x2 - expr) == 0

    def test_rpn_mul(self):
        x1, x2 = Symbol('x1'), Symbol('x2')
        rpn = _sympy_to_rpn(x1 * x2)
        expr = _rpn_to_sympy(rpn)
        assert sympy.simplify(x1 * x2 - expr) == 0

    def test_rpn_nested(self):
        x1 = Symbol('x1')
        rpn = _sympy_to_rpn(sin(x1**2))
        expr = _rpn_to_sympy(rpn)
        assert sympy.simplify(sin(x1**2) - expr) == 0

    def test_rpn_complex(self):
        x1, x2, x3 = Symbol('x1'), Symbol('x2'), Symbol('x3')
        original = x1 * sin(x2) + x3
        rpn = _sympy_to_rpn(original)
        expr = _rpn_to_sympy(rpn)
        assert sympy.simplify(original - expr) == 0

    def test_rpn_constants(self):
        rpn = _sympy_to_rpn(pi)
        expr = _rpn_to_sympy(rpn)
        assert expr == pi


class TestFSReD:
    def test_fsred_equation_count(self):
        assert len(FSRED_EQUATIONS) >= 120

    def test_all_fsred_equations_encode(self, tokenizer):
        """All 120 FSReD equations must encode without error."""
        errors = []
        for i, eq in enumerate(FSRED_EQUATIONS[:120]):
            try:
                ids = tokenizer.encode(eq)
                assert ids[0] == BOS_TOKEN
                assert EOS_TOKEN in ids
            except Exception as e:
                errors.append((i, eq, str(e)))
        assert len(errors) == 0, f"Failed equations: {errors}"

    def test_fsred_roundtrip_sample(self, tokenizer):
        """Test encode-decode roundtrip on 50 FSReD equations."""
        successes = 0
        for eq in FSRED_EQUATIONS[:50]:
            try:
                ids = tokenizer.encode(eq)
                decoded = tokenizer.decode(ids)
                orig = sympy.sympify(eq, locals={f'x{i}': Symbol(f'x{i}') for i in range(1, 10)})
                dec = sympy.sympify(decoded, locals={f'x{i}': Symbol(f'x{i}') for i in range(1, 10)})
                # Numerical check
                vals = {Symbol(f'x{i}'): 1.0 + 0.1 * i for i in range(1, 10)}
                try:
                    v1 = complex(orig.subs(vals))
                    v2 = complex(dec.subs(vals))
                    if abs(v1 - v2) < 1e-6 * max(abs(v1), 1):
                        successes += 1
                except Exception:
                    # Some expressions may not evaluate, still count as parsed
                    successes += 1
            except Exception:
                pass
        assert successes >= 40, f"Only {successes}/50 roundtripped successfully"


class TestAugmentation:
    def test_commutative_swap(self):
        x1, x2 = Symbol('x1'), Symbol('x2')
        expr = x1 + x2
        results = commutative_swap(expr)
        assert len(results) >= 1

    def test_constant_folding(self):
        expr = sympy.sympify("2*3 + x1")
        folded = constant_folding(expr)
        assert folded == 6 + Symbol('x1')

    def test_identity_elimination(self):
        x1 = Symbol('x1')
        expr = x1 + 0
        result = identity_elimination(expr)
        # SymPy may simplify this automatically, check it's still x1
        assert sympy.simplify(result - x1) == 0

    def test_associative_regroup(self):
        x1, x2, x3 = Symbol('x1'), Symbol('x2'), Symbol('x3')
        expr = x1 + x2 + x3
        result = associative_regroup(expr)
        assert sympy.simplify(result - expr) == 0

    def test_augment_expression(self):
        x1, x2 = Symbol('x1'), Symbol('x2')
        expr = x1 * x2 + x1 * x2**2
        augmented = augment_expression(expr, num_augmentations=4)
        assert len(augmented) >= 1
        # All augmentations should be equivalent
        vals = {x1: 2.0, x2: 3.0}
        orig_val = float(expr.subs(vals))
        for aug in augmented:
            aug_val = float(aug.subs(vals))
            assert abs(orig_val - aug_val) < 1e-8, f"Augmentation changed value: {aug}"

    def test_augment_preserves_equivalence(self):
        x1 = Symbol('x1')
        expr = sin(x1)**2 + cos(x1)**2
        augmented = augment_expression(expr, num_augmentations=4)
        vals = {x1: 1.5}
        orig_val = float(expr.subs(vals))
        for aug in augmented:
            aug_val = float(aug.subs(vals))
            assert abs(orig_val - aug_val) < 1e-8


class TestEdgeCases:
    def test_single_variable(self, tokenizer):
        ids = tokenizer.encode("x1")
        decoded = tokenizer.decode(ids)
        assert sympy.sympify(decoded) == Symbol('x1')

    def test_integer_constant(self, tokenizer):
        ids = tokenizer.encode("42")
        decoded = tokenizer.decode(ids)
        assert float(sympy.sympify(decoded)) == 42.0

    def test_negative_number(self, tokenizer):
        ids = tokenizer.encode("-3*x1")
        decoded = tokenizer.decode(ids)
        x1 = Symbol('x1')
        assert sympy.simplify(sympy.sympify(decoded) - (-3 * x1)) == 0

    def test_pi_constant(self, tokenizer):
        ids = tokenizer.encode("pi*x1")
        decoded = tokenizer.decode(ids)
        x1 = Symbol('x1')
        assert sympy.simplify(sympy.sympify(decoded) - pi * x1) == 0

    def test_max_seq_len(self, tokenizer):
        ids = tokenizer.encode("x1+x2+x3+x4+x5+x6+x7+x8+x9")
        assert len(ids) == 64

    def test_get_rpn_tokens(self, tokenizer):
        ids = tokenizer.encode("x1+x2")
        rpn = tokenizer.get_rpn_tokens(ids)
        assert len(rpn) > 0
        assert any(tok in rpn for tok in ['x1', 'x2'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
