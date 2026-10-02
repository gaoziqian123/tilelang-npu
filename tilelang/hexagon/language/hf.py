"""Explicit HF rounding boundaries; QFloat temporaries never escape a leaf.

These operations do not change the semantics of T.exp or T.reduce_sum.
The exp2_hf approximation contract is documented in docs/hexagon/hf_contract.md.
"""
from tvm import tirx
from tvm.ir import Op


def exp2_hf(x):
    if str(x.dtype).split('x')[0] != 'float16':
        raise TypeError('exp2_hf requires an explicitly rounded float16 input')
    return tirx.call_intrin(x.dtype, Op.get('tl.hexagon.exp2_hf'), x)


def sub_hf(a, b):
    """HF inputs -> QF16 subtract -> HF output (one explicit rounding)."""
    if a.dtype != b.dtype or str(a.dtype).split('x')[0] != 'float16':
        raise TypeError('sub_hf requires matching float16 operands')
    return tirx.call_intrin(a.dtype, Op.get('tl.hexagon.sub_hf'), a, b)
