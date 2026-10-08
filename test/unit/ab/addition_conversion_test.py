from itertools import product

import pytest

from qlbm.components.common.primitives import AdditionConversion
from qlbm.tools.utils import bit_value
from test.builders import CircuitBuilder

from .qarp_helpers import sample_register_counts


@pytest.mark.parametrize(
    "nq,state_in,state_out", list(product([4], [1, 4, 7, 11, 14], [0, 2, 8, 9, 12]))
)
def test_addition_conversion(nq, state_in, state_out):
    builder = CircuitBuilder(nq + 1)
    for q in range(nq):
        if bit_value(state_in, q):
            builder.x(q)

    builder.compose(AdditionConversion(nq, state_in, state_out))

    counts = sample_register_counts(builder, range(nq + 1), shots=128)

    assert all(outcome == state_out for outcome in counts), (
        f"{state_in} handled incorrectly. Expected {state_out}, got {counts}."
    )
