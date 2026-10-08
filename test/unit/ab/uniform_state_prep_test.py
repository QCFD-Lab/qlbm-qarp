import numpy as np
import pytest

from qlbm.components.common.primitives import UniformStatePrep
from test.builders import CircuitBuilder

from .qarp_helpers import simulate_statevector


@pytest.mark.parametrize(
    "num_states",
    list(range(1, 10)),
)
def test_uniform_State_prep(num_states):
    nq = 5
    builder = CircuitBuilder(nq)
    builder.compose(UniformStatePrep(nq, num_states))

    state = simulate_statevector(builder)

    expected = 1.0 / np.sqrt(num_states)

    assert np.allclose(np.abs(state)[:num_states], expected, atol=1e-8), (
        "Uniform state prep results in wrong magnitudes for the first k basis states"
    )
    assert np.allclose(np.abs(state)[num_states:], 0.0, atol=1e-8), (
        "Uniform state prep results in wrong magnitudes for the trailing basis states"
    )
