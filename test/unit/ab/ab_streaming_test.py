"""``ABStreamingOperator`` and its explicit inverse."""

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.ab import ABStreamingOperator
from qlbm.lattice import ABLattice, OHLattice


def apply(block, state: np.ndarray) -> np.ndarray:
    """``state`` after the block's gates."""
    return np.asarray(
        qx.QarpSimulator().statevector(
            list(block.flatten()), block.n_qubits, initial_state=state
        )
    )


@pytest.mark.parametrize(
    "lattice_class,dims,velocities",
    [
        (ABLattice, {"x": 8}, "d1q3"),
        (ABLattice, {"x": 4, "y": 4}, "d2q9"),
        (OHLattice, {"x": 4, "y": 4}, "d2q9"),
    ],
)
def test_inverse_streaming_is_the_dagger_of_streaming(lattice_class, dims, velocities):
    """``inverse=True`` undoes streaming: it acts as the daggered operator."""
    lattice = lattice_class(
        {"lattice": {"dim": dims, "velocities": velocities}, "geometry": []}
    )
    rng = np.random.default_rng(3)
    state = rng.standard_normal(1 << lattice.n_qubits) + 1j * rng.standard_normal(
        1 << lattice.n_qubits
    )
    state /= np.linalg.norm(state)

    inverse = apply(ABStreamingOperator(lattice, inverse=True), state)
    dagger = apply(~ABStreamingOperator(lattice), state)

    np.testing.assert_allclose(inverse, dagger, atol=1e-9, rtol=0)
