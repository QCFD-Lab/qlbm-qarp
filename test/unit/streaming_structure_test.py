"""Each dimension's QFT shift is one block, and that block is a permutation."""

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.ab import ABStreamingOperator
from qlbm.components.ms import ControlledIncrementer
from qlbm.lattice import ABLattice, MSLattice, OHLattice

AB_LATTICES = [
    (ABLattice, {"x": 8}, "d1q3"),
    (ABLattice, {"x": 4, "y": 4}, "d2q9"),
    (ABLattice, {"x": 8, "y": 4}, "d2q9"),
    (OHLattice, {"x": 4, "y": 4}, "d2q9"),
]

MS_LATTICES = [
    ({"x": 4, "y": 4}, None),
    ({"x": 8, "y": 4}, "bounceback"),
    ({"x": 2, "y": 2, "z": 2}, None),
]


def maps_basis_states_to_basis_states(block) -> bool:
    """Whether ``block`` sends basis states to basis states with amplitude 1."""
    commands, n = list(block.flatten()), block.n_qubits
    simulator = qx.QarpSimulator()
    # Every basis state on small registers, a fixed sample of 512 above.
    indices = (
        range(1 << n)
        if n <= 11
        else np.random.default_rng(0).choice(1 << n, 512, replace=False)
    )
    for index in indices:
        state = np.zeros(1 << n, dtype=np.complex128)
        state[index] = 1
        out = np.asarray(simulator.statevector(commands, n, initial_state=state))
        if not np.isclose(out[np.argmax(np.abs(out))], 1, atol=1e-9, rtol=0):
            return False
    return True


@pytest.mark.parametrize("lattice_class,dims,velocities", AB_LATTICES)
def test_ab_streaming_has_one_permutation_block_per_dimension(
    lattice_class, dims, velocities
):
    """QFT, phase shifts and inverse QFT of a dimension form one child."""
    lattice = lattice_class(
        {"lattice": {"dim": dims, "velocities": velocities}, "geometry": []}
    )

    children = list(ABStreamingOperator(lattice, []).children())

    assert len(children) == lattice.num_dims
    assert all(maps_basis_states_to_basis_states(child) for child in children)


@pytest.mark.parametrize("dims,reflection", MS_LATTICES)
def test_ms_incrementer_has_one_permutation_block_per_dimension(dims, reflection):
    """QFT, the two phase shifts and inverse QFT of a dimension form one child."""
    geometry = (
        [{"shape": "cuboid", "x": [1, 2], "y": [1, 2], "boundary": reflection}]
        if reflection
        else []
    )
    lattice = MSLattice(
        {
            "lattice": {"dim": dims, "velocities": {axis: 4 for axis in dims}},
            "geometry": geometry,
        }
    )

    children = list(ControlledIncrementer(lattice, reflection=reflection).children())

    assert len(children) == lattice.num_dims
    assert all(maps_basis_states_to_basis_states(child) for child in children)
