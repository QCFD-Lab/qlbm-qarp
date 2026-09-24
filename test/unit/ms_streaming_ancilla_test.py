"""Zero-width velocity registers in :class:`.StreamingAncillaPreparation`."""

import numpy as np

from qlbm.components.ms import StreamingAncillaPreparation
from qlbm.lattice import MSLattice


def two_velocity_lattice() -> MSLattice:
    """A 4x4 lattice with 2 velocities per dimension: no velocity qubits."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 2, "y": 2}},
            "geometry": [],
        }
    )


def test_velocity_register_has_zero_width():
    """Two velocities per dimension leave only the direction qubit."""
    lattice = two_velocity_lattice()

    assert lattice.velocity_index(0) == []
    assert lattice.velocity_index(1) == []


def test_preparation_flips_the_streaming_ancilla():
    """With nothing to control on, the ancilla of the dimension is set outright."""
    lattice = two_velocity_lattice()
    for dim in (0, 1):
        ancilla = lattice.ancillae_velocity_index(dim)[0]
        block = StreamingAncillaPreparation(lattice, velocities=[0], dim=dim)

        size = 1 << lattice.n_qubits
        expected = np.zeros((size, size))
        for basis in range(size):
            expected[basis ^ (1 << ancilla), basis] = 1.0
        np.testing.assert_allclose(
            np.asarray(block.unitary_matrix()), expected, atol=1e-12
        )
