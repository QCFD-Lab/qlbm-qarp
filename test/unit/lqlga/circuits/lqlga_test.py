"""The :class:`.LQLGA` time step is only defined where reflection is."""

import pytest

from qlbm.components.lqlga import (
    LQLGA,
    LQLGAMGReflectionOperator,
    LQLGAReflectionOperator,
)
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.tools.exceptions import CircuitException

UNSUPPORTED = [
    ({"x": 2, "y": 2}, "D2Q4"),
    ({"x": 2, "y": 1, "z": 1}, "D3Q6"),
]


def lattice_without_geometry(dims, velocities) -> LQLGALattice:
    """An LQLGA lattice with no shapes."""
    return LQLGALattice(
        {"lattice": {"dim": dims, "velocities": velocities}, "geometry": []}
    )


@pytest.mark.parametrize("dims,velocities", UNSUPPORTED)
def test_time_step_without_geometry_is_rejected(dims, velocities):
    """Above one dimension the time step is rejected even with nothing to reflect off."""
    with pytest.raises(CircuitException, match="Reflection Operator unsupported"):
        LQLGA(lattice_without_geometry(dims, velocities))


@pytest.mark.parametrize("dims,velocities", UNSUPPORTED)
def test_reflection_without_shapes_is_rejected(dims, velocities):
    """Both reflection operators check the discretization before their shapes."""
    lattice = lattice_without_geometry(dims, velocities)

    with pytest.raises(CircuitException, match="Reflection Operator unsupported"):
        LQLGAReflectionOperator(lattice, [])
    with pytest.raises(CircuitException, match="Reflection Operator unsupported"):
        LQLGAMGReflectionOperator(lattice, [[], []])


def test_d2q4_step_with_geometry_is_rejected():
    """Reflection is only defined for D1Q2 and D1Q3."""
    lattice = LQLGALattice(
        {
            "lattice": {"dim": {"x": 2, "y": 2}, "velocities": "D2Q4"},
            "geometry": [
                {"shape": "cuboid", "x": [1, 1], "y": [1, 1], "boundary": "bounceback"}
            ],
        }
    )

    with pytest.raises(CircuitException, match="Reflection Operator unsupported"):
        LQLGA(lattice)


@pytest.mark.parametrize("velocities", ["D1Q2", "D1Q3"])
def test_one_dimensional_step_without_geometry_builds(velocities):
    """In one dimension a lattice without shapes still has a time step."""
    lattice = lattice_without_geometry({"x": 4}, velocities)

    assert LQLGA(lattice).n_qubits == lattice.n_qubits
