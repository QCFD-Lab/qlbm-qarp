"""A time step gives the same state through qarp's structured execution as through its gates."""

import numpy as np
import pytest

from qlbm.components import CQLBM, MSQLBM
from qlbm.components.ab import ABQLBM
from qlbm.components.common import StreamingShift
from qlbm.lattice import ABLattice, MSLattice

CUBOID = {"shape": "cuboid", "x": [5, 8], "y": [2, 5]}


def ab_lattice(boundary: str) -> ABLattice:
    """A 16x8 D2Q9 lattice with one obstacle of the given boundary condition."""
    return ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            "geometry": [dict(CUBOID, boundary=boundary)],
        }
    )


def ms_lattice() -> MSLattice:
    """An 8x8 lattice, four velocities per dimension, one bounce-back obstacle."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
            "geometry": [
                {"shape": "cuboid", "x": [2, 5], "y": [3, 5], "boundary": "bounceback"}
            ],
        }
    )


STEPS = {
    "ab_bounceback": lambda: CQLBM(ab_lattice("bounceback")),
    "ab_specular": lambda: CQLBM(ab_lattice("specular")),
    "ab_zone_agnostic_bounceback": lambda: ABQLBM(ab_lattice("bounceback"), True),
    "ab_zone_agnostic_specular": lambda: ABQLBM(ab_lattice("specular"), True),
    "ms_bounceback": lambda: MSQLBM(ms_lattice()),
}


@pytest.mark.parametrize("name", STEPS)
def test_structured_step_matches_the_gate_path(name):
    """The planned kernels and the plain gates agree on a random state."""
    step = STEPS[name]()
    rng = np.random.default_rng(11)
    state = rng.standard_normal(1 << step.n_qubits) + 1j * rng.standard_normal(
        1 << step.n_qubits
    )
    state /= np.linalg.norm(state)

    structured = step.statevector(initial_state=state, structured=True)
    gates = step.statevector(initial_state=state, structured=False)

    np.testing.assert_allclose(structured, gates, atol=1e-9, rtol=0)


# Steps whose every gate belongs to a declared or derived permutation.
PERMUTATION_ONLY = {"ab_bounceback", "ab_specular"}


@pytest.mark.parametrize("name", STEPS)
def test_step_plans_to_permutation_kernels(name):
    """A step runs as permutation kernels, plus gate runs where comparators remain; never dense."""
    kinds = STEPS[name]().kernels()

    assert kinds is not None
    assert "permutation" in kinds
    assert "dense" not in kinds
    if name in PERMUTATION_ONLY:
        assert set(kinds) == {"permutation"}


@pytest.mark.parametrize("name", STEPS)
def test_planner_reads_the_declared_shift(name, monkeypatch):
    """Structured execution takes the shift from its declaration, not from its gates.

    The planner derives a span it cannot read a declaration for, which is
    correct and slow; this is the one place that would notice.
    """
    declared = StreamingShift.classical_action
    consulted = []

    def spy(self, indices):
        consulted.append(self)
        return declared(self, indices)

    monkeypatch.setattr(StreamingShift, "classical_action", spy)
    step = STEPS[name]()

    step.statevector(structured=True)

    assert len(consulted) >= step.lattice.num_dims
