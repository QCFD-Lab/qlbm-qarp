"""Differential tests: MS components vs the qiskit-based qlbm.

MS components are wider than 6 qubits, so each is compared through its action
on seeded probe states, global phase included.
"""

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.ms import (
    MSQLBM,
    BounceBackReflectionOperator,
    BounceBackWallComparator,
    ControlledIncrementer,
    EdgeComparator,
    GridMeasurement,
    MSInitialConditions,
    MSInitialConditions3DSlim,
    MSStreamingOperator,
    SpecularReflectionOperator,
    SpecularWallComparator,
    StreamingAncillaPreparation,
)
from qlbm.lattice import MSLattice
from test.oracle import cases_ms
from test.unit.differential.fixtures import load_npz

_cases_module = cases_ms

CASES = _cases_module.CASES
LATTICES = _cases_module.LATTICES
probe_state = _cases_module.probe_state

_LATTICE_CACHE: dict = {}


def get_lattice(name):
    """Build (and memoise) the lattice named ``name`` in ``cases_ms.LATTICES``."""
    if name not in _LATTICE_CACHE:
        _LATTICE_CACHE[name] = MSLattice(LATTICES[name])
    return _LATTICE_CACHE[name]


def boundary_operator(case, lattice):
    """Instantiate the reflection operator matching ``case['boundary']``."""
    boundary = case["boundary"]
    blocks = lattice.shapes[boundary]
    if boundary == "specular":
        return SpecularReflectionOperator(lattice, blocks), blocks
    return BounceBackReflectionOperator(lattice, blocks), blocks


def select_wall(blocks, spec):
    """Resolve ``["inside"|"outside", dim, index]`` against ``blocks[0]``."""
    side, dim, index = spec
    walls = blocks[0].walls_inside if side == "inside" else blocks[0].walls_outside
    return walls[dim][index]


def select_edge(blocks, spec):
    """Resolve ``["near_corner"|"corner", index]`` against ``blocks[0]``."""
    kind, index = spec
    edges = (
        blocks[0].near_corner_edges_3d
        if kind == "near_corner"
        else blocks[0].corner_edges_3d
    )
    return edges[index]


def select_point(blocks, spec):
    """Resolve a point selector against ``blocks[0]``."""
    kind, index = spec
    lookup = {
        "corner_outside": blocks[0].corners_outside,
        "near_corner_2d": blocks[0].near_corner_points_2d,
        "overlapping_near_corner_3d": blocks[0].overlapping_near_corner_edge_points_3d,
    }
    return lookup[kind][index]


def _build_block(case):
    """Build the qarp block for ``case``."""
    kind = case["component"]
    lattice = get_lattice(case["lattice"])

    if kind == "streaming_ancilla_prep":
        return StreamingAncillaPreparation(lattice, case["velocities"], case["dim"])
    if kind == "controlled_incrementer":
        return ControlledIncrementer(lattice, reflection=case["reflection"])
    if kind == "streaming_operator":
        return MSStreamingOperator(lattice, case["velocities"])
    if kind == "initial_conditions":
        return MSInitialConditions(lattice)
    if kind == "initial_conditions_3d_slim":
        return MSInitialConditions3DSlim(lattice)
    if kind == "grid_measurement":
        return GridMeasurement(lattice)
    if kind == "specular_wall_comparator":
        blocks = lattice.shapes["specular"]
        return SpecularWallComparator(lattice, select_wall(blocks, case["wall"]))
    if kind == "bounceback_wall_comparator":
        blocks = lattice.shapes["bounceback"]
        return BounceBackWallComparator(lattice, select_wall(blocks, case["wall"]))
    if kind == "edge_comparator":
        blocks = lattice.shapes["specular"] or lattice.shapes["bounceback"]
        return EdgeComparator(lattice, select_edge(blocks, case["edge"]))
    if kind == "specular_reflection_operator":
        return SpecularReflectionOperator(lattice, lattice.shapes["specular"])
    if kind == "bounceback_reflection_operator":
        return BounceBackReflectionOperator(lattice, lattice.shapes["bounceback"])
    if kind == "msqlbm":
        return MSQLBM(lattice, group_velocities=case["group_velocities"])

    # Sub-circuit cases: one reflection-operator method's block.
    operator, blocks = boundary_operator(case, lattice)
    if kind == "reflect_wall":
        return operator.reflect_wall(select_wall(blocks, case["wall"]))
    if kind == "reset_edge_state":
        return operator.reset_edge_state(select_edge(blocks, case["edge"]))
    if kind == "reset_point_state":
        return operator.reset_point_state(select_point(blocks, case["point"]))
    if kind == "flip_and_stream":
        return operator.flip_and_stream()
    raise ValueError(f"Unknown component kind: {kind}")


def build_block(case):
    """Build the component for ``case`` and return it built."""
    block = _build_block(case)
    block.build()
    return block


def measurement_pairs(block):
    """Ordered ``(qubit, cbit)`` pairs of every ``Measure`` in ``block``."""
    pairs = [
        (command.qubits[0], command.cbits[0])
        for command in block.flatten()
        if command.gate.name == "Measure"
    ]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def load_fixture(case):
    """Load ``case``'s fixture, skipping the test when it has not been generated."""
    return load_npz(case["id"], "generate_fixtures_ms")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_ms_component_matches_reference(case):
    """The MS component acts exactly as the qiskit-based one on the probe states."""
    fixture = load_fixture(case)
    block = build_block(case)
    n = int(fixture["n_qubits"])
    assert block.n_qubits == n

    kind = str(fixture["kind"])
    if kind == "measure":
        np.testing.assert_array_equal(measurement_pairs(block), fixture["pairs"])
        return

    simulator = qx.QarpSimulator()
    commands = block.flatten()

    if kind == "statevector":
        psi_in = fixture["psi_in"]
        for k, psi in enumerate(psi_in):
            actual = np.asarray(simulator.statevector(commands, n, initial_state=psi))
            np.testing.assert_allclose(
                actual, fixture["psi_out"][k], atol=1e-9, rtol=0, err_msg=f"probe {k}"
            )
        return

    assert kind == "statevector_sampled"
    indices = fixture["sample_indices"]
    for k, expected in enumerate(fixture["psi_out_samples"]):
        actual = np.asarray(
            simulator.statevector(commands, n, initial_state=probe_state(n, k))
        )
        np.testing.assert_allclose(
            actual[indices], expected, atol=1e-9, rtol=0, err_msg=f"probe {k}"
        )


def test_probe_states_are_normalised():
    """The deterministic probe generator returns unit vectors, not near-zero ones."""
    for n in (3, 8, 13):
        for k in range(3):
            psi = probe_state(n, k)
            assert psi.shape == (1 << n,)
            np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12, rtol=0)
        # distinct probes must not collide, or the oracle would be blind
        assert abs(np.vdot(probe_state(n, 0), probe_state(n, 1))) < 0.9
