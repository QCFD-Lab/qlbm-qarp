"""Differential tests: LQLGA operators vs the qiskit-based qlbm.

Small lattices are compared as full unitaries, wider ones through seeded probe
states, element-wise with global phase included.
"""

import numpy as np
import pytest
import qarpx as qx
from qarp.blocks import CompositeBlock

from qlbm.components.lqlga import (
    LQLGA,
    GenericLQLGACollisionOperator,
    LQGLAInitialConditions,
    LQLGAGridVelocityMeasurement,
    LQLGAMGReflectionOperator,
    LQLGAReflectionOperator,
    LQLGAStreamingOperator,
)
from qlbm.components.lqlga.initial import LQGLAAveragedInitialConditions
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from test.oracle import cases_lqlga
from test.unit.differential.fixtures import load_npz

_cases_module = cases_lqlga
CASES_LQLGA = _cases_module.CASES_LQLGA


def build_lattice(case):
    """Build the ``LQLGALattice`` for ``case``, geometries applied."""
    lattice = LQLGALattice(case["lattice"])
    if "geometries" in case:
        lattice.set_geometries(case["geometries"])
    if "accumulation" in case:
        lattice.use_accumulation_register(
            case["accumulation"]["size"], case["accumulation"]["indices"]
        )
    return lattice


def grid_data_of(case):
    """Convert the JSON-shaped ``grid_data`` manifest entry to qlbm tuples."""
    return [
        (tuple(gridpoint), tuple(bool(v) for v in profile))
        for gridpoint, profile in case["grid_data"]
    ]


def _build_block(case):
    """Build the qarp component for ``case`` and return its block."""
    lattice = build_lattice(case)
    kind = case["component"]

    if kind == "lqlga_streaming":
        return LQLGAStreamingOperator(lattice)
    if kind == "lqlga_collision":
        return GenericLQLGACollisionOperator(lattice)
    if kind == "lqlga_reflection":
        return LQLGAReflectionOperator(
            lattice, lattice.shapes["bounceback"] + lattice.shapes["specular"]
        )
    if kind == "lqlga_mg_reflection":
        return LQLGAMGReflectionOperator(
            lattice,
            [gdict["bounceback"] + gdict["specular"] for gdict in lattice.geometries],
        )
    if kind == "lqlga_initial":
        return LQGLAInitialConditions(lattice, grid_data_of(case))
    if kind == "lqlga_averaged_initial":
        return LQGLAAveragedInitialConditions(lattice, case["gridpoints"])
    if kind == "lqlga_measurement":
        return LQLGAGridVelocityMeasurement(lattice)
    if kind == "lqlga":
        return LQLGA(lattice)
    if kind == "lqlga_evolution":
        return CompositeBlock(
            [LQGLAInitialConditions(lattice, grid_data_of(case))]
            + [LQLGA(lattice) for _ in range(case["steps"])],
            lattice.n_qubits,
        )
    raise ValueError(f"Unknown component kind: {kind}")


def build_block(case):
    """Build the component for ``case`` and return it built."""
    block = _build_block(case)
    block.build()
    return block


def _fixture(case):
    return load_npz(case["id"], "generate_fixtures_lqlga")


def _unitary(block):
    return np.asarray(
        qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits)
    )


def _evolve(block, psi_in=None):
    initial_state = (
        None if psi_in is None else np.ascontiguousarray(psi_in, dtype=np.complex128)
    )
    return np.asarray(
        qx.QarpSimulator().statevector(
            block.flatten(), block.n_qubits, initial_state=initial_state
        )
    )


_UNITARY_CASES = [c for c in CASES_LQLGA if c["mode"] == "unitary"]
_PROBE_CASES = [c for c in CASES_LQLGA if c["mode"] == "probes"]
_MEASUREMENT_CASES = [c for c in CASES_LQLGA if c["mode"] == "measurement"]
_EVOLUTION_CASES = [c for c in CASES_LQLGA if c["mode"] == "evolution"]


@pytest.mark.parametrize("case", _UNITARY_CASES, ids=[c["id"] for c in _UNITARY_CASES])
def test_lqlga_unitary_matches_reference(case):
    """The unitary equals the qiskit-based one exactly (global phase included)."""
    expected = _fixture(case)["unitary"]
    actual = _unitary(build_block(case))

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("case", _PROBE_CASES, ids=[c["id"] for c in _PROBE_CASES])
def test_lqlga_probes_match_reference(case):
    """The action on the oracle's random probe states is identical."""
    fixture = _fixture(case)
    psi_in, psi_out = fixture["psi_in"], fixture["psi_out"]

    block = build_block(case)
    assert 2**block.n_qubits == psi_in.shape[1]

    for k in range(psi_in.shape[0]):
        actual = _evolve(block, psi_in[k])
        np.testing.assert_allclose(actual, psi_out[k], atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "case", _MEASUREMENT_CASES, ids=[c["id"] for c in _MEASUREMENT_CASES]
)
def test_lqlga_measurement_matches_reference(case):
    """Measured ``(qubit, clbit)`` pairs and their order match the qiskit-based ones."""
    fixture = _fixture(case)
    expected_pairs = fixture["pairs"]

    commands = list(build_block(case).flatten())
    measures = [c for c in commands if c.gate == qx.GateType.Measure]

    # num_ops pins that the oracle circuit carries no non-measure prefix.
    assert len(commands) == int(fixture["num_ops"])
    assert len(measures) == len(commands)

    actual_pairs = np.array(
        [(c.qubits[0], c.cbits[0]) for c in measures], dtype=np.int64
    ).reshape(-1, 2)
    np.testing.assert_array_equal(actual_pairs, expected_pairs)


@pytest.mark.parametrize(
    "case", _EVOLUTION_CASES, ids=[c["id"] for c in _EVOLUTION_CASES]
)
def test_lqlga_evolution_matches_reference(case):
    """Initial conditions plus ``steps`` LQLGA steps reproduce the oracle state."""
    expected = _fixture(case)["psi_out"]
    actual = _evolve(build_block(case))

    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
