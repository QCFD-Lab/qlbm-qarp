"""Differential tests: AB components vs the qiskit-based qlbm.

Each case compares an exact unitary (up to 6 qubits), seeded probe states, or
the measured ``(qubit, cbit)`` pairs, element-wise with global phase included.
"""

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.ab import (
    ABQLBM,
    ABDiscreteUniformInitialConditions,
    ABGridMeasurement,
    ABInitialConditions,
    ABParallelDiscreteUniformInitialConditions,
    ABStreamingOperator,
    BinaryToOHPermutation,
)
from qlbm.components.ab.averaged_collision import ABEAveragedCollisionOperator
from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.reflection import (
    ABBounceBackReflectionPermutation,
    ABReflectionOperator,
    ABSpecularReflectionPermutation,
    ABZoneAgnosticReflectionOperator,
    ABZoneAgnosticReflectionOracle,
)
from qlbm.components.ab.reflection.agnosotic_reflection import ABZoneAgnosticSRCheck
from qlbm.components.common.arithmetic import RGQFTMultiplier
from qlbm.components.cqlbm import CQLBM
from qlbm.lattice import ABLattice, OHLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from test.oracle import cases_ab
from test.unit.differential.fixtures import load_npz

_cases_module = cases_ab
CASES = _cases_module.CASES


def build_lattice(case):
    """Instantiate the lattice described by ``case``."""
    lattice_class = {"ABLattice": ABLattice, "OHLattice": OHLattice}[
        case["lattice_class"]
    ]
    lattice = lattice_class(case["lattice"])
    if case.get("marker_qubits"):
        lattice.set_num_marker_qubits(case["marker_qubits"])
    return lattice


def _build_block(case):
    """Build the qarp component for ``case`` and return its block."""
    kind = case["component"]

    if kind == "rgqft_multiplier":
        return RGQFTMultiplier(case["n"], case["m"])

    if kind == "bb_permutation":
        return ABBounceBackReflectionPermutation(
            case["n"],
            LatticeDiscretization.D2Q9,
            ABEncodingType[case["encoding"]],
            num_ctrl_qubits=case["ctrl"],
        )

    if kind == "sr_permutation":
        return ABSpecularReflectionPermutation(
            case["n"],
            LatticeDiscretization.D2Q9,
            ABEncodingType[case["encoding"]],
            tuple(case["reflect"]),
            num_ctrl_qubits=case["ctrl"],
        )

    lattice = build_lattice(case)

    if kind == "binary_to_oh":
        return BinaryToOHPermutation(lattice)
    if kind == "streaming":
        return ABStreamingOperator(lattice, case["controls"])
    if kind == "averaged_collision":
        return ABEAveragedCollisionOperator(lattice)
    if kind == "discrete_uniform_initial":
        return ABDiscreteUniformInitialConditions(
            lattice,
            case["velocities"],
            tuple(list(g) for g in case["grid_superpose"]),
        )
    if kind == "initial":
        return ABInitialConditions(lattice)
    if kind == "parallel_initial":
        return ABParallelDiscreteUniformInitialConditions(
            lattice,
            case["velocities"],
            [tuple(list(d) for d in g) for g in case["grid_superpose"]],
        )
    if kind == "grid_measurement":
        return ABGridMeasurement(lattice, case["measure_velocity"])
    if kind == "za_oracle":
        return ABZoneAgnosticReflectionOracle(
            lattice,
            lattice.shapes[case["shape_key"]][0],
            target_obstacle_index=case["target_obstacle_index"],
        )
    if kind == "za_sr_check":
        return ABZoneAgnosticSRCheck(
            lattice,
            lattice.discretization,
            lattice.shapes["specular"],
            check_negative_direction=case["check_negative_direction"],
        )
    if kind == "standard_reflection":
        return ABReflectionOperator(lattice)
    if kind == "agnostic_reflection":
        return ABZoneAgnosticReflectionOperator(lattice)
    if kind == "abqlbm":
        return ABQLBM(lattice, use_agnostic_bcs=case["use_agnostic_bcs"])
    if kind == "cqlbm":
        return CQLBM(lattice, use_agnostic_bcs=case["use_agnostic_bcs"])

    raise ValueError(f"Unknown component kind: {kind}")


def build_block(case):
    """Build the component for ``case`` and return it built."""
    block = _build_block(case)
    block.build()
    return block


def _load(case):
    return load_npz(case["id"], "generate_fixtures_ab")


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["mode"] == "unitary"],
    ids=[c["id"] for c in CASES if c["mode"] == "unitary"],
)
def test_ab_unitary_matches_reference(case):
    """The unitary equals the qiskit-based one exactly (global phase included)."""
    expected = _load(case)["unitary"]

    block = build_block(case)
    actual = np.asarray(
        qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits)
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["mode"] == "probe"],
    ids=[c["id"] for c in CASES if c["mode"] == "probe"],
)
def test_ab_probe_states_match_reference(case):
    """The block maps every probe state exactly as the qiskit-based one does."""
    fixture = _load(case)
    psi_in, psi_out = fixture["psi_in"], fixture["psi_out"]

    block = build_block(case)
    commands = block.flatten()
    assert 2**block.n_qubits == psi_in.shape[1]

    simulator = qx.QarpSimulator()
    for index in range(psi_in.shape[0]):
        initial = np.ascontiguousarray(psi_in[index], dtype=complex)
        actual = np.asarray(
            simulator.statevector(commands, block.n_qubits, initial_state=initial)
        )
        np.testing.assert_allclose(
            actual, psi_out[index], atol=1e-9, rtol=0, err_msg=f"probe {index}"
        )


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["mode"] == "measure"],
    ids=[c["id"] for c in CASES if c["mode"] == "measure"],
)
def test_ab_measurement_pairs_match_reference(case):
    """The measurement records the same ordered (qubit, cbit) pairs."""
    expected = _load(case)["pairs"]

    block = build_block(case)
    actual = np.array(
        [
            [command.qubits[0], command.cbits[0]]
            for command in block.flatten()
            if command.gate == qx.GateType.Measure
        ],
        dtype=int,
    )

    np.testing.assert_array_equal(actual, expected)
