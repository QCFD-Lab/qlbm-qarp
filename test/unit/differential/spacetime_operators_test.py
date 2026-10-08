"""Differential tests: Space-Time operators vs the qiskit-based qlbm.

Comparison is element-wise with global phase included; the modes are listed
in ``test/oracle/cases_spacetime.py``.
"""

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.spacetime import (
    PointWiseSpaceTimeInitialConditions,
    PointWiseSpaceTimeReflectionOperator,
    SpaceTimeD2Q4CollisionOperator,
    SpaceTimeGridVelocityMeasurement,
    SpaceTimeQLBM,
    SpaceTimeStreamingOperator,
    VolumetricSpaceTimeInitialConditions,
)
from qlbm.components.spacetime.collision.eqc_collision import (
    GenericSpaceTimeCollisionOperator,
)
from qlbm.components.spacetime.measurement import SpaceTimePointWiseMassMeasurement
from qlbm.components.spacetime.reflection.volumetric import (
    VolumetricSpaceTimeReflectionOperator,
)
from qlbm.lattice import SpaceTimeLattice
from test.oracle import cases_spacetime
from test.unit.differential.fixtures import load_npz

_cases_module = cases_spacetime
CASES = _cases_module.CASES_SPACETIME
NUM_PROBES = _cases_module.NUM_PROBES


def _make_lattice(descriptor):
    return SpaceTimeLattice(
        descriptor["num_timesteps"],
        descriptor["lattice_data"],
        include_measurement_qubit=descriptor["include_measurement_qubit"],
        use_volumetric_ops=descriptor["use_volumetric_ops"],
    )


def _build_block(case):
    """Build the qarp component for ``case`` and return its block."""
    lattice = _make_lattice(case["lattice"])
    kind = case["component"]

    if kind == "st_streaming":
        return SpaceTimeStreamingOperator(lattice, case["timestep"])
    if kind == "st_pointwise_initial":
        return PointWiseSpaceTimeInitialConditions(
            lattice, case["grid_data"], case["filter_inside_blocks"]
        )
    if kind == "st_volumetric_initial":
        return VolumetricSpaceTimeInitialConditions(
            lattice, case["cuboid_bounds"], case["velocity_profile"]
        )
    if kind == "st_pointwise_reflection":
        return PointWiseSpaceTimeReflectionOperator(
            lattice,
            case["timestep"],
            lattice.shapes["bounceback"],
            case["filter_inside_blocks"],
        )
    if kind == "st_volumetric_reflection":
        return VolumetricSpaceTimeReflectionOperator(
            lattice,
            case["timestep"],
            lattice.shapes["bounceback"],
            case["filter_inside_blocks"],
        )
    if kind == "st_eqc_collision":
        return GenericSpaceTimeCollisionOperator(lattice, case["timestep"])
    if kind == "st_d2q4_collision":
        return SpaceTimeD2Q4CollisionOperator(lattice, case["timestep"])
    if kind == "st_d2q4_local_collision":
        return SpaceTimeD2Q4CollisionOperator(lattice, 1).local_collision_circuit(
            case["reset_state"]
        )
    if kind == "st_grid_velocity_measurement":
        return SpaceTimeGridVelocityMeasurement(lattice)
    if kind == "st_mass_measurement":
        return SpaceTimePointWiseMassMeasurement(
            lattice, case["gridpoint"], case["velocity_index_to_measure"]
        )
    if kind == "st_qlbm":
        return SpaceTimeQLBM(lattice, case["filter_inside_blocks"])
    raise ValueError(f"Unknown component kind: {kind}")


def build_block(case):
    """Build the component for ``case`` and return it built."""
    block = _build_block(case)
    block.build()
    return block


def _load_fixture(case):
    return load_npz(case["id"], "generate_fixtures_spacetime")


def _split_measurements(block):
    """Split a flattened block into gate commands and ``(qubit, cbit)`` pairs."""
    gates, pairs = [], []
    for command in block.flatten():
        if command.cbits:
            pairs.append((command.qubits[0], command.cbits[0]))
        else:
            gates.append(command)
    return gates, pairs


def _evolve(commands, num_qubits, psi_in):
    return np.asarray(
        qx.QarpSimulator().statevector(
            commands,
            num_qubits,
            initial_state=np.ascontiguousarray(psi_in, dtype=np.complex128),
        )
    )


def _densify(indices, amplitudes, num_qubits):
    dense = np.zeros(2**num_qubits, dtype=np.complex128)
    dense[indices] = amplitudes
    return dense


def _assert_probes(fixture, commands, num_qubits):
    for k in range(NUM_PROBES):
        actual = _evolve(commands, num_qubits, fixture[f"psi_in_{k}"])
        np.testing.assert_allclose(
            actual, fixture[f"psi_out_{k}"], atol=1e-9, rtol=0, err_msg=f"probe {k}"
        )


def _assert_sparse(fixture, commands, num_qubits):
    for k in range(NUM_PROBES):
        psi_in = _densify(fixture[f"in_idx_{k}"], fixture[f"in_amp_{k}"], num_qubits)
        expected = _densify(
            fixture[f"out_idx_{k}"], fixture[f"out_amp_{k}"], num_qubits
        )
        actual = _evolve(commands, num_qubits, psi_in)
        np.testing.assert_allclose(
            actual, expected, atol=1e-9, rtol=0, err_msg=f"sparse probe {k}"
        )


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_spacetime_matches_reference(case):
    """The operator reproduces the qiskit-based action exactly."""
    fixture = _load_fixture(case)
    block = build_block(case)
    num_qubits = block.n_qubits
    assert num_qubits == int(fixture["n_qubits"])

    mode = case["mode"]
    if mode == "unitary":
        actual = np.asarray(
            qx.QarpSimulator().unitary_matrix(block.flatten(), num_qubits)
        )
        np.testing.assert_allclose(actual, fixture["unitary"], atol=1e-9, rtol=0)
        return

    gates, pairs = _split_measurements(block)

    if mode == "measurement":
        expected_pairs = fixture["measure_pairs"]
        assert np.array_equal(
            np.asarray(pairs, dtype=np.int64).reshape(-1, 2), expected_pairs
        )

    if mode in ("probes", "measurement"):
        _assert_probes(fixture, gates, num_qubits)
    elif mode == "sparse":
        _assert_sparse(fixture, gates, num_qubits)
    else:
        raise ValueError(f"Unknown mode: {mode}")
