"""Differential tests: QarpRunner trajectories vs the qiskit-based qlbm.

States, not counts, are compared, so the sampling RNG does not enter; the
Space-Time fixture is stored only if two Aer seeds give the same trajectory.
"""

import numpy as np
import pytest

from qlbm.components.common import EmptyPrimitive
from qlbm.components.lqlga import (
    LQLGA,
    LQGLAInitialConditions,
    LQLGAGridVelocityMeasurement,
)
from qlbm.components.ms import MSQLBM, GridMeasurement, MSInitialConditions
from qlbm.components.spacetime import (
    SpaceTimeGridVelocityMeasurement,
    SpaceTimeQLBM,
)
from qlbm.components.spacetime.initial.pointwise import (
    PointWiseSpaceTimeInitialConditions,
)
from qlbm.infra import QarpRunner, SimulationConfig
from qlbm.lattice import MSLattice
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from test.oracle import cases_infra
from test.unit.differential.fixtures import load_npz

_cases_module = cases_infra
CASES_TRAJECTORY = _cases_module.CASES_TRAJECTORY


def load_fixture(case_id: str):
    """Load the oracle fixture for ``case_id``, skipping when it is absent."""
    return load_npz(case_id, "generate_fixtures_infra")


def build_lattice(case):
    """Build the lattice for ``case``."""
    family = case["family"]
    if family == "ms":
        return MSLattice(case["lattice"])
    if family == "lqlga":
        return LQLGALattice(case["lattice"])
    if family == "spacetime":
        return SpaceTimeLattice(case["num_timesteps"], case["lattice"])
    raise ValueError(f"Unknown lattice family: {family}")


def grid_data_of(case):
    """Convert the JSON-shaped ``grid_data`` manifest entry to qlbm tuples."""
    return [
        (tuple(gridpoint), tuple(bool(v) for v in profile))
        for gridpoint, profile in case["grid_data"]
    ]


def build_circuits(case, lattice):
    """Build the four algorithmic components for a trajectory case."""
    family = case["family"]
    if family == "ms":
        return (
            MSInitialConditions(lattice),
            MSQLBM(lattice),
            EmptyPrimitive(lattice),
            GridMeasurement(lattice),
        )
    if family == "lqlga":
        return (
            LQGLAInitialConditions(lattice, grid_data_of(case)),
            LQLGA(lattice),
            EmptyPrimitive(lattice),
            LQLGAGridVelocityMeasurement(lattice),
        )
    if family == "spacetime":
        return (
            PointWiseSpaceTimeInitialConditions(lattice, grid_data_of(case)),
            SpaceTimeQLBM(lattice),
            EmptyPrimitive(lattice),
            SpaceTimeGridVelocityMeasurement(lattice),
        )
    raise ValueError(f"Unknown family: {family}")


def build_config(case, lattice, shots=None, optimization_level=0):
    """Build and prepare the :class:`.SimulationConfig` for ``case``."""
    initial_conditions, algorithm, postprocessing, measurement = build_circuits(
        case, lattice
    )
    config = SimulationConfig(
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        optimization_level=optimization_level,
        shots=case["num_shots"] if shots is None else shots,
    )
    config.validate()
    config.prepare_for_simulation()
    return config


def run_trajectory(case, directory, snapshots=True, shots=None, seed=7):
    """Run the runner and return its per-time-step statevectors."""
    lattice = build_lattice(case)
    config = build_config(case, lattice, shots=shots)
    runner = QarpRunner(config, lattice, save_statevector_to_disk=True, seed=seed)
    runner.run(
        case["num_steps"],
        None,
        str(directory),
        statevector_snapshots=snapshots,
    )
    return [
        np.load(f"{directory}/statevectors/step_{step}.npy")
        for step in range(case["num_steps"] + 1)
    ]


class TestSnapshotTrajectoryAgainstReference:
    """The injection-based snapshot loop reproduces the qiskit-based states."""

    @pytest.mark.parametrize("case", CASES_TRAJECTORY, ids=lambda case: case["id"])
    def test_statevector_trajectory_matches_reference(self, case, tmp_path):
        """Every time step's state equals the qiskit-based runner's."""
        fixture = load_fixture(case["id"])

        trajectory = run_trajectory(case, tmp_path)

        assert len(trajectory) == case["num_steps"] + 1
        for step, statevector in enumerate(trajectory):
            np.testing.assert_allclose(
                statevector,
                fixture[f"psi_{step}"],
                atol=1e-9,
                rtol=0,
                err_msg=f"{case['id']} diverges at step {step}",
            )


class TestLoopEquivalence:
    """The O(N) snapshot loop agrees with the naive re-run-from-zero loop."""

    @pytest.mark.parametrize(
        "case",
        [
            case
            for case in CASES_TRAJECTORY
            # Space-Time re-synthesizes from counts between steps, so the two
            # loops are physically different algorithms there by design.
            if case["family"] != "spacetime"
        ],
        ids=lambda case: case["id"],
    )
    def test_snapshot_loop_equals_naive_loop(self, case, tmp_path):
        """Carrying the state forward equals re-running from the initial conditions."""
        snapshot = run_trajectory(case, tmp_path / "snapshot", snapshots=True)
        naive = run_trajectory(case, tmp_path / "naive", snapshots=False)

        assert len(snapshot) == len(naive) >= 3
        for step, (carried, rerun) in enumerate(zip(snapshot, naive, strict=True)):
            np.testing.assert_allclose(
                carried, rerun, atol=1e-9, rtol=0, err_msg=f"step {step}"
            )
