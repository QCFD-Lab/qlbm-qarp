"""Write the fixtures of ``cases_infra.py``: per-step states, decoded fields, or
reinitializer pairs and states.

Trajectories are generated under two Aer seeds and written only if they agree,
which for Space-Time shows the shot budget covers the state's support.

Run with the qiskit-based qlbm at ``fixture_store.REFERENCE_COMMIT`` installed; the
README's Testing section has the command.
"""

from tempfile import TemporaryDirectory

import numpy as np
from cases_infra import CASES_DECODE, CASES_REINIT, CASES_TRAJECTORY
from fixture_store import save_npz
from qiskit_aer import AerSimulator

# Import order matters: `qlbm.lattice` alone trips a circular import in the
# qiskit-based package, `qlbm.components` first resolves it.
import qlbm.components  # noqa: F401


def build_lattice(case):
    """Build the qiskit-based lattice for ``case``."""
    from qlbm.lattice import MSLattice
    from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
    from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice

    family = case.get("family", case.get("lattice_family"))
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
    """Build the four qiskit-based algorithmic components for a trajectory case."""
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


def run_reference_trajectory(case, seed):
    """Run the qiskit-based snapshot loop once and return its per-step statevectors."""
    from qlbm.infra.runner import QiskitRunner
    from qlbm.infra.runner.simulation_config import SimulationConfig

    lattice = build_lattice(case)
    initial_conditions, algorithm, postprocessing, measurement = build_circuits(
        case, lattice
    )

    # The qiskit-based runner only emits a `save_statevector` when the reinitializer asks
    # for the state or statevector sampling is on; Space-Time needs neither, so
    # its cases turn sampling on purely to make the snapshot observable.
    statevector_sampling = case.get("statevector_sampling", False)

    config = SimulationConfig(
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        statevector_sampling=statevector_sampling,
        execution_backend=AerSimulator(method="statevector", seed_simulator=seed),
        sampling_backend=AerSimulator(method="statevector", seed_simulator=seed)
        if statevector_sampling
        else None,
    )
    config.validate()
    config.prepare_for_simulation()

    with TemporaryDirectory() as directory:
        runner = QiskitRunner(config, lattice, save_statevector_to_disk=True)
        runner.run(
            case["num_steps"],
            case["num_shots"],
            directory,
            statevector_snapshots=True,
        )
        return [
            np.asarray(
                np.load(f"{directory}/statevectors/step_{step}.npy"),
                dtype=np.complex128,
            )
            for step in range(case["num_steps"] + 1)
        ]


def counts_of(case):
    """Render the case's synthetic ``(cbit_key, value)`` pairs as MSB bitstrings."""
    width = case["n_cbits"]
    return {format(key, f"0{width}b"): float(value) for key, value in case["counts"]}


def capture_decoded_field(case):
    """Push the case's synthetic counts through the qiskit-based decoder."""
    from qlbm.infra.result.amplitude_result import AmplitudeResult
    from qlbm.infra.result.lqlga_result import LQLGAResult
    from qlbm.infra.result.spacetime_result import SpaceTimeResult

    decoders = {
        "amplitude": AmplitudeResult,
        "spacetime": SpaceTimeResult,
        "lqlga": LQLGAResult,
    }

    lattice = build_lattice(case)
    with TemporaryDirectory() as directory:
        result = decoders[case["decoder"]](lattice, directory)

        captured = {}

        def capture(numpy_res, *_args, **_kwargs):
            captured["field"] = np.asarray(numpy_res, dtype=np.float64)

        result.save_timestep_array = capture
        result.save_timestep_counts(counts_of(case), 0, create_vis=False)
        return captured["field"]


def capture_reinitialization(case):
    """Push the case's synthetic counts through the qiskit-based SpaceTimeReinitializer."""
    from qiskit.quantum_info import Statevector

    from qlbm.components.spacetime.initial.pointwise import (
        PointWiseSpaceTimeInitialConditions,
    )
    from qlbm.infra.compiler import CircuitCompiler
    from qlbm.infra.reinitialize.spacetime_reinitializer import SpaceTimeReinitializer

    lattice = build_lattice({**case, "lattice_family": "spacetime"})
    reinitializer = SpaceTimeReinitializer(lattice, CircuitCompiler("QISKIT", "QISKIT"))
    pairs = reinitializer.counts_to_velocity_pairs(counts_of(case))

    grid = np.asarray([grid for grid, _ in pairs], dtype=np.int64)
    velocity = np.asarray([velocity for _, velocity in pairs], dtype=bool)

    # 2D Space-Time lattices are >20 qubits wide; storing their state would
    # blow the fixture budget, so those cases pin the decoding only.
    if not case.get("store_state", True):
        return grid, velocity, None

    circuit = PointWiseSpaceTimeInitialConditions(
        lattice, pairs, lattice.filter_inside_blocks
    ).circuit

    return (
        grid,
        velocity,
        np.asarray(Statevector(circuit).data, dtype=np.complex128),
    )


def main():
    """Generate every infra fixture."""

    for case in CASES_TRAJECTORY:
        first = run_reference_trajectory(case, seed=1234)
        second = run_reference_trajectory(case, seed=987654)
        for step, (a, b) in enumerate(zip(first, second, strict=True)):
            if not np.allclose(a, b, atol=1e-12):
                raise RuntimeError(
                    f"{case['id']}: trajectory is seed-dependent at step {step}; "
                    "raise num_shots or shrink the state's support."
                )
        save_npz(
            case["id"],
            **{f"psi_{step}": psi for step, psi in enumerate(first)},
        )
        print(f"{case['id']}: {len(first)} statevectors of dim {first[0].shape[0]}")

    for case in CASES_DECODE:
        field = capture_decoded_field(case)
        save_npz(case["id"], field=field)
        print(f"{case['id']}: field {field.shape}")

    for case in CASES_REINIT:
        grid, velocity, psi = capture_reinitialization(case)
        arrays = {"grid": grid, "velocity": velocity}
        if psi is not None:
            arrays["psi"] = psi
        save_npz(case["id"], **arrays)
        print(
            f"{case['id']}: {grid.shape[0]} pairs"
            + (f", psi dim {psi.shape[0]}" if psi is not None else ", no state")
        )


if __name__ == "__main__":
    main()
