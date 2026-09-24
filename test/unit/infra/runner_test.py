"""Unit tests for :class:`.QarpRunner` that need no oracle fixtures."""

from typing import List, Tuple

import numpy as np
import pytest
import qarpx as qx
from qarp import EXACT

from qlbm.components.common import EmptyPrimitive
from qlbm.components.lqlga import (
    LQLGA,
    LQGLAInitialConditions,
    LQLGAGridVelocityMeasurement,
)
from qlbm.infra import QarpRunner, SimulationConfig
from qlbm.infra.runner.qarp_runner import NORM_KEPT_DRIFT
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.tools.exceptions import ExecutionException
from test.builders import CircuitBuilder

# Two particles on a 4-site D1Q2 ring: 8 qubits, a deterministic basis state.
GRID_DATA: List[Tuple[Tuple[int, ...], Tuple[bool, ...]]] = [
    ((0,), (True, False)),
    ((2,), (False, True)),
]


@pytest.fixture
def lattice():
    """A small LQLGA lattice."""
    return LQLGALattice(
        {"lattice": {"dim": {"x": 4}, "velocities": "D1Q2"}, "geometry": []}
    )


def build_config(lattice, shots=128, initial_conditions=None):
    """A validated, prepared LQLGA config on ``lattice``."""
    config = SimulationConfig(
        initial_conditions=(
            LQGLAInitialConditions(lattice, GRID_DATA)
            if initial_conditions is None
            else initial_conditions
        ),
        algorithm=LQLGA(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=LQLGAGridVelocityMeasurement(lattice),
        shots=shots,
    )
    config.validate()
    config.prepare_for_simulation()
    return config


def initial_state(runner, config) -> np.ndarray:
    """The state the config's initial conditions prepare."""
    return np.asarray(
        qx.QarpSimulator().statevector(
            list(config.initial_conditions.flatten()), runner.num_qubits
        )
    )


def commands(circuit) -> list:
    """The command stream of a config attribute, checked to be a built block."""
    assert isinstance(circuit, qx.Block)
    return list(circuit.flatten())


class TestSampling:
    """Shot sampling and exact readout agree, and both land in cbit space."""

    def test_exact_counts_sum_to_one(self, lattice):
        """Exact readout returns a normalized distribution over the cbits."""
        config = build_config(lattice, shots=EXACT)
        runner = QarpRunner(config, lattice, seed=3)

        counts = runner._sample(initial_state(runner, config), EXACT, 0)

        assert sum(counts.values()) == pytest.approx(1.0)
        assert all(0 <= key < (1 << runner.num_cbits) for key in counts)

    def test_sampled_counts_converge_to_the_exact_distribution(self, lattice):
        """Sampling reproduces the exact distribution within shot noise."""
        config = build_config(lattice)
        runner = QarpRunner(config, lattice, seed=17)
        state = initial_state(runner, config)

        exact = runner._sample(state, EXACT, 0)
        sampled = runner._sample(state, 8192, 0)

        assert sum(sampled.values()) == 8192
        assert set(sampled) <= set(exact)
        for key, probability in exact.items():
            assert sampled.get(key, 0) / 8192 == pytest.approx(probability, abs=0.05)

    def test_counts_are_rekeyed_onto_the_classical_register(self):
        """Qubit-space sampling counts are folded onto the declared cbits."""
        # qarpx keys SamplingResult.counts on the *qubit* register; the runner
        # must fold that onto the cbits the measurement component declared.
        lattice = LQLGALattice(
            {"lattice": {"dim": {"x": 2}, "velocities": "D1Q2"}, "geometry": []}
        )
        builder = CircuitBuilder(lattice.n_qubits, name="permuted_measurement")
        # Reversed qubit -> cbit mapping: qubit 3 lands in cbit 0.
        for cbit, qubit in enumerate(reversed(range(4))):
            builder.measure(qubit, cbit)

        preparation = CircuitBuilder(lattice.n_qubits, name="prep")
        preparation.x(3)

        config = SimulationConfig(
            initial_conditions=preparation.build(),
            algorithm=EmptyPrimitive(lattice),
            postprocessing=EmptyPrimitive(lattice),
            measurement=builder.build(),
            shots=64,
        )
        config.validate()
        config.prepare_for_simulation()
        runner = QarpRunner(config, lattice, seed=5)

        statevector = np.asarray(
            qx.QarpSimulator().statevector(
                commands(config.initial_conditions), runner.num_qubits
            )
        )

        assert runner._sample(statevector, 64, 0) == {1: 64}
        assert runner._sample(statevector, EXACT, 0) == {1: pytest.approx(1.0)}

    def test_gates_in_the_measurement_component_are_applied(self):
        """Gates preceding the ``measure`` commands act on the sampled state."""
        # Measurement components such as the Space-Time mass measurement
        # prepare an ancilla with X/MCX before measuring it.
        lattice = LQLGALattice(
            {"lattice": {"dim": {"x": 2}, "velocities": "D1Q2"}, "geometry": []}
        )
        measurement = CircuitBuilder(lattice.n_qubits, name="flip_then_measure")
        measurement.x(2)
        measurement.cx(2, 3)
        measurement.measure(3, 0)
        measurement.measure(2, 1)

        config = SimulationConfig(
            initial_conditions=EmptyPrimitive(lattice),
            algorithm=EmptyPrimitive(lattice),
            postprocessing=EmptyPrimitive(lattice),
            measurement=measurement.build(),
            shots=64,
        )
        config.validate()
        config.prepare_for_simulation()
        runner = QarpRunner(config, lattice, seed=5)
        vacuum = np.zeros(1 << runner.num_qubits, dtype=np.complex128)
        vacuum[0] = 1.0

        assert runner._sample(vacuum, 64, 0) == {0b11: 64}
        assert runner._sample(vacuum, EXACT, 0) == {0b11: pytest.approx(1.0)}


class TestStateHandling:
    """Carried states are renormalised against rounding drift, never beyond it."""

    def test_rounding_drift_is_renormalised(self, lattice):
        """A state a little off unit norm evolves and samples as the exact one."""
        config = build_config(lattice, shots=EXACT)
        runner = QarpRunner(config, lattice, seed=5)
        exact = runner._evolve(initial_state(runner, config), 1)
        drifted = exact * (1 - 1e-9)

        np.testing.assert_allclose(
            runner._evolve(drifted, 1), runner._evolve(exact, 1), atol=1e-12
        )
        assert runner._sample(drifted, EXACT, 0) == pytest.approx(
            runner._sample(exact, EXACT, 0)
        )

    def test_drift_below_the_rescale_bound_is_accepted(self, lattice):
        """Drift the runner leaves unscaled still passes qarpx's norm check."""
        config = build_config(lattice, shots=EXACT)
        runner = QarpRunner(config, lattice, seed=5)
        exact = runner._evolve(initial_state(runner, config), 1)
        drifted = exact * (1 - 0.9 * NORM_KEPT_DRIFT)

        np.testing.assert_allclose(
            runner._evolve(drifted, 1), runner._evolve(exact, 1), atol=1e-12
        )
        assert runner._sample(drifted, EXACT, 0) == pytest.approx(
            runner._sample(exact, EXACT, 0)
        )

    def test_unnormalised_state_is_rejected(self, lattice):
        """A state far from unit norm is an input error, not drift."""
        config = build_config(lattice, shots=EXACT)
        runner = QarpRunner(config, lattice, seed=5)
        doubled = 2 * initial_state(runner, config)

        with pytest.raises(ExecutionException, match="norm"):
            runner._evolve(doubled, 1)
        with pytest.raises(ExecutionException, match="norm"):
            runner._sample(doubled, EXACT, 0)

    def test_injected_statevector_seeds_the_run(self, lattice, tmp_path):
        """An injected statevector becomes the run's step-zero state."""
        reference = np.asarray(
            qx.QarpSimulator().statevector(
                LQGLAInitialConditions(lattice, GRID_DATA).flatten(), lattice.n_qubits
            )
        )
        config = build_config(lattice, initial_conditions=reference)
        runner = QarpRunner(config, lattice, save_statevector_to_disk=True, seed=1)
        runner.run(0, None, str(tmp_path), statevector_snapshots=True)

        np.testing.assert_allclose(
            np.load(f"{tmp_path}/statevectors/step_0.npy"), reference, atol=1e-12
        )


class TestRunArguments:
    """``run`` resolves and validates its shot budget and seeds every step."""

    def test_shots_default_to_the_config(self, lattice, tmp_path, monkeypatch):
        """``num_shots=None`` samples every step with the config's budget."""
        config = build_config(lattice, shots=256)
        runner = QarpRunner(config, lattice, seed=2)
        budgets = []
        sample = runner._sample

        def record(statevector, num_shots, step):
            counts = sample(statevector, num_shots, step)
            budgets.append((num_shots, sum(counts.values())))
            return counts

        monkeypatch.setattr(runner, "_sample", record)
        runner.run(2, None, str(tmp_path), statevector_snapshots=True)

        assert budgets == [(256, 256)] * 3

    @pytest.mark.parametrize("num_shots", [0, -5, 2.5])
    def test_invalid_shot_budget_is_rejected(self, lattice, tmp_path, num_shots):
        """A non-positive or fractional budget fails before anything runs."""
        runner = QarpRunner(build_config(lattice), lattice, seed=2)

        with pytest.raises(ExecutionException, match="shot budget"):
            runner.run(1, num_shots, str(tmp_path), statevector_snapshots=True)

    def test_step_seeds_are_reproducible_and_distinct(self, lattice):
        """A seed fixes every step's stream; neighbouring seeds share none."""
        config = build_config(lattice)
        first = QarpRunner(config, lattice, seed=7)
        again = QarpRunner(config, lattice, seed=7)
        neighbour = QarpRunner(config, lattice, seed=8)

        steps = range(4)
        assert [first._step_seed(s) for s in steps] == [
            again._step_seed(s) for s in steps
        ]
        assert len({first._step_seed(s) for s in steps}) == 4
        assert not {first._step_seed(s) for s in steps} & {
            neighbour._step_seed(s) for s in steps
        }
        assert QarpRunner(config, lattice)._step_seed(0) is None


class TestRunnerSurface:
    """Structural guarantees of the runner's public surface."""

    def test_classical_register_width_comes_from_the_command_stream(self, lattice):
        """The classical register is sized from measure commands, not n_cbits."""
        measurement = LQLGAGridVelocityMeasurement(lattice)
        assert measurement.n_cbits == lattice.num_base_qubits

        runner = QarpRunner(build_config(lattice), lattice)

        assert runner.num_cbits == lattice.num_base_qubits
        assert runner.num_qubits == lattice.n_qubits
