"""qarp-specific implementation of the :class:`CircuitRunner`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import Dict, List, Tuple

import numpy as np
import qarpx as qx
from qarp import EXACT
from qarp.algorithms import Sampler
from qarp.blocks import CompositeBlock
from qarp.engines import QarpEngine
from typing_extensions import override

from qlbm.infra.reinitialize import IdentityReinitializer
from qlbm.infra.result import QBMResult
from qlbm.lattice import Lattice
from qlbm.tools.exceptions import ExecutionException

from .base import CircuitRunner
from .simulation_config import SimulationConfig

# Carried states lose norm to rounding every step, and qarpx rejects an
# initial state more than 1e-10 off; drift up to this bound is renormalised.
NORM_DRIFT_TOLERANCE = 1e-6
# Drift this small is left in place: well inside qarpx's 1e-10, and the
# rescale is a full copy of the state.
NORM_KEPT_DRIFT = 1e-12


class QarpRunner(CircuitRunner):
    """
    qarp-specific implementation of the :class:`CircuitRunner`.

    The snapshot loop carries the statevector from step to step, so each step
    costs one application of the algorithm; counts are sampled from that state.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`config`              The :class:`.SimulationConfig` containing the simulation information.
    :attr:`lattice`             The :class:`.Lattice` of the simulated system.
    :attr:`reinitializer`       The :class:`.Reinitializer` that performs the transition between time steps.
    :attr:`device`              Currently ignored.
    :attr:`seed`                Base RNG seed for shot sampling; ``None`` samples nondeterministically.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``.
    =========================== ======================================================================
    """

    def __init__(
        self,
        config: SimulationConfig,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
        device: str = "CPU",  # ! TODO reimplement
        save_statevector_to_disk: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(config, lattice, logger, device)
        self.statevector_to_disk = save_statevector_to_disk
        self.seed = seed
        self.num_qubits = self.__infer_num_qubits()
        # The measurement component declares the (qubit, cbit) pairs; counts are keyed on the cbits.
        self.measure_pairs = self.__measure_pairs()
        self.num_cbits = (
            max(cbit for _, cbit in self.measure_pairs) + 1 if self.measure_pairs else 0
        )
        # The measurement component may rotate or prepare qubits before its
        # ``measure`` commands, so its gates belong to the sampled circuit.
        self.sampling_ket = CompositeBlock(
            [self.config.postprocessing, self.config.measurement],  # type: ignore[list-item]
            self.num_qubits,
        )
        self.sampling_ket.build()

    def __infer_num_qubits(self) -> int:
        widths = [
            block.n_qubits
            for block in (
                self.config.initial_conditions,
                self.config.algorithm,
                self.config.postprocessing,
                self.config.measurement,
            )
            if isinstance(block, qx.Block)
        ]
        if not widths:
            raise ExecutionException(
                "Cannot infer the simulation width: the config holds no built blocks. Call prepare_for_simulation() first."
            )
        return max(widths)

    def __measure_pairs(self) -> List[Tuple[int, int]]:
        return [
            (command.qubits[0], command.cbits[0])
            for command in self.config.measurement.flatten()  # type: ignore[union-attr]
            if command.cbits
        ]

    @override
    def run(
        self,
        num_steps: int,
        num_shots: int | None,
        output_directory: str,
        output_file_name: str = "step",
        statevector_snapshots: bool = False,
    ) -> QBMResult:
        # Replaying the algorithm from the initial conditions skips the
        # reinitializer, which is only equivalent when it passes the state on.
        if not statevector_snapshots and not isinstance(
            self.reinitializer, IdentityReinitializer
        ):
            raise ExecutionException(
                f"{type(self.reinitializer).__name__} re-encodes the state between "
                "time steps, which only the snapshot loop performs: run with "
                "statevector_snapshots=True."
            )
        shots = self.config.shots if num_shots is None else num_shots
        if shots is not EXACT and (not isinstance(shots, int) or shots < 1):
            raise ExecutionException(
                f"Unsupported shot budget {shots}. Provide a positive integer or qarp.EXACT."
            )
        simulation_result = self.new_result(output_directory, output_file_name)
        simulation_result.visualize_geometry()

        self.logger.info(
            f"Simulation start: QARP with config {self.config} with num_steps={num_steps}, num_shots={shots}, snapshots={statevector_snapshots}"
        )
        runner_start_time = perf_counter_ns()
        simulation_result = (
            self._run_snapshot_time_loop(num_steps, shots, simulation_result)
            if statevector_snapshots
            else self._run_time_loop(num_steps, shots, simulation_result)
        )
        self.logger.info(
            f"Entire simulation took {perf_counter_ns() - runner_start_time} (ns)"
        )
        return simulation_result

    @staticmethod
    def _normalised(state: np.ndarray) -> np.ndarray:
        """``state`` rescaled to unit norm, rejecting more than rounding drift."""
        amplitudes = np.ascontiguousarray(state, dtype=np.complex128)
        # einsum, not np.linalg.norm or vdot: BLAS wakes OpenBLAS's spinning
        # thread pool, which starves the simulator's OpenMP threads.
        parts = amplitudes.view(np.float64)
        norm = float(np.sqrt(np.einsum("i,i->", parts, parts)))
        if abs(norm - 1.0) > NORM_DRIFT_TOLERANCE:
            raise ExecutionException(f"State norm {norm} is not 1.")
        if abs(norm - 1.0) <= NORM_KEPT_DRIFT:
            return amplitudes
        return amplitudes / norm

    @staticmethod
    def _state_of(seed: "qx.Block | np.ndarray") -> np.ndarray:
        """The statevector a seed denotes: itself, or the state a block prepares from |0...0>."""
        if isinstance(seed, np.ndarray):
            return seed
        seed.build()
        return np.asarray(seed.statevector())

    def _evolve(self, seed: "qx.Block | np.ndarray", steps: int) -> np.ndarray:
        """Apply ``steps`` time steps of the algorithm to the state ``seed`` denotes."""
        state = self._normalised(self._state_of(seed))
        if steps == 0:
            return state
        evolution = self.config.algorithm**steps  # type: ignore[operator]
        evolution.build()
        return np.asarray(evolution.statevector(initial_state=state))

    def _run_snapshot_time_loop(
        self,
        num_steps: int,
        num_shots: int,
        simulation_result: QBMResult,
    ) -> QBMResult:
        # Each step advances the carried state by one algorithm application;
        # the reinitializer decides what seeds the next step (the state itself
        # or a re-synthesized initial-conditions block).
        seed: "qx.Block | np.ndarray" = self.config.initial_conditions  # type: ignore[assignment]

        for step in range(num_steps + 1):
            step_start_time = perf_counter_ns()
            statevector = self._evolve(seed, min(step, 1))
            counts = self._sample(statevector, num_shots, step)
            simulation_result.save_timestep_counts(counts, step, n_cbits=self.num_cbits)
            if self.statevector_to_disk:
                simulation_result.save_statevector(statevector, step=step)
            # Logged before the early exit so the final step is timed too —
            # reinitialization below belongs to the *next* step's work.
            self.logger.info(
                f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
            )
            if step == num_steps:
                break

            seed = self.reinitializer.reinitialize(
                statevector=statevector
                if self.reinitializer.requires_statevector()
                else np.zeros(0, dtype=np.complex128),
                counts=counts,
                n_cbits=self.num_cbits,
                optimization_level=self.config.optimization_level,
            )

        return simulation_result

    def _run_time_loop(
        self,
        num_steps: int,
        num_shots: int,
        simulation_result: QBMResult,
    ) -> QBMResult:
        # Every step is re-simulated from the initial conditions.
        for step in range(num_steps + 1):
            step_start_time = perf_counter_ns()
            statevector = self._evolve(self.config.initial_conditions, step)  # type: ignore[arg-type]
            counts = self._sample(statevector, num_shots, step)
            simulation_result.save_timestep_counts(counts, step, n_cbits=self.num_cbits)
            if self.statevector_to_disk:
                simulation_result.save_statevector(statevector, step=step)
            self.logger.info(
                f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
            )

        return simulation_result

    def _step_seed(self, step: int) -> int | None:
        """An independent sampling seed per time step, derived from :attr:`seed`."""
        if self.seed is None:
            return None
        return int(np.random.SeedSequence([self.seed, step]).generate_state(1)[0])

    def _sample(
        self, statevector: np.ndarray, num_shots: int, step: int
    ) -> Dict[int, float]:
        """
        Sample the measurement component applied to ``statevector`` after postprocessing, keyed on the classical bits.

        Parameters
        ----------
        statevector : np.ndarray
            The state to measure.
        num_shots : int
            The number of shots, or ``qarp.EXACT`` for the Born distribution.
        step : int
            The time step, which offsets the seed so steps draw independently.

        Returns
        -------
        Dict[int, float]
            Counts (or probabilities under ``EXACT``) per classical-register value.
        """
        sampler = Sampler(
            ket=self.sampling_ket,
            n_shots=num_shots,
            measured_qubits=[qubit for qubit, _ in self.measure_pairs],
            initial_state=self._normalised(statevector),
        )
        engine = QarpEngine(seed=self._step_seed(step))
        engine.build([sampler])
        distribution = engine.run()[0]

        counts: Dict[int, float] = {}
        for bits, probability in distribution.items():
            key = sum(
                bit << cbit
                for bit, (_, cbit) in zip(bits, self.measure_pairs, strict=True)
            )
            value = (
                probability if num_shots is EXACT else round(probability * num_shots)
            )
            counts[key] = counts.get(key, 0) + value
        return counts
