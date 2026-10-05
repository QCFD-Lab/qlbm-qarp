"""Base class for all simulator-specific runners."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger

from qlbm.infra.reinitialize.base import Reinitializer
from qlbm.infra.result.base import QBMResult
from qlbm.lattice import Lattice
from qlbm.tools.exceptions import ExecutionException

from .simulation_config import SimulationConfig


class CircuitRunner(ABC):
    """
    Base class for all simulator-specific runners.

    A ``CircuitRunner`` object uses the information provided in a :class:`.SimulationConfig`
    to efficiently simulate the QLBM circuit.
    This includes carrying the quantum state between time steps,
    concatenating command streams together, performing reinitialization,
    and processing results.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`config`              The :class:`.SimulationConfig` containing the simulation information.
    :attr:`lattice`             The :class:`.Lattice` of the simulated system.
    :attr:`reinitializer`       The :class:`.Reinitializer` that performs the transition between time steps.
    :attr:`device`              The simulation device; only ``"CPU"`` is supported.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``.
    =========================== ======================================================================
    """

    def __init__(
        self,
        config: SimulationConfig,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
        device: str = "CPU",  # ! TODO reimplement
    ) -> None:
        super().__init__()
        if device != "CPU":
            raise ExecutionException(
                f"Unsupported device {device!r}. Only 'CPU' is supported."
            )
        self.config = config
        self.lattice = lattice
        self.logger = logger
        self.device = device
        self.reinitializer = self.new_reinitializer()

    @abstractmethod
    def run(
        self,
        num_steps: int,
        num_shots: int | None,
        output_directory: str,
        output_file_name: str = "step",
        statevector_snapshots: bool = False,
    ) -> QBMResult:
        """
        Simulates the provided configuration.

        Parameters
        ----------
        num_steps : int
            The number of time steps to simulate the system for.
        num_shots : int | None
            The number of shots to perform for each time step.
            ``None`` falls back to the config's ``shots``; ``qarp.EXACT``
            yields exact probabilities instead of sampled counts.
        output_directory : str
            The directory to which output will be stored.
        output_file_name : str, optional
            The root name for files containing time step artifacts, by default "step".
        statevector_snapshots : bool, optional
            Whether to utilize statevector snapshots, by default False.

        Returns
        -------
        QBMResult
            The parsed result of the simulation.
        """
        pass

    def new_result(self, output_directory: str, output_file_name: str) -> QBMResult:
        """
        Get a new result object for the current runner.

        Delegates to the lattice's :meth:`.Lattice.create_result` factory method.

        Parameters
        ----------
        output_directory : str
            The directory where the result data will be stored.
        output_file_name : str
            The file name of the result data within the directory.

        Returns
        -------
        QBMResult
            An empty result object.
        """
        return self.lattice.create_result(output_directory, output_file_name)

    def new_reinitializer(self) -> Reinitializer:
        """
        Creates a new reinitializer for a simulated algorithm.

        Delegates to the lattice's :meth:`.Lattice.create_reinitializer` factory method.

        Returns
        -------
        Reinitializer
            A suitable reinitializer.
        """
        return self.lattice.create_reinitializer(
            self.config.get_execution_compiler(), self.logger
        )
