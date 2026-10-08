"""A ``SimulationConfig`` ties together the four algorithmic circuits, the optimization level, and the shot budget."""

from logging import Logger, getLogger
from numbers import Integral
from typing import Any, List

import numpy as np
import qarpx as qx
from qarp import EXACT, Shots

from qlbm.infra.compiler import CircuitCompiler
from qlbm.tools.exceptions import ExecutionException


def resolve_shots(shots: Any) -> "int | Shots":
    """
    The shot budget as a built-in ``int``, or ``qarp.EXACT`` unchanged.

    Parameters
    ----------
    shots : Any
        The requested budget: a positive whole number of any integral type, or ``qarp.EXACT``.

    Returns
    -------
    int | Shots
        ``qarp.EXACT``, or the budget as a built-in ``int``.

    Raises
    ------
    ExecutionException
        If ``shots`` is neither ``qarp.EXACT`` nor a positive whole number.
    """
    if shots is EXACT:
        return EXACT
    # bool is an integral type, and True would sample a single shot.
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ExecutionException(
            f"Unsupported shot budget {shots}. Provide a positive integer or qarp.EXACT."
        )
    return int(shots)


class SimulationConfig:
    """
    A ``SimulationConfig`` ties together the algorithmic quantum components, the compiler optimization level, and the shot budget.

    This is the most convenient access point for performing simulations with ``qlbm``.

    Algorithmic attributes specify the complete, end-to-end, QLBM algorithm.
    This includes initial conditions, the time step circuit,
    an optional postprocessing step, and a final measurement procedure.

    .. list-table:: Algorithmic attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`initial_conditions`
          - The initial conditions of the simulation. Either a qarp ``Block`` (every component is one) or an LSB-indexed statevector ``np.ndarray``.
        * - :attr:`algorithm`
          - The algorithm that performs the QLBM time step computation. For example, :class:`.CQLBM` or :class:`.SpaceTimeQLBM`.
        * - :attr:`postprocessing`
          - The quantum component concatenated to the ``algorithm``. Usually :class:`.EmptyPrimitive`.
        * - :attr:`measurement`
          - The circuit that samples the quantum state. For example, :class:`.GridMeasurement` or :class:`.SpaceTimeGridVelocityMeasurement`.

    .. list-table:: Execution attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`optimization_level`
          - The optimization level qarp applies at execution to the gate runs left after planning a block's structure; 0 (the default) runs them as they are, 1 cancels and fuses adjacent gates, 2 also cancels across commuting gates.
        * - :attr:`shots`
          - The default number of shots per time step. ``qarp.EXACT`` returns exact probabilities instead of sampled counts.

    .. note::
        Example configuration: simulating :class:`.SpaceTimeQLBM`.

        .. code-block:: python

            cfg = SimulationConfig(
                initial_conditions=PointWiseSpaceTimeInitialConditions(
                    lattice, grid_data=[((1, 5), (True, True, True, True))]
                ),
                algorithm=SpaceTimeQLBM(lattice),
                postprocessing=EmptyPrimitive(lattice),
                measurement=SpaceTimeGridVelocityMeasurement(lattice),
                optimization_level=0,
                shots=4096,
            )

            cfg.validate()
            cfg.prepare_for_simulation()

        The circuits are lowered in place, which makes it easy to plug the
        ``cfg`` object into a :class:`.QarpRunner`:

        .. code-block:: python

            runner = QarpRunner(cfg, lattice)
            runner.run(10, 4096, "output_dir", statevector_snapshots=True)
    """

    initial_conditions: "qx.Block | np.ndarray"
    algorithm: "qx.Block"
    postprocessing: "qx.Block"
    measurement: "qx.Block"
    optimization_level: int
    shots: int
    logger: Logger

    circuit_types: List[Any] = [qx.Block]
    """The accepted types of the four algorithmic attributes."""

    initial_conditions_types: List[Any] = [qx.Block, np.ndarray]
    """The accepted types of :attr:`initial_conditions` (a statevector is also allowed)."""

    def __init__(
        self,
        initial_conditions: "qx.Block | np.ndarray",
        algorithm: "qx.Block",
        postprocessing: "qx.Block",
        measurement: "qx.Block",
        optimization_level: int = 0,
        shots: int = 1024,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        # Circuits
        self.initial_conditions = initial_conditions
        self.algorithm = algorithm
        self.postprocessing = postprocessing
        self.measurement = measurement

        # Simulation details
        self.optimization_level = optimization_level
        self.shots = shots
        self.logger = logger

    def validate(
        self,
    ) -> None:
        """
        Validates the configuration.

        This includes the following checks:

        #. The algorithmic attributes are of compatible types.
        #. The optimization level is supported by the compiler.
        #. The shot budget is either a positive integer or ``qarp.EXACT``.

        This function simply checks that the provided attributes are
        suitable - it does not perform any conversions.

        Raises
        ------
        ExecutionException
            If any of the above checks fail.
        """
        self.__is_compatible_type(
            self.initial_conditions, self.initial_conditions_types, "Initial conditions"
        )
        self.__is_compatible_type(self.algorithm, self.circuit_types, "Algorithm")
        self.__is_compatible_type(
            self.postprocessing, self.circuit_types, "Postprocessing"
        )
        self.__is_compatible_type(self.measurement, self.circuit_types, "Measurement")

        if self.optimization_level not in CircuitCompiler.supported_optimization_levels:
            raise ExecutionException(
                f"Unsupported optimization level {self.optimization_level}. Supported optimization levels are {CircuitCompiler.supported_optimization_levels}."
            )

        resolve_shots(self.shots)

    def __is_compatible_type(
        self,
        object_to_validate: Any,
        accepted_types: List[Any],
        object_name: str,
    ) -> None:
        if not any(isinstance(object_to_validate, t) for t in accepted_types):
            raise ExecutionException(
                f"{object_name} object of type {type(object_to_validate)} is not in supported types {accepted_types}",
            )

    def prepare_for_simulation(
        self,
    ) -> None:
        """Lowers all algorithmic components into built qarp blocks, in place.

        A statevector supplied as :attr:`initial_conditions` is left untouched
        - the runner injects it directly into the simulator.
        """
        compiler = self.get_execution_compiler()

        if not isinstance(self.initial_conditions, np.ndarray):
            self.initial_conditions = compiler.compile(
                self.initial_conditions, self.optimization_level
            )

        self.algorithm = compiler.compile(self.algorithm, self.optimization_level)
        self.postprocessing = compiler.compile(
            self.postprocessing, self.optimization_level
        )
        # Measurement blocks are never optimized: the peephole passes are
        # defined over unitaries, and the cbit mapping is the result contract.
        self.measurement = compiler.compile(self.measurement, 0)

    def get_execution_compiler(self) -> CircuitCompiler:
        """
        Get the :class:`CircuitCompiler` that lowers the algorithmic attributes into qarp blocks.

        Returns
        -------
        CircuitCompiler
            A compatible circuit compiler.
        """
        return CircuitCompiler(self.logger)

    def __str__(self) -> str:
        """
        String representation of the configuration.

        Returns
        -------
        str
            The string representation.
        """
        return f"[SimulationConfig with optimization_level={self.optimization_level}, shots={self.shots}]"
