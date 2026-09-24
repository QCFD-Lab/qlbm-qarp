""":class:`.SpaceTimeQLBM`-specific implementation of the :class:`.Reinitializer`."""

from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

import numpy as np
import qarpx as qx
from typing_extensions import override

from qlbm.components.spacetime.initial.pointwise import (
    PointWiseSpaceTimeInitialConditions,
)
from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.exceptions import ExecutionException

from .base import Reinitializer


class SpaceTimeReinitializer(Reinitializer):
    r"""
    :class:`.SpaceTimeQLBM`-specific implementation of the :class:`.Reinitializer`.

    The Space-Time encoding can only be evolved for as many time steps as the
    lattice was constructed with, after which the state must be re-synthesized.
    This reinitializer decodes the counts sampled by
    :class:`.SpaceTimeGridVelocityMeasurement` into ``(gridpoint, velocity profile)``
    pairs and rebuilds a :class:`.PointWiseSpaceTimeInitialConditions` circuit
    from them. No copy of the statevector is required.

    Counts are keyed by LSB classical-bit integers: the grid coordinates occupy
    cbits :math:`0 \dots n_g - 1` (dimension-major) and the origin's velocity
    profile occupies cbits :math:`n_g \dots n_g + n_v - 1`.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.SpaceTimeLattice` of the simulated system.
    :attr:`compiler`            The compiler that lowers novel initial conditions circuits.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``
    =========================== ======================================================================
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, compiler, logger)
        self.lattice = lattice
        self.logger = logger
        self.x_grid_qubits = self.lattice.num_gridpoints[0].bit_length()
        self.y_grid_qubits = (
            self.lattice.num_gridpoints[1].bit_length()
            if self.lattice.num_dims > 1
            else 0
        )
        self.num_grid_qubits = self.x_grid_qubits + self.y_grid_qubits
        self.num_velocities_per_point = (
            self.lattice.properties.get_num_velocities_per_point()
        )

    @override
    def reinitialize(
        self,
        statevector: np.ndarray,
        counts: Dict[int, float],
        n_cbits: int | None = None,
        optimization_level: int = 0,
    ) -> "qx.Block":
        """
        Converts the input ``counts`` into a new :class:`.PointWiseSpaceTimeInitialConditions` block that seeds the following time step.

        Parameters
        ----------
        statevector : np.ndarray
            Ignored.
        counts : Dict[int, float]
            The counts obtained from :class:`.SpaceTimeGridVelocityMeasurement`,
            keyed by LSB classical-bit integer.
        n_cbits : int | None, optional
            Ignored; the register layout follows from the lattice.
        optimization_level : int, optional
            The compiler optimization level.

        Returns
        -------
        qx.Block
            The initial conditions block to apply from the all-zero state.
        """
        return self.compiler.compile(
            PointWiseSpaceTimeInitialConditions(
                self.lattice,
                self.counts_to_velocity_pairs(counts),
                self.lattice.filter_inside_blocks,
            ),
            optimization_level=optimization_level,
        )

    def counts_to_velocity_pairs(
        self,
        counts: Dict[int, float],
    ) -> List[Tuple[Tuple[int, ...], Tuple[bool, ...]]]:
        """
        Converts all counts into their grid and velocity components.

        Outcomes whose origin velocity profile is empty carry no population and
        are dropped.

        Parameters
        ----------
        counts : Dict[int, float]
            The LSB-integer-keyed counts of the simulation.

        Returns
        -------
        List[Tuple[Tuple[int, ...], Tuple[bool, ...]]]
            The input counts split into their grid position and velocity profile.
        """
        return [
            self.split_count(key) for key in counts if (key >> self.num_grid_qubits) > 0
        ]

    def split_count(self, key: int) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
        """
        Splits a given count key into its position and velocity components.

        Counts are assumed to be obtained from :class:`.SpaceTimeGridVelocityMeasurement`,
        and the split format is the same as the input to :class:`.PointWiseSpaceTimeInitialConditions`.

        Parameters
        ----------
        key : int
            The LSB classical-bit integer key of one measurement outcome.

        Returns
        -------
        Tuple[Tuple[int, ...], Tuple[bool, ...]]
            The input count split into its grid position and velocity profile.
            The position tuple has one entry per lattice dimension.

        Raises
        ------
        ExecutionException
            If the lattice has more than 2 dimensions.
        """
        velocities = cast(
            Tuple[bool, ...],
            tuple(
                bool((key >> (self.num_grid_qubits + v)) & 1)
                for v in range(self.num_velocities_per_point)
            ),
        )

        if self.lattice.num_dims == 1:
            return ((key & ((1 << self.x_grid_qubits) - 1),), velocities)
        elif self.lattice.num_dims == 2:
            return (
                (
                    key & ((1 << self.x_grid_qubits) - 1),
                    (key >> self.x_grid_qubits) & ((1 << self.y_grid_qubits) - 1),
                ),
                velocities,
            )

        raise ExecutionException(
            f"Reinitialization not supported for lattice with {self.lattice.num_dims} dimensions."
        )

    @override
    def requires_statevector(self) -> bool:
        return False
