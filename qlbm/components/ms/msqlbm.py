"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`."""

from logging import getLogger

from typing_extensions import override

from qlbm.components.base import LBMAlgorithm
from qlbm.lattice import MSLattice
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import get_time_series

from .bounceback_reflection import BounceBackReflectionOperator
from .specular_reflection import SpecularReflectionOperator
from .streaming import MSStreamingOperator, StreamingAncillaPreparation

logger = getLogger("qlbm")


class MSQLBM(LBMAlgorithm):
    """The end-to-end algorithm of the Multi-Speed Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`.

    This implementation supports 2D and 3D simulations with with cuboid objects
    with either bounce-back or specular reflection boundary conditions.

    The algorithm is composed of three steps that are repeated according to a CFL counter:

    #. Streaming performed by the :class:`.MSStreamingOperator` increments or decrements the positions of particles on the grid.
    #. :class:`.BounceBackReflectionOperator` and :class:`.SpecularReflectionOperator` reflect the particles that come in contact with :class:`.Block` obstacles encoded in the :class:`.MSLattice`.
    #. The :class:`.StreamingAncillaPreparation` resets the state of the ancilla qubits for the next CFL counter substep.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`group_velocities`  Whether to group velocities into 1 streaming step in the CFL series.
    ========================= ======================================================================
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice, group_velocities: bool = False) -> None:
        self.group_velocities = group_velocities
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        # Assumes equal velocities in all dimensions
        time_series = get_time_series(
            2 ** self.lattice.num_velocities[0].bit_length(),
            group_velocities=self.group_velocities,
        )
        logger.debug(f"MSQLBM CFL time series: {time_series}")

        for velocities_to_increment in time_series:
            self.place(MSStreamingOperator(self.lattice, velocities_to_increment))

            if self.lattice.shapes["specular"]:
                if not all(
                    isinstance(shape, Block)
                    for shape in self.lattice.shapes["specular"]
                ):
                    raise LatticeException(
                        "All shapes with the 'specular' boundary condition must be of type Block for the MSQLBM algorithm. "
                    )
                self.place(
                    SpecularReflectionOperator(
                        self.lattice,
                        self.lattice.shapes["specular"],  # type: ignore
                    )
                )

            for bc in ["bounceback", "specular"]:
                if self.lattice.shapes[bc]:
                    if not all(
                        isinstance(shape, Block)
                        for shape in self.lattice.shapes["specular"]
                    ):
                        raise LatticeException(
                            f"All shapes with the {bc} boundary condition must be cuboids for the MSQLBM algorithm. "
                        )
                self.place(
                    BounceBackReflectionOperator(
                        self.lattice,
                        self.lattice.shapes["bounceback"],  # type: ignore
                    )
                )

            for dim in range(self.lattice.num_dims):
                self.place(
                    StreamingAncillaPreparation(
                        self.lattice, velocities_to_increment, dim
                    )
                )

    @override
    def __str__(self) -> str:
        return f"[Algorithm MSQLBM with lattice {self.lattice}]"
