"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm, or Quantum Transport Method first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`.

This is a common entrypoint that supports implementations based on the :class:`.MSLattice` and :class:`.ABLattice`.

Implementations can be found in the :class:`MSQLBM` and :class:`.ABQLBM`, respectively.
"""

from typing import cast

from typing_extensions import override

from qlbm.components.ab.ab import ABQLBM
from qlbm.components.base import LBMAlgorithm
from qlbm.components.ms.msqlbm import MSQLBM
from qlbm.lattice import MSLattice
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.tools.exceptions import CircuitException, LatticeException


class CQLBM(LBMAlgorithm):
    """The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`.

    Implementations based on lattices with the DdQq discretization use the :class:`.ABQLBM`:

    .. code-block:: python

        from qlbm.components import CQLBM
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        CQLBM(lattice).plot()

    Implementations where the number of velocities is defined per dimension delegate to the :class:`.MSQLBM`.

    .. code-block:: python

        from qlbm.components import CQLBM
        from qlbm.lattice import MSLattice

        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        CQLBM(lattice).plot()

    """

    lattice: AmplitudeLattice

    def __init__(
        self, lattice: AmplitudeLattice, use_agnostic_bcs: bool = False
    ) -> None:
        if isinstance(lattice, MSLattice) and use_agnostic_bcs:
            raise CircuitException("Agnostic BCs are not supported for the MSQLBM.")
        if not isinstance(lattice, (MSLattice, ABLattice)):
            raise LatticeException(
                f"CQLBM does not support lattices of type {type(lattice)}"
            )
        self.use_agnostic_bcs = use_agnostic_bcs
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        if isinstance(self.lattice, MSLattice):
            self.place(MSQLBM(cast(MSLattice, self.lattice), group_velocities=True))
        else:
            self.place(
                ABQLBM(
                    cast(ABLattice, self.lattice),
                    use_agnostic_bcs=self.use_agnostic_bcs,
                )
            )

    @override
    def __str__(self) -> str:
        return f"[Algorithm CQLBM with lattice {self.lattice}]"
