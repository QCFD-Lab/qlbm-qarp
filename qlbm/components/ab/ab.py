"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`."""

from typing_extensions import override

from qlbm.components.ab.reflection import ABZoneAgnosticReflectionOperator
from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.components.base import LBMAlgorithm
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.tools.exceptions import CircuitException

from .streaming import ABStreamingOperator


class ABQLBM(LBMAlgorithm):
    """
    Implementation of the **A** mplitude **B** ased QLBM (ABQLBM).

    The algorithm consists of interleaving steps of streaming and boundary conditions.
    Note that there is **no** collision in this algorithm as of yet.
    Details of the general framework can be found in :cite:`collisionless`.
    The ABQLBM works with :math:`D_dQ_q` discretizations only.
    For multi-speed alternatives, see :class:`.MSQLBM`.

    .. warning::

        The complete algorithm currently supports D2Q9 only because its
        reflection operators are implemented only for D2Q9. The standalone
        :class:`.ABStreamingOperator` additionally supports D1Q3 with an
        :class:`.ABLattice`.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABQLBM
        from qlbm.lattice import ABLattice

        # Example with streaming only for simplicity.
        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABQLBM(lattice).plot()
    """

    lattice: ABLattice

    def __init__(self, lattice: ABLattice, use_agnostic_bcs: bool = False) -> None:
        if use_agnostic_bcs and lattice.has_multiple_geometries():
            raise CircuitException(
                "Zone-agnostic boundary conditions are not supported "
                "with multiple geometries. Use use_agnostic_bcs=False "
                "or specify a single geometry."
            )
        self.use_agnostic_bcs = use_agnostic_bcs
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        self.place(ABStreamingOperator(self.lattice))
        if self.use_agnostic_bcs:
            self.place(ABZoneAgnosticReflectionOperator(self.lattice))
        else:
            self.place(ABReflectionOperator(self.lattice))

    @override
    def __str__(self) -> str:
        return f"[Algorithm ABQLBM with lattice {self.lattice}]"
