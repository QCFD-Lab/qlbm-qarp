"""Measurement operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class LQLGAGridVelocityMeasurement(LatticePrimitive):
    # TODO: Improve documentation
    """
    Measurement operator for the :class:`.LQLGA` algorithm.

    This operator measures the velocity qubits at each grid point in the LQLGA lattice.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.lqlga import LQLGAGridVelocityMeasurement
        from qlbm.lattice import LQLGALattice

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 5},
                    "velocities": "D1Q3",
                },
                "geometry": [],
            },
        )

        LQLGAGridVelocityMeasurement(lattice=lattice).plot()
    """

    lattice: LQLGALattice

    def __init__(self, lattice: LQLGALattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        self.measure([(qubit, qubit) for qubit in range(self.lattice.num_base_qubits)])

    @override
    def __str__(self) -> str:
        return f"[LQLGAGridVelocityMeasurement for lattice {self.lattice}]"
