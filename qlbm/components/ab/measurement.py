"""Quantum circuits used for measurement in the :class:`ABQLBM` algorithm."""

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.ab_lattice import ABLattice


class ABGridMeasurement(LatticePrimitive):
    """
    Grid measurement for the :class:`ABQLBM` algorithm.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABGridMeasurement
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 32, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABGridMeasurement(lattice).plot()

    """

    lattice: ABLattice

    def __init__(
        self, lattice: ABLattice, measure_velocity_qubits: bool = False
    ) -> None:
        self.measure_velocity_qubits = measure_velocity_qubits
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        qubits_to_measure = self.lattice.grid_index() + (
            self.lattice.velocity_index() if self.measure_velocity_qubits else []
        )
        self.measure([(qubit, cbit) for cbit, qubit in enumerate(qubits_to_measure)])

    @override
    def __str__(self) -> str:
        return f"[Primitive ABEGridMeasurement with lattice {self.lattice}]"
