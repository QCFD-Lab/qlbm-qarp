"""Quantum circuits used for measurement in the :class:`ABQLBM` algorithm."""

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.ab_lattice import ABLattice


class ABGridMeasurement(LatticePrimitive):
    """
    Grid measurement for the :class:`ABQLBM` algorithm.

    By default, this component measures only the grid register. Setting
    ``measure_velocity_qubits=True`` appends the velocity register to the same
    classical register. For a ``16 x 8`` D2Q9 lattice, this produces 7 grid bits
    by default, 11 total bits for an :class:`.ABLattice` with velocity
    measurement, or 16 total bits for an :class:`.OHLattice` with velocity
    measurement.

    With :class:`.ABLattice`, the measured velocity bits encode the binary D2Q9
    channel index. With :class:`.OHLattice`, they form a nine-bit one-hot value.

    .. warning::

        Qiskit displays classical bit strings from the highest classical-bit
        index on the left to index 0 on the right. This circuit maps x-coordinate
        bits first, followed by the remaining dimensions and then velocity bits.
        For the 2D example below, the displayed groups therefore appear in the
        reverse order: velocity, y, then x.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABGridMeasurement
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABGridMeasurement(lattice).plot()

        # Include the four binary velocity-index qubits as well.
        ABGridMeasurement(lattice, measure_velocity_qubits=True).draw("mpl")

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
