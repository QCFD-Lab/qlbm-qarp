"""Measurement operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from typing import Tuple

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.utils import flatten, get_qubits_to_invert


class SpaceTimeGridVelocityMeasurement(LatticePrimitive):
    """A primitive that implements a measurement operation on the grid and the local velocity qubits.

    Used at the end of the simulation to extract information from the quantum state.
    Together, the information from the local and grid qubits can be used for on-the-fly reinitialization.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    ========================= ======================================================================


    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime import SpaceTimeGridVelocityMeasurement
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
                "geometry": [],
            },
        )

        # Draw the measurement circuit
        SpaceTimeGridVelocityMeasurement(lattice=lattice).plot()
    """

    lattice: SpaceTimeLattice

    def __init__(self, lattice: SpaceTimeLattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        qubits_to_measure = self.lattice.grid_index() + self.lattice.velocity_index(0)
        self.measure([(qubit, cbit) for cbit, qubit in enumerate(qubits_to_measure)])

    @override
    def __str__(self) -> str:
        return f"[SpaceTimeGridVelocityMeasurement for lattice {self.lattice}]"


class SpaceTimePointWiseMassMeasurement(LatticePrimitive):
    """
    A primitive that performs mass measurement.

    WIP.
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        gridpoint: Tuple[int, int],
        velocity_index_to_measure: int,
    ) -> None:
        self.gridpoint = gridpoint
        self.velocity_index_to_measure = velocity_index_to_measure
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        qubits_to_invert = flatten(
            [
                [
                    qubit + self.lattice.properties.get_num_previous_grid_qubits(dim)
                    for qubit in get_qubits_to_invert(
                        coord, self.lattice.num_gridpoints[dim].bit_length()
                    )
                ]
                for dim, coord in enumerate(self.gridpoint)
            ]
        )
        control_qubits = self.lattice.grid_index() + self.lattice.velocity_index(
            0, self.velocity_index_to_measure
        )
        target_qubits = self.lattice.ancilla_mass_index()

        if qubits_to_invert:
            self.x(qubits_to_invert)
        for target_qubit in target_qubits:
            self.mcx(*control_qubits, target_qubit)
        self.measure(target_qubits[0], 0)

    @override
    def __str__(self) -> str:
        return f"[SpaceTimePointWiseMassMeasurement for lattice {self.lattice}, gridpoint {self.gridpoint}, velocity {self.velocity_index_to_measure}]"
