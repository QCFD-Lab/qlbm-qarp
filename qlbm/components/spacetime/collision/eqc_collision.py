"""Collision operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from typing_extensions import override

from qlbm.components.base import LBMOperator
from qlbm.components.common.cbse_collision.cbse_collision import EQCCollisionOperator
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


class GenericSpaceTimeCollisionOperator(LBMOperator):
    """
    A generic space-time collision operator that can be used to apply any gate to the velocities of a grid location.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`gate`              The gate to apply to the velocities.
    ========================= ======================================================================
    """

    lattice: SpaceTimeLattice

    def __init__(self, lattice: SpaceTimeLattice, timestep: int) -> None:
        self.timestep = timestep
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        properties = self.lattice.properties
        stride = properties.get_num_velocities_per_point()
        first = properties.get_num_grid_qubits()
        for start in range(
            first, first + properties.get_num_velocity_qubits(self.timestep), stride
        ):
            self.place(
                EQCCollisionOperator(properties.get_discretization()),
                range(start, start + stride),
            )

    @override
    def __str__(self) -> str:
        return f"[GenericSpaceTimeCollisionOperator for discretization {self.lattice.properties.get_discretization()}]"
