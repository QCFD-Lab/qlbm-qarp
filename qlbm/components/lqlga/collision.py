"""Collision operators for the :class:`.LQLGA` algorithm."""

from typing_extensions import override

from qlbm.components.base import LBMOperator
from qlbm.components.common.cbse_collision.cbse_collision import EQCCollisionOperator
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class GenericLQLGACollisionOperator(LBMOperator):
    """
    Equivalence class-based LGA collision operator for the :class:`.LQLGA` algorithm.

    This operator applies the :class:`.EQCCollisionOperator` operator to all velocity qubits at each grid point.
    """

    lattice: LQLGALattice

    def __init__(self, lattice: LQLGALattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        stride = self.lattice.num_velocities_per_point
        for start in range(0, self.lattice.num_base_qubits, stride):
            self.place(
                EQCCollisionOperator(self.lattice.discretization),
                range(start, start + stride),
            )

    @override
    def __str__(self) -> str:
        return f"[GenericSpaceTimeCollisionOperator for discretization {self.lattice.discretization}]"
