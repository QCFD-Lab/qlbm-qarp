"""WIP."""

from typing_extensions import override

from qlbm.components.base import LBMOperator
from qlbm.components.common.primitives import TruncatedQFT
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException


class ABEAveragedCollisionOperator(LBMOperator):
    """WIP."""

    lattice: ABLattice

    def __init__(self, lattice: ABLattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        if self.lattice.discretization != LatticeDiscretization.D1Q3:
            raise LatticeException("ABE only currently supported in D1Q3")
        self.place(
            TruncatedQFT(
                self.lattice.num_velocity_qubits, self.lattice.num_velocities_per_point
            ),
            self.lattice.velocity_index(),
        )

    @override
    def __str__(self) -> str:
        return f"[Operator ABEAveragedCollision with lattice {self.lattice}]"
