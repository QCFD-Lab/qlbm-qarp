"""Collision operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from typing_extensions import override

from qlbm.components.base import LBMComposite
from qlbm.components.common.cbse_collision.cbse_permutation import (
    EQCPermutation,
)
from qlbm.components.common.cbse_collision.cbse_redistribution import (
    EQCRedistribution,
)
from qlbm.lattice.eqc.eqc_generator import (
    EquivalenceClassGenerator,
)
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)


class EQCCollisionOperator(LBMComposite):
    """
    Collision operator based on the equivalence class abstraction described in section 5 of :cite:`spacetime2`.

    Consists of a permutation, redistribution, and inverse permutation of the velocity qubits.
    This operator is designed to be applied to a single velocity register, which can be repeated depending on the encoding.
    Used in the :class:`.GenericSpaceTimeCollisionOperator` and :class:`.GenericLQLGACollisionOperator`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`discretization`    The discretization for which this collision operator is defined.
    :attr:`num_velocities`    The number of velocities in the discretization.
    ========================= ======================================================================

    Simple D2Q4 example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import EQCCollisionOperator
        from qlbm.lattice import LatticeDiscretization

        # Select a discretization and draw its circuit
        EQCCollisionOperator(
            LatticeDiscretization.D2Q4
        ).plot()

    More complex D3Q6 example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import EQCCollisionOperator
        from qlbm.lattice import LatticeDiscretization

        # Select a discretization and draw its circuit
        EQCCollisionOperator(
            LatticeDiscretization.D3Q6
        ).plot()
    """

    discretization: LatticeDiscretization
    """
    The discretization of the lattice for which this collision operator is defined.
    """

    num_velocities: int
    """
    The number of velocities in the discretization."""

    def __init__(self, discretization: LatticeDiscretization) -> None:
        self.discretization = discretization
        self.num_velocities = LatticeDiscretizationProperties.get_num_velocities(
            discretization
        )
        super().__init__(self.num_velocities)

    @override
    def build_vanilla(self) -> None:
        """Applies the collision operator of all equivalence classes onto one velocity register."""
        for eqc in EquivalenceClassGenerator(
            self.discretization
        ).generate_equivalence_classes():
            self.place(EQCPermutation(eqc))
            self.place(EQCRedistribution(eqc))
            self.place(EQCPermutation(eqc, inverse=True))

    @override
    def __str__(self) -> str:
        return f"[EQCCollisionOperator for equi {self.discretization}]"
