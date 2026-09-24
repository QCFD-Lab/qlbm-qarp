"""Permutations of states belonging to equivalence classes, based on the computational basis state encoding."""

from typing import List, Tuple, override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.eqc.eqc import EquivalenceClass
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import CircuitException


class EQCPermutation(LBMPrimitive):
    """Applies a permutation to the velocity qubits of an equivalence Sclass in the CBSE encoding.

    This is used as part of the PRP collision operator described in section 5 of :cite:`spacetime2`.
    Utilized in the :class:`.EQCCollisionOperator`.

    ============================== =================================================================
    Attribute              Summary
    ============================== =================================================================
    :attr:`equivalence_class`      The equivalence class of the operator.
    :attr:`inverse`                Whether to apply the inverse permutation.
    ============================== =================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import EQCPermutation
        from qlbm.lattice import LatticeDiscretization
        from qlbm.lattice.eqc import EquivalenceClassGenerator

        # Generate some equivalence classes
        eqcs = EquivalenceClassGenerator(
            LatticeDiscretization.D3Q6
        ).generate_equivalence_classes()

        # Select one at random and draw its circuit
        EQCPermutation(eqcs.pop(), inverse=False).plot()

    """

    equivalence_class: EquivalenceClass
    """
    The equivalence class for which the permutation is defined.
    """

    inverse: bool
    """
    Whether to apply the inverse permutation.
    """

    def __init__(
        self, equivalence_class: EquivalenceClass, inverse: bool = False
    ) -> None:
        self.equivalence_class = equivalence_class
        self.inverse = inverse
        super().__init__(
            LatticeDiscretizationProperties.get_num_velocities(
                equivalence_class.discretization
            )
        )

    @override
    def build_vanilla(self) -> None:
        # Every gate is self-inverse, so the inverse permutation is the reversed sequence.
        gates = self.__gates()
        for gate, *qubits in reversed(gates) if self.inverse else gates:
            getattr(self, gate)(*qubits)

    def __gates(self) -> List[Tuple]:
        match self.equivalence_class.discretization:
            case LatticeDiscretization.D1Q3:
                return [("cx", 0, 1), ("cx", 0, 2)]
            case LatticeDiscretization.D2Q4:
                return [("cx", 1, 2), ("cx", 0, 1), ("cx", 0, 3)]
            case LatticeDiscretization.D3Q6:
                return self.__gates_d3q6()
            case _:
                raise CircuitException(
                    f"Collision not yet supported for discretization {self.equivalence_class.discretization}."
                )

    def __gates_d3q6(self) -> List[Tuple]:
        match self.equivalence_class.id():
            case (2, [0, 0, 0]):
                return [
                    ("cx", 2, 3),
                    ("cx", 5, 4),
                    ("cx", 1, 2),
                    ("cx", 1, 3),
                    ("cx", 1, 5),
                    ("cx", 0, 2),
                    ("cx", 0, 4),
                    ("cx", 0, 5),
                ]
            case (4, [0, 0, 0]):
                return [
                    ("ccx", 4, 5, 3),
                    ("ccx", 0, 2, 4),
                    ("ccx", 0, 1, 2),
                    ("ccx", 0, 1, 5),
                    ("x", 1),
                    ("cx", 1, 0),
                ]
            case (3, [1, 0, 0]):
                return [
                    ("cx", 0, 3),
                    ("cx", 1, 2),
                    ("ccx", 0, 5, 4),
                    ("cx", 1, 5),
                    ("swap", 0, 1),
                ]
            case (3, [-1, 0, 0]):
                return [
                    ("cx", 1, 2),
                    ("cx", 5, 4),
                    ("cx", 1, 5),
                    ("x", 0),
                    ("swap", 0, 1),
                ]
            case (3, [0, 1, 0]):
                return [("cx", 0, 2), ("cx", 5, 3), ("cx", 0, 5), ("x", 4)]
            case (3, [0, -1, 0]):
                return [("cx", 5, 3), ("cx", 0, 2), ("cx", 0, 5), ("x", 1)]
            case (3, [0, 0, 1]):
                return [("cx", 1, 3), ("cx", 0, 1), ("cx", 0, 4), ("x", 5)]
            case (3, [0, 0, -1]):
                return [("cx", 0, 1), ("cx", 4, 3), ("cx", 0, 4), ("x", 2)]
            case _:
                raise CircuitException(
                    f"Collision not yet supported for discretization {self.equivalence_class.discretization} and equivalence class {self.equivalence_class.id()}."
                )

    @override
    def __str__(self) -> str:
        return f"[EQCPermutation for eqc {self.equivalence_class}]"
