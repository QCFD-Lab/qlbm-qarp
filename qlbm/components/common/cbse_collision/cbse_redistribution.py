"""Permutations of states belonging to equivalence classes, based on the computational basis state encoding."""

from math import pi
from typing import override

from qarp.blocks import SimpleBlock

from qlbm.components.base import LBMComposite
from qlbm.components.common.primitives import TruncatedQFT
from qlbm.lattice.eqc.eqc import EquivalenceClass
from qlbm.lattice.spacetime.properties_base import LatticeDiscretizationProperties
from qlbm.tools.utils import is_two_pow


class EQCRedistribution(LBMComposite):
    """
    Redistribution operator for equivalence classes in the CBSE encoding.

    The operator is mathematically described in section 4 of :cite:`spacetime2`.
    Redistribution is applied before and after permutations, and consists of a controlled
    unitary operator composed of a discrete Fourier transform (DFT)-block matrix.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`equivalence_class`  The equivalence class of the operator.
    :attr:`decompose_block`    Whether to decompose the DFT block into a circuit.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import EQCRedistribution
        from qlbm.lattice import LatticeDiscretization
        from qlbm.lattice.eqc import EquivalenceClassGenerator

        # Generate some equivalence classes
        eqcs = EquivalenceClassGenerator(
            LatticeDiscretization.D3Q6
        ).generate_equivalence_classes()

        # Select one at random and draw its circuit in the schematic form
        EQCRedistribution(eqcs.pop(), decompose_block=False).plot()

    The `decompose_block` parameter can be set to ``True`` to decompose the DFT block into a circuit:

    .. plot::
        :include-source:

        from qlbm.components.common import EQCRedistribution
        from qlbm.lattice import LatticeDiscretization
        from qlbm.lattice.eqc import EquivalenceClassGenerator

        # Generate some equivalence classes
        eqcs = EquivalenceClassGenerator(
            LatticeDiscretization.D3Q6
        ).generate_equivalence_classes()

        # Select one at random and draw its decomposed circuit
        EQCRedistribution(eqcs.pop(), decompose_block=True).plot()

    """

    equivalence_class: EquivalenceClass
    """
    The equivalence class for which the redistribution is defined.
    """

    decompose_block: bool
    """
    Whether to decompose the DFT block into a circuit.
    Both settings produce the same block; qarp lowers blocks itself.
    Defaults to ``True``.
    """

    def __init__(
        self, equivalence_class: EquivalenceClass, decompose_block: bool = True
    ) -> None:
        self.equivalence_class = equivalence_class
        self.decompose_block = decompose_block
        super().__init__(
            LatticeDiscretizationProperties.get_num_velocities(
                equivalence_class.discretization
            )
        )

    @override
    def build_vanilla(self) -> None:
        num_velocities = self.n_qubits
        size = self.equivalence_class.size()
        num_qubits = (size - 1).bit_length()

        if is_two_pow(size):
            redistribution = SimpleBlock(num_qubits, name="redistribution")
            redistribution.ry([(qubit, pi / 2) for qubit in range(num_qubits)])
        else:
            redistribution = TruncatedQFT(num_qubits, size)

        # Controls on the high velocity qubits, the block on the low ones, both reversed.
        self.place_controlled(
            redistribution,
            range(num_velocities - 1, num_qubits - 1, -1),
            range(num_qubits - 1, -1, -1),
        )

    @override
    def __str__(self):
        return f"[EQCRedistribution({self.equivalence_class})]"
