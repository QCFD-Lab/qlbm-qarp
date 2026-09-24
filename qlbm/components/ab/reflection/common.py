"""Common utilities for reflection in the :class:`.ABQLBM` algorithm."""

from typing import List, Tuple

from qarp.blocks import SimpleBlock
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.base import ControllableComponent
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException


class _Permutation:
    """
    Emits gates on a velocity-register leaf, addressing qubits in the encoding's bit order.

    The amplitude-based encoding numbers its velocity qubits from the most
    significant bit, so ``reverse_bits`` mirrors the indices of the gate tables.
    """

    def __init__(self, num_qubits: int, reverse_bits: bool, name: str) -> None:
        self.block = SimpleBlock(num_qubits, name=name)
        self.num_qubits = num_qubits
        self.reverse_bits = reverse_bits

    def q(self, index: int) -> int:
        return self.num_qubits - 1 - index if self.reverse_bits else index

    def x(self, indices: List[int]) -> None:
        self.block.x([self.q(index) for index in indices])

    def cx(self, control: int, target: int) -> None:
        self.block.cx(self.q(control), self.q(target))

    def mcx(self, controls: List[int], target: int) -> None:
        self.block.mcx(*(self.q(c) for c in controls), self.q(target))

    def swap(self, first: int, second: int) -> None:
        self.block.swap(self.q(first), self.q(second))


class ABBounceBackReflectionPermutation(ControllableComponent):
    """
    Permutes velocity state to implement bounce-back reflection in the amplitude-based encoding for :math:`D_dQ_q` discretizations.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABEncodingType, ABBounceBackReflectionPermutation
        from qlbm.lattice import LatticeDiscretization

        ABBounceBackReflectionPermutation(
            4, LatticeDiscretization.D2Q9, ABEncodingType.AB
        ).plot()

    """

    discretization: LatticeDiscretization
    """
    The lattice discretization the permutation adheres to.
    """

    encoding: ABEncodingType
    """
    The type of encoding to permute for.
    """

    controls_first = True
    """The additional control qubits lead the velocity qubits."""

    def __init__(
        self,
        num_qubits: int,
        discretization: LatticeDiscretization,
        encoding: ABEncodingType,
        num_ctrl_qubits: int = 0,
    ) -> None:
        self.discretization = discretization
        self.encoding = encoding
        super().__init__(num_qubits, num_ctrl_qubits, name="ab_bb_permutation")

    @override
    def build_core(self) -> SimpleBlock:
        if self.discretization != LatticeDiscretization.D2Q9:
            raise LatticeException("AB reflection only currently supported in D2Q9")
        gate = _Permutation(
            self.num_qubits, self.encoding == ABEncodingType.AB, "ab_bb_permutation"
        )
        match self.encoding:
            case ABEncodingType.OH:
                gate.swap(1, 3)
                gate.swap(2, 4)
                gate.swap(5, 7)
                gate.swap(6, 8)
            case ABEncodingType.AB:
                # 1 <-> 3
                gate.x([0, 1])
                gate.mcx([0, 1, 3], 2)
                gate.x([0, 1])
                # 2 <-> 4
                gate.x([0, 3])
                gate.cx(1, 2)
                gate.mcx([0, 2, 3], 1)
                gate.cx(1, 2)
                gate.x([0, 3])
                # 5 <-> 7
                gate.x([0])
                gate.mcx([0, 1, 3], 2)
                gate.x([0])
                # 6 <-> 8
                gate.cx(0, 1)
                gate.cx(0, 2)
                gate.x([3])
                gate.mcx([1, 2, 3], 0)
                gate.cx(0, 2)
                gate.cx(0, 1)
                gate.x([3])
            case _:
                raise LatticeException(f"Unsupported lattice encoding: {self.encoding}")
        return gate.block

    @override
    def __str__(self) -> str:
        return f"[Primitive ABBounceBackReflectionPermutation with {self.num_qubits} qubits on {self.discretization}]"


class ABSpecularReflectionPermutation(ControllableComponent):
    """
    Permutes velocity state to implement bounce-back reflection in the amplitude-based encoding for :math:`D_dQ_q` discretizations.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABEncodingType, ABSpecularReflectionPermutation
        from qlbm.lattice import LatticeDiscretization

        ABSpecularReflectionPermutation(
            4, LatticeDiscretization.D2Q9, ABEncodingType.AB, (True, True)
        ).plot()

    """

    discretization: LatticeDiscretization
    """
    The lattice discretization the permutation adheres to.
    """

    encoding: ABEncodingType
    """
    The type of encoding to permute for.
    """

    supported_reflection_types: List[str] = ["bounceback", "specular"]

    reflect_in_dim: Tuple[bool, ...]
    controls_first = True
    """The additional control qubits lead the velocity qubits."""

    def __init__(
        self,
        num_qubits: int,
        discretization: LatticeDiscretization,
        encoding: ABEncodingType,
        reflect_in_dim: Tuple[bool, ...],
        num_ctrl_qubits: int = 0,
    ) -> None:
        self.discretization = discretization
        self.encoding = encoding
        self.reflect_in_dim = reflect_in_dim
        super().__init__(num_qubits, num_ctrl_qubits, name="ab_sr_permutation")

    @override
    def build_core(self) -> SimpleBlock:
        if self.discretization != LatticeDiscretization.D2Q9:
            raise LatticeException("AB reflection only currently supported in D2Q9")
        gate = _Permutation(
            self.num_qubits, self.encoding == ABEncodingType.AB, "ab_sr_permutation"
        )
        match self.encoding:
            case ABEncodingType.OH:
                if self.reflect_in_dim[0]:
                    gate.swap(1, 3)
                    gate.swap(5, 6)
                    gate.swap(8, 7)
                if self.reflect_in_dim[1]:
                    gate.swap(2, 4)
                    gate.swap(5, 8)
                    gate.swap(6, 7)
            case ABEncodingType.AB:
                if self.reflect_in_dim[0]:
                    # 1 <-> 3
                    gate.x([0, 1])
                    gate.mcx([0, 1, 3], 2)
                    gate.x([0, 1])
                    # 5 <-> 6
                    gate.x([0])
                    gate.cx(3, 2)
                    gate.mcx([0, 1, 2], 3)
                    gate.cx(3, 2)
                    gate.x([0])
                    # 8 <-> 7
                    gate.cx(0, 1)
                    gate.cx(0, 2)
                    gate.cx(0, 3)
                    gate.mcx([1, 2, 3], 0)
                    gate.cx(0, 3)
                    gate.cx(0, 2)
                    gate.cx(0, 1)
                if self.reflect_in_dim[1]:
                    # 2 <-> 4
                    gate.x([0, 3])
                    gate.cx(1, 2)
                    gate.mcx([0, 2, 3], 1)
                    gate.cx(1, 2)
                    gate.x([0, 3])
                    # 5 <-> 8
                    gate.x([2])
                    gate.cx(0, 1)
                    gate.cx(0, 3)
                    gate.mcx([1, 2, 3], 0)
                    gate.cx(0, 3)
                    gate.cx(0, 1)
                    gate.x([2])
                    # 6 <-> 7
                    gate.x([0])
                    gate.mcx([0, 1, 2], 3)
                    gate.x([0])
            case _:
                raise LatticeException(f"Unsupported lattice encoding: {self.encoding}")
        return gate.block

    @override
    def __str__(self) -> str:
        return f"[Primitive ABSpecularReflectionPermutation with {self.num_qubits} qubits on {self.discretization}, reflection {self.reflect_in_dim}]"
