"""Quantum circuits that perform arithmetic comparison operations."""

from typing import List

from qarp.blocks import XnBlock
from typing_extensions import override

from qlbm.components.base import LBMComposite
from qlbm.components.common.adders import ParameterizedDraperAdder
from qlbm.components.common.arithmetic import DraperQFTAdder
from qlbm.tools import ComparatorMode


class TwoRegisterComparator(LBMComposite):
    """
    Quantum comparator primitive that compares the states of 2 registers of ``num_qubits`` qubits a :class:`~qlbm.tools.ComparatorMode`.

    The generate circuit is of size ``2*num_qubits+1``, where the last qubit of the register holds the boolean result.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common.comparators import TwoRegisterComparator
        from qlbm.tools import ComparatorMode

        # Compare two registers of size 4
        TwoRegisterComparator(num_qubits=4, mode=ComparatorMode.LT).plot()
    """

    def __init__(self, num_qubits: int, mode: ComparatorMode) -> None:
        self.num_qubits = num_qubits
        self.mode = mode
        super().__init__(2 * num_qubits + 1)

    @override
    def build_vanilla(self) -> None:
        x_register = list(range(self.num_qubits))
        y_register = list(range(self.num_qubits, 2 * self.num_qubits))
        output_qubit = 2 * self.num_qubits

        match self.mode:
            case ComparatorMode.GT | ComparatorMode.LE:
                self.__place_gt(x_register, y_register, output_qubit)
            case ComparatorMode.LT | ComparatorMode.GE:
                self.__place_gt(y_register, x_register, output_qubit)
            case _:
                raise ValueError("Invalid Comparator Mode")

        if self.mode in (ComparatorMode.LE, ComparatorMode.GE):
            self.place(XnBlock(1), [output_qubit])

    def __place_gt(
        self, x_register: List[int], y_register: List[int], output_qubit: int
    ) -> None:
        n = self.num_qubits
        self.place(XnBlock(n), y_register)
        self.place(
            DraperQFTAdder(n, kind="half"), x_register + y_register + [output_qubit]
        )
        self.place(~DraperQFTAdder(n, kind="fixed"), x_register + y_register)
        self.place(XnBlock(n), y_register)

    @override
    def __str__(self) -> str:
        return f"[Primitive TwoRegisterComparator of {self.num_qubits} qubits, mode={self.mode}]"


class SingleRegisterComparator(LBMComposite):
    """
    Quantum comparator primitive that compares a quantum state of ``num_qubits`` qubits and an integer ``num_to_compare`` with respect to a :class:`~qlbm.tools.ComparatorMode`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        Number of qubits encoding the integer to compare.
    :attr:`num_to_compare`    The integer to compare against.
    :attr:`mode`              The :class:`~qlbm.tools.ComparatorMode` used to compare the two numbers.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common.comparators import SingleRegisterComparator
        from qlbm.tools.utils import ComparatorMode

        # On a 5 qubit register, compare the number 3
        SingleRegisterComparator(num_qubits=5,
                                num_to_compare=3,
                                mode=ComparatorMode.LT).plot()
    """

    def __init__(
        self, num_qubits: int, num_to_compare: int, mode: ComparatorMode
    ) -> None:
        self.num_qubits = num_qubits
        self.num_to_compare = num_to_compare
        self.mode = mode
        super().__init__(num_qubits)

    @override
    def build_vanilla(self) -> None:
        n, num, mode = self.num_qubits, self.num_to_compare, self.mode
        largest = 2 ** (n - 1) - 1

        # LE and GT reduce to LT and GE on a shifted number; GT of the largest
        # encodable number is the identity.
        match mode:
            case ComparatorMode.LE:
                num, mode = (
                    (0, ComparatorMode.GE)
                    if num == largest
                    else (num + 1, ComparatorMode.LT)
                )
            case ComparatorMode.GT:
                if num == largest:
                    return
                num, mode = num + 1, ComparatorMode.GE
            case ComparatorMode.LT | ComparatorMode.GE:
                pass
            case _:
                raise ValueError("Invalid Comparator Mode")

        self.place(ParameterizedDraperAdder(n, num, positive=False))
        self.place(ParameterizedDraperAdder(n - 1, num, positive=True), range(n - 1))
        if mode == ComparatorMode.GE:
            self.place(XnBlock(1), [n - 1])

    @override
    def __str__(self) -> str:
        return f"[Primitive Comparator of {self.num_qubits} and {self.num_to_compare}, mode={self.mode}]"
