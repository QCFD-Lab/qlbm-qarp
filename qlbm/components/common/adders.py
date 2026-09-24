"""Circuits implementing components of quantum adders. See :cite:`draper` and :cite:`adder`."""

from math import pi

import numpy as np
from qarp.blocks import QFTBlock, SimpleBlock
from typing_extensions import override

from qlbm.components.base import ControllableComponent, LBMComposite, LBMPrimitive
from qlbm.tools import bit_value


class ParameterizedPhaseShift(ControllableComponent):
    r"""A primitive that applies the phase-shift as part of the :class:`.ParameterizedDraperAdder` used in :class:`.Comparator`\ s.

    The rotation applied is :math:`\pm \frac{\pi}{2^{n_q - 1 - j}}`, with :math:`j` the position of the qubit (indexed starting with 0).
    Unlike the regular :class:`.PhaseShift`, the parameterized version additionally adds a phase relative to the number supplied.
    For an in-depth mathematical explanation of the procedure, consult Sections 4 and 5.5 of :cite:t:`collisionless`.
    """

    num_to_add: int
    """The number to add to the basis states encoded in the qubits."""

    positive: bool
    """Whether the operation is an addition or a subtraction."""

    def __init__(
        self,
        num_qubits: int,
        num_to_add: int,
        positive: bool = False,
        num_ctrl_qubits: int = 0,
    ) -> None:
        self.num_to_add = num_to_add
        self.positive = positive
        super().__init__(num_qubits, num_ctrl_qubits)

    @override
    def build_core(self) -> SimpleBlock:
        n = self.num_qubits
        angles = np.zeros(n)
        for qubit_index in range(n):
            dig = bit_value(self.num_to_add, qubit_index)
            for i in range(n - qubit_index):
                # (2 * positive - 1) will flip the sign if positive is False
                # This effectively inverts the circuit
                angles[i] += (
                    (2 * self.positive - 1)
                    * dig
                    * pi
                    / (2 ** (n - qubit_index - i - 1))
                )

        core = SimpleBlock(n, name="parameterized_phase_shift")
        core.p([(qubit_index, angles[qubit_index]) for qubit_index in range(n)])
        return core

    @override
    def __str__(self) -> str:
        return f"[Primitive ParameterizedPhaseShift of {self.num_qubits} qubits, num {self.num_to_add}, in direction {self.positive}, ctrl {self.num_ctrl_qubits}]"


class ParameterizedDraperAdder(LBMComposite):
    r"""A QFT-based incrementer used to perform streaming in the algorithms based on amplitude encodings.

    Incrementation and decerementation are performed as rotations on grid qubits
    that have been previously mapped to the Fourier basis.
    This happens by nesting a :class:`.ParameterizedPhaseShift` primitive
    between regular and inverse :math:`QFT`\ s.
    """

    num_qubits: int
    """The number of qubits the phase shift is performed on."""

    num_to_add: int
    """The number to add to the basis states encoded in the qubits."""

    positive: bool
    """Whether the operation is an addition or a subtraction."""

    num_ctrl_qubits: int
    """Optional additional qubits to control the operation on. If any, the control qubits trail the target qubits."""

    def __init__(
        self,
        num_qubits: int,
        num_to_add: int,
        positive: bool,
        num_ctrl_qubits: int = 0,
    ) -> None:
        self.num_qubits = num_qubits
        self.num_to_add = num_to_add
        self.positive = positive
        self.num_ctrl_qubits = num_ctrl_qubits
        super().__init__(num_qubits + num_ctrl_qubits)

    @override
    def build_vanilla(self) -> None:
        target_register = list(range(self.num_qubits))

        self.place(QFTBlock(self.num_qubits), target_register)
        self.place(
            ParameterizedPhaseShift(
                self.num_qubits, self.num_to_add, self.positive, self.num_ctrl_qubits
            )
        )
        self.place(~QFTBlock(self.num_qubits), target_register)

    @override
    def __str__(self) -> str:
        return f"[Primitive SimpleAdder on {self.num_qubits} qubits, on velocity {self.num_to_add}, in direction {self.positive}]"


class PhaseShift(LBMPrimitive):
    r"""
    A primitive that applies the phase-shift as part of the :class:`.ControlledIncrementer` used in the :class:`.MSStreamingOperator`.

    The rotation applied is :math:`\pm\frac{\pi}{2^{n_q - 1 - j}}`, with :math:`j` the position of the qubit (indexed starting with 0).
    For an in-depth mathematical explanation of the procedure and its use within QLBM,
    consult Section 4 of :cite:t:`collisionless`.
    The Draper adder was originally formulated in :cite:`draper`, while the version implemented
    here uses the one-register approach, which was
    first described in :cite:`qftadder`.
    """

    def __init__(self, num_qubits: int, positive: bool = False) -> None:
        self.num_qubits = num_qubits
        self.positive = positive
        super().__init__(num_qubits)

    @override
    def build_vanilla(self) -> None:
        # (2 * positive - 1) will flip the sign if positive is False
        # This effectively inverts the circuit
        sign = 2 * self.positive - 1
        self.p(
            [
                (qubit_index, sign * pi / (2 ** (self.num_qubits - 1 - qubit_index)))
                for qubit_index in range(self.num_qubits)
            ]
        )

    @override
    def __str__(self) -> str:
        return f"[Primitive PhaseShift of {self.num_qubits} qubits, in direction {self.positive}]"
