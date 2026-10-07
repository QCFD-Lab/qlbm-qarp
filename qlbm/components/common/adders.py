"""Circuits implementing components of quantum adders. See :cite:`draper` and :cite:`adder`."""

from math import pi
from typing import Iterable, NamedTuple, Sequence, Tuple

import numpy as np
from qarp.blocks import QFTBlock, SimpleBlock
from typing_extensions import override

from qlbm.components.base import ControllableComponent, LBMComposite, LBMPrimitive
from qlbm.tools import bit_value


class ShiftTerm(NamedTuple):
    """One shift of a :class:`StreamingShift`, in the block's local frame."""

    positive: bool
    """Whether the register moves up (``True``) or down."""

    controls: Tuple[int, ...]
    """The control qubits the shift is conditioned on, as local indices."""

    ctrl_state: Tuple[bool, ...]
    """The control values that activate the shift."""


def shift_on(
    positive: bool,
    controls: Sequence[int],
    on_qubits: Sequence[int],
    inverted: Iterable[int] = (),
) -> ShiftTerm:
    r"""
    A shift of a :class:`StreamingShift` whose control qubits are ``controls``.

    Parameters
    ----------
    positive : bool
        Whether to increment or decrement.
    controls : Sequence[int]
        The parent qubits the shift block is controlled on, in placement order.
    on_qubits : Sequence[int]
        The controls this shift is conditioned on, a subset of ``controls``.
    inverted : Iterable[int]
        The controls active on :math:`\ket{0}`.

    Returns
    -------
    ShiftTerm
        The shift in the block's local frame.
    """
    open_controls = set(inverted)
    return ShiftTerm(
        positive,
        tuple(controls.index(qubit) for qubit in on_qubits),
        tuple(qubit not in open_controls for qubit in on_qubits),
    )


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


class StreamingShift(LBMComposite):
    r"""
    Moves a register one position up or down for every control pattern that matches.

    The register is mapped to the Fourier basis by a :math:`QFT`, each shift
    is a :class:`.PhaseShift` controlled on its pattern, and an inverse
    :math:`QFT` maps the register back.  Together the gates send a basis
    state :math:`\ket{c}\ket{r}` to :math:`\ket{c}\ket{(r + \sum_j s_j) \bmod 2^n}`,
    where the sum runs over the shifts whose controls match :math:`c` and
    :math:`s_j = \pm 1`; :meth:`classical_action` declares this permutation.
    The control qubits precede the register.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        The number of qubits of the shifted register.
    :attr:`num_ctrl_qubits`   The number of control qubits, which precede the register.
    :attr:`shifts`            The shifts, each a :class:`ShiftTerm`.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import StreamingShift

        # A 3-qubit register moved up when the control is |1> and down when it is |0>
        StreamingShift(3, 1, [(True, [0], [True]), (False, [0], [False])]).plot()
    """

    num_qubits: int
    """The number of qubits of the shifted register."""

    num_ctrl_qubits: int
    """The number of control qubits, which precede the register."""

    shifts: Tuple[ShiftTerm, ...]
    """The shifts, with control indices below :attr:`num_ctrl_qubits`.  Immutable:
    the declared action must stay the one the gates were built from."""

    def __init__(
        self,
        num_qubits: int,
        num_ctrl_qubits: int,
        shifts: Sequence[Tuple[bool, Sequence[int], Sequence[bool]]],
    ) -> None:
        self.num_qubits = num_qubits
        self.num_ctrl_qubits = num_ctrl_qubits
        self.shifts = tuple(
            ShiftTerm(bool(positive), tuple(controls), tuple(map(bool, ctrl_state)))
            for positive, controls, ctrl_state in shifts
        )
        super().__init__(num_ctrl_qubits + num_qubits)

    @override
    def build_vanilla(self) -> None:
        register = list(range(self.num_ctrl_qubits, self.n_qubits))
        self.place(QFTBlock(self.num_qubits), register)
        for positive, controls, ctrl_state in self.shifts:
            self.place_controlled(
                PhaseShift(self.num_qubits, positive),
                controls,
                register,
                ctrl_state,
            )
        self.place(~QFTBlock(self.num_qubits), register)

    def classical_action(self, indices: np.ndarray) -> np.ndarray:
        """
        The image of each basis state, declared for qarp's structured execution.

        Parameters
        ----------
        indices : np.ndarray
            Basis-state indices of this block, control qubits in the low bits.

        Returns
        -------
        np.ndarray
            The indices the gates map them to, ``int64`` of the same shape.
        """
        states = np.asarray(indices, dtype=np.int64)
        controls = states & ((1 << self.num_ctrl_qubits) - 1)
        offset = np.zeros_like(states)
        for positive, qubits, ctrl_state in self.shifts:
            mask = sum(1 << qubit for qubit in qubits)
            active = sum(
                1 << qubit for qubit, value in zip(qubits, ctrl_state) if value
            )
            offset += np.where(controls & mask == active, 1 if positive else -1, 0)
        register = (states >> self.num_ctrl_qubits) + offset
        return controls | (
            (register & ((1 << self.num_qubits) - 1)) << self.num_ctrl_qubits
        )

    @override
    def __str__(self) -> str:
        return f"[Primitive StreamingShift of {self.num_qubits} qubits, {len(self.shifts)} shifts, ctrl {self.num_ctrl_qubits}]"
