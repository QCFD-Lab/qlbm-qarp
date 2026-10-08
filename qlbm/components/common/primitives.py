"""Common primitives used for multiple encodings."""

from math import acos, log2, sqrt
from typing import List, Tuple

import numpy as np
from qarp.blocks import QFTBlock, SimpleBlock, XnBlock
from typing_extensions import override

from qlbm.components.base import (
    ControllableComponent,
    LatticePrimitive,
    LBMComposite,
    LBMPrimitive,
)
from qlbm.components.common.adders import ParameterizedDraperAdder
from qlbm.lattice import Lattice
from qlbm.tools.exceptions import CircuitException
from qlbm.tools.utils import get_qubits_to_invert


class EmptyPrimitive(LatticePrimitive):
    """
    Empty primitive used for effectively not specifying parts of the QLBM algorithm.

    Useful in situations where testing the end-to-end implementation of the algorithm
    where one part of the algorithm is left out or not yet implemented.
    """

    @override
    def build_vanilla(self) -> None:
        pass

    @override
    def __str__(self) -> str:
        return f"[Primitive EmptyPrimitive with lattice {self.lattice}]"


class MCSwap(LatticePrimitive):
    """
    Decomposition of a Multi-Controlled Swap Gate into 1 multi-controlled :math:`X` gate and 2 single-controlled :math:`X` gates.

    Decomposition taken from :cite:t:`mcswap`.
    """

    def __init__(
        self,
        lattice: Lattice,
        control_qubits: List[int],
        swap_qubits: Tuple[int, int],
    ) -> None:
        self.control_qubits = control_qubits
        self.swap_qubits = swap_qubits
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        first, second = self.swap_qubits
        self.cx(second, first)
        self.mcx(*self.control_qubits, first, second)
        self.cx(second, first)

    @override
    def __str__(self) -> str:
        return f"[Primitive MCSwap with lattice {self.lattice}]"


class HammingWeightAdder(LBMComposite):
    """
    QFT-based Hamming Weight adder.

    This primitive adds the hamming weight (number of 1s) in a given register :math:`x`
    to the binary-encoded value of a second register :math:`y`.
    """

    x_register_size: int
    """
    The size of the register encoding the hamming weight value to add.
    """

    y_register_size: int
    """
    The size of the register to which the hamming weight is added.
    """

    def __init__(self, x_register_size: int, y_register_size: int) -> None:
        self.x_register_size = x_register_size
        self.y_register_size = y_register_size
        super().__init__(x_register_size + y_register_size)

    @override
    def build_vanilla(self) -> None:
        y_register = list(range(self.x_register_size, self.n_qubits))

        ladder = SimpleBlock(self.n_qubits, name="hamming_weight_ladder")
        ladder.cp(
            [
                (xi, yi, 2 * np.pi / (2 ** (self.y_register_size - k)))
                for xi in range(self.x_register_size)
                for k, yi in enumerate(y_register)
            ]
        )

        self.place(QFTBlock(self.y_register_size), y_register)
        self.place(ladder)
        self.place(~QFTBlock(self.y_register_size), y_register)

    @override
    def __str__(self):
        return f"[Primitive HWAdder with with register size {self.x_register_size} and {self.y_register_size}]"


class TruncatedQFT(LBMPrimitive):
    r"""Truncated Quantum Fourier Transform primitive used to create an equal magnitude superposition.

    For a superposition of the first :math:`k` basis states encoded in :math:`n` qubits,
    the operator consists of discrete fourier transform block of size :math:`k\times k`,
    padded with :math:`2^n - k` :math:`1`\ s on the main diagonal.
    The rationale and properties of this operator are described in :cite:`spacetime2`.
    Synthesized via qarpx Quantum Shannon Decomposition (the truncated DFT block
    is not a structured QFT circuit for general :math:`k`); construction raises a
    :class:`.CircuitException` when the synthesized unitary misses the target.
    """

    num_qubits: int
    """The number of qubits the operator acts on."""

    dft_size: int
    """The size of the discrete Fourier transform block."""

    synthesis_tolerance: float = 1e-8
    """The largest element-wise deviation from the target unitary accepted."""

    def __init__(self, num_qubits: int, dft_size: int) -> None:
        self.num_qubits = num_qubits
        self.dft_size = dft_size
        super().__init__(num_qubits)
        # The synthesis is numerical, so its result is checked, not trusted.
        error = np.max(np.abs(np.asarray(self.unitary_matrix()) - self.target()))
        if error > self.synthesis_tolerance:
            raise CircuitException(
                f"Synthesized TruncatedQFT({num_qubits}, {dft_size}) deviates from "
                f"the target unitary by {error:.1e}."
            )

    def target(self) -> np.ndarray:
        """The unitary: a normalised DFT on the first :attr:`dft_size` states, identity elsewhere."""
        k = self.dft_size
        i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        matrix = np.eye(2**self.num_qubits, dtype=complex)
        matrix[:k, :k] = np.exp(2j * np.pi * i * j / k) / np.sqrt(k)
        return matrix

    @override
    def build_vanilla(self) -> None:
        self.unitary_synthesis(self.target())

    @override
    def __str__(self):
        return f"[Primitive TuncatedQFT({self.num_qubits}, {self.dft_size})]"


def _open_controlled_h(block: SimpleBlock, control: int, targets: range) -> None:
    r"""Hadamards on ``targets`` conditioned on ``control`` being :math:`\ket{0}`."""
    block.x(control)
    for target in targets:
        block.ch(control, target)
    block.x(control)


class UniformStatePrep(ControllableComponent):
    r"""Efficient uniform state preparation primitive used to create an equal magnitude superposition over the first :math:`k` basis states.

    This is an implementation of Algorithm 1 described by :cite:t:`uniprep`.
    It is used to create an uniform magnitude superposition over arbitrary
    velocity states in :class:`.ABDiscreteUniformInitialConditions`.
    """

    num_states: int
    """The number of states to generate."""

    def __init__(
        self, num_qubits: int, num_states: int, num_ctrl_qubits: int = 0
    ) -> None:
        self.num_states = num_states
        super().__init__(num_qubits, num_ctrl_qubits)

    @override
    def build_core(self) -> SimpleBlock:
        core = SimpleBlock(self.num_qubits, name="uniform_state_prep")
        num_states = self.num_states

        # M = 1 : do nothing, stays in |0...0>
        if num_states == 1:
            return core

        # If M is a power of two, the solution is trivial: Hadamards on log2(M) qubits
        if num_states & (num_states - 1) == 0:
            core.h(list(range(int(log2(num_states)))))
            return core

        # --- General case: Algorithm 1 (Section 2.1 of the paper) ---

        # We only need n_eff = ceil(log2 M) active qubits; the rest stay in |0>
        n_eff = num_states.bit_length()
        if n_eff > self.num_qubits:
            raise CircuitException("Internal error: n_eff > num_qubits.")

        # Binary decomposition: M = \Sum_j 2^{l_j}, with 0 <= l0 < l1 < ... < lk
        bit_positions = [i for i in range(n_eff) if (num_states >> i) & 1]
        l0, l1 = bit_positions[0], bit_positions[1]
        k = len(bit_positions) - 1  # number of "higher" bits

        # Helper: safe acos for numerical stability
        def safe_acos(x: float) -> float:
            return acos(max(-1.0, min(1.0, x)))

        # Step 4: Apply X on qubits at positions l1, l2, ..., lk
        core.x(bit_positions[1:])

        # Step 5: M0 = 2^{l0}
        m_prev = 2**l0

        # Step 6–7: If l0 > 0, apply H on qubits 0..(l0-1)
        if l0 > 0:
            core.h(list(range(l0)))

        # Step 8: Apply RY(theta0) on |q_{l1}>, theta0 = -2 arccos( sqrt(M0 / M) )
        core.ry(l1, -2.0 * safe_acos(sqrt(m_prev / num_states)))

        # Step 9: Controlled H on qubits i in [l0, l1) with open control on q_{l1} == |0>
        _open_controlled_h(core, l1, range(l0, l1))

        # Steps 10–13: For-loop over remaining bits
        for m in range(1, k):
            l_m, l_next = bit_positions[m], bit_positions[m + 1]

            # Step 11: Controlled RY(theta_m) on q_{l_{m+1}} with open control on q_{l_m} == |0>
            theta_m = -2.0 * safe_acos(sqrt(2**l_m / (num_states - m_prev)))
            core.x(l_m)
            core.cry(l_m, l_next, theta_m)
            core.x(l_m)

            # Step 12: Controlled H on qubits i in [l_m, l_{m+1}) with open control on q_{l_{m+1}} == |0>
            _open_controlled_h(core, l_next, range(l_m, l_next))

            # Step 13: M_m = M_{m-1} + 2^{l_m}
            m_prev += 2**l_m

        return core

    @override
    def __str__(self):
        return f"[Primitive UniformStatePrep({self.num_qubits}, {self.num_states})]"


class AdditionConversion(LBMComposite):
    """
    Converts one basis state to another by incrementation/decrementation.

    Useful for performing permutations in which the initial superposition contains no basis states
    of the target superposition.

    The circuit utilizes a :class:`.ParameterizedDraperAdder` which controlled on the state
    of an ancilla qubit to add the difference only to the target basis state.
    """

    num_qubits: int
    """The number of qubits the states are encoded in."""

    state_from: int
    """The starting state to convert."""

    state_to: int
    """The state to convert to."""

    num_ctrl_qubits: int
    """The number of qubits to control the operation."""

    def __init__(
        self,
        num_qubits: int,
        state_from: int,
        state_to: int,
        num_ctrl_qubits: int = 0,
    ) -> None:
        self.num_qubits = num_qubits
        self.state_from = state_from
        self.state_to = state_to
        self.num_ctrl_qubits = num_ctrl_qubits
        super().__init__(num_qubits + num_ctrl_qubits + 1)

    @override
    def build_vanilla(self) -> None:
        data_register = list(range(self.num_qubits))
        ancilla = self.num_qubits
        control_qubits = list(range(ancilla + 1, self.n_qubits))

        def flag(state: int) -> None:
            # Flip the ancilla iff the data register holds |state> (and the controls are set).
            self.place(StateSetter(self.num_qubits, state), data_register)
            self.place_controlled(XnBlock(1), data_register + control_qubits, [ancilla])
            self.place(StateSetter(self.num_qubits, state), data_register)

        flag(self.state_from)
        self.place(
            ParameterizedDraperAdder(
                self.num_qubits,
                abs(self.state_to - self.state_from),
                self.state_to > self.state_from,
                self.num_ctrl_qubits + 1,
            )
        )
        flag(self.state_to)

    @override
    def __str__(self):
        return f"[Primitive AdditionConversion({self.num_qubits}, {self.state_from}, {self.state_to})]"


class StateSetter(LBMPrimitive):
    r"""
    Permutes the superposition such that a target state :math:`\ket{k}` is permuted to :math:`\ket{1}^{\otimes n}`.

    The primitive acts a single layer of :math:`\mathrm{X}` gates on the qubit
    indices that have value :math:`\ket{0}` for the input state.
    """

    num_qubits: int
    """The number of qubits the state is encoded in."""

    state_to_set: int
    r"""The state to convert to :math:`\ket{1}^{\otimes n}`"""

    def __init__(self, num_qubits: int, state_to_set: int) -> None:
        self.num_qubits = num_qubits
        self.state_to_set = state_to_set
        super().__init__(num_qubits)

    @override
    def build_vanilla(self) -> None:
        qubits_to_invert = get_qubits_to_invert(self.state_to_set, self.num_qubits)
        if qubits_to_invert:
            self.x(qubits_to_invert)

    @override
    def __str__(self):
        return f"[Primitive StateSetter({self.num_qubits}, {self.state_to_set}]"
