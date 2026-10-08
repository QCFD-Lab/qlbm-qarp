"""QFT arithmetic blocks that qarp does not ship.

:class:`~qarp.blocks.QFTBlock` always includes the swap layer, so the phase
ladders target bit-reversed positions inside the sum register and the two
swap layers cancel exactly.
"""

from math import pi
from typing import List

from qarp.blocks import QFTBlock, SimpleBlock
from typing_extensions import override

from qlbm.components.base import LBMComposite


def ccp(
    block: SimpleBlock, control_a: int, control_b: int, target: int, theta: float
) -> None:
    r"""
    Emit a doubly-controlled phase gate, exact including global phase.

    Uses the standard 5-gate ``cp``/``cx`` identity: the gate is diagonal, so
    the identity is exact and stays inside one leaf block.

    Parameters
    ----------
    block : SimpleBlock
        The leaf to append to.
    control_a : int
        The first control qubit.
    control_b : int
        The second control qubit.
    target : int
        The target qubit.
    theta : float
        The phase applied to the :math:`\ket{111}` component, in radians.
    """
    block.cp(control_a, target, theta / 2)
    block.cx(control_a, control_b)
    block.cp(control_b, target, -theta / 2)
    block.cx(control_a, control_b)
    block.cp(control_b, target, theta / 2)


class DraperQFTAdder(LBMComposite):
    r"""
    In-place QFT adder :cite:`draper`.

    Computes :math:`\ket{a, b} \mapsto \ket{a, a + b}` with ``a`` on qubits
    ``[0, n)`` and ``b`` on ``[n, 2n)``; ``kind="fixed"`` adds modulo
    :math:`2^n`, ``kind="half"`` carries into an extra qubit at ``2n``.
    """

    num_state_qubits: int
    """The size of each input register."""

    kind: str
    """``"fixed"`` (modular) or ``"half"`` (with carry-out)."""

    def __init__(self, num_state_qubits: int, kind: str = "fixed") -> None:
        self.num_state_qubits = num_state_qubits
        self.kind = kind
        num_sum_qubits = num_state_qubits + (1 if kind == "half" else 0)
        super().__init__(num_state_qubits + num_sum_qubits, name=f"draper_{kind}")

    @override
    def build_vanilla(self) -> None:
        n = self.num_state_qubits
        num_sum_qubits = self.n_qubits - n
        a_register = list(range(n))
        sum_register = list(range(n, self.n_qubits))

        ladder = SimpleBlock(self.n_qubits, name="draper_ladder")
        ladder.cp(
            [
                (a_register[j], sum_register[num_sum_qubits - 1 - (j + k)], pi / (2**k))
                for j in range(n)
                for k in range(n - j)
            ]
        )
        if self.kind == "half":
            ladder.cp(
                [
                    (
                        a_register[n - j - 1],
                        sum_register[num_sum_qubits - 1 - n],
                        pi / (2 ** (j + 1)),
                    )
                    for j in range(n)
                ]
            )

        self.place(QFTBlock(num_sum_qubits), sum_register)
        self.place(ladder)
        self.place(~QFTBlock(num_sum_qubits), sum_register)

    @override
    def __str__(self) -> str:
        return f"[Primitive DraperQFTAdder({self.num_state_qubits}, {self.kind})]"


class RGQFTMultiplier(LBMComposite):
    r"""
    Out-of-place QFT multiplier.

    Maps :math:`\ket{a}\ket{b}\ket{0} \mapsto \ket{a}\ket{b}
    \ket{a \cdot b \bmod 2^m}` for :math:`m` result qubits, following
    Ruiz-Perez et al. (arXiv:1411.5949), Fig. 3 and 5.  Register layout:
    ``a`` on :math:`[0, n)`, ``b`` on :math:`[n, 2n)`, result on
    :math:`[2n, 2n + m)`.
    """

    num_state_qubits: int
    """The size of each of the two input registers."""

    num_result_qubits: int
    """The size of the result register; the product is computed modulo :math:`2^m`."""

    def __init__(self, num_state_qubits: int, num_result_qubits: int) -> None:
        self.num_state_qubits = num_state_qubits
        self.num_result_qubits = num_result_qubits
        super().__init__(
            2 * num_state_qubits + num_result_qubits, name="rgqft_multiplier"
        )

    @override
    def build_vanilla(self) -> None:
        n, m = self.num_state_qubits, self.num_result_qubits
        a_register: List[int] = list(range(n))
        b_register: List[int] = list(range(n, 2 * n))
        out_register: List[int] = list(range(2 * n, 2 * n + m))

        ladder = SimpleBlock(self.n_qubits, name="multiplier_ladder")
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                for k in range(1, m + 1):
                    # Bit-reversed target: compensates QFTBlock's swap layer.
                    ccp(
                        ladder,
                        a_register[n - j],
                        b_register[n - i],
                        out_register[m - k],
                        (2 * pi) / (2 ** (i + j + k - 2 * n)),
                    )

        self.place(QFTBlock(m), out_register)
        self.place(ladder)
        self.place(~QFTBlock(m), out_register)

    @override
    def __str__(self) -> str:
        return f"[Primitive RGQFTMultiplier({self.num_state_qubits}, {self.num_result_qubits})]"
