"""Utilities for the Amplitude-Based QLBM."""

from typing import List, Tuple

from qarp.blocks import SimpleBlock
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.ab_lattice import ABLattice


def binary_to_onehot_permutation(num_qubits: int) -> List[int]:
    """Basis-state permutation mapping the first :math:`n` binary states to one-hot states.

    ``perm[column]`` is the basis state that ``column`` is mapped to.  The
    first ``num_qubits`` columns go to the one-hot states :math:`2^j`; the
    remaining columns take the unused states in ascending order.

    Parameters
    ----------
    num_qubits : int
        The number of qubits the permutation acts on.

    Returns
    -------
    List[int]
        The image of every basis state, indexed by the state itself.
    """
    dim = 2**num_qubits
    permutation = [-1] * dim
    used_rows = set()

    for j in range(num_qubits):
        permutation[j] = 1 << j
        used_rows.add(1 << j)

    remaining_rows = [r for r in range(dim) if r not in used_rows]
    for offset, column in enumerate(range(num_qubits, dim)):
        permutation[column] = remaining_rows[offset]

    return permutation


def _transpositions(permutation: List[int]) -> List[Tuple[int, int]]:
    """Decompose a permutation into transpositions, in circuit-application order.

    A cycle ``c0 -> c1 -> ... -> c_{k-1} -> c0`` equals the composition
    ``t(c0,c1) . t(c1,c2) . ... . t(c_{k-2},c_{k-1})``; composition applies
    right to left, so the circuit applies the list reversed.
    """
    seen = [False] * len(permutation)
    result: List[Tuple[int, int]] = []

    for start in range(len(permutation)):
        if seen[start] or permutation[start] == start:
            seen[start] = True
            continue

        cycle = []
        node = start
        while not seen[node]:
            seen[node] = True
            cycle.append(node)
            node = permutation[node]

        pairs = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
        result.extend(reversed(pairs))

    return result


def _emit_transposition(
    block: SimpleBlock, num_qubits: int, state_a: int, state_b: int
) -> None:
    """Append the basis-state transposition ``|state_a> <-> |state_b>``.

    The two states are first mapped, by CX/X conjugation, onto a pair that
    differs only in the pivot qubit and is all-ones elsewhere; a single MCX
    then swaps them without touching any other basis state.
    """
    difference = state_a ^ state_b
    pivot = (difference & -difference).bit_length() - 1
    linked = [i for i in range(num_qubits) if i != pivot and (difference >> i) & 1]

    pivot_bit = (state_a >> pivot) & 1
    conjugated_bits = [
        ((state_a >> i) & 1) ^ (pivot_bit if i in linked else 0)
        for i in range(num_qubits)
    ]
    to_invert = [i for i in range(num_qubits) if i != pivot and conjugated_bits[i] == 0]
    controls = [i for i in range(num_qubits) if i != pivot]

    for target in linked:
        block.cx(pivot, target)
    if to_invert:
        block.x(to_invert)

    if controls:
        block.mcx(*controls, pivot)
    else:
        block.x(pivot)

    if to_invert:
        block.x(to_invert)
    for target in reversed(linked):
        block.cx(pivot, target)


class BinaryToOHPermutation(LBMPrimitive):
    """
    Permutes the first :math:`q` basis states of the binary encoding into the :math:`q` one-hot states of the OH encoding.

    The permutation is emitted as reversible logic (X / CX / MCX), which is
    exact and needs no unitary synthesis.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import BinaryToOHPermutation
        from qlbm.lattice import OHLattice

        lattice = OHLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        BinaryToOHPermutation(lattice).plot()
    """

    lattice: ABLattice

    def __init__(self, lattice: ABLattice) -> None:
        self.lattice = lattice
        super().__init__(lattice.num_velocity_qubits, name="binary_to_onehot")

    @override
    def build_vanilla(self) -> None:
        permutation = binary_to_onehot_permutation(self.n_qubits)
        for state_a, state_b in _transpositions(permutation):
            _emit_transposition(self, self.n_qubits, state_a, state_b)

    @override
    def __str__(self) -> str:
        return f"[Primitive BinaryOHPermutation with lattice {self.lattice}]"
