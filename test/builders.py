"""Test scaffolding: a mutable circuit for preparing inputs around components.

Tests prepare basis states and compose components into them.  ``CircuitBuilder``
collects direct gate calls into leaf blocks and composed blocks into children,
and ``build()`` returns the assembled :class:`~qarp.blocks.CompositeBlock`.
"""

from typing import List, Optional, Sequence

from qarp.blocks import AnyBlock, CompositeBlock, SimpleBlock


class CircuitBuilder:
    """Mutable circuit: gates and composed blocks in call order, built on demand."""

    def __init__(self, n_qubits: int, name: str = "test_circuit") -> None:
        self.n_qubits = n_qubits
        self.name = name
        self._children: List[AnyBlock] = []
        self._leaf: Optional[SimpleBlock] = None

    def _open_leaf(self) -> SimpleBlock:
        if self._leaf is None:
            self._leaf = SimpleBlock(self.n_qubits, name=f"{self.name}_gates")
        return self._leaf

    def _flush(self) -> None:
        if self._leaf is not None:
            self._children.append(self._leaf)
            self._leaf = None

    def x(self, qubits) -> None:
        """X on a qubit or a list of qubits."""
        self._open_leaf().x(qubits)

    def h(self, qubits) -> None:
        """Hadamard on a qubit or a list of qubits."""
        self._open_leaf().h(qubits)

    def cx(self, control: int, target: int) -> None:
        """CNOT."""
        self._open_leaf().cx(control, target)

    def swap(self, first: int, second: int) -> None:
        """SWAP."""
        self._open_leaf().swap(first, second)

    def ry(self, qubit: int, theta: float) -> None:
        """Y-rotation by ``theta`` radians."""
        self._open_leaf().ry(qubit, theta)

    def measure(self, qubit: int, cbit: int) -> None:
        """Measure ``qubit`` into ``cbit``."""
        self._open_leaf().measure(qubit, cbit)

    def compose(
        self, block: AnyBlock, qubits: Optional[Sequence[int]] = None
    ) -> "CircuitBuilder":
        """Add ``block`` as a child placed on ``qubits`` (identity if ``None``).

        The placement is stored on ``block`` itself, so a block instance can be
        composed more than once only onto the same qubits.
        """
        self._flush()
        if isinstance(block, CircuitBuilder):
            block = block.build()
        if qubits is not None:
            block.target_qubits = list(qubits)
        self._children.append(block)
        return self

    def build(self) -> CompositeBlock:
        """The assembled, built block."""
        self._flush()
        block = CompositeBlock(list(self._children), self.n_qubits, name=self.name)
        block.build()
        return block
