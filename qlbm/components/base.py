"""Base classes for the quantum components of QLBMs.

Every component is a qarp block: leaves (:class:`LBMPrimitive`) emit gates,
trees (:class:`LBMComposite`) place child blocks.  Components build on
construction, so subclasses set every attribute ``build_vanilla`` reads
before calling ``super().__init__``.
"""

from typing import Iterable, List, Optional, Sequence

from qarp.blocks import (
    AnyBlock,
    CompositeBlock,
    CompositeBlockBase,
    ControlledBlock,
    SimpleBlock,
    XnBlock,
)

from qlbm.lattice import Lattice


def on(block: AnyBlock, qubits: Sequence[int]) -> AnyBlock:
    """
    Return ``block`` placed on ``qubits`` of the block it is added to.

    Parameters
    ----------
    block : AnyBlock
        The block to place; it must not be placed anywhere else.
    qubits : Sequence[int]
        The parent qubits the block acts on, in the block's order.

    Returns
    -------
    AnyBlock
        The same block, with its ``target_qubits`` set.
    """
    block.target_qubits = list(qubits)
    return block


def x_layer(qubits: Sequence[int]) -> XnBlock:
    """
    A layer of :math:`X` gates placed on ``qubits``.

    Parameters
    ----------
    qubits : Sequence[int]
        The parent qubits to invert; must be non-empty.

    Returns
    -------
    XnBlock
        The placed layer.
    """
    return XnBlock(len(qubits), target_qubits=list(qubits))


def controlled(
    inner: AnyBlock,
    controls: Sequence[int],
    targets: Sequence[int],
    ctrl_state: Optional[Sequence[bool]] = None,
) -> AnyBlock:
    """
    ``inner`` controlled on ``controls`` and placed on ``targets``.

    With no controls the bare ``inner`` is placed, so callers need no
    special case for an unconditional operation.

    Parameters
    ----------
    inner : AnyBlock
        The block to control.
    controls : Sequence[int]
        The parent's control qubits.
    targets : Sequence[int]
        The parent qubits ``inner`` acts on, in its order.
    ctrl_state : Sequence[bool] | None
        The control values that activate ``inner``; all ``True`` if ``None``.

    Returns
    -------
    AnyBlock
        The placed block.
    """
    if not controls:
        return on(inner, targets)
    return ControlledBlock(
        inner,
        num_controls=len(controls),
        ctrl_state=None if ctrl_state is None else list(ctrl_state),
        target_qubits=list(controls) + list(targets),
    )


def flip_if(
    controls: Sequence[int], targets: Sequence[int], inverted: Iterable[int] = ()
) -> AnyBlock:
    r"""
    Flip ``targets`` when every control holds its active value.

    The active value is :math:`\ket{1}`, or :math:`\ket{0}` for the
    ``inverted`` controls.  An ``inverted`` qubit that is not a control is
    ignored.

    Parameters
    ----------
    controls : Sequence[int]
        The parent's control qubits.
    targets : Sequence[int]
        The parent qubits to flip.
    inverted : Iterable[int]
        The controls active on :math:`\ket{0}`.

    Returns
    -------
    AnyBlock
        The placed multi-controlled multi-target :math:`X`.
    """
    inverted = set(inverted)
    return controlled(
        XnBlock(len(targets)),
        controls,
        targets,
        ctrl_state=[qubit not in inverted for qubit in controls],
    )


class SequenceBlock(CompositeBlock):
    """
    A composite whose children are its declared parts.

    qarp's structured execution plans a block part by part when it declares
    them, instead of deriving the block's span as a whole first.  Use it for
    a sequence whose every child plans well on its own (declared permutations,
    small classical blocks); a sequence of loose gates plans better as a span.
    """

    def structure(self) -> List[AnyBlock]:
        """
        The children in placement order, declared for qarp's structured execution.

        Returns
        -------
        List[AnyBlock]
            The blocks this sequence was given, built or not.
        """
        return list(self.blocks)


class LBMPrimitive(SimpleBlock):
    """
    Base class for all primitive-level quantum components.

    A primitive is a small, isolated, and structurally parameterizable
    circuit that emits gates directly.  Subclasses implement
    ``build_vanilla`` with the native gate builders (``x``, ``cx``, ``swap``,
    ``mcx``, ``cp``, ``ry``, ... in scalar or bulk-list form).
    """

    def __init__(self, n_qubits: int, name: Optional[str] = None) -> None:
        super().__init__(n_qubits, name=name or type(self).__name__)
        self.build()


class LatticePrimitive(LBMPrimitive):
    """
    A primitive spanning the full width of a :class:`.Lattice`.

    Qubits are addressed by the global indices the lattice's register
    helpers return.
    """

    lattice: Lattice

    def __init__(self, lattice: Lattice, name: Optional[str] = None) -> None:
        self.lattice = lattice
        super().__init__(lattice.n_qubits, name)


class LBMComposite(CompositeBlockBase):
    """
    Base class for components assembled from other blocks.

    Subclasses implement ``build_vanilla`` by placing children with
    :meth:`place`; a composite never emits gates of its own.
    """

    def __init__(self, n_qubits: int, name: Optional[str] = None) -> None:
        super().__init__(n_qubits, name=name or type(self).__name__)
        self.build()

    def place(self, child: AnyBlock, qubits: Optional[Sequence[int]] = None) -> None:
        """
        Wire ``child`` onto ``qubits`` of this block (identity mapping if ``None``).

        Parameters
        ----------
        child : AnyBlock
            The block to add; it must not be placed anywhere else.
        qubits : Sequence[int] | None
            The qubits of this block the child acts on, in the child's order.
        """
        self.add_wired_child(child if qubits is None else on(child, qubits))

    def invert(self, qubits: Sequence[int]) -> None:
        """
        Place a layer of :math:`X` gates on ``qubits`` (nothing if empty).

        Parameters
        ----------
        qubits : Sequence[int]
            The qubits of this block to invert.
        """
        if qubits:
            self.place(x_layer(qubits))

    def place_controlled(
        self,
        inner: AnyBlock,
        controls: Sequence[int],
        targets: Sequence[int],
        ctrl_state: Optional[Sequence[bool]] = None,
    ) -> None:
        """
        Wire ``inner`` controlled on ``controls`` and acting on ``targets``.

        With no controls the bare ``inner`` is placed, so callers need no
        special case for an unconditional operation.

        Parameters
        ----------
        inner : AnyBlock
            The block to control.
        controls : Sequence[int]
            The control qubits of this block.
        targets : Sequence[int]
            The qubits of this block ``inner`` acts on, in its order.
        ctrl_state : Sequence[bool] | None
            The control values that activate ``inner``; all ``True`` if ``None``.
        """
        self.place(controlled(inner, controls, targets, ctrl_state))


class ControllableComponent(LBMComposite):
    """
    A component whose whole circuit is optionally controlled on extra qubits.

    The ``num_ctrl_qubits`` control qubits trail the ``num_qubits`` data
    qubits, or precede them when ``controls_first`` is set.  Subclasses
    implement :meth:`build_core`, the uncontrolled circuit over the data qubits.
    """

    num_qubits: int
    """The number of data qubits."""

    num_ctrl_qubits: int
    """Optional additional qubits to control the operation on."""

    controls_first: bool = False
    """Whether the control qubits precede the data qubits."""

    def __init__(
        self, num_qubits: int, num_ctrl_qubits: int = 0, name: Optional[str] = None
    ) -> None:
        self.num_qubits = num_qubits
        self.num_ctrl_qubits = num_ctrl_qubits
        super().__init__(num_qubits + num_ctrl_qubits, name)

    def build_core(self) -> AnyBlock:
        """
        The uncontrolled circuit over the ``num_qubits`` data qubits.

        Returns
        -------
        AnyBlock
            A block of width ``num_qubits``.
        """
        raise NotImplementedError

    def build_vanilla(self) -> None:
        """Place the core, controlled on the control qubits if there are any."""
        if self.controls_first:
            controls = range(self.num_ctrl_qubits)
            data = range(self.num_ctrl_qubits, self.n_qubits)
        else:
            data = range(self.num_qubits)
            controls = range(self.num_qubits, self.n_qubits)
        self.place_controlled(self.build_core(), controls, data)


class LBMOperator(LBMComposite):
    """
    Base class for all operator-level quantum components.

    An operator component implements a specific physical operation
    corresponding to the classical LBM (streaming, collision, etc.).
    Operators are inferred based on the structure of a :class:`.Lattice`
    object of an appropriate encoding and span its full width.
    """

    lattice: Lattice

    def __init__(self, lattice: Lattice, name: Optional[str] = None) -> None:
        self.lattice = lattice
        super().__init__(lattice.n_qubits, name)


class LBMAlgorithm(LBMComposite):
    """
    Base class for all algorithm-level quantum components.

    An algorithm composes operators into one time step of a QLBM over the
    full width of a :class:`.Lattice`.
    """

    lattice: Lattice

    def __init__(self, lattice: Lattice, name: Optional[str] = None) -> None:
        self.lattice = lattice
        super().__init__(lattice.n_qubits, name)
