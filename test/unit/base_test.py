"""Placement helpers and component bases in :mod:`qlbm.components.base`.

Every case is a classical reversible map, so the oracle is the permutation
matrix of the basis-state map, compared exactly including global phase.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from qarp.blocks import AnyBlock, XnBlock

from qlbm.components.base import (
    ControllableComponent,
    LBMComposite,
    SequenceBlock,
    controlled,
    flip_if,
    on,
    x_layer,
)


class Placed(LBMComposite):
    """A composite that places the given children."""

    def __init__(
        self,
        n_qubits: int,
        children: Sequence[Tuple[AnyBlock, Optional[Sequence[int]]]],
    ):
        self.children_to_place = list(children)
        super().__init__(n_qubits)

    def build_vanilla(self) -> None:
        """Place every child on its qubits."""
        for child, qubits in self.children_to_place:
            self.place(child, qubits)


class ControlledFlip(ControllableComponent):
    """An X on one data qubit, optionally controlled."""

    def __init__(self, num_ctrl_qubits: int, controls_first: bool) -> None:
        self.controls_first = controls_first
        super().__init__(1, num_ctrl_qubits)

    def build_core(self) -> AnyBlock:
        """A single X."""
        return XnBlock(1)


class Inverter(LBMComposite):
    """A composite whose only content is :meth:`LBMComposite.invert`."""

    def __init__(self, n_qubits: int, qubits: List[int]) -> None:
        self.qubits = qubits
        super().__init__(n_qubits)

    def build_vanilla(self) -> None:
        """Invert the configured qubits."""
        self.invert(self.qubits)


def permutation(n_qubits: int, image: Callable[[int], int]) -> np.ndarray:
    """The unitary sending basis state ``b`` to ``image(b)``."""
    matrix = np.zeros((1 << n_qubits, 1 << n_qubits))
    for basis in range(1 << n_qubits):
        matrix[image(basis), basis] = 1.0
    return matrix


def bit(basis: int, qubit: int) -> int:
    """The value of ``qubit`` in basis state ``basis``."""
    return (basis >> qubit) & 1


def assert_unitary(block, expected: np.ndarray) -> None:
    """The block's unitary equals ``expected``, global phase included."""
    np.testing.assert_allclose(np.asarray(block.unitary_matrix()), expected, atol=1e-12)


def test_on_places_a_block_on_the_given_qubits():
    """An X placed on qubit 2 of three flips bit 2 only."""
    block = Placed(3, [(on(XnBlock(1), [2]), None)])

    assert_unitary(block, permutation(3, lambda b: b ^ 0b100))


def test_place_maps_the_child_qubits_in_order():
    """A two-qubit child's qubit 0 lands on the first listed parent qubit."""
    child = Placed(2, [(XnBlock(1), [0])])
    block = Placed(3, [(child, [2, 0])])

    assert_unitary(block, permutation(3, lambda b: b ^ 0b100))


def test_sequence_block_acts_as_its_children_in_order():
    """A sequence of two placed X layers flips bits 0 and 2."""
    sequence = SequenceBlock([on(XnBlock(1), [0]), on(XnBlock(1), [2])], 3)
    block = Placed(3, [(sequence, None)])

    assert_unitary(block, permutation(3, lambda b: b ^ 0b101))


def test_sequence_block_declares_its_children_as_parts():
    """The declared parts are the children, in placement order."""
    first, second = on(XnBlock(1), [0]), on(XnBlock(1), [2])

    assert SequenceBlock([first, second], 3).structure() == [first, second]


def test_sequence_block_parts_are_built_before_the_sequence_is():
    """The planner flattens the parts of an unbuilt sequence, so they are built."""
    inner = Placed(2, [(XnBlock(1), [0])])
    sequence = SequenceBlock([on(inner, [1, 2]), on(XnBlock(1), [0])], 3)

    parts = sequence.structure()

    assert [len(part.flatten()) for part in parts] == [1, 1]


def test_x_layer_flips_every_listed_qubit():
    """``x_layer([0, 2])`` flips bits 0 and 2."""
    block = Placed(3, [(x_layer([0, 2]), None)])

    assert_unitary(block, permutation(3, lambda b: b ^ 0b101))


def test_controlled_without_controls_is_the_bare_block():
    """No controls places ``inner`` unconditionally."""
    block = Placed(2, [(controlled(XnBlock(1), [], [1]), None)])

    assert_unitary(block, permutation(2, lambda b: b ^ 0b10))


@pytest.mark.parametrize(
    ("ctrl_state", "active"),
    [(None, (1, 1)), ([True, False], (1, 0)), ([False, False], (0, 0))],
)
def test_controlled_fires_only_on_the_control_state(ctrl_state, active):
    """The target flips exactly when the controls hold ``ctrl_state``."""
    block = Placed(3, [(controlled(XnBlock(1), [0, 1], [2], ctrl_state), None)])

    def image(b: int) -> int:
        return b ^ 0b100 if (bit(b, 0), bit(b, 1)) == active else b

    assert_unitary(block, permutation(3, image))


def test_flip_if_treats_inverted_controls_as_open():
    """Targets flip when control 0 is set and inverted control 1 is clear."""
    block = Placed(4, [(flip_if([0, 1], [2, 3], inverted=[1]), None)])

    def image(b: int) -> int:
        return b ^ 0b1100 if (bit(b, 0), bit(b, 1)) == (1, 0) else b

    assert_unitary(block, permutation(4, image))


def test_flip_if_ignores_inverted_qubits_that_are_not_controls():
    """An inverted non-control would be conjugated by cancelling X gates."""
    with_extra = Placed(3, [(flip_if([0], [2], inverted=[1]), None)])
    without = Placed(3, [(flip_if([0], [2]), None)])

    assert_unitary(with_extra, np.asarray(without.unitary_matrix()))
    assert_unitary(without, permutation(3, lambda b: b ^ 0b100 if bit(b, 0) else b))


@pytest.mark.parametrize(
    ("controls_first", "control_qubit", "data_qubit"), [(False, 1, 0), (True, 0, 1)]
)
def test_controllable_component_layout(controls_first, control_qubit, data_qubit):
    """Controls trail the data qubits unless ``controls_first`` is set."""
    block = ControlledFlip(1, controls_first)

    def image(b: int) -> int:
        return b ^ (1 << data_qubit) if bit(b, control_qubit) else b

    assert block.n_qubits == 2
    assert_unitary(block, permutation(2, image))


def test_controllable_component_without_controls_is_the_core():
    """Zero control qubits leaves the core unconditional."""
    assert_unitary(ControlledFlip(0, False), permutation(1, lambda b: b ^ 1))


def test_invert_with_no_qubits_places_nothing():
    """``invert([])`` adds no gates."""
    block = Inverter(2, [])

    assert len(block.flatten()) == 0
    assert_unitary(block, np.eye(4))
