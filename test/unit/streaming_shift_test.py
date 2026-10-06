"""``StreamingShift`` declares the basis-state permutation its gates implement."""

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.common import StreamingShift

# (num_qubits, num_ctrl_qubits, shifts): every control pattern the sites use.
SHAPES = [
    (3, 0, [(True, [], [])]),
    (2, 1, [(True, [0], [True])]),
    (3, 1, [(False, [0], [False])]),
    (3, 2, [(True, [0, 1], [True, False]), (False, [0, 1], [False, True])]),
    (3, 2, [(True, [0, 1], [True, True]), (True, [0, 1], [True, True])]),
    (3, 1, [(True, [0], [True]), (False, [0], [True])]),
    (4, 3, [(True, [0], [True]), (True, [1], [True]), (False, [2], [True])]),
    (2, 3, [(True, [0, 2], [True, False]), (False, [1], [False])]),
]


def gate_permutation(block) -> np.ndarray:
    """The image of every basis state under the block's gates, phase included.

    The simulator only sees commands, so the declaration plays no part.
    """
    commands, n = list(block.flatten()), block.n_qubits
    simulator = qx.QarpSimulator()
    images = np.empty(1 << n, dtype=np.int64)
    for index in range(1 << n):
        state = np.zeros(1 << n, dtype=np.complex128)
        state[index] = 1
        out = np.asarray(simulator.statevector(commands, n, initial_state=state))
        images[index] = int(np.argmax(np.abs(out)))
        assert np.isclose(out[images[index]], 1, atol=1e-9, rtol=0), index
    return images


def test_unconditional_positive_shift_increments_modulo_register_size():
    """With no controls a positive shift is ``r -> (r + 1) mod 2^n``."""
    shift = StreamingShift(3, 0, [(True, [], [])])

    images = shift.classical_action(np.arange(8, dtype=np.int64))

    assert images.tolist() == [1, 2, 3, 4, 5, 6, 7, 0]


def test_unconditional_negative_shift_decrements_modulo_register_size():
    """With no controls a negative shift is ``r -> (r - 1) mod 2^n``."""
    shift = StreamingShift(3, 0, [(False, [], [])])

    images = shift.classical_action(np.arange(8, dtype=np.int64))

    assert images.tolist() == [7, 0, 1, 2, 3, 4, 5, 6]


def test_controlled_shift_moves_the_register_only_on_the_active_control():
    """Control ``|1>`` on the low qubit shifts the two register qubits above it."""
    shift = StreamingShift(2, 1, [(True, [0], [True])])

    images = shift.classical_action(np.arange(8, dtype=np.int64))

    # index = control | register << 1; register 3 wraps to 0.
    assert images.tolist() == [0, 3, 2, 5, 4, 7, 6, 1]


@pytest.mark.parametrize("num_qubits,num_ctrl_qubits,shifts", SHAPES)
def test_declared_action_matches_the_gates(num_qubits, num_ctrl_qubits, shifts):
    """The declared images are the ones the gates produce, phase included."""
    shift = StreamingShift(num_qubits, num_ctrl_qubits, shifts)
    indices = np.arange(1 << shift.n_qubits, dtype=np.int64)

    np.testing.assert_array_equal(
        shift.classical_action(indices), gate_permutation(shift)
    )


def test_width_is_controls_plus_register():
    """The block spans the control qubits followed by the register."""
    assert StreamingShift(3, 2, [(True, [0], [True])]).n_qubits == 5
