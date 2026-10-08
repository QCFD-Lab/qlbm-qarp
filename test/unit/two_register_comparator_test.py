"""Exhaustive and superposition tests for :class:`.TwoRegisterComparator`."""

from typing import List, Sequence, Tuple

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.common.comparators import TwoRegisterComparator
from qlbm.tools.utils import ComparatorMode
from test.builders import CircuitBuilder


def _state_index(x_value: int, y_value: int, out_value: int, num_qubits: int) -> int:
    return x_value + (y_value << num_qubits) + (out_value << (2 * num_qubits))


def _extract_register_value(state_index: int, start: int, num_qubits: int) -> int:
    # Reconstruct an integer from a contiguous little-endian qubit slice.
    return sum(((state_index >> (start + bit)) & 1) << bit for bit in range(num_qubits))


def _builder(num_total_qubits: int, prep: Sequence[Tuple[str, int]]) -> CircuitBuilder:
    # Fresh builder per call: a built block cannot be re-opened for more gates.
    builder = CircuitBuilder(num_total_qubits, name="comparator_test")
    for gate, qubit in prep:
        getattr(builder, gate)(qubit)
    return builder


def _statevector(builder: CircuitBuilder) -> np.ndarray:
    # LSB-indexed.
    block = builder.build()
    return np.asarray(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


def _assert_equiv(actual: np.ndarray, expected: np.ndarray) -> None:
    # Equality up to a global phase.
    assert actual.shape == expected.shape
    pivot = int(np.argmax(np.abs(expected)))
    assert abs(expected[pivot]) > 1e-8
    phase = actual[pivot] / expected[pivot]
    assert abs(abs(phase) - 1.0) < 1e-8
    np.testing.assert_allclose(actual, expected * phase, atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    "mode",
    [ComparatorMode.LT, ComparatorMode.LE, ComparatorMode.GT, ComparatorMode.GE],
)
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_two_register_comparator_all_modes(mode: ComparatorMode, num_qubits: int):
    """Comparator preserves both inputs and writes the inequality to the ancilla."""
    # Build one comparator circuit per (mode, width) and reuse it for all inputs.
    comparator = TwoRegisterComparator(num_qubits=num_qubits, mode=mode)
    operator = mode.to_operator()

    # Exhaustively verify all (x, y) inputs for this register width.
    for x_value in range(2**num_qubits):
        for y_value in range(2**num_qubits):
            # Prepare |x>|y>|0> in computational basis (LSB at the lower qubit index).
            prep: List[Tuple[str, int]] = []
            for bit in range(num_qubits):
                if (x_value >> bit) & 1:
                    prep.append(("x", bit))
                if (y_value >> bit) & 1:
                    prep.append(("x", num_qubits + bit))

            builder = _builder(2 * num_qubits + 1, prep)

            # Apply the reversible two-register comparator.
            builder.compose(comparator)

            # The circuit is deterministic on basis input; a single basis state should have unit amplitude.
            amplitudes = _statevector(builder)
            max_index = max(
                range(len(amplitudes)), key=lambda idx: abs(amplitudes[idx])
            )
            max_amplitude = amplitudes[max_index]

            # Ensure no superposition/leakage due to incorrect uncomputation.
            assert abs(abs(max_amplitude) - 1.0) < 1e-9

            x_after = _extract_register_value(max_index, 0, num_qubits)
            y_after = _extract_register_value(max_index, num_qubits, num_qubits)
            out_after = (max_index >> (2 * num_qubits)) & 1

            # Comparator must preserve both input registers exactly.
            assert x_after == x_value
            assert y_after == y_value

            # Output ancilla must encode the selected inequality mode.
            assert out_after == int(operator(x_value, y_value))


@pytest.mark.parametrize(
    "mode",
    [ComparatorMode.LT, ComparatorMode.LE, ComparatorMode.GT, ComparatorMode.GE],
)
def test_two_register_comparator_superposition_inputs(mode: ComparatorMode):
    """Comparator routes every |x,y,0> amplitude to |x,y,f(x,y)> coherently."""
    num_qubits = 2
    num_total_qubits = 2 * num_qubits + 1
    comparator = TwoRegisterComparator(num_qubits=num_qubits, mode=mode)
    operator = mode.to_operator()

    def assert_expected_superposition(prep: Sequence[Tuple[str, int]]):
        input_state = _statevector(_builder(num_total_qubits, prep))

        # Build expected output by routing each |x,y,0> amplitude to |x,y,f(x,y)>.
        expected = np.zeros(2**num_total_qubits, dtype=complex)
        for x_value in range(2**num_qubits):
            for y_value in range(2**num_qubits):
                source_idx = _state_index(x_value, y_value, 0, num_qubits)
                target_idx = _state_index(
                    x_value,
                    y_value,
                    int(operator(x_value, y_value)),
                    num_qubits,
                )
                expected[target_idx] = input_state[source_idx]

        evolved = _builder(num_total_qubits, prep)
        evolved.compose(comparator)
        actual_state = _statevector(evolved)

        _assert_equiv(actual_state, expected)

    # Superposition over x only (y fixed to 2).
    assert_expected_superposition(
        [("h", 0), ("h", 1), ("x", num_qubits + 1)],
    )

    # Superposition over y only (x fixed to 1).
    assert_expected_superposition(
        [("x", 0), ("h", num_qubits), ("h", num_qubits + 1)],
    )

    # Superposition over both registers.
    assert_expected_superposition(
        [("h", 0), ("h", 1), ("h", num_qubits), ("h", num_qubits + 1)],
    )
