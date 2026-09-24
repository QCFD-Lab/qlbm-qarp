"""Tests for :class:`.TruncatedQFT` against the exact truncated DFT."""

import numpy as np
import pytest

from qlbm.components.common.primitives import TruncatedQFT
from qlbm.tools.exceptions import CircuitException


def truncated_dft(num_qubits: int, dft_size: int) -> np.ndarray:
    """Normalised DFT of size ``dft_size`` padded with the identity."""
    omega = np.exp(2j * np.pi / dft_size)
    matrix = np.eye(1 << num_qubits, dtype=complex)
    for row in range(dft_size):
        for column in range(dft_size):
            matrix[row, column] = omega ** (row * column) / np.sqrt(dft_size)
    return matrix


@pytest.mark.parametrize(
    ("num_qubits", "dft_size"),
    [(1, 2), (2, 3), (2, 4), (3, 5), (3, 6), (4, 10), (5, 30), (6, 64)],
)
def test_unitary_is_the_truncated_dft(num_qubits, dft_size):
    """The operator equals the padded DFT exactly, global phase included."""
    block = TruncatedQFT(num_qubits, dft_size)

    np.testing.assert_allclose(
        np.asarray(block.unitary_matrix()),
        truncated_dft(num_qubits, dft_size),
        atol=1e-9,
    )


def test_uniform_superposition_over_the_first_states():
    """|0> maps to an equal superposition of the first ``dft_size`` states."""
    block = TruncatedQFT(3, 6)
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0

    output = np.asarray(block.statevector(initial_state=state))

    np.testing.assert_allclose(output[:6], np.full(6, 1 / np.sqrt(6)), atol=1e-9)
    np.testing.assert_allclose(output[6:], 0, atol=1e-9)


def test_inaccurate_synthesis_is_rejected(monkeypatch):
    """A synthesis that misses the target raises instead of returning it."""
    monkeypatch.setattr(
        TruncatedQFT,
        "build_vanilla",
        lambda self: self.unitary_synthesis(np.eye(1 << self.num_qubits)),
    )
    with pytest.raises(CircuitException, match="deviates from the target"):
        TruncatedQFT(2, 3)
