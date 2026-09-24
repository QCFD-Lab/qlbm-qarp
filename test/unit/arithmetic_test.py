"""Basis-state tests for the QFT arithmetic in :mod:`qlbm.components.common.arithmetic`.

Every input basis state must map to exactly one output basis state, the one
integer arithmetic predicts, with amplitude 1: the adder and multiplier are
permutations, so any stray phase or leakage is a defect.
"""

from cmath import exp

import numpy as np
import pytest
from qarp.blocks import SimpleBlock

from qlbm.components.common.arithmetic import DraperQFTAdder, RGQFTMultiplier, ccp


def _apply(block, basis_index: int) -> np.ndarray:
    state = np.zeros(1 << block.n_qubits, dtype=np.complex128)
    state[basis_index] = 1.0
    return np.asarray(block.statevector(initial_state=state))


def _assert_maps_to(block, basis_in: int, basis_out: int) -> None:
    output = _apply(block, basis_in)
    expected = np.zeros_like(output)
    expected[basis_out] = 1.0
    np.testing.assert_allclose(output, expected, atol=1e-10)


@pytest.mark.parametrize("num_state_qubits", [1, 2, 3])
def test_fixed_adder_adds_modulo_the_register_size(num_state_qubits):
    """``|a>|b> -> |a>|a + b mod 2^n>`` for every pair of inputs."""
    n = num_state_qubits
    adder = DraperQFTAdder(n, kind="fixed")
    for a in range(1 << n):
        for b in range(1 << n):
            _assert_maps_to(adder, a | (b << n), a | (((a + b) % (1 << n)) << n))


@pytest.mark.parametrize("num_state_qubits", [1, 2, 3])
def test_half_adder_carries_into_the_extra_qubit(num_state_qubits):
    """``|a>|b>|0> -> |a>|a + b>`` with the sum register one qubit wider."""
    n = num_state_qubits
    adder = DraperQFTAdder(n, kind="half")
    assert adder.n_qubits == 2 * n + 1
    for a in range(1 << n):
        for b in range(1 << n):
            _assert_maps_to(adder, a | (b << n), a | ((a + b) << n))


@pytest.mark.parametrize(
    ("num_state_qubits", "num_result_qubits"),
    [(1, 2), (2, 4), (2, 3), (3, 3), (3, 2)],
)
def test_multiplier_writes_the_product_modulo_the_result_size(
    num_state_qubits, num_result_qubits
):
    """``|a>|b>|0> -> |a>|b>|a * b mod 2^m>`` for every pair of inputs."""
    n, m = num_state_qubits, num_result_qubits
    multiplier = RGQFTMultiplier(n, m)
    assert multiplier.n_qubits == 2 * n + m
    for a in range(1 << n):
        for b in range(1 << n):
            product = (a * b) % (1 << m)
            _assert_maps_to(
                multiplier, a | (b << n), a | (b << n) | (product << (2 * n))
            )


@pytest.mark.parametrize("theta", [0.3, np.pi / 2, -1.7])
def test_ccp_is_the_doubly_controlled_phase(theta):
    """The unitary is ``diag(1, ..., 1, e^{i theta})`` on its three qubits."""
    block = SimpleBlock(3, name="ccp")
    ccp(block, 0, 1, 2, theta)
    block.build()
    expected = np.diag([1.0] * 7 + [exp(1j * theta)])
    np.testing.assert_allclose(np.asarray(block.unitary_matrix()), expected, atol=1e-12)
