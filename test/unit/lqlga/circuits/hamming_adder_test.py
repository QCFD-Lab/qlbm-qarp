"""Sampling tests for the Hamming-weight adder used by the LQLGA family.

``QarpSimulator.run`` returns ``dict[int, int]`` keyed by the LSB-indexed
classical register, so a key
``k`` carries qubit ``i`` in bit ``i``: the 4-qubit ``x`` register occupies
bits 0-3 and the 4-qubit ``y`` register bits 4-7, ``y`` LSB first.
"""

import qarpx as qx

from qlbm.components.common import HammingWeightAdder
from test.builders import CircuitBuilder

NUM_QUBITS = 8
SHOTS = 128
SEED = 7


def hamming_weight(value: int) -> int:
    """Number of set bits in ``value``."""
    return int(value).bit_count()


def x_register(key: int, size: int = 4) -> int:
    """Value of the low ``size``-qubit register in a counts key."""
    return key & ((1 << size) - 1)


def y_register(key: int, offset: int = 4, size: int = 4) -> int:
    """Value of the register starting at qubit ``offset`` in a counts key."""
    return (key >> offset) & ((1 << size) - 1)


def get_counts(builder: CircuitBuilder, num_shots: int = SHOTS):
    """Measure every qubit into the matching cbit and sample the circuit."""
    for qubit in range(builder.n_qubits):
        builder.measure(qubit, qubit)
    block = builder.build()
    return (
        qx.QarpSimulator()
        .run(block.flatten(), block.n_qubits, num_shots, seed=SEED)
        .counts
    )


def test_hamming_adder_all_0s():
    """Adding onto ``|0...0>`` leaves the register in the all-zero state."""
    builder = CircuitBuilder(NUM_QUBITS)
    builder.compose(HammingWeightAdder(3, 5))
    counts = get_counts(builder)

    assert len(counts) == 1
    assert all(key == 0 for key in counts)


def test_hamming_adder_1plus0():
    """Weight 1 in ``x`` added onto ``y = 0``; ``x`` is preserved."""
    builder = CircuitBuilder(NUM_QUBITS)
    builder.x(3)
    builder.compose(HammingWeightAdder(4, 4))
    counts = get_counts(builder)

    assert len(counts) == 1
    assert all(x_register(key) == 0b1000 for key in counts)
    assert all(y_register(key) == 1 for key in counts)


def test_hamming_adder_2plus0():
    """Weight 2 in ``x`` added onto ``y = 0``."""
    builder = CircuitBuilder(NUM_QUBITS)
    builder.x([0, 3])
    builder.compose(HammingWeightAdder(4, 4))
    counts = get_counts(builder)

    assert len(counts) == 1
    assert all(y_register(key) == 2 for key in counts)


def test_hamming_adder_2plus4():
    """Weight 2 in ``x`` added onto ``y = 4``."""
    builder = CircuitBuilder(NUM_QUBITS)
    builder.x([0, 3, 6])
    builder.compose(HammingWeightAdder(4, 4))
    counts = get_counts(builder)

    assert len(counts) == 1
    assert all(y_register(key) == 6 for key in counts)


def test_hamming_adder_twice():
    """Applying the adder twice accumulates the weight twice."""
    builder = CircuitBuilder(NUM_QUBITS)
    builder.x([0, 3])
    # Fresh instance per placement: a block child carries exactly one placement.
    builder.compose(HammingWeightAdder(4, 4))
    builder.compose(HammingWeightAdder(4, 4))
    counts = get_counts(builder)

    assert len(counts) == 1
    assert all(y_register(key) == 4 for key in counts)


def test_hamming_adder_superposition_x():
    """A uniform ``x`` superposition adds its own weight onto ``y = 4``."""
    builder = CircuitBuilder(NUM_QUBITS)
    builder.h([0, 1, 2, 3])
    builder.x(6)
    builder.compose(HammingWeightAdder(4, 4))
    counts = get_counts(builder, num_shots=4096)

    assert len(counts) == 16
    assert all(y_register(key) == hamming_weight(x_register(key)) + 4 for key in counts)
