"""Shared qarp helpers for the AB unit tests."""

from typing import Dict

import numpy as np
import qarpx as qx

from test.builders import CircuitBuilder


def lattice_builder(lattice) -> CircuitBuilder:
    """Fresh full-width circuit builder for ``lattice``."""
    return CircuitBuilder(lattice.n_qubits)


def simulate_statevector(builder) -> np.ndarray:
    """Final statevector of a builder (or built block), LSB-indexed."""
    block = builder.build() if isinstance(builder, CircuitBuilder) else builder
    return np.asarray(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


def marginal_probabilities(statevector: np.ndarray, qubits) -> np.ndarray:
    """Marginal distribution over ``qubits``, ``qubits[0]`` least significant."""
    amplitudes = np.abs(statevector) ** 2
    indices = np.arange(len(statevector))
    keys = np.zeros(len(indices), dtype=int)
    for position, qubit in enumerate(qubits):
        keys |= ((indices >> qubit) & 1) << position
    result = np.zeros(2 ** len(list(qubits)))
    np.add.at(result, keys, amplitudes)
    return result


def statevectors_equivalent(first: np.ndarray, second: np.ndarray, atol=1e-8) -> bool:
    """Equality up to global phase, mirroring ``Statevector.equiv``."""
    overlap = np.vdot(first, second)
    return bool(
        np.isclose(
            abs(overlap), np.linalg.norm(first) * np.linalg.norm(second), atol=atol
        )
    )


def sample_counts(builder, shots: int, seed: int = 1234) -> Dict[int, int]:
    """Sample a builder that ends in measurements.

    ``SamplingResult.counts`` is keyed on the *qubit* index space (bit ``q``
    of the key is the outcome of qubit ``q``), not on the classical register.
    """
    block = builder.build() if isinstance(builder, CircuitBuilder) else builder
    result = qx.QarpSimulator().run(block.flatten(), block.n_qubits, shots, seed)
    return dict(result.counts)


def sample_register_counts(
    builder, qubits, shots: int, seed: int = 1234
) -> Dict[int, int]:
    """Measure ``qubits`` and return counts keyed on the register's integer value.

    ``qubits[0]`` is the least significant bit.
    """
    qubits = list(qubits)
    for cbit, qubit in enumerate(qubits):
        builder.measure(qubit, cbit)

    counts: Dict[int, int] = {}
    for key, count in sample_counts(builder, shots, seed).items():
        value = 0
        for position, qubit in enumerate(qubits):
            value |= ((key >> qubit) & 1) << position
        counts[value] = counts.get(value, 0) + count
    return counts
