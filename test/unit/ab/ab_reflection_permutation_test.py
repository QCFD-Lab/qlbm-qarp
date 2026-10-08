import pytest

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.reflection import ABBounceBackReflectionPermutation
from qlbm.components.ab.reflection.common import ABSpecularReflectionPermutation
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.utils import bit_value
from test.builders import CircuitBuilder

from .qarp_helpers import sample_register_counts

NUM_QUBITS = 4


def _run_permutation(state_in: int, permutation_block) -> set:
    """Prepare ``|state_in>``, apply the permutation, return the outcome set."""
    builder = CircuitBuilder(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        if bit_value(state_in, q):
            builder.x(q)

    builder.compose(permutation_block)

    return set(sample_register_counts(builder, range(NUM_QUBITS), shots=128))


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 3), (2, 4), (3, 1), (4, 2), (5, 7), (6, 8), (7, 5), (8, 6)]
    + [(i, i) for i in range(9, 16)],
)
def test_reflectionpermutation_outcomes_d2q9_bounceback(permutation_outcome_pairs):
    outcomes = _run_permutation(
        permutation_outcome_pairs[0],
        ABBounceBackReflectionPermutation(
            NUM_QUBITS, LatticeDiscretization.D2Q9, ABEncodingType.AB
        ),
    )

    assert outcomes == {permutation_outcome_pairs[1]}, (
        f"{permutation_outcome_pairs} handled incorrectly. "
        f"Expected {permutation_outcome_pairs[1]}, got {outcomes}."
    )


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 3), (3, 1), (5, 6), (6, 5), (8, 7), (7, 8), (2, 2), (4, 4)]
    + [(i, i) for i in range(9, 16)],
)
def test_reflectionpermutation_outcomes_d2q9_specular_rx(permutation_outcome_pairs):
    outcomes = _run_permutation(
        permutation_outcome_pairs[0],
        ABSpecularReflectionPermutation(
            NUM_QUBITS,
            LatticeDiscretization.D2Q9,
            ABEncodingType.AB,
            reflect_in_dim=(True, False),
        ),
    )

    assert outcomes == {permutation_outcome_pairs[1]}, (
        f"{permutation_outcome_pairs} handled incorrectly. "
        f"Expected {permutation_outcome_pairs[1]}, got {outcomes}."
    )


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 1), (3, 3), (2, 4), (4, 2), (5, 8), (8, 5), (6, 7), (7, 6)]
    + [(i, i) for i in range(9, 16)],
)
def test_reflectionpermutation_outcomes_d2q9_specular_ry(permutation_outcome_pairs):
    outcomes = _run_permutation(
        permutation_outcome_pairs[0],
        ABSpecularReflectionPermutation(
            NUM_QUBITS,
            LatticeDiscretization.D2Q9,
            ABEncodingType.AB,
            reflect_in_dim=(False, True),
        ),
    )

    assert outcomes == {permutation_outcome_pairs[1]}, (
        f"{permutation_outcome_pairs} handled incorrectly. "
        f"Expected {permutation_outcome_pairs[1]}, got {outcomes}."
    )


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 3), (2, 4), (3, 1), (4, 2), (5, 7), (6, 8), (7, 5), (8, 6)]
    + [(i, i) for i in range(9, 16)],
)
def test_reflectionpermutation_outcomes_d2q9_specular_rxry(permutation_outcome_pairs):
    outcomes = _run_permutation(
        permutation_outcome_pairs[0],
        ABSpecularReflectionPermutation(
            NUM_QUBITS,
            LatticeDiscretization.D2Q9,
            ABEncodingType.AB,
            reflect_in_dim=(True, True),
        ),
    )

    assert outcomes == {permutation_outcome_pairs[1]}, (
        f"{permutation_outcome_pairs} handled incorrectly. "
        f"Expected {permutation_outcome_pairs[1]}, got {outcomes}."
    )
