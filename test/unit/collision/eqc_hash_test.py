"""Hash/equality contract of :class:`.EquivalenceClass`.

The oracle here is the Python data model, not the implementation: objects that
compare equal must hash equal, or set membership and deduplication silently
break.  ``EquivalenceClass`` stores its configurations in a ``set``, whose
iteration order depends on insertion history and on the interpreter's set
layout, so the contract only holds if the hash is order-invariant.
"""

from itertools import permutations

import pytest

from qlbm.lattice.eqc.eqc import EquivalenceClass
from qlbm.lattice.eqc.eqc_generator import EquivalenceClassGenerator
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization

DISCRETIZATIONS_WITH_CLASSES = [
    LatticeDiscretization.D1Q3,
    LatticeDiscretization.D2Q4,
    LatticeDiscretization.D3Q6,
]


def _rebuild(eqc: EquivalenceClass, order) -> EquivalenceClass:
    rebuilt = set()
    for configuration in order:
        rebuilt.add(configuration)
    return EquivalenceClass(eqc.discretization, rebuilt)


@pytest.mark.parametrize("discretization", DISCRETIZATIONS_WITH_CLASSES)
def test_equal_classes_hash_equal_under_any_insertion_order(discretization):
    """Equal classes hash equal however their configurations were inserted."""
    for eqc in EquivalenceClassGenerator(discretization).generate_equivalence_classes():
        for order in permutations(eqc.velocity_configurations):
            rebuilt = _rebuild(eqc, order)
            assert rebuilt == eqc
            assert hash(rebuilt) == hash(eqc), (
                f"{discretization} class {eqc.id()} hashes differently when its "
                f"configurations are inserted in order {order}"
            )


@pytest.mark.parametrize("discretization", DISCRETIZATIONS_WITH_CLASSES)
def test_reordered_classes_remain_set_members(discretization):
    """A class rebuilt in reverse order is still found in the generated set."""
    generated = EquivalenceClassGenerator(discretization).generate_equivalence_classes()

    for eqc in generated:
        reversed_order = tuple(reversed(list(eqc.velocity_configurations)))
        assert _rebuild(eqc, reversed_order) in generated


def test_bool_and_int_configurations_are_interchangeable():
    """The generator yields ``0``/``1`` tuples; literals in tests use ``bool``."""
    integer_form = EquivalenceClass(
        LatticeDiscretization.D2Q4,
        {(1, 0, 1, 0), (0, 1, 0, 1)},  # type: ignore[arg-type]
    )
    boolean_form = EquivalenceClass(
        LatticeDiscretization.D2Q4,
        {(False, True, False, True), (True, False, True, False)},
    )

    assert integer_form == boolean_form
    assert hash(integer_form) == hash(boolean_form)
    assert integer_form in {boolean_form}
