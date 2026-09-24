"""Tests for :class:`.Register` and :func:`.assign_offsets`."""

import pytest

from qlbm.lattice.registers import Register, assign_offsets


def test_offsets_accumulate_in_declaration_order():
    """Each register starts where the previous one ends."""
    grid, velocity, ancilla = assign_offsets(
        [Register(3, "g"), Register(2, "v"), Register(1, "a")]
    )

    assert (grid.offset, velocity.offset, ancilla.offset) == (0, 3, 5)
    assert list(grid) == [0, 1, 2]
    assert list(velocity) == [3, 4]
    assert list(ancilla) == [5]


def test_indexing_yields_global_qubits():
    """Positive, negative and slice keys map to global indices."""
    register = Register(4, "g", offset=10)

    assert register[0] == 10
    assert register[3] == 13
    assert register[-1] == 13
    assert register[-4] == 10
    assert register[1:3] == [11, 12]
    assert register[::-1] == [13, 12, 11, 10]


@pytest.mark.parametrize("key", [4, -5])
def test_out_of_range_index_raises(key):
    """Indices outside the register fail loudly."""
    with pytest.raises(IndexError, match="out of range"):
        Register(4, "g", offset=10)[key]


def test_zero_width_register_is_empty():
    """A zero-width register takes no global indices."""
    empty, after = assign_offsets([Register(0, "v"), Register(2, "a")])

    assert len(empty) == 0
    assert list(empty) == []
    assert list(after) == [0, 1]


def test_negative_size_is_rejected():
    """A register cannot have negative width."""
    with pytest.raises(ValueError, match="non-negative"):
        Register(-1, "g")


def test_equality_compares_size_name_and_offset():
    """Registers are equal only when all three attributes agree."""
    assert Register(2, "g", 1) == Register(2, "g", 1)
    assert Register(2, "g", 1) != Register(2, "g", 0)
    assert Register(2, "g", 1) != Register(2, "v", 1)
    assert Register(2, "g", 1) != Register(3, "g", 1)
