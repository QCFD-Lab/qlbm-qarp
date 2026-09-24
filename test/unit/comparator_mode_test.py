import pytest

from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import ComparatorMode


def test_comparator_mode_from_string():
    assert ComparatorMode.from_string("<") == ComparatorMode.LT
    assert ComparatorMode.from_string("<=") == ComparatorMode.LE
    assert ComparatorMode.from_string(">") == ComparatorMode.GT
    assert ComparatorMode.from_string(">=") == ComparatorMode.GE


def test_comparator_mode_from_string_whitespace():
    assert ComparatorMode.from_string("  <  ") == ComparatorMode.LT


def test_comparator_mode_from_string_invalid():
    with pytest.raises(LatticeException, match="Unsupported comparator mode"):
        ComparatorMode.from_string("!=")
