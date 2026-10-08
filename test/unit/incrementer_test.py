"""Structural regression pins for ``ControlledIncrementer``.

Correctness is covered differentially in ``test/unit/differential/ms_operators_test.py``;
these pins only guard against silent structural drift.  The counts are
flattened-command counts.
"""

import pytest

from qlbm.components.ms.streaming import ControlledIncrementer
from qlbm.lattice import MSLattice


@pytest.fixture
def lattice_asymmetric_medium_3d():
    """8x16x8 grid, 4 velocities per dimension, no obstacles."""
    return MSLattice("test/resources/asymmetric_3d_no_obstacles.json")


@pytest.fixture
def lattice_symmetric_small_2d():
    """16x16 grid, 4 velocities per dimension, no obstacles."""
    return MSLattice("test/resources/symmetric_2d_no_obstacles.json")


def test_3d_incrementer_size(lattice_asymmetric_medium_3d):
    """3D incrementer spans the whole lattice and has a pinned command count."""
    inc: ControlledIncrementer = ControlledIncrementer(lattice_asymmetric_medium_3d)

    assert inc.n_qubits == lattice_asymmetric_medium_3d.n_qubits
    assert len(inc.flatten()) == 158


def test_2d_incrementer_size(lattice_symmetric_small_2d):
    """2D incrementer spans the whole lattice and has a pinned command count."""
    inc: ControlledIncrementer = ControlledIncrementer(lattice_symmetric_small_2d)

    assert inc.n_qubits == lattice_symmetric_small_2d.n_qubits
    assert len(inc.flatten()) == 132
