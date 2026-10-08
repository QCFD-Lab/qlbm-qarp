"""Integration tests for :class:`CircuitCompiler`.

Every MS streaming operator lowers to a built ``qx.Block`` of the right width,
and the compiler hands every optimization level back the same built block.
"""

from itertools import product

import pytest
import qarpx as qx

from qlbm.components import MSStreamingOperator
from qlbm.infra import (
    CircuitCompiler,
)
from qlbm.lattice import MSLattice
from qlbm.tools.exceptions import CompilerException
from qlbm.tools.utils import get_time_series


@pytest.fixture
def lattice_asymmetric_medium_3d():
    """A 3D 8x16x8 MS lattice with 4 velocities per dimension."""
    return MSLattice("test/resources/asymmetric_3d_no_obstacles.json")


@pytest.fixture
def lattice_symmetric_small_2d():
    """A 2D 16x16 MS lattice with 4 velocities per dimension."""
    return MSLattice("test/resources/symmetric_2d_no_obstacles.json")


def streaming_operator(lattice, velocity):
    """The MS streaming operator for the ``velocity``-th time series entry."""
    num_velocities = (
        lattice.num_velocities[0] + 1
    )  # +1 because velocities are between 0 and n_vi
    velocities = get_time_series(num_velocities)[velocity]
    return MSStreamingOperator(lattice, velocities)


@pytest.mark.parametrize(
    "lattice_fixture,velocity,optimization_level",
    list(
        product(
            ["lattice_symmetric_small_2d", "lattice_asymmetric_medium_3d"],
            list(range(3)),
            [0, 1, 2],
        )
    ),
)
def test_qarp_target_compilation(
    lattice_fixture, velocity, optimization_level, request
):
    """Every MS streaming operator lowers to a built block of the lattice's width."""
    lattice = request.getfixturevalue(lattice_fixture)
    op = streaming_operator(lattice, velocity)
    compiler = CircuitCompiler()

    compiled_circuit = compiler.compile(op, optimization_level)

    assert isinstance(compiled_circuit, qx.Block)
    assert compiled_circuit.n_qubits == lattice.n_qubits
    assert len(compiled_circuit.flatten()) > 0


def test_compiling_a_prebuilt_block_is_idempotent(lattice_symmetric_small_2d):
    """A ``qx.Block`` handed to the compiler comes back unchanged at level 0."""
    op = streaming_operator(lattice_symmetric_small_2d, 0)
    compiler = CircuitCompiler()

    once = compiler.compile(op, 0)
    twice = compiler.compile(once, 0)

    assert twice is once


def test_unsupported_optimization_level_is_rejected(lattice_symmetric_small_2d):
    """Compilation rejects optimization levels outside the supported set."""
    op = streaming_operator(lattice_symmetric_small_2d, 0)
    compiler = CircuitCompiler()

    with pytest.raises(CompilerException):
        compiler.compile(op, 3)
