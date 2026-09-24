"""MCSwap placement on Space-Time lattice registers.

The command-stream assertions pin the qubit indices the lattice resolves and
the MCZ-sandwich lowering of the multi-controlled X (qarpx has no MCX gate).
The unitary assertions are the independent oracle: an analytically built
multi-controlled SWAP permutation matrix.
"""

import itertools

import numpy as np
import qarpx as qx

from qlbm.components.common.primitives import MCSwap
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


def _commands(component):
    return [str(command) for command in component.flatten()]


def _unitary(component):
    return np.asarray(
        qx.QarpSimulator().unitary_matrix(component.flatten(), component.n_qubits)
    )


def _controlled_swap_matrix(num_qubits, controls, targets):
    """Permutation matrix that swaps ``targets`` when all ``controls`` are set."""
    matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for state in range(2**num_qubits):
        image = state
        if all((state >> control) & 1 for control in controls):
            if ((state >> targets[0]) & 1) != ((state >> targets[1]) & 1):
                image = state ^ (1 << targets[0]) ^ (1 << targets[1])
        matrix[image, state] = 1
    return matrix


def _d2q4_lattice() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": "D2Q4",
            },
            "geometry": [],
        },
    )


def _d1q2_lattice() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 4},
                "velocities": "D1Q2",
            },
            "geometry": [],
        },
    )


def test_mcswap_13ctrl():
    """Explicit control/target indices lower to the expected MCZ sandwich."""
    lattice = _d2q4_lattice()

    mcswap = MCSwap(lattice, [0, 2, 3], (5, 6))

    assert mcswap.n_qubits == lattice.n_qubits
    assert _commands(mcswap) == [
        "CX [q6, q5]",
        "H [q6]",
        "MCZ [q0, q2, q3, q5, q6]",
        "H [q6]",
        "CX [q6, q5]",
    ]


def test_mcswap_grid_ctrl():
    """Grid-register controls resolve to qubits 0-7, velocities to 9 and 11."""
    lattice = _d2q4_lattice()

    mcswap = MCSwap(
        lattice,
        lattice.grid_index(),
        (lattice.velocity_index(0, 1)[0], lattice.velocity_index(0, 3)[0]),
    )

    grid = ", ".join(f"q{qubit}" for qubit in range(8))
    assert _commands(mcswap) == [
        "CX [q11, q9]",
        "H [q11]",
        f"MCZ [{grid}, q9, q11]",
        "H [q11]",
        "CX [q11, q9]",
    ]


def test_mcswap_grid_controlled_is_a_multi_controlled_swap():
    """Oracle: the analytic controlled-SWAP permutation on the lattice's qubits."""
    lattice = _d1q2_lattice()
    controls = lattice.grid_index()
    targets = (lattice.velocity_index(0, 0)[0], lattice.velocity_index(1, 1)[0])

    mcswap = MCSwap(lattice, controls, targets)

    np.testing.assert_allclose(
        _unitary(mcswap),
        _controlled_swap_matrix(lattice.n_qubits, controls, targets),
        atol=1e-12,
        rtol=0,
    )


def test_mcswap_single_control_placements():
    """Every control/target placement on a narrow lattice matches the oracle."""
    lattice = _d1q2_lattice()

    for control, targets in itertools.product([0, 1], [(2, 3), (4, 7), (5, 2)]):
        mcswap = MCSwap(lattice, [control], targets)

        np.testing.assert_allclose(
            _unitary(mcswap),
            _controlled_swap_matrix(lattice.n_qubits, [control], targets),
            atol=1e-12,
            rtol=0,
        )
