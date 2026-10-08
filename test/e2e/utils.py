"""Shared utilities for end-to-end tests."""

from typing import Dict, Tuple

import numpy as np
import qarpx as qx

from test.builders import CircuitBuilder


class E2ECircuit:
    """Composable circuit wrapper around :class:`.CircuitBuilder`."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self._builder = CircuitBuilder(n_qubits, name="e2e")

    def x(self, qubit: int) -> None:
        """Pauli-X on ``qubit``."""
        self._builder.x(qubit)

    def compose(self, block, qubits=None) -> "E2ECircuit":
        """Append a block or another E2ECircuit, placed on ``qubits`` if given."""
        if isinstance(block, E2ECircuit):
            block = block.build()
        self._builder.compose(block, qubits=qubits)
        return self

    def build(self) -> "qx.Block":
        """Return the finished, built block."""
        return self._builder.build()


def as_circuit(block) -> E2ECircuit:
    """Wrap an already-built block in a composable :class:`.E2ECircuit`.

    Blocks have no ``copy()`` and are not composable in place.
    """
    block.build()
    circuit = E2ECircuit(block.n_qubits)
    circuit.compose(block)
    return circuit


def _as_block(circuit) -> "qx.Block":
    """Resolve an E2ECircuit or a block to a built Block."""
    if isinstance(circuit, E2ECircuit):
        return circuit.build()
    block = circuit
    block.build()
    return block


def run_statevector(circuit) -> np.ndarray:
    """Simulate a circuit and return the final statevector.

    Parameters
    ----------
    circuit : E2ECircuit | qx.Block
        The circuit to simulate.

    Returns
    -------
    np.ndarray
        The resulting statevector, LSB-indexed (qubit 0 = least significant
        bit), which is the order the decode helpers below assume.
    """
    block = _as_block(circuit)
    return np.asarray(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


def get_nonzero_amplitudes(sv, threshold: float = 1e-8) -> Dict[int, complex]:
    """Return a dict mapping basis-state index to amplitude for nonzero entries.

    Parameters
    ----------
    sv : np.ndarray
        The statevector to inspect.
    threshold : float
        Amplitude magnitude below which entries are treated as zero.

    Returns
    -------
    Dict[int, complex]
        Mapping from statevector index to complex amplitude.
    """
    data = np.array(sv)
    nonzero_idx = np.where(np.abs(data) > threshold)[0]
    return {int(idx): complex(data[idx]) for idx in nonzero_idx}


def decode_state(
    index: int, qubit_ranges: Dict[str, Tuple[int, int]]
) -> Dict[str, int]:
    """Decode a statevector index into named register values.

    Parameters
    ----------
    index : int
        The statevector basis-state index.
    qubit_ranges : Dict[str, Tuple[int, int]]
        Mapping from register name to ``(start_qubit, num_qubits)``.

    Returns
    -------
    Dict[str, int]
        Mapping from register name to the integer value stored in that register.
    """
    result: Dict[str, int] = {}
    for name, (start, size) in qubit_ranges.items():
        value = 0
        for i in range(size):
            if index & (1 << (start + i)):
                value |= 1 << i
        result[name] = value
    return result


def make_ab_qubit_layout(lattice) -> Dict[str, Tuple[int, int]]:
    """Build a qubit-layout dictionary for an ABLattice.

    Parameters
    ----------
    lattice : ABLattice
        The lattice whose register structure to describe.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Mapping ``{register_name: (start_qubit, size)}``.
    """
    dim_names = ["g_x", "g_y", "g_z"]
    layout: Dict[str, Tuple[int, int]] = {}
    for dim in range(lattice.num_dims):
        layout[dim_names[dim]] = (
            lattice.grid_index(dim)[0],
            len(lattice.grid_index(dim)),
        )
    layout["v"] = (lattice.velocity_index()[0], lattice.num_velocity_qubits)
    if lattice.num_comparator_qubits > 0:
        layout["a_c"] = (
            lattice.ancillae_comparator_index()[0],
            lattice.num_comparator_qubits,
        )
    layout["a_o"] = (
        lattice.ancillae_obstacle_index()[0],
        lattice.num_obstacle_qubits,
    )
    return layout


def make_ms_qubit_layout(lattice) -> Dict[str, Tuple[int, int]]:
    """Build a qubit-layout dictionary for an MSLattice.

    Parameters
    ----------
    lattice : MSLattice
        The lattice whose register structure to describe.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Mapping ``{register_name: (start_qubit, size)}``.
    """
    dim_names = ["g_x", "g_y", "g_z"]
    layout: Dict[str, Tuple[int, int]] = {}
    layout["a_v"] = (
        lattice.ancillae_velocity_index()[0],
        len(lattice.ancillae_velocity_index()),
    )
    layout["a_o"] = (
        lattice.ancillae_obstacle_index()[0],
        len(lattice.ancillae_obstacle_index()),
    )
    if lattice.ancillae_comparator_index():
        layout["a_c"] = (
            lattice.ancillae_comparator_index()[0],
            len(lattice.ancillae_comparator_index()),
        )
    for dim in range(lattice.num_dims):
        layout[dim_names[dim]] = (
            lattice.grid_index(dim)[0],
            len(lattice.grid_index(dim)),
        )
    for dim in range(lattice.num_dims):
        vi = lattice.velocity_index(dim)
        if vi:
            layout[f"v_{dim_names[dim][-1]}"] = (vi[0], len(vi))
    for dim in range(lattice.num_dims):
        layout[f"vd_{dim_names[dim][-1]}"] = (
            lattice.velocity_dir_index(dim)[0],
            len(lattice.velocity_dir_index(dim)),
        )
    return layout


def prepare_single_particle(
    lattice, grid_pos: Tuple[int, ...], velocity_channel: int
) -> E2ECircuit:
    """Prepare a circuit with one particle at a specific position and velocity.

    Parameters
    ----------
    lattice : ABLattice | MSLattice
        The lattice that defines the register layout.
    grid_pos : Tuple[int, ...]
        Grid coordinates, one per dimension.
    velocity_channel : int
        Integer index of the velocity channel (binary-encoded into velocity qubits).

    Returns
    -------
    E2ECircuit
        A circuit that prepares the desired initial state.
    """
    circuit = E2ECircuit(lattice.n_qubits)
    for dim, pos in enumerate(grid_pos):
        for i in range(lattice.num_gridpoints[dim].bit_length()):
            if (pos >> i) & 1:
                circuit.x(lattice.grid_index(dim)[i])
    for i in range(lattice.num_velocity_qubits):
        if (velocity_channel >> i) & 1:
            circuit.x(lattice.velocity_index()[i])
    return circuit


def prepare_ms_particle(
    lattice,
    grid_pos: Tuple[int, ...],
    velocity_mag: Tuple[int, ...],
    velocity_dir: Tuple[int, ...],
) -> E2ECircuit:
    """Prepare a circuit with one MS particle at a given position and velocity.

    Parameters
    ----------
    lattice : MSLattice
        The lattice that defines the register layout.
    grid_pos : Tuple[int, ...]
        Grid coordinates, one per dimension.
    velocity_mag : Tuple[int, ...]
        Velocity magnitude per dimension (binary-encoded).
    velocity_dir : Tuple[int, ...]
        Velocity direction per dimension (1 = positive, 0 = negative).

    Returns
    -------
    E2ECircuit
        A circuit that prepares the desired initial state.
    """
    circuit = E2ECircuit(lattice.n_qubits)
    for dim, pos in enumerate(grid_pos):
        for i in range(len(lattice.grid_index(dim))):
            if (pos >> i) & 1:
                circuit.x(lattice.grid_index(dim)[i])
    for dim in range(lattice.num_dims):
        for i in range(len(lattice.velocity_index(dim))):
            if (velocity_mag[dim] >> i) & 1:
                circuit.x(lattice.velocity_index(dim)[i])
        if velocity_dir[dim]:
            circuit.x(lattice.velocity_dir_index(dim)[0])
    return circuit
