"""Write the fixtures of ``cases_lqlga.py``: ``unitary``, ``psi_in``/``psi_out``, ``pairs``
with ``num_ops`` (no gates may precede the measurements), or ``psi_out``.

Run with the qiskit-based qlbm at ``fixture_store.REFERENCE_COMMIT`` installed; the
README's Testing section has the command.
"""

import zlib

import numpy as np
from cases_lqlga import CASES_LQLGA, NUM_PROBES
from fixture_store import save_npz
from qiskit.quantum_info import Operator, Statevector

# Import order matters: `qlbm.lattice` alone trips a circular import in the
# qiskit-based package, `qlbm.components` first resolves it.
import qlbm.components  # noqa: F401


def build_lattice(case):
    """Build the qiskit-based ``LQLGALattice`` for ``case``, geometries applied."""
    from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice

    lattice = LQLGALattice(case["lattice"])
    if "geometries" in case:
        lattice.set_geometries(case["geometries"])
    if "accumulation" in case:
        lattice.use_accumulation_register(
            case["accumulation"]["size"], case["accumulation"]["indices"]
        )
    return lattice


def grid_data_of(case):
    """Convert the JSON-shaped ``grid_data`` manifest entry to qlbm tuples."""
    return [
        (tuple(gridpoint), tuple(bool(v) for v in profile))
        for gridpoint, profile in case["grid_data"]
    ]


def build_reference(case):
    """Build the qiskit component for ``case`` and return its circuit."""
    from qlbm.components.lqlga import (
        LQLGA,
        GenericLQLGACollisionOperator,
        LQGLAInitialConditions,
        LQLGAGridVelocityMeasurement,
        LQLGAMGReflectionOperator,
        LQLGAReflectionOperator,
        LQLGAStreamingOperator,
    )
    from qlbm.components.lqlga.initial import LQGLAAveragedInitialConditions

    lattice = build_lattice(case)
    kind = case["component"]

    if kind == "lqlga_streaming":
        return LQLGAStreamingOperator(lattice).circuit
    if kind == "lqlga_collision":
        return GenericLQLGACollisionOperator(lattice).circuit
    if kind == "lqlga_reflection":
        return LQLGAReflectionOperator(
            lattice, lattice.shapes["bounceback"] + lattice.shapes["specular"]
        ).circuit
    if kind == "lqlga_mg_reflection":
        return LQLGAMGReflectionOperator(
            lattice,
            [gdict["bounceback"] + gdict["specular"] for gdict in lattice.geometries],
        ).circuit
    if kind == "lqlga_initial":
        return LQGLAInitialConditions(lattice, grid_data_of(case)).circuit
    if kind == "lqlga_averaged_initial":
        return LQGLAAveragedInitialConditions(lattice, case["gridpoints"]).circuit
    if kind == "lqlga_measurement":
        return LQLGAGridVelocityMeasurement(lattice).circuit
    if kind == "lqlga":
        return LQLGA(lattice).circuit
    if kind == "lqlga_evolution":
        circuit = lattice.circuit.copy()
        circuit.compose(
            LQGLAInitialConditions(lattice, grid_data_of(case)).circuit, inplace=True
        )
        for _ in range(case["steps"]):
            circuit.compose(LQLGA(lattice).circuit, inplace=True)
        return circuit
    raise ValueError(f"Unknown component kind: {kind}")


def random_states(num_qubits, seed):
    """``NUM_PROBES`` Haar-ish random unit statevectors, LSB index basis."""
    rng = np.random.default_rng(seed)
    dim = 2**num_qubits
    raw = rng.normal(size=(NUM_PROBES, dim)) + 1j * rng.normal(size=(NUM_PROBES, dim))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def measurement_pairs(circuit):
    """Ordered ``(qubit, clbit)`` index pairs of every measure instruction."""
    pairs = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        pairs.append(
            (
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.clbits[0]).index,
            )
        )
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def main():
    """Write one fixture file per LQLGA case."""
    for case in CASES_LQLGA:
        circuit = build_reference(case)
        mode = case["mode"]

        if mode == "unitary":
            unitary = Operator(circuit).data
            save_npz(case["id"], unitary=unitary)
            print(f"{case['id']}: unitary {unitary.shape[0]}x{unitary.shape[0]}")
        elif mode == "probes":
            num_qubits = circuit.num_qubits
            seed = zlib.crc32(case["id"].encode())
            psi_in = random_states(num_qubits, seed)
            psi_out = np.array(
                [Statevector(psi).evolve(circuit).data for psi in psi_in]
            )
            save_npz(case["id"], psi_in=psi_in, psi_out=psi_out)
            print(f"{case['id']}: {NUM_PROBES} probes on {num_qubits} qubits")
        elif mode == "measurement":
            pairs = measurement_pairs(circuit)
            save_npz(case["id"], pairs=pairs, num_ops=len(circuit.data))
            print(f"{case['id']}: {len(pairs)}/{len(circuit.data)} measurement pairs")
        elif mode == "evolution":
            psi_out = (
                Statevector.from_int(0, 2**circuit.num_qubits).evolve(circuit).data
            )
            save_npz(case["id"], psi_out=psi_out)
            print(f"{case['id']}: evolution on {circuit.num_qubits} qubits")
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
