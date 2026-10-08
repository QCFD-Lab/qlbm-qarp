"""Write the fixtures of ``cases_ab.py``: ``unitary``, ``psi_in``/``psi_out`` or ``pairs``.

Run with the qiskit-based qlbm at ``fixture_store.REFERENCE_COMMIT`` installed; the
README's Testing section has the command.
"""

import zlib

import numpy as np
from cases_ab import CASES
from fixture_store import save_npz
from qiskit import QuantumCircuit
from qiskit.circuit.library import RGQFTMultiplier
from qiskit.quantum_info import Operator, Statevector

NUM_BASIS_PROBES = 4
NUM_SUPERPOSITION_PROBES = 2
SUPERPOSITION_SUPPORT = 16


def make_probes(case_id: str, num_qubits: int) -> np.ndarray:
    """Seeded probe states for ``case_id`` as a ``[K][2**n]`` array."""
    dim = 2**num_qubits
    rng = np.random.default_rng(zlib.crc32(case_id.encode()))
    probes = []

    for _ in range(NUM_BASIS_PROBES):
        state = np.zeros(dim, dtype=complex)
        state[int(rng.integers(0, dim))] = 1.0
        probes.append(state)

    for _ in range(NUM_SUPERPOSITION_PROBES):
        support = rng.choice(dim, size=min(SUPERPOSITION_SUPPORT, dim), replace=False)
        state = np.zeros(dim, dtype=complex)
        state[support] = np.exp(2j * np.pi * rng.random(len(support)))
        state /= np.linalg.norm(state)
        probes.append(state)

    return np.array(probes)


def build_lattice(case):
    """Instantiate the qiskit-based lattice described by ``case``."""
    import qlbm.components  # noqa: F401  (import order dodges a circular import)
    from qlbm.lattice import ABLattice, OHLattice

    lattice_class = {"ABLattice": ABLattice, "OHLattice": OHLattice}[
        case["lattice_class"]
    ]
    lattice = lattice_class(case["lattice"])
    if case.get("marker_qubits"):
        lattice.set_num_marker_qubits(case["marker_qubits"])
    return lattice


def build_reference(case):
    """Build the qiskit component for ``case`` and return its circuit."""
    from qlbm.components.ab import (
        ABQLBM,
        ABDiscreteUniformInitialConditions,
        ABGridMeasurement,
        ABInitialConditions,
        ABParallelDiscreteUniformInitialConditions,
        ABStreamingOperator,
        BinaryToOHPermutation,
    )
    from qlbm.components.ab.averaged_collision import ABEAveragedCollisionOperator
    from qlbm.components.ab.encodings import ABEncodingType
    from qlbm.components.ab.reflection import (
        ABBounceBackReflectionPermutation,
        ABReflectionOperator,
        ABSpecularReflectionPermutation,
        ABZoneAgnosticReflectionOperator,
        ABZoneAgnosticReflectionOracle,
    )
    from qlbm.components.ab.reflection.agnosotic_reflection import ABZoneAgnosticSRCheck
    from qlbm.components.cqlbm import CQLBM
    from qlbm.lattice.spacetime.properties_base import LatticeDiscretization

    kind = case["component"]

    if kind == "rgqft_multiplier":
        return RGQFTMultiplier(
            num_state_qubits=case["n"], num_result_qubits=case["m"]
        ).decompose()

    if kind in ("bb_permutation", "sr_permutation"):
        encoding = ABEncodingType[case["encoding"]]
        if kind == "bb_permutation":
            circuit = ABBounceBackReflectionPermutation(
                case["n"], LatticeDiscretization.D2Q9, encoding
            ).circuit
        else:
            circuit = ABSpecularReflectionPermutation(
                case["n"],
                LatticeDiscretization.D2Q9,
                encoding,
                tuple(case["reflect"]),
            ).circuit
        if case["ctrl"] == 0:
            return circuit
        controlled = circuit.control(case["ctrl"])
        wrapper = QuantumCircuit(case["ctrl"] + case["n"])
        wrapper.append(controlled, range(case["ctrl"] + case["n"]))
        return wrapper.decompose()

    lattice = build_lattice(case)

    if kind == "binary_to_oh":
        return BinaryToOHPermutation(lattice).circuit
    if kind == "streaming":
        return ABStreamingOperator(lattice, case["controls"]).circuit
    if kind == "averaged_collision":
        return ABEAveragedCollisionOperator(lattice).circuit
    if kind == "discrete_uniform_initial":
        return ABDiscreteUniformInitialConditions(
            lattice,
            case["velocities"],
            tuple(list(g) for g in case["grid_superpose"]),
        ).circuit
    if kind == "initial":
        return ABInitialConditions(lattice).circuit
    if kind == "parallel_initial":
        return ABParallelDiscreteUniformInitialConditions(
            lattice,
            case["velocities"],
            [tuple(list(d) for d in g) for g in case["grid_superpose"]],
        ).circuit
    if kind == "grid_measurement":
        return ABGridMeasurement(lattice, case["measure_velocity"]).circuit
    if kind == "za_oracle":
        return ABZoneAgnosticReflectionOracle(
            lattice,
            lattice.shapes[case["shape_key"]][0],
            target_obstacle_index=case["target_obstacle_index"],
        ).circuit
    if kind == "za_sr_check":
        return ABZoneAgnosticSRCheck(
            lattice,
            lattice.discretization,
            lattice.shapes["specular"],
            check_negative_direction=case["check_negative_direction"],
        ).circuit
    if kind == "standard_reflection":
        return ABReflectionOperator(lattice).circuit
    if kind == "agnostic_reflection":
        return ABZoneAgnosticReflectionOperator(lattice).circuit
    if kind == "abqlbm":
        return ABQLBM(lattice, use_agnostic_bcs=case["use_agnostic_bcs"]).circuit
    if kind == "cqlbm":
        return CQLBM(lattice, use_agnostic_bcs=case["use_agnostic_bcs"]).circuit

    raise ValueError(f"Unknown component kind: {kind}")


def measurement_pairs(circuit) -> np.ndarray:
    """Ordered ``(qubit, cbit)`` pairs of every measurement in ``circuit``."""
    pairs = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        pairs.append(
            [
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.clbits[0]).index,
            ]
        )
    return np.array(pairs, dtype=int)


def main():
    """Write one fixture file per case."""
    total_bytes = 0
    for case in CASES:
        circuit = build_reference(case)

        if case["mode"] == "unitary":
            unitary = Operator(circuit).data
            assert circuit.num_qubits <= 6, (
                f"{case['id']}: unitary mode needs <= 6 qubits, "
                f"got {circuit.num_qubits}"
            )
            path = save_npz(case["id"], unitary=unitary)
            detail = f"unitary {unitary.shape[0]}x{unitary.shape[0]}"
        elif case["mode"] == "probe":
            psi_in = make_probes(case["id"], circuit.num_qubits)
            psi_out = np.array(
                [Statevector(state).evolve(circuit).data for state in psi_in]
            )
            path = save_npz(case["id"], psi_in=psi_in, psi_out=psi_out)
            detail = f"probe {psi_in.shape[0]}x2^{circuit.num_qubits}"
        elif case["mode"] == "measure":
            pairs = measurement_pairs(circuit)
            path = save_npz(case["id"], pairs=pairs)
            detail = f"measure {len(pairs)} pairs"
        else:
            raise ValueError(f"Unknown mode: {case['mode']}")

        size = path.stat().st_size
        total_bytes += size
        print(f"{case['id']}: {detail}, {size / 1024:.1f} KiB")

    print(f"total: {total_bytes / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
