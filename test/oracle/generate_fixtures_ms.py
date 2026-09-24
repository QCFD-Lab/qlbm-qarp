"""Write the fixtures of ``cases_ms.py``: the action on ``cases_ms.probe_state`` probes.

``kind`` is ``statevector``, ``statevector_sampled`` (``psi_out_samples`` at
``sample_indices``) or ``measure``; ``n_qubits`` is cross-checked by the tests.

Run with the qiskit-based qlbm at ``fixture_store.REFERENCE_COMMIT`` installed; the
README's Testing section has the command.
"""

import sys
from pathlib import Path

import numpy as np
from fixture_store import save_npz

sys.path.insert(0, str(Path(__file__).parent))

from cases_ms import (  # noqa: E402
    CASES,
    FULL_STATE_MAX_QUBITS,
    LATTICES,
    NUM_PROBES,
    probe_state,
    sample_indices,
)
from qiskit.quantum_info import Statevector  # noqa: E402

import qlbm.components  # noqa: F401,E402  (import first: qlbm.lattice is circular)
from qlbm.components.ms import (  # noqa: E402
    MSQLBM,
    BounceBackReflectionOperator,
    BounceBackWallComparator,
    ControlledIncrementer,
    EdgeComparator,
    GridMeasurement,
    MSInitialConditions,
    MSInitialConditions3DSlim,
    MSStreamingOperator,
    SpecularReflectionOperator,
    SpecularWallComparator,
    StreamingAncillaPreparation,
)
from qlbm.lattice import MSLattice  # noqa: E402

_LATTICE_CACHE: dict = {}


def get_lattice(name):
    """Build (and memoise) the lattice named ``name`` in ``cases_ms.LATTICES``."""
    if name not in _LATTICE_CACHE:
        _LATTICE_CACHE[name] = MSLattice(LATTICES[name])
    return _LATTICE_CACHE[name]


def boundary_operator(case, lattice):
    """Instantiate the reflection operator matching ``case['boundary']``."""
    boundary = case["boundary"]
    blocks = lattice.shapes[boundary]
    if boundary == "specular":
        return SpecularReflectionOperator(lattice, blocks), blocks
    return BounceBackReflectionOperator(lattice, blocks), blocks


def select_wall(blocks, spec):
    """Resolve ``["inside"|"outside", dim, index]`` against ``blocks[0]``."""
    side, dim, index = spec
    walls = blocks[0].walls_inside if side == "inside" else blocks[0].walls_outside
    return walls[dim][index]


def select_edge(blocks, spec):
    """Resolve ``["near_corner"|"corner", index]`` against ``blocks[0]``."""
    kind, index = spec
    edges = (
        blocks[0].near_corner_edges_3d
        if kind == "near_corner"
        else blocks[0].corner_edges_3d
    )
    return edges[index]


def select_point(blocks, spec):
    """Resolve a point selector against ``blocks[0]``."""
    kind, index = spec
    lookup = {
        "corner_outside": blocks[0].corners_outside,
        "near_corner_2d": blocks[0].near_corner_points_2d,
        "overlapping_near_corner_3d": blocks[0].overlapping_near_corner_edge_points_3d,
    }
    return lookup[kind][index]


def build_reference(case):
    """Build the qiskit circuit for ``case``."""
    kind = case["component"]
    lattice = get_lattice(case["lattice"])

    if kind == "streaming_ancilla_prep":
        return StreamingAncillaPreparation(
            lattice, case["velocities"], case["dim"]
        ).circuit
    if kind == "controlled_incrementer":
        return ControlledIncrementer(lattice, reflection=case["reflection"]).circuit
    if kind == "streaming_operator":
        return MSStreamingOperator(lattice, case["velocities"]).circuit
    if kind == "initial_conditions":
        return MSInitialConditions(lattice).circuit
    if kind == "initial_conditions_3d_slim":
        return MSInitialConditions3DSlim(lattice).circuit
    if kind == "grid_measurement":
        return GridMeasurement(lattice).circuit
    if kind == "specular_wall_comparator":
        blocks = lattice.shapes["specular"]
        return SpecularWallComparator(
            lattice, select_wall(blocks, case["wall"])
        ).circuit
    if kind == "bounceback_wall_comparator":
        blocks = lattice.shapes["bounceback"]
        return BounceBackWallComparator(
            lattice, select_wall(blocks, case["wall"])
        ).circuit
    if kind == "edge_comparator":
        blocks = lattice.shapes["specular"] or lattice.shapes["bounceback"]
        return EdgeComparator(lattice, select_edge(blocks, case["edge"])).circuit
    if kind == "specular_reflection_operator":
        return SpecularReflectionOperator(lattice, lattice.shapes["specular"]).circuit
    if kind == "bounceback_reflection_operator":
        return BounceBackReflectionOperator(
            lattice, lattice.shapes["bounceback"]
        ).circuit
    if kind == "msqlbm":
        return MSQLBM(lattice, group_velocities=case["group_velocities"]).circuit

    # Sub-circuit cases: run one reflection-operator method on a fresh circuit.
    operator, blocks = boundary_operator(case, lattice)
    circuit = lattice.circuit.copy()
    if kind == "reflect_wall":
        operator.reflect_wall(circuit, select_wall(blocks, case["wall"]))
    elif kind == "reset_edge_state":
        operator.reset_edge_state(circuit, select_edge(blocks, case["edge"]))
    elif kind == "reset_point_state":
        operator.reset_point_state(circuit, select_point(blocks, case["point"]))
    elif kind == "flip_and_stream":
        operator.flip_and_stream(circuit)
    else:
        raise ValueError(f"Unknown component kind: {kind}")
    return circuit


def measurement_pairs(circuit):
    """Ordered ``(qubit, clbit)`` pairs of every measurement in ``circuit``."""
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
    """Write one fixture file per MS case."""
    for case in CASES:
        circuit = build_reference(case)
        n = circuit.num_qubits

        if case["component"] == "grid_measurement":
            save_npz(
                case["id"],
                kind="measure",
                n_qubits=n,
                pairs=measurement_pairs(circuit),
            )
            print(f"{case['id']}: measure, n={n}")
            continue

        psi_in = np.stack([probe_state(n, k) for k in range(NUM_PROBES)])
        psi_out = np.stack(
            [np.asarray(Statevector(psi).evolve(circuit).data) for psi in psi_in]
        )

        if n <= FULL_STATE_MAX_QUBITS:
            save_npz(
                case["id"],
                kind="statevector",
                n_qubits=n,
                psi_in=psi_in,
                psi_out=psi_out,
            )
            print(f"{case['id']}: statevector, n={n}, size={circuit.size()}")
        else:
            indices = sample_indices(n)
            save_npz(
                case["id"],
                kind="statevector_sampled",
                n_qubits=n,
                sample_indices=indices,
                psi_out_samples=psi_out[:, indices],
            )
            print(
                f"{case['id']}: sampled statevector, n={n}, "
                f"size={circuit.size()}, kept={len(indices)}"
            )


if __name__ == "__main__":
    main()
