"""Write the fixtures of ``cases_spacetime.py``; wide lattices are evolved with Aer.

Run with the qiskit-based qlbm at ``fixture_store.REFERENCE_COMMIT`` installed; the
README's Testing section has the command.
"""

import zlib

import numpy as np
from cases_spacetime import CASES_SPACETIME, NUM_PROBES, SPARSE_SUPPORT
from fixture_store import save_npz
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

# Above this width a dense statevector fixture is no longer storable.
DENSE_LIMIT = 11


def make_lattice(descriptor):
    """Instantiate the qiskit-based SpaceTimeLattice from a case descriptor."""
    from qlbm.lattice import SpaceTimeLattice

    return SpaceTimeLattice(
        descriptor["num_timesteps"],
        descriptor["lattice_data"],
        include_measurement_qubit=descriptor["include_measurement_qubit"],
        use_volumetric_ops=descriptor["use_volumetric_ops"],
    )


def build_reference(case):
    """Build the qiskit component for ``case`` and return its circuit."""
    from qlbm.components.spacetime import (
        PointWiseSpaceTimeInitialConditions,
        PointWiseSpaceTimeReflectionOperator,
        SpaceTimeD2Q4CollisionOperator,
        SpaceTimeGridVelocityMeasurement,
        SpaceTimeQLBM,
        SpaceTimeStreamingOperator,
        VolumetricSpaceTimeInitialConditions,
    )
    from qlbm.components.spacetime.collision.eqc_collision import (
        GenericSpaceTimeCollisionOperator,
    )
    from qlbm.components.spacetime.measurement import SpaceTimePointWiseMassMeasurement
    from qlbm.components.spacetime.reflection.volumetric import (
        VolumetricSpaceTimeReflectionOperator,
    )

    class ConcreteVolumetricReflection(VolumetricSpaceTimeReflectionOperator):
        """Upstream leaves ``__str__`` abstract, making the class uninstantiable."""

        def __str__(self):
            return "[VolumetricSpaceTimeReflectionOperator]"

    lattice = make_lattice(case["lattice"])
    kind = case["component"]

    if kind == "st_streaming":
        return SpaceTimeStreamingOperator(lattice, case["timestep"]).circuit
    if kind == "st_pointwise_initial":
        return PointWiseSpaceTimeInitialConditions(
            lattice, case["grid_data"], case["filter_inside_blocks"]
        ).circuit
    if kind == "st_volumetric_initial":
        return VolumetricSpaceTimeInitialConditions(
            lattice, case["cuboid_bounds"], case["velocity_profile"]
        ).circuit
    if kind == "st_pointwise_reflection":
        return PointWiseSpaceTimeReflectionOperator(
            lattice,
            case["timestep"],
            lattice.shapes["bounceback"],
            case["filter_inside_blocks"],
        ).circuit
    if kind == "st_volumetric_reflection":
        return ConcreteVolumetricReflection(
            lattice,
            case["timestep"],
            lattice.shapes["bounceback"],
            case["filter_inside_blocks"],
        ).circuit
    if kind == "st_eqc_collision":
        return GenericSpaceTimeCollisionOperator(lattice, case["timestep"]).circuit
    if kind == "st_d2q4_collision":
        return SpaceTimeD2Q4CollisionOperator(lattice, case["timestep"]).circuit
    if kind == "st_d2q4_local_collision":
        return SpaceTimeD2Q4CollisionOperator(lattice, 1).local_collision_circuit(
            case["reset_state"]
        )
    if kind == "st_grid_velocity_measurement":
        return SpaceTimeGridVelocityMeasurement(lattice).circuit
    if kind == "st_mass_measurement":
        return SpaceTimePointWiseMassMeasurement(
            lattice, case["gridpoint"], case["velocity_index_to_measure"]
        ).circuit
    if kind == "st_qlbm":
        return SpaceTimeQLBM(lattice, case["filter_inside_blocks"]).circuit
    raise ValueError(f"Unknown component kind: {kind}")


def case_rng(case):
    """Deterministic per-case generator, stable across regenerations."""
    return np.random.default_rng(zlib.crc32(case["id"].encode()))


def random_state(rng, dim):
    """Normalized dense random complex state of dimension ``dim``."""
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


def random_sparse_state(rng, dim, support):
    """Normalized sparse random state as ``(indices, amplitudes)``."""
    indices = np.sort(rng.choice(dim, size=support, replace=False))
    amps = rng.standard_normal(support) + 1j * rng.standard_normal(support)
    return indices.astype(np.int64), amps / np.linalg.norm(amps)


def evolve_dense(circuit, psi_in):
    """Statevector of ``circuit`` acting on ``psi_in`` (exact, no transpile)."""
    return Statevector(psi_in).evolve(circuit).data


def evolve_aer(circuit, psi_in):
    """Same as :func:`evolve_dense`, via Aer — the only tractable path >12 qubits."""
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(method="statevector")
    wrapper = QuantumCircuit(circuit.num_qubits)
    wrapper.set_statevector(Statevector(psi_in))
    wrapper.compose(circuit, inplace=True)
    wrapper.save_statevector()

    result = simulator.run(transpile(wrapper, simulator, optimization_level=0)).result()
    return np.asarray(result.get_statevector(0).data)


def strip_measurements(circuit):
    """Circuit with the final measurements (and their cregs) removed."""
    return circuit.remove_final_measurements(inplace=False)


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
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def probe_payload(circuit, rng, num_qubits):
    """``NUM_PROBES`` dense (input, output) statevector pairs."""
    payload = {}
    for k in range(NUM_PROBES):
        psi_in = random_state(rng, 2**num_qubits)
        payload[f"psi_in_{k}"] = psi_in
        payload[f"psi_out_{k}"] = evolve_dense(circuit, psi_in)
    return payload


def sparse_payload(circuit, rng, num_qubits):
    """``NUM_PROBES`` sparse-input / sparse-output pairs for wide lattices."""
    dim = 2**num_qubits
    payload = {}
    for k in range(NUM_PROBES):
        indices, amps = random_sparse_state(rng, dim, SPARSE_SUPPORT)
        psi_in = np.zeros(dim, dtype=complex)
        psi_in[indices] = amps

        psi_out = evolve_aer(circuit, psi_in)
        (nonzero,) = np.nonzero(np.abs(psi_out) > 1e-12)
        if nonzero.size > 1 << 16:
            raise RuntimeError(
                f"output support {nonzero.size} too large to store sparsely"
            )

        payload[f"in_idx_{k}"] = indices
        payload[f"in_amp_{k}"] = amps
        payload[f"out_idx_{k}"] = nonzero.astype(np.int64)
        payload[f"out_amp_{k}"] = psi_out[nonzero]
    return payload


def build_payload(case):
    """Fixture arrays for one case, keyed by the case ``mode``."""
    circuit = build_reference(case)
    num_qubits = circuit.num_qubits
    rng = case_rng(case)
    payload = {"n_qubits": np.asarray(num_qubits)}

    mode = case["mode"]
    if mode == "unitary":
        assert num_qubits <= 6, f"{case['id']}: {num_qubits} qubits is too wide"
        payload["unitary"] = Operator(circuit).data
    elif mode == "probes":
        assert num_qubits <= DENSE_LIMIT, f"{case['id']}: {num_qubits} qubits"
        payload.update(probe_payload(circuit, rng, num_qubits))
    elif mode == "sparse":
        payload.update(sparse_payload(circuit, rng, num_qubits))
    elif mode == "measurement":
        payload["measure_pairs"] = measurement_pairs(circuit)
        payload.update(probe_payload(strip_measurements(circuit), rng, num_qubits))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return num_qubits, payload


def self_check():
    """Pin the Aer path against the exact path on a small circuit."""
    case = next(c for c in CASES_SPACETIME if c["mode"] == "probes")
    circuit = build_reference(case)
    psi_in = random_state(np.random.default_rng(0), 2**circuit.num_qubits)
    deviation = np.max(
        np.abs(evolve_aer(circuit, psi_in) - evolve_dense(circuit, psi_in))
    )
    print(f"aer-vs-exact max deviation: {deviation:.3e}")
    assert deviation < 1e-12, "Aer disagrees with Statevector.evolve"


def main():
    """Write one fixture file per case."""
    self_check()
    for case in CASES_SPACETIME:
        num_qubits, payload = build_payload(case)
        save_npz(case["id"], **payload)
        print(f"{case['id']}: {case['mode']}, {num_qubits} qubits written")


if __name__ == "__main__":
    main()
