"""Cases for ``generate_fixtures_ms.py``; case ids double as fixture names.

``probe_state`` must give bit-identical vectors under both environments' numpy
versions, so it uses splitmix64 instead of ``numpy.random``.
"""

from typing import Any, Dict, List

import numpy as np

# Above this width the full output vector is too large to keep on disk, so
# fixtures store a fixed random subset of amplitudes instead.
FULL_STATE_MAX_QUBITS = 17

# Number of amplitudes retained for cases wider than FULL_STATE_MAX_QUBITS.
SAMPLE_SIZE = 8192

# Probe states per statevector case.
NUM_PROBES = 3

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GAMMA = np.uint64(0x9E3779B97F4A7C15)


def _splitmix64(state: np.ndarray) -> np.ndarray:
    """Finalizer of splitmix64 over a uint64 array (wraps mod 2**64)."""
    z = state.astype(np.uint64)
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def _uniform(state: np.ndarray) -> np.ndarray:
    """Map a uint64 array to float64 in [0, 1) using the top 53 bits."""
    return (_splitmix64(state) >> np.uint64(11)).astype(np.float64) * (2.0**-53)


def probe_state(n_qubits: int, index: int) -> np.ndarray:
    """Return probe state ``index`` for an ``n_qubits`` circuit.

    Deterministic and independent of the numpy version: the amplitudes come
    from splitmix64 applied to the basis index, so both venvs agree bit for
    bit.  LSB qubit ordering, matching qiskit and qarp alike.

    Parameters
    ----------
    n_qubits : int
        Width of the circuit the state is fed to.
    index : int
        Which of the probe states to build; ``0 <= index < NUM_PROBES``.

    Returns
    -------
    numpy.ndarray
        A normalised ``complex128`` vector of length ``2 ** n_qubits``.
    """
    dim = 1 << n_qubits
    # Python-int arithmetic for the offset: uint64 scalar multiply wraps but warns.
    offset = np.uint64(((index + 1) * int(_GAMMA)) % (1 << 64))
    base = np.arange(dim, dtype=np.uint64) + offset
    real = _uniform(base) - 0.5
    imag = _uniform(base ^ _M1) - 0.5
    psi = real + 1j * imag
    return psi / np.linalg.norm(psi)


def sample_indices(n_qubits: int) -> np.ndarray:
    """Return the fixed amplitude indices kept for wide cases.

    Deterministic, sorted, duplicate-free.  Used only when
    ``n_qubits > FULL_STATE_MAX_QUBITS``.

    Parameters
    ----------
    n_qubits : int
        Width of the circuit.

    Returns
    -------
    numpy.ndarray
        Sorted ``int64`` indices into a ``2 ** n_qubits`` state vector.
    """
    dim = 1 << n_qubits
    raw = _splitmix64(np.arange(SAMPLE_SIZE, dtype=np.uint64) + _GAMMA)
    return np.unique((raw % np.uint64(dim)).astype(np.int64))


# lattice specifications
#
# Velocities are 4 per dimension throughout: the MSLattice velocity register
# holds ``(velocities // 2).bit_length() - 1`` qubits, so 2 velocities/dim
# gives a zero-width register, which the qiskit-based qlbm rejects.

LATTICES: Dict[str, Any] = {
    # 2D, no geometry
    "2d_4x4_v4": {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    },
    "2d_4x8_v4": {
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    },
    # 2D, specular obstacle
    "2d_4x4_v4_spec": {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
        "geometry": [
            {"shape": "cuboid", "x": [1, 2], "y": [1, 2], "boundary": "specular"}
        ],
    },
    "2d_8x8_v4_spec": {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [
            {"shape": "cuboid", "x": [5, 6], "y": [1, 2], "boundary": "specular"}
        ],
    },
    # 2D, bounce-back obstacle
    "2d_4x4_v4_bb": {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
        "geometry": [
            {"shape": "cuboid", "x": [1, 2], "y": [1, 2], "boundary": "bounceback"}
        ],
    },
    "2d_8x8_v4_bb": {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [
            {"shape": "cuboid", "x": [5, 6], "y": [1, 2], "boundary": "bounceback"}
        ],
    },
    # 3D, no geometry (smallest 3D lattice the qiskit-based qlbm supports)
    "3d_2x2x2_v4": {
        "lattice": {
            "dim": {"x": 2, "y": 2, "z": 2},
            "velocities": {"x": 4, "y": 4, "z": 4},
        },
        "geometry": [],
    },
    # 3D, obstacles (smallest 3D geometry lattices: n=20 / n=18)
    "3d_4x2x2_v4_spec": {
        "lattice": {
            "dim": {"x": 4, "y": 2, "z": 2},
            "velocities": {"x": 4, "y": 4, "z": 4},
        },
        "geometry": [
            {
                "shape": "cuboid",
                "x": [1, 2],
                "y": [0, 1],
                "z": [0, 1],
                "boundary": "specular",
            }
        ],
    },
    "3d_4x2x2_v4_bb": {
        "lattice": {
            "dim": {"x": 4, "y": 2, "z": 2},
            "velocities": {"x": 4, "y": 4, "z": 4},
        },
        "geometry": [
            {
                "shape": "cuboid",
                "x": [1, 2],
                "y": [0, 1],
                "z": [0, 1],
                "boundary": "bounceback",
            }
        ],
    },
}


CASES: List[Dict[str, Any]] = [
    # StreamingAncillaPreparation
    {
        "id": "ms_sap_2d_4x8_d0_v1",
        "component": "streaming_ancilla_prep",
        "lattice": "2d_4x8_v4",
        "velocities": [1],
        "dim": 0,
    },
    {
        "id": "ms_sap_2d_4x8_d1_v01",
        "component": "streaming_ancilla_prep",
        "lattice": "2d_4x8_v4",
        "velocities": [0, 1],
        "dim": 1,
    },
    {
        "id": "ms_sap_3d_2x2x2_d2_v1",
        "component": "streaming_ancilla_prep",
        "lattice": "3d_2x2x2_v4",
        "velocities": [1],
        "dim": 2,
    },
    # ControlledIncrementer (all three reflection modes)
    {
        "id": "ms_ctrlinc_2d_4x4",
        "component": "controlled_incrementer",
        "lattice": "2d_4x4_v4",
        "reflection": None,
    },
    {
        "id": "ms_ctrlinc_2d_4x8",
        "component": "controlled_incrementer",
        "lattice": "2d_4x8_v4",
        "reflection": None,
    },
    {
        "id": "ms_ctrlinc_2d_4x4_spec",
        "component": "controlled_incrementer",
        "lattice": "2d_4x4_v4_spec",
        "reflection": "specular",
    },
    {
        "id": "ms_ctrlinc_2d_4x4_bb",
        "component": "controlled_incrementer",
        "lattice": "2d_4x4_v4_bb",
        "reflection": "bounceback",
    },
    {
        "id": "ms_ctrlinc_3d_2x2x2",
        "component": "controlled_incrementer",
        "lattice": "3d_2x2x2_v4",
        "reflection": None,
    },
    # MSStreamingOperator
    {
        "id": "ms_stream_2d_4x4_v1",
        "component": "streaming_operator",
        "lattice": "2d_4x4_v4",
        "velocities": [1],
    },
    {
        "id": "ms_stream_2d_4x8_v01",
        "component": "streaming_operator",
        "lattice": "2d_4x8_v4",
        "velocities": [0, 1],
    },
    {
        "id": "ms_stream_3d_2x2x2_v1",
        "component": "streaming_operator",
        "lattice": "3d_2x2x2_v4",
        "velocities": [1],
    },
    # initial conditions
    {
        "id": "ms_init_2d_4x8",
        "component": "initial_conditions",
        "lattice": "2d_4x8_v4",
    },
    {
        "id": "ms_init_3d_2x2x2",
        "component": "initial_conditions",
        "lattice": "3d_2x2x2_v4",
    },
    {
        "id": "ms_init_slim_3d_2x2x2",
        "component": "initial_conditions_3d_slim",
        "lattice": "3d_2x2x2_v4",
    },
    # GridMeasurement (measurement-pair oracle)
    {
        "id": "ms_gridmeas_2d_4x8",
        "component": "grid_measurement",
        "lattice": "2d_4x8_v4",
    },
    {
        "id": "ms_gridmeas_3d_2x2x2",
        "component": "grid_measurement",
        "lattice": "3d_2x2x2_v4",
    },
    # comparators
    {
        "id": "ms_specwallcomp_2d_8x8_in",
        "component": "specular_wall_comparator",
        "lattice": "2d_8x8_v4_spec",
        "wall": ["inside", 0, 0],
    },
    {
        "id": "ms_specwallcomp_2d_8x8_out",
        "component": "specular_wall_comparator",
        "lattice": "2d_8x8_v4_spec",
        "wall": ["outside", 1, 1],
    },
    {
        "id": "ms_bbwallcomp_2d_8x8_in",
        "component": "bounceback_wall_comparator",
        "lattice": "2d_8x8_v4_bb",
        "wall": ["inside", 0, 0],
    },
    {
        "id": "ms_bbwallcomp_2d_8x8_out",
        "component": "bounceback_wall_comparator",
        "lattice": "2d_8x8_v4_bb",
        "wall": ["outside", 1, 1],
    },
    {
        "id": "ms_edgecomp_3d_4x2x2_nce0",
        "component": "edge_comparator",
        "lattice": "3d_4x2x2_v4_spec",
        "edge": ["near_corner", 0],
    },
    {
        "id": "ms_edgecomp_3d_4x2x2_ce3",
        "component": "edge_comparator",
        "lattice": "3d_4x2x2_v4_spec",
        "edge": ["corner", 3],
    },
    # reflection sub-circuits, 2D
    {
        "id": "ms_spec_reflect_wall_2d_in",
        "component": "reflect_wall",
        "boundary": "specular",
        "lattice": "2d_8x8_v4_spec",
        "wall": ["inside", 0, 0],
    },
    {
        "id": "ms_spec_reflect_wall_2d_out",
        "component": "reflect_wall",
        "boundary": "specular",
        "lattice": "2d_8x8_v4_spec",
        "wall": ["outside", 1, 1],
    },
    {
        "id": "ms_bb_reflect_wall_2d_in",
        "component": "reflect_wall",
        "boundary": "bounceback",
        "lattice": "2d_8x8_v4_bb",
        "wall": ["inside", 0, 0],
    },
    {
        "id": "ms_bb_reflect_wall_2d_out",
        "component": "reflect_wall",
        "boundary": "bounceback",
        "lattice": "2d_8x8_v4_bb",
        "wall": ["outside", 1, 1],
    },
    {
        "id": "ms_spec_reset_point_2d",
        "component": "reset_point_state",
        "boundary": "specular",
        "lattice": "2d_8x8_v4_spec",
        "point": ["corner_outside", 0],
    },
    {
        "id": "ms_bb_reset_point_2d",
        "component": "reset_point_state",
        "boundary": "bounceback",
        "lattice": "2d_8x8_v4_bb",
        "point": ["near_corner_2d", 1],
    },
    {
        "id": "ms_spec_flip_stream_2d",
        "component": "flip_and_stream",
        "boundary": "specular",
        "lattice": "2d_8x8_v4_spec",
    },
    {
        "id": "ms_bb_flip_stream_2d",
        "component": "flip_and_stream",
        "boundary": "bounceback",
        "lattice": "2d_8x8_v4_bb",
    },
    # reflection sub-circuits, 3D (edge / point reset paths)
    {
        "id": "ms_spec_reset_edge_3d_nce0",
        "component": "reset_edge_state",
        "boundary": "specular",
        "lattice": "3d_4x2x2_v4_spec",
        "edge": ["near_corner", 0],
    },
    {
        "id": "ms_spec_reset_edge_3d_ce3",
        "component": "reset_edge_state",
        "boundary": "specular",
        "lattice": "3d_4x2x2_v4_spec",
        "edge": ["corner", 3],
    },
    {
        "id": "ms_bb_reset_edge_3d_nce7",
        "component": "reset_edge_state",
        "boundary": "bounceback",
        "lattice": "3d_4x2x2_v4_bb",
        "edge": ["near_corner", 7],
    },
    {
        "id": "ms_bb_reset_edge_3d_ce0",
        "component": "reset_edge_state",
        "boundary": "bounceback",
        "lattice": "3d_4x2x2_v4_bb",
        "edge": ["corner", 0],
    },
    {
        "id": "ms_spec_reset_point_3d_onc5",
        "component": "reset_point_state",
        "boundary": "specular",
        "lattice": "3d_4x2x2_v4_spec",
        "point": ["overlapping_near_corner_3d", 5],
    },
    {
        "id": "ms_bb_reset_point_3d_co2",
        "component": "reset_point_state",
        "boundary": "bounceback",
        "lattice": "3d_4x2x2_v4_bb",
        "point": ["corner_outside", 2],
    },
    {
        "id": "ms_spec_reflect_wall_3d_in",
        "component": "reflect_wall",
        "boundary": "specular",
        "lattice": "3d_4x2x2_v4_spec",
        "wall": ["inside", 2, 0],
    },
    {
        "id": "ms_bb_reflect_wall_3d_out",
        "component": "reflect_wall",
        "boundary": "bounceback",
        "lattice": "3d_4x2x2_v4_bb",
        "wall": ["outside", 2, 1],
    },
    {
        "id": "ms_spec_flip_stream_3d",
        "component": "flip_and_stream",
        "boundary": "specular",
        "lattice": "3d_4x2x2_v4_spec",
    },
    {
        "id": "ms_bb_flip_stream_3d",
        "component": "flip_and_stream",
        "boundary": "bounceback",
        "lattice": "3d_4x2x2_v4_bb",
    },
    # whole reflection operators (2D)
    {
        "id": "ms_specular_op_2d_4x4",
        "component": "specular_reflection_operator",
        "lattice": "2d_4x4_v4_spec",
    },
    {
        "id": "ms_bounceback_op_2d_4x4",
        "component": "bounceback_reflection_operator",
        "lattice": "2d_4x4_v4_bb",
    },
    # end-to-end algorithm
    {
        "id": "ms_msqlbm_2d_4x4",
        "component": "msqlbm",
        "lattice": "2d_4x4_v4",
        "group_velocities": False,
    },
    {
        "id": "ms_msqlbm_2d_4x4_bb",
        "component": "msqlbm",
        "lattice": "2d_4x4_v4_bb",
        "group_velocities": True,
    },
]
