"""Cases for ``generate_fixtures_infra.py``; case ids double as fixture names.

``mode`` is ``trajectory``, ``decode`` or ``reinit``.  Counts are ``(cbit_key, value)``
pairs with LSB integer keys; ``n_cbits`` is per case because ancilla, marker and
accumulation qubits stay unmeasured.
"""

from typing import Any, Dict, List

# lattices
_MS_2D_4X4 = {
    "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 2, "y": 2}},
    "geometry": [],
}
# MSQLBM needs at least 2 velocity qubits per dimension; the decode-only
# lattices above stay narrower because only their grid register is measured.
_MS_2D_4X4_V4 = {
    "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
    "geometry": [],
}
_MS_2D_4X4_V4_BB = {
    "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
    "geometry": [
        {"shape": "cuboid", "x": [2, 2], "y": [1, 2], "boundary": "bounceback"}
    ],
}
_MS_3D_4X4X4 = {
    "lattice": {
        "dim": {"x": 4, "y": 4, "z": 4},
        "velocities": {"x": 2, "y": 2, "z": 2},
    },
    "geometry": [],
}
_MS_1D_X8 = {"lattice": {"dim": {"x": 8}, "velocities": {"x": 2}}, "geometry": []}

_ST_1D_X8 = {"lattice": {"dim": {"x": 8}, "velocities": "D1Q2"}, "geometry": []}
_ST_2D_4X4 = {
    "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
    "geometry": [],
}

_LQLGA_1D_X4 = {"lattice": {"dim": {"x": 4}, "velocities": "D1Q2"}, "geometry": []}
_LQLGA_1D_X4_BB = {
    "lattice": {"dim": {"x": 4}, "velocities": "D1Q2"},
    "geometry": [{"shape": "cuboid", "x": [2, 2], "boundary": "bounceback"}],
}
_LQLGA_1D_X3_Q3 = {"lattice": {"dim": {"x": 3}, "velocities": "D1Q3"}, "geometry": []}

# trajectories
CASES_TRAJECTORY: List[Dict[str, Any]] = [
    {
        "id": "infra_traj_ms_2d_4x4",
        "mode": "trajectory",
        "family": "ms",
        "lattice": _MS_2D_4X4_V4,
        "num_steps": 2,
        "num_shots": 4096,
    },
    {
        "id": "infra_traj_ms_2d_4x4_bb",
        "mode": "trajectory",
        "family": "ms",
        "lattice": _MS_2D_4X4_V4_BB,
        "num_steps": 2,
        "num_shots": 4096,
    },
    {
        "id": "infra_traj_lqlga_1d_x4",
        "mode": "trajectory",
        "family": "lqlga",
        "lattice": _LQLGA_1D_X4,
        "grid_data": [[[0], [1, 0]], [[2], [0, 1]]],
        "num_steps": 3,
        "num_shots": 4096,
    },
    {
        "id": "infra_traj_lqlga_1d_x4_bb",
        "mode": "trajectory",
        "family": "lqlga",
        "lattice": _LQLGA_1D_X4_BB,
        "grid_data": [[[0], [1, 0]]],
        "num_steps": 2,
        "num_shots": 4096,
    },
    {
        # Space-Time re-synthesizes its state from counts every step, so this
        # case also exercises the reinitializer inside the runner loop.
        "id": "infra_traj_spacetime_1d_x8",
        "mode": "trajectory",
        "family": "spacetime",
        "lattice": _ST_1D_X8,
        "num_timesteps": 1,
        "grid_data": [[[2], [1, 0]]],
        "num_steps": 3,
        "num_shots": 16384,
        "statevector_sampling": True,
    },
]

# decoders
# Amplitude decoder cases stay on square/cubic lattices: the reference's
# x/y(/z) transposition is invisible exactly there.
CASES_DECODE: List[Dict[str, Any]] = [
    {
        "id": "infra_decode_amplitude_1d_grid",
        "mode": "decode",
        "decoder": "amplitude",
        "lattice_family": "ms",
        "lattice": _MS_1D_X8,
        "n_cbits": 3,
        "counts": [[0, 100], [1, 50], [5, 25], [7, 12]],
    },
    {
        # 4 cbits = 3 grid + 1 velocity, which turns the "rest bonus" on.
        "id": "infra_decode_amplitude_1d_grid_velocity",
        "mode": "decode",
        "decoder": "amplitude",
        "lattice_family": "ms",
        "lattice": _MS_1D_X8,
        "n_cbits": 4,
        "counts": [[0, 100], [1, 50], [8, 25], [13, 12], [15, 7]],
    },
    {
        "id": "infra_decode_amplitude_2d_square",
        "mode": "decode",
        "decoder": "amplitude",
        "lattice_family": "ms",
        "lattice": _MS_2D_4X4,
        "n_cbits": 4,
        "counts": [[0, 10], [1, 20], [4, 30], [5, 40], [10, 50], [15, 60]],
    },
    {
        "id": "infra_decode_amplitude_3d_cube",
        "mode": "decode",
        "decoder": "amplitude",
        "lattice_family": "ms",
        "lattice": _MS_3D_4X4X4,
        "n_cbits": 6,
        "counts": [[0, 10], [1, 20], [9, 30], [21, 40], [42, 50], [63, 60]],
    },
    {
        "id": "infra_decode_spacetime_1d",
        "mode": "decode",
        "decoder": "spacetime",
        "lattice_family": "spacetime",
        "lattice": _ST_1D_X8,
        "num_timesteps": 1,
        "n_cbits": 5,
        "counts": [[0, 10], [2, 20], [10, 30], [19, 40], [31, 50]],
    },
    {
        "id": "infra_decode_spacetime_2d",
        "mode": "decode",
        "decoder": "spacetime",
        "lattice_family": "spacetime",
        "lattice": _ST_2D_4X4,
        "num_timesteps": 1,
        "n_cbits": 8,
        "counts": [[0, 10], [17, 20], [90, 30], [175, 40], [255, 50]],
    },
    {
        "id": "infra_decode_lqlga_1d_q2",
        "mode": "decode",
        "decoder": "lqlga",
        "lattice_family": "lqlga",
        "lattice": _LQLGA_1D_X4,
        "n_cbits": 8,
        "counts": [[1, 10], [2, 20], [66, 30], [129, 40], [255, 50]],
    },
    {
        "id": "infra_decode_lqlga_1d_q3",
        "mode": "decode",
        "decoder": "lqlga",
        "lattice_family": "lqlga",
        "lattice": _LQLGA_1D_X3_Q3,
        "n_cbits": 9,
        "counts": [[1, 10], [4, 20], [73, 30], [292, 40], [511, 50]],
    },
]

# reinitializers
CASES_REINIT: List[Dict[str, Any]] = [
    {
        "id": "infra_reinit_spacetime_1d",
        "mode": "reinit",
        "lattice": _ST_1D_X8,
        "num_timesteps": 1,
        "n_cbits": 5,
        # Keys with a zero velocity profile must be filtered out.
        "counts": [[0, 10], [2, 20], [10, 30], [19, 40], [31, 50]],
    },
    {
        "id": "infra_reinit_spacetime_2d",
        "mode": "reinit",
        # A 2D Space-Time lattice is 24 qubits wide; pin the decoding only.
        "store_state": False,
        "lattice": _ST_2D_4X4,
        "num_timesteps": 1,
        "n_cbits": 8,
        "counts": [[0, 10], [17, 20], [90, 30], [175, 40], [255, 50]],
    },
]

CASES_INFRA = CASES_TRAJECTORY + CASES_DECODE + CASES_REINIT
