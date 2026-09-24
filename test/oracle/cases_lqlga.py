"""Cases for ``generate_fixtures_lqlga.py``; case ids double as fixture names.

``mode`` is ``unitary`` (at most 6 qubits), ``probes``, ``measurement`` or ``evolution``.
``dim`` entries are one larger than ``lattice.num_gridpoints``.
"""

from typing import Any, Dict, List

NUM_PROBES = 3

# Lattice specs (the "lattice" key follows the qlbm input-dict format).
_D1Q2_X2 = {"lattice": {"dim": {"x": 2}, "velocities": "D1Q2"}, "geometry": []}
_D1Q2_X3 = {"lattice": {"dim": {"x": 3}, "velocities": "D1Q2"}, "geometry": []}
_D1Q2_X4 = {"lattice": {"dim": {"x": 4}, "velocities": "D1Q2"}, "geometry": []}
_D1Q3_X2 = {"lattice": {"dim": {"x": 2}, "velocities": "D1Q3"}, "geometry": []}
_D1Q3_X3 = {"lattice": {"dim": {"x": 3}, "velocities": "D1Q3"}, "geometry": []}
# 2 gridpoints x 4 velocities = 8 qubits; the smallest 2D discretization.
_D2Q4_2X1 = {
    "lattice": {"dim": {"x": 2, "y": 1}, "velocities": "D2Q4"},
    "geometry": [],
}

_D1Q2_X3_BB = {
    "lattice": {"dim": {"x": 3}, "velocities": "D1Q2"},
    "geometry": [{"shape": "cuboid", "x": [1, 1], "boundary": "bounceback"}],
}
_D1Q3_X2_BB = {
    "lattice": {"dim": {"x": 2}, "velocities": "D1Q3"},
    "geometry": [{"shape": "cuboid", "x": [1, 1], "boundary": "bounceback"}],
}
_D1Q3_X3_BB = {
    "lattice": {"dim": {"x": 3}, "velocities": "D1Q3"},
    "geometry": [{"shape": "cuboid", "x": [1, 1], "boundary": "bounceback"}],
}
_D1Q2_X4_SPEC = {
    "lattice": {"dim": {"x": 4}, "velocities": "D1Q2"},
    "geometry": [{"shape": "cuboid", "x": [2, 3], "boundary": "specular"}],
}
_D1Q2_X2_BB = {
    "lattice": {"dim": {"x": 2}, "velocities": "D1Q2"},
    "geometry": [{"shape": "cuboid", "x": [1, 1], "boundary": "bounceback"}],
}

# Multi-geometry configurations (applied via lattice.set_geometries).
_GEOMS_2 = [
    [{"shape": "cuboid", "x": [1, 1], "boundary": "bounceback"}],
    [{"shape": "cuboid", "x": [0, 0], "boundary": "specular"}],
]
_GEOMS_3 = [
    [{"shape": "cuboid", "x": [1, 1], "boundary": "bounceback"}],
    [{"shape": "cuboid", "x": [0, 0], "boundary": "specular"}],
    [{"shape": "cuboid", "x": [0, 1], "boundary": "bounceback"}],
]

CASES_LQLGA: List[Dict[str, Any]] = [
    # streaming
    {
        "id": "lqlga_stream_d1q2_x3",
        "component": "lqlga_streaming",
        "mode": "unitary",
        "lattice": _D1Q2_X3,
    },
    {
        "id": "lqlga_stream_d1q3_x2",
        "component": "lqlga_streaming",
        "mode": "unitary",
        "lattice": _D1Q3_X2,
    },
    {
        "id": "lqlga_stream_d1q2_x4",
        "component": "lqlga_streaming",
        "mode": "probes",
        "lattice": _D1Q2_X4,
    },
    {
        "id": "lqlga_stream_d1q3_x3",
        "component": "lqlga_streaming",
        "mode": "probes",
        "lattice": _D1Q3_X3,
    },
    {
        "id": "lqlga_stream_d2q4_2x1",
        "component": "lqlga_streaming",
        "mode": "probes",
        "lattice": _D2Q4_2X1,
    },
    # collision
    # D1Q2 has no non-trivial equivalence class: pins the empty-block path.
    {
        "id": "lqlga_coll_d1q2_x3",
        "component": "lqlga_collision",
        "mode": "unitary",
        "lattice": _D1Q2_X3,
    },
    {
        "id": "lqlga_coll_d1q3_x2",
        "component": "lqlga_collision",
        "mode": "unitary",
        "lattice": _D1Q3_X2,
    },
    {
        "id": "lqlga_coll_d1q3_x3",
        "component": "lqlga_collision",
        "mode": "probes",
        "lattice": _D1Q3_X3,
    },
    {
        "id": "lqlga_coll_d2q4_2x1",
        "component": "lqlga_collision",
        "mode": "probes",
        "lattice": _D2Q4_2X1,
    },
    # reflection (single geometry)
    {
        "id": "lqlga_refl_d1q2_x3_bb",
        "component": "lqlga_reflection",
        "mode": "unitary",
        "lattice": _D1Q2_X3_BB,
    },
    {
        "id": "lqlga_refl_d1q3_x2_bb",
        "component": "lqlga_reflection",
        "mode": "unitary",
        "lattice": _D1Q3_X2_BB,
    },
    {
        "id": "lqlga_refl_d1q2_x4_spec",
        "component": "lqlga_reflection",
        "mode": "probes",
        "lattice": _D1Q2_X4_SPEC,
    },
    {
        "id": "lqlga_refl_d1q3_x3_bb",
        "component": "lqlga_reflection",
        "mode": "probes",
        "lattice": _D1Q3_X3_BB,
    },
    # reflection (multiple geometries)
    {
        "id": "lqlga_mgrefl_d1q2_x2_g2",
        "component": "lqlga_mg_reflection",
        "mode": "unitary",
        "lattice": _D1Q2_X2,
        "geometries": _GEOMS_2,
    },
    {
        "id": "lqlga_mgrefl_d1q2_x2_g3",
        "component": "lqlga_mg_reflection",
        "mode": "unitary",
        "lattice": _D1Q2_X2,
        "geometries": _GEOMS_3,
    },
    {
        "id": "lqlga_mgrefl_d1q3_x2_g2",
        "component": "lqlga_mg_reflection",
        "mode": "probes",
        "lattice": _D1Q3_X2,
        "geometries": _GEOMS_2,
    },
    # initial conditions
    # Includes an all-False velocity profile (skipped-gridpoint branch).
    {
        "id": "lqlga_ic_d1q2_x3",
        "component": "lqlga_initial",
        "mode": "unitary",
        "lattice": _D1Q2_X3,
        "grid_data": [
            [[0], [True, False]],
            [[1], [False, False]],
            [[2], [True, True]],
        ],
    },
    {
        "id": "lqlga_ic_d1q3_x2",
        "component": "lqlga_initial",
        "mode": "unitary",
        "lattice": _D1Q3_X2,
        "grid_data": [[[1], [True, True, False]]],
    },
    {
        "id": "lqlga_ic_mg_d1q2_x2",
        "component": "lqlga_initial",
        "mode": "unitary",
        "lattice": _D1Q2_X2,
        "geometries": _GEOMS_2,
        "grid_data": [[[0], [True, False]]],
    },
    {
        "id": "lqlga_ic_acc_d1q2_x2",
        "component": "lqlga_initial",
        "mode": "unitary",
        "lattice": _D1Q2_X2,
        "accumulation": {"size": 2, "indices": [0, 1]},
        "grid_data": [[[1], [False, True]]],
    },
    # averaged initial conditions
    {
        "id": "lqlga_avgic_d1q2_x3",
        "component": "lqlga_averaged_initial",
        "mode": "unitary",
        "lattice": _D1Q2_X3,
        "gridpoints": [0, 2],
    },
    {
        "id": "lqlga_avgic_d1q3_x2",
        "component": "lqlga_averaged_initial",
        "mode": "unitary",
        "lattice": _D1Q3_X2,
        "gridpoints": [1],
    },
    # measurement
    {
        "id": "lqlga_meas_d1q2_x3",
        "component": "lqlga_measurement",
        "mode": "measurement",
        "lattice": _D1Q2_X3,
    },
    {
        "id": "lqlga_meas_acc_d1q2_x2",
        "component": "lqlga_measurement",
        "mode": "measurement",
        "lattice": _D1Q2_X2,
        "accumulation": {"size": 2, "indices": [0, 1]},
    },
    {
        "id": "lqlga_meas_mg_d1q2_x2",
        "component": "lqlga_measurement",
        "mode": "measurement",
        "lattice": _D1Q2_X2,
        "geometries": _GEOMS_2,
    },
    # LQLGA end-to-end operator
    {
        "id": "lqlga_e2e_d1q2_x3_bb",
        "component": "lqlga",
        "mode": "unitary",
        "lattice": _D1Q2_X3_BB,
    },
    {
        "id": "lqlga_e2e_d1q3_x2_bb",
        "component": "lqlga",
        "mode": "unitary",
        "lattice": _D1Q3_X2_BB,
    },
    {
        "id": "lqlga_e2e_mg_d1q2_x2_g2",
        "component": "lqlga",
        "mode": "unitary",
        "lattice": _D1Q2_X2,
        "geometries": _GEOMS_2,
    },
    {
        "id": "lqlga_e2e_d1q2_x4_free",
        "component": "lqlga",
        "mode": "probes",
        "lattice": _D1Q2_X4,
    },
    # end-to-end evolution statevector (smallest lattice)
    {
        "id": "lqlga_evol_d1q2_x2_bb",
        "component": "lqlga_evolution",
        "mode": "evolution",
        "lattice": _D1Q2_X2_BB,
        "grid_data": [[[0], [True, False]]],
        "steps": 2,
    },
    {
        "id": "lqlga_evol_d1q3_x2_bb",
        "component": "lqlga_evolution",
        "mode": "evolution",
        "lattice": _D1Q3_X2_BB,
        "grid_data": [[[0], [True, False, True]]],
        "steps": 3,
    },
]
