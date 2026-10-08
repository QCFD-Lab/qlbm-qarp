"""Cases for ``generate_fixtures_spacetime.py``; case ids double as fixture names.

``mode`` is ``unitary`` (at most 6 qubits), ``probes``, ``sparse`` (index/amplitude
pairs for lattices too wide for dense vectors) or ``measurement``.  The grid must
hold the stencil: D1Q2 needs ``2 * t + 1`` gridpoints, so ``dim.x == 4`` allows one
time step; the narrowest lattice has 8 qubits, so ``unitary`` covers only the
register-local D2Q4 collision sub-circuits.
"""

from typing import Any, Dict, List

NUM_PROBES = 3

# Amplitude support of one ``sparse``-mode input vector.
SPARSE_SUPPORT = 4

# Widest lattice still stored as dense probe vectors.
DENSE_LIMIT = 11


def _lattice(
    lattice_data,
    num_timesteps,
    include_measurement_qubit=False,
    use_volumetric_ops=False,
):
    """Build the (pure-data) lattice descriptor consumed by both sides."""
    return {
        "lattice_data": lattice_data,
        "num_timesteps": num_timesteps,
        "include_measurement_qubit": include_measurement_qubit,
        "use_volumetric_ops": use_volumetric_ops,
    }


_D1Q2_X4 = {"lattice": {"dim": {"x": 4}, "velocities": "D1Q2"}, "geometry": []}
_D1Q2_X4_BB = {
    "lattice": {"dim": {"x": 4}, "velocities": "D1Q2"},
    "geometry": [{"shape": "cuboid", "x": [1, 2], "boundary": "bounceback"}],
}
_D1Q2_X4_BB1 = {
    "lattice": {"dim": {"x": 4}, "velocities": "D1Q2"},
    "geometry": [{"shape": "cuboid", "x": [2, 2], "boundary": "bounceback"}],
}
_D1Q2_X8 = {"lattice": {"dim": {"x": 8}, "velocities": "D1Q2"}, "geometry": []}
_D1Q2_X8_BB = {
    "lattice": {"dim": {"x": 8}, "velocities": "D1Q2"},
    "geometry": [{"shape": "cuboid", "x": [3, 4], "boundary": "bounceback"}],
}
_D2Q4_2X2 = {"lattice": {"dim": {"x": 2, "y": 2}, "velocities": "D2Q4"}, "geometry": []}
_D2Q4_2X2_BB = {
    "lattice": {"dim": {"x": 2, "y": 2}, "velocities": "D2Q4"},
    "geometry": [
        {"shape": "cuboid", "x": [1, 1], "y": [1, 1], "boundary": "bounceback"}
    ],
}

CASES_SPACETIME: List[Dict[str, Any]] = [
    # streaming
    {
        "id": "st_stream_d1q2_x4_t1",
        "component": "st_streaming",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4, 1),
        "timestep": 1,
    },
    {
        "id": "st_stream_d1q2_x8_t1",
        "component": "st_streaming",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X8, 1),
        "timestep": 1,
    },
    {
        "id": "st_stream_d1q2_x8_t2_ts1",
        "component": "st_streaming",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8, 2),
        "timestep": 1,
    },
    {
        "id": "st_stream_d1q2_x8_t2_ts2",
        "component": "st_streaming",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8, 2),
        "timestep": 2,
    },
    {
        "id": "st_stream_d1q2_x8_t3_ts3",
        "component": "st_streaming",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8, 3),
        "timestep": 3,
    },
    {
        "id": "st_stream_d2q4_2x2_t1",
        "component": "st_streaming",
        "mode": "sparse",
        "lattice": _lattice(_D2Q4_2X2, 1),
        "timestep": 1,
    },
    # point-wise initial conditions
    {
        "id": "st_init_pw_d1q2_x4_t1",
        "component": "st_pointwise_initial",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4, 1),
        "grid_data": [((0,), (True, False))],
        "filter_inside_blocks": True,
    },
    {
        "id": "st_init_pw_d1q2_x8_t2",
        "component": "st_pointwise_initial",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8, 2),
        "grid_data": [((1,), (True, False)), ((6,), (False, True))],
        "filter_inside_blocks": True,
    },
    # the obstacle swallows one of the two populations and part of a stencil
    {
        "id": "st_init_pw_d1q2_x8_t2_bb_filtered",
        "component": "st_pointwise_initial",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8_BB, 2),
        "grid_data": [((3,), (True, True)), ((6,), (True, False))],
        "filter_inside_blocks": True,
    },
    {
        "id": "st_init_pw_d1q2_x8_t2_bb_unfiltered",
        "component": "st_pointwise_initial",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8_BB, 2),
        "grid_data": [((3,), (True, True)), ((6,), (True, False))],
        "filter_inside_blocks": False,
    },
    {
        "id": "st_init_pw_d2q4_2x2_t1",
        "component": "st_pointwise_initial",
        "mode": "sparse",
        "lattice": _lattice(_D2Q4_2X2, 1),
        "grid_data": [((0, 0), (True, False, True, False))],
        "filter_inside_blocks": True,
    },
    # volumetric initial conditions
    {
        "id": "st_init_vol_d1q2_x4_t1",
        "component": "st_volumetric_initial",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4, 1, use_volumetric_ops=True),
        "cuboid_bounds": [(1, 2)],
        "velocity_profile": (1, 1),
    },
    {
        "id": "st_init_vol_d1q2_x8_t1",
        "component": "st_volumetric_initial",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X8, 1, use_volumetric_ops=True),
        "cuboid_bounds": [(2, 5)],
        "velocity_profile": (1, 0),
    },
    # bounds that wrap around the periodic boundary
    {
        "id": "st_init_vol_d1q2_x8_t2",
        "component": "st_volumetric_initial",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8, 2, use_volumetric_ops=True),
        "cuboid_bounds": [(6, 7)],
        "velocity_profile": (0, 1),
    },
    # point-wise reflection
    {
        "id": "st_refl_pw_d1q2_x4_t1_bb",
        "component": "st_pointwise_reflection",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4_BB, 1),
        "timestep": 1,
        "filter_inside_blocks": True,
    },
    {
        "id": "st_refl_pw_d1q2_x4_t1_bb1",
        "component": "st_pointwise_reflection",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4_BB1, 1),
        "timestep": 1,
        "filter_inside_blocks": False,
    },
    {
        "id": "st_refl_pw_d1q2_x8_t2_ts1_bb",
        "component": "st_pointwise_reflection",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8_BB, 2),
        "timestep": 1,
        "filter_inside_blocks": True,
    },
    {
        "id": "st_refl_pw_d1q2_x8_t2_ts2_bb",
        "component": "st_pointwise_reflection",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8_BB, 2),
        "timestep": 2,
        "filter_inside_blocks": True,
    },
    {
        "id": "st_refl_pw_d2q4_2x2_t1_bb",
        "component": "st_pointwise_reflection",
        "mode": "sparse",
        "lattice": _lattice(_D2Q4_2X2_BB, 1),
        "timestep": 1,
        "filter_inside_blocks": True,
    },
    # volumetric reflection (comparators only)
    {
        "id": "st_refl_vol_d1q2_x4_t1_bb",
        "component": "st_volumetric_reflection",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4_BB, 1, use_volumetric_ops=True),
        "timestep": 1,
        "filter_inside_blocks": True,
    },
    {
        "id": "st_refl_vol_d1q2_x8_t2_bb",
        "component": "st_volumetric_reflection",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8_BB, 2, use_volumetric_ops=True),
        "timestep": 2,
        "filter_inside_blocks": True,
    },
    # equivalence-class collision
    # D1Q2 has only singleton equivalence classes: the operator is empty.
    {
        "id": "st_coll_eqc_d1q2_x4_t1",
        "component": "st_eqc_collision",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4, 1),
        "timestep": 1,
    },
    {
        "id": "st_coll_eqc_d2q4_2x2_t1",
        "component": "st_eqc_collision",
        "mode": "sparse",
        "lattice": _lattice(_D2Q4_2X2, 1),
        "timestep": 1,
    },
    # D2Q4 collision (d2q4_old)
    {
        "id": "st_coll_d2q4old_local_set",
        "component": "st_d2q4_local_collision",
        "mode": "unitary",
        "lattice": _lattice(_D2Q4_2X2, 1),
        "reset_state": False,
    },
    {
        "id": "st_coll_d2q4old_local_reset",
        "component": "st_d2q4_local_collision",
        "mode": "unitary",
        "lattice": _lattice(_D2Q4_2X2, 1),
        "reset_state": True,
    },
    {
        "id": "st_coll_d2q4old_2x2_t1",
        "component": "st_d2q4_collision",
        "mode": "sparse",
        "lattice": _lattice(_D2Q4_2X2, 1),
        "timestep": 1,
    },
    # measurement
    {
        "id": "st_meas_gridvel_d1q2_x4_t1",
        "component": "st_grid_velocity_measurement",
        "mode": "measurement",
        "lattice": _lattice(_D1Q2_X4, 1),
    },
    {
        "id": "st_meas_gridvel_d1q2_x8_t1",
        "component": "st_grid_velocity_measurement",
        "mode": "measurement",
        "lattice": _lattice(_D1Q2_X8, 1),
    },
    {
        "id": "st_meas_mass_d1q2_x4_t1_v0",
        "component": "st_mass_measurement",
        "mode": "measurement",
        "lattice": _lattice(_D1Q2_X4, 1, include_measurement_qubit=True),
        "gridpoint": (1,),
        "velocity_index_to_measure": 0,
    },
    {
        "id": "st_meas_mass_d1q2_x4_t1_v1",
        "component": "st_mass_measurement",
        "mode": "measurement",
        "lattice": _lattice(_D1Q2_X4, 1, include_measurement_qubit=True),
        "gridpoint": (2,),
        "velocity_index_to_measure": 1,
    },
    # end-to-end algorithm
    {
        "id": "st_qlbm_d1q2_x4_t1_bb",
        "component": "st_qlbm",
        "mode": "probes",
        "lattice": _lattice(_D1Q2_X4_BB, 1),
        "filter_inside_blocks": True,
    },
    {
        "id": "st_qlbm_d1q2_x8_t2_bb",
        "component": "st_qlbm",
        "mode": "sparse",
        "lattice": _lattice(_D1Q2_X8_BB, 2),
        "filter_inside_blocks": True,
    },
    {
        "id": "st_qlbm_d2q4_2x2_t1_bb",
        "component": "st_qlbm",
        "mode": "sparse",
        "lattice": _lattice(_D2Q4_2X2_BB, 1),
        "filter_inside_blocks": True,
    },
]
