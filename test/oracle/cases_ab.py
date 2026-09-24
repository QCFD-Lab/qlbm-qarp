"""Cases for ``generate_fixtures_ab.py``; case ids double as fixture names.

``mode`` is ``unitary`` (at most 6 qubits), ``probe`` or ``measure``.  AB circuits
permute basis states, so probes are basis states plus sparse random-phase
superpositions: small on disk, yet sensitive to relative phases.
"""

from typing import Any, Dict, List

# Smallest AB lattices that still exercise every code path.
LATTICE_D1Q3 = {"lattice": {"dim": {"x": 8}, "velocities": "d1q3"}, "geometry": []}
LATTICE_D2Q9 = {
    "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
    "geometry": [],
}
LATTICE_D2Q9_BB = {
    "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
    "geometry": [
        {"shape": "cuboid", "x": [1, 2], "y": [1, 2], "boundary": "bounceback"}
    ],
}
LATTICE_D2Q9_SR = {
    "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
    "geometry": [{"shape": "cuboid", "x": [1, 2], "y": [1, 2], "boundary": "specular"}],
}
LATTICE_YMONOMIAL: Dict[str, Any] = {
    "lattice": {"dim": {"x": 2, "y": 4}, "velocities": "d2q9"},
    "geometry": [
        {
            "shape": "ymonomial",
            "exponent": 2,
            "comparator": "<",
            "boundary": "bounceback",
        }
    ],
}

CASES: List[Dict[str, Any]] = [
    # QFT multiplier
    {
        "id": "ab_rgqft_1_2",
        "component": "rgqft_multiplier",
        "n": 1,
        "m": 2,
        "mode": "unitary",
    },
    {
        "id": "ab_rgqft_2_2",
        "component": "rgqft_multiplier",
        "n": 2,
        "m": 2,
        "mode": "unitary",
    },
    {
        "id": "ab_rgqft_2_3",
        "component": "rgqft_multiplier",
        "n": 2,
        "m": 3,
        "mode": "probe",
    },
    {
        "id": "ab_rgqft_3_6",
        "component": "rgqft_multiplier",
        "n": 3,
        "m": 6,
        "mode": "probe",
    },
    # BinaryToOHPermutation
    {
        "id": "ab_binoh_d1q3",
        "component": "binary_to_oh",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D1Q3,
        "mode": "unitary",
    },
    {
        "id": "ab_binoh_d2q9",
        "component": "binary_to_oh",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "mode": "unitary",
    },
    {
        "id": "ab_binoh_oh_d2q9",
        "component": "binary_to_oh",
        "lattice_class": "OHLattice",
        "lattice": LATTICE_D2Q9,
        "mode": "probe",
    },
    # reflection permutations, uncontrolled and controlled
    {
        "id": "ab_bbperm_ab",
        "component": "bb_permutation",
        "n": 4,
        "encoding": "AB",
        "ctrl": 0,
        "mode": "unitary",
    },
    {
        "id": "ab_bbperm_ab_c1",
        "component": "bb_permutation",
        "n": 4,
        "encoding": "AB",
        "ctrl": 1,
        "mode": "probe",
    },
    {
        "id": "ab_bbperm_ab_c4",
        "component": "bb_permutation",
        "n": 4,
        "encoding": "AB",
        "ctrl": 4,
        "mode": "probe",
    },
    {
        "id": "ab_bbperm_oh",
        "component": "bb_permutation",
        "n": 9,
        "encoding": "OH",
        "ctrl": 0,
        "mode": "probe",
    },
    {
        "id": "ab_bbperm_oh_c1",
        "component": "bb_permutation",
        "n": 9,
        "encoding": "OH",
        "ctrl": 1,
        "mode": "probe",
    },
    {
        "id": "ab_srperm_ab_x",
        "component": "sr_permutation",
        "n": 4,
        "encoding": "AB",
        "reflect": [True, False],
        "ctrl": 0,
        "mode": "unitary",
    },
    {
        "id": "ab_srperm_ab_y",
        "component": "sr_permutation",
        "n": 4,
        "encoding": "AB",
        "reflect": [False, True],
        "ctrl": 0,
        "mode": "unitary",
    },
    {
        "id": "ab_srperm_ab_xy",
        "component": "sr_permutation",
        "n": 4,
        "encoding": "AB",
        "reflect": [True, True],
        "ctrl": 0,
        "mode": "unitary",
    },
    {
        "id": "ab_srperm_ab_x_c1",
        "component": "sr_permutation",
        "n": 4,
        "encoding": "AB",
        "reflect": [True, False],
        "ctrl": 1,
        "mode": "probe",
    },
    {
        "id": "ab_srperm_ab_xy_c4",
        "component": "sr_permutation",
        "n": 4,
        "encoding": "AB",
        "reflect": [True, True],
        "ctrl": 4,
        "mode": "probe",
    },
    {
        "id": "ab_srperm_oh_y",
        "component": "sr_permutation",
        "n": 9,
        "encoding": "OH",
        "reflect": [False, True],
        "ctrl": 0,
        "mode": "probe",
    },
    # streaming
    {
        "id": "ab_stream_d1q3",
        "component": "streaming",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D1Q3,
        "controls": [],
        "mode": "unitary",
    },
    {
        "id": "ab_stream_d1q3_ctrl",
        "component": "streaming",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D1Q3,
        # The obstacle ancilla: streaming under a boundary-condition control.
        "controls": [5],
        "mode": "unitary",
    },
    {
        "id": "ab_stream_d2q9",
        "component": "streaming",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "controls": [],
        "mode": "probe",
    },
    {
        "id": "ab_stream_d2q9_ctrl",
        "component": "streaming",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "controls": [8],
        "mode": "probe",
    },
    # averaged collision (D1Q3 only)
    {
        "id": "ab_abecoll_d1q3",
        "component": "averaged_collision",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D1Q3,
        "mode": "unitary",
    },
    # initial conditions
    {
        "id": "ab_init_d1q3_v01",
        "component": "discrete_uniform_initial",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D1Q3,
        "velocities": [0, 1],
        "grid_superpose": [[]],
        "mode": "unitary",
    },
    {
        "id": "ab_init_d2q9_v012",
        "component": "discrete_uniform_initial",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "velocities": [0, 1, 2],
        "grid_superpose": [[], []],
        "mode": "probe",
    },
    {
        "id": "ab_init_d2q9_v4507_g",
        "component": "discrete_uniform_initial",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "velocities": [4, 5, 0, 7],
        "grid_superpose": [[0], [1]],
        "mode": "probe",
    },
    {
        "id": "ab_init_plain_d2q9",
        "component": "initial",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "mode": "probe",
    },
    {
        "id": "ab_init_parallel_d2q9",
        "component": "parallel_initial",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "marker_qubits": 1,
        "velocities": [[0, 1], [0, 3]],
        "grid_superpose": [[[0], []], [[], [0]]],
        "mode": "probe",
    },
    # measurement
    {
        "id": "ab_meas_d2q9",
        "component": "grid_measurement",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "measure_velocity": False,
        "mode": "measure",
    },
    {
        "id": "ab_meas_d2q9_v",
        "component": "grid_measurement",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9,
        "measure_velocity": True,
        "mode": "measure",
    },
    # zone-agnostic oracle
    {
        "id": "ab_zaoracle_block",
        "component": "za_oracle",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_BB,
        "shape_key": "bounceback",
        "target_obstacle_index": 0,
        "mode": "probe",
    },
    {
        "id": "ab_zaoracle_ymonomial",
        "component": "za_oracle",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_YMONOMIAL,
        "shape_key": "bounceback",
        "target_obstacle_index": 0,
        "mode": "probe",
    },
    # zone-agnostic SR check
    {
        "id": "ab_srcheck_neg",
        "component": "za_sr_check",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_SR,
        "check_negative_direction": True,
        "mode": "probe",
    },
    {
        "id": "ab_srcheck_pos",
        "component": "za_sr_check",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_SR,
        "check_negative_direction": False,
        "mode": "probe",
    },
    # standard reflection
    {
        "id": "ab_stdrefl_bb",
        "component": "standard_reflection",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_BB,
        "mode": "probe",
    },
    {
        "id": "ab_stdrefl_sr",
        "component": "standard_reflection",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_SR,
        "mode": "probe",
    },
    # zone-agnostic reflection
    {
        "id": "ab_agnrefl_bb",
        "component": "agnostic_reflection",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_BB,
        "mode": "probe",
    },
    {
        "id": "ab_agnrefl_sr",
        "component": "agnostic_reflection",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_SR,
        "mode": "probe",
    },
    # end-to-end
    {
        "id": "ab_abqlbm_bb",
        "component": "abqlbm",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_BB,
        "use_agnostic_bcs": False,
        "mode": "probe",
    },
    {
        "id": "ab_abqlbm_bb_agnostic",
        "component": "abqlbm",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_BB,
        "use_agnostic_bcs": True,
        "mode": "probe",
    },
    {
        "id": "ab_cqlbm_bb",
        "component": "cqlbm",
        "lattice_class": "ABLattice",
        "lattice": LATTICE_D2Q9_BB,
        "use_agnostic_bcs": False,
        "mode": "probe",
    },
]
