"""Write ``lattice_layouts.json``: register layouts, qubit counts, ``num_*`` values
and index-helper outputs per lattice, with the lattice spec embedded.

Run with the qiskit-based qlbm at ``fixture_store.REFERENCE_COMMIT`` installed; the
README's Testing section has the command.
"""

import json
from pathlib import Path

from fixture_store import save_text

RESOURCE_DIR = Path(__file__).parent.parent / "resources"

ATTRIBUTES = [
    "num_dims",
    "num_total_qubits",
    "num_grid_qubits",
    "num_velocity_qubits",
    "num_ancilla_qubits",
    "num_base_qubits",
    "num_marker_qubits",
    "num_accumulation_qubits",
    "num_obstacle_qubits",
    "num_comparator_qubits",
    "num_copy_qubits",
    "num_monomial_qubits",
    "num_velocities_per_point",
    "num_timesteps",
]


def _res(name):
    """Load a lattice config from ``test/resources``.

    Parameters
    ----------
    name : str
        The resource file name.

    Returns
    -------
    dict
        The parsed lattice configuration.
    """
    return json.loads((RESOURCE_DIR / name).read_text())


def _cuboid(boundary, *bounds):
    """Build a cuboid geometry dict with per-dimension ``bounds``.

    Parameters
    ----------
    boundary : str
        The boundary condition, ``"specular"`` or ``"bounceback"``.
    *bounds : list[int]
        One ``[lo, hi]`` pair per dimension, in x, y, z order.

    Returns
    -------
    dict
        The geometry entry.
    """
    entry = {"shape": "cuboid"}
    for letter, pair in zip("xyz", bounds, strict=False):
        entry[letter] = list(pair)
    entry["boundary"] = boundary
    return entry


def build_cases():
    """Return the list of lattice case specs (embedded into the fixture).

    Returns
    -------
    list[dict]
        One spec per case: id, class name, lattice_data, constructor
        kwargs, and post-construction mutation calls.
    """
    return [
        # MSLattice (CFL multi-speed encoding)
        {
            "id": "ms_2d_8x8_specular",
            "class": "MSLattice",
            "lattice_data": _res("symmetric_2d_1_obstacle.json"),
        },
        {
            # v=2 per dim -> zero-size velocity magnitude registers
            "id": "ms_2d_4x8_q2_specular",
            "class": "MSLattice",
            "lattice_data": _res("symmetric_2d_1_obstacle_q4.json"),
        },
        {
            "id": "ms_2d_16x16_no_obstacles",
            "class": "MSLattice",
            "lattice_data": _res("symmetric_2d_no_obstacles.json"),
        },
        {
            "id": "ms_3d_8x16x8_no_obstacles",
            "class": "MSLattice",
            "lattice_data": _res("asymmetric_3d_no_obstacles.json"),
        },
        {
            "id": "ms_2d_16x64_asym_velocities",
            "class": "MSLattice",
            "lattice_data": {
                "lattice": {
                    "dim": {"x": 16, "y": 64},
                    "velocities": {"x": 16, "y": 4},
                },
                "geometry": [_cuboid("specular", [4, 6], [3, 12])],
            },
        },
        {
            # bounceback-only -> single obstacle ancilla (adaptable register)
            "id": "ms_2d_16x16_bounceback_only",
            "class": "MSLattice",
            "lattice_data": {
                "lattice": {
                    "dim": {"x": 16, "y": 16},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [_cuboid("bounceback", [4, 6], [3, 12])],
            },
        },
        {
            "id": "ms_3d_8x8x8_mixed",
            "class": "MSLattice",
            "lattice_data": {
                "lattice": {
                    "dim": {"x": 8, "y": 8, "z": 8},
                    "velocities": {"x": 4, "y": 4, "z": 4},
                },
                "geometry": [
                    _cuboid("specular", [1, 2], [1, 2], [1, 2]),
                    _cuboid("bounceback", [5, 6], [5, 6], [5, 6]),
                ],
            },
        },
        # ABLattice (fully compressed amplitude encoding)
        {
            "id": "ab_1d_16_d1q2_cuboid_bb",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16}, "velocities": "D1Q2"},
                "geometry": [_cuboid("bounceback", [4, 6])],
            },
        },
        {
            "id": "ab_1d_256_d1q3_nogeom",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 256}, "velocities": "D1Q3"},
            },
        },
        {
            "id": "ab_2d_16x16_d2q4_cuboid_bb",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [_cuboid("bounceback", [2, 6], [5, 10])],
            },
        },
        {
            # specular -> d+2 obstacle qubits
            "id": "ab_2d_16x16_d2q4_cuboid_specular",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [_cuboid("specular", [2, 6], [5, 10])],
            },
        },
        {
            "id": "ab_2d_16x16_d2q9_nogeom",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q9"},
                "geometry": [],
            },
        },
        {
            # ymonomial -> copy + monomial registers
            "id": "ab_2d_16x16_ymonomial",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [
                    {
                        "shape": "ymonomial",
                        "exponent": 2,
                        "comparator": "<=",
                        "boundary": "bounceback",
                    }
                ],
            },
        },
        {
            "id": "ab_3d_8x8x8_d3q6_cuboid_bb",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 8, "y": 8, "z": 8}, "velocities": "D3Q6"},
                "geometry": [_cuboid("bounceback", [1, 2], [1, 2], [1, 2])],
            },
        },
        {
            # marker register via set_geometries
            "id": "ab_2d_16x16_multi_geometry",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [_cuboid("bounceback", [2, 6], [5, 10])],
            },
            "mutations": [
                {
                    "method": "set_geometries",
                    "args": [
                        [
                            [_cuboid("bounceback", [2, 6], [5, 10])],
                            [_cuboid("bounceback", [1, 3], [1, 3])],
                            [_cuboid("specular", [8, 12], [8, 12])],
                        ]
                    ],
                }
            ],
        },
        {
            # accumulation register
            "id": "ab_2d_16x16_accumulation",
            "class": "ABLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [_cuboid("bounceback", [2, 6], [5, 10])],
            },
            "mutations": [{"method": "use_accumulation_register", "args": []}],
        },
        # OHLattice (one-hot velocity encoding)
        {
            "id": "oh_1d_256_d1q3_nogeom",
            "class": "OHLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 256}, "velocities": "D1Q3"},
            },
        },
        {
            "id": "oh_2d_8x8_d2q9_nogeom",
            "class": "OHLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "D2Q9"},
                "geometry": [],
            },
        },
        {
            "id": "oh_2d_16x16_d2q4_cuboid_bb",
            "class": "OHLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [_cuboid("bounceback", [2, 6], [5, 10])],
            },
        },
        {
            "id": "oh_3d_8x8x8_d3q6_cuboid_bb",
            "class": "OHLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 8, "y": 8, "z": 8}, "velocities": "D3Q6"},
                "geometry": [_cuboid("bounceback", [1, 2], [1, 2], [1, 2])],
            },
        },
        # SpaceTimeLattice
        {
            "id": "st_1d_16_t2_d1q2",
            "class": "SpaceTimeLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16}, "velocities": "D1Q2"},
                "geometry": [],
            },
            "kwargs": {"num_timesteps": 2},
        },
        {
            "id": "st_1d_16_t3_d1q2_meas_vol",
            "class": "SpaceTimeLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16}, "velocities": "D1Q2"},
                "geometry": [],
            },
            "kwargs": {
                "num_timesteps": 3,
                "include_measurement_qubit": True,
                "use_volumetric_ops": True,
            },
        },
        {
            "id": "st_2d_16x16_t1_d2q4",
            "class": "SpaceTimeLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [],
            },
            "kwargs": {"num_timesteps": 1},
        },
        {
            "id": "st_2d_16x16_t2_d2q4",
            "class": "SpaceTimeLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
                "geometry": [],
            },
            "kwargs": {"num_timesteps": 2},
        },
        {
            "id": "st_2d_4x8_t2_d2q4_meas_vol",
            "class": "SpaceTimeLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
                "geometry": [],
            },
            "kwargs": {
                "num_timesteps": 2,
                "include_measurement_qubit": True,
                "use_volumetric_ops": True,
            },
        },
        {
            "id": "st_3d_4x4x4_t1_d3q6",
            "class": "SpaceTimeLattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 4, "y": 4, "z": 4}, "velocities": "D3Q6"},
                "geometry": [],
            },
            "kwargs": {"num_timesteps": 1},
        },
        # LQLGALattice
        {
            "id": "lqlga_1d_8_d1q2",
            "class": "LQLGALattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 8}, "velocities": "D1Q2"},
            },
        },
        {
            "id": "lqlga_1d_8_d1q3",
            "class": "LQLGALattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 8}, "velocities": "D1Q3"},
            },
        },
        {
            "id": "lqlga_2d_4x4_d2q4",
            "class": "LQLGALattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
            },
        },
        {
            "id": "lqlga_2d_2x2_d2q9",
            "class": "LQLGALattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 2, "y": 2}, "velocities": "D2Q9"},
            },
        },
        {
            "id": "lqlga_3d_2x2x2_d3q6",
            "class": "LQLGALattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 2, "y": 2, "z": 2}, "velocities": "D3Q6"},
            },
        },
        {
            "id": "lqlga_1d_8_d1q2_multi_geometry",
            "class": "LQLGALattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 8}, "velocities": "D1Q2"},
            },
            "mutations": [
                {
                    "method": "set_geometries",
                    "args": [
                        [
                            [_cuboid("bounceback", [3, 4])],
                            [_cuboid("specular", [1, 2])],
                            [_cuboid("specular", [1, 4])],
                        ]
                    ],
                }
            ],
        },
        {
            "id": "lqlga_1d_8_d1q2_accumulation",
            "class": "LQLGALattice",
            "lattice_data": {
                "lattice": {"dim": {"x": 8}, "velocities": "D1Q2"},
            },
            "mutations": [
                {"method": "use_accumulation_register", "args": [3, [0, 1, 2]]}
            ],
        },
    ]


def build_lattice(spec, module):
    """Instantiate a lattice from a case spec and apply its mutations.

    Parameters
    ----------
    spec : dict
        The case spec (class, lattice_data, kwargs, mutations).
    module : module
        The ``qlbm.lattice`` module to draw lattice classes from.

    Returns
    -------
    Lattice
        The constructed (and mutated) lattice.
    """
    cls = getattr(module, spec["class"])
    lattice = cls(lattice_data=spec["lattice_data"], **spec.get("kwargs", {}))
    for mutation in spec.get("mutations", []):
        getattr(lattice, mutation["method"])(*mutation["args"])
    return lattice


def _neighbor_sample(num_neighbors):
    """Deterministic subset of neighborhood indices for large stencils.

    Parameters
    ----------
    num_neighbors : int
        The total number of neighborhood points.

    Returns
    -------
    list[int]
        All indices when small, otherwise edges plus the midpoint.
    """
    if num_neighbors <= 16:
        return list(range(num_neighbors))
    return sorted(
        {0, 1, 2, 3, num_neighbors // 2, num_neighbors - 2, num_neighbors - 1}
    )


def helper_calls(lattice, cls_name):
    """Enumerate every valid public index-helper call for ``lattice``.

    The enumeration only produces non-raising calls; the differential test
    replays the recorded (helper, args) pairs verbatim.

    Parameters
    ----------
    lattice : Lattice
        The lattice to enumerate helpers for.
    cls_name : str
        The lattice class name (drives which helpers exist).

    Returns
    -------
    list[tuple[str, tuple]]
        The (helper name, args) pairs to record.
    """
    dims = lattice.num_dims
    calls = []
    if cls_name == "MSLattice":
        for helper in [
            "ancillae_velocity_index",
            "grid_index",
            "velocity_index",
            "velocity_dir_index",
        ]:
            calls.append((helper, (None,)))
            calls.extend((helper, (dim,)) for dim in range(dims))
        calls.append(("ancillae_obstacle_index", (None,)))
        calls.extend(
            ("ancillae_obstacle_index", (i,))
            for i in range(lattice.num_obstacle_qubits)
        )
        calls.append(("ancillae_comparator_index", (None,)))
        calls.extend(("ancillae_comparator_index", (i,)) for i in range(dims - 1))
    elif cls_name in ("ABLattice", "OHLattice"):
        calls.append(("grid_index", (None,)))
        calls.extend(("grid_index", (dim,)) for dim in range(dims))
        calls.append(("velocity_index", (None,)))
        calls.append(("ancillae_comparator_index", (None,)))
        if lattice.num_comparator_qubits > 0:
            calls.append(("ancillae_comparator_index", (0,)))
        calls.append(("ancillae_obstacle_index", (None,)))
        calls.extend(
            ("ancillae_obstacle_index", (i,))
            for i in range(lattice.num_obstacle_qubits)
        )
        if lattice.num_copy_qubits > 0:
            calls.append(("ancillae_copy_index", ()))
        if lattice.num_monomial_qubits > 0:
            calls.append(("ancillae_monomial_index", ()))
        if cls_name == "ABLattice":
            calls.append(("marker_index", ()))
            calls.append(("accumulation_index", ()))
    elif cls_name == "SpaceTimeLattice":
        calls.append(("grid_index", (None,)))
        calls.extend(("grid_index", (dim,)) for dim in range(dims))
        velocities = lattice.num_velocities_per_point
        num_neighbors = lattice.properties.get_num_velocity_qubits() // velocities
        for neighbor in _neighbor_sample(num_neighbors):
            calls.append(("velocity_index", (neighbor, None)))
            calls.extend(("velocity_index", (neighbor, v)) for v in range(velocities))
        if lattice.include_measurement_qubit:
            calls.append(("ancilla_mass_index", ()))
        if lattice.use_volumetric_ops:
            calls.append(("ancilla_comparator_index", (None,)))
            calls.extend(("ancilla_comparator_index", (i,)) for i in range(dims))
    elif cls_name == "LQLGALattice":
        from itertools import product

        velocities = lattice.num_velocities_per_point
        gridpoint_tuples = list(
            product(*[range(ng + 1) for ng in lattice.num_gridpoints])
        )
        for flat, gp_tuple in enumerate(gridpoint_tuples):
            calls.append(("gridpoint_index_tuple", (gp_tuple,)))
            calls.append(("gridpoint_index_flat", (flat,)))
            calls.extend(("velocity_index_flat", (flat, v)) for v in range(velocities))
            calls.extend(
                ("velocity_index_tuple", (gp_tuple, v)) for v in range(velocities)
            )
        calls.append(("marker_index", ()))
        calls.append(("accumulation_index", ()))
        calls.extend(
            ("get_velocity_qubits_of_line", (line,))
            for line in range(velocities // 2 + 1)
        )
    else:
        raise ValueError(f"Unknown lattice class: {cls_name}")
    return calls


def _norm(value):
    """JSON-normalize a helper result (tuples become lists).

    Parameters
    ----------
    value : Any
        The helper return value.

    Returns
    -------
    Any
        The JSON-compatible equivalent.
    """
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value


def dump_layout(lattice):
    """Extract the register layout of a qiskit-based lattice.

    Global qubit indices are taken from the blueprint circuit itself via
    ``circuit.find_bit`` — the layout authority, not an assumption about
    declaration-order offsets.

    Parameters
    ----------
    lattice : Lattice
        The qiskit-based lattice.

    Returns
    -------
    tuple[list, int]
        The per-register ``[name, size, qubit indices]`` list and the
        blueprint circuit's total qubit count.
    """
    registers = [
        [
            register.name,
            register.size,
            [lattice.circuit.find_bit(qubit).index for qubit in register],
        ]
        for register in lattice.registers
    ]
    return registers, lattice.circuit.num_qubits


def main():
    """Write the layout fixture for every case."""
    import qlbm.lattice as lattice_module

    entries = []
    for spec in build_cases():
        lattice = build_lattice(spec, lattice_module)
        registers, total_qubits = dump_layout(lattice)
        attributes = {
            name: getattr(lattice, name, None)
            for name in ATTRIBUTES
            if getattr(lattice, name, None) is not None
        }
        recorded_calls = [
            {
                "helper": helper,
                "args": _norm(list(args)),
                "result": _norm(getattr(lattice, helper)(*args)),
            }
            for helper, args in helper_calls(lattice, spec["class"])
        ]
        entries.append(
            {
                "spec": spec,
                "registers": registers,
                "total_qubits": total_qubits,
                "attributes": attributes,
                "calls": recorded_calls,
            }
        )
        print(
            f"{spec['id']}: {len(registers)} registers, "
            f"{total_qubits} qubits, {len(recorded_calls)} helper calls"
        )

    path = save_text("lattice_layouts.json", json.dumps(entries, indent=1))
    print(f"Wrote {len(entries)} cases to {path}")


if __name__ == "__main__":
    main()
