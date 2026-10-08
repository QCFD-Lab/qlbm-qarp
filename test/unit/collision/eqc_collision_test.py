"""Shot-based tests for the equivalence-class collision operator."""

from typing import Dict, List

import pytest
import qarpx as qx
from qarp.endianness import bits_to_label

from qlbm.components.spacetime.collision.eqc_collision import (
    EQCCollisionOperator,
)
from qlbm.lattice.eqc.eqc_generator import (
    EquivalenceClassGenerator,
)
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.utils import flatten
from test.builders import CircuitBuilder

# Fixed so the shot-based assertions below are deterministic.
SIMULATION_SEED = 42


def int_to_bool_list(num, num_bits):
    """Bool list of the ``num_bits``-wide binary expansion of ``num``."""
    return bit_string_to_bool_list(format(num, f"0{num_bits}b"))


def bit_string_to_bool_list(bitstring):
    """Bool list of a ``"0"``/``"1"`` string, one entry per character."""
    return [x == "1" for x in bitstring]


@pytest.fixture
def d2q4_equivalence_class_bitstrings() -> List[List[str]]:
    """Velocity configurations of every D2Q4 equivalence class."""
    return [
        eqc.get_bitstrings()
        for eqc in EquivalenceClassGenerator(
            LatticeDiscretization.D2Q4
        ).generate_equivalence_classes()
    ]


@pytest.fixture
def d3q6_equivalence_class_bitstrings() -> List[List[str]]:
    """Velocity configurations of every D3Q6 equivalence class."""
    return sorted(
        [
            eqc.get_bitstrings()
            for eqc in EquivalenceClassGenerator(
                LatticeDiscretization.D3Q6
            ).generate_equivalence_classes()
        ],
        key=lambda x: x[0],
    )  # Sort by first element


def simulate_counts(
    input_state: str,
    collision_circuit: "qx.Block",
    num_shots: int = 128,
) -> Dict[int, int]:
    """Sample the collision operator applied to a computational-basis input.

    Parameters
    ----------
    input_state : str
        Velocity configuration, character ``c`` giving the state of qubit ``c``.
    collision_circuit : qx.Block
        The collision operator to apply.
    num_shots : int
        Number of shots to sample.

    Returns
    -------
    Dict[int, int]
        ``SamplingResult`` counts, keyed by LSB integer basis-state label.
    """
    builder = CircuitBuilder(collision_circuit.n_qubits, name="eqc_collision_test")
    for c, b in enumerate(bit_string_to_bool_list(input_state)):
        if b:
            builder.x(c)
    builder.compose(collision_circuit)
    circuit = builder.build()

    return dict(
        qx.QarpSimulator()
        .run(
            circuit.flatten(),
            circuit.n_qubits,
            num_shots,
            seed=SIMULATION_SEED,
        )
        .counts
    )


def verify_simulation_outcome(
    input_state: str,
    expected_outcomes: List[str],
    collision_circuit: "qx.Block",
    num_shots=128,
    verify_negative_cases: bool = True,
):
    """Assert the sampled support is exactly ``expected_outcomes``.

    Parameters
    ----------
    input_state : str
        Velocity configuration prepared before the collision operator.
    expected_outcomes : List[str]
        Velocity configurations that must all be observed.
    collision_circuit : qx.Block
        The collision operator under test.
    num_shots : int
        Number of shots to sample.
    verify_negative_cases : bool
        Also assert that nothing outside ``expected_outcomes`` is observed.
    """
    counts = simulate_counts(input_state, collision_circuit, num_shots)
    expected_labels = [bits_to_label(b) for b in expected_outcomes]

    assert all(label in counts for label in expected_labels), (
        f"{expected_outcomes} not covered by counts {counts}"
    )

    if verify_negative_cases:
        all_bitstrings = [
            format(i, f"0{collision_circuit.n_qubits}b")
            for i in range(2**collision_circuit.n_qubits)
        ]
        assert all(
            bits_to_label(b) not in counts
            for b in all_bitstrings
            if b not in expected_outcomes
        )


@pytest.mark.parametrize(
    "equivalence_class_index",
    list(
        range(
            len(
                EquivalenceClassGenerator(
                    LatticeDiscretization.D2Q4
                ).generate_equivalence_classes()
            )
        )
    ),
)
def test_d2q4_collision_positive_cases(
    d2q4_equivalence_class_bitstrings, equivalence_class_index
):
    """Each D2Q4 class input collides onto exactly that class's configurations."""
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
            "geometry": [],
        },
    )

    local_circuit = EQCCollisionOperator(lattice.properties.get_discretization())
    assert local_circuit.n_qubits == 4
    eqc = d2q4_equivalence_class_bitstrings[equivalence_class_index]
    for velocity_cfg in eqc:
        verify_simulation_outcome(
            velocity_cfg, eqc, local_circuit, verify_negative_cases=True
        )


def test_d2q4_collision_negative_cases(d2q4_equivalence_class_bitstrings):
    """Configurations outside every D2Q4 class are left untouched."""
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
            "geometry": [],
        },
    )

    local_circuit = EQCCollisionOperator(lattice.properties.get_discretization())
    assert local_circuit.n_qubits == 4

    for b in [
        format(i, f"0{local_circuit.n_qubits}b")
        for i in range(2**local_circuit.n_qubits)
        if format(i, f"0{local_circuit.n_qubits}b")
        not in flatten(d2q4_equivalence_class_bitstrings)
    ]:
        verify_simulation_outcome(
            b,
            [b],
            local_circuit,
            verify_negative_cases=True,
        )


@pytest.mark.parametrize(
    "equivalence_class_index",
    list(
        range(
            len(
                EquivalenceClassGenerator(
                    LatticeDiscretization.D3Q6
                ).generate_equivalence_classes()
            )
        )
    ),
)
def test_d3q6_collision_positive_cases(
    d3q6_equivalence_class_bitstrings, equivalence_class_index
):
    """Each D3Q6 class input collides onto exactly that class's configurations."""
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 2, "y": 2, "z": 2},
                "velocities": "D3Q6",
            },
            "geometry": [],
        },
    )

    local_circuit = EQCCollisionOperator(lattice.properties.get_discretization())
    assert local_circuit.n_qubits == 6
    eqc = d3q6_equivalence_class_bitstrings[equivalence_class_index]
    for velocity_cfg in eqc:
        verify_simulation_outcome(
            velocity_cfg, eqc, local_circuit, verify_negative_cases=True
        )
