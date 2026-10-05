"""Integration tests for :class:`SimulationConfig` on the MS family.

Validation of the config attributes, and the guarantee that preparation
lowers the four algorithmic components into built qarp blocks.
"""

from itertools import product

import numpy as np
import pytest
import qarpx as qx
from qarp import EXACT

from qlbm.components.common import EmptyPrimitive
from qlbm.components.ms import (
    MSQLBM,
    GridMeasurement,
    MSInitialConditions,
)
from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.runner.simulation_config import SimulationConfig
from qlbm.lattice import MSLattice
from qlbm.tools.exceptions import ExecutionException

CIRCUIT_ATTRIBUTES = [
    "initial_conditions",
    "algorithm",
    "postprocessing",
    "measurement",
]


def commands(circuit) -> list:
    """The command stream of a config attribute, checked to be a built block."""
    assert isinstance(circuit, qx.Block)
    return list(circuit.flatten())


def circuits_for(lattice: MSLattice):
    """The four algorithmic components of the MS algorithm on ``lattice``."""
    return {
        "initial_conditions": MSInitialConditions(lattice),
        "algorithm": MSQLBM(lattice),
        "postprocessing": EmptyPrimitive(lattice),
        "measurement": GridMeasurement(lattice),
    }


@pytest.fixture
def symmetric_2d_no_osbtacle_circuits():
    """The four MS components on the obstacle-free 16x16 lattice."""
    return circuits_for(MSLattice("test/resources/symmetric_2d_no_obstacles.json"))


@pytest.fixture
def symmetric_2d_one_osbtacle_circuits():
    """The four MS components on the 8x8 lattice with one specular obstacle."""
    return circuits_for(MSLattice("test/resources/symmetric_2d_1_obstacle.json"))


CIRCUIT_FIXTURES = [
    "symmetric_2d_no_osbtacle_circuits",
    "symmetric_2d_one_osbtacle_circuits",
]


@pytest.mark.parametrize(
    "circuits,optimization_level,shots",
    list(product(CIRCUIT_FIXTURES, [0, 1, 2], [1, 1024, EXACT])),
)
def test_simulation_configuration_validation(
    circuits, optimization_level, shots, request
):
    """Every supported (optimization level, shot budget) combination validates."""
    cfg = SimulationConfig(
        **request.getfixturevalue(circuits),
        optimization_level=optimization_level,
        shots=shots,
    )

    cfg.validate()


@pytest.mark.parametrize("optimization_level", [-1, 3, 100])
def test_unsupported_optimization_level_is_rejected(
    symmetric_2d_no_osbtacle_circuits, optimization_level
):
    """Validation rejects optimization levels the compiler does not implement."""
    cfg = SimulationConfig(
        **symmetric_2d_no_osbtacle_circuits,
        optimization_level=optimization_level,
    )

    with pytest.raises(ExecutionException) as excinfo:
        cfg.validate()

    assert "optimization level" in str(excinfo.value)


@pytest.mark.parametrize("shots", [0, -1, 2.5, "many", None, True, np.float64(32.0)])
def test_unsupported_shot_budget_is_rejected(symmetric_2d_no_osbtacle_circuits, shots):
    """Validation rejects shot budgets that are neither positive whole numbers nor EXACT."""
    cfg = SimulationConfig(**symmetric_2d_no_osbtacle_circuits, shots=shots)

    with pytest.raises(ExecutionException) as excinfo:
        cfg.validate()

    assert "shot budget" in str(excinfo.value)


@pytest.mark.parametrize("shots", [np.int64(32), np.uint8(7), np.int32(1)])
def test_numpy_integer_shot_budget_validates(symmetric_2d_no_osbtacle_circuits, shots):
    """Validation accepts a positive whole number of any integral type."""
    cfg = SimulationConfig(**symmetric_2d_no_osbtacle_circuits, shots=shots)

    cfg.validate()


@pytest.mark.parametrize("attribute", CIRCUIT_ATTRIBUTES)
def test_non_circuit_attribute_is_rejected(
    symmetric_2d_no_osbtacle_circuits, attribute
):
    """Validation rejects any algorithmic attribute that is not a circuit."""
    circuits = dict(symmetric_2d_no_osbtacle_circuits)
    circuits[attribute] = "not a circuit"
    cfg = SimulationConfig(**circuits)

    with pytest.raises(ExecutionException) as excinfo:
        cfg.validate()

    assert "not in supported types" in str(excinfo.value)


@pytest.mark.parametrize("attribute", ["algorithm", "postprocessing", "measurement"])
def test_statevector_is_only_accepted_as_initial_conditions(
    symmetric_2d_no_osbtacle_circuits, attribute
):
    """A statevector is a valid initial condition but never a valid circuit."""
    lattice_circuits = symmetric_2d_no_osbtacle_circuits
    state = np.zeros(2 ** lattice_circuits["algorithm"].lattice.n_qubits, dtype=complex)
    state[0] = 1.0

    circuits = dict(lattice_circuits)
    circuits[attribute] = state

    with pytest.raises(ExecutionException):
        SimulationConfig(**circuits).validate()

    circuits = dict(lattice_circuits)
    circuits["initial_conditions"] = state
    SimulationConfig(**circuits).validate()


@pytest.mark.parametrize(
    "circuits,optimization_level",
    list(product(CIRCUIT_FIXTURES, [0, 1, 2])),
)
def test_simulation_configuration_preparation(circuits, optimization_level, request):
    """Preparation lowers all four algorithmic components into built qarp blocks."""
    cfg = SimulationConfig(
        **request.getfixturevalue(circuits),
        optimization_level=optimization_level,
    )

    cfg.validate()
    cfg.prepare_for_simulation()

    for attribute in CIRCUIT_ATTRIBUTES:
        block = getattr(cfg, attribute)
        assert isinstance(block, qx.Block)
        assert block.n_qubits > 0


@pytest.mark.parametrize("optimization_level", [1, 2])
def test_preparation_never_optimizes_the_measurement_block(
    symmetric_2d_one_osbtacle_circuits, optimization_level
):
    """Measurement is compiled at level 0 regardless of the config's level.

    Peephole optimization is defined over unitaries, and the qubit -> cbit
    mapping is the result contract, so the measurement stream must survive
    preparation verbatim.
    """
    lattice = symmetric_2d_one_osbtacle_circuits["algorithm"].lattice
    reference = CircuitCompiler().compile(GridMeasurement(lattice), 0)

    cfg = SimulationConfig(
        **symmetric_2d_one_osbtacle_circuits,
        optimization_level=optimization_level,
    )
    cfg.prepare_for_simulation()

    prepared = [
        (command.qubits, command.cbits) for command in commands(cfg.measurement)
    ]
    expected = [(command.qubits, command.cbits) for command in reference.flatten()]
    assert prepared == expected


@pytest.mark.parametrize("optimization_level", [1, 2])
def test_preparation_preserves_the_simulated_state(
    symmetric_2d_one_osbtacle_circuits, optimization_level
):
    """Optimized preparation produces the same state as the unoptimized one."""
    lattice = symmetric_2d_one_osbtacle_circuits["algorithm"].lattice

    plain = SimulationConfig(**circuits_for(lattice), optimization_level=0)
    plain.prepare_for_simulation()
    optimized = SimulationConfig(
        **circuits_for(lattice), optimization_level=optimization_level
    )
    optimized.prepare_for_simulation()

    simulator = qx.QarpSimulator()
    np.testing.assert_allclose(
        np.asarray(
            simulator.statevector(
                commands(plain.initial_conditions) + commands(plain.algorithm),
                lattice.n_qubits,
            )
        ),
        np.asarray(
            simulator.statevector(
                commands(optimized.initial_conditions) + commands(optimized.algorithm),
                lattice.n_qubits,
            )
        ),
        atol=1e-9,
    )


def test_statevector_initial_conditions_survive_preparation(
    symmetric_2d_no_osbtacle_circuits,
):
    """A statevector initial condition is handed to the runner untouched."""
    lattice = symmetric_2d_no_osbtacle_circuits["algorithm"].lattice
    state = np.zeros(2**lattice.n_qubits, dtype=complex)
    state[0] = 1.0

    circuits = dict(symmetric_2d_no_osbtacle_circuits)
    circuits["initial_conditions"] = state
    cfg = SimulationConfig(**circuits)

    cfg.validate()
    cfg.prepare_for_simulation()

    assert cfg.initial_conditions is state
    for attribute in ["algorithm", "postprocessing", "measurement"]:
        assert isinstance(getattr(cfg, attribute), qx.Block)


def test_execution_compiler_carries_the_config_logger(
    symmetric_2d_no_osbtacle_circuits,
):
    """The compiler the config hands out logs through the config's logger."""
    cfg = SimulationConfig(**symmetric_2d_no_osbtacle_circuits)

    compiler = cfg.get_execution_compiler()

    assert isinstance(compiler, CircuitCompiler)
    assert compiler.logger is cfg.logger
