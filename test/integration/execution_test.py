"""End-to-end execution of the MS and Space-Time algorithms through the runner.

The tests parametrize on ``statevector_snapshots``, which picks between the
O(N) carried-state loop and the O(N^2) re-run-from-zero loop.
"""

import os

import numpy as np
import pytest
from qarp import EXACT

from qlbm.components.common import EmptyPrimitive
from qlbm.components.ms import (
    MSQLBM,
    GridMeasurement,
    MSInitialConditions,
)
from qlbm.components.spacetime import (
    SpaceTimeGridVelocityMeasurement,
    SpaceTimeQLBM,
)
from qlbm.components.spacetime.initial.pointwise import (
    PointWiseSpaceTimeInitialConditions,
)
from qlbm.infra.result.base import QBMResult
from qlbm.infra.runner import QarpRunner
from qlbm.infra.runner.simulation_config import SimulationConfig
from qlbm.lattice import MSLattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.exceptions import ExecutionException

NUM_STEPS = 2


@pytest.fixture
def collisionless_circuits():
    """The four MS components on a 2D 8x8 lattice with one specular obstacle."""
    lattice = MSLattice("test/resources/symmetric_2d_1_obstacle.json")

    return {
        "initial_conditions": MSInitialConditions(lattice),
        "algorithm": MSQLBM(lattice),
        "postprocessing": EmptyPrimitive(lattice),
        "measurement": GridMeasurement(lattice),
        "lattice": lattice,
    }


@pytest.fixture
def spacetime_circuits():
    """The four Space-Time components on a 1D D1Q2 lattice with a bounceback obstacle."""
    # D2Q4 needs >= 22 qubits for its neighbourhood registers, so the case is
    # D1Q2 with a bounceback obstacle: 9 qubits, and it exercises the
    # reflection path.
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {"dim": {"x": 8}, "velocities": "D1Q2"},
            "geometry": [{"shape": "cuboid", "x": [3, 4], "boundary": "bounceback"}],
        },
    )

    return {
        "initial_conditions": PointWiseSpaceTimeInitialConditions(lattice),
        "algorithm": SpaceTimeQLBM(lattice),
        "postprocessing": EmptyPrimitive(lattice),
        "measurement": SpaceTimeGridVelocityMeasurement(lattice),
        "lattice": lattice,
    }


def build_config(circuits, shots):
    """A validated, prepared config from a circuits fixture."""
    cfg = SimulationConfig(
        initial_conditions=circuits["initial_conditions"],
        algorithm=circuits["algorithm"],
        postprocessing=circuits["postprocessing"],
        measurement=circuits["measurement"],
        optimization_level=0,
        shots=shots,
    )
    cfg.validate()
    cfg.prepare_for_simulation()
    return cfg


@pytest.fixture
def recorded_fields(monkeypatch):
    """The decoded field of every time step, in the order the runner saves them."""
    fields = []
    save = QBMResult.save_timestep_array

    def record(self, numpy_res, timestep, *args, **kwargs):
        fields.append(np.array(numpy_res, dtype=float))
        return save(self, numpy_res, timestep, *args, **kwargs)

    monkeypatch.setattr(QBMResult, "save_timestep_array", record)
    return fields


def assert_visualization_artifacts(output_directory: str, num_steps: int):
    """One Paraview frame per time step, plus the serialized lattice."""
    assert os.path.isfile(f"{output_directory}/lattice.json")
    for step in range(num_steps + 1):
        assert os.path.isfile(f"{output_directory}/paraview/step_{step:03d}.vti")


@pytest.mark.parametrize("statevector_snapshots", [True, False])
def test_collisionless_execution(
    collisionless_circuits, statevector_snapshots, recorded_fields, tmp_path
):
    """Every shot lands in the fluid: none inside the specular obstacle, none lost.

    The obstacle of ``symmetric_2d_1_obstacle.json`` spans x in [5, 6] and
    y in [1, 2]; the recorded fields index ``[y][x]``.
    """
    cfg = build_config(collisionless_circuits, shots=2048)
    runner = QarpRunner(cfg, collisionless_circuits["lattice"], seed=11)

    output_directory = str(tmp_path / f"collisionless-{int(statevector_snapshots)}")
    result = runner.run(
        NUM_STEPS,
        2048,  # Number of shots per time step
        output_directory,
        statevector_snapshots=statevector_snapshots,
    )

    assert result is not None
    assert_visualization_artifacts(output_directory, NUM_STEPS)
    assert len(recorded_fields) == NUM_STEPS + 1
    for field in recorded_fields:
        assert field.sum() == 2048
        assert field[1:3, 5:7].sum() == 0


def test_spacetime_execution(spacetime_circuits, recorded_fields, tmp_path):
    """The Space-Time lattice gas conserves its particle number across steps."""
    cfg = build_config(spacetime_circuits, shots=EXACT)
    runner = QarpRunner(cfg, spacetime_circuits["lattice"], seed=13)

    output_directory = str(tmp_path / "spacetime")
    result = runner.run(NUM_STEPS, EXACT, output_directory, statevector_snapshots=True)

    assert result is not None
    assert_visualization_artifacts(output_directory, NUM_STEPS)
    totals = [field.sum() for field in recorded_fields]
    assert len(totals) == NUM_STEPS + 1
    assert totals[0] > 0
    assert totals == pytest.approx([totals[0]] * len(totals))


def test_spacetime_rejects_the_rerun_loop(spacetime_circuits, tmp_path):
    """Space-Time re-encodes between steps, which replaying from zero would skip."""
    cfg = build_config(spacetime_circuits, shots=512)
    runner = QarpRunner(cfg, spacetime_circuits["lattice"], seed=13)

    with pytest.raises(ExecutionException, match="statevector_snapshots=True"):
        runner.run(NUM_STEPS, 512, str(tmp_path), statevector_snapshots=False)
