"""The optimization level reaches qarp's execution with the block tree intact."""

import pytest
from qarp.blocks import CompositeBlockBase, SimpleBlock

from qlbm.components import CQLBM
from qlbm.components.ab import (
    ABDiscreteUniformInitialConditions,
    ABGridMeasurement,
)
from qlbm.components.common import EmptyPrimitive
from qlbm.infra import CircuitCompiler, QarpRunner, SimulationConfig
from qlbm.infra.runner import qarp_runner
from qlbm.lattice import ABLattice


def lattice() -> ABLattice:
    """A 16x8 D2Q9 lattice with one bounce-back obstacle."""
    return ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            "geometry": [
                {"shape": "cuboid", "x": [5, 8], "y": [2, 5], "boundary": "bounceback"}
            ],
        }
    )


@pytest.mark.parametrize("level", [0, 1, 2])
def test_compile_keeps_the_block_tree(level):
    """Compilation at any level returns the component with its children."""
    step = CQLBM(lattice())

    compiled = CircuitCompiler().compile(step, optimization_level=level)

    assert compiled is step
    assert len(list(compiled.children())) == len(list(step.children()))


@pytest.mark.parametrize("level", [1, 2])
def test_runner_hands_the_level_to_qarp(level, monkeypatch, tmp_path):
    """The step evolves through ``statevector`` and samples through ``QarpEngine`` at the configured level."""
    seen: dict[str, list[int | None]] = {"statevector": [], "engine": []}

    def record(cls):
        original = cls.statevector

        def spy(self, initial_state=None, **kwargs):
            seen["statevector"].append(kwargs.get("optimization_level"))
            return original(self, initial_state, **kwargs)

        monkeypatch.setattr(cls, "statevector", spy)

    # ``statevector`` is defined on each base class, not inherited.
    for cls in (SimpleBlock, CompositeBlockBase):
        record(cls)

    class RecordingEngine(qarp_runner.QarpEngine):
        def __init__(self, *args, **kwargs):
            seen["engine"].append(kwargs.get("optimization_level"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(qarp_runner, "QarpEngine", RecordingEngine)
    lat = lattice()
    config = SimulationConfig(
        initial_conditions=ABDiscreteUniformInitialConditions(lat, [1, 5], ([], [])),
        algorithm=CQLBM(lat),
        postprocessing=EmptyPrimitive(lat),
        measurement=ABGridMeasurement(lat),
        optimization_level=level,
        shots=64,
    )
    config.prepare_for_simulation()

    QarpRunner(config, lat).run(
        num_steps=1, num_shots=64, output_directory=str(tmp_path)
    )

    assert level in seen["statevector"]
    assert seen["engine"] == [level] * len(seen["engine"]) and seen["engine"]
