"""Each dimension's QFT shift is one declared ``StreamingShift`` block."""

import pytest

from qlbm.components.ab import ABStreamingOperator
from qlbm.components.ab.reflection.agnosotic_reflection import (
    ABZoneAgnosticReflectionOperator,
    ABZoneAgnosticSRCheck,
)
from qlbm.components.ab.reflection.standard_reflection import (
    ABSpecularReflectionOperator,
)
from qlbm.components.base import SequenceBlock
from qlbm.components.common import StreamingShift
from qlbm.components.ms import ControlledIncrementer
from qlbm.lattice import ABLattice, MSLattice, OHLattice


def specular_lattice() -> ABLattice:
    """An 8x8 D2Q9 lattice with one specular obstacle."""
    return ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
            "geometry": [
                {"shape": "cuboid", "x": [2, 4], "y": [3, 5], "boundary": "specular"}
            ],
        }
    )


def shifts_under(block, parent_type):
    """The ``StreamingShift`` descendants of ``block`` whose parent is a ``parent_type``."""
    found = []
    children = getattr(block, "children", None)
    for child in children() if callable(children) else []:
        if isinstance(child, StreamingShift) and isinstance(block, parent_type):
            found.append(child)
        found.extend(shifts_under(child, parent_type))
    return found


AB_LATTICES = [
    (ABLattice, {"x": 8}, "d1q3"),
    (ABLattice, {"x": 4, "y": 4}, "d2q9"),
    (ABLattice, {"x": 8, "y": 4}, "d2q9"),
    (OHLattice, {"x": 4, "y": 4}, "d2q9"),
]

MS_LATTICES = [
    ({"x": 4, "y": 4}, None),
    ({"x": 8, "y": 4}, "bounceback"),
    ({"x": 2, "y": 2, "z": 2}, None),
]


@pytest.mark.parametrize("lattice_class,dims,velocities", AB_LATTICES)
def test_ab_streaming_is_one_shift_per_dimension(lattice_class, dims, velocities):
    """The operator's children are the per-dimension shifts, and so are its parts."""
    lattice = lattice_class(
        {"lattice": {"dim": dims, "velocities": velocities}, "geometry": []}
    )

    operator = ABStreamingOperator(lattice, [])
    children = list(operator.children())

    assert len(children) == lattice.num_dims
    assert all(isinstance(child, StreamingShift) for child in children)
    assert operator.structure() == children


@pytest.mark.parametrize("dims,reflection", MS_LATTICES)
def test_ms_incrementer_is_one_shift_per_dimension(dims, reflection):
    """The incrementer's children are the per-dimension shifts, and so are its parts."""
    geometry = (
        [{"shape": "cuboid", "x": [1, 2], "y": [1, 2], "boundary": reflection}]
        if reflection
        else []
    )
    lattice = MSLattice(
        {
            "lattice": {"dim": dims, "velocities": {axis: 4 for axis in dims}},
            "geometry": geometry,
        }
    )

    incrementer = ControlledIncrementer(lattice, reflection=reflection)
    children = list(incrementer.children())

    assert len(children) == lattice.num_dims
    assert all(isinstance(child, StreamingShift) for child in children)
    assert incrementer.structure() == children


def test_specular_reflection_streams_with_one_shift_per_dimension():
    """The dimension-selective stream is a sequence of per-dimension shifts."""
    lattice = specular_lattice()

    operator = ABSpecularReflectionOperator(lattice, lattice.shapes["specular"])

    assert len(shifts_under(operator, SequenceBlock)) == lattice.num_dims
    assert len(shifts_under(operator, object)) == lattice.num_dims


def test_zone_agnostic_reflection_streams_with_one_shift_per_dimension():
    """The dimension-selective stream is a sequence of per-dimension shifts."""
    lattice = specular_lattice()

    operator = ABZoneAgnosticReflectionOperator(lattice)

    assert len(shifts_under(operator, SequenceBlock)) == lattice.num_dims


def test_sr_check_unstreams_and_restreams_with_one_shift_per_dimension():
    """Each dimension is unstreamed and streamed back by one shift each."""
    lattice = specular_lattice()

    check = ABZoneAgnosticSRCheck(
        lattice, lattice.discretization, lattice.shapes["specular"]
    )

    assert len(shifts_under(check, ABZoneAgnosticSRCheck)) == 2 * lattice.num_dims
