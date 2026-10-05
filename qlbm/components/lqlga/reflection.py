"""Reflection operator for the :class:`.LQLGA` algorithm :cite:`spacetime` that swaps particles one gridpoint at a time."""

from typing import List, Tuple, cast

from qarp.blocks import XnBlock
from typing_extensions import override

from qlbm.components.base import LatticePrimitive, LBMOperator
from qlbm.components.common.primitives import MCSwap
from qlbm.lattice.geometry.shapes.base import LQLGAShape, Shape
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException
from qlbm.tools.utils import get_qubits_to_invert


def _reject_unsupported(lattice: LQLGALattice) -> None:
    """Raise unless reflection is implemented for the discretization of ``lattice``."""
    # Checked even without shapes: LQLGA streaming only moves the first
    # x-line, so this is what keeps a 2D or 3D time step from building.
    if lattice.discretization not in (
        LatticeDiscretization.D1Q2,
        LatticeDiscretization.D1Q3,
    ):
        raise CircuitException(
            f"Reflection Operator unsupported for {lattice.discretization}."
        )


def _swap_pairs(lattice: LQLGALattice, shape: LQLGAShape) -> List[Tuple[int, int]]:
    """The velocity-qubit pairs ``shape`` reflects on ``lattice``."""
    match lattice.discretization:
        case LatticeDiscretization.D1Q2:
            reflection_data = shape.get_lqlga_reflection_data_d1q2()
        case LatticeDiscretization.D1Q3:
            reflection_data = shape.get_lqlga_reflection_data_d1q3()
        case _:
            raise CircuitException(
                f"Reflection Operator unsupported for {lattice.discretization}."
            )
    return [
        (
            lattice.velocity_index_tuple(
                data.gridpoints[0], data.velocity_indices_to_swap[0]
            ),
            lattice.velocity_index_tuple(
                data.gridpoints[1], data.velocity_indices_to_swap[1]
            ),
        )
        for data in reflection_data
    ]


class LQLGAReflectionOperator(LatticePrimitive):
    """
    Operator implementing reflection in the :class:`.LQLGA` algorithm.

    Reflections in this algorithm can be entirely implemented by swap gates.
    The number of gates scales with the number of gridpoints of the solid geometry.
    The depth of the operator is 1.

    ============================ ======================================================================
    Attribute                     Summary
    ============================ ======================================================================
    :attr:`lattice`              The lattice the operator acts on.
    :attr:`shapes`               A list of boundary-conditioned shapes.
    ============================ ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.lqlga import LQLGAReflectionOperator
        from qlbm.lattice import LQLGALattice

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 7},
                    "velocities": "D1Q3",
                },
                "geometry": [{"shape": "cuboid", "x": [3, 5], "boundary": "bounceback"}],
            },
        )
        reflection_operator = LQLGAReflectionOperator(
            lattice, shapes=lattice.shapes["bounceback"]
        )
        reflection_operator.plot()

    """

    shapes: List[LQLGAShape]
    """
    A list of shapes that require reflection at the boundaries.
    """

    lattice: LQLGALattice

    def __init__(self, lattice: LQLGALattice, shapes: List[Shape]) -> None:
        self.shapes = cast(List[LQLGAShape], shapes)
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        _reject_unsupported(self.lattice)
        for shape in self.shapes:
            pairs = _swap_pairs(self.lattice, shape)
            if pairs:
                self.swap(pairs)

    @override
    def __str__(self) -> str:
        return f"[PointWiseLQLGAReflectionOperator for lattice {self.lattice}, shapes {self.shapes}]"


class LQLGAMGReflectionOperator(LBMOperator):
    """
    Operator implementing reflection in the :class:`.LQLGA` algorithm with multiple geometries.

    Reflections in this algorithm can be entirely implemented a series of controlled swap gates.
    This is equivalent to multiple controlled realizations of the :class:`.LQLGAReflectionOperator`
    applied in series.

    ============================ ======================================================================
    Attribute                     Summary
    ============================ ======================================================================
    :attr:`lattice`              The lattice the operator acts on.
    :attr:`shapes`               A list of boundary-conditioned shapes.
    ============================ ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.lqlga import LQLGAReflectionOperator
        from qlbm.lattice import LQLGALattice

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 7},
                    "velocities": "D1Q3",
                },
                "geometry": [{"shape": "cuboid", "x": [3, 5], "boundary": "bounceback"}],
            },
        )
        reflection_operator = LQLGAReflectionOperator(
            lattice, shapes=lattice.shapes["bounceback"]
        )
        reflection_operator.plot()

    """

    shapes: List[List[LQLGAShape]]
    """
    A list of shapes that require reflection at the boundaries.
    """

    lattice: LQLGALattice

    def __init__(self, lattice: LQLGALattice, shapes: List[List[Shape]]) -> None:
        self.shapes = cast(List[List[LQLGAShape]], shapes)
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        _reject_unsupported(self.lattice)
        marker = self.lattice.marker_index()
        for c, geometry in enumerate(self.shapes):
            # Prepare the |1> state in the marker register for geometry c
            qubits_to_invert = [
                marker[0] + q
                for q in get_qubits_to_invert(c, self.lattice.num_marker_qubits)
            ]
            if qubits_to_invert:
                self.place(XnBlock(len(qubits_to_invert)), qubits_to_invert)
            for shape in geometry:
                for pair in _swap_pairs(self.lattice, shape):
                    self.place(MCSwap(self.lattice, marker, pair))
            if qubits_to_invert:
                self.place(XnBlock(len(qubits_to_invert)), qubits_to_invert)

    @override
    def __str__(self) -> str:
        return f"[LQLGAMGReflectionOperator for lattice {self.lattice}, shapes {self.shapes}]"
