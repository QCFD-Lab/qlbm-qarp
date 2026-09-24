"""Primitives for the implementation of the Collisionless Quantum Lattice Boltzmann Method introduced in :cite:t:`collisionless`."""

from typing_extensions import override

from qlbm.components.base import LatticePrimitive, LBMOperator
from qlbm.components.common.comparators import SingleRegisterComparator
from qlbm.lattice import MSLattice
from qlbm.lattice.geometry.encodings.ms import ReflectionResetEdge
from qlbm.tools import flatten
from qlbm.tools.utils import ComparatorMode


class GridMeasurement(LatticePrimitive):
    """A primitive that implements a measurement operation on the grid qubits.

    Used at the end of the time step circuit to extract information from the quantum state.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import GridMeasurement
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice({
            "lattice": {
                "dim": {
                        "x": 8,
                        "y": 8
                    },
                    "velocities": {
                        "x": 4,
                        "y": 4
                }
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [5, 6],
                    "y": [1, 2],
                    "boundary": "specular"
                }
            ]
        })

        # Draw the measurement circuit
        GridMeasurement(lattice).plot()
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        all_grid_qubits = flatten(
            [self.lattice.grid_index(dim) for dim in range(self.lattice.num_dims)]
        )
        # Classical bits are flat indices in qarp; one bit per grid qubit.
        self.measure([(qubit, cbit) for cbit, qubit in enumerate(all_grid_qubits)])

    @override
    def __str__(self) -> str:
        return f"[Primitive DVGridMeasurement with lattice {self.lattice}]"


class MSInitialConditions(LatticePrimitive):
    """A primitive that creates the quantum circuit to prepare the flow field in its initial conditions for the :class:`.MSLattice`.

    The initial conditions create a quantum state spanning half the grid
    in the x-axis, and the entirety of the y (and z)-axes (if 3D).
    All velocities are pointing in the positive direction.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import MSInitialConditions
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice({
            "lattice": {
                "dim": {
                        "x": 8,
                        "y": 8
                    },
                    "velocities": {
                        "x": 4,
                        "y": 4
                }
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [5, 6],
                    "y": [1, 2],
                    "boundary": "specular"
                }
            ]
        })

        # Draw the initial conditions circuit
        MSInitialConditions(lattice).plot()
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        self.x(
            [
                self.lattice.velocity_dir_index(dim)[0]
                for dim in range(self.lattice.num_dims)
            ]
        )
        superposed = self.lattice.grid_index(0)[:-1] + flatten(
            [self.lattice.grid_index(dim) for dim in range(1, self.lattice.num_dims)]
        )
        if superposed:
            self.h(superposed)

    @override
    def __str__(self) -> str:
        return f"[Primitive InitialConditions with lattice {self.lattice}]"


class MSInitialConditions3DSlim(LatticePrimitive):
    r"""
    A primitive that creates the quantum circuit to prepare the flow field in its initial conditions for 3 dimensions.

    The initial conditions create the quantum state
    :math:`\Sigma_{j}\ket{0}^{\otimes n_{g_x}}\ket{0}^{\otimes n_{g_y}}\ket{j}` over the grid qubits,
    that is, spanning the z-axis at the bottom of the x- and y-axes.
    This is helpful for debugging edge cases around the corners of 3D obstacles.
    All velocities are pointing in the positive direction.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import MSInitialConditions3DSlim
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice({
            "lattice": {
                "dim": {
                "x": 8,
                "y": 8,
                "z": 8
                },
                "velocities": {
                "x": 4,
                "y": 4,
                "z": 4
                }
            },
            "geometry": []
        })

        # Draw the initial conditions circuit
        MSInitialConditions3DSlim(lattice).plot()
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        self.x(self.lattice.velocity_dir_index())
        if self.lattice.num_dims > 2:
            self.h(self.lattice.grid_index(2))

    @override
    def __str__(self) -> str:
        return f"[Primitive InitialConditions with lattice {self.lattice}]"


class EdgeComparator(LBMOperator):
    """
    A primitive used in the 3D collisionless :class:`SpecularReflectionOperator` and :class:`BounceBackReflectionOperator` described in :cite:t:`collisionless`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`edge`              The coordinates of the edge within the grid.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import EdgeComparator
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {
                    "dim": {"x": 8, "y": 8, "z": 8},
                    "velocities": {"x": 4, "y": 4, "z": 4},
                },
                "geometry": [{"shape":"cuboid", "x": [2, 5], "y": [2, 5], "z": [2, 5], "boundary": "specular"}],
            }
        )

        # Draw the edge comparator circuit for one specific corner edge
        EdgeComparator(lattice, lattice.shape_list[0].corner_edges_3d[0]).plot()
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice, edge: ReflectionResetEdge) -> None:
        self.edge = edge
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        dim = self.edge.dim_disconnected
        num_qubits = self.lattice.num_gridpoints[dim].bit_length() + 1
        grid_index = self.lattice.grid_index(dim)
        # Two comparator ancillae per relevant dimension: one for the lower bound, one for the upper.
        ancillae = self.lattice.ancillae_comparator_index(0)

        self.place(
            SingleRegisterComparator(
                num_qubits, self.edge.bounds_disconnected_dim[0], ComparatorMode.GE
            ),
            grid_index + ancillae[:-1],
        )
        self.place(
            SingleRegisterComparator(
                num_qubits, self.edge.bounds_disconnected_dim[1], ComparatorMode.LE
            ),
            grid_index + ancillae[1:],
        )

    @override
    def __str__(self) -> str:
        return f"[Primitive SpecularEdgeComparator on edge={self.edge}]"
