"""Initial conditions for the :class:`.LQLGA` algorithm."""

from typing import List, Tuple

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.tools.utils import flatten


class LQGLAInitialConditions(LatticePrimitive):
    """
    Primitive for setting initial conditions in the :class:`.LQLGA` algorithm.

    This operator allows the construction of arbitrary deterministic initial conditions for the LQLGA algorithm.
    The number of gates required by this operator is equal to the number of enabled velocity qubits across all grid points.
    The depth of the circuit is 1, as all gates are applied in parallel at each grid point.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.lattice import LQLGALattice
        from qlbm.components.lqlga import LQGLAInitialConditions

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 4},
                    "velocities": "D1Q3",
                },
                "geometry": [],
            },
        )
        initial_conditions = LQGLAInitialConditions(lattice, [(tuple([2]), (True, True, True))])
        initial_conditions.plot()
    """

    grid_data: List[Tuple[Tuple[int, ...], Tuple[bool, ...]]]
    """
    Grid data for the initial conditions, where each tuple contains:
    #. A tuple of grid point indices (e.g., `(x, y, z)`).
    #. A tuple of booleans indicating which velocity qubits are enabled at that grid point.
    """

    lattice: LQLGALattice

    def __init__(
        self,
        lattice: LQLGALattice,
        grid_data: List[Tuple[Tuple[int, ...], Tuple[bool, ...]]],
    ) -> None:
        self.grid_data = grid_data
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        stride = self.lattice.num_velocities_per_point
        enabled = [
            self.lattice.gridpoint_index_tuple(gridpoint) * stride + velocity
            for gridpoint, velocity_profile in self.grid_data
            for velocity, is_enabled in enumerate(velocity_profile)
            if is_enabled
        ]
        if enabled:
            self.x(enabled)
        if self.lattice.has_multiple_geometries():
            self.h(self.lattice.marker_index())

    @override
    def __str__(self):
        return f"[Primitive LQGLAInitialConditions on lattice={self.lattice}, grid_data={self.grid_data})]"


class LQGLAAveragedInitialConditions(LatticePrimitive):
    """
    Primitive for setting initial conditions in the :class:`.LQLGA` algorithm.

    This operator creates an equal magnitude superposition over a set of gridpoints.
    This is equivalent to starting the QLGA algorithm in all possible configurations
    over the given set of gridpoints.


    Example usage:

    .. plot::
        :include-source:

        from qlbm.lattice import LQLGALattice
        from qlbm.components.lqlga import LQGLAAveragedInitialConditions

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 5},
                    "velocities": "D1Q3",
                },
                "geometry": [],
            },
        )
        initial_conditions = LQGLAAveragedInitialConditions(lattice, [0, 2, 3])
        initial_conditions.plot()

    """

    gridpoints: List[int]
    """The gridpoints to create the uniform superposition over."""

    lattice: LQLGALattice

    def __init__(self, lattice: LQLGALattice, gridpoints: List[int]) -> None:
        self.gridpoints = gridpoints
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        stride = self.lattice.num_velocities_per_point
        self.h(
            flatten(
                [
                    list(range(gp * stride, gp * stride + stride))
                    for gp in self.gridpoints
                ]
            )
        )

    @override
    def __str__(self):
        return f"[Primitive LQGLAAveragedInitialConditions on lattice={self.lattice}, gps={self.gridpoints})]"
