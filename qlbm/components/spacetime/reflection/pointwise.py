"""Reflection operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime` that swaps particles one gridpoint at a time."""

from typing import List, Tuple, cast

from typing_extensions import override

from qlbm.components.base import LBMOperator
from qlbm.components.common.primitives import MCSwap
from qlbm.lattice.geometry.shapes.base import Shape, SpaceTimeShape
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException


class PointWiseSpaceTimeReflectionOperator(LBMOperator):
    """Operator implementing reflection in the :class:`.SpaceTimeQLBM` algorithm, one gridpoint at a time.

    Work in progress.

    ============================ ======================================================================
    Attribute                     Summary
    ============================ ======================================================================
    :attr:`lattice`              The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`timestep`             The timestep for to which to perform reflection.
    :attr:`shapes`               A list of  :class:`.Shape` objects for which to generate the BB boundary condition circuits.
    :attr:`filter_inside_blocks` A ``bool`` that, when enabled, disregards operations that would have happened inside blocks.
    ============================ ======================================================================

    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        shapes: List[Shape],
        filter_inside_blocks: bool = True,
    ) -> None:
        if timestep < 1 or timestep > lattice.num_timesteps:
            raise CircuitException(
                f"Invalid time step {timestep}, select a value between 1 and {lattice.num_timesteps}"
            )
        self.timestep = timestep
        self.shapes = cast(List[SpaceTimeShape], shapes)
        self.filter_inside_blocks = filter_inside_blocks
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        grid_index = self.lattice.grid_index()
        for shape in self.shapes:
            for reflection_data in self.__reflection_data(shape):
                # Inverting the qubits that are 0 turns the grid qubit state
                # encoding this point to |11...1>, which allows controlling on it.
                grid_qubit_indices_to_invert = [
                    grid_index[0] + qubit for qubit in reflection_data.qubits_to_invert
                ]
                self.invert(grid_qubit_indices_to_invert)
                # Controlled on the grid qubits, swap the velocities affected by reflection
                for neighbor_velocity_pair in reflection_data.neighbor_velocity_pairs:
                    self.place(
                        MCSwap(
                            self.lattice,
                            grid_index,
                            cast(
                                Tuple[int, int],
                                tuple(
                                    self.lattice.velocity_index(nvp[0], nvp[1])[0]
                                    for nvp in neighbor_velocity_pair
                                ),
                            ),
                        )
                    )
                self.invert(grid_qubit_indices_to_invert)

    def __reflection_data(self, shape: SpaceTimeShape) -> List:
        match self.lattice.properties.get_discretization():
            case LatticeDiscretization.D1Q2:
                return shape.get_spacetime_reflection_data_d1q2(
                    self.lattice.properties, self.timestep
                )
            case LatticeDiscretization.D2Q4:
                reflection_data_points = shape.get_spacetime_reflection_data_d2q4(
                    self.lattice.properties, self.timestep
                )
                if self.filter_inside_blocks:
                    reflection_data_points = [
                        rdp
                        for rdp in reflection_data_points
                        if not self.lattice.is_inside_an_obstacle(rdp.gridpoint_encoded)
                    ]
                return reflection_data_points
            case discretization:
                raise CircuitException(
                    f"Reflection Operator unsupported for {discretization}."
                )

    @override
    def __str__(self) -> str:
        return f"[PointWiseSpaceTimeReflectionOperator for lattice {self.lattice}, blocks {self.shapes}, filtered {self.filter_inside_blocks}]"
