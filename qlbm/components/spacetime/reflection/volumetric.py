"""Reflection operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime` that swaps particles for simultaneously for fixed volumes."""

from typing import List, Tuple

from qarp.blocks import HnBlock
from typing_extensions import override

from qlbm.components.base import LBMOperator
from qlbm.components.spacetime.initial.volumetric import (
    stencil_neighbors,
    volume_bounds_at,
    volume_comparators,
)
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.exceptions import CircuitException


class VolumetricSpaceTimeReflectionOperator(LBMOperator):
    """
    Prepares the initial state for the :class:`.SpaceTimeQLBM` for a volumetric region.

    Work in progress.
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        blocks: List[Block],
        filter_inside_blocks: bool = True,
    ) -> None:
        if timestep < 1 or timestep > lattice.num_timesteps:
            raise CircuitException(
                f"Invalid time step {timestep}, select a value between 1 and {lattice.num_timesteps}"
            )
        self.timestep = timestep
        self.blocks = blocks
        self.filter_inside_blocks = filter_inside_blocks
        self.all_cuboid_bounds: List[List[Tuple[int, int]]] = [
            block.bounds for block in blocks
        ]
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        grid_index = self.lattice.grid_index()
        self.place(HnBlock(len(grid_index)), grid_index)

        # For all blocks, all "layers" of gridpoints, all stencil positions in the layer
        for block in self.blocks:
            for manhattan_distance in range(self.lattice.num_timesteps + 1):
                for neighbor in stencil_neighbors(self.lattice, manhattan_distance):
                    comparators = volume_comparators(
                        self.lattice,
                        volume_bounds_at(self.lattice, block.bounds, neighbor),
                    )
                    # Only the comparators are placed: they compute and
                    # uncompute their ancillae, the reflection is a no-op.
                    for comparator, qubits in comparators:
                        self.place(comparator, qubits)
                    for comparator, qubits in comparators:
                        self.place(comparator, qubits)
