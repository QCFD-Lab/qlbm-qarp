"""Quantum comparator circuits for the implementation of bounce-back boundary conditions as described in :cite:t:`qmem`."""

from typing import List

from qarp.blocks import AnyBlock, CompositeBlock
from typing_extensions import override

from qlbm.components.base import LBMOperator, flip_if
from qlbm.components.common.comparators import SingleRegisterComparator
from qlbm.components.ms.specular_reflection import SpecularWallComparator
from qlbm.lattice import MSLattice
from qlbm.lattice.geometry.encodings.ms import (
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
)
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.tools.exceptions import CircuitException
from qlbm.tools.utils import ComparatorMode, flatten

from .primitives import EdgeComparator
from .streaming import ControlledIncrementer


class BounceBackWallComparator(LBMOperator):
    r"""
    A primitive used in the collision :class:`BounceBackReflectionOperator` that implements the comparator for the BB boundary conditions as described in :cite:t:`qmem`.

    The comparator sets an ancilla qubit to :math:`\ket{1}` for the components of
    the quantum state whose grid qubits fall within the range spanned by the wall.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`wall`              The :class:`.ReflectionWall` encoding the range spanned by the wall.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import BounceBackWallComparator
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [{"shape":"cuboid", "x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
            }
        )

        # Comparing on the indices of the inside x-wall on the lower-bound of the obstacle
        BounceBackWallComparator(
            lattice=lattice, wall=lattice.shape_list[0].walls_inside[0][0]
        ).plot()
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice, wall: ReflectionWall) -> None:
        self.wall = wall
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        # If the wall is inside the object, the comparators are strict so as not to overlap
        for c, dim in enumerate(self.wall.alignment_dims):
            loose = self.wall.bounceback_loose_bounds[self.wall.dim][c]
            num_qubits = self.lattice.num_gridpoints[dim].bit_length() + 1
            grid_index = self.lattice.grid_index(dim)
            # Two comparator ancillae per relevant dimension: one for the lower bound, one for the upper.
            ancillae = self.lattice.ancillae_comparator_index(c)

            self.place(
                SingleRegisterComparator(
                    num_qubits,
                    self.wall.lower_bounds[c],
                    ComparatorMode.GE if loose else ComparatorMode.GT,
                ),
                grid_index + ancillae[:-1],
            )
            self.place(
                SingleRegisterComparator(
                    num_qubits,
                    self.wall.upper_bounds[c],
                    ComparatorMode.LE if loose else ComparatorMode.LT,
                ),
                grid_index + ancillae[1:],
            )

    @override
    def __str__(self) -> str:
        return f"[Primitive BounceBackWallComparator on wall={self.wall}]"


class BounceBackReflectionOperator(LBMOperator):
    """
    Operator implementing the 2D and 3D Bounce-Back (BB) boundary conditions as described in :cite:t:`qmem`.

    The operator parses information encoded in :class:`.Block` objects to detect particles that
    have virtually streamed into the solid domain before placing them back to their
    previous positions in the fluid domain.
    The pseudocode for this procedure is as follows:

    #. Components of the quantum state that encode particles that have streamed inside the obstacle are identified with :class:`.BounceBackWallComparator` objects;
    #. These components have their velocity direction qubits flipped in all three dimensions;
    #. Particles are streamed outside the solid domain with inverted velocity directions;
    #. Once streamed outside the solid domain, components encoding affected particles have their obstacle ancilla qubit reset based on grid position, velocity direction, and whether they have streamed in the CFL timestep.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`blocks`            A list of  :class:`.Block` objects for which to generate the BB boundary condition circuits.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import BounceBackReflectionOperator
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [{"shape":"cuboid", "x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
            }
        )

        BounceBackReflectionOperator(lattice=lattice, blocks=lattice.shape_list)
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice, blocks: List[Block]) -> None:
        self.blocks = blocks
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        # Reflect the particles that have streamed into the inner walls of the object
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                for wall in block.walls_inside[dim]:
                    self.place(self.reflect_wall(wall))

        # Perform streaming
        self.place(self.flip_and_stream())

        # Reset the ancilla qubits for the particles that have been reflected
        # onto the outer walls of the object
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                for wall in block.walls_outside[dim]:
                    self.place(self.reflect_wall(wall))

        if self.lattice.num_dims == 2:
            # Reset state for near-corner points
            for block in self.blocks:
                for near_corner_point in block.near_corner_points_2d:
                    self.place(self.reset_point_state(near_corner_point))
        elif self.lattice.num_dims == 3:
            for block in self.blocks:
                # Reset the near-corner edges (24x)
                for near_corner_edge in block.near_corner_edges_3d:
                    self.place(self.reset_edge_state(near_corner_edge))
                # Reset the corner edges (12x)
                for corner_edge in block.corner_edges_3d:
                    self.place(self.reset_edge_state(corner_edge))
                for point in block.overlapping_near_corner_edge_points_3d:
                    self.place(self.reset_point_state(point))
        else:
            raise CircuitException(
                f"CQBM specular reflection is not supported for {self.lattice.num_dims} dimensions."
            )

        # Reset state for outside corners
        for block in self.blocks:
            # Reset the individual points at the corners of the block
            # (8x for 3D, 4x for 2D)
            for corner in block.corners_outside:
                self.place(self.reset_point_state(corner))

    def reflect_wall(self, wall: ReflectionWall) -> AnyBlock:
        """
        Build the block that reflects the particles hitting ``wall``.

        The comparator marks the relevant gridpoints on its ancillae, the
        obstacle ancillae are flipped conditioned on that mark, and the
        comparator uncomputes its ancillae.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        # If the wall is outside the obstacle, the two comparators behave identically
        comparator = (
            BounceBackWallComparator(self.lattice, wall)
            if not wall.data.is_outside_obstacle_bounds
            else SpecularWallComparator(self.lattice, wall)
        )
        direction = self.lattice.velocity_dir_index(wall.dim)
        control_qubits = (
            self.lattice.ancillae_velocity_index(wall.dim)
            + (direction if wall.data.is_outside_obstacle_bounds else [])
            + self.lattice.grid_index(wall.dim)
            + self.lattice.ancillae_comparator_index()
        )
        # Empty only for the |11..1> grid position in the reflected dimension.
        inverted = [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in wall.data.qubits_to_invert
        ] + (direction if wall.data.invert_velocity else [])

        return CompositeBlock(
            [
                comparator,
                flip_if(
                    control_qubits, self.lattice.ancillae_obstacle_index(0), inverted
                ),
                comparator,
            ],
            self.n_qubits,
            name="reflect_wall",
        )

    def reset_edge_state(self, edge: ReflectionResetEdge) -> AnyBlock:
        """
        Build the block that resets the obstacle ancilla along ``edge``.

        The comparator marks the relevant gridpoints on its ancillae, the
        obstacle ancillae are flipped conditioned on that mark, and the
        comparator uncomputes its ancillae.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        comparator = EdgeComparator(self.lattice, edge)
        control_qubits = flatten(
            [
                self.lattice.ancillae_velocity_index(dim)
                + self.lattice.velocity_dir_index(dim)
                + self.lattice.grid_index(dim)
                for dim in edge.dims_of_edge
            ]
        ) + self.lattice.ancillae_comparator_index(0)
        inverted = [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in flatten([wall.qubits_to_invert for wall in edge.walls_joining])
        ] + flatten(
            [
                self.lattice.velocity_dir_index(dim)
                for c, dim in enumerate(edge.dims_of_edge)
                if edge.invert_velocity_in_dimension[c]
            ]
        )

        return CompositeBlock(
            [
                comparator,
                flip_if(
                    control_qubits, self.lattice.ancillae_obstacle_index(0), inverted
                ),
                comparator,
            ],
            self.n_qubits,
            name="reset_edge_state",
        )

    def reset_point_state(self, corner: ReflectionPoint) -> AnyBlock:
        """
        Build the block that resets the obstacle ancilla at the gridpoint ``corner``.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        control_qubits = (
            self.lattice.ancillae_velocity_index()
            + self.lattice.velocity_dir_index()
            + self.lattice.grid_index()
        )
        # Empty only for the |11...1> grid point, i.e. the bottom-right most one.
        inverted = [
            self.lattice.grid_index(0)[0] + qubit for qubit in corner.qubits_to_invert
        ] + flatten(
            [
                self.lattice.velocity_dir_index(dim)
                for dim in range(corner.num_dims)
                if corner.invert_velocity_in_dimension[dim]
            ]
        )
        return CompositeBlock(
            [
                flip_if(
                    control_qubits, self.lattice.ancillae_obstacle_index(0), inverted
                )
            ],
            self.n_qubits,
            name="reset_point_state",
        )

    def flip_and_stream(self) -> AnyBlock:
        """
        Build the block that flips the velocity direction of reflected particles and streams them.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        return CompositeBlock(
            [
                flip_if(
                    self.lattice.ancillae_obstacle_index(0),
                    self.lattice.velocity_dir_index(),
                ),
                ControlledIncrementer(self.lattice, reflection="bounceback"),
            ],
            self.n_qubits,
            name="flip_and_stream",
        )

    @override
    def __str__(self) -> str:
        return f"[Operator BounceBackReflectionOperator against shapes {self.lattice.shapes}]"
