"""Reflection utilities for the :class:`.ABQLBM` algorithm; generalizations of :cite:`collisionless`."""

from itertools import product
from typing import Dict, List, Tuple, cast

from qarp.blocks import AnyBlock, CompositeBlock
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.reflection.common import (
    ABBounceBackReflectionPermutation,
    ABSpecularReflectionPermutation,
)
from qlbm.components.ab.streaming import (
    STREAMING_POPULATIONS,
    ABStreamingOperator,
    velocity_qubits_to_invert,
)
from qlbm.components.base import LBMOperator, flip_if, on
from qlbm.components.common.adders import StreamingShift, shift_on
from qlbm.components.ms.specular_reflection import SpecularWallComparator
from qlbm.lattice.geometry.encodings.ms import ReflectionPoint, ReflectionWall
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten, get_qubits_to_invert


def set_ancilla_of_point_state(
    lattice: AmplitudeLattice,
    points_data: List[Tuple[ReflectionPoint, List[int] | None]],
    ignore_velocity_data: bool,
    control_on_marker_state: bool = False,
    target_obstacle_index: int = 0,
) -> AnyBlock:
    """
    Flip an obstacle ancilla for particles at the given gridpoints (and velocities).

    Parameters
    ----------
    lattice : AmplitudeLattice
        The lattice whose registers are addressed.
    points_data : List[Tuple[ReflectionPoint, List[int] | None]]
        The gridpoints, each with the velocity indices that select the flip
        (ignored when ``ignore_velocity_data`` is set).
    ignore_velocity_data : bool
        Whether to flip for every velocity at the point.
    control_on_marker_state : bool
        Whether to additionally control on the multi-geometry marker register.
    target_obstacle_index : int
        Which obstacle ancilla to flip.

    Returns
    -------
    AnyBlock
        The block over the full lattice width.
    """
    marker = lattice.marker_index() if control_on_marker_state else []
    target_qubits = lattice.ancillae_obstacle_index(target_obstacle_index)
    grid_index = lattice.grid_index()
    children = []
    for point, velocities in points_data:
        grid_inverted = [grid_index[0] + qubit for qubit in point.qubits_to_invert]
        if ignore_velocity_data:
            children.append(flip_if(grid_index + marker, target_qubits, grid_inverted))
            continue
        for velocity in velocities or []:
            match lattice.get_encoding():
                case ABEncodingType.AB:
                    children.append(
                        flip_if(
                            grid_index + lattice.velocity_index() + marker,
                            target_qubits,
                            grid_inverted
                            + velocity_qubits_to_invert(lattice, velocity),
                        )
                    )
                case ABEncodingType.OH:
                    children.append(
                        flip_if(
                            grid_index + [lattice.velocity_index()[velocity]] + marker,
                            target_qubits,
                            grid_inverted,
                        )
                    )
                case _:
                    raise LatticeException(
                        f"Unsupported lattice encoding: {lattice.get_encoding()}"
                    )
    return CompositeBlock(children, lattice.n_qubits, name="ab_set_ancilla")


class ABReflectionOperator(LBMOperator):
    r"""
    Implements reflection boundary conditions in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_dQ_q` discretizations.

    This is the top-level entrypoint that delegates to :class:`ABBounceBackReflectionOperator` and :class:`ABSpecularReflectionOperator` based on the boundary condition types present in the geometry.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABReflectionOperator
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [1, 3],
                        "y": [1, 3],
                        "boundary": "bounceback",
                    }
                ],
            }
        )

        ABReflectionOperator(lattice).plot()

    """

    lattice: ABLattice

    def __init__(
        self, lattice: ABLattice, shapes: Dict[str, List[Shape]] | None = None
    ) -> None:
        if shapes is not None:
            self.shapes: Dict[str, List[Shape]] | List[Dict[str, List[Shape]]] = (
                shapes if not lattice.has_multiple_geometries() else [shapes]
            )
        elif not lattice.has_multiple_geometries():
            self.shapes = lattice.geometries[0]
        else:
            self.shapes = list(lattice.geometries)
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        if self.lattice.discretization not in [LatticeDiscretization.D2Q9]:
            raise LatticeException("AB reflection only currently supported in D2Q9")

        if not self.lattice.has_multiple_geometries():
            self.__place_d2q9(cast(Dict[str, List[Shape]], self.shapes))
            return

        marker = self.lattice.marker_index()
        for c, shapes_for_geometry in enumerate(
            cast(List[Dict[str, List[Shape]]], self.shapes)
        ):
            # Map the marker state of geometry c to |1...1> around its reflection
            qubits_to_invert = [
                marker[0] + q
                for q in get_qubits_to_invert(c, self.lattice.num_marker_qubits)
            ]
            self.invert(qubits_to_invert)
            self.__place_d2q9(shapes_for_geometry, control_on_marker_state=True)
            self.invert(qubits_to_invert)

    def __place_d2q9(
        self,
        shapes_dict: Dict[str, List[Shape]],
        control_on_marker_state: bool = False,
    ) -> None:
        bb_blocks = shapes_dict.get("bounceback", [])
        sr_blocks = shapes_dict.get("specular", [])
        if bb_blocks:
            self.place(
                ABBounceBackReflectionOperator(
                    self.lattice, bb_blocks, control_on_marker_state
                )
            )
        if sr_blocks:
            self.place(
                ABSpecularReflectionOperator(
                    self.lattice, sr_blocks, control_on_marker_state
                )
            )

    @override
    def __str__(self) -> str:
        return f"[Operator ABReflection with lattice {self.lattice}]"


class ABBounceBackReflectionOperator(LBMOperator):
    r"""
    Implements bounce-back reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_2Q_9`.

    Bounce-back reflection reverses all velocity components of particles
    that have entered the obstacle walls. The algorithm proceeds as:

    1. **Mark inner walls** -- Set the obstacle ancilla for gridpoints
       inside each wall segment.
    2. **Mark inner corners** -- Set the obstacle ancilla for the inner
       corner gridpoints.
    3. **Permute and stream** -- Apply the bounce-back velocity
       permutation (full reversal) and stream, both controlled on the
       obstacle ancilla.
    4. **Reset outer walls** -- Reset the obstacle ancilla using the
       outside wall comparators.
    5. **Corner corrections** -- Fix near-corner and outside-corner
       ancilla residuals.
    """

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        blocks: List[Shape],
        control_on_marker_state: bool = False,
    ) -> None:
        self.blocks = blocks
        self.control_on_marker_state = control_on_marker_state
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        if self.lattice.discretization != LatticeDiscretization.D2Q9:
            raise LatticeException("AB bounce-back reflection only supported in D2Q9")

        for block in self.blocks:
            self.place(self.set_inside_wall_ancilla_state(block))  # type: ignore[arg-type]
        self.place(
            set_ancilla_of_point_state(
                self.lattice,
                flatten(
                    [[(p, None) for p in block.corners_inside] for block in self.blocks]  # type: ignore[attr-defined]
                ),
                ignore_velocity_data=True,
                control_on_marker_state=self.control_on_marker_state,
            )
        )

        self.place(self.permute_and_stream())

        for block in self.blocks:
            self.place(self.reset_outside_wall_ancilla_state(block))  # type: ignore[arg-type]

        point_data: List[Tuple[ReflectionPoint, List[int] | None]] = []
        for block in self.blocks:
            for dim in range(self.lattice.num_dims):
                for c, bounds in enumerate(
                    product(*[[False, True]] * self.lattice.num_dims)
                ):
                    point_data.append(
                        (
                            block.near_corner_points_2d[dim * 4 + c],  # type: ignore[attr-defined]
                            block.get_lbm_near_corner_velocity_indices_to_reflect(  # type: ignore[attr-defined]
                                self.lattice.discretization, dim, bounds
                            ),
                        )
                    )
            for c, bounds in enumerate(
                product(*[[False, True]] * self.lattice.num_dims)
            ):
                point_data.append(
                    (
                        block.corners_outside[c],  # type: ignore[attr-defined]
                        block.get_lbm_outside_corner_indices_to_reflect(  # type: ignore[attr-defined]
                            self.lattice.discretization, bounds
                        ),
                    )
                )
        self.place(
            set_ancilla_of_point_state(
                self.lattice,
                point_data,
                ignore_velocity_data=False,
                control_on_marker_state=self.control_on_marker_state,
            )
        )

    def _marker_controls(self) -> List[int]:
        return self.lattice.marker_index() if self.control_on_marker_state else []

    def _grid_inverted(self, wall: ReflectionWall) -> List[int]:
        # Empty only for the |11..1> grid position in the reflected dimension.
        return [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in wall.data.qubits_to_invert
        ]

    def _velocity_controls(self, velocity: int) -> Tuple[List[int], List[int]]:
        """Control qubits selecting ``velocity`` and, of those, the ones active on |0>."""
        match self.lattice.get_encoding():
            case ABEncodingType.AB:
                return (
                    self.lattice.velocity_index(),
                    velocity_qubits_to_invert(self.lattice, velocity),
                )
            case ABEncodingType.OH:
                return [self.lattice.velocity_index()[velocity]], []
            case _:
                raise LatticeException(
                    f"Unsupported lattice encoding: {self.lattice.get_encoding()}"
                )

    def set_inside_wall_ancilla_state(self, block: Block) -> AnyBlock:
        """
        Build the block that marks the particles that streamed into the inner walls of ``block``.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        children = []
        for dim in range(self.lattice.num_dims):
            for wall in block.walls_inside[dim]:
                comparator = SpecularWallComparator(self.lattice, wall)
                children += [
                    comparator,
                    flip_if(
                        self.lattice.grid_index(wall.dim)
                        + self.lattice.ancillae_comparator_index()
                        + self._marker_controls(),
                        self.lattice.ancillae_obstacle_index(0),
                        self._grid_inverted(wall),
                    ),
                    comparator,
                ]
        return CompositeBlock(children, self.n_qubits, name="ab_bb_inner_wall")

    def reset_outside_wall_ancilla_state(self, block: Block) -> AnyBlock:
        """
        Build the block that unmarks the particles reflected onto the outer walls of ``block``.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        children = []
        for dim in range(self.lattice.num_dims):
            for bound, wall in enumerate(block.walls_outside[dim]):
                comparator = SpecularWallComparator(self.lattice, wall)
                children.append(comparator)
                for v in block.get_lbm_wall_velocity_indices_to_reflect(
                    self.lattice.discretization, dim, bool(bound)
                ):
                    velocity_controls, velocity_inverted = self._velocity_controls(v)
                    children.append(
                        flip_if(
                            self.lattice.grid_index(wall.dim)
                            + self.lattice.ancillae_comparator_index()
                            + velocity_controls
                            + self._marker_controls(),
                            self.lattice.ancillae_obstacle_index(0),
                            self._grid_inverted(wall) + velocity_inverted,
                        )
                    )
                children.append(comparator)
        return CompositeBlock(children, self.n_qubits, name="ab_bb_outer_wall")

    def permute_and_stream(self) -> AnyBlock:
        """
        Build the block that bounces the marked velocities back and streams them.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        return CompositeBlock(
            [
                on(
                    ABBounceBackReflectionPermutation(
                        self.lattice.num_velocity_qubits,
                        self.lattice.discretization,
                        self.lattice.get_encoding(),
                        num_ctrl_qubits=1,
                    ),
                    self.lattice.ancillae_obstacle_index(0)
                    + self.lattice.velocity_index(),
                ),
                ABStreamingOperator(
                    self.lattice, self.lattice.ancillae_obstacle_index(0)
                ),
            ],
            self.n_qubits,
            name="ab_permute_and_stream",
        )

    @override
    def __str__(self) -> str:
        return f"[Operator ABBounceBackReflection with lattice {self.lattice}]"


class ABSpecularReflectionOperator(LBMOperator):
    r"""
    Implements specular reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_2Q_9`.

    This operator uses per-dimension obstacle ancillae (``a_x``, ``a_y``)
    to track which wall a particle has entered through. The algorithm
    proceeds in five phases:

    1. **Mark inner walls** -- For each dimension, set the per-dimension
       obstacle ancilla for all gridpoints lying inside the walls of
       each block.
    2. **Specular permutations** -- Apply per-dimension velocity
       permutations controlled on the corresponding obstacle ancilla.
    3. **Dimension-selective stream** -- For each dimension *d*, stream
       the grid qubits of *d* controlled on ``ancilla[d]``. This
       ensures a particle reflected off an x-wall only moves in x,
       while a corner particle (both ancillae set) moves in both.
    4. **Reset outer walls** -- For each dimension, reset the
       per-dimension obstacle ancilla using the outside wall
       comparators.
    5. **Corner corrections** -- Fix near-corner and outside-corner
       ancilla residuals caused by the wall-based reset overshoot and
       by cardinal velocities at inner corners.

    Parameters
    ----------
    lattice : ABLattice
        The lattice on which to build the reflection circuit.
    blocks : List[Shape]
        The list of specular :class:`.Block` objects.
    control_on_marker_state : bool
        Whether to control all MCX gates on the marker register.
    """

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        blocks: List[Shape],
        control_on_marker_state: bool = False,
    ) -> None:
        self.blocks = blocks
        self.control_on_marker_state = control_on_marker_state
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        if self.lattice.discretization != LatticeDiscretization.D2Q9:
            raise LatticeException("AB specular reflection only supported in D2Q9")

        # Phase 1: Mark inner walls per dimension
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                self.place(self._set_inside_wall_ancilla_per_dim(block, dim))  # type: ignore[arg-type]
        # Phase 1b: Correct inner corner ancillae
        self.place(self._correct_inner_corner_ancillae())
        # Phase 2: Per-dimension specular permutations
        self.place(self._specular_permutations())
        # Phase 3: Dimension-selective streaming
        self.place(self._dim_selective_stream())
        # Phase 4: Reset outer walls per dimension
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                self.place(self._reset_outside_wall_ancilla_per_dim(block, dim))  # type: ignore[arg-type]
        # Phase 5: Corner corrections
        self.place(self._corner_corrections())

    def _marker_controls(self) -> List[int]:
        return self.lattice.marker_index() if self.control_on_marker_state else []

    def _grid_inverted(self, wall: ReflectionWall) -> List[int]:
        # Empty only for the |11..1> grid position in the reflected dimension.
        return [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in wall.data.qubits_to_invert
        ]

    def _set_inside_wall_ancilla_per_dim(self, block: Block, dim: int) -> AnyBlock:
        children = []
        for wall in block.walls_inside[dim]:
            comparator = SpecularWallComparator(self.lattice, wall)
            children += [
                comparator,
                flip_if(
                    self.lattice.grid_index(wall.dim)
                    + self.lattice.ancillae_comparator_index()
                    + self._marker_controls(),
                    self.lattice.ancillae_obstacle_index(dim),
                    self._grid_inverted(wall),
                ),
                comparator,
            ]
        return CompositeBlock(children, self.n_qubits, name="ab_sr_inner_wall")

    def _correct_inner_corner_ancillae(self) -> AnyBlock:
        children = []
        for block in self.blocks:
            for c, bounds in enumerate(
                product(*[[False, True]] * self.lattice.num_dims)
            ):
                corner_point = block.corners_inside[c]  # type: ignore[attr-defined]
                for dim in range(self.lattice.num_dims):
                    velocities_to_unset = (
                        block.get_lbm_sr_inner_corner_non_entering_velocity_indices(  # type: ignore[attr-defined]
                            self.lattice.discretization, dim, bounds[dim]
                        )
                    )
                    children.append(
                        set_ancilla_of_point_state(
                            self.lattice,
                            [(corner_point, velocities_to_unset)],
                            ignore_velocity_data=False,
                            control_on_marker_state=self.control_on_marker_state,
                            target_obstacle_index=dim,
                        )
                    )
        return CompositeBlock(children, self.n_qubits, name="ab_sr_inner_corner")

    def _specular_permutations(self) -> AnyBlock:
        return CompositeBlock(
            [
                on(
                    ABSpecularReflectionPermutation(
                        self.lattice.num_velocity_qubits,
                        self.lattice.discretization,
                        self.lattice.get_encoding(),
                        tuple(d == dim for d in range(self.lattice.num_dims)),
                        num_ctrl_qubits=1,
                    ),
                    self.lattice.ancillae_obstacle_index(dim)
                    + self.lattice.velocity_index(),
                )
                for dim in range(self.lattice.num_dims)
            ],
            self.n_qubits,
            name="ab_sr_permute",
        )

    def _dim_selective_stream(self) -> AnyBlock:
        children = []
        for dim, dim_population_to_update in enumerate(
            STREAMING_POPULATIONS[LatticeDiscretization.D2Q9]
        ):
            grid_index = self.lattice.grid_index(dim)
            controls = (
                self.lattice.ancillae_obstacle_index(dim)
                + self.lattice.velocity_index()
            )
            shifts = [
                shift_on(
                    direction == 0,
                    controls,
                    controls,
                    velocity_qubits_to_invert(self.lattice, index),
                )
                for direction, indices in enumerate(dim_population_to_update)
                for index in indices
            ]
            children.append(
                on(
                    StreamingShift(len(grid_index), len(controls), shifts),
                    controls + grid_index,
                )
            )
        return CompositeBlock(children, self.n_qubits, name="ab_sr_stream")

    def _reset_outside_wall_ancilla_per_dim(self, block: Block, dim: int) -> AnyBlock:
        children = []
        for bound, wall in enumerate(block.walls_outside[dim]):
            comparator = SpecularWallComparator(self.lattice, wall)
            children.append(comparator)
            for v in block.get_lbm_wall_velocity_indices_to_reflect(
                self.lattice.discretization, dim, bool(bound)
            ):
                children.append(
                    flip_if(
                        self.lattice.grid_index(wall.dim)
                        + self.lattice.ancillae_comparator_index()
                        + self.lattice.velocity_index()
                        + self._marker_controls(),
                        self.lattice.ancillae_obstacle_index(dim),
                        self._grid_inverted(wall)
                        + velocity_qubits_to_invert(self.lattice, v),
                    )
                )
            children.append(comparator)
        return CompositeBlock(children, self.n_qubits, name="ab_sr_outer_wall")

    def _corner_corrections(self) -> AnyBlock:
        children = []
        for block in self.blocks:
            # 5a: same-dimension near-corner corrections
            for dim in range(self.lattice.num_dims):
                same_dim_data: List[Tuple[ReflectionPoint, List[int] | None]] = [
                    (
                        block.near_corner_points_2d[dim * 4 + c],  # type: ignore[attr-defined]
                        block.get_lbm_near_corner_velocity_indices_to_reflect(  # type: ignore[attr-defined]
                            self.lattice.discretization, dim, bounds
                        ),
                    )
                    for c, bounds in enumerate(
                        product(*[[False, True]] * self.lattice.num_dims)
                    )
                ]
                children.append(
                    set_ancilla_of_point_state(
                        self.lattice,
                        same_dim_data,
                        ignore_velocity_data=False,
                        control_on_marker_state=self.control_on_marker_state,
                        target_obstacle_index=dim,
                    )
                )
            # 5b: outside corner corrections (both dimensions)
            for target_dim in range(self.lattice.num_dims):
                corner_data: List[Tuple[ReflectionPoint, List[int] | None]] = [
                    (
                        block.corners_outside[c],  # type: ignore[attr-defined]
                        block.get_lbm_outside_corner_indices_to_reflect(  # type: ignore[attr-defined]
                            self.lattice.discretization, bounds
                        ),
                    )
                    for c, bounds in enumerate(
                        product(*[[False, True]] * self.lattice.num_dims)
                    )
                ]
                children.append(
                    set_ancilla_of_point_state(
                        self.lattice,
                        corner_data,
                        ignore_velocity_data=False,
                        control_on_marker_state=self.control_on_marker_state,
                        target_obstacle_index=target_dim,
                    )
                )
        return CompositeBlock(children, self.n_qubits, name="ab_sr_corners")

    @override
    def __str__(self) -> str:
        return f"[Operator ABSpecularReflection with lattice {self.lattice}]"
