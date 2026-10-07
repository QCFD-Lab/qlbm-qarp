"""Zone-agnostic reflection utilities for the :class:`.ABQLBM` algorithm."""

from logging import getLogger
from typing import Dict, List, Tuple, cast

from qarp.blocks import AnyBlock, CompositeBlock, SimpleBlock
from typing_extensions import override

from qlbm.components.ab.reflection.common import (
    ABBounceBackReflectionPermutation,
    ABSpecularReflectionPermutation,
)
from qlbm.components.ab.streaming import (
    STREAMING_POPULATIONS,
    ABStreamingOperator,
    velocity_qubits_to_invert,
)
from qlbm.components.base import (
    LBMOperator,
    SequenceBlock,
    controlled,
    flip_if,
    on,
    x_layer,
)
from qlbm.components.common.adders import (
    ParameterizedDraperAdder,
    StreamingShift,
    shift_on,
)
from qlbm.components.common.arithmetic import RGQFTMultiplier
from qlbm.components.common.comparators import (
    SingleRegisterComparator,
    TwoRegisterComparator,
)
from qlbm.lattice.geometry.shapes import Block, Circle, YMonomial
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException, LatticeException
from qlbm.tools.utils import ComparatorMode, flatten, get_qubits_to_invert

logger = getLogger("qlbm")

# Diagonal velocities and the streaming direction (positive?) in each dimension.
DIAGONAL_VELOCITIES: Dict[int, Tuple[bool, ...]] = {
    5: (True, True),
    6: (False, True),
    7: (False, False),
    8: (True, False),
}


class ABZoneAgnosticReflectionOperator(LBMOperator):
    """
    Implements bounceback reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_dQ_q` discretizations.

    Uses a zone-agnostic approach that relies on the existence of an oracle that marks the
    basis states belonging to the inside of the solid geometry.
    For more details on the oracle, see :class:`.ABZoneAgnosticReflectionOracle`.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABZoneAgnosticReflectionOperator
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

        ABZoneAgnosticReflectionOperator(lattice).plot()

    """

    lattice: ABLattice

    shapes: Dict[str, List[Shape]]

    def __init__(
        self, lattice: ABLattice, shapes: Dict[str, List[Shape]] | None = None
    ) -> None:
        if shapes is None:
            if lattice.has_multiple_geometries():
                self.markered_shapes = lattice.geometries
                raise CircuitException(
                    "Multigeometry only currently supported for standard boundary condition imposition."
                )
            self.shapes = lattice.geometries[0]
        else:
            self.shapes = shapes

        supported_shapes = ["cuboid", "ymonomial"]
        if any(x.name() not in supported_shapes for x in flatten(self.shapes.values())):  # type: ignore
            raise CircuitException(
                f"Agnostic reflection operator only supports the following shapes: {supported_shapes}."
            )
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        if self.lattice.discretization not in [LatticeDiscretization.D2Q9]:
            raise LatticeException("AB reflection only currently supported in D2Q9")
        if self.lattice.has_multiple_geometries():
            self.place(self.__multi_geometry())
        else:
            self.place(self.__single_geometry_bounceback())
            self.place(self.__single_geometry_sr())

    def __single_geometry_bounceback(self) -> AnyBlock:
        if not self.shapes.get("bounceback"):
            return CompositeBlock([], self.n_qubits, name="ab_agnostic_bb")
        oracle = self.__build_oracle("bounceback")
        return CompositeBlock(
            [
                oracle,
                self.permute_and_stream_bounceback(),
                ABStreamingOperator(self.lattice, inverse=True),
                oracle,
                ABStreamingOperator(self.lattice),
            ],
            self.n_qubits,
            name="ab_agnostic_bb",
        )

    def __single_geometry_sr(self) -> AnyBlock:
        if not self.shapes.get("specular"):
            return CompositeBlock([], self.n_qubits, name="ab_agnostic_sr")
        oracle = self.__build_oracle("specular")
        return CompositeBlock(
            [
                # Step 1: Oracle (sets a_o)
                oracle,
                # Step 2: SR check (sets a_x, a_y for diagonals)
                ABZoneAgnosticSRCheck(
                    self.lattice,
                    self.lattice.discretization,
                    self.shapes["specular"],
                    check_negative_direction=True,
                ),
                # Step 3: Permutations (velocity changes, position unchanged)
                self.__apply_permutations_sr(),
                # Step 4: Dim-selective stream for diagonals (ctrl a_o AND a_{d+1})
                self.__dim_selective_stream(),
                # Step 5: Inverse stream
                ABStreamingOperator(self.lattice, inverse=True),
                # Step 6: SR check in the positive direction
                ABZoneAgnosticSRCheck(
                    self.lattice,
                    self.lattice.discretization,
                    self.shapes["specular"],
                    check_negative_direction=False,
                ),
                # Step 7: Oracle (unitarily uncomputes a_o)
                oracle,
                # Step 8: Stream
                ABStreamingOperator(self.lattice),
            ],
            self.n_qubits,
            name="ab_agnostic_sr",
        )

    def __build_oracle(self, boundary_condition: str) -> AnyBlock:
        return CompositeBlock(
            [
                ABZoneAgnosticReflectionOracle(self.lattice, shape)
                for shape in self.shapes[boundary_condition]
            ],
            self.n_qubits,
            name="ab_oracle",
        )

    def __multi_geometry(self) -> AnyBlock:
        oracle = self.build_combined_oracle("bounceback")
        return CompositeBlock(
            [
                oracle,
                self.permute_and_stream_bounceback(),
                ABStreamingOperator(self.lattice, inverse=True),
                oracle,
                ABStreamingOperator(self.lattice),
            ],
            self.n_qubits,
            name="ab_agnostic_multi",
        )

    def build_combined_oracle(self, boundary_condition: str) -> AnyBlock:
        """
        Build the oracle of every geometry, each controlled on its marker state.

        Parameters
        ----------
        boundary_condition : str
            Which boundary condition's shapes to include.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        children = []
        marker = self.lattice.marker_index()
        for c, shapes_for_geometry in enumerate(self.markered_shapes):
            qubits_to_invert = [
                marker[0] + q
                for q in get_qubits_to_invert(c, self.lattice.num_marker_qubits)
            ]
            if qubits_to_invert:
                children.append(x_layer(qubits_to_invert))
            children += [
                ABZoneAgnosticReflectionOracle(
                    self.lattice, shape, control_on_marker_state=True
                )
                for shape in shapes_for_geometry[boundary_condition]
            ]
            if qubits_to_invert:
                children.append(x_layer(qubits_to_invert))
        return CompositeBlock(children, self.n_qubits, name="ab_combined_oracle")

    def __apply_permutations_sr(self) -> AnyBlock:
        a_x = self.lattice.ancillae_obstacle_index(1)
        a_y = self.lattice.ancillae_obstacle_index(2)
        a_xy = self.lattice.ancillae_obstacle_index(3)
        all_obstacle = self.lattice.ancillae_obstacle_index()
        velocity = self.lattice.velocity_index()

        def specular(reflect_in_dim: Tuple[bool, ...]) -> AnyBlock:
            return ABSpecularReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                reflect_in_dim,
            )

        # Each permutation is selected by the obstacle ancillae (a_o, a_x, a_y, a_xy),
        # with the listed ancillae required to be |0>.
        cases = [
            (specular((True, False)), a_y + a_xy),  # x-wall hit
            (specular((False, True)), a_x + a_xy),  # y-wall hit
            (specular((True, True)), a_xy),  # corner hit, reflect both
            # Cardinal velocity, reflect both (BB equivalent)
            (
                ABBounceBackReflectionPermutation(
                    self.lattice.num_velocity_qubits,
                    self.lattice.discretization,
                    self.lattice.get_encoding(),
                ),
                a_x + a_y,
            ),
        ]
        return CompositeBlock(
            [
                controlled(
                    permutation,
                    all_obstacle,
                    velocity,
                    ctrl_state=[qubit not in open_controls for qubit in all_obstacle],
                )
                for permutation, open_controls in cases
            ],
            self.n_qubits,
            name="ab_agnostic_perm",
        )

    def __dim_selective_stream(self) -> AnyBlock:
        obstacle = self.lattice.ancillae_obstacle_index()
        last, middle = obstacle[-1], obstacle[1:-1]
        velocity = self.lattice.velocity_index()
        controls = obstacle + velocity
        children = []

        for dim, dim_population_to_update in enumerate(
            STREAMING_POPULATIONS[LatticeDiscretization.D2Q9]
        ):
            grid_index = self.lattice.grid_index(dim)
            wall_controls = (
                self.lattice.ancillae_obstacle_index(0)  # a_{o, 0}
                + self.lattice.ancillae_obstacle_index(dim + 1)  # a_{o, x/y}
                + [last]  # a_{o, xy}, required |0>
                + velocity
            )
            shifts = [
                shift_on(
                    direction == 0,
                    controls,
                    wall_controls,
                    [last] + velocity_qubits_to_invert(self.lattice, index),
                )
                for direction, indices in enumerate(dim_population_to_update)
                for index in indices
            ]
            # Diagonal velocities that hit a concave corner: control on |001> of (a_x, a_y, a_xy)
            shifts += [
                shift_on(
                    positive_in_dim[dim],
                    controls,
                    controls,
                    middle + velocity_qubits_to_invert(self.lattice, diagonal_velocity),
                )
                for diagonal_velocity, positive_in_dim in DIAGONAL_VELOCITIES.items()
            ]
            children.append(
                on(
                    StreamingShift(len(grid_index), len(controls), shifts),
                    controls + grid_index,
                )
            )
        return SequenceBlock(children, self.n_qubits, name="ab_agnostic_dim_stream")

    def permute_and_stream_bounceback(self) -> AnyBlock:
        """
        Build the block that bounces the marked velocities back and streams them.

        Returns
        -------
        AnyBlock
            The block over the full lattice width.
        """
        return SequenceBlock(
            [
                # Permute the velocities according to reflection rules
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
            name="ab_agnostic_permute_stream",
        )

    @override
    def __str__(self) -> str:
        return f"[Operator ABZoneAgnosticReflection with lattice {self.lattice}]"


class ABZoneAgnosticReflectionOracle(LBMOperator):
    r"""
    Implementation of the oracle required for :class:`.ABZoneAgnosticReflectionOperator`.

    An oracle is an operator :math:`U_{\Omega}` for an obstacle's region
    :math:`\Omega` such that, in the amplitude-based encoding,
    :math:`U_\Omega\ket{x}\ket{v}\ket{0}_\mathbb{o} = \ket{x}\ket{v}\ket{x \in \Omega}_\mathbb{o}`.
    Intuitively, the operator flips the object ancilla qubit if and only if the position :math:`x`
    falls within the bounds of the object.

    Currently, the only available implementation is for 2D axis-aligned :class:`.Block`
    and :class:`.YMonomial` objects.

    .. important::

        The ``YMonomial`` implementation is a work in progress.
        At present, only the :math:`x^2` monomial case is supported,
        and only when the monomial result register width matches the :math:`y`
        grid register width.

    This is an improvement in asymptotic and practical complexity compared to
    the methods described in :cite:`collisionless`.
    This operation relies on basic arithmetic through the :class:`.ParameterizedDraperAdder` class
    and comparison operation through the :class:`Comparator` circuits.

    When ``control_on_marker_state`` is ``True``, the oracle additionally conditions
    the obstacle ancilla flip on the marker register being in the all-ones state.
    This is used for parallel boundary conditions where multiple geometries
    are simulated on the same lattice, each identified by a marker state.
    Only the central MCX gate (for cuboids) is controlled on the marker,
    since the surrounding adder and comparator operations are self-inverse
    and their net effect on the grid register is zero.

    .. important::

        Marker-controlled oracles for :class:`.YMonomial` shapes are not yet supported.
        Passing ``control_on_marker_state=True`` with a ``YMonomial`` shape will raise
        a :class:`.CircuitException`.

    Example usage for a cuboid :class:`.Block`:

    .. code-block:: python

        from qlbm.components.ab.reflection import ABZoneAgnosticReflectionOracle
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 16}, "velocities": "d2q9"},
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

        ABZoneAgnosticReflectionOracle(
            lattice, shape=lattice.shapes["bounceback"][0]
        ).plot()

    And for a :class:`.YMonomial`:

    .. code-block:: python

        from qlbm.components.ab.reflection import ABZoneAgnosticReflectionOracle
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 16}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "ymonomial",
                        "exponent": 2,
                        "comparator": "<",
                        "boundary": "bounceback",
                    }
                ],
            }
        )

        ABZoneAgnosticReflectionOracle(
            lattice, shape=lattice.shapes["bounceback"][0]
        ).plot()


    """

    target_obstacle_index: int
    """Index within the obstacle ancilla register to target with the oracle flip."""

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        shape: Shape,
        control_on_marker_state: bool = False,
        additional_control_qubits: List[int] = [],
        target_obstacle_index: int = 0,
    ) -> None:
        self.shape = shape
        self.control_on_marker_state = control_on_marker_state
        self.additional_control_qubits = additional_control_qubits
        self.target_obstacle_index = target_obstacle_index
        if isinstance(shape, YMonomial):
            if control_on_marker_state:
                raise CircuitException(
                    "Marker-controlled oracles for YMonomial shapes are not yet supported. "
                    "Parallel boundary conditions with YMonomial geometries require a future extension."
                )
            self.__validate_ymonomial(lattice, shape)
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        match self.shape:
            case Block():
                self.__place_block()
            case Circle():
                raise CircuitException("Not implemented")
            case YMonomial():
                self.__place_ymonomial()
            case _:
                raise CircuitException(
                    f"Unsupported shape {self.shape} for the oracle."
                )

    def __place_block(self) -> None:
        block: Block = cast(Block, self.shape)
        comparator_ancillae = self.lattice.ancillae_comparator_index(0)

        def bounds_check(dim: int, positive: bool) -> List[AnyBlock]:
            # Shift the grid by the lower bound so the comparison is against the width
            grid_index = self.lattice.grid_index(dim)
            adder = on(
                ParameterizedDraperAdder(
                    len(grid_index), block.bounds[dim][0], positive
                ),
                grid_index,
            )
            comparator = on(
                SingleRegisterComparator(
                    num_qubits=len(grid_index) + 1,
                    num_to_compare=block.bounds[dim][1] - block.bounds[dim][0],
                    mode=ComparatorMode.LE,
                ),
                grid_index + [comparator_ancillae[dim]],
            )
            return [comparator, adder] if positive else [adder, comparator]

        for dim in range(self.lattice.num_dims):
            for child in bounds_check(dim, positive=False):
                self.place(child)

        control_qubits = (
            comparator_ancillae[: self.lattice.num_dims]
            + self.additional_control_qubits
        )
        if self.control_on_marker_state:
            control_qubits = control_qubits + self.lattice.marker_index()
        self.place(
            flip_if(
                control_qubits,
                self.lattice.ancillae_obstacle_index(self.target_obstacle_index)[:1],
            )
        )

        for dim in range(self.lattice.num_dims):
            for child in bounds_check(dim, positive=True):
                self.place(child)

    @staticmethod
    def __validate_ymonomial(lattice: ABLattice, ym: YMonomial) -> None:
        if ym.exponent != 2:
            raise CircuitException(
                "YMonomial oracle is a work in progress: only exponent=2 (x^2) is currently supported."
            )
        n_y = len(lattice.grid_index(1))
        n_monomial = len(lattice.ancillae_monomial_index())
        n_copy = len(lattice.ancillae_copy_index())
        if n_monomial != n_y:
            padding_needed = abs(n_monomial - n_y)
            if padding_needed > n_copy:
                raise CircuitException(
                    f"YMonomial oracle: register size mismatch requires "
                    f"{padding_needed} padding qubits but only {n_copy} "
                    f"copy-register qubits are available. "
                    f"Grid must satisfy |2*n_x - n_y| <= n_x."
                )
            logger.warning(
                "YMonomial oracle: monomial register (%d qubits) differs "
                "from y grid register (%d qubits). Using %d copy-register "
                "qubits as zero-padding for the comparison.",
                n_monomial,
                n_y,
                padding_needed,
            )

    def __place_ymonomial(self) -> None:
        ym: YMonomial = cast(YMonomial, self.shape)
        grid_x_qubits = self.lattice.grid_index(0)
        grid_y_qubits = self.lattice.grid_index(1)
        copy_qubits = self.lattice.ancillae_copy_index()
        result_qubits = self.lattice.ancillae_monomial_index()
        obstacle_qubits = self.lattice.ancillae_obstacle_index(
            self.target_obstacle_index
        )
        n_y, n_monomial = len(grid_y_qubits), len(result_qubits)

        # Copy x into the copy register; the same block undoes the copy.
        copy = SimpleBlock(self.n_qubits, name="copy_x")
        copy.cx(list(zip(grid_x_qubits, copy_qubits, strict=True)))
        multiplication_qubits = grid_x_qubits + copy_qubits + result_qubits

        self.place(copy)
        # Multiply x * copy -> result
        self.place(
            RGQFTMultiplier(len(grid_x_qubits), n_monomial), multiplication_qubits
        )

        if n_monomial != n_y:
            # Free the copy register by undoing the copy operation.  This
            # leaves all copy qubits in |0>, so a subset can serve as
            # zero-padding for the shorter register in the comparator, which
            # preserves both input registers; the copy is then restored.
            self.place(copy)
            padding_qubits = copy_qubits[: abs(n_monomial - n_y)]
            if n_monomial > n_y:
                # Pad y with zeros in the high bits
                comparator_x_reg = grid_y_qubits + padding_qubits
                comparator_y_reg = result_qubits
            else:
                # Pad result with zeros in the high bits
                comparator_x_reg = grid_y_qubits
                comparator_y_reg = result_qubits + padding_qubits
            self.place(
                TwoRegisterComparator(max(n_y, n_monomial), ym.comparator_mode),
                comparator_x_reg + comparator_y_reg + obstacle_qubits,
            )
            self.place(copy)
        else:
            self.place(
                TwoRegisterComparator(n_y, ym.comparator_mode),
                grid_y_qubits + result_qubits + obstacle_qubits,
            )

        # Undo multiplication and copy
        self.place(
            ~RGQFTMultiplier(len(grid_x_qubits), n_monomial), multiplication_qubits
        )
        self.place(copy)

    @override
    def __str__(self) -> str:
        return f"[Primitive ABZoneAgnosticReflectionOracle with lattice {self.lattice}, shape={self.shape}]"


class ABZoneAgnosticSRCheck(LBMOperator):
    r"""Determines which spatial dimensions caused a diagonal particle to enter the obstacle.

    For each dimension :math:`d`, this primitive:

    1. Unstreams only the diagonal velocities in dimension :math:`d`.
    2. Applies an oracle targeting :math:`a_{d+1}` to check whether the particle
       is still inside the obstacle after the partial unstream.
    3. Flips :math:`a_{d+1}` so that :math:`a_{d+1} = 1` means
       "dimension :math:`d` caused the entry" (i.e., unstreaming in :math:`d`
       took the particle outside the obstacle).
    4. Restreams the diagonal velocities to restore the original position.

    The oracle for each dimension is constructed internally and targets
    ``ancillae_obstacle_index(dim + 1)`` directly, avoiding the need for
    swap-based ancilla management.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab.reflection.agnosotic_reflection import (
            ABZoneAgnosticSRCheck,
        )
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [1, 3],
                        "y": [1, 3],
                        "boundary": "specular",
                    }
                ],
            }
        )

        ABZoneAgnosticSRCheck(
            lattice, lattice.discretization, lattice.shapes["specular"]
        ).plot()

    """

    sr_velocities_to_unstream: Dict[
        LatticeDiscretization, Dict[int, Tuple[bool | None, ...]]
    ] = {
        LatticeDiscretization.D2Q9: {
            5: (True, True),
            6: (False, True),
            7: (False, False),
            8: (True, False),
            1: (True, None),
            2: (None, True),
            3: (False, None),
            4: (None, False),
        }
    }

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        discretization: LatticeDiscretization,
        shapes: List[Shape],
        check_negative_direction: bool = False,
        additional_control_qubit_indices: List[int] = [],
    ) -> None:
        if discretization not in self.sr_velocities_to_unstream:
            raise LatticeException(
                f"Specular Reflection BCs not supported for {discretization}"
            )
        self.discretization = discretization
        self.shapes = shapes
        self.additional_control_qubit_indices = additional_control_qubit_indices
        self.check_negative_direction = check_negative_direction
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        obstacle = self.lattice.ancillae_obstacle_index()
        # If we are inside the obstacle, but neither x nor y contributed
        # individually to getting here, then it must have been their combination.
        combination = flip_if(obstacle[:-1], [obstacle[-1]], obstacle[1:-1])

        if not self.check_negative_direction:
            self.place(combination)
        for dim in range(self.lattice.num_dims):
            self.__place_streaming(dim, unstream=True)
            # After unstreaming dimension dim, the oracle checks whether the
            # particle is still inside; if so a_{dim+1} is set.  The X gate
            # inverts the meaning: a_{dim+1} = 1 means unstreaming took the
            # particle OUT, i.e. dimension dim caused the entry.
            self.place(self.__build_oracle_for_dim(dim))
            self.place(x_layer(self.lattice.ancillae_obstacle_index(dim + 1)))
            self.__place_streaming(dim, unstream=False)
        if self.check_negative_direction:
            self.place(combination)

    def __place_streaming(self, dim: int, unstream: bool) -> None:
        grid_index = self.lattice.grid_index(dim)
        controls = self.additional_control_qubit_indices + self.lattice.velocity_index()
        shifts = []
        for velocity_idx, vel_signs in self.sr_velocities_to_unstream[
            self.discretization
        ].items():
            sign = vel_signs[dim]
            if sign is None:
                continue
            # Unstream = reverse the streaming direction for this dim, restream = original
            positive = sign ^ (not unstream) ^ self.check_negative_direction
            shifts.append(
                shift_on(
                    positive,
                    controls,
                    controls,
                    velocity_qubits_to_invert(self.lattice, velocity_idx),
                )
            )
        self.place(
            StreamingShift(len(grid_index), len(controls), shifts),
            controls + grid_index,
        )

    def __build_oracle_for_dim(self, dim: int) -> AnyBlock:
        return CompositeBlock(
            [
                ABZoneAgnosticReflectionOracle(
                    self.lattice,
                    shape,
                    additional_control_qubits=self.lattice.ancillae_obstacle_index(0),
                    target_obstacle_index=dim + 1,
                )
                for shape in self.shapes
            ],
            self.n_qubits,
            name="ab_sr_check_oracle",
        )

    @override
    def __str__(self):
        return f"[Primitive ABZoneAgnosticSRCheck with lattice {self.lattice}]"
