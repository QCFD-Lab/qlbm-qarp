"""Quantum circuits used for setting the initial state in the :class:`ABQLBM` algorithm."""

from typing import List, Tuple

from qarp.blocks import HnBlock
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.utils import BinaryToOHPermutation
from qlbm.components.base import LBMOperator, controlled
from qlbm.components.common.primitives import (
    AdditionConversion,
    StateSetter,
    UniformStatePrep,
)
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.tools.exceptions import CircuitException, LatticeException
from qlbm.tools.utils import dimension_letter


def _validate_configuration(
    lattice: ABLattice,
    velocity_indices: List[int],
    grid_qubits_to_superpose: Tuple[List[int], ...],
) -> None:
    if any(v not in range(lattice.num_velocities_per_point) for v in velocity_indices):
        raise LatticeException(
            f"Velocity indices should be in the interval 0..{lattice.num_velocities_per_point}"
        )
    if len(grid_qubits_to_superpose) != lattice.num_dims:
        raise LatticeException(
            f"Lattice has {lattice.num_dims} dimensions, but provided grid qubit information has {len(grid_qubits_to_superpose)} entries."
        )
    for dim in range(lattice.num_dims):
        if any(
            q not in range(lattice.num_gridpoints[dim].bit_length())
            for q in grid_qubits_to_superpose[dim]
        ):
            raise LatticeException(
                f"Grid qubit specification in dimension {dimension_letter(dim)} out of range."
            )


def _conversions(velocity_indices: List[int]) -> List[Tuple[int, int]]:
    """The (from, to) basis-state conversions that move the uniform superposition onto ``velocity_indices``."""
    states_from = list(range(len(velocity_indices)))
    states_to = velocity_indices.copy()
    # Remove indices that are already in place
    for v in velocity_indices:
        if v < len(velocity_indices):
            states_from.remove(v)
            states_to.remove(v)
    return list(zip(states_from, states_to, strict=True))


class ABInitialConditions(LBMOperator):
    """
    Initial conditions for the :class:`ABQLBM` algorithm.

    This component creates an equal magnitude superposition of all velocity
    basis states using the :class:`.UniformStatePrep`. Its spatial state depends
    on the lattice encoding:

    * With an :class:`.ABLattice`, ``x`` remains 0 and every ``y`` coordinate is
      placed in superposition. For a ``16 x 8`` grid, the populated points are
      therefore ``(0, 0)`` through ``(0, 7)``.
    * With an :class:`.OHLattice`, the grid register remains at ``(0, 0)``.

    When the lattice contains multiple geometries, the component also places
    every marker qubit in superposition.

    .. warning::

        If the number of geometries is not a power of two, applying a Hadamard
        gate to every marker qubit also creates marker states with no associated
        geometry. Use :class:`.ABParallelDiscreteUniformInitialConditions` when
        each marker state needs an explicitly configured initial condition.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABInitialConditions(lattice).plot()
    """

    lattice: ABLattice

    def __init__(self, lattice: ABLattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        self.place(
            UniformStatePrep(
                self.lattice.num_velocity_qubits, self.lattice.num_velocities_per_point
            ),
            self.lattice.velocity_index()[: self.lattice.num_velocity_qubits],
        )
        match self.lattice.get_encoding():
            case ABEncodingType.AB:
                grid_index = self.lattice.grid_index(1)
                self.place(HnBlock(len(grid_index)), grid_index)
            case ABEncodingType.OH:
                self.place(
                    BinaryToOHPermutation(self.lattice), self.lattice.velocity_index()
                )
            case _:
                raise LatticeException(
                    f"Encoding {self.lattice.get_encoding()} not supported."
                )
        if self.lattice.has_multiple_geometries():
            marker = self.lattice.marker_index()
            self.place(HnBlock(len(marker)), marker)

    @override
    def __str__(self) -> str:
        return f"[Primitive ABEInitialConditions with lattice {self.lattice}]"


class ABDiscreteUniformInitialConditions(LBMOperator):
    """
    Initial conditions for the :class:`ABQLBM` algorithm.

    This component creates an equal magnitude superposition of a configurable
    set of velocity and grid indices. The selected velocities are alternative
    basis states; they do not represent several independently occupied channels
    at the same grid point.

    ``velocity_indices`` lists zero-based velocity channels. The tuple
    ``grid_qubits_to_superpose`` lists zero-based coordinate bit positions for
    each dimension (x, then y, then z when present); selected bits vary between
    0 and 1, while unselected bits remain 0. Thus ``[1, 3, 4], ([], [])``
    prepares channels 1 (+x), 3 (-x), and 4 (-y) at grid point (0, 0). On a 2D
    lattice, ``[1, 3, 4], ([0, 1], [0])`` prepares the same velocity channels
    over x=0..3 and y=0..1.

    .. warning::

        ``grid_qubits_to_superpose`` does not accept coordinate ranges and
        cannot select an arbitrary interval. Selecting bit positions
        ``Q`` produces coordinates of the form
        ``sum(b_q * 2**q for q in Q)``, where each ``b_q`` is 0 or 1.
        For example, ``[0, 1]`` produces coordinates 0, 1, 2, and 3, but no
        bit-position list produces exactly the interval 2 through 7. Preparing
        such an interval requires a separate state-preparation circuit.

        ``velocity_indices`` must be nonempty and contain unique indices. Each
        index must be between 0 and ``num_velocities_per_point - 1``.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABDiscreteUniformInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        ABDiscreteUniformInitialConditions(lattice, [1, 3, 4], ([], [])).plot()

    The primitive can also be applied to the :class:`.OHLattice`. The same
    channel indices are used, but each selected channel becomes a separate
    one-hot basis state:

    .. code-block:: python

        from qlbm.components.ab import ABDiscreteUniformInitialConditions
        from qlbm.lattice import OHLattice

        lattice = OHLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        ABDiscreteUniformInitialConditions(lattice, [0, 1], ([0, 1], [0])).plot()
    """

    velocity_indices: List[int]

    grid_qubits_to_superpose: Tuple[List[int], ...]

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        velocity_indices: List[int],
        grid_qubits_to_superpose: Tuple[List[int], ...],
    ) -> None:
        _validate_configuration(lattice, velocity_indices, grid_qubits_to_superpose)
        self.velocity_indices = sorted(velocity_indices)
        self.grid_qubits_to_superpose = grid_qubits_to_superpose
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        num_states = len(self.velocity_indices)
        nq = (num_states - 1).bit_length()
        self.place(UniformStatePrep(nq, num_states), self.lattice.velocity_index()[:nq])
        for v_from, v_to in _conversions(self.velocity_indices):
            self.place(
                AdditionConversion(self.lattice.num_velocity_qubits, v_from, v_to),
                # Additional guard necessary of OH
                self.lattice.velocity_index()[: self.lattice.num_velocities_per_point]
                + self.lattice.ancillae_obstacle_index(0),
            )
        if self.lattice.get_encoding() == ABEncodingType.OH:
            self.place(
                BinaryToOHPermutation(self.lattice), self.lattice.velocity_index()
            )
        for dim in range(self.lattice.num_dims):
            if self.grid_qubits_to_superpose[dim]:
                qubits = [
                    self.lattice.grid_index(dim)[0] + q
                    for q in self.grid_qubits_to_superpose[dim]
                ]
                self.place(HnBlock(len(qubits)), qubits)

    @override
    def __str__(self) -> str:
        return f"[Primitive ABDiscreteUniformInitialConditions with lattice {self.lattice}, v={self.velocity_indices}, g={self.grid_qubits_to_superpose}]"


class ABParallelDiscreteUniformInitialConditions(LBMOperator):
    """
    Marker-sensitive initial conditions for the :class:`ABQLBM` algorithm.

    This component creates marker-sensitive initial conditions for parallel
    configurations. Entry ``i`` of ``velocity_indices_list`` and
    ``grid_qubits_to_superpose_list`` is prepared when the marker register is in
    basis state ``|i>``. For example, with two marker qubits, entries 0 through 3
    correspond to ``|00>``, ``|01>``, ``|10>``, and ``|11>``, respectively.

    Each grid tuple uses the same bit-position convention and has the same range
    restrictions as :class:`.ABDiscreteUniformInitialConditions`. Each velocity
    list must be nonempty and contain unique channel indices.

    .. warning::

        Parallel initial conditions support :class:`.ABLattice` only. The number
        of configurations cannot exceed the number of marker basis states,
        ``2**lattice.num_marker_qubits``.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABParallelDiscreteUniformInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        lattice.set_num_marker_qubits(2)

        ABParallelDiscreteUniformInitialConditions(
            lattice,
            [[0, 1], [0, 3], [0], [0, 5]],
            [([0], [0])] * 4,
        ).plot()

    In this example, all four configurations cover grid points ``(0, 0)``,
    ``(1, 0)``, ``(0, 1)``, and ``(1, 1)``. Marker states ``|00>``, ``|01>``,
    ``|10>``, and ``|11>`` select velocity lists ``[0, 1]``, ``[0, 3]``,
    ``[0]``, and ``[0, 5]``, respectively.

    """

    velocity_indices: List[List[int]]

    grid_qubits_to_superpose: List[Tuple[List[int], ...]]

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        velocity_indices_list: List[List[int]],
        grid_qubits_to_superpose_list: List[Tuple[List[int], ...]],
    ) -> None:
        if lattice.get_encoding() == ABEncodingType.OH:
            raise LatticeException(
                "OHLattice does not currently support parallel initial conditions."
            )
        if len(velocity_indices_list) != len(grid_qubits_to_superpose_list):
            raise CircuitException("Input lists have mismatched lengths.")
        if len(velocity_indices_list) > 2**lattice.num_marker_qubits:
            raise LatticeException(
                f"{lattice.num_marker_qubits} cannot encode {len(velocity_indices_list)} configurations."
            )
        for velocity_indices, grid_qubits_to_superpose in zip(
            velocity_indices_list, grid_qubits_to_superpose_list, strict=True
        ):
            _validate_configuration(lattice, velocity_indices, grid_qubits_to_superpose)
        self.velocity_indices_list = [
            sorted(velocity_indices) for velocity_indices in velocity_indices_list
        ]
        self.grid_qubits_to_superpose_list = grid_qubits_to_superpose_list
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        marker = self.lattice.marker_index()
        num_marker_qubits = self.lattice.num_marker_qubits

        # Uniform superposition over the marker index
        self.place(
            UniformStatePrep(num_marker_qubits, len(self.velocity_indices_list)), marker
        )

        for marker_index, (velocity_indices, grid_qubits_to_superpose) in enumerate(
            zip(
                self.velocity_indices_list,
                self.grid_qubits_to_superpose_list,
                strict=True,
            )
        ):
            num_states = len(velocity_indices)
            nq = (num_states - 1).bit_length()
            # Everything below is controlled on the marker register holding marker_index
            self.place(StateSetter(num_marker_qubits, marker_index), marker)
            self.place(
                UniformStatePrep(nq, num_states, num_ctrl_qubits=num_marker_qubits),
                self.lattice.velocity_index()[:nq] + marker,
            )
            for v_from, v_to in _conversions(velocity_indices):
                self.place(
                    AdditionConversion(
                        self.lattice.num_velocity_qubits,
                        v_from,
                        v_to,
                        num_ctrl_qubits=num_marker_qubits,
                    ),
                    # Additional guard necessary of OH
                    self.lattice.velocity_index()[
                        : self.lattice.num_velocities_per_point
                    ]
                    + self.lattice.ancillae_obstacle_index(0)
                    + marker,
                )
            for dim in range(self.lattice.num_dims):
                if grid_qubits_to_superpose[dim]:
                    qubits = [
                        self.lattice.grid_index(dim)[0] + q
                        for q in grid_qubits_to_superpose[dim]
                    ]
                    self.place(controlled(HnBlock(len(qubits)), marker, qubits))
            self.place(StateSetter(num_marker_qubits, marker_index), marker)

    @override
    def __str__(self) -> str:
        return f"[Primitive ABParallelDiscreteUniformInitialConditions with lattice {self.lattice}, v={self.velocity_indices_list}, g={self.grid_qubits_to_superpose_list}]"
