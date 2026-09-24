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
    basis states at position ``(0, 0)`` using the :class:`.UniformStatePrep`.

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

    This component creates an equal magnitude superposition of a configurable set of velocity and grid indices.

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

    The primitive can also applied to the :class:`.OHLattice`:

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

    This component creates an equal magnitude superposition of a configurable set of velocity and grid indices,
    entangled with the state of the marker register.
    Used in parallel realizations of configurations.

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
