"""Prepares the initial state for the :class:`.SpaceTimeQLBM` one gridpoint at a time."""

from typing import List, Tuple

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import VonNeumannNeighbor
from qlbm.tools.utils import bit_value, flatten


class PointWiseSpaceTimeInitialConditions(LatticePrimitive):
    r"""Prepares the initial state for the :class:`.SpaceTimeQLBM`.

    Initial conditions are supplied in a ``List[Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]]``
    containing, for each population to be initialized, two nested tuples.

    The first tuple position of the population(s) on the grid (i.e., ``(2, 5)``).
    The second tuple contains velocity of the population(s) at that location.
    Since the maximum number of velocities is pre-determined and the computational basis state encoding favors boolean logic,
    the input is provided as a tuple of ``boolean``\ s.
    That is, ``(True, True, False, False)`` would mean there are two populations at the same gridpoint,
    with velocities :math:`q_0` and :math:`q_1` according to the :math:`D_2Q_4` discretization.
    Together, the ``grid_data`` argument of the constructor can be supplied as, for instance, ``[((3, 7), (False, True, False, True))]``.

    The initialization follows the following steps:

    * For each (position, velocity) pair:
        #. Set the grid qubits encoding the position to :math:`\ket{1}^{\otimes n_g}` using :math:`X` gates;
        #. Set each of the toggled velocities to :math:`\ket{1}` by means of :math:`MCX` gates, controlled on the qubits set in the previous step;
        #. Undo the operation of step 1 (i.e., repeat the :math:`X` gates);
        #. Repeat steps 1-3 for all neighboring velocity qubits, adjusting for grid position and relative velocity index.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`grid_data`         The information encoding the particle probability distribution, formatted as (position, velocity) tuples.
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime.initial import PointWiseSpaceTimeInitialConditions
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
                "geometry": [],
            },
        )

        # Draw the initial conditions for two particles at (3, 7), traveling in the +y and -y directions
        PointWiseSpaceTimeInitialConditions(lattice=lattice, grid_data=[((3, 7), (False, True, False, True))]).plot()
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        grid_data: List[Tuple[Tuple[int, ...], Tuple[bool, ...]]] = [
            ((2, 5), (True, True, True, True)),
            ((3, 4), (False, True, False, True)),
        ],
        filter_inside_blocks: bool = True,
    ) -> None:
        self.grid_data = grid_data
        self.filter_inside_blocks = filter_inside_blocks
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        self.h(self.lattice.grid_index())

        for point_coordinates, velocity_values in self.grid_data:
            if self.filter_inside_blocks and self.lattice.is_inside_an_obstacle(
                point_coordinates
            ):
                continue
            # Set the velocity state for the origin
            self.set_point_velocity(point_coordinates, velocity_values, 0)

            # Set the velocity state for neighbors in increasing velocity
            for manhattan_distance in range(1, self.lattice.num_timesteps + 1):
                for neighbor in self.lattice.extreme_point_indices[manhattan_distance]:
                    self.set_neighbor_velocity(
                        point_coordinates, velocity_values, neighbor
                    )

                # No intermediate points at Manhattan distance 1
                # Or in 1D
                if manhattan_distance < 2 or self.lattice.num_dims < 2:
                    continue

                for neighbor in flatten(
                    list(
                        self.lattice.intermediate_point_indices[
                            manhattan_distance
                        ].values()
                    )
                ):
                    self.set_neighbor_velocity(
                        point_coordinates, velocity_values, neighbor
                    )

    def set_grid_value(self, point_coordinates: Tuple[int, ...]) -> None:
        r"""
        Emit the :math:`X` gates that map the grid state ``point_coordinates`` onto :math:`\ket{1\ldots1}`.

        Parameters
        ----------
        point_coordinates : Tuple[int, ...]
            The gridpoint to select.
        """
        qubits_to_invert = [
            self.lattice.grid_index(dim)[0] + qubit_index
            for dim, num_gp in enumerate(self.lattice.num_gridpoints)
            for qubit_index in range(num_gp.bit_length())
            if not bit_value(point_coordinates[dim], qubit_index)
        ]
        if qubits_to_invert:
            self.x(qubits_to_invert)

    def set_point_velocity(
        self,
        point_coordinates: Tuple[int, ...],
        velocity_values: Tuple[bool, ...],
        neighbor_index: int,
    ) -> None:
        """
        Emit the gates that enable ``velocity_values`` in the velocity register ``neighbor_index`` of gridpoint ``point_coordinates``.

        Parameters
        ----------
        point_coordinates : Tuple[int, ...]
            The gridpoint whose grid state controls the operation.
        velocity_values : Tuple[bool, ...]
            Which velocities to enable.
        neighbor_index : int
            The velocity register (stencil position) to set.
        """
        self.set_grid_value(point_coordinates)
        for target_qubit in flatten(
            [
                self.lattice.velocity_index(neighbor_index, c)
                for c, is_velocity_enabled in enumerate(velocity_values)
                if is_velocity_enabled
            ]
        ):
            self.mcx(*self.lattice.grid_index(), target_qubit)
        self.set_grid_value(point_coordinates)

    def set_neighbor_velocity(
        self,
        point_coordinates: Tuple[int, ...],
        velocity_values: Tuple[bool, ...],
        neighbor: VonNeumannNeighbor,
    ) -> None:
        """
        Emit the gates that set the velocities of ``neighbor`` relative to ``point_coordinates``.

        Parameters
        ----------
        point_coordinates : Tuple[int, ...]
            The origin gridpoint.
        velocity_values : Tuple[bool, ...]
            Which velocities to enable.
        neighbor : VonNeumannNeighbor
            The stencil neighbor whose register is set.
        """
        absolute_neighbor_coordinates = neighbor.get_absolute_values(
            point_coordinates, relative=False
        )
        if self.filter_inside_blocks and self.lattice.is_inside_an_obstacle(
            absolute_neighbor_coordinates
        ):
            return
        self.set_point_velocity(
            absolute_neighbor_coordinates, velocity_values, neighbor.neighbor_index
        )

    @override
    def __str__(self) -> str:
        return f"[Primitive PointWiseSpaceTimeInitialConditions with data={self.grid_data} and lattice={self.lattice}]"
