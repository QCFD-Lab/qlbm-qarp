"""Prepares the initial state for the :class:`.SpaceTimeQLBM` for a volumetric region."""

from typing import List, Tuple, cast

from qarp.blocks import HnBlock
from typing_extensions import override

from qlbm.components.base import LBMOperator, flip_if
from qlbm.components.common.comparators import SingleRegisterComparator
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import VonNeumannNeighbor
from qlbm.tools.utils import ComparatorMode, flatten


def stencil_neighbors(lattice: SpaceTimeLattice, manhattan_distance: int) -> List:
    """
    The stencil positions at ``manhattan_distance`` from the origin (the origin itself at 0).

    Parameters
    ----------
    lattice : SpaceTimeLattice
        The lattice whose stencil is walked.
    manhattan_distance : int
        The layer of the stencil.

    Returns
    -------
    List
        The :class:`.VonNeumannNeighbor` objects of the layer.
    """
    if manhattan_distance == 0:
        return [lattice.properties.origin]
    return lattice.extreme_point_indices[manhattan_distance] + (
        flatten(list(lattice.intermediate_point_indices[manhattan_distance].values()))
        if manhattan_distance in lattice.intermediate_point_indices
        else []
    )


def volume_bounds_at(
    lattice: SpaceTimeLattice,
    cuboid_bounds: List[Tuple[int, int]],
    neighbor: VonNeumannNeighbor,
) -> List[Tuple[Tuple[int, int], Tuple[bool, bool]]]:
    """
    The periodic comparator bounds of ``cuboid_bounds`` shifted to the stencil position ``neighbor``.

    Parameters
    ----------
    lattice : SpaceTimeLattice
        The lattice providing the periodic wrap-around.
    cuboid_bounds : List[Tuple[int, int]]
        The cuboid bounds per dimension.
    neighbor : VonNeumannNeighbor
        The stencil position whose relative coordinates offset the bounds.

    Returns
    -------
    List[Tuple[Tuple[int, int], Tuple[bool, bool]]]
        Per dimension, the shifted bounds and whether each wraps around.
    """
    return lattice.comparator_periodic_volume_bounds(
        cast(
            List[Tuple[int, int]],
            [
                tuple(
                    dim_bound[i] + neighbor.coordinates_relative[dim]
                    for i in range(len(dim_bound))
                )
                for dim, dim_bound in enumerate(cuboid_bounds)
            ],
        )
    )


def volume_comparators(
    lattice: SpaceTimeLattice,
    periodic_volume_bounds: List[Tuple[Tuple[int, int], Tuple[bool, bool]]],
) -> List[Tuple[SingleRegisterComparator, List[int]]]:
    """
    The lower- and upper-bound comparators of a volume, each with the qubits it is placed on.

    Parameters
    ----------
    lattice : SpaceTimeLattice
        The lattice whose grid and comparator ancillae are addressed.
    periodic_volume_bounds : List[Tuple[Tuple[int, int], Tuple[bool, bool]]]
        The bounds per dimension, as :func:`volume_bounds_at` returns them.

    Returns
    -------
    List[Tuple[SingleRegisterComparator, List[int]]]
        The comparators in placement order, with their qubits.
    """
    return [
        (
            SingleRegisterComparator(
                lattice.properties.get_num_grid_qubits() + 1,
                pvb[0][bound],
                ComparatorMode.LE if bound else ComparatorMode.GE,
            ),
            lattice.grid_index() + [lattice.ancilla_comparator_index(dim)[bound]],
        )
        for dim, pvb in enumerate(periodic_volume_bounds)
        for bound in [False, True]
    ]


class VolumetricSpaceTimeInitialConditions(LBMOperator):
    """
    Prepares the initial state for the :class:`.SpaceTimeQLBM` for a volumetric region.

    Work in progress.
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        cuboid_bounds: List[Tuple[int, int]],
        velocity_profile: Tuple[int, ...],
    ) -> None:
        self.cuboid_bounds = cuboid_bounds
        self.velocity_profile = velocity_profile
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        grid_index = self.lattice.grid_index()
        self.place(HnBlock(len(grid_index)), grid_index)

        for manhattan_distance in range(self.lattice.num_timesteps + 1):
            for neighbor in stencil_neighbors(self.lattice, manhattan_distance):
                periodic_volume_bounds = volume_bounds_at(
                    self.lattice, self.cuboid_bounds, neighbor
                )
                comparators = volume_comparators(self.lattice, periodic_volume_bounds)
                # The sum of the profile is the number of velocities set to true.
                target_qubits = flatten(
                    [
                        self.lattice.velocity_index(neighbor.neighbor_index, c)
                        for c, is_velocity_enabled in enumerate(self.velocity_profile)
                        if is_velocity_enabled
                    ]
                )

                # The comparators compute the volume membership on their
                # ancillae and uncompute it after the velocities are set.
                for comparator, qubits in comparators:
                    self.place(comparator, qubits)
                for (
                    control_qubit_sequence
                ) in self.lattice.volumetric_ancilla_qubit_combinations(
                    [any(pvb[1]) for pvb in periodic_volume_bounds]
                ):
                    self.place(flip_if(control_qubit_sequence, target_qubits))
                for comparator, qubits in comparators:
                    self.place(comparator, qubits)

    @override
    def __str__(self) -> str:
        return f"[Primitive VolumetricSpaceTimeInitialConditions with range={self.cuboid_bounds}, profile = {self.velocity_profile}, and lattice={self.lattice}]"
