"""Streaming operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from typing import List, Tuple

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class LQLGAStreamingOperator(LatticePrimitive):
    # TODO: Improve documentation
    """
    Streaming operator for the :class:`.LQLGA` algorithm.

    Streaming is implemented by a series of swap gates as described in :cite:`spacetime`.
    The number of gates scales linearly with size of the grid,
    while the depth scales logarithmically.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.lqlga import LQLGAStreamingOperator
        from qlbm.lattice import LQLGALattice

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 4},
                    "velocities": "D1Q3",
                },
                "geometry": [],
            },
        )
        streaming_operator = LQLGAStreamingOperator(lattice)
        streaming_operator.plot()
    """

    lattice: LQLGALattice

    def __init__(self, lattice: LQLGALattice) -> None:
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        # ! TODO Generalize in 2 and 3D
        num_gps = self.lattice.num_gridpoints[0] + 1
        for direction, velocity_qubit_of_line in enumerate(
            self.lattice.get_velocity_qubits_of_line(0)
        ):
            for layer in self.logarithmic_depth_streaming_line_swaps(
                num_gps, negative_direction=bool(direction)
            ):
                self.swap(
                    [
                        (
                            self.lattice.velocity_index_flat(i, velocity_qubit_of_line),
                            self.lattice.velocity_index_flat(j, velocity_qubit_of_line),
                        )
                        for i, j in layer
                    ]
                )

    def logarithmic_depth_streaming_line_swaps(
        self, num_gridpoints: int, negative_direction: bool
    ) -> List[List[Tuple[int, int]]]:
        """
        Compute the swap layers that stream one velocity line in logarithmic depth.

        Parameters
        ----------
        num_gridpoints : int
            The number of gridpoints along the line.
        negative_direction : bool
            Whether the line streams towards decreasing indices.

        Returns
        -------
        List[List[Tuple[int, int]]]
            The layers of gridpoint pairs to swap, in application order.
        """
        if num_gridpoints < 2:
            return []
        layers: List[List[Tuple[int, int]]] = []
        stride = 1
        while stride < num_gridpoints:
            layer: List[Tuple[int, int]] = []
            for i in range(0, num_gridpoints, 2 * stride):
                if i + stride < num_gridpoints:
                    layer.append(
                        (i, i + stride)
                        if not negative_direction
                        else (num_gridpoints - 1 - i, num_gridpoints - 1 - i - stride)
                    )
            layers.append(layer)
            stride *= 2
        return layers

    @override
    def __str__(self):
        return f"[LQLGAStreamingOperator on lattice={self.lattice}]"
