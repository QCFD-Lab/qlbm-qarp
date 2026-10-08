"""Streaming operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from typing import List

from typing_extensions import override

from qlbm.components.base import LatticePrimitive
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException


class SpaceTimeStreamingOperator(LatticePrimitive):
    """An operator that performs streaming as a series of :math:`SWAP` gates as part of the :class:`.SpaceTimeQLBM` algorithm.

    The velocities corresponding to neighboring gridpoints are streamed "into" the gridpoint affected relative to the ``timestep``.
    The register setup of the :class:`.SpaceTimeLattice` is such that following each
    time step, an additional "layer" neighboring velocity qubits can be discarded,
    since the information they encode can never reach the relative origin in the remaining number of time steps.
    As such, the complexity of the streaming operator decreases with the number of steps (still) to be simulated.
    For an in-depth mathematical explanation of the procedure, consult pages 15-18 of :cite:t:`spacetime`.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`timestep`          The time step for which to perform streaming.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime import SpaceTimeStreamingOperator
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
                "geometry": [],
            },
        )

        # Draw the streaming operator for 1 time step
        SpaceTimeStreamingOperator(lattice=lattice, timestep=1).plot()
    """

    lattice: SpaceTimeLattice

    def __init__(self, lattice: SpaceTimeLattice, timestep: int) -> None:
        if timestep < 1 or timestep > lattice.num_timesteps:
            raise CircuitException(
                f"Invalid time step {timestep}, select a value between 1 and {lattice.num_timesteps}"
            )
        self.timestep = timestep
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        properties = self.lattice.properties
        match properties.get_discretization():
            case LatticeDiscretization.D1Q2:
                # (dimension, positive direction) -> velocity direction index
                schedule = [(0, True, 0), (0, False, 1)]
            case LatticeDiscretization.D2Q4:
                schedule = [(0, True, 0), (0, False, 2), (1, True, 1), (1, False, 3)]
            case discretization:
                raise CircuitException(
                    f"Streaming Operator unsupported for {discretization}."
                )
        for dim, positive, velocity_direction in schedule:
            self.stream_lines(
                properties.get_streaming_lines(dim, positive, self.timestep),
                velocity_direction,
            )

    def stream_lines(
        self, streaming_lines: List[List[int]], velocity_direction: int
    ) -> None:
        """
        Emit the swaps that stream one velocity direction along ``streaming_lines``.

        Parameters
        ----------
        streaming_lines : List[List[int]]
            The gridpoint neighbor indices of each line, in streaming order.
        velocity_direction : int
            The velocity direction to stream.
        """
        pairs = [
            (
                self.lattice.velocity_index(neighbor, velocity_direction)[0],
                self.lattice.velocity_index(next_neighbor, velocity_direction)[0],
            )
            for streaming_line in streaming_lines
            for neighbor, next_neighbor in zip(
                streaming_line, streaming_line[1:], strict=False
            )
        ]
        if pairs:
            self.swap(pairs)

    @override
    def __str__(self) -> str:
        return f"[SpaceTimeStreamingOperator for lattice {self.lattice}]"
