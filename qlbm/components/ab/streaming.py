"""Quantum circuits used for streaming in the :class:`ABQLBM` algorithm."""

from typing import List, Sequence, Tuple

from qarp.blocks import AnyBlock
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.base import LBMOperator
from qlbm.components.common.adders import ShiftTerm, StreamingShift, shift_on
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import get_qubits_to_invert

# Velocity indices that stream in the positive and negative direction of each dimension.
STREAMING_POPULATIONS = {
    LatticeDiscretization.D1Q3: [[[1], [2]]],
    LatticeDiscretization.D2Q9: [
        [[1, 5, 8], [3, 6, 7]],  # f1, f5, f8 x <- x + 1; f3, f6, f7 x <- x - 1
        [[2, 5, 6], [4, 7, 8]],  # f2, f5, f6 y <- y + 1; f4, f7, f8 y <- y - 1
    ],
}


def velocity_qubits_to_invert(
    lattice: AmplitudeLattice, velocity_index: int
) -> List[int]:
    r"""
    The velocity qubits that are :math:`\ket{0}` in the binary encoding of ``velocity_index``.

    Parameters
    ----------
    lattice : AmplitudeLattice
        The lattice whose velocity register is addressed.
    velocity_index : int
        The velocity whose encoding is inspected.

    Returns
    -------
    List[int]
        Global qubit indices.
    """
    velocity_register = lattice.velocity_index()
    return [
        velocity_register[qubit]
        for qubit in get_qubits_to_invert(velocity_index, lattice.num_velocity_qubits)
    ]


class ABStreamingOperator(LBMOperator):
    """
    Streaming operator for the :class:`ABQLBM` algorithm.

    Uses a variant of the Draper adder described in :cite:`collisionless`.
    The operator works by applying QFTs in parallel to each dimension of the grid,
    followed by phase gates that perform incrementation in the Fourier basis, and, finally,
    by applying an inverse QFT mapping the qubits back to the computational basis.

    Populations are streamed one after the other in Fourier space
    by controlling phase gates on the state of the velocity qubits.
    Additional controls qubits can be specified to restrict this operation.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABStreamingOperator
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABStreamingOperator(lattice).plot()

    """

    lattice: AmplitudeLattice
    """The lattice to construct the component for."""

    additional_control_qubit_indices: List[int]
    """The qubits (if any) that streaming should be controlled over.
    This makes the operator useful for the application of boundary conditions.
    Controls need only be applied to the phase gates and not the QFT blocks."""

    inverse: bool
    """Whether to stream every population backwards, undoing the forward operator."""

    def __init__(
        self,
        lattice: AmplitudeLattice,
        additional_control_qubit_indices: List[int] = [],
        inverse: bool = False,
    ) -> None:
        self.additional_control_qubit_indices = additional_control_qubit_indices
        self.inverse = inverse
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        discretization = self.lattice.discretization
        if discretization not in STREAMING_POPULATIONS:
            raise LatticeException("ABE only currently supported in D1Q3 and D2Q9")
        # The one-hot encoding is only used in D2Q9.
        encoding = (
            self.lattice.get_encoding()
            if discretization == LatticeDiscretization.D2Q9
            else ABEncodingType.AB
        )
        if encoding not in (ABEncodingType.AB, ABEncodingType.OH):
            raise LatticeException(
                f"Unsupported lattice encoding: {self.lattice.get_encoding()}"
            )

        for dim, dim_population_to_update in enumerate(
            STREAMING_POPULATIONS[discretization]
        ):
            grid_index = self.lattice.grid_index(dim)
            controls, shifts = self._shifts(dim_population_to_update, encoding)
            self.place(
                StreamingShift(len(grid_index), len(controls), shifts),
                controls + grid_index,
            )

    def _shifts(
        self, populations: Sequence[Sequence[int]], encoding: ABEncodingType
    ) -> Tuple[List[int], List[ShiftTerm]]:
        """The control qubits of one dimension's shift and its shifts, one per population."""
        extra = self.additional_control_qubit_indices
        velocity = self.lattice.velocity_index()
        if encoding == ABEncodingType.OH:
            controls = extra + [
                velocity[index] for indices in populations for index in indices
            ]
        else:
            controls = extra + velocity
        shifts = []
        for direction, indices in enumerate(populations):
            positive = (direction == 0) != self.inverse
            for index in indices:
                if encoding == ABEncodingType.OH:
                    shifts.append(
                        shift_on(positive, controls, extra + [velocity[index]])
                    )
                else:
                    shifts.append(
                        shift_on(
                            positive,
                            controls,
                            controls,
                            velocity_qubits_to_invert(self.lattice, index),
                        )
                    )
        return controls, shifts

    def structure(self) -> List[AnyBlock]:
        """
        The per-dimension shifts in order, declared for qarp's structured execution.

        Returns
        -------
        List[AnyBlock]
            The children of this operator.
        """
        return list(self.children())

    @override
    def __str__(self) -> str:
        return f"[Operator ABStreaming with lattice {self.lattice}]"
