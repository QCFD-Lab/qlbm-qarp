"""Quantum circuits used for streaming in the :class:`ABQLBM` algorithm."""

from typing import List, Sequence

from qarp.blocks import AnyBlock, QFTBlock
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.base import LBMOperator, controlled
from qlbm.components.common.adders import PhaseShift
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


def controlled_phase_shift(
    num_qubits: int,
    positive: bool,
    control_qubits: Sequence[int],
    target_qubits: Sequence[int],
    inverted: Sequence[int] = (),
) -> AnyBlock:
    r"""
    The streaming phase shift on ``target_qubits`` (in the Fourier basis), controlled on ``control_qubits``.

    Parameters
    ----------
    num_qubits : int
        The width of the phase shift.
    positive : bool
        Whether to increment or decrement.
    control_qubits : Sequence[int]
        The control qubits.
    target_qubits : Sequence[int]
        The grid qubits shifted.
    inverted : Sequence[int]
        The controls active on :math:`\ket{0}`.

    Returns
    -------
    AnyBlock
        The placed block.
    """
    open_controls = set(inverted)
    return controlled(
        PhaseShift(num_qubits, positive),
        control_qubits,
        target_qubits,
        ctrl_state=[qubit not in open_controls for qubit in control_qubits],
    )


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

    def __init__(
        self,
        lattice: AmplitudeLattice,
        additional_control_qubit_indices: List[int] = [],
    ) -> None:
        self.additional_control_qubit_indices = additional_control_qubit_indices
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

        for dim, dim_population_to_update in enumerate(
            STREAMING_POPULATIONS[discretization]
        ):
            grid_index = self.lattice.grid_index(dim)
            self.place(QFTBlock(len(grid_index)), grid_index)
            for direction, indices in enumerate(dim_population_to_update):
                positive = direction == 0
                for index in indices:
                    match encoding:
                        case ABEncodingType.OH:
                            control_qubits = self.additional_control_qubit_indices + [
                                self.lattice.velocity_index()[index]
                            ]
                            inverted: List[int] = []
                        case ABEncodingType.AB:
                            control_qubits = (
                                self.additional_control_qubit_indices
                                + self.lattice.velocity_index()
                            )
                            inverted = velocity_qubits_to_invert(self.lattice, index)
                        case _:
                            raise LatticeException(
                                f"Unsupported lattice encoding: {self.lattice.get_encoding()}"
                            )
                    self.place(
                        controlled_phase_shift(
                            len(grid_index),
                            positive,
                            control_qubits,
                            grid_index,
                            inverted,
                        )
                    )
            self.place(~QFTBlock(len(grid_index)), grid_index)

    @override
    def __str__(self) -> str:
        return f"[Operator ABStreaming with lattice {self.lattice}]"
