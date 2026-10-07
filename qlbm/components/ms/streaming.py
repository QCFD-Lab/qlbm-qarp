"""Quantum circuits for the implementation of QFT-based streaming as described in :cite:t:`collisionless`."""

from typing import List

from qarp.blocks import AnyBlock
from typing_extensions import override

from qlbm.components.base import LatticePrimitive, LBMOperator
from qlbm.components.common.adders import StreamingShift, shift_on
from qlbm.lattice import MSLattice
from qlbm.tools import CircuitException, bit_value


class StreamingAncillaPreparation(LatticePrimitive):
    r"""
    A primitive used in :class:`.MSStreamingOperator` that implements the preparatory step of streaming necessary for the :class:`.MSQLBM` method.

    This operator sets the ancilla qubits to :math:`\ket{1}` for the velocities that
    will be streamed in the next CFL time step.  With two velocities per
    dimension the velocity register is empty and the ancilla is set outright.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`velocities`        The velocities that need to be streamed within the next time step.
    :attr:`dim`               The dimension to which the velocities correspond.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import StreamingAncillaPreparation
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        # Streaming velocities indexed 2 in the y (1) dimension
        StreamingAncillaPreparation(lattice=lattice, velocities=[2], dim=1).plot()
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice, velocities: List[int], dim: int) -> None:
        self.velocities = velocities
        self.dim = dim
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        control_qubits = self.lattice.velocity_index(self.dim)
        target_qubit = self.lattice.ancillae_velocity_index(self.dim)[0]
        # Ignore the directional qubit
        num_velocity_qubits = self.lattice.num_velocities[self.dim].bit_length() - 1

        for velocity in self.velocities:
            # Inverting the qubits that are 0 turns the velocity state in this
            # dimension to |11...1>, which allows controlling on this one velocity.
            qubits_to_invert = [
                control_qubits[velocity_qubit]
                for velocity_qubit in range(num_velocity_qubits)
                if bit_value(velocity, velocity_qubit) == 0
            ]
            if qubits_to_invert:
                self.x(qubits_to_invert)
            if control_qubits:
                self.mcx(*control_qubits, target_qubit)
            else:
                # Zero-size velocity register (2 velocities per dimension):
                # an MCX with no controls degenerates to a plain X.
                self.x(target_qubit)
            if qubits_to_invert:
                self.x(qubits_to_invert)

    @override
    def __str__(self) -> str:
        return f"[Primitive StreamingAncillaPreparation on dimension {self.dim}, for velocities {self.velocities}]"


class ControlledIncrementer(LBMOperator):
    r"""
    A primitive used in :class:`.MSStreamingOperator` that implements the streaming operation on the states for which the ancilla qubits are in the state :math:`\ket{1}`.

    This primitive is applied after the primitive :class:`.StreamingAncillaPreparation` to compose the streaming operator.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`reflection`        The reflection attribute decides the type of reflection that will take place. This should
                              be either "specular", "bounceback", or ``None``, and defaults to None. This parameter
                              governs which qubits are used as controls for the Fourier space phase shifts.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import ControlledIncrementer
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        # Streaming velocities indexed 2 in the y (1) dimension
        ControlledIncrementer(lattice=lattice).plot()
    """

    supported_reflection: List[str] = ["specular", "bounceback"]

    lattice: MSLattice

    def __init__(self, lattice: MSLattice, reflection: str | None = None) -> None:
        if reflection and reflection not in self.supported_reflection:
            raise CircuitException(
                f'Controlled Incrementer does not support reflection type "{reflection}". Supported types are {self.supported_reflection}'
            )
        self.reflection = reflection
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        for dim in range(self.lattice.num_dims):
            grid_index = self.lattice.grid_index(dim)
            direction = self.lattice.velocity_dir_index(dim)

            # The gate is controlled by the velocity direction and the velocity (or obstacle) ancilla qubits
            match self.reflection:
                case "specular":
                    ancilla = self.lattice.ancillae_obstacle_index(dim)
                case "bounceback":
                    ancilla = self.lattice.ancillae_obstacle_index(0)
                case _:
                    ancilla = self.lattice.ancillae_velocity_index(dim)
            control_qubits = ancilla + direction

            # UP+ when the direction qubit is |1>, UP- when it is |0>
            self.place(
                StreamingShift(
                    len(grid_index),
                    len(control_qubits),
                    [
                        shift_on(True, control_qubits, control_qubits),
                        shift_on(False, control_qubits, control_qubits, direction),
                    ],
                ),
                control_qubits + grid_index,
            )

    def structure(self) -> List[AnyBlock]:
        """
        The per-dimension shifts in order, declared for qarp's structured execution.

        Returns
        -------
        List[AnyBlock]
            The children of this primitive.
        """
        return list(self.children())

    @override
    def __str__(self) -> str:
        return f"[Primitive ControlledIncrementer with reflection {self.reflection}]"


class MSStreamingOperator(LBMOperator):
    """An operator that performs streaming in Fourier space as part of the :class:`.MSQLBM` algorithm.

    Streaming is broken down into the following steps:

    #. A :class:`.StreamingAncillaPreparation` object prepares the ancilla velocity qubits for CFL time step. This happens independently for all dimensions, and it is assumed the velocity discretization is uniform across dimensions.
    #. A :class:`.ControlledIncrementer` performs incrementation or decrementation in the Fourier space, controlled on the ancilla qubits set in the previous steps.
    #. For efficiency reasons, the velocity qubits set in step 1 are **not** reset, as they will be reused in the subsequent reflection step. Another instance of the :class:`.StreamingAncillaPreparation` would be required to consistently end the step.

    For an in-depth mathematical explanation of the procedure, consult Section 4 of :cite:t:`collisionless`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`velocities`        A list of velocities to increment. This is computed according to CFL counter.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import MSStreamingOperator
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        # Streaming the velocity with index 2
        MSStreamingOperator(lattice=lattice, velocities=[2]).plot()
    """

    lattice: MSLattice

    def __init__(self, lattice: MSLattice, velocities: List[int]) -> None:
        self.velocities_to_stream = velocities
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        for dim in range(self.lattice.num_dims):
            self.place(
                StreamingAncillaPreparation(
                    self.lattice, self.velocities_to_stream, dim
                )
            )
        self.place(ControlledIncrementer(self.lattice))

    @override
    def __str__(self) -> str:
        return (
            f"[Operator StreamingOperator for velocities {self.velocities_to_stream}]"
        )
