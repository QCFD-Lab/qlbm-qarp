"""Collision operator for the :math:`D_2Q_4` discretization :class:`.SpaceTimeQLBM` algorithm as described in :cite:`spacetime`."""

from math import pi

from qarp.blocks import AnyBlock, CompositeBlock, SimpleBlock
from typing_extensions import override

from qlbm.components.base import LBMOperator, controlled
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


class SpaceTimeD2Q4CollisionOperator(LBMOperator):
    r"""An operator that performs collision part of the :class:`.SpaceTimeQLBM` algorithm.

    Collision is a local operation that is performed simultaneously on all velocity qubits corresponding to a grid location.
    In practice, this means the same circuit is repeated across all "local" qubit register chunks.
    Collision can be understood as follows:

    #. For each group of qubits, the states encoding velocities belonging to a particular equivalence class are first isolated with a series of :math:`X` and :math:`CX` gates. This leaves qubits not affected by the rotation in :math:`\ket{1}^{\otimes n_v-1}` state.
    #. A rotation gate is applied to the qubit(s) relevant to the equivalence class shift, controlled on the qubits set in the previous step.
    #. The operation performed in Step 1 is undone.

    The register setup of the :class:`.SpaceTimeLattice` is such that following each
    time step, an additional "layer" neighboring velocity qubits can be discarded,
    since the information they encode can never reach the relative origin in the remaining number of time steps.
    As such, the complexity of the collision operator decreases with the number of steps (still) to be simulated.
    For an in-depth mathematical explanation of the procedure, consult pages 11-15 of :cite:t:`spacetime`.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`timestep`          The time step for which to perform streaming.
    :attr:`gate_to_apply`     A built single-qubit ``qx.Block`` to apply to the velocities matching equivalence classes. ``None`` (the default) means :math:`R_y(\frac{\pi}{2})`.
    ========================= ======================================================================


    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime.collision.d2q4_old import SpaceTimeD2Q4CollisionOperator
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
                "geometry": [],
            },
        )

        # Draw the collision operator for 1 time step
        SpaceTimeD2Q4CollisionOperator(lattice=lattice, timestep=1).plot()
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        gate_to_apply: AnyBlock | None = None,
    ) -> None:
        self.timestep = timestep
        self.gate_to_apply = gate_to_apply
        super().__init__(lattice)

    @override
    def build_vanilla(self) -> None:
        properties = self.lattice.properties
        stride = properties.get_num_velocities_per_point()
        first = properties.get_num_grid_qubits()
        # Append the collision circuit at each step
        for start in range(
            first, first + properties.get_num_velocity_qubits(self.timestep), stride
        ):
            self.place(self.__local_collision_operator(), range(start, start + stride))

    def __local_collision_operator(self) -> AnyBlock:
        num_velocities = self.lattice.properties.get_num_velocities_per_point()
        control_qubits = list(range(1, num_velocities))
        if self.gate_to_apply is None:
            gate = SimpleBlock(1, name="ry")
            gate.ry(0, pi / 2)
        else:
            gate = self.gate_to_apply
        return CompositeBlock(
            [
                self.local_collision_circuit(reset_state=False),
                controlled(gate, control_qubits, [0]),
                self.local_collision_circuit(reset_state=True),
            ],
            num_velocities,
            name="local_collision",
        )

    def local_collision_circuit(self, reset_state: bool) -> AnyBlock:
        """
        The permutation that maps the colliding velocity states onto the control pattern (or back).

        Parameters
        ----------
        reset_state : bool
            Whether to apply the mirrored circuit that undoes the mapping.

        Returns
        -------
        AnyBlock
            The block over one velocity register.
        """
        num_velocities = self.lattice.properties.get_num_velocities_per_point()
        block = SimpleBlock(num_velocities, name="local_collision_map")
        if not reset_state:
            block.cx(0, 2)
            block.x(0)
            block.cx(1, 3)
            block.cx(0, 1)
            block.x(list(range(num_velocities)))
        # Same circuit, but mirrored
        else:
            block.x(list(range(num_velocities)))
            block.cx(0, 1)
            block.cx(1, 3)
            block.x(0)
            block.cx(0, 2)
        return block

    @override
    def __str__(self) -> str:
        return "Space Time Collision Operator"
