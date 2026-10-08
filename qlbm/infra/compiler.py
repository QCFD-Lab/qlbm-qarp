"""Optimization pass that lowers ``qlbm`` components into runnable qarp blocks."""

from logging import Logger, getLogger
from typing import List

import qarpx as qx

from qlbm.tools.exceptions import CompilerException
from qlbm.tools.utils import get_circuit_properties


class CircuitCompiler:
    r"""
    Lowers ``qlbm`` components into built, optionally optimized qarp ``Block``\ s.

    qarp consumes ``Block`` command streams directly, so compilation reduces
    to building the component and applying :meth:`qarp.blocks.Block.optimize`.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``.
    =========================== ======================================================================

    Example usage:

    .. code-block:: python

        from qlbm.components.spacetime import SpaceTimeQLBM
        from qlbm.infra import CircuitCompiler
        from qlbm.lattice import SpaceTimeLattice

        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
                "geometry": [],
            },
        )

        compiler = CircuitCompiler()
        block = compiler.compile(SpaceTimeQLBM(lattice), optimization_level=1)
    """

    supported_optimization_levels: List[int] = [0, 1, 2]
    """The optimization levels accepted by :meth:`compile`."""

    def __init__(
        self,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__()

        self.logger = logger

        logger.info(str(self))

    def compile(
        self,
        block: "qx.Block",
        optimization_level: int = 0,
    ) -> "qx.Block":
        """
        Lowers the provided object into a built qarp ``Block``.

        Parameters
        ----------
        block : qx.Block
            The block to compile; a component is a block.
        optimization_level : int, optional
            The optimization level, by default 0. It is checked here and
            applied at execution, where qarp plans the block's structure
            first and optimizes the remaining gate runs at that level; the
            block itself is never flattened.

        Returns
        -------
        qx.Block
            The built block.

        Raises
        ------
        CompilerException
            If the optimization level is not supported.
        """
        if optimization_level not in self.supported_optimization_levels:
            raise CompilerException(
                f"Unsupported optimization level {optimization_level}. Supported optimization levels are {self.supported_optimization_levels}."
            )

        block.build()

        self.logger.info(
            f"{str(self)}: Compiled block with properties {get_circuit_properties(block)} for opt={optimization_level}"
        )

        return block

    def __str__(self) -> str:
        """
        String representation of the compiler.

        Returns
        -------
        str
            The string representation.
        """
        return "[Compiler targeting qarp]"
