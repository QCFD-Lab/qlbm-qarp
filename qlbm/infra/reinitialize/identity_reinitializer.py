"""Identity reinitializer used for the :class:`.CQLBM` and :class:`.LQLGA` algorithms."""

from logging import Logger, getLogger
from typing import Dict

import numpy as np
from typing_extensions import override

from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.reinitialize.base import Reinitializer
from qlbm.lattice.lattices.base import Lattice


class IdentityReinitializer(Reinitializer):
    r"""
    Implementation of the :class:`.Reinitializer` that passes the statevector along to the following time step.

    Useful for the :class:`.CQLBM` and :class:`.LQLGA` algorithms: the state at
    the end of one time step is the initial state of the next.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.Lattice` of the simulated system.
    :attr:`compiler`            The compiler that lowers novel initial conditions circuits. Unused.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``
    =========================== ======================================================================
    """

    lattice: Lattice

    def __init__(
        self,
        lattice: Lattice,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, compiler, logger)
        self.lattice = lattice
        self.logger = logger

    @override
    def reinitialize(
        self,
        statevector: np.ndarray,
        counts: Dict[int, float],
        n_cbits: int | None = None,
        optimization_level: int = 0,
    ) -> np.ndarray:
        """
        Returns the provided ``statevector`` unchanged, for the runner to inject into the next time step.

        Parameters
        ----------
        statevector : np.ndarray
            The LSB-indexed statevector at the end of the simulated time step.
        counts : Dict[int, float]
            Ignored.
        n_cbits : int | None, optional
            Ignored.
        optimization_level : int, optional
            Ignored.

        Returns
        -------
        np.ndarray
            The statevector to seed the next time step with.
        """
        return statevector

    @override
    def requires_statevector(self) -> bool:
        return True
