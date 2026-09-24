"""Base class for all algorithm-specific reinitializers."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Dict

import numpy as np
import qarpx as qx

from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice import Lattice


class Reinitializer(ABC):
    r"""
    Base class for all algorithm-specific reinitializers.

    A ``Reinitializer`` uses the information available
    at the end of the simulation of 1 or more time steps
    to derive new initial conditions for the following time steps.
    Such information includes the quantum state and counts extracted from it.
    For convenience, all reinitializers provide a uniform :meth:`reinitialize` interface,
    which takes as input both the quantum state and the counts performed during simulation.
    Its implementation may choose to ignore one of those inputs, depending on the
    algorithm and implementation.

    The return value discriminates the two transition mechanisms the runner
    supports, and callers must branch on its type:

    * an ``np.ndarray`` is the LSB-indexed statevector the next time step is
      seeded with (state carried by injection, no circuit needed);
    * a qarp ``Block`` is an initial conditions circuit the next time step
      applies starting from :math:`\ket{0}^{\otimes n}`.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.Lattice` of the simulated system.
    :attr:`compiler`            The compiler that lowers novel initial conditions circuits.
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
        self.lattice = lattice
        self.compiler = compiler
        self.logger = logger

    @abstractmethod
    def reinitialize(
        self,
        statevector: np.ndarray,
        counts: Dict[int, float],
        n_cbits: int | None = None,
        optimization_level: int = 0,
    ) -> "qx.Block | np.ndarray":
        """
        Parses the input statevector and counts and derives the initial conditions of the next time step.

        Parameters
        ----------
        statevector : np.ndarray
            The LSB-indexed statevector at the end of the simulated time step.
        counts : Dict[int, float]
            The counts extracted from the statevector, keyed by LSB classical-bit integer.
        n_cbits : int | None, optional
            The width of the classical register the counts were sampled into, by default None.
        optimization_level : int, optional
            The optimization level to pass to the circuit compiler, by default 0.

        Returns
        -------
        qx.Block | np.ndarray
            Either the statevector to seed the next time step with, or the
            initial conditions block to apply from the all-zero state.
        """
        pass

    @abstractmethod
    def requires_statevector(self) -> bool:
        """
        Whether the reinitializer requires a copy of the statevector.

        Returns
        -------
        bool
            Whether a statevector is needed.
        """
        pass
