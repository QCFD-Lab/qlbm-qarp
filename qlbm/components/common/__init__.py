"""Common primitives used for multiple encodings."""

from .adders import ParameterizedDraperAdder, ParameterizedPhaseShift, PhaseShift
from .arithmetic import DraperQFTAdder, RGQFTMultiplier, ccp
from .cbse_collision import EQCCollisionOperator, EQCPermutation, EQCRedistribution
from .comparators import SingleRegisterComparator, TwoRegisterComparator
from .primitives import (
    AdditionConversion,
    EmptyPrimitive,
    HammingWeightAdder,
    MCSwap,
    StateSetter,
    TruncatedQFT,
    UniformStatePrep,
)

__all__ = [
    "AdditionConversion",
    "DraperQFTAdder",
    "RGQFTMultiplier",
    "ccp",
    "EmptyPrimitive",
    "EQCCollisionOperator",
    "EQCPermutation",
    "EQCRedistribution",
    "HammingWeightAdder",
    "ParameterizedDraperAdder",
    "ParameterizedPhaseShift",
    "PhaseShift",
    "StateSetter",
    "TruncatedQFT",
    "UniformStatePrep",
    "MCSwap",
    "SingleRegisterComparator",
    "TwoRegisterComparator",
]
