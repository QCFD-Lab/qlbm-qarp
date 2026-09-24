"""Differential tests: common primitives vs the qiskit-based qlbm.

Exact unitaries are compared element-wise, global phase included.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from qlbm.components.common import (
    AdditionConversion,
    EmptyPrimitive,
    EQCCollisionOperator,
    EQCPermutation,
    EQCRedistribution,
    HammingWeightAdder,
    MCSwap,
    SingleRegisterComparator,
    StateSetter,
    TruncatedQFT,
    TwoRegisterComparator,
    UniformStatePrep,
)
from qlbm.components.common.adders import (
    ParameterizedDraperAdder,
    ParameterizedPhaseShift,
    PhaseShift,
)
from qlbm.lattice.eqc.eqc_generator import EquivalenceClassGenerator
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.utils import ComparatorMode
from test.oracle import cases
from test.unit.differential.fixtures import load_npz

_cases_module = cases
CASES = _cases_module.CASES


def _fake_lattice(n: int):
    return SimpleNamespace(n_qubits=n)


def _find_equivalence_class(discretization_name, mass, momentum):
    discretization = LatticeDiscretization[discretization_name]
    for eqc in EquivalenceClassGenerator(discretization).generate_equivalence_classes():
        eqc_mass, eqc_momentum = eqc.id()
        if eqc_mass == mass and list(eqc_momentum) == list(momentum):
            return eqc
    raise ValueError(
        f"No equivalence class ({mass}, {momentum}) in {discretization_name}"
    )


def build_block(case):
    """Build the qarp component for ``case``."""
    kind = case["component"]
    if kind == "empty":
        return EmptyPrimitive(_fake_lattice(case["n"]))
    if kind == "state_setter":
        return StateSetter(case["n"], case["state"])
    if kind == "mcswap":
        return MCSwap(
            _fake_lattice(case["n"]), case["controls"], tuple(case["targets"])
        )
    if kind == "hwadder":
        return HammingWeightAdder(case["x"], case["y"])
    if kind == "truncated_qft":
        return TruncatedQFT(case["n"], case["dft_size"])
    if kind == "uniform_state_prep":
        return UniformStatePrep(case["n"], case["states"], case["ctrl"])
    if kind == "phase_shift":
        return PhaseShift(case["n"], case["positive"])
    if kind == "param_phase_shift":
        return ParameterizedPhaseShift(
            case["n"], case["add"], case["positive"], case["ctrl"]
        )
    if kind == "draper_adder":
        return ParameterizedDraperAdder(
            case["n"], case["add"], case["positive"], case["ctrl"]
        )
    if kind == "addition_conversion":
        return AdditionConversion(
            case["n"], case["state_from"], case["state_to"], case["ctrl"]
        )
    if kind == "two_register_comparator":
        return TwoRegisterComparator(case["n"], ComparatorMode[case["mode"]])
    if kind == "single_register_comparator":
        return SingleRegisterComparator(
            case["n"], case["num"], ComparatorMode[case["mode"]]
        )
    if kind == "eqc_permutation":
        return EQCPermutation(
            _find_equivalence_class(
                case["discretization"], case["eqc_mass"], case["eqc_momentum"]
            ),
            case["inverse"],
        )
    if kind == "eqc_redistribution":
        return EQCRedistribution(
            _find_equivalence_class(
                case["discretization"], case["eqc_mass"], case["eqc_momentum"]
            ),
            case["decompose"],
        )
    if kind == "eqc_collision":
        return EQCCollisionOperator(LatticeDiscretization[case["discretization"]])
    raise ValueError(f"Unknown component kind: {kind}")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_primitive_matches_reference(case):
    """The unitary equals the qiskit-based one exactly (global phase included)."""
    expected = load_npz(case["id"], "generate_fixtures")["unitary"]

    block = build_block(case)
    block.build()
    actual = np.asarray(block.unitary_matrix())

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)


def test_qft_block_is_dft_matrix():
    """Pin QFTBlock (swaps included) to the analytic DFT matrix.

    LSB index basis, the convention the adders and comparators rely on.
    """
    from qarp.blocks import QFTBlock

    for n in range(1, 6):
        dim = 2**n
        block = QFTBlock(n)
        block.build()
        actual = np.asarray(block.unitary_matrix())
        j, k = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
        expected = np.exp(2j * np.pi * j * k / dim) / np.sqrt(dim)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
