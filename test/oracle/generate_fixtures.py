"""Write the fixtures of ``cases.py``: exact ``Operator`` matrices under ``unitary``.

Run with the qiskit-based qlbm at ``fixture_store.REFERENCE_COMMIT`` installed; the
README's Testing section has the command.
"""

from types import SimpleNamespace

from cases import CASES
from fixture_store import save_npz
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def find_equivalence_class(discretization_name, mass, momentum):
    """Select the equivalence class with ``id() == (mass, momentum)``."""
    from qlbm.lattice.eqc.eqc_generator import EquivalenceClassGenerator
    from qlbm.lattice.spacetime.properties_base import LatticeDiscretization

    discretization = LatticeDiscretization[discretization_name]
    for eqc in EquivalenceClassGenerator(discretization).generate_equivalence_classes():
        eqc_mass, eqc_momentum = eqc.id()
        if eqc_mass == mass and list(eqc_momentum) == list(momentum):
            return eqc
    raise ValueError(
        f"No equivalence class ({mass}, {momentum}) in {discretization_name}"
    )


def build_reference(case):
    """Build the qiskit component for ``case`` and return its circuit."""
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
    from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
    from qlbm.tools.utils import ComparatorMode

    kind = case["component"]
    if kind == "empty":
        lattice = SimpleNamespace(circuit=QuantumCircuit(case["n"]))
        return EmptyPrimitive(lattice).circuit
    if kind == "state_setter":
        return StateSetter(case["n"], case["state"]).circuit
    if kind == "mcswap":
        lattice = SimpleNamespace(circuit=QuantumCircuit(case["n"]))
        return MCSwap(lattice, case["controls"], tuple(case["targets"])).circuit
    if kind == "hwadder":
        return HammingWeightAdder(case["x"], case["y"]).circuit
    if kind == "truncated_qft":
        return TruncatedQFT(case["n"], case["dft_size"]).circuit
    if kind == "uniform_state_prep":
        return UniformStatePrep(case["n"], case["states"], case["ctrl"]).circuit
    if kind == "phase_shift":
        return PhaseShift(case["n"], case["positive"]).circuit
    if kind == "param_phase_shift":
        return ParameterizedPhaseShift(
            case["n"], case["add"], case["positive"], case["ctrl"]
        ).circuit
    if kind == "draper_adder":
        return ParameterizedDraperAdder(
            case["n"], case["add"], case["positive"], case["ctrl"]
        ).circuit
    if kind == "addition_conversion":
        return AdditionConversion(
            case["n"], case["state_from"], case["state_to"], case["ctrl"]
        ).circuit
    if kind == "two_register_comparator":
        return TwoRegisterComparator(case["n"], ComparatorMode[case["mode"]]).circuit
    if kind == "single_register_comparator":
        return SingleRegisterComparator(
            case["n"], case["num"], ComparatorMode[case["mode"]]
        ).circuit
    if kind == "eqc_permutation":
        return EQCPermutation(
            find_equivalence_class(
                case["discretization"], case["eqc_mass"], case["eqc_momentum"]
            ),
            case["inverse"],
        ).circuit
    if kind == "eqc_redistribution":
        return EQCRedistribution(
            find_equivalence_class(
                case["discretization"], case["eqc_mass"], case["eqc_momentum"]
            ),
            case["decompose"],
        ).circuit
    if kind == "eqc_collision":
        return EQCCollisionOperator(
            LatticeDiscretization[case["discretization"]]
        ).circuit
    raise ValueError(f"Unknown component kind: {kind}")


def main():
    """Write one fixture file per case."""
    for case in CASES:
        circuit = build_reference(case)
        unitary = Operator(circuit).data
        save_npz(case["id"], unitary=unitary)
        print(f"{case['id']}: {unitary.shape[0]}x{unitary.shape[0]} written")


if __name__ == "__main__":
    main()
