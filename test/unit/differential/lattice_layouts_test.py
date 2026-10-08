"""Differential tests: lattice register layouts vs the qiskit-based qlbm.

The fixture embeds each lattice spec, the register names, sizes and global
qubit indices, and the output of every public index helper.
"""

import pytest

import qlbm.lattice as lattice_module
from test.unit.differential.fixtures import load_json

ENTRIES = load_json("lattice_layouts.json", "generate_lattice_layouts")

# Helpers whose first argument is a gridpoint tuple (JSON stores lists).
TUPLE_ARG_HELPERS = {"gridpoint_index_tuple", "velocity_index_tuple"}


def build_lattice(spec):
    """Instantiate a lattice from an embedded fixture spec.

    Parameters
    ----------
    spec : dict
        The case spec (class, lattice_data, kwargs, mutations).

    Returns
    -------
    Lattice
        The constructed (and mutated) lattice.
    """
    cls = getattr(lattice_module, spec["class"])
    lattice = cls(lattice_data=spec["lattice_data"], **spec.get("kwargs", {}))
    for mutation in spec.get("mutations", []):
        getattr(lattice, mutation["method"])(*mutation["args"])
    return lattice


def _norm(value):
    """JSON-normalize a helper result (tuples become lists).

    Parameters
    ----------
    value : Any
        The helper return value.

    Returns
    -------
    Any
        The JSON-compatible equivalent.
    """
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value


def _decode_args(helper, args):
    """Convert JSON-decoded helper args back to their call form.

    Parameters
    ----------
    helper : str
        The helper name.
    args : list
        The JSON-decoded argument list.

    Returns
    -------
    list
        The arguments to call the helper with.
    """
    if helper in TUPLE_ARG_HELPERS:
        return [tuple(args[0]), *args[1:]]
    return args


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["spec"]["id"])
def test_register_layout_matches_reference(entry):
    """Register names, sizes, and global qubit indices match the qiskit-based lattice."""
    lattice = build_lattice(entry["spec"])

    layout = [
        [register.name, register.size, list(register)] for register in lattice.registers
    ]
    assert layout == entry["registers"]

    # Offsets are the first global index of each (non-empty) register.
    for register, (_, size, qubits) in zip(
        lattice.registers, entry["registers"], strict=True
    ):
        if size > 0:
            assert register.offset == qubits[0]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["spec"]["id"])
def test_total_qubits_matches_reference(entry):
    """``n_qubits`` equals the reference lattice circuit's qubit count."""
    lattice = build_lattice(entry["spec"])

    assert lattice.n_qubits == entry["total_qubits"]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["spec"]["id"])
def test_num_attributes_match_reference(entry):
    """All recorded ``num_*`` attributes match the qiskit-based values."""
    lattice = build_lattice(entry["spec"])

    for name, expected in entry["attributes"].items():
        assert getattr(lattice, name) == expected, name


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["spec"]["id"])
def test_index_helpers_match_reference(entry):
    """Every recorded index-helper call reproduces the qiskit-based output."""
    lattice = build_lattice(entry["spec"])

    for call in entry["calls"]:
        helper = call["helper"]
        args = _decode_args(helper, call["args"])
        result = getattr(lattice, helper)(*args)
        assert _norm(result) == call["result"], f"{helper}({args})"
