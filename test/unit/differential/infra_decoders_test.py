"""Differential tests: result decoders and reinitializer vs the qiskit-based qlbm.

Every case feeds a fixed synthetic counts dictionary to both packages, so the
comparison carries no RNG; the decoded field arrays must agree exactly.
"""

from typing import Dict

import numpy as np
import pytest
import qarpx as qx

from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.reinitialize.spacetime_reinitializer import SpaceTimeReinitializer
from qlbm.infra.result.amplitude_result import AmplitudeResult
from qlbm.infra.result.lqlga_result import LQLGAResult
from qlbm.infra.result.spacetime_result import SpaceTimeResult
from qlbm.lattice import MSLattice
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from test.oracle import cases_infra
from test.results import decode_field
from test.unit.differential.fixtures import load_npz

_cases_module = cases_infra
CASES_DECODE = _cases_module.CASES_DECODE
CASES_REINIT = _cases_module.CASES_REINIT

_DECODERS = {
    "amplitude": AmplitudeResult,
    "spacetime": SpaceTimeResult,
    "lqlga": LQLGAResult,
}


def load_fixture(case_id: str):
    """Load the oracle fixture for ``case_id``, skipping when it is absent."""
    return load_npz(case_id, "generate_fixtures_infra")


def build_lattice(case):
    """Build the lattice for ``case``."""
    family = case.get("family", case.get("lattice_family"))
    if family == "ms":
        return MSLattice(case["lattice"])
    if family == "lqlga":
        return LQLGALattice(case["lattice"])
    if family == "spacetime":
        return SpaceTimeLattice(case["num_timesteps"], case["lattice"])
    raise ValueError(f"Unknown lattice family: {family}")


def counts_of(case) -> Dict[int, float]:
    """The case's synthetic counts as LSB classical-bit integer keys."""
    return {int(key): float(value) for key, value in case["counts"]}


class TestDecodersAgainstReference:
    """Every decoder reproduces the qiskit-based field array from the same counts."""

    @pytest.mark.parametrize("case", CASES_DECODE, ids=lambda case: case["id"])
    def test_field_matches_reference(self, case, tmp_path):
        """The decoded field equals the qiskit-based one, element for element."""
        fixture = load_fixture(case["id"])
        lattice = build_lattice(case)
        result = _DECODERS[case["decoder"]](lattice, str(tmp_path))

        field = decode_field(result, counts_of(case), case["n_cbits"])

        assert field.shape == fixture["field"].shape
        np.testing.assert_allclose(field, fixture["field"], atol=1e-12, rtol=0)


class TestSpaceTimeReinitializerAgainstReference:
    """The counts decoding and the re-synthesized circuit both match the qiskit-based ones."""

    @pytest.mark.parametrize("case", CASES_REINIT, ids=lambda case: case["id"])
    def test_velocity_pairs_match_reference(self, case):
        """Counts decode into the same (gridpoint, velocity profile) pairs."""
        fixture = load_fixture(case["id"])
        lattice = build_lattice({**case, "lattice_family": "spacetime"})
        reinitializer = SpaceTimeReinitializer(lattice, CircuitCompiler())

        pairs = reinitializer.counts_to_velocity_pairs(counts_of(case))

        grid = np.asarray([gridpoint for gridpoint, _ in pairs], dtype=np.int64)
        velocity = np.asarray([profile for _, profile in pairs], dtype=bool)

        # Fixtures hold a 2-wide gridpoint tuple, zero-padded for 1D lattices;
        # the reinitializer emits one entry per dimension.
        assert grid.shape == (fixture["grid"].shape[0], lattice.num_dims)
        np.testing.assert_array_equal(grid, fixture["grid"][:, : lattice.num_dims])
        np.testing.assert_array_equal(velocity, fixture["velocity"])

    @pytest.mark.parametrize(
        "case",
        [case for case in CASES_REINIT if case.get("store_state", True)],
        ids=lambda case: case["id"],
    )
    def test_resynthesized_state_matches_reference(self, case):
        """The rebuilt initial conditions prepare the qiskit-based state."""
        fixture = load_fixture(case["id"])
        lattice = build_lattice({**case, "lattice_family": "spacetime"})
        reinitializer = SpaceTimeReinitializer(lattice, CircuitCompiler())

        block = reinitializer.reinitialize(
            statevector=np.zeros(0, dtype=np.complex128),
            counts=counts_of(case),
            n_cbits=case["n_cbits"],
        )
        statevector = np.asarray(
            qx.QarpSimulator().statevector(block.flatten(), block.n_qubits)
        )

        np.testing.assert_allclose(statevector, fixture["psi"], atol=1e-9, rtol=0)
