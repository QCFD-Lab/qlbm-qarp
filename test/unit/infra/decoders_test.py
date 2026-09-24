"""Analytic tests for the result decoders and the Space-Time reinitializer."""

import numpy as np
import pytest
import qarpx as qx

from qlbm.components.spacetime.initial.pointwise import (
    PointWiseSpaceTimeInitialConditions,
)
from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.reinitialize.spacetime_reinitializer import SpaceTimeReinitializer
from qlbm.infra.result.amplitude_result import AmplitudeResult
from qlbm.infra.result.lqlga_result import LQLGAResult
from qlbm.infra.result.spacetime_result import SpaceTimeResult
from qlbm.lattice import MSLattice
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretizationProperties
from qlbm.tools.exceptions import ResultsException
from test.results import decode_field


class TestAmplitudeCoordinates:
    """The 2D/3D amplitude decoder reads coordinates from the right registers."""

    def test_non_square_2d_places_counts_at_the_measured_gridpoint(self, tmp_path):
        """Non-square lattices land at the gridpoint the measurement encoded."""
        # 8 x 4 lattice: 3 x-cbits then 2 y-cbits, so x and y are not
        # interchangeable.
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 4}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            }
        )
        result = AmplitudeResult(lattice, str(tmp_path))

        # cbits 0..2 = x = 6, cbits 3..4 = y = 2.
        field = decode_field(result, {0b10110: 42.0}, 5)

        # The field is transposed on the way out, so it indexes [y][x].
        assert field.shape == (4, 8)
        assert field[2][6] == 42.0
        assert field.sum() == 42.0

    def test_square_2d_places_counts_at_the_measured_gridpoint(self, tmp_path):
        """Square lattices place counts at the gridpoint the measurement encoded."""
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            }
        )
        result = AmplitudeResult(lattice, str(tmp_path))

        field = decode_field(result, {0b1001: 7.0}, 4)

        assert field[2][1] == 7.0

    def test_counts_sharing_a_gridpoint_accumulate(self, tmp_path):
        """Keys that differ only in the non-grid cbits add up at their gridpoint."""
        # With the velocity register measured too, several keys land on one
        # cell; the field holds their sum, as the 1D decoder already does.
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            }
        )
        result = AmplitudeResult(lattice, str(tmp_path))
        grid_key = 0b1001  # x = 1, y = 2
        counts = {grid_key: 10.0, grid_key | (1 << 4): 20.0, grid_key | (1 << 5): 5.0}

        field = decode_field(result, counts, 6)

        assert field[2][1] == 35.0
        assert field.sum() == 35.0


class TestLQLGAMultiDimensional:
    """The LQLGA decoder handles lattices above one dimension."""

    def test_2d_lattice_decodes_into_a_2d_field(self, tmp_path):
        """A 2D lattice decodes into a 2D field."""
        lattice = LQLGALattice(
            {
                "lattice": {"dim": {"x": 2, "y": 2}, "velocities": "D2Q4"},
                "geometry": [],
            }
        )
        result = LQLGAResult(lattice, str(tmp_path))

        # 4 gridpoints x 4 velocity channels = 16 cbits; occupy channel 0 of
        # gridpoint 3, which is (x, y) = (1, 1).
        field = decode_field(result, {1 << 12: 100.0}, 16)

        assert field.shape == (2, 2)
        assert field[1][1] == pytest.approx(1.0)
        assert field.sum() == pytest.approx(1.0)

    def test_1d_slices_the_key_per_gridpoint(self, tmp_path):
        """Each gridpoint's mass comes from its own slice of the key."""
        lattice = LQLGALattice(
            {"lattice": {"dim": {"x": 4}, "velocities": "D1Q2"}, "geometry": []}
        )
        channel_masses = LatticeDiscretizationProperties.get_channel_masses(
            lattice.discretization
        )
        result = LQLGAResult(lattice, str(tmp_path))

        # Cbits 4 and 5 are both channels of gridpoint 2; every other
        # gridpoint stays empty, and counts are normalized by their total.
        field = decode_field(result, {0b00110000: 100.0}, 8)

        expected = np.zeros((2, 4))
        expected[0][2] = expected[1][2] = float(channel_masses.sum())
        np.testing.assert_allclose(field, expected, atol=1e-12)


class TestSpaceTimeResultGuards:
    """Space-Time results are defined up to two dimensions."""

    def test_three_dimensional_lattice_raises(self, tmp_path):
        """3D lattices raise a diagnosable exception."""
        lattice = SpaceTimeLattice(
            1,
            {
                "lattice": {"dim": {"x": 2, "y": 2, "z": 2}, "velocities": "D3Q6"},
                "geometry": [],
            },
        )
        result = SpaceTimeResult(lattice, str(tmp_path))

        with pytest.raises(ResultsException):
            result.save_timestep_counts({0: 1.0}, 0, create_vis=False, n_cbits=9)


class TestSpaceTimeReinitializer:
    """Counts decode into the (gridpoint, velocity profile) pairs they encode."""

    def test_empty_velocity_profiles_are_filtered(self):
        """Outcomes with no populations carry no initial conditions."""
        lattice = SpaceTimeLattice(
            1, {"lattice": {"dim": {"x": 8}, "velocities": "D1Q2"}, "geometry": []}
        )
        reinitializer = SpaceTimeReinitializer(lattice, CircuitCompiler())

        # 3 grid cbits, 2 velocity cbits: keys below 8 carry no population.
        pairs = reinitializer.counts_to_velocity_pairs({0: 5.0, 3: 5.0, 11: 5.0})

        assert pairs == [((3,), (True, False))]

    def test_split_count_reads_the_low_cbits_as_the_grid(self):
        """The grid occupies the low cbits, the velocity profile the high ones."""
        lattice = SpaceTimeLattice(
            1, {"lattice": {"dim": {"x": 8}, "velocities": "D1Q2"}, "geometry": []}
        )
        reinitializer = SpaceTimeReinitializer(lattice, CircuitCompiler())

        # cbits 0..2 = x = 5, cbits 3..4 = velocity profile (False, True).
        assert reinitializer.split_count(0b10101) == ((5,), (False, True))


class TestPointWiseReinitializationRoundTrip:
    """Decoding a state's own measurement reproduces that state."""

    def test_pairs_round_trip_through_the_initial_conditions(self):
        """Decoding a state's own measurement rebuilds that state."""
        lattice = SpaceTimeLattice(
            1, {"lattice": {"dim": {"x": 8}, "velocities": "D1Q2"}, "geometry": []}
        )
        grid_data: list[tuple[tuple[int, ...], tuple[bool, ...]]] = [
            ((3,), (True, False)),
            ((5,), (False, True)),
        ]

        block = PointWiseSpaceTimeInitialConditions(lattice, grid_data)
        reinitializer = SpaceTimeReinitializer(lattice, CircuitCompiler())

        # The counts a perfect measurement of that state would produce.
        counts = {
            3 | (0b01 << 3): 1.0,
            5 | (0b10 << 3): 1.0,
        }
        assert reinitializer.counts_to_velocity_pairs(counts) == grid_data

        rebuilt = reinitializer.reinitialize(
            statevector=np.zeros(0, dtype=np.complex128), counts=counts
        )
        simulator = qx.QarpSimulator()
        np.testing.assert_allclose(
            np.asarray(simulator.statevector(rebuilt.flatten(), rebuilt.n_qubits)),
            np.asarray(simulator.statevector(block.flatten(), block.n_qubits)),
            atol=1e-12,
        )
