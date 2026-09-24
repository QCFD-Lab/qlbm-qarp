from itertools import product

import pytest

from qlbm.components.ab.initial import ABDiscreteUniformInitialConditions

from .qarp_helpers import lattice_builder, sample_register_counts


@pytest.mark.parametrize(
    "velocities,lattice_fixture",
    list(
        product(
            [[0], [0, 1], [6, 7, 8], [4, 5, 0, 7], list(range(9))],
            ["ab_lattice_d2q9_8x8"],
        )
    ),
)
def test_initial_ab_no_gird_superposition(velocities, lattice_fixture, request):
    lattice = request.getfixturevalue(lattice_fixture)

    builder = lattice_builder(lattice)
    builder.compose(
        ABDiscreteUniformInitialConditions(lattice, velocities, ([], [])),
    )

    counts = sample_register_counts(builder, lattice.velocity_index(), shots=256)
    output_velocities = sorted(counts.keys())

    assert output_velocities == sorted(velocities), (
        f"Expected output velocities to be {velocities}, got {output_velocities}"
    )


@pytest.mark.parametrize(
    "velocities,lattice_fixture",
    list(
        product(
            [[0], [0, 1], [6, 7, 8], [4, 5, 0, 7], list(range(9))],
            ["oh_lattice_d2q9_8x8"],
        )
    ),
)
def test_initial_oh_no_gird_superposition(velocities, lattice_fixture, request):
    lattice = request.getfixturevalue(lattice_fixture)

    builder = lattice_builder(lattice)
    builder.compose(
        ABDiscreteUniformInitialConditions(lattice, velocities, ([], [])),
    )

    counts = sample_register_counts(builder, lattice.velocity_index(), shots=256)
    output_velocities = sorted(counts.keys())

    assert output_velocities == sorted([2**v for v in velocities]), (
        f"Expected output velocities to be {velocities}, got {output_velocities}"
    )
