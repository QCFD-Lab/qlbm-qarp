"""Implementation of the :class:`.QBMResult`. for :class:`AmplitudeLattice`-based algorithms."""

import re
from os import listdir
from typing import Dict

import numpy as np
import vtk
from typing_extensions import override
from vtkmodules.util import numpy_support

from qlbm.lattice.lattices.base import AmplitudeLattice

from .base import QBMResult


class AmplitudeResult(QBMResult):
    """
    Implementation of the :class:`.QBMResult` for lattices inheriting from :class:`.AmplitudeLattice`.

    Processes counts sampled from :class:`.GridMeasurement` primitives.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.AmplitudeLattice` of the simulated system.
    :attr:`directory`           The directory to which the results outputs data to.
    :attr:`output_file_name`    The root name for files containing time step artifacts, by default "step".
    =========================== ======================================================================
    """

    num_steps: int
    """The time step to which this result corresponds."""

    directory: str
    """The output directory for the results."""

    output_file_name: str
    """The name of the file to output the artifacts to."""

    lattice: AmplitudeLattice
    """The lattice the result corresponds to."""

    def __init__(
        self,
        lattice: AmplitudeLattice,
        directory: str,
        output_file_name: str = "step",
    ) -> None:
        super().__init__(lattice, directory, output_file_name)

    @override
    def save_timestep_counts(
        self,
        counts: Dict[int, float],
        timestep: int,
        create_vis: bool = True,
        save_array: bool = False,
        n_cbits: int | None = None,
    ):
        num_grid_bits = self.lattice.num_grid_qubits
        if n_cbits is None:
            n_cbits = num_grid_bits

        # Grid measurements write dimension d into the cbits directly above
        # dimension d-1, so every coordinate is a shift plus a mask.
        dimension_bit_counts = [
            self.lattice.num_gridpoints[dim].bit_length()
            for dim in range(self.lattice.num_dims)
        ]
        dimension_offsets = [
            sum(dimension_bit_counts[:dim]) for dim in range(self.lattice.num_dims)
        ]

        def coordinate(key: int, dim: int) -> int:
            return (key >> dimension_offsets[dim]) & (
                (1 << dimension_bit_counts[dim]) - 1
            )

        if self.lattice.num_dims == 1:
            # The second dimension is a dirty rendering trick for VTK and ParaView
            count_history = np.zeros((self.lattice.num_gridpoints[0] + 1, 2))
            # The rest bonus only applies when the velocity register was
            # measured in full alongside the grid register.
            has_velocity_bits = (
                n_cbits - num_grid_bits == self.lattice.num_velocity_qubits
                and self.lattice.num_velocity_qubits > 0
            )
            for key, value in counts.items():
                x = key & ((1 << num_grid_bits) - 1)
                rest_bonus = int(has_velocity_bits and (key >> num_grid_bits) == 0)
                # Another dirty rendering trick for VTK and ParaView
                count_history[x][0] += value * (1 + rest_bonus)
                count_history[x][1] += value * (1 + rest_bonus)

        elif self.lattice.num_dims == 2:
            count_history = np.zeros(
                (self.lattice.num_gridpoints[0] + 1, self.lattice.num_gridpoints[1] + 1)
            )

            for key, value in counts.items():
                count_history[coordinate(key, 0)][coordinate(key, 1)] += value

        elif self.lattice.num_dims == 3:
            count_history = np.zeros(  # type: ignore
                (
                    self.lattice.num_gridpoints[0] + 1,
                    self.lattice.num_gridpoints[1] + 1,
                    self.lattice.num_gridpoints[2] + 1,
                )
            )

            for key, value in counts.items():
                count_history[coordinate(key, 0)][coordinate(key, 1)][
                    coordinate(key, 2)
                ] += value

        # Transposing puts the x-axis last, which is the order VTK flattens in.
        self.save_timestep_array(
            np.transpose(count_history),
            timestep,
            create_vis=create_vis,
            save_counts_array=save_array,
        )

    @override
    def visualize_all_numpy_data(self):
        # Filter the algorithm output files
        r = re.compile("[a-zA-Z0-9]+_[0-9]+.csv")
        for data_file_name in filter(r.match, listdir(self.directory)):
            data = np.genfromtxt(
                f"{self.directory}/{data_file_name}",
                dtype=None,
                delimiter=",",
                autostrip=True,
            )
            vtk_data = numpy_support.numpy_to_vtk(
                num_array=data, deep=True, array_type=vtk.VTK_FLOAT
            )
            img = vtk.vtkImageData()
            img.SetDimensions(
                self.lattice.num_gridpoints[0] + 1,
                self.lattice.num_gridpoints[1] + 1 if self.lattice.num_dims > 1 else 1,
                self.lattice.num_gridpoints[2] + 1 if self.lattice.num_dims > 2 else 1,
            )
            img.GetPointData().SetScalars(vtk_data)

            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(
                f"{self.paraview_dir}/{data_file_name.split('.')[0]}.vti"
            )
            writer.SetInputData(img)
            writer.Write()
