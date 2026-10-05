""":class:`.SpaceTimeQLBM`-specific implementation of the :class:`.QBMResult`."""

import re
from os import listdir
from typing import Dict

import numpy as np
import vtk
from typing_extensions import override
from vtkmodules.util import numpy_support

from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.exceptions import ResultsException

from .base import QBMResult


class SpaceTimeResult(QBMResult):
    """
    :class:`.SpaceTimeQLBM`-specific implementation of the :class:`.QBMResult`.

    Processes counts sampled from :class:`.SpaceTimeGridVelocityMeasurement` primitives.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.SpaceTimeLattice` of the simulated system.
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

    lattice: SpaceTimeLattice
    """The lattice the result corresponds to."""

    def __init__(
        self,
        lattice: SpaceTimeLattice,
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
        dimension_bit_counts = (
            self.lattice.num_gridpoints[0].bit_length(),
            self.lattice.num_gridpoints[1].bit_length()
            if self.lattice.num_dims > 1
            else 0,
            self.lattice.num_gridpoints[2].bit_length()
            if self.lattice.num_dims > 2
            else 0,
        )

        if self.lattice.num_dims == 1:
            # The second dimension is a dirty rendering trick for VTK and ParaView
            count_history = np.zeros((self.lattice.num_gridpoints[0] + 1, 2))
            num_grid_bits = dimension_bit_counts[0]
            for key, value in counts.items():
                x = key & ((1 << num_grid_bits) - 1)
                # The number of 1s above the grid register is the number of populations
                num_populations = (key >> num_grid_bits).bit_count()
                # Another dirty rendering trick for VTK and ParaView
                count_history[x][0] += value * num_populations
                count_history[x][1] += value * num_populations
        elif self.lattice.num_dims == 2:
            count_history = np.zeros(
                (self.lattice.num_gridpoints[0] + 1, self.lattice.num_gridpoints[1] + 1)
            )
            num_grid_bits = dimension_bit_counts[0] + dimension_bit_counts[1]
            for key, value in counts.items():
                x = key & ((1 << dimension_bit_counts[0]) - 1)
                y = (key >> dimension_bit_counts[0]) & (
                    (1 << dimension_bit_counts[1]) - 1
                )
                num_populations = (key >> num_grid_bits).bit_count()
                count_history[x][y] += value * num_populations
        else:
            raise ResultsException(
                f"Space-Time results are not supported for lattices with {self.lattice.num_dims} dimensions."
            )
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
                num_array=data.flatten, deep=True, array_type=vtk.VTK_FLOAT
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
