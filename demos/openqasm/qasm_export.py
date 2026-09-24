import logging.config
from logging import getLogger
from os import mkdir
from os.path import exists
from time import perf_counter_ns

from qlbm.components import (
    MSQLBM,
)
from qlbm.lattice import MSLattice
from qlbm.tools.utils import create_directory_and_parents

if __name__ == "__main__":
    cfgs = [
        # "demos/lattices/2d_8x8_0_obstacle.json",
        # "demos/lattices/2d_8x8_1_obstacle.json",
        # "demos/lattices/2d_8x8_2_obstacle.json",
        # "demos/lattices/2d_16x16_0_obstacle.json",
        # "demos/lattices/2d_16x16_1_obstacle.json",
        "demos/lattices/2d_16x16_2_obstacle.json",
        # "demos/lattices/2d_16x16_3_obstacle.json",
        # "demos/lattices/2d_32x32_0_obstacle.json",
        # "demos/lattices/2d_32x32_1_obstacle.json",
        # "demos/lattices/2d_32x32_2_obstacle.json",
        # "demos/lattices/2d_32x32_3_obstacle.json",
    ]
    output_dir = "qlbm-output/openqasm-circuits"
    create_directory_and_parents(output_dir)

    if not exists(output_dir):
        mkdir(output_dir)

    dummy_logger = getLogger("dummy_logger")

    logging.config.fileConfig("demos/openqasm/qasm_export_logging.conf")
    logger = getLogger("qlbm")

    for lattice_file in cfgs:
        lattice_name = lattice_file.split("/")[-1].split(".")[0].replace("_", "-")
        lattice = MSLattice(lattice_file, logger=dummy_logger)

        # Compile initial condition circuit
        algorithm = MSQLBM(lattice)

        start_time_qasm3 = perf_counter_ns()
        logger.info(f"Converting {lattice_name} to QASM3...")

        with open(f"{output_dir}/{lattice_name}.qasm3", "w+") as f:
            f.write(algorithm.to_qasm3())

        end_time_qasm3 = perf_counter_ns()
        logger.info(
            f"Conversion of {lattice_name} to QASM3 took {end_time_qasm3 - start_time_qasm3} (ns)"
        )
