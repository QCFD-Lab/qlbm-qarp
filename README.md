# `qlbm`

![GitHub License](https://img.shields.io/github/license/qcfd-lab/qlbm?color=%2300A6D6) ![GitHub top language](https://img.shields.io/github/languages/top/qcfd-lab/qlbm?color=%2300A6D6) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FQCFD-Lab%2Fqlbm%2Frefs%2Fheads%2Fdev%2Fpyproject.toml?color=%2300A6D6) ![PyPI - Version](https://img.shields.io/pypi/v/qlbm?color=%2300A6D6) ![GitHub commits since latest release](https://img.shields.io/github/commits-since/qcfd-lab/qlbm/latest?color=%2300A6D6) ![GitHub branch check runs](https://img.shields.io/github/check-runs/qcfd-lab/qlbm/main?color=%2300A6D6) <a href="https://arxiv.org/abs/2411.19439">![Static Badge](https://img.shields.io/badge/preprint-blue?style=flat&label=arXiv&color=%2300A6D6)</a>

`qlbm` is a package for the development, simulation, and analysis of **Q**uantum **L**attice **B**oltzmann **M**ethods.

---

`qlbm` is a rapidly evolving, research-oriented piece of software. It contains building blocks for constructing quantum circuits for quantum LBMs and connects these with quantum software infrastructure. `qlbm` is built with end-to-end development environment in mind, including:

- Parsing human-readable `JSON` specifications for QLBMs
- Constructing quantum circuits in `qarp`/`qarpx` that implement QLBMs
- Compiling and optimizing those circuits through `qarp`'s block optimizer
- Simulating quantum circuits on classical hardware with the `qarpx` simulator
- Visualizing results in [ParaView](https://www.paraview.org/)
- Analyzing the properties , scalability, and performance of quantum algorithms

<p align="center">
<a href="https://qcfd-lab.github.io/qlbm/">
<img width=400 centered alt="Static Badge" src="https://img.shields.io/badge/Documentation-00A6D6%20?style=flat&logo=BookStack&logoColor=%23FFFFFF&logoSize=10&label=Web&color=%2300A6D6&">
</a>
</p>

## PyPI installation

`qlbm` can be installed through `pip`. We recommend the use of a Python 3.12, 3.13 or 3.14 virtual environment:

```bash
python -m venv qlbm-venv
pip install --upgrade pip
pip install qlbm
```

Note that `qlbm` evolves quickly and it is likely that the GitHub repository contains new features that the PyPI installation does not. To get the latest developments, we recommend the source installation.

## Install from source

`qlbm` builds its circuits on [OpenQARP](https://github.com/OpenQARP/openqarp) (`openqarp` on PyPI), which ships a compiled statevector simulator, so no further quantum SDK is needed. Alternatively, you can install the latest version of `qlbm` by cloning the repository and installing from source as follows (again using Python 3.12, 3.13 or 3.14):

```bash
git clone https://github.com/QCFD-Lab/qlbm.git
cd qlbm
python -m venv qlbm-venv
source qlbm-venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,docs]
```

If you are using `zsh` (which is the default shell on macOS) you need to replace the last line by

```bash
pip install -e .\[dev,docs\]
```

We also provide a `make` script for this purpose, which will create the environment from scratch:

```bash
make install
source qlbm-venv/bin/activate
```

To override the default Python executable, pass `PYTHON` on the command line:

```bash
make install PYTHON=your-python-binary
```

## Container installation

The `Docker` directory contains a Dockerfile for running `qlbm` in a containerized environment.

Build the image from the repository root:

```bash
docker build -f ./Docker/build_cpu.Dockerfile -t qlbm-cpu .
```

After the build completes, start an interactive container and mount the local `qlbm` source directory as read-only:

```bash
docker run --rm -it \
  -v "$(pwd)/qlbm:/qlbm/qlbm:ro" \
  qlbm-cpu
```

On systems using SELinux, add the `Z` option to relabel the mounted directory:

```bash
docker run --rm -it \
  -v "$(pwd)/qlbm:/qlbm/qlbm:ro,Z" \
  qlbm-cpu
```

## Components are OpenQARP blocks

Every `qlbm` component is an OpenQARP block: primitives extend `qarp.blocks.SimpleBlock` and emit gates in `build_vanilla`; operators and algorithms extend `qarp.blocks.CompositeBlockBase` and place child blocks. Components are built on construction, so input validation raises in the constructor. A component has `n_qubits`, `n_cbits`, `flatten()`, `statevector()`, `unitary_matrix()`, `plot()`, `to_qasm3()` and every other block method, and `ControlledBlock(op, k)`, `~op` and `op ** k` apply to it directly.

```python
from qlbm.components.ms import MSQLBM
from qlbm.lattice import MSLattice

lattice = MSLattice("demos/lattices/2d_8x8_1_obstacle.json")
algorithm = MSQLBM(lattice)
print(algorithm.n_qubits, len(algorithm.flatten()))
```

Simulation goes through `SimulationConfig` and `QarpRunner`. The runner carries the statevector between time steps, samples counts from it directly, and accepts `num_shots=qarp.EXACT` to return exact probabilities instead of sampled counts. Counts are keyed by the classical-bit integer (bit 0 least significant); the values are shot counts, or probabilities under `qarp.EXACT`.

Operators and algorithms are built from `LBMPrimitive`, `LBMComposite`, `LBMOperator`, `LBMAlgorithm` and `ControllableComponent` in `qlbm.components.base`. The QFT arithmetic OpenQARP does not ship (`DraperQFTAdder`, `RGQFTMultiplier`, `ccp`) lives in `qlbm.components.common.arithmetic`. GPU execution is not wired into the runner.

### Testing

`test/unit/differential/` compares components, lattices, the runner and the result decoders against fixtures generated from the qiskit-based `qlbm` at commit `7f8ef844` by the `test/oracle/generate_*.py` scripts. Fixtures up to 200 KB are committed; the larger ones are written to `test/oracle/fixtures/large/`, which is gitignored, and the tests that need them skip when they are absent. To regenerate all fixtures, install the reference in a separate environment and run each script from the repository root:

```bash
python -m venv ../qlbm-oracle-venv
../qlbm-oracle-venv/bin/pip install "qlbm[cpu] @ git+https://github.com/QCFD-Lab/qlbm@7f8ef844b128a81062c94b31662b82b0b22ee135"
../qlbm-oracle-venv/bin/python test/oracle/generate_fixtures.py
```

## Algorithms and Usage

Currently, `qlbm` supports two algorithms:
 - The Quantum Transport Method (Collisionless QLBM) described in [Efficient and fail-safe quantum algorithm for the transport equation](https://doi.org/10.1016/j.jcp.2024.112816) ([arXiv:2211.14269](https://arxiv.org/abs/2211.14269)) by M.A. Schalkers and M. Möller.
 - The Space-Time QLBM/QLGA described in [On the importance of data encoding in quantum Boltzmann methods](https://link.springer.com/article/10.1007/s11128-023-04216-6) by M.A. Schalkers and M. Möller and expanded in [Fully Quantum Lattice Gas Automata Building Blocks for Computational Basis State Encodings](https://doi.org/10.1016/j.jcp.2025.114595).
 - The Linear-encoding Quantum Lattice Gas Automata (LQLGA) described in [On quantum extensions of hydrodynamic lattice gas automata](https://www.mdpi.com/2410-3896/4/2/48) by P. Love and [Fully Quantum Lattice Gas Automata Building Blocks for Computational Basis State Encodings](https://doi.org/10.1016/j.jcp.2025.114595).

The `demos` directory contains several use cases for simulating and analyzing these algorithms. Each demo requires minimal setup once the virtual environment has been configured. Consult the `README.md` file in the `demos` directory for further details.

> **Note on visualization**: we rely on  ParaView for visualizing the flow field of the simulation. You can install ParaView from [this link](https://www.paraview.org/download/).

## Configuration

`qlbm` uses quantum circuits to simulate systems that users can specify in simple `JSON` configuration files. For instance, the following configuration describes a 2D system of 64x32 gridpoints, 4 discrete velocities per dimension, and with 3 solid objects placed in the fluid domain:

```JSON
{
  "lattice": {
    "dim": {
      "x": 64,
      "y": 32
    },
    "velocities": {
      "x": 4,
      "y": 4
    }
  },
  "geometry": [
    { 
      "shape": "cuboid",
      "x": [18, 20],
      "y": [6, 25],
      "boundary": "specular"
    },
    {
      "shape": "cuboid",
      "x": [23, 25],
      "y": [3, 17],
      "boundary": "bounceback"
    },
    {
      "shape": "cuboid",
      "x": [28, 29],
      "y": [16, 29],
      "boundary": "specular"
    }
  ]
}
```

## Citation

An open access peer-reviewed article describing `qlbm` is available [here](https://doi.org/10.1016/j.cpc.2025.109699). If you use `qlbm`, you can cite it as:

```
@article{georgescu2025qlbm,
title = {qlbm – A quantum lattice Boltzmann software framework},
journal = {Computer Physics Communications},
volume = {315},
pages = {109699},
year = {2025},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2025.109699},
url = {https://www.sciencedirect.com/science/article/pii/S0010465525002012},
author = {C\u{{a}}lin A. Georgescu and Merel A. Schalkers and Matthias M\"{o}ller},
}
```

## Contact

In addition to opening issues, you can contact the developers of `qlbm` at `qcfd-EWI@tudelft.nl`.
