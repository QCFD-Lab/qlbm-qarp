from os.path import exists

from qarp.absorb import QASM3Absorber

from qlbm.tools.utils import get_circuit_properties

if __name__ == "__main__":
    # Written by demos/openqasm/qasm_export.py.  qarp's OpenQASM 2 emitter
    # rejects the multi-controlled gates of the algorithms, so only QASM 3 exists.
    qasm_circuit_dir = "qlbm-output/openqasm-circuits"

    if not exists(qasm_circuit_dir):
        raise RuntimeError(f'Directory "{qasm_circuit_dir}" does not exist.')

    with open(f"{qasm_circuit_dir}/2d-16x16-2-obstacle.qasm3") as f:
        circuit = QASM3Absorber().absorb(f.read())
    print(get_circuit_properties(circuit))
