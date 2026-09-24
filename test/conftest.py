"""Collection settings shared by every test directory."""

# The fixture generators run against the qiskit-based qlbm in a separate
# environment and import qiskit at module level; ``--doctest-modules`` must
# not import them.
collect_ignore_glob = ["oracle/generate_*.py"]
