r"""Register bookkeeping for lattice qubit layouts.

A :class:`Register` is a named, contiguous slice of the global qubit index
space; iteration and indexing yield *global* qubit indices (plain ``int``\ s).
Offsets are cumulative in register declaration order and are assigned by
:func:`assign_offsets` after the flat register tuple is assembled.
"""

from typing import Iterable, Iterator, List, Tuple, Union


class Register:
    """A named group of qubits at a fixed offset in the global index space.

    Supports ``size``, ``name``, ``len()``, iteration and indexing; element
    access yields global qubit indices as ``int``.

    Parameters
    ----------
    size : int
        The number of qubits in the register.
    name : str
        The register label (e.g. ``"g_x"``).
    offset : int, optional
        The global index of the register's first qubit, by default ``0``.
        Normally assigned after construction via :func:`assign_offsets`.
    """

    def __init__(self, size: int, name: str, offset: int = 0) -> None:
        if size < 0:
            raise ValueError(f"Register size must be non-negative, got {size}.")
        self.size = size
        self.name = name
        self.offset = offset

    def __len__(self) -> int:
        """Return the number of qubits in the register."""
        return self.size

    def __iter__(self) -> Iterator[int]:
        """Iterate over the global qubit indices of the register."""
        return iter(range(self.offset, self.offset + self.size))

    def __getitem__(self, key: Union[int, slice]) -> Union[int, List[int]]:
        """Return the global qubit index (or indices) at ``key``.

        Parameters
        ----------
        key : int | slice
            The register-local position(s); negative indices are supported.

        Returns
        -------
        int | List[int]
            The global qubit index for an ``int`` key, or a list of global
            qubit indices for a ``slice`` key.
        """
        if isinstance(key, slice):
            return list(range(self.offset, self.offset + self.size))[key]
        index = key + self.size if key < 0 else key
        if not 0 <= index < self.size:
            raise IndexError(
                f"Index {key} out of range for register '{self.name}' of size {self.size}."
            )
        return self.offset + index

    def __eq__(self, other: object) -> bool:
        """Compare two registers by size, name, and offset."""
        if not isinstance(other, Register):
            return NotImplemented
        return (
            self.size == other.size
            and self.name == other.name
            and self.offset == other.offset
        )

    def __repr__(self) -> str:
        """Return a debug representation of the register."""
        return f"Register({self.size}, '{self.name}', offset={self.offset})"


def assign_offsets(registers: Iterable[Register]) -> Tuple[Register, ...]:
    """Assign cumulative offsets to ``registers`` in declaration order.

    The declaration order defines every component's qubit layout.

    Parameters
    ----------
    registers : Iterable[Register]
        The flat register sequence, in declaration order.

    Returns
    -------
    Tuple[Register, ...]
        The same register objects, with ``offset`` set, as a tuple.
    """
    result = tuple(registers)
    offset = 0
    for register in result:
        register.offset = offset
        offset += register.size
    return result
