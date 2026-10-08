"""Location of the differential fixtures.

The fixtures are generated from the qiskit-based qlbm at :data:`REFERENCE_COMMIT`.
Files up to :data:`COMMITTED_LIMIT` bytes live in ``fixtures/`` and are
committed; larger ones go to ``fixtures/large/``, which is gitignored.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np

REFERENCE_COMMIT = "7f8ef844b128a81062c94b31662b82b0b22ee135"
COMMITTED_LIMIT = 200_000
FIXTURE_DIR = Path(__file__).parent / "fixtures"
LARGE_DIR = FIXTURE_DIR / "large"


def _route(path: Path) -> Path:
    # A fixture lives in exactly one of the two directories.
    large = LARGE_DIR / path.name
    if path.stat().st_size > COMMITTED_LIMIT:
        LARGE_DIR.mkdir(parents=True, exist_ok=True)
        path.replace(large)
        return large
    large.unlink(missing_ok=True)
    return path


def save_npz(case_id: str, **arrays: Any) -> Path:
    """Write ``arrays`` as the compressed fixture of ``case_id``."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{case_id}.npz"
    np.savez_compressed(path, **arrays)
    return _route(path)


def save_text(filename: str, text: str) -> Path:
    """Write a text fixture named ``filename``."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / filename
    path.write_text(text)
    return _route(path)


def find(filename: str) -> Optional[Path]:
    """The path of fixture ``filename``, or ``None`` when it was not generated."""
    for directory in (FIXTURE_DIR, LARGE_DIR):
        if (directory / filename).exists():
            return directory / filename
    return None
