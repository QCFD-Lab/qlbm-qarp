"""Fixture access for the differential tests; a missing fixture skips the test."""

import json
from typing import Any

import numpy as np
import pytest

from test.oracle.fixture_store import find


def load_npz(case_id: str, generator: str) -> Any:
    """The arrays of ``case_id``, written by ``test/oracle/<generator>.py``."""
    path = find(f"{case_id}.npz")
    if path is None:
        pytest.skip(
            f"Missing oracle fixture {case_id}.npz; run test/oracle/{generator}.py"
        )
    return np.load(path)


def load_json(filename: str, generator: str) -> Any:
    """The decoded JSON fixture ``filename``, written by ``test/oracle/<generator>.py``."""
    path = find(filename)
    if path is None:
        pytest.skip(
            f"Missing oracle fixture {filename}; run test/oracle/{generator}.py",
            allow_module_level=True,
        )
    return json.loads(path.read_text())
