"""Capture the field a result decoder would write, without writing files."""

import numpy as np


def decode_field(result, counts, n_cbits) -> np.ndarray:
    """Run ``result``'s decoder on ``counts`` and return the field array."""
    captured = {}

    def capture(numpy_res, *_args, **_kwargs):
        captured["field"] = np.asarray(numpy_res, dtype=np.float64)

    result.save_timestep_array = capture
    result.save_timestep_counts(counts, 0, create_vis=False, n_cbits=n_cbits)
    return captured["field"]
