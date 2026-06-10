"""Numeric comparison helper shared by all tests.

Everything is marshalled to numpy before comparison, so this one function works
for every backend. On failure it optionally emits diff heatmap PNGs (opt-in via
``--heatmaps``), mirroring the visualization from ``tritongrad/testing.py``.
"""
import numpy as np


def assert_close_np(actual, expected, name: str, atol: float, rtol: float,
                    heatmaps: bool = False):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"[{name}] shape mismatch: backend {actual.shape} vs torch {expected.shape}"
        )

    try:
        np.testing.assert_allclose(
            actual, expected, atol=atol, rtol=rtol,
            err_msg=f"[{name}] value mismatch (atol={atol}, rtol={rtol})",
        )
    except AssertionError:
        if heatmaps:
            from tests.core.heatmaps import save_heatmaps
            save_heatmaps(expected, actual, name.replace(":", "_"),
                          atol=atol, rtol=rtol)
        raise
