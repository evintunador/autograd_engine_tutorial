"""Opt-in failure visualization (numpy-only port of tritongrad/testing.py).

When a comparison fails and ``--heatmaps`` is set, we write PNGs showing where
the backend output diverges from the torch reference: raw absolute difference,
absolute-tolerance failure mask, and relative-tolerance failure mask. Handles
2D/3D/4D tensors by saving one set per leading batch/head slice.
"""
import os

import numpy as np

FOLDER = os.path.join(os.path.dirname(__file__), "..", "heatmaps")


def save_heatmaps(expected: np.ndarray, actual: np.ndarray, test_name: str,
                  atol: float = 1e-3, rtol: float = 1e-3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FOLDER, exist_ok=True)

    expected = np.asarray(expected)
    actual = np.asarray(actual)
    abs_diff = np.abs(expected - actual)
    abs_fail = (abs_diff > atol).astype(np.int32)
    rel_fail = (abs_diff > rtol * np.abs(expected)).astype(np.int32)

    def save_figure(matrix, title, filename, cmap="hot"):
        matrix = np.atleast_2d(matrix)
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap=cmap, aspect="auto")
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(FOLDER, filename))
        plt.close()

    def save_set(diff, am, rm, suffix):
        save_figure(diff, f"{test_name} {suffix} raw diff", f"{test_name}_{suffix}_raw.png")
        save_figure(am, f"{test_name} {suffix} abs-fail", f"{test_name}_{suffix}_absfail.png", cmap="Reds")
        save_figure(rm, f"{test_name} {suffix} rel-fail", f"{test_name}_{suffix}_relfail.png", cmap="Reds")

    if expected.ndim == 4:
        B, H = expected.shape[:2]
        for b in range(B):
            for h in range(H):
                save_set(abs_diff[b, h], abs_fail[b, h], rel_fail[b, h], f"b{b}_h{h}")
    elif expected.ndim == 3:
        for b in range(expected.shape[0]):
            save_set(abs_diff[b], abs_fail[b], rel_fail[b], f"b{b}")
    else:
        save_set(abs_diff, abs_fail, rel_fail, "diff")
