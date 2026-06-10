"""Isolated importing of each backend's modules.

micrograd, minigrad and tritongrad each ship top-level modules with the SAME
names (``engine``, ``ops``, ``nn``, ``modules``) and rely on bare imports like
``from engine import Value``. Importing two of them naively would collide in
``sys.modules``. ``load_backend`` imports a backend's modules inside a window
where only that backend's directory is on ``sys.path`` and any conflicting cached
module names are temporarily removed, then restores ``sys.modules`` afterwards.
The returned module objects keep working because their intra-package references
were bound during the window.
"""
import importlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_CACHE = {}


def load_backend(subdir: str, names: list) -> dict:
    """Import ``names`` (in order) from ``REPO_ROOT/subdir`` in isolation.

    Returns ``{name: module}``. Results are cached per (subdir, names) so repeated
    calls across tests don't re-import.
    """
    key = (subdir, tuple(names))
    if key in _CACHE:
        return _CACHE[key]

    dirpath = os.path.join(REPO_ROOT, subdir)
    saved_path = list(sys.path)
    saved_mods = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, dirpath)
    out = {}
    try:
        for n in names:
            out[n] = importlib.import_module(n)
    finally:
        sys.path[:] = saved_path
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved_mods)

    _CACHE[key] = out
    return out
