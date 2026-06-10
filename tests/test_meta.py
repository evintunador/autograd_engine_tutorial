"""Sanity checks on the harness itself: adapters conform, registry is well-formed."""
import numpy as np

from tests.adapters import ADAPTERS
from tests.core.base_adapter import AdapterABC
from tests.core.registry import OP_REGISTRY, MODULE_REGISTRY


def test_adapters_conform():
    names = set()
    for A in ADAPTERS:
        assert issubclass(A, AdapterABC), f"{A} is not an AdapterABC"
        assert isinstance(A.name, str) and A.name, f"{A} has no name"
        assert A.name not in names, f"duplicate adapter name {A.name}"
        names.add(A.name)
        assert isinstance(A.OPS, set)
        assert isinstance(A.MODULES, set)
        ok, reason = A.available()
        assert isinstance(ok, bool)
        if not ok:
            assert reason, f"{A.name} unavailable but gave no reason"


def test_registry_well_formed():
    op_names = [o.name for o in OP_REGISTRY]
    assert len(op_names) == len(set(op_names)), "duplicate op names"
    mod_names = [m.name for m in MODULE_REGISTRY]
    assert len(mod_names) == len(set(mod_names)), "duplicate module names"

    g = np.random.default_rng(0)
    for op in OP_REGISTRY:
        arrs = op.make_inputs(g)
        assert isinstance(arrs, list) and arrs
        for a in arrs:
            assert isinstance(a, np.ndarray)


def test_adapter_ops_are_registered():
    """Every op/module an adapter claims must exist in the registry."""
    known_ops = {o.name for o in OP_REGISTRY}
    known_mods = {m.name for m in MODULE_REGISTRY}
    for A in ADAPTERS:
        assert A.OPS <= known_ops, f"{A.name} claims unknown ops {A.OPS - known_ops}"
        assert A.MODULES <= known_mods, f"{A.name} claims unknown modules {A.MODULES - known_mods}"
