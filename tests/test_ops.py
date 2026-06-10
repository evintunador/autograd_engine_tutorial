"""Tensor-op parity: every backend's forward + backward vs PyTorch.

Parametrized over (adapter x op). A case is skipped when the backend is
unavailable (e.g. CUDA absent) or doesn't support the op.
"""
import numpy as np
import pytest
import torch

from tests.adapters import ADAPTERS
from tests.core.registry import OP_REGISTRY
from tests.core.compare import assert_close_np

CASES = [(A, op) for A in ADAPTERS for op in OP_REGISTRY]
IDS = [f"{A.name}-{op.name}" for A, op in CASES]


@pytest.mark.parametrize("Adapter,spec", CASES, ids=IDS)
def test_op(Adapter, spec, request):
    ok, reason = Adapter.available()
    if not ok:
        pytest.skip(f"{Adapter.name} unavailable: {reason}")
    adapter = Adapter()
    if not adapter.supports_op(spec.name):
        pytest.skip(f"{Adapter.name} does not support op '{spec.name}'")
    if spec.name in adapter.xfail_ops:
        request.node.add_marker(pytest.mark.xfail(reason=adapter.xfail_ops[spec.name],
                                                  strict=False))

    seed = request.config.getoption("seed")
    heat = request.config.getoption("heatmaps")
    g = np.random.default_rng(seed)
    arrs = spec.make_inputs(g)

    # ---- torch reference (fwd + bwd) ----
    t_in = [torch.tensor(a, dtype=torch.float32, requires_grad=True) for a in arrs]
    t_out = spec.torch_fn(*t_in)
    gg = np.random.default_rng(seed + 12345)
    grad_out = gg.standard_normal(tuple(t_out.shape)).astype(np.float32)
    t_out.backward(torch.tensor(grad_out))

    # ---- backend under test (fwd + seeded bwd) ----
    b_in = [adapter.from_numpy(a, True) for a in arrs]
    handle = adapter.forward_op(spec.name, b_in)
    adapter.backward(handle, grad_out)

    atol, rtol = adapter.tol_for(spec.name, spec.atol, spec.rtol)
    assert_close_np(adapter.to_numpy(handle.output), t_out.detach().numpy(),
                    f"{adapter.name}:{spec.name}:fwd", atol, rtol, heat)
    for i, leaf in enumerate(b_in):
        assert_close_np(adapter.grad_of(leaf), t_in[i].grad.numpy(),
                        f"{adapter.name}:{spec.name}:grad{i}", atol, rtol, heat)
