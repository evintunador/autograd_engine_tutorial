"""nn-module parity: every backend's module fwd + param/input grads vs PyTorch.

The torch reference module is built and its parameters read in canonical layout;
the adapter loads those same params into its own module (handling weight-layout
differences) so the comparison is apples-to-apples.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.adapters import ADAPTERS
from tests.core.registry import MODULE_REGISTRY
from tests.core.compare import assert_close_np

CASES = [(A, m) for A in ADAPTERS for m in MODULE_REGISTRY]
IDS = [f"{A.name}-{m.name}" for A, m in CASES]


@pytest.mark.parametrize("Adapter,spec", CASES, ids=IDS)
def test_module(Adapter, spec, request):
    ok, reason = Adapter.available()
    if not ok:
        pytest.skip(f"{Adapter.name} unavailable: {reason}")
    adapter = Adapter()
    if not adapter.supports_module(spec.name):
        pytest.skip(f"{Adapter.name} does not support module '{spec.name}'")
    if spec.name in adapter.xfail_modules:
        request.node.add_marker(pytest.mark.xfail(reason=adapter.xfail_modules[spec.name],
                                                  strict=False))

    seed = request.config.getoption("seed")
    heat = request.config.getoption("heatmaps")
    g = np.random.default_rng(seed)
    inputs = spec.make_inputs(g)
    gg = np.random.default_rng(seed + 777)

    # ---- torch reference ----
    if spec.kind == "attention":
        q, k, v = [torch.tensor(a, dtype=torch.float32, requires_grad=True) for a in inputs]
        t_out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               scale=spec.config["scale"])
        grad_out = gg.standard_normal(tuple(t_out.shape)).astype(np.float32)
        t_out.backward(torch.tensor(grad_out))
        ref_params = {}
        ref_param_grads = {}
        ref_input_grads = {0: q.grad.numpy(), 1: k.grad.numpy(), 2: v.grad.numpy()}
    else:
        ref = spec.build_torch(spec.config)
        named = dict(ref.named_parameters())
        ref_params = {n: named[n].detach().numpy().copy() for n in spec.param_names}
        t_inputs, float_idx = [], []
        for i, a in enumerate(inputs):
            if np.issubdtype(a.dtype, np.integer):
                t_inputs.append(torch.tensor(a, dtype=torch.long))
            else:
                ti = torch.tensor(a, dtype=torch.float32, requires_grad=True)
                t_inputs.append(ti)
                float_idx.append(i)
        t_out = ref(*t_inputs)
        grad_out = gg.standard_normal(tuple(t_out.shape)).astype(np.float32)
        t_out.backward(torch.tensor(grad_out))
        ref_param_grads = {n: named[n].grad.numpy().copy() for n in spec.param_names}
        ref_input_grads = {i: t_inputs[i].grad.numpy() for i in float_idx}

    # ---- backend under test ----
    res = adapter.run_module(spec, ref_params, inputs, grad_out)

    atol, rtol = adapter.tol_for(spec.name, spec.atol, spec.rtol)
    assert_close_np(res.out, t_out.detach().numpy(),
                    f"{adapter.name}:{spec.name}:fwd", atol, rtol, heat)
    for n in spec.param_names:
        assert_close_np(res.param_grads[n], ref_param_grads[n],
                        f"{adapter.name}:{spec.name}:dparam[{n}]", atol, rtol, heat)
    for i, gnp in ref_input_grads.items():
        if i in res.input_grads:
            assert_close_np(res.input_grads[i], gnp,
                            f"{adapter.name}:{spec.name}:dinput[{i}]", atol, rtol, heat)
