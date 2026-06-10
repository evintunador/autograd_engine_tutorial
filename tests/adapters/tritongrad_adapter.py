"""Adapter for tritongrad (torch-backed TritonTensor, CUDA + Triton).

tritongrad's ``engine.py`` touches ``torch.cuda.current_device()`` at import time,
so on a non-CUDA host (e.g. a Mac) importing it crashes. ``available()`` checks
for CUDA/Triton WITHOUT importing the backend, so collection on such hosts simply
skips every tritongrad test. The backend is only imported once availability is
confirmed (inside ``__init__``), which happens on a GPU box.

This adapter is therefore exercised on GPU only; on a Mac it is never
instantiated. Its op/module wiring mirrors tritongrad/testing.py.
"""
import math

import numpy as np

from tests.core.base_adapter import AdapterABC, GraphHandle, ModuleResult
from tests.core.loader import load_backend


class TritongradAdapter(AdapterABC):
    name = "tritongrad"
    # NOTE: no "softmax" — tritongrad's TritonTensor.softmax() is an unimplemented
    # stub (`pass`). Reductions are last-dim only, which the registry already uses.
    OPS = {"add", "sub", "mul", "div", "matmul", "exp", "log", "relu",
           "sum_lastdim", "mean", "var"}
    MODULES = {"linear", "embedding", "layernorm", "attention"}
    # matmul/linear gradient accumulation and the many-op attention kernel are
    # sensitive at fp32, exactly as documented in tritongrad/testing.py.
    tol_overrides = {
        "matmul": {"atol": 5e-2, "rtol": 1e5},
        "linear": {"atol": 5e-2, "rtol": 1e5},
        "attention": {"atol": 2e-3, "rtol": 1e-1},
    }
    # Known tritongrad bug surfaced by this suite: the var FORWARD reduction
    # subtracts sum(x) instead of mean(x) (kernels/vectorwise.py:58 is missing the
    # `/ row_len`), so the forward value is garbage. (The var backward kernel just
    # below computes the mean correctly.) tritongrad also uses sample variance
    # /(n-1) while torch's default is population /n.
    xfail_ops = {"var": "tritongrad var forward subtracts sum(x) not mean(x) "
                        "(kernels/vectorwise.py:58 missing /row_len)"}

    @classmethod
    def available(cls):
        try:
            import torch
        except Exception as e:  # pragma: no cover
            return False, f"torch import failed: {e}"
        if not torch.cuda.is_available():
            return False, "CUDA device not available"
        try:
            import triton  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"triton import failed: {e}"
        # torch.cuda.is_available() can be True while the installed torch lacks
        # compiled kernels for this GPU's compute capability (e.g. a Blackwell
        # sm_120 card with a torch built only up to sm_90). A real kernel launch
        # is the only reliable check; catch it here so the suite skips with an
        # actionable message instead of failing every case.
        try:
            (torch.ones(8, device="cuda") + 1.0).sum().item()
        except Exception as e:  # pragma: no cover
            cap = "".join(str(x) for x in torch.cuda.get_device_capability())
            return False, (
                f"CUDA present but torch cannot launch kernels on this GPU "
                f"(sm_{cap}): {e}. Install a torch build that supports it "
                f"(e.g. --index-url https://download.pytorch.org/whl/cu128)."
            )
        return True, ""

    def __init__(self):
        import torch
        self._torch = torch
        mods = load_backend("tritongrad", ["engine", "nn"])
        self._engine = mods["engine"]
        self._nn = mods["nn"]
        self.TritonTensor = self._engine.TritonTensor
        self._device = self._engine.DEVICE

    def _t(self, arr):
        return self._torch.tensor(np.asarray(arr), dtype=self._torch.float32,
                                  device=self._device).contiguous()

    # ---- ops --------------------------------------------------------------
    def from_numpy(self, arr, requires_grad):
        return self.TritonTensor(self._t(arr), requires_grad=requires_grad)

    def to_numpy(self, x):
        return x.data.detach().cpu().numpy()

    def grad_of(self, leaf):
        return leaf.grad.detach().cpu().numpy()

    def forward_op(self, op_name, inputs):
        a = inputs[0]
        b = inputs[1] if len(inputs) > 1 else None
        if op_name == "add":
            out = a + b
        elif op_name == "sub":
            out = a - b
        elif op_name == "mul":
            out = a * b
        elif op_name == "div":
            out = a / b
        elif op_name == "matmul":
            out = a @ b
        elif op_name == "exp":
            out = a.exp()
        elif op_name == "log":
            out = a.log()
        elif op_name == "relu":
            out = a.relu()
        elif op_name == "sum_lastdim":
            out = a.sum()
        elif op_name == "mean":
            out = a.mean()
        elif op_name == "var":
            out = a.var()
        else:
            raise KeyError(op_name)
        return GraphHandle(out, inputs)

    def _seeded_backward(self, out, grad_output):
        """Drive tritongrad's backward correctly under Triton autotuning.

        tritongrad's backward kernels accumulate into ``.grad`` with ``+=``, and
        Triton's autotuner runs each config many times on a kernel's first call —
        so a naive single ``backward()`` adds the gradient thousands of times. We
        first run a backward seeded with ZEROS (autotuning runs, but accumulates
        nothing), reset all grads, then run the real backward once (configs now
        cached). This mirrors the warmup dance in tritongrad/testing.py.
        """
        g = self._t(grad_output)
        out.backward(self._torch.zeros_like(g))
        out.zero_grad_backward()
        out.backward(g)

    def backward(self, handle, grad_output):
        self._seeded_backward(handle.output, grad_output)

    # ---- modules ----------------------------------------------------------
    def run_module(self, spec, ref_params, input_arrays, grad_output):
        torch = self._torch
        nn = self._nn

        if spec.name == "linear":
            mod = nn.Linear(spec.config["in"], spec.config["out"], bias=True)
            # torch weight is (out, in); tritongrad stores (in, out)
            mod.weight.data = self._t(ref_params["weight"].T)
            mod.weight.grad = torch.zeros_like(mod.weight.data)
            mod.bias.data = self._t(ref_params["bias"])
            mod.bias.grad = torch.zeros_like(mod.bias.data)
            x = self.from_numpy(input_arrays[0], requires_grad=True)
            out = mod(x)
            self._seeded_backward(out, grad_output)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": mod.weight.grad.detach().cpu().numpy().T,
                             "bias": mod.bias.grad.detach().cpu().numpy()},
                input_grads={0: self.grad_of(x)},
            )

        if spec.name == "embedding":
            mod = nn.Embedding(spec.config["num"], spec.config["dim"])
            # tritongrad's Embedding reads tokens.data/.shape/.numel, so wrap tokens
            tokens = self.TritonTensor(
                torch.tensor(input_arrays[0].astype(np.int64), device=self._device))
            weight = self.TritonTensor(self._t(ref_params["weight"]), requires_grad=True)
            mod.weight = weight
            out = mod(tokens)
            self._seeded_backward(out, grad_output)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": weight.grad.detach().cpu().numpy()},
                input_grads={},
            )

        if spec.name == "layernorm":
            mod = nn.LayerNorm(spec.config["dim"])
            weight = self.TritonTensor(self._t(ref_params["weight"]), requires_grad=True)
            bias = self.TritonTensor(self._t(ref_params["bias"]), requires_grad=True)
            mod.weight = weight
            mod.bias = bias
            x = self.from_numpy(input_arrays[0], requires_grad=True)
            out = mod(x)
            self._seeded_backward(out, grad_output)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": weight.grad.detach().cpu().numpy(),
                             "bias": bias.grad.detach().cpu().numpy()},
                input_grads={0: self.grad_of(x)},
            )

        if spec.name == "attention":
            q = self.from_numpy(input_arrays[0], requires_grad=True)
            k = self.from_numpy(input_arrays[1], requires_grad=True)
            v = self.from_numpy(input_arrays[2], requires_grad=True)
            out = nn.FlashAttention()(q, k, v, scale=spec.config["scale"])
            self._seeded_backward(out, grad_output)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={},
                input_grads={0: self.grad_of(q), 1: self.grad_of(k), 2: self.grad_of(v)},
            )

        raise NotImplementedError(spec.name)
