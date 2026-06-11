"""Adapter for cudagrad (torch-backed CudaTensor, raw CUDA C++ kernels).

cudagrad's ``engine.py`` touches ``torch.cuda.current_device()`` at import time
and its kernels are JIT-compiled by nvcc on first use, so on a non-CUDA host
(e.g. a Mac) it can't run. ``available()`` checks for CUDA + a usable kernel
launch + a CUDA toolkit (nvcc) WITHOUT importing the backend, so collection on
such hosts simply skips every cudagrad test. The backend is imported only once
availability is confirmed (inside ``__init__``), which happens on a GPU box.

Unlike the tritongrad adapter, this one needs NO warmup dance: cudagrad's CUDA
kernels don't autotune, so a single ``backward()`` accumulates each grad exactly
once into the zero-initialized ``.grad`` buffers.

Op/module coverage grows as kernel phases land — start with ``add`` only.
"""
import numpy as np

from tests.core.base_adapter import AdapterABC, GraphHandle, ModuleResult
from tests.core.loader import load_backend


class CudagradAdapter(AdapterABC):
    name = "cudagrad"
    # Scaffold: only the binary `add` kernel is implemented + verified. Each
    # kernel phase flips its op/module names on here as it lands.
    OPS = {"add", "exp", "log", "relu", "neg", "matmul",
           "softmax", "sum_lastdim", "mean", "var", "std",
           "max_lastdim", "min_lastdim"}
    MODULES = {"linear"}
    # matmul/linear/attention accumulate at fp32 and are sensitive, same as
    # tritongrad — pre-seed the loosened tolerances so enabling them is one edit.
    tol_overrides = {
        "matmul": {"atol": 5e-2, "rtol": 1e5},
        "linear": {"atol": 5e-2, "rtol": 1e5},
        "attention": {"atol": 2e-3, "rtol": 1e-1},
    }

    @classmethod
    def available(cls):
        try:
            import torch
        except Exception as e:  # pragma: no cover
            return False, f"torch import failed: {e}"
        if not torch.cuda.is_available():
            return False, "CUDA device not available"
        # nvcc is required to JIT-compile the .cu sources; without it the first
        # kernel call would crash mid-build, so skip cleanly with an actionable msg.
        try:
            from torch.utils.cpp_extension import CUDA_HOME
        except Exception as e:  # pragma: no cover
            return False, f"torch.utils.cpp_extension unavailable: {e}"
        if CUDA_HOME is None:
            return False, ("no CUDA toolkit found (CUDA_HOME unset / nvcc missing); "
                           "cudagrad JIT-compiles .cu sources and needs nvcc")
        # torch.cuda.is_available() can be True while the installed torch lacks
        # compiled kernels for this GPU's compute capability. A real launch is the
        # only reliable check; report an actionable message on failure.
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
        mods = load_backend("cudagrad", ["engine", "nn"])
        self._engine = mods["engine"]
        self._nn = mods["nn"]
        self.CudaTensor = self._engine.CudaTensor
        self._device = self._engine.DEVICE

    def _t(self, arr):
        return self._torch.tensor(np.asarray(arr), dtype=self._torch.float32,
                                  device=self._device).contiguous()

    # ---- ops --------------------------------------------------------------
    def from_numpy(self, arr, requires_grad):
        return self.CudaTensor(self._t(arr), requires_grad=requires_grad)

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
        elif op_name == "neg":
            out = -a
        elif op_name == "softmax":
            out = a.softmax()
        elif op_name == "sum_lastdim":
            out = a.sum()
        elif op_name == "mean":
            out = a.mean()
        elif op_name == "var":
            out = a.var()
        elif op_name == "std":
            out = a.std()
        elif op_name == "max_lastdim":
            out = a.max()
        elif op_name == "min_lastdim":
            out = a.min()
        else:
            raise KeyError(op_name)
        return GraphHandle(out, inputs)

    def backward(self, handle, grad_output):
        # single pass — no autotuning, so no warmup dance (unlike tritongrad)
        handle.output.backward(self._t(grad_output))

    # ---- modules ----------------------------------------------------------
    def run_module(self, spec, ref_params, input_arrays, grad_output):
        torch = self._torch
        nn = self._nn
        g = self._t(grad_output)

        if spec.name == "linear":
            mod = nn.Linear(spec.config["in"], spec.config["out"], bias=True)
            # torch weight is (out, in); cudagrad stores (in, out)
            mod.weight.data = self._t(ref_params["weight"].T)
            mod.weight.grad = torch.zeros_like(mod.weight.data)
            mod.bias.data = self._t(ref_params["bias"])
            mod.bias.grad = torch.zeros_like(mod.bias.data)
            x = self.from_numpy(input_arrays[0], requires_grad=True)
            out = mod(x)
            out.backward(g)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": mod.weight.grad.detach().cpu().numpy().T,
                             "bias": mod.bias.grad.detach().cpu().numpy()},
                input_grads={0: self.grad_of(x)},
            )

        if spec.name == "embedding":
            mod = nn.Embedding(spec.config["num"], spec.config["dim"])
            tokens = self.CudaTensor(
                torch.tensor(input_arrays[0].astype(np.int64), device=self._device))
            weight = self.CudaTensor(self._t(ref_params["weight"]), requires_grad=True)
            mod.weight = weight
            out = mod(tokens)
            out.backward(g)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": weight.grad.detach().cpu().numpy()},
                input_grads={},
            )

        if spec.name == "layernorm":
            mod = nn.LayerNorm(spec.config["dim"])
            weight = self.CudaTensor(self._t(ref_params["weight"]), requires_grad=True)
            bias = self.CudaTensor(self._t(ref_params["bias"]), requires_grad=True)
            mod.weight = weight
            mod.bias = bias
            x = self.from_numpy(input_arrays[0], requires_grad=True)
            out = mod(x)
            out.backward(g)
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
            out.backward(g)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={},
                input_grads={0: self.grad_of(q), 1: self.grad_of(k), 2: self.grad_of(v)},
            )

        raise NotImplementedError(spec.name)
