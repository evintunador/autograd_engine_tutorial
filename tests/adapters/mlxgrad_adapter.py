"""Adapter for mlxgrad (MLX-array-backed MLXTensor, raw Metal kernels).

Unlike tritongrad/cudagrad (which need an NVIDIA GPU and so only run over SSH on
a cloud box), mlxgrad runs on Apple Silicon — so on the dev Mac these tests
actually execute. ``available()`` gates on ``mx.metal.is_available()`` WITHOUT
importing the backend, so non-Apple hosts skip every mlxgrad test cleanly.

Like cudagrad, this needs NO warmup dance: ``mx.fast.metal_kernel`` does not
autotune, so a single ``backward()`` accumulates each grad exactly once. (The
accumulation is functional — backward kernels return ``grad_in + contribution``
and the engine rebinds ``.grad`` — but that's internal to the engine; the adapter
just drives one backward pass.)

Op/module coverage grows as kernel phases land — start with ``add`` only.
"""
import numpy as np

from tests.core.base_adapter import AdapterABC, GraphHandle, ModuleResult
from tests.core.loader import load_backend


class MlxgradAdapter(AdapterABC):
    name = "mlxgrad"
    # elementwise + vectorwise + matmul complete; modules phase adds the rest.
    OPS = {"add", "sub", "mul", "div", "exp", "log", "relu", "neg", "matmul",
           "softmax", "sum_lastdim", "mean", "var", "std",
           "max_lastdim", "min_lastdim"}
    MODULES = {"linear", "embedding", "layernorm", "attention"}
    # matmul/linear/attention accumulate at fp32 and are sensitive, same as
    # cudagrad/tritongrad — pre-seed the loosened tolerances so enabling them is
    # one edit when those phases land.
    tol_overrides = {
        "matmul": {"atol": 5e-2, "rtol": 1e5},
        "linear": {"atol": 5e-2, "rtol": 1e5},
        "attention": {"atol": 2e-3, "rtol": 1e-1},
    }

    @classmethod
    def available(cls):
        try:
            import mlx.core as mx
        except Exception as e:  # pragma: no cover
            return False, f"mlx import failed: {e} (pip install mlx; Apple Silicon only)"
        try:
            if not mx.metal.is_available():
                return False, "Metal GPU not available (mlx needs Apple Silicon)"
        except Exception as e:  # pragma: no cover
            return False, f"mlx.metal unavailable: {e}"
        # A real launch is the only reliable check that kernels actually run here.
        try:
            mx.eval(mx.ones(8) + 1.0)
        except Exception as e:  # pragma: no cover
            return False, f"mlx present but cannot run on this device: {e}"
        return True, ""

    def __init__(self):
        import mlx.core as mx
        self._mx = mx
        mods = load_backend("mlxgrad", ["engine", "nn"])
        self._engine = mods["engine"]
        self._nn = mods["nn"]
        self.MLXTensor = self._engine.MLXTensor

    def _a(self, arr):
        return self._mx.array(np.asarray(arr, dtype=np.float32))

    # ---- ops --------------------------------------------------------------
    def from_numpy(self, arr, requires_grad):
        return self.MLXTensor(self._a(arr), requires_grad=requires_grad)

    def to_numpy(self, x):
        return np.array(x.data)  # forces MLX lazy eval + converts

    def grad_of(self, leaf):
        return np.array(leaf.grad)

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
        handle.output.backward(self._a(grad_output))

    # ---- modules ----------------------------------------------------------
    def run_module(self, spec, ref_params, input_arrays, grad_output):
        mx = self._mx
        nn = self._nn
        g = self._a(grad_output)

        if spec.name == "linear":
            mod = nn.Linear(spec.config["in"], spec.config["out"], bias=True)
            # torch weight is (out, in); mlxgrad stores (in, out)
            mod.weight.data = self._a(ref_params["weight"].T)
            mod.weight.grad = mx.zeros_like(mod.weight.data)
            mod.bias.data = self._a(ref_params["bias"])
            mod.bias.grad = mx.zeros_like(mod.bias.data)
            x = self.from_numpy(input_arrays[0], requires_grad=True)
            out = mod(x)
            out.backward(g)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": np.array(mod.weight.grad).T,
                             "bias": np.array(mod.bias.grad)},
                input_grads={0: self.grad_of(x)},
            )

        if spec.name == "embedding":
            mod = nn.Embedding(spec.config["num"], spec.config["dim"])
            tokens = self.MLXTensor(mx.array(input_arrays[0].astype(np.int32)))
            weight = self.MLXTensor(self._a(ref_params["weight"]), requires_grad=True)
            mod.weight = weight
            out = mod(tokens)
            out.backward(g)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": np.array(weight.grad)},
                input_grads={},
            )

        if spec.name == "layernorm":
            mod = nn.LayerNorm(spec.config["dim"])
            weight = self.MLXTensor(self._a(ref_params["weight"]), requires_grad=True)
            bias = self.MLXTensor(self._a(ref_params["bias"]), requires_grad=True)
            mod.weight = weight
            mod.bias = bias
            x = self.from_numpy(input_arrays[0], requires_grad=True)
            out = mod(x)
            out.backward(g)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": np.array(weight.grad),
                             "bias": np.array(bias.grad)},
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
