"""Adapter for minigrad (numpy-backed Tensor, CPU)."""
import numpy as np

from tests.core.base_adapter import AdapterABC, GraphHandle, ModuleResult
from tests.core.loader import load_backend


def _seeded_backward(out, grad_output):
    """Run minigrad's topological backward but seed the output with a supplied
    gradient instead of ones (minigrad's own ``backward()`` hard-seeds ones)."""
    topo, visited = [], set()

    def build(v):
        if id(v) not in visited:
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)
    build(out)

    for node in topo:
        if getattr(node, "requires_grad", False) and node.grad is not None:
            node.grad = np.zeros_like(node.grad)
    out.grad = np.asarray(grad_output, dtype=np.float32).reshape(out.shape)
    for node in reversed(topo):
        node._backward()


class MinigradAdapter(AdapterABC):
    name = "minigrad"
    OPS = {"add", "sub", "mul", "div", "matmul", "exp", "log", "relu",
           "softmax", "sum_lastdim", "mean", "var"}
    MODULES = {"linear", "embedding", "layernorm"}
    # Known minigrad bug surfaced by this suite: Tensor.sum()'s backward
    # (engine.py:220) broadcasts out.grad to self.shape without re-inserting the
    # reduced last axis, so reductions over dim=-1 with keepdim=False crash on
    # backward. mean/var are built on sum() and inherit it. Forward is fine.
    _SUM_BUG = "minigrad Tensor.sum backward missing expand_dims for keepdim=False (engine.py:220)"
    xfail_ops = {"sum_lastdim": _SUM_BUG, "mean": _SUM_BUG, "var": _SUM_BUG}

    @classmethod
    def available(cls):
        return True, ""

    def __init__(self):
        mods = load_backend("minigrad", ["engine", "nn"])
        self._engine = mods["engine"]
        self._nn = mods["nn"]
        self.Tensor = self._engine.Tensor

    # ---- ops --------------------------------------------------------------
    def from_numpy(self, arr, requires_grad):
        return self.Tensor(np.asarray(arr, dtype=np.float32), requires_grad=requires_grad)

    def to_numpy(self, x):
        return np.asarray(x.data, dtype=np.float32)

    def grad_of(self, leaf):
        return np.asarray(leaf.grad, dtype=np.float32)

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
        elif op_name == "softmax":
            out = a.softmax(dim=-1)
        elif op_name == "sum_lastdim":
            out = a.sum(dim=-1)
        elif op_name == "mean":
            out = a.mean(dim=-1)
        elif op_name == "var":
            out = a.var(dim=-1)
        else:
            raise KeyError(op_name)
        return GraphHandle(out, inputs)

    def backward(self, handle, grad_output):
        _seeded_backward(handle.output, grad_output)

    # ---- modules ----------------------------------------------------------
    def run_module(self, spec, ref_params, input_arrays, grad_output):
        Tensor = self.Tensor
        nn = self._nn

        if spec.name == "linear":
            mod = nn.Linear(spec.config["in"], spec.config["out"], bias=True)
            # torch weight is (out, in); minigrad stores (in, out)
            mod.w.data = np.ascontiguousarray(ref_params["weight"].T, dtype=np.float32)
            mod.w.grad = np.zeros_like(mod.w.data)
            mod.b.data = ref_params["bias"].reshape(1, spec.config["out"]).astype(np.float32)
            mod.b.grad = np.zeros_like(mod.b.data)
            w_param, b_param = mod.w, mod.b  # capture before __call__ reassigns mod.b
            x = Tensor(input_arrays[0].astype(np.float32), requires_grad=True)
            out = mod(x)
            _seeded_backward(out, grad_output)
            return ModuleResult(
                out=np.asarray(out.data),
                param_grads={"weight": np.asarray(w_param.grad).T,
                             "bias": np.asarray(b_param.grad).reshape(spec.config["out"])},
                input_grads={0: np.asarray(x.grad)},
            )

        if spec.name == "embedding":
            mod = nn.Embedding(spec.config["num"], spec.config["dim"])
            mod.w.data = ref_params["weight"].astype(np.float32).copy()
            mod.w.grad = np.zeros_like(mod.w.data)
            tokens = input_arrays[0].astype(np.int64)
            out = mod(tokens)
            _seeded_backward(out, grad_output)
            return ModuleResult(
                out=np.asarray(out.data),
                param_grads={"weight": np.asarray(mod.w.grad)},
                input_grads={},
            )

        if spec.name == "layernorm":
            mod = nn.LayerNorm(spec.config["dim"], elementwise_affine=True)
            mod.affine.data = ref_params["weight"].astype(np.float32).copy()
            mod.affine.grad = np.zeros_like(mod.affine.data)
            mod.bias.data = ref_params["bias"].astype(np.float32).copy()
            mod.bias.grad = np.zeros_like(mod.bias.data)
            a_param, b_param = mod.affine, mod.bias
            x = Tensor(input_arrays[0].astype(np.float32), requires_grad=True)
            out = mod(x)
            _seeded_backward(out, grad_output)
            return ModuleResult(
                out=np.asarray(out.data),
                param_grads={"weight": np.asarray(a_param.grad),
                             "bias": np.asarray(b_param.grad)},
                input_grads={0: np.asarray(x.grad)},
            )

        raise NotImplementedError(spec.name)
