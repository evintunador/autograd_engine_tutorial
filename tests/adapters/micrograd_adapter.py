"""Adapter for micrograd (scalar Value objects in nested Python lists).

The hard parts:
  * Marshalling np.ndarray <-> nested lists of ``Value``.
  * micrograd's ``Value.backward()`` is scalar-only and hard-seeds grad to 1.0.
    For a tensor output (a forest of ``Value``s) seeded with an arbitrary upstream
    gradient, we bypass it: flatten the output values, build one topo order over
    the union of their graphs, zero every node, seed each output scalar with its
    slice of ``grad_output`` (``+=`` so shared nodes accumulate), then run one
    reverse pass. ``Value._backward`` closures only read/accumulate, so this
    yields the correct vector-Jacobian product.
"""
import numpy as np

from tests.core.base_adapter import AdapterABC, GraphHandle, ModuleResult
from tests.core.loader import load_backend


class MicrogradAdapter(AdapterABC):
    name = "micrograd"
    # core ops only; micrograd has no broadcasting, no dim-reductions beyond a
    # full last-dim sum, no shape ops, and only add/mul elementwise.
    OPS = {"add", "sub", "mul", "div", "neg", "matmul", "exp", "log", "relu", "softmax",
           "sum_lastdim", "mean", "var", "std", "max_lastdim", "min_lastdim"}
    MODULES = {"linear", "embedding", "layernorm"}

    @classmethod
    def available(cls):
        return True, ""

    def __init__(self):
        mods = load_backend("micrograd", ["engine", "ops", "modules"])
        self._engine = mods["engine"]
        self._ops = mods["ops"]
        self._modules = mods["modules"]
        self.Value = self._engine.Value

    # ---- marshalling ------------------------------------------------------
    def from_numpy(self, arr, requires_grad=True):
        arr = np.asarray(arr)
        Value = self.Value

        def build(a):
            if a.ndim == 0:
                return Value(float(a))
            return [build(a[i]) for i in range(a.shape[0])]
        return build(arr)

    def to_numpy(self, x):
        Value = self.Value

        def read(v):
            if isinstance(v, Value):
                return v.data
            return [read(s) for s in v]
        return np.array(read(x), dtype=np.float32)

    def grad_of(self, leaf):
        Value = self.Value

        def read(v):
            if isinstance(v, Value):
                return v.grad
            return [read(s) for s in v]
        return np.array(read(leaf), dtype=np.float32)

    def _flatten(self, x):
        if isinstance(x, self.Value):
            return [x]
        out = []
        for s in x:
            out += self._flatten(s)
        return out

    # ---- ops --------------------------------------------------------------
    def forward_op(self, op_name, inputs):
        ops = self._ops
        a = inputs[0]
        b = inputs[1] if len(inputs) > 1 else None
        if op_name == "add":
            out = ops.entry_wise_add(a, b)
        elif op_name == "sub":
            out = ops.entry_wise_sub(a, b)
        elif op_name == "mul":
            out = ops.entry_wise_mult(a, b)
        elif op_name == "div":
            out = ops.entry_wise_div(a, b)
        elif op_name == "matmul":
            out = ops.tensor_matmul(a, b)
        elif op_name == "exp":
            out = ops.vector_wise_apply(ops.exp, a)
        elif op_name == "log":
            out = ops.vector_wise_apply(ops.log, a)
        elif op_name == "neg":
            out = ops.vector_wise_apply(ops.neg, a)
        elif op_name == "relu":
            out = ops.vector_wise_apply(ops.relu, a)
        elif op_name == "softmax":
            out = ops.vector_wise_apply(ops.softmax, a)
        elif op_name == "sum_lastdim":
            out = ops.vector_wise_apply(ops.sum, a)
        elif op_name == "mean":
            out = ops.vector_wise_apply(ops.mean, a)
        elif op_name == "var":
            out = ops.vector_wise_apply(ops.var, a)
        elif op_name == "std":
            out = ops.vector_wise_apply(ops.std, a)
        elif op_name == "max_lastdim":
            out = ops.vector_wise_apply(ops.max, a)
        elif op_name == "min_lastdim":
            out = ops.vector_wise_apply(ops.min, a)
        else:
            raise KeyError(op_name)
        return GraphHandle(out, inputs)

    def backward(self, handle, grad_output):
        self._seeded_backward(handle.output, grad_output)

    def _seeded_backward(self, output, grad_output):
        out_vals = self._flatten(output)
        grad_flat = np.asarray(grad_output, dtype=np.float32).reshape(-1)
        assert len(out_vals) == grad_flat.size, \
            f"output has {len(out_vals)} scalars but grad has {grad_flat.size}"

        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        for v in out_vals:
            build(v)

        for node in topo:
            node.grad = 0.0
        for v, g in zip(out_vals, grad_flat):
            v.grad += float(g)
        for node in reversed(topo):
            node._backward()

    # ---- modules ----------------------------------------------------------
    def run_module(self, spec, ref_params, input_arrays, grad_output):
        ops = self._ops
        modules = self._modules

        if spec.name == "linear":
            in_dim, out_dim = spec.config["in"], spec.config["out"]
            mod = modules.Linear(in_dim, out_dim)
            # torch weight is (out, in); micrograd neuron j holds output j's weights,
            # so neuron[j].w[i] == W[j, i] directly (no transpose).
            W = ref_params["weight"]
            bias = ref_params["bias"]
            for j, neuron in enumerate(mod.neurons):
                for i, wv in enumerate(neuron.w):
                    wv.data = float(W[j, i])
                neuron.b.data = float(bias[j])

            x = self.from_numpy(input_arrays[0].astype(np.float32))
            out = ops.vector_wise_apply(mod, x)
            self._seeded_backward(out, grad_output)

            W_grad = np.array([[neuron.w[i].grad for i in range(in_dim)]
                               for neuron in mod.neurons], dtype=np.float32)
            b_grad = np.array([neuron.b.grad for neuron in mod.neurons], dtype=np.float32)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": W_grad, "bias": b_grad},
                input_grads={0: self.grad_of(x)},
            )

        if spec.name == "layernorm":
            dim = spec.config["dim"]
            mod = modules.LayerNorm(dim)
            # torch LayerNorm weight/bias are both 1-D of length `dim`; load directly.
            W = ref_params["weight"]
            bias = ref_params["bias"]
            for d in range(dim):
                mod.weight[d].data = float(W[d])
                mod.bias[d].data = float(bias[d])

            x = self.from_numpy(input_arrays[0].astype(np.float32))
            out = ops.vector_wise_apply(mod, x)
            self._seeded_backward(out, grad_output)

            # weight/bias are shared across every normalized vector, so their grads
            # accumulate over all positions -- matching torch's reduction.
            W_grad = np.array([mod.weight[d].grad for d in range(dim)], dtype=np.float32)
            b_grad = np.array([mod.bias[d].grad for d in range(dim)], dtype=np.float32)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": W_grad, "bias": b_grad},
                input_grads={0: self.grad_of(x)},
            )

        if spec.name == "embedding":
            num, dim = spec.config["num"], spec.config["dim"]
            mod = modules.Embedding(num, dim)
            W = ref_params["weight"]
            for c in range(num):
                for d in range(dim):
                    mod.weight[c][d].data = float(W[c, d])

            tokens_nested = input_arrays[0].astype(int).tolist()
            out = ops.vector_wise_apply(mod, tokens_nested)
            self._seeded_backward(out, grad_output)

            W_grad = np.array([[mod.weight[c][d].grad for d in range(dim)]
                               for c in range(num)], dtype=np.float32)
            return ModuleResult(
                out=self.to_numpy(out),
                param_grads={"weight": W_grad},
                input_grads={},
            )

        raise NotImplementedError(spec.name)
