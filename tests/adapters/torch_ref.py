"""Reference adapter backed by PyTorch itself.

This is mostly a sanity check on the harness: if torch-vs-torch ever fails, the
marshalling / seeded-backward plumbing is broken, not a backend. It participates
in op tests only (modules would be tautological against the reference).
"""
import numpy as np
import torch

from tests.core.base_adapter import AdapterABC, GraphHandle

_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "matmul": lambda a, b: a @ b,
    "exp": lambda a: a.exp(),
    "log": lambda a: a.log(),
    "relu": lambda a: a.relu(),
    "softmax": lambda a: torch.softmax(a, dim=-1),
    "sum_lastdim": lambda a: a.sum(dim=-1),
    "mean": lambda a: a.mean(dim=-1),
    "var": lambda a: a.var(dim=-1, unbiased=False),
    "std": lambda a: a.std(dim=-1, unbiased=False),
}


class TorchAdapter(AdapterABC):
    name = "torch"
    OPS = set(_OPS.keys())
    MODULES = set()  # the reference; no self-comparison for modules

    @classmethod
    def available(cls):
        return True, ""

    def from_numpy(self, arr, requires_grad):
        return torch.tensor(np.asarray(arr), dtype=torch.float32,
                            requires_grad=requires_grad)

    def to_numpy(self, x):
        return x.detach().numpy()

    def grad_of(self, leaf):
        return leaf.grad.detach().numpy()

    def forward_op(self, op_name, inputs):
        out = _OPS[op_name](*inputs)
        return GraphHandle(out, inputs)

    def backward(self, handle, grad_output):
        handle.output.backward(torch.tensor(np.asarray(grad_output), dtype=torch.float32))
