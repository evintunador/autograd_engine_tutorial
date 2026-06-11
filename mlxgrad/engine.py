"""
MLXTensor is the foundation of mlxgrad's autograd engine — the Apple-Metal tier
of this tutorial. It is a thin wrapper around an ``mlx.core`` float32 array that
routes every forward/backward operation through our own custom Metal kernels
(the ``.metal`` sources under ``kernels/``, launched via ``mlx_kernels``) and
never MLX's own autograd or high-level math ops. This is the sibling of
``cudagrad/engine.py``: where cudagrad writes raw CUDA C++, here we write Metal
Shading Language — the closest Apple analog to thread-level CUDA. (Apple has no
first-party tile-level DSL, so there is no mlxgrad analog of tritongrad's Triton
tier.)

Terminology, as in cudagrad: "mx.array" is MLX's array, "MLXTensor" is our
wrapper around it, and "tensor" means either.

Unlike cudagrad/tritongrad (which need a CUDA GPU), this runs on Apple Silicon —
so it imports and runs on the dev machine. The test adapter still gates on
``mx.metal.is_available()`` so non-Apple hosts skip cleanly.

KEY DESIGN NOTE — functional gradient accumulation. MLX arrays are immutable and
lazily evaluated, so we cannot accumulate gradients in place the way cudagrad's
CUDA kernels do (``dx += ...``). Instead each backward kernel takes the running
gradient ``grad_in`` and returns ``grad_in + contribution``; the ``_backward``
closures here rebind ``child.grad = mlx_kernels.<op>_backward(child.grad, ...)``.
The math (including the ``+=``) still happens entirely inside the Metal kernels.
"""
from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import mlx.core as mx

import mlx_kernels as kn

DEVICE = mx.default_device()


class MLXTensor:
    '''
    Stores a tensor and its gradient information. A wrapper around mx.array so we
    get Python operator sugar (__add__, __mul__, ...) while all real math runs in
    our custom Metal kernels.
    '''
    def __init__(self,
                 data: Union[float, int, list, np.ndarray, mx.array],
                 requires_grad: bool = False,
                 device: Optional[object] = None,
                 _children: Tuple['MLXTensor', ...] = ()):

        # Convert input to an mx.array; enforce fp32 throughout for simplicity
        # (matches the suite's reference). MLX uses one unified-memory device on
        # Apple Silicon, so there is no host/device copy to manage.
        if isinstance(data, mx.array):
            self.data = data.astype(mx.float32)
        else:
            self.data = mx.array(data).astype(mx.float32)

        # Tensor metadata.
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        self.device = DEVICE
        self.numel = lambda: self.data.size

        # Gradient state.
        self.requires_grad = requires_grad
        self.grad = mx.zeros_like(self.data) if requires_grad else None

        # Autograd graph.
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"MLXTensor:\n{self.data}"

    # ---- elementwise binary ops -------------------------------------------
    def _binary(self, other, op):
        """elementwise binary op supporting broadcasting of `other` up to `self`.

        We pass `loop_stride = other.numel()` to the kernel, which broadcasts
        `other` via `i % loop_stride` (same contract as cudagrad/tritongrad)."""
        other = other if isinstance(other, MLXTensor) else MLXTensor(other)

        n_elements = self.numel()
        loop_stride = other.numel()
        assert n_elements >= loop_stride, \
            "first input must have >= as many entries as the second"
        assert n_elements % loop_stride == 0, \
            "first input's entry count must be a multiple of the second's"

        # restrict to logically-broadcastable shapes (else the kernel's modulo
        # indexing would compute nonsense) — same guard as cudagrad
        if self.shape != other.shape and other.shape != (1,):
            ptr = 0
            for d in self.shape:
                if ptr == other.ndim:
                    break
                if d == other.shape[ptr]:
                    ptr += 1
            assert ptr == other.ndim, \
                f"broadcasting requires b's dims ({other.shape}) be a subsequence of a's ({self.shape})"

        output = kn.binary_forward(self.data, other.data, loop_stride, op)

        out = MLXTensor(
            output,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                self.grad = kn.binary_backward_dx(self.grad, other.data, out.grad,
                                                  loop_stride, op)
            if other.requires_grad:
                other.grad = kn.binary_backward_dy(other.grad, self.data, other.data,
                                                   out.grad, loop_stride, op)
        out._backward = _backward

        return out

    def __add__(self, other):
        return self._binary(other, op='add')

    def __mul__(self, other):
        return self._binary(other, op='mul')

    def __sub__(self, other):
        return self._binary(other, op='sub')

    def __truediv__(self, other):
        return self._binary(other, op='div')

    def __neg__(self):
        return self._unary(op='neg')

    # ---- matmul -----------------------------------------------------------
    def __matmul__(self, other):
        """A @ B for matrices and (batched) tensors. NOT YET IMPLEMENTED — the
        matmul phase fills in the kernel + may refine this call site."""
        assert self.ndim >= 2 and other.ndim >= 2, \
            f'matmul needs >=2D inputs, got {self.ndim} and {other.ndim}'
        assert self.shape[-1] == other.shape[-2], \
            f'incompatible matmul dims, A: {self.shape}, B: {other.shape}'

        out_data = kn.matmul_forward(self.data, other.data)

        out = MLXTensor(
            out_data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                self.grad = kn.matmul_backward_dA(self.grad, other.data, out.grad)
            if other.requires_grad:
                other.grad = kn.matmul_backward_dB(other.grad, self.data, out.grad)
        out._backward = _backward

        return out

    # ---- elementwise unary ops --------------------------------------------
    def _unary(self, op):
        """elementwise unary op (exp/log/relu/neg)."""
        output = kn.unary_forward(self.data, op)

        out = MLXTensor(output, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad = kn.unary_backward(self.grad, self.data, out.data,
                                              out.grad, op)
        out._backward = _backward

        return out

    def exp(self):
        return self._unary(op='exp')

    def log(self):
        return self._unary(op='log')

    def relu(self):
        return self._unary(op='relu')

    # ---- reductions over the final dimension ------------------------------
    def _reduction(self, op):
        """reduce along the final dim (sum/mean/max/min/var/std). NOT YET
        IMPLEMENTED. Note for the implementer: var/std must subtract the row mean
        and use population (/n) normalization to agree with torch.var(unbiased=
        False) — the exact bug the suite caught in tritongrad (see PROMPT.md)."""
        n_cols = self.shape[-1]
        n_rows = self.numel() // n_cols
        output = kn.reduction_forward(self.data, n_rows, n_cols, op)

        out = MLXTensor(output, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad = kn.reduction_backward(self.grad, self.data, out.grad,
                                                  out.data, n_rows, n_cols, op)
        out._backward = _backward

        return out

    def sum(self):
        return self._reduction(op='sum')

    def mean(self):
        return self._reduction(op='mean')

    def max(self):
        return self._reduction(op='max')

    def min(self):
        return self._reduction(op='min')

    def var(self):
        return self._reduction(op='var')

    def std(self):
        return self._reduction(op='std')

    def softmax(self):
        """numerically-stable softmax along the final dim. NOT YET IMPLEMENTED."""
        n_cols = self.shape[-1]
        n_rows = self.numel() // n_cols
        output = kn.softmax_forward(self.data, n_rows, n_cols)

        out = MLXTensor(output, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad = kn.softmax_backward(self.grad, out.data, out.grad,
                                                n_rows, n_cols)
        out._backward = _backward

        return out

    # ---- shape ops (no custom kernels; grads flow through MLX array ops) ----
    # As in cudagrad, these reshape/move data only — no arithmetic — so their
    # gradients are plain MLX index/shape ops, not Metal kernels.
    def contiguous(self):
        """No-op: MLX copies non-contiguous inputs to row-contiguous at each
        kernel launch (ensure_row_contiguous), so an explicit pass is unneeded."""
        return self

    def transpose(self, dim0=None, dim1=None):
        if dim0 is None and dim1 is None:
            dim0, dim1 = -1, -2
        out = MLXTensor(mx.swapaxes(self.data, dim0, dim1),
                        self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + mx.swapaxes(out.grad, dim0, dim1)
        out._backward = _backward
        return out

    def squeeze(self, dim):
        out = MLXTensor(mx.squeeze(self.data, dim),
                        self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + mx.expand_dims(out.grad, dim)
        out._backward = _backward
        return out

    def unsqueeze(self, dim):
        out = MLXTensor(mx.expand_dims(self.data, dim),
                        self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + mx.squeeze(out.grad, dim)
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = MLXTensor(mx.reshape(self.data, shape),
                        self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + mx.reshape(out.grad, self.shape)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = MLXTensor(self.data[idx], self.requires_grad, self.device, (self,))
        def _backward():
            # backward through slicing is unused in this tutorial; left as a no-op
            pass
        out._backward = _backward
        return out

    # ---- autograd driver --------------------------------------------------
    def zero_grad(self):
        self.grad = mx.zeros_like(self.data) if self.requires_grad else None

    def backward(self, grad=None):
        """Backpropagate from this tensor (typically a scalar loss).

        Like cudagrad, mlxgrad needs NO warmup dance: ``mx.fast.metal_kernel``
        does not autotune, so a single pass accumulates each grad exactly once
        into buffers that start at zero."""
        self.grad = mx.ones_like(self.data) if grad is None else grad
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node._backward()

    def zero_grad_backward(self):
        """Zero every grad in the graph reachable from this node."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node.grad = mx.zeros_like(node.grad) if node.grad is not None else None


class Parameter(MLXTensor):
    """A trainable MLXTensor (weights/biases); requires_grad is always True."""
    def __init__(self, data: Union[float, int, mx.array], device=None):
        super().__init__(data, requires_grad=True, device=device)
