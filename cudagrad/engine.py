"""
CudaTensor is the foundation of cudagrad's autograd engine — the raw-CUDA tier of
this tutorial. It is a thin wrapper around a ``torch.cuda`` float32 tensor that
routes every forward/backward operation through our own custom CUDA kernels
(compiled from the ``.cu`` sources under ``kernels/`` via ``cuda_kernels``) and
never PyTorch's own math ops. This mirrors ``tritongrad/engine.py`` one
abstraction level lower: where tritongrad writes tile-level Triton kernels, here
we write plain CUDA C++.

Terminology, as in tritongrad: "torch.tensor" is PyTorch's tensor, "CudaTensor"
is our wrapper around it, and "tensor" means either.

NOTE: this module touches ``torch.cuda.current_device()`` at import time, so it
only imports on a CUDA box. The test adapter checks availability WITHOUT
importing this, so non-GPU hosts skip cleanly.
"""
from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import torch

import cuda_kernels as kn

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


class CudaTensor:
    '''
    Stores a tensor and its gradient information. A wrapper around torch.tensor so
    we get Python operator sugar (__add__, __mul__, ...) while all real math runs
    in our custom CUDA kernels.
    '''
    def __init__(self,
                 data: Union[float, int, list, np.ndarray, torch.Tensor],
                 requires_grad: bool = False,
                 device: Optional[Union[str, torch.device]] = None,
                 _children: Tuple['CudaTensor', ...] = ()):

        # Convert input data to torch.Tensor if it isn't already.
        if isinstance(data, torch.Tensor):
            # we enforce fp32 throughout for simplicity (matches the suite's reference)
            self.data = data.to(torch.float32)
        else:
            # requires_grad=False so torch never tracks its own grad / wastes memory
            self.data = torch.tensor(data, dtype=torch.float32, requires_grad=False)

        # Move to device (default CUDA if available).
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = self.data.to(device)

        # Tensor metadata.
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        self.device = self.data.device
        self.numel = lambda: self.data.numel()

        # Gradient state.
        self.requires_grad = requires_grad
        self.grad = torch.zeros_like(self.data, requires_grad=False) if requires_grad else None

        # Autograd graph.
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"CudaTensor:\n{self.data}"

    # ---- elementwise binary ops -------------------------------------------
    def _binary(self, other, op):
        """elementwise binary op supporting broadcasting of `other` up to `self`.

        We pass `loop_stride = other.numel()` to the kernel, which broadcasts
        `other` via `i % loop_stride` (same contract as tritongrad)."""
        other = other if isinstance(other, CudaTensor) else CudaTensor(other)

        assert self.device == other.device, \
            f'tensors must share a device, got {self.device} and {other.device}'
        assert self.data.is_contiguous() and other.data.is_contiguous()

        n_elements = self.numel()
        loop_stride = other.numel()
        assert n_elements >= loop_stride, \
            "first input must have >= as many entries as the second"
        assert n_elements % loop_stride == 0, \
            "first input's entry count must be a multiple of the second's"

        # restrict to logically-broadcastable shapes (else the kernel's modulo
        # indexing would compute nonsense) — same guard as tritongrad
        if self.shape != other.shape and other.shape != (1,):
            ptr = 0
            for d in self.shape:
                if ptr == other.ndim:
                    break
                if d == other.shape[ptr]:
                    ptr += 1
            assert ptr == other.ndim, \
                f"broadcasting requires b's dims ({other.shape}) be a subsequence of a's ({self.shape})"

        output = torch.empty_like(self.data)
        kn.binary_forward(self.data, other.data, output, loop_stride, op)

        out = CudaTensor(
            output,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                kn.binary_backward_dx(other.data, self.grad, out.grad, loop_stride, op)
            if other.requires_grad:
                kn.binary_backward_dy(self.data, other.data, other.grad, out.grad, loop_stride, op)
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
        """A @ B for matrices and (batched) tensors, supporting the shapes we need
        for linear layers and attention. NOT YET IMPLEMENTED — the matmul phase
        fills in the kernel + may refine this call site."""
        assert self.ndim >= 2 and other.ndim >= 2, \
            f'matmul needs >=2D inputs, got {self.ndim} and {other.ndim}'
        assert self.shape[-1] == other.shape[-2], \
            f'incompatible matmul dims, A: {self.shape}, B: {other.shape}'
        assert self.data.is_contiguous()

        (m, k), n = self.shape[-2:], other.shape[-1]
        out_data = torch.empty(self.shape[:-2] + (m, n), device=self.device, dtype=torch.float32)
        kn.matmul_forward(self.data, other.data, out_data)

        out = CudaTensor(
            out_data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                kn.matmul_backward_dA(other.data, self.grad, out.grad)
            if other.requires_grad:
                kn.matmul_backward_dB(self.data, other.grad, out.grad)
        out._backward = _backward

        return out

    # ---- elementwise unary ops --------------------------------------------
    def _unary(self, op):
        """elementwise unary op (exp/log/relu/neg). NOT YET IMPLEMENTED."""
        assert self.data.is_contiguous()
        output = torch.empty_like(self.data)
        kn.unary_forward(self.data, output, op)

        out = CudaTensor(output, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                kn.unary_backward(self.data, self.grad, out.data, out.grad, op)
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
        and use population (/n) normalization to agree with the forward, backward,
        and torch.var(unbiased=False) — the exact bug the suite caught in
        tritongrad (see PROMPT.md)."""
        assert self.data.is_contiguous()
        output = torch.empty(self.data.shape[:-1], dtype=self.dtype,
                             device=self.device, requires_grad=False)
        n_cols = self.shape[-1]
        n_rows = self.data.numel() // n_cols
        kn.reduction_forward(self.data, output, n_rows, n_cols, op)

        out = CudaTensor(output, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                kn.reduction_backward(self.data, self.grad, out.grad, out.data,
                                      n_rows, n_cols, op)
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
        assert self.data.is_contiguous()
        output = torch.empty_like(self.data)
        n_cols = self.shape[-1]
        n_rows = self.data.numel() // n_cols
        kn.softmax_forward(self.data, output, n_rows, n_cols)

        out = CudaTensor(output, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                kn.softmax_backward(out.data, self.grad, out.grad, n_rows, n_cols)
        out._backward = _backward

        return out

    # ---- shape ops (no custom kernels; gradients flow through torch ops) ----
    def contiguous(self):
        """contiguous copy (no-op passthrough if already contiguous)."""
        if self.data.is_contiguous():
            return self
        out = CudaTensor(self.data.contiguous(), self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
        out._backward = _backward
        return out

    def transpose(self, dim0=None, dim1=None):
        if dim0 is None and dim1 is None:
            dim0, dim1 = -1, -2
        out = CudaTensor(self.data.transpose(dim0, dim1),
                         self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(dim0, dim1)
        out._backward = _backward
        return out

    def squeeze(self, dim):
        out = CudaTensor(torch.squeeze(self.data, dim),
                         self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += torch.unsqueeze(out.grad, dim)
        out._backward = _backward
        return out

    def unsqueeze(self, dim):
        out = CudaTensor(torch.unsqueeze(self.data, dim),
                         self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += torch.squeeze(out.grad, dim)
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = CudaTensor(torch.reshape(self.data, shape),
                         self.requires_grad, self.device, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += torch.reshape(out.grad, self.shape)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = CudaTensor(self.data[idx], self.requires_grad, self.device, (self,))
        def _backward():
            # backward through slicing is unused in this tutorial; left as a no-op
            pass
        out._backward = _backward
        return out

    # ---- autograd driver --------------------------------------------------
    def zero_grad(self):
        self.grad = torch.zeros_like(self.data) if self.requires_grad else None

    def backward(self, grad=None):
        """Backpropagate from this tensor (typically a scalar loss).

        Unlike tritongrad, cudagrad needs NO warmup dance: our CUDA kernels don't
        autotune, so a single pass accumulates each grad exactly once into buffers
        that start at zero."""
        self.grad = torch.ones_like(self.grad) if grad is None else grad
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
        self.grad = torch.zeros_like(self.grad) if self.grad is not None else None
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
            node.grad = torch.zeros_like(node.grad) if node.grad is not None else None


class Parameter(CudaTensor):
    """A trainable CudaTensor (weights/biases); requires_grad is always True."""
    def __init__(self, data: Union[float, int, torch.Tensor], device=None):
        super().__init__(data, requires_grad=True, device=device)
