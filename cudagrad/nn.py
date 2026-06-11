"""Neural-network modules for cudagrad, mirroring tritongrad/nn.py one
abstraction level lower (CUDA kernels instead of Triton).

Linear is fully functional once the matmul kernel lands (it's pure CudaTensor
algebra). Embedding / LayerNorm / FlashAttention are structural skeletons whose
forward/backward route through `cuda_kernels` wrappers that currently raise
NotImplementedError — the modules phase + flash phase fill in those kernels and
may refine these call sites together with the wrapper signatures.
"""
import math

import torch

from engine import CudaTensor, Parameter
import cuda_kernels as kn

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


class Module:  # mirrors pytorch's nn.Module surface
    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def children(self):
        return []

    def parameters(self):
        out = []
        for child in self.children():
            if child.parameters() is not None:
                out += child.parameters()
        return out if out else None


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True, device=None):
        super().__init__()
        self.weight = Parameter(
            data=torch.randn((in_features, out_features)) * math.sqrt(1 / in_features),
            device=device,
        )
        self.bias = Parameter(
            torch.randn((out_features,)) * math.sqrt(1 / in_features),
            device=device,
        ) if bias else None

    def __call__(self, x: CudaTensor):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"nn.Linear\nWeight:\n{self.weight}"


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            data=torch.randn((num_embeddings, embedding_dim)),
            device=device,
        )

    def __call__(self, tokens):
        B, N = tokens.shape
        # out-of-range indices would read garbage in the kernel; validate up front
        assert int(tokens.data.min().item()) >= 0 and int(tokens.data.max().item()) < self.num_embeddings, \
            f"token indices must be in [0, {self.num_embeddings})"

        output = torch.empty((B, N, self.embedding_dim), dtype=self.weight.dtype,
                             device=self.weight.device, requires_grad=False)
        kn.embedding_forward(tokens.data, self.weight.data, output,
                             N, self.embedding_dim, self.num_embeddings)

        out = CudaTensor(output, requires_grad=True, _children=(tokens, self.weight))

        def _backward():
            kn.embedding_backward(tokens.data, self.weight.grad, out.grad,
                                  N, self.embedding_dim, self.num_embeddings)
        out._backward = _backward

        return out

    def parameters(self):
        return [self.weight]


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.device = DEVICE if device is None else device

        # weight/bias are always materialized (ones/zeros when affine is off) so the
        # single fused kernel always has valid read/write targets; non-affine ones
        # are plain CudaTensors excluded from parameters(). Mirrors tritongrad.
        if elementwise_affine:
            self.weight = Parameter(torch.ones((normalized_shape,)), device=device)
        else:
            self.weight = CudaTensor(torch.ones((normalized_shape,)), requires_grad=False, device=device)
            self.weight.grad = torch.zeros((normalized_shape,), device=self.weight.device)

        if elementwise_affine and bias:
            self.bias = Parameter(torch.zeros((normalized_shape,)), device=device)
        else:
            self.bias = CudaTensor(torch.zeros((normalized_shape,)), requires_grad=False, device=device)
            self.bias.grad = torch.zeros((normalized_shape,), device=self.bias.device)

    def __call__(self, x: CudaTensor):
        D = x.shape[-1]
        assert D == self.normalized_shape
        rows = math.prod(x.shape[:-1])

        output = torch.empty_like(x.data, requires_grad=False)
        # saved for the backward pass (computed in forward)
        mean = torch.empty(rows, dtype=torch.float32, device=self.device, requires_grad=False)
        rstd = torch.empty(rows, dtype=torch.float32, device=self.device, requires_grad=False)

        kn.layernorm_forward(x.data, self.weight.data, self.bias.data, output,
                             mean, rstd, rows, D, self.eps)

        out = CudaTensor(output, requires_grad=True, _children=(x, self.weight, self.bias))

        def _backward():
            kn.layernorm_backward(x.data, self.weight.data, self.bias.data,
                                  x.grad, out.grad, self.weight.grad, self.bias.grad,
                                  mean, rstd, rows, D)
        out._backward = _backward

        return out

    def parameters(self):
        return [p for p in (self.weight, self.bias) if isinstance(p, Parameter)]


class FlashAttention(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, Q: CudaTensor, K: CudaTensor, V: CudaTensor, scale: float = None):
        assert Q.shape == K.shape == V.shape
        B, H, N, D = Q.shape
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        O = torch.empty_like(Q.data)
        LSE = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)
        kn.flash_attention_forward(Q.data, K.data, V.data, O, LSE, scale, B, H, N, D)

        out = CudaTensor(O, requires_grad=True, _children=(Q, K, V))

        def _backward():
            out.grad = out.grad.contiguous()
            kn.flash_attention_backward(Q.data, K.data, V.data, O, out.grad,
                                        Q.grad, K.grad, V.grad, LSE, scale, B, H, N, D)
        out._backward = _backward

        return out

    def parameters(self):
        return []
