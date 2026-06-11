"""Neural-network modules for mlxgrad, mirroring cudagrad/nn.py with MLX arrays
and Metal kernels instead of torch.cuda + CUDA.

Linear is fully functional once the matmul kernel lands (it's pure MLXTensor
algebra). Embedding / LayerNorm / FlashAttention are structural skeletons whose
forward/backward route through `mlx_kernels` wrappers that currently raise
NotImplementedError — the modules phase + flash phase fill in those kernels and
may refine these call sites together with the wrapper signatures.
"""
import math

import mlx.core as mx

from engine import MLXTensor, Parameter
import mlx_kernels as kn

DEVICE = mx.default_device()


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
            data=mx.random.normal((in_features, out_features)) * math.sqrt(1 / in_features),
            device=device,
        )
        self.bias = Parameter(
            mx.random.normal((out_features,)) * math.sqrt(1 / in_features),
            device=device,
        ) if bias else None

    def __call__(self, x: MLXTensor):
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
            data=mx.random.normal((num_embeddings, embedding_dim)),
            device=device,
        )

    def __call__(self, tokens):
        B, N = tokens.shape
        # out-of-range indices would read garbage in the kernel; validate up front
        assert int(mx.min(tokens.data).item()) >= 0 and \
            int(mx.max(tokens.data).item()) < self.num_embeddings, \
            f"token indices must be in [0, {self.num_embeddings})"

        output = kn.embedding_forward(tokens.data, self.weight.data,
                                      N, self.embedding_dim, self.num_embeddings)

        out = MLXTensor(output, requires_grad=True, _children=(tokens, self.weight))

        def _backward():
            self.weight.grad = kn.embedding_backward(
                self.weight.grad, tokens.data, out.grad,
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

        # weight/bias are always materialized (ones/zeros when affine is off) so the
        # single fused kernel always has valid read/write targets; non-affine ones
        # are plain MLXTensors excluded from parameters(). Mirrors cudagrad.
        if elementwise_affine:
            self.weight = Parameter(mx.ones((normalized_shape,)), device=device)
        else:
            self.weight = MLXTensor(mx.ones((normalized_shape,)), requires_grad=False, device=device)
            self.weight.grad = mx.zeros((normalized_shape,))

        if elementwise_affine and bias:
            self.bias = Parameter(mx.zeros((normalized_shape,)), device=device)
        else:
            self.bias = MLXTensor(mx.zeros((normalized_shape,)), requires_grad=False, device=device)
            self.bias.grad = mx.zeros((normalized_shape,))

    def __call__(self, x: MLXTensor):
        D = x.shape[-1]
        assert D == self.normalized_shape
        rows = math.prod(x.shape[:-1])

        # forward returns the normalized output plus per-row mean/rstd saved for bwd
        output, mean, rstd = kn.layernorm_forward(
            x.data, self.weight.data, self.bias.data, rows, D, self.eps)

        out = MLXTensor(output, requires_grad=True, _children=(x, self.weight, self.bias))

        def _backward():
            x.grad, self.weight.grad, self.bias.grad = kn.layernorm_backward(
                x.data, self.weight.data, self.bias.data,
                x.grad, self.weight.grad, self.bias.grad,
                out.grad, mean, rstd, rows, D)
        out._backward = _backward

        return out

    def parameters(self):
        return [p for p in (self.weight, self.bias) if isinstance(p, Parameter)]


class FlashAttention(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, Q: MLXTensor, K: MLXTensor, V: MLXTensor, scale: float = None):
        assert Q.shape == K.shape == V.shape
        B, H, N, D = Q.shape
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        # forward returns O plus the per-row logsumexp LSE reused in backward
        O, LSE = kn.flash_attention_forward(Q.data, K.data, V.data, scale, B, H, N, D)

        out = MLXTensor(O, requires_grad=True, _children=(Q, K, V))

        def _backward():
            Q.grad, K.grad, V.grad = kn.flash_attention_backward(
                Q.data, K.data, V.data, O, out.grad,
                Q.grad, K.grad, V.grad, LSE, scale, B, H, N, D)
        out._backward = _backward

        return out

    def parameters(self):
        return []
