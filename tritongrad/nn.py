import torch
import triton
import triton.language as tl
import math

from engine import TritonTensor, Parameter
from kernels import modules

class Module: # just to make our syntax the same as pytorch's
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
        # returns a list or iterator of immediate children modules
        # this will be useful for recursively setting training mode
        return []
    
    def parameters(self):
        '''
        default parameter-yielding method
        modules which actually have parameters should overwrite this method
        '''
        out = []
        for child in self.children():
            if child.parameters() is not None:
                out += child.parameters()
        return out if out else None

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias = True, device = None):
        super().__init__()
        self.weight = Parameter(
            data = torch.randn((in_features, out_features)) * math.sqrt(1/in_features),
            device = device
        )
        if bias: 
            self.bias = Parameter(
                torch.randn((out_features,)) * math.sqrt(1/in_features),
                device = device
            )
        else:
            self.bias = None

    def __call__(self, x: TritonTensor):
        # First compute matmul
        out = x @ self.weight
        # Then add bias if it exists
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"nn.Linear\nWeight:\n{self.weight}" + f"\nBias:\n{self.bias}" if self.bias is not None else ""

class Embedding(Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        #padding_idx: int = None,
        device = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            data = torch.randn((num_embeddings, embedding_dim)),
            device = device
        )

    def __call__(self, tokens):
        B, N = tokens.shape
        # TODO assert vals in tokens are all below num_embeddings
        
        # pre-allocate output
        output = torch.empty(
            (B, N, self.embedding_dim), 
            dtype=self.weight.dtype, 
            device=self.weight.device, 
            requires_grad=True
        )

        grid = lambda meta: (
            triton.cdiv(B*N, meta['BLOCK_SIZE_ROWS']), 
            triton.cdiv(self.embedding_dim, meta['BLOCK_SIZE_COLS'])
            )
        modules.embedding_forward[grid](
            tokens.data,
            self.weight.data,
            output,
            tokens.data.stride(0), tokens.data.stride(1),
            self.weight.data.stride(0), self.weight.data.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            N, self.embedding_dim, self.num_embeddings,
            tokens.numel(), self.weight.data.numel(), output.numel(),
        )

        # Wrap output in TritonTensor with autograd information
        out = TritonTensor(
            output, 
            requires_grad = True, # since self.weight always requires it 
            _children = (tokens, self.weight)
        )

        def _backward():
            modules.embedding_backward[grid](
                tokens.data,
                self.weight.grad,
                out.grad,
                tokens.data.stride(0), tokens.data.stride(1),
                self.weight.grad.stride(0), self.weight.grad.stride(1),
                out.grad.stride(0), out.grad.stride(1), out.grad.stride(2),
                N, self.embedding_dim, self.num_embeddings,
                tokens.numel(), self.weight.grad.numel(), out.numel(),
            )
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Emedding:\n({self.weight.data})"

    def parameters(self):
        return [self.weight]
