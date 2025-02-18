import torch
import triton
import triton.language as tl
import math

from engine import TritonTensor, Parameter
from kernels import modules, flash_attention

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # each SM has a fixed amount of SRAM that it can access
    # if one SM isn't using all its available SRAM then another can be spun up to use the remainder

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
            requires_grad=False
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

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device = None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.device = torch.device(f'cuda:{torch.cuda.current_device()}') if device is None else device

        self.weight = Parameter(
            data = torch.ones((normalized_shape,), device=device),
            device = device
        ) if elementwise_affine else None
        self.bias = Parameter(
            torch.zeros((normalized_shape,), device=device),
            device = device
        ) if elementwise_affine and bias else None
        """
        TODO rn the thing prolly breaks if you do elementwise_affine=False and/or bias=False.
        to fix this you'd set it so that instead of initializing self.weight and self.bias to None,
         they'd be TritonTensors with requires_grad = False but still the same 1's and 0's values.
        that way we can still use the same fused kernel but not have it mess anything up in the graph
         which yes is unnecessarily slow but i'm fs too lazy to write a whole separate kernel without 
         the fused weights and biases when i know for a fact i'm never gonna use it.
        so feel free to do that if you want
        """

    def __call__(self, x: TritonTensor):
        D = x.shape[-1]
        assert D == self.normalized_shape
        preceeding_dims = math.prod(x.shape[:-1])

        # get tensor dimensions to ensure our parallelization scheme will work
        BLOCK_SIZE_COLS = triton.next_power_of_2(D)
        # 4 for the 4 bytes in fp32
        assert BLOCK_SIZE_COLS * 4 < TOTAL_SRAM_PER_SM, \
            f"vectors (each size {BLOCK_SIZE_COLS * 4}) too large to fit into SRAM size {TOTAL_SRAM_PER_SM}"

        # pre-allocate output
        output = torch.empty_like(x.data, requires_grad=False)
        # and pre-allocate mean & reciprocal standard deviation for use in the backward pass later
        mean = torch.empty(math.prod(x.shape[:-1]), dtype=torch.float32, device=self.device, requires_grad=False)
        rstd = torch.empty(math.prod(x.shape[:-1]), dtype=torch.float32, device=self.device, requires_grad=False)

        grid = lambda meta: (triton.cdiv(preceeding_dims, meta['BLOCK_SIZE_ROWS']),)
        modules.layernorm_forward[grid](
            x.data, self.weight.data, self.bias.data, output,
            x.data.stride(-2), x.data.stride(-1),
            self.weight.data.stride(0), self.bias.data.stride(0),
            output.stride(-2), output.stride(-1),
            preceeding_dims, D,
            self.eps,
            mean, rstd,
            BLOCK_SIZE_COLS
        )

        # wrap output in a triton tensor to add it to our graph
        out = TritonTensor(
            output, 
            requires_grad=True,
            _children = (x, self.weight, self.bias)
        )

        def _backward():
            # this module assumes everything needs a gradient, we're not gonna bother giving the option
            modules.layernorm_backward[grid](
                x.data, self.weight.data, self.bias.data,
                x.grad, out.grad,
                self.weight.grad, self.bias.grad,
                mean, rstd,
                x.data.stride(-2), x.data.stride(-1),
                self.weight.data.stride(-1),
                self.bias.data.stride(-1),
                x.grad.stride(-2), x.grad.stride(-1),
                out.grad.stride(-2), out.grad.stride(-1),
                self.weight.grad.stride(-1),
                self.bias.grad.stride(-1),
                mean.stride(-1),
                rstd.stride(-1),
                preceeding_dims, D,
                BLOCK_SIZE_COLS
            )
        out._backward = _backward

        return out

    def parameters(self):
        return [self.weight] \
                + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"nn.LayerNorm\nWeight:\n{self.weight}" + f"\nBias:\n{self.bias}" if self.bias is not None else ""


class FlashAttention(Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        Q: TritonTensor,
        K: TritonTensor,
        V: TritonTensor,
        scale: float = None,
    ):
        assert Q.shape == K.shape == V.shape
        assert Q.shape[-1] in (128, 256), \
            f'flash attention only supports head dimension of 128 or 256 but got {q.shape[-1]}'
            # the kernel actually isn't this limited but too much larger and it would break
        B, H, N, D = Q.shape

        # pre-allocate output tensor
        O = torch.empty_like(Q.data) # output tensor will be pre head concatenation and mixing
        # and pre-allocate the tensor where we hold the logsumexp
        LSE = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)
        
        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]), # primary parallelizatoin is across sequence length
            B * H, # further parallelize across the dimensions that don't matter
        )
        flash_attention.attn_fwd[grid](
            Q.data, K.data, V.data, O, LSE, 
            scale,
            Q.data.stride(0), Q.data.stride(1), Q.data.stride(2), Q.data.stride(3),
            K.data.stride(0), K.data.stride(1), K.data.stride(2), K.data.stride(3),
            V.data.stride(0), V.data.stride(1), V.data.stride(2), V.data.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, N, D,
        )

        # wrap output in a triton tensor to add it to our graph
        out = TritonTensor(
            O, 
            requires_grad = True,
            _children = (Q, K, V)
        )

        def _backward():
            # this module assumes everything needs a gradient, we're not gonna bother giving the option

            # for some reason out.grad isn't contiguous by default; prolly a skill issue tbh
            out.grad = out.grad.contiguous()
            assert Q.data.stride() == K.data.stride() == V.data.stride() == out.data.stride() == out.grad.stride()
            
            Delta = torch.empty_like(LSE)

            # the ordering of your grid matters because it determines which programs end up sharing the same SRAM
            pre_grid = lambda meta: (N // meta["PRE_BLOCK_SIZE_ROW"], B * H)
                # in this case, we want the parallelizations along the N dimension to be near each other so they can
                #  share data, while parallelization across batches & heads don't necessitate any sharing
            flash_attention.attn_backward_preprocess[pre_grid](
                out.data, out.grad, Delta,
                N, D,
            )

            grid = lambda meta: (N // meta["BLOCK_SIZE_MACRO"], B * H) 
            flash_attention.attn_backward[grid](
                Q.data, K.data, V.data,
                out.grad, Q.grad, K.grad, V.grad,
                LSE, Delta,
                scale,
                Q.data.stride(0), Q.data.stride(1), Q.data.stride(2), Q.data.stride(3), # all tensors should share same stride
                H, N, D,
            )
        out._backward = _backward

        return out

    def parameters(self):
        return []

    def __repr__(self):
        return f"nn.FlashAttention Module"
