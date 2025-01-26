"""
This document and the object TritonTensor are the foundation of our autograd engine that everything else runs off of.
Triton only works on pytorch's torch.tensor object so what we're doing here is building a wrapper function TritonTensor
 around torch.tensor that allows us to only use our own custom kernel operations and never any of pytorch's. 

in this document we'll use "torch.tensor" to refer to pytorch's implementation of tensors, 
 "TritonTensor" to refer to our implementation which is itself a wrapper around pytorch's, 
  and "tensor" to refer to them both at once
"""

from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import torch
import triton
import triton.language as tl
from kernels import hadamard

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

class TritonTensor:
    '''
    Stores a tensor and its gradient information.
    Functions as a wrapper around torch.tensor so that we can take advantage of in-built ops like __add__ 
     and __mul__ rather than writing Triton kernels in the traditional way.
    '''
    def __init__(self, 
                 data: Union[float, int, list, np.ndarray, torch.Tensor], 
                 requires_grad: bool = False,
                 device: Optional[Union[str, torch.device]] = None,
                 _children: Tuple['Tensor', ...] = ()):
        
        # Convert input data to torch.Tensor if it isn't already
        if isinstance(data, torch.Tensor):
            self.data = data
        else:
            self.data = torch.tensor(data, dtype=torch.float32, requires_grad=False)
                # requires_grad=False prevents pytorch from taking up memory by keeping track of its own gradient
        
        # Move tensor to specified device (default to CUDA if available)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = self.data.to(device)
        
        # Store tensor metadata
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        self.device = self.data.device
        self.numel = lambda: self.data.numel()
        
        # Gradient-related attributes
        self.requires_grad = requires_grad
        self.grad = torch.zeros_like(self.data, requires_grad=False) if requires_grad else None
            # requires_grad=False prevents pytorch from taking up memory by keeping track of its own gradient
        
        # Autograd graph information
        self._prev = set(_children)
        self._backward = lambda: None  # function to compute local gradient updates

    def __repr__(self):
        return f"Tensor:\n{self.data}\nGrad:{self.grad}"
    
    def _hadamard(self, other, op):
        """a simple hadamard (entry-wise) operation that supports broadcasting of other up to size self
        """
        # Ensures all tensors are on the same GPU device
        assert self.device == other.device, \
            f'tensors must be on same device but got self.device: {self.device}, other.device: {other.device}'

        # Convert other to Tensor if needed
        other = other if isinstance(other, TritonTensor) else TritonTensor(other)

        # getting the total number of entries of each of our inputs
        n_elements = self.numel()
        loop_stride = other.numel() # the name `loop_stride` will make sense in the kernel

        # forcing other to be the smaller tensor that needs to get broadcasted into self makes our logic simpler
        assert n_elements >= loop_stride, "for addition, the first input must have more than or as many entries as the second"
        assert n_elements % loop_stride == 0, "the number of entries in the first input must be a multiple of the second"

        # restricting the possible set of inputs to those which are logically broadcastable.
        # if we didn't do this then later our kernel would compute nonsensical broadcasting values
        if self.shape != other.shape:
            ptr = 0
            for d in self.shape:
                if ptr == other.ndim: break
                if d == other.shape[ptr]:
                    ptr += 1
            assert ptr == other.ndim, f"for broadcasting to work, all dims in x ({x.shape}) must be a subset of those in y ({y.shape})"
        # TODO is this restriction good enough? am i missing something?
        
        # Preallocating the output
        output = torch.empty_like(self.data)
        
        # Define grid based on tensor dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        # Launch kernel
        hadamard.binary_op_forward[grid](
            self.data, other.data, output, 
            n_elements, loop_stride,
            OP=op
        )
        
        # Wrap output in Tensor with autograd information
        out = TritonTensor(
            output,
            requires_grad = (self.requires_grad or other.requires_grad),
            _children = (self, other)
        )
        
        # define our backward pass
        def _backward():
            # Only run backward kernel if at least one input requires grad
            if not self.requires_grad and not other.requires_grad:
                pass

            # reusing the same grid from earlier
            hadamard.binary_op_backward[grid](
                self.data, other.data,
                self.grad, other.grad, out.grad, 
                n_elements, loop_stride,
                OP=op,
            )
            # doesn't return anything because it writes to the self.grad and self.other tensors in-place

        out._backward = _backward
        
        return out

    def __add__(self, other):
        return self._hadamard(other, op='add')

    def __mul__(self, other):
        return self._hadamard(other, op='mul')

    def __sub__(self, other):
        return self._hadamard(other, op='sub')

    def __truediv__(self, other):
        return self._hadamard(other, op='div')

    def __neg__(self):
        """Placeholder for negation"""
        # TODO: Implement Triton kernel for negation
        raise NotImplementedError("Negation kernel not yet implemented")

    def __matmul__(self, other):
        """Placeholder for matrix multiplication"""
        # TODO: Implement Triton kernel for matrix multiplication
        raise NotImplementedError("Matrix multiplication kernel not yet implemented")

    def sum(self, dim=None, keepdim=False):
        """Placeholder for sum reduction"""
        # TODO: Implement Triton kernel for sum reduction
        raise NotImplementedError("Sum reduction kernel not yet implemented")

    def mean(self, dim=None, keepdim=False):
        """Placeholder for mean reduction"""
        # TODO: Implement Triton kernel for mean reduction
        raise NotImplementedError("Mean reduction kernel not yet implemented")

    def exp(self):
        """Placeholder for exponential operation"""
        # TODO: Implement Triton kernel for exponential
        raise NotImplementedError("Exponential kernel not yet implemented")

    def log(self):
        """Placeholder for logarithm operation"""
        # TODO: Implement Triton kernel for logarithm
        raise NotImplementedError("Logarithm kernel not yet implemented")

    def relu(self):
        """Placeholder for ReLU activation"""
        # TODO: Implement Triton kernel for ReLU
        raise NotImplementedError("ReLU kernel not yet implemented")

    def max(self, dim=None):
        """Placeholder for max reduction"""
        # TODO: Implement Triton kernel for max reduction
        raise NotImplementedError("Max reduction kernel not yet implemented")

    def min(self, dim=None):
        """Placeholder for min reduction"""
        # TODO: Implement Triton kernel for min reduction
        raise NotImplementedError("Min reduction kernel not yet implemented")

    def softmax(self, dim=-1):
        """Placeholder for softmax operation"""
        # TODO: Implement Triton kernel for softmax
        raise NotImplementedError("Softmax kernel not yet implemented")

    def __pow__(self, power):
        """Placeholder for power operation"""
        # TODO: Implement Triton kernel for power
        raise NotImplementedError("Power kernel not yet implemented")

    def var(self, dim=-1, keepdim=False):
        """Placeholder for variance reduction"""
        # TODO: Implement Triton kernel for variance
        raise NotImplementedError("Variance kernel not yet implemented")

    def sd(self, dim=-1, keepdim=False):
        """Placeholder for standard deviation reduction"""
        # TODO: Implement Triton kernel for standard deviation
        raise NotImplementedError("Standard deviation kernel not yet implemented")

    def transpose(self, axes=None):
        """Placeholder for transpose operation"""
        # TODO: Implement Triton kernel for transpose
        raise NotImplementedError("Transpose kernel not yet implemented")

    def squeeze(self, dim):
        """Placeholder for squeeze operation"""
        # TODO: Implement Triton kernel for squeeze
        raise NotImplementedError("Squeeze kernel not yet implemented")

    def unsqueeze(self, dim):
        """Placeholder for unsqueeze operation"""
        # TODO: Implement Triton kernel for unsqueeze
        raise NotImplementedError("Unsqueeze kernel not yet implemented")

    def broadcast_to(self, shape):
        """Placeholder for broadcast operation"""
        # TODO: Implement Triton kernel for broadcast
        raise NotImplementedError("Broadcast kernel not yet implemented")

    def reshape(self, shape):
        """Placeholder for reshape operation"""
        # TODO: Implement Triton kernel for reshape
        raise NotImplementedError("Reshape kernel not yet implemented")

    def __getitem__(self, idx):
        """Placeholder for indexing operation"""
        # TODO: Implement Triton kernel for indexing
        raise NotImplementedError("Indexing kernel not yet implemented")

    def masked_fill(self, mask, fill_value):
        """Placeholder for masked fill operation"""
        # TODO: Implement Triton kernel for masked fill
        raise NotImplementedError("Masked fill kernel not yet implemented")

    def zero_grad(self):
        self.grad = torch.zeros_like(self.data) if self.requires_grad else None

    def backward(self, grad: None):
        """
        Run backpropagation starting from this tensor. 
        Typically called on a scalar loss tensor.
        """
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



def test_operation(op_name: str,
                  triton_fn,
                  torch_fn,
                  input_shapes: list,
                  dtype=torch.float32,
                  device=DEVICE,
                  rtol=1e-3,
                  atol=1e-3):
    """
    Test TritonTensor operations against PyTorch for correctness.
    
    Args:
        op_name: Name of operation being tested
        triton_fn: Function that takes TritonTensor inputs and returns TritonTensor output
        torch_fn: Function that takes torch.Tensor inputs and returns torch.Tensor output
        input_shapes: List of shapes for input tensors
        dtype: Data type for tensors
        device: Device to run on
        rtol: Relative tolerance for comparing outputs
        atol: Absolute tolerance for comparing outputs
    """
    print(f"\nTesting {op_name}...")
    
    # Generate random inputs
    torch_inputs = [torch.randn(shape, dtype=dtype, device=device, requires_grad=True) 
                   for shape in input_shapes]
    triton_inputs = [TritonTensor(x, requires_grad=True) for x in torch_inputs]
    
    # Forward pass
    torch_out = torch_fn(*torch_inputs)
    triton_out = triton_fn(*triton_inputs)
    
    # Check forward pass
    torch.testing.assert_close(triton_out.data, torch_out, rtol=rtol, atol=atol)
    print(f"✓ Forward pass matches")
    
    # before computing the backward pass, we need to let the autotuner run.
    # this needs to be done bc otherwise the gradient accumulation of each run would compound
    #  to incorrect values
    zero_grad = torch.zeros_like(torch_out)
    triton_out.backward(zero_grad)
    # and in order to avoid any potential divide by zero Nan's, we also set all gradients to 0
    triton_out.zero_grad_backward()

    # Backward pass
    grad_output = torch.randn_like(torch_out)
    torch_out.backward(grad_output)
    triton_out.backward(grad_output)
    
    # Check gradients
    for i, (torch_input, triton_input) in enumerate(zip(torch_inputs, triton_inputs)):
        torch.testing.assert_close(triton_input.grad, torch_input.grad, rtol=rtol, atol=atol)
    print(f"✓ Backward pass matches")


if __name__ == "__main__":
    B, N, H, D = 32, 1024, 8, 64

    ### ADDITION
    def triton_add(x, y): return x + y
    def torch_add(x, y): return x + y
    test_operation(
        f"addition: ({B}, {N}, {D}) + ({B}, {N}, {D})",
        triton_add,
        torch_add,
        [(B, N, D), (B, N, D)]
    )
    test_operation(
        f"addition with broadcasting: ({B}, {N}, {D}) + ({D})",
        triton_add,
        torch_add,
        [(B, N, D), (D)]
    )

    ### MULTIPLICATION
    def triton_mul(x, y): return x * y
    def torch_mul(x, y): return x * y
    test_operation(
        f"multiplication: ({B}, {N}, {D}) * ({B}, {N}, {D})",
        triton_mul,
        torch_mul,
        [(B, N, D), (B, N, D)]
    )
    test_operation(
        f"multiplication with broadcasting: ({B}, {N}, {D}) * ({D})",
        triton_mul,
        torch_mul,
        [(B, N, D), (D)]
    )

    ### SUBTRACTION
    def triton_sub(x, y): return x - y
    def torch_sub(x, y): return x - y
    test_operation(
        f"subtraction: ({B}, {N}, {D}) + ({B}, {N}, {D})",
        triton_sub,
        torch_sub,
        [(B, N, D), (B, N, D)]
    )
    test_operation(
        f"subtraction with broadcasting: ({B}, {N}, {D}) + ({D})",
        triton_sub,
        torch_sub,
        [(B, N, D), (D)]
    )

    ### DIVISION
    def triton_div(x, y): return x / y
    def torch_div(x, y): return x / y
    test_operation(
        f"division: ({B}, {N}, {D}) + ({B}, {N}, {D})",
        triton_div,
        torch_div,
        [(B, N, D), (B, N, D)]
    )
    test_operation(
        f"division with broadcasting: ({B}, {N}, {D}) + ({D})",
        triton_div,
        torch_div,
        [(B, N, D), (D)]
    )