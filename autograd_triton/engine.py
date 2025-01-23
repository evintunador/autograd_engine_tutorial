from typing import Union, Tuple, Optional

import torch
import numpy as np

import triton
import triton.language as tl
from kernels import hadamard

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

class Tensor:
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
            self.data = torch.tensor(data, dtype=torch.float32)
        
        # Move tensor to specified device (default to CUDA if available)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = self.data.to(device)
        
        # Store tensor metadata
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        self.device = self.data.device
        
        # Gradient-related attributes
        self.requires_grad = requires_grad
        self.grad = torch.zeros_like(self.data) if requires_grad else None
        
        # Autograd graph information
        self._prev = set(_children)
        self._backward = lambda: None  # function to compute local gradient updates

    def __repr__(self):
        return f"Tensor:\n{self.data}\nGrad:{self.grad}"
    
    def __add__(self, other):
        # Convert other to Tensor if needed
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Preallocating the output
        output = torch.empty_like(self.data)
        
        # Ensures all tensors are on the same GPU device
        assert self.device == other.device, \
            f'tensors must be on same device but got self.device: {self.device}, other.device: {other.device}'
        
        # Handle broadcasting
        if self.shape != other.shape:
            # Broadcast tensors to compatible shapes
            target_shape = torch.broadcast_shapes(self.shape, other.shape)
            self_data = self.data.expand(target_shape)
            other_data = other.data.expand(target_shape)
            output = torch.empty(target_shape, device=self.device)
        else:
            self_data = self.data
            other_data = other.data
        
        # Getting total number of elements
        n_elements = output.numel()
        
        # Define grid based on tensor dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        
        # Launch kernel
        hadamard.add_kernel[grid](
            self_data, other_data, output, n_elements, 
            BLOCK_SIZE=1024
        )
        
        # Wrap output in Tensor with autograd information
        out = Tensor(
            output,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other)
        )
        
        def _backward():
            # Only run backward kernel if at least one input requires grad
            if self.requires_grad or other.requires_grad:
                # Get flattened gradients after broadcasting
                if self.shape != out.shape:
                    # Broadcast gradients to match output shape
                    grad_self = torch.zeros_like(self_data)
                    axes = tuple(i for i, (self_dim, out_dim) in enumerate(zip(self.shape, out.shape)) 
                                if self_dim != out_dim)
                    # Launch backward kernel
                    grid = lambda meta: (triton.cdiv(grad_self.numel(), meta['BLOCK_SIZE']), )
                    hadamard.add_backward_kernel[grid](
                        out.grad, grad_self, grad_other,
                        grad_self.numel(), BLOCK_SIZE=1024
                    )
                    # Sum along broadcast axes and reshape
                    if self.requires_grad:
                        self.grad += grad_self.sum(dim=axes).reshape(self.shape)
                    if other.requires_grad:
                        other.grad += grad_other.sum(dim=axes).reshape(other.shape)
                else:
                    # No broadcasting needed, run kernel directly
                    grid = lambda meta: (triton.cdiv(out.grad.numel(), meta['BLOCK_SIZE']), )
                    hadamard.add_backward_kernel[grid](
                        out.grad,
                        self.grad if self.requires_grad else torch.zeros_like(self.data),
                        other.grad if other.requires_grad else torch.zeros_like(other.data),
                        out.grad.numel(),
                        BLOCK_SIZE=1024
                    )

            out._backward = _backward
        
        return out

    def zero_grad(self):
        self.grad = torch.zeros_like(self.data) if self.requires_grad else None

    def backward(self, grad: None):
        """
        Run backpropagation starting from this tensor. 
        Typically called on a scalar loss Tensor.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = torch.ones_like(self.grad) if grad is None else grad
        for node in reversed(topo):
            node._backward()

def test_operation(op_name: str,
                  triton_fn,
                  torch_fn,
                  input_shapes: list,
                  dtype=torch.float32,
                  device=DEVICE,
                  rtol=1e-3,
                  atol=1e-3):
    """
    Test Triton operation against PyTorch for correctness.
    
    Args:
        op_name: Name of operation being tested
        triton_fn: Function that takes Tensor inputs and returns Tensor output
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
    triton_inputs = [Tensor(x, requires_grad=True) for x in torch_inputs]
    
    # Forward pass
    torch_out = torch_fn(*torch_inputs)
    triton_out = triton_fn(*triton_inputs)
    
    # Check forward pass
    torch.testing.assert_close(triton_out.data, torch_out, rtol=rtol, atol=atol)
    print(f"✓ Forward pass matches")
    
    # Backward pass
    grad_output = torch.randn_like(torch_out)
    torch_out.backward(grad_output)
    triton_out.backward(Tensor(grad_output))
    
    # Check gradients
    for i, (torch_input, triton_input) in enumerate(zip(torch_inputs, triton_inputs)):
        torch.testing.assert_close(triton_input.grad, torch_input.grad, rtol=rtol, atol=atol)
    print(f"✓ Backward pass matches")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to vary
        x_vals=[2**i for i in range(12, 25, 1)],  # Different input sizes
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['torch', 'triton'],  # Possible values for line_arg
        line_names=['PyTorch', 'Triton'],  # Label name for different lines
        styles=[('blue', '-'), ('red', '-')],  # Line styles
        ylabel='TFLOPS',  # Label name for y-axis
        plot_name='operation-performance',  # Name for plot
        args={},  # Values for constant arguments
    )
)
def benchmark_operation(size, provider,
                       triton_fn, torch_fn,
                       input_shapes_fn,
                       device=DEVICE):
    """
    Benchmark Triton operation against PyTorch.
    
    Args:
        size: Size parameter to generate input shapes
        provider: 'torch' or 'triton'
        triton_fn: Triton implementation
        torch_fn: PyTorch implementation
        input_shapes_fn: Function that takes size and returns list of input shapes
        device: Device to run on
    """
    # Generate input shapes and data
    shapes = input_shapes_fn(size)
    inputs = [torch.randn(shape, device=device) for shape in shapes]
    
    # Select implementation
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        triton_inputs = [Tensor(x) for x in inputs]
        fn = lambda: triton_fn(*triton_inputs)
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # Calculate TFLOPS
    flops = sum(2 * prod(shape) for shape in shapes)  # Adjust FLOPS calculation per operation
    return flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    # Test addition
    def triton_add(x, y): return x + y
    def torch_add(x, y): return x + y
    test_operation(
        "addition",
        triton_add,
        torch_add,
        [(1024, 1024), (1024, 1024)]
    )
    
    # Benchmark addition
    def add_shapes(size): return [(size, size), (size, size)]
    benchmark_operation.run(
        print_data=True,
        triton_fn=triton_add,
        torch_fn=torch_add,
        input_shapes_fn=add_shapes
    )
    
    # Test broadcasting
    test_operation(
        "addition with broadcasting",
        triton_add,
        torch_add,
        [(1024, 1024), (1, 1024)]
    )