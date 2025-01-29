from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import torch
import triton
import triton.language as tl

from engine import TritonTensor
from kernels import hadamard

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
BATCH, N_HEADS, SEQ_LEN, DIM = 32, 8, 1024, 64 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM

########################################################################################
########################### Elementwise Operations ############################################
########################################################################################

class _hadamard(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, op_name): 
        """a simple hadamard (entry-wise) operation that supports broadcasting of `other` up to size `self`"""
        # Ensures all tensors are on the same GPU device and of the same dtype
        assert a.device == b.device
        assert a.is_contiguous() and b.is_contiguous()

        # getting the total number of entries of each of our inputs
        n_elements = a.numel()
        loop_stride = b.numel() 

        # restricting the possible set of inputs to those which are logically broadcastable.
        # if we didn't do this then later our kernel would compute nonsensical broadcasting values
        if a.shape != b.shape:
            ptr = 0
            for d in a.shape:
                if ptr == b.ndim: break
                if d == b.shape[ptr]:
                    ptr += 1
            assert ptr == b.ndim, \
            f"for broadcasting to work, all dims in a ({a.shape}) must be a subset of those in b ({b.shape})"

        # Preallocating the output
        c = torch.empty_like(a)

        # Define grid based on tensor dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        # Launch kernel
        hadamard.binary_op_forward[grid](
            a, b, c, 
            n_elements, loop_stride,
            OP=op_name, # designates which operation to run (addition, subtraction, multiplication, division)
        )

        ctx.save_for_backward(a, b)
        ctx.grid = grid
        ctx.n_elements = n_elements
        ctx.loop_stride = loop_stride
        ctx.op_name = op_name
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        # reusing the same grid from earlier
        hadamard.binary_op_backward[ctx.grid](
            a, b,
            da, db, dc, 
            ctx.n_elements, ctx.loop_stride,
            OP=ctx.op_name, # designates which operation to run (addition, subtraction, multiplication, division
        )
        return da, db, None


hadamard_fn = _hadamard.apply

########################################################################################
########################### Elementwise Addition ############################################
########################################################################################

addition_configs = []
for mode in ["fwd", "bwd"]:
    for broadcasting in [True, False]:
        addition_configs.append(
            triton.testing.Benchmark(
                x_names=['total_elements'],  # Argument names to vary
                x_vals=[2**i for i in range(12, 24, 1)],  # Different input sizes
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                line_vals=['torch', 'triton'],  # Possible values for line_arg
                line_names=['PyTorch', 'Triton'],  # Label name for different lines
                styles=[('blue', '-'), ('red', '-')],  # Line styles
                ylabel='TFLOPS',  # Label name for y-axis
                xlabel="Total Elements (millions)", # Label name for x-axis
                plot_name=f'add_{mode}_broadcasting={broadcasting}',  # Name for plot
                args={"mode": mode, "broadcasting": broadcasting,},
            ))
@triton.testing.perf_report(addition_configs)
def benchmark_addition(total_elements, provider,
                       triton_fn, torch_fn,
                       input_shapes_fn,
                       mode,
                       broadcasting,
                       device=DEVICE):
    """
    Benchmark Triton addition against PyTorch.
    
    Args:
        total_elements: Total number of elements in the tensors
        provider: 'torch' or 'triton'
        triton_fn: Triton implementation
        torch_fn: PyTorch implementation
        input_shapes_fn: Function that takes size and returns list of input shapes
        device: Device to run on
    """
    # Generate input shapes and data
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
    
    # Select implementation
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        fn = lambda: triton_fn(*(inputs + ["add"]))
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # TODO make TFLOPS calc a function that you pass in, and separate it between forward & backward
    # Calculate TFLOPS
    flops = sum(2 * prod(shape) for shape in shapes)  # Adjust FLOPS calculation per operation
    return flops * 1e-12 / (ms * 1e-3)

########################################################################################
########################### Elementwise Subtraction ############################################
########################################################################################

# Create "sub" configs
sub_configs = []
for mode in ["fwd", "bwd"]:
    for broadcasting in [True, False]:
        sub_configs.append(
            triton.testing.Benchmark(
                x_names=['total_elements'],  # Argument names to vary
                x_vals=[2**i for i in range(12, 24, 1)],  # Different input sizes
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                line_vals=['torch', 'triton'],  # Possible values for line_arg
                line_names=['PyTorch', 'Triton'],  # Label name for different lines
                styles=[('blue', '-'), ('red', '-')],  # Line styles
                ylabel='TFLOPS',  # Label name for y-axis
                xlabel="Total Elements (millions)", # Label name for x-axis
                plot_name=f'sub_{mode}_broadcasting={broadcasting}',  # Name for plot
                args={"mode": mode, "broadcasting": broadcasting,},
            ))

@triton.testing.perf_report(sub_configs)
def benchmark_sub(total_elements, provider,
                  triton_fn, torch_fn,
                  input_shapes_fn,
                  mode,
                  broadcasting,
                  device=DEVICE):
    # Generate input shapes and data
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
    
    # Select implementation
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        fn = lambda: triton_fn(*(inputs + ["sub"]))
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # TODO make TFLOPS calc a function that you pass in, and separate it between forward & backward
    # Calculate TFLOPS
    flops = sum(2 * prod(shape) for shape in shapes)  # Adjust FLOPS calculation per operation
    return flops * 1e-12 / (ms * 1e-3)

########################################################################################
########################### Elementwise Multiplication ############################################
########################################################################################

# Create "mul" configs
mul_configs = []
for mode in ["fwd", "bwd"]:
    for broadcasting in [True, False]:
        mul_configs.append(
            triton.testing.Benchmark(
                x_names=['total_elements'],  # Argument names to vary
                x_vals=[2**i for i in range(12, 24, 1)],  # Different input sizes
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                line_vals=['torch', 'triton'],  # Possible values for line_arg
                line_names=['PyTorch', 'Triton'],  # Label name for different lines
                styles=[('blue', '-'), ('red', '-')],  # Line styles
                ylabel='TFLOPS',  # Label name for y-axis
                xlabel="Total Elements (millions)", # Label name for x-axis
                plot_name=f'mul_{mode}_broadcasting={broadcasting}',  # Name for plot
                args={"mode": mode, "broadcasting": broadcasting,},
            ))

@triton.testing.perf_report(mul_configs)
def benchmark_mul(total_elements, provider,
                  triton_fn, torch_fn,
                  input_shapes_fn,
                  mode,
                  broadcasting,
                  device=DEVICE):
    # Generate input shapes and data
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
    
    # Select implementation
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        fn = lambda: triton_fn(*(inputs + ["mul"]))
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # TODO make TFLOPS calc a function that you pass in, and separate it between forward & backward
    # Calculate TFLOPS
    flops = sum(2 * prod(shape) for shape in shapes)  # Adjust FLOPS calculation per operation
    return flops * 1e-12 / (ms * 1e-3)

########################################################################################
########################### Elementwise Division ############################################
########################################################################################

# Create "div" configs
div_configs = []
for mode in ["fwd", "bwd"]:
    for broadcasting in [True, False]:
        div_configs.append(
            triton.testing.Benchmark(
                x_names=['total_elements'],  # Argument names to vary
                x_vals=[2**i for i in range(12, 24, 1)],  # Different input sizes
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                line_vals=['torch', 'triton'],  # Possible values for line_arg
                line_names=['PyTorch', 'Triton'],  # Label name for different lines
                styles=[('blue', '-'), ('red', '-')],  # Line styles
                ylabel='TFLOPS',  # Label name for y-axis
                xlabel="Total Elements (millions)", # Label name for x-axis
                plot_name=f'div_{mode}_broadcasting={broadcasting}',  # Name for plot
                args={"mode": mode, "broadcasting": broadcasting,},
            ))

@triton.testing.perf_report(div_configs)
def benchmark_div(total_elements, provider,
                  triton_fn, torch_fn,
                  input_shapes_fn,
                  mode,
                  broadcasting,
                  device=DEVICE):
    # Generate input shapes and data
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]
    
    # Select implementation
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        fn = lambda: triton_fn(*(inputs + ["div"]))
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # TODO make TFLOPS calc a function that you pass in, and separate it between forward & backward
    # Calculate TFLOPS
    flops = sum(2 * prod(shape) for shape in shapes)  # Adjust FLOPS calculation per operation
    return flops * 1e-12 / (ms * 1e-3)

########################################################################################
########################### Matrix Multiplication ############################################
########################################################################################

# Create "matmul" configs
matmul_configs = []
for mode in ["fwd", "bwd"]:
    for broadcasting in [True, False]:
        matmul_configs.append(
            triton.testing.Benchmark(
                x_names=['total_elements'],  # Argument names to vary
                x_vals=[2**i for i in range(12, 24, 1)],  # Different input sizes
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                line_vals=['torch', 'triton'],  # Possible values for line_arg
                line_names=['PyTorch', 'Triton'],  # Label name for different lines
                styles=[('blue', '-'), ('red', '-')],  # Line styles
                ylabel='TFLOPS',  # Label name for y-axis
                xlabel="Total Elements (millions)", # Label name for x-axis
                plot_name=f'matmul_{mode}_broadcasting={broadcasting}',  # Name for plot
                args={"mode": mode, "broadcasting": broadcasting,},
            ))

@triton.testing.perf_report(matmul_configs)
def benchmark_matmul(total_elements, provider,
                  triton_fn, torch_fn,
                  input_shapes_fn,
                  mode,
                  broadcasting,
                  device=DEVICE):
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device) for shape in shapes]
    # TODO fix shape and TFLOP calculation to make sense w/ each other
    
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        triton_inputs = [TritonTensor(x) for x in inputs]
        fn = lambda: triton_fn(*triton_inputs)
    
    ms = triton.testing.do_bench(fn)
    flops = sum(2 * prod(shape) for shape in shapes)
    return flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmarks for Triton operations')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--add', action='store_true', help='Run addition benchmarks')
    parser.add_argument('--sub', action='store_true', help='Run subtraction benchmarks')
    parser.add_argument('--mul', action='store_true', help='Run multiplication benchmarks')
    parser.add_argument('--div', action='store_true', help='Run division benchmarks')
    parser.add_argument('--matmul', action='store_true', help='Run matrix multiplication benchmarks')
    
    args = parser.parse_args()
    
    # If no args are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)
    
    if args.all or args.add:
        print("\nRunning addition benchmarks...")
        def triton_add(x, y, op): return hadamard_fn(x, y, op)
        def torch_add(x, y): return x + y
        def add_shapes(size, broadcasting): 
            return [(size, size), (size,)] if broadcasting else [(size, size), (size, size)] 
        benchmark_addition.run(
            print_data=True,
            triton_fn=triton_add,
            torch_fn=torch_add,
            input_shapes_fn=add_shapes,
            save_path='./benchmarks/'
        )

    if args.all or args.sub:
        print("\nRunning subtraction benchmarks...")
        def triton_sub(x, y, op): return hadamard_fn(x, y, op)
        def torch_sub(x, y): return x - y
        def sub_shapes(size, broadcasting):
            return [(size, size), (size,)] if broadcasting else [(size, size), (size, size)]
        benchmark_sub.run(
            print_data=True,
            triton_fn=triton_sub,
            torch_fn=torch_sub,
            input_shapes_fn=sub_shapes,
            save_path='./benchmarks/'
        )

    if args.all or args.mul:
        print("\nRunning multiplication benchmarks...")
        def triton_mul(x, y, op): return hadamard_fn(x, y, op)
        def torch_mul(x, y): return x * y
        def mul_shapes(size, broadcasting):
            return [(size, size), (size,)] if broadcasting else [(size, size), (size, size)]
        benchmark_mul.run(
            print_data=True,
            triton_fn=triton_mul,
            torch_fn=torch_mul,
            input_shapes_fn=mul_shapes,
            save_path='./benchmarks/'
        )

    if args.all or args.div:
        print("\nRunning division benchmarks...")
        def triton_div(x, y, op): return hadamard_fn(x, y, op)
        def torch_div(x, y): return x / y
        def div_shapes(size, broadcasting):
            return [(size, size), (size,)] if broadcasting else [(size, size), (size, size)]
        benchmark_div.run(
            print_data=True,
            triton_fn=triton_div,
            torch_fn=torch_div,
            input_shapes_fn=div_shapes,
            save_path='./benchmarks/'
        )

    if args.all or args.matmul:
        print("\nRunning matmul benchmarks...")
        def triton_matmul(x, y): return x@ y
        def torch_matmul(x, y): return x @ y
        def matmul_shapes(size, broadcasting):
            return [(size, size, size // 2), (size // 2, size)] if broadcasting else [(size, size // 2), (size // 2, size * 2)]
            # TODO fix TFLOPS calculation in graphs
        benchmark_matmul.run(
            print_data=True,
            triton_fn=triton_matmul,
            torch_fn=torch_matmul,
            input_shapes_fn=matmul_shapes,
            save_path='./benchmarks/'
        )