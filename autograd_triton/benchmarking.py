from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import torch
import triton
import triton.language as tl

from engine import TritonTensor

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
BATCH, N_HEADS, SEQ_LEN, DIM = 32, 8, 1024, 64 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM



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
    inputs = [torch.randn(shape, device=device) for shape in shapes]
    
    # Select implementation
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        triton_inputs = [TritonTensor(x) for x in inputs]
        fn = lambda: triton_fn(*triton_inputs)
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # TODO make TFLOPS calc a function that you pass in, and separate it between forward & backward
    # Calculate TFLOPS
    flops = sum(2 * prod(shape) for shape in shapes)  # Adjust FLOPS calculation per operation
    return flops * 1e-12 / (ms * 1e-3)

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
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device) for shape in shapes]
    
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        triton_inputs = [TritonTensor(x) for x in inputs]
        fn = lambda: triton_fn(*triton_inputs)
    
    ms = triton.testing.do_bench(fn)
    flops = sum(2 * prod(shape) for shape in shapes)
    return flops * 1e-12 / (ms * 1e-3)

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
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device) for shape in shapes]
    
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        triton_inputs = [TritonTensor(x) for x in inputs]
        fn = lambda: triton_fn(*triton_inputs)
    
    ms = triton.testing.do_bench(fn)
    flops = sum(2 * prod(shape) for shape in shapes)
    return flops * 1e-12 / (ms * 1e-3)

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
    shapes = input_shapes_fn(int(total_elements ** 0.5), broadcasting)  # Take square root for 2D tensors
    inputs = [torch.randn(shape, device=device) for shape in shapes]
    
    if provider == 'torch':
        fn = lambda: torch_fn(*inputs)
    else:
        triton_inputs = [TritonTensor(x) for x in inputs]
        fn = lambda: triton_fn(*triton_inputs)
    
    ms = triton.testing.do_bench(fn)
    flops = sum(2 * prod(shape) for shape in shapes)
    return flops * 1e-12 / (ms * 1e-3)

# Create "matmul" configs
matmul_configs = []
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
    
    args = parser.parse_args()
    
    # If no args are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)
    
    if args.all or args.add:
        print("\nRunning addition benchmarks...")
        def triton_add(x, y): return x + y
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
        def triton_sub(x, y): return x - y
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
        def triton_mul(x, y): return x * y
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
        def triton_div(x, y): return x / y
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
        benchmark_div.run(
            print_data=True,
            triton_fn=triton_matmul,
            torch_fn=torch_matmul,
            input_shapes_fn=matmul_shapes,
            save_path='./benchmarks/'
        )