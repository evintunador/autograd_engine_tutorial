"""
we're going to have to construct our own benchmarking setup since Triton's built-in
doesn't work with the backward pass of our TritonTensor wrapper class
"""
from typing import Union, Tuple, Optional
import numpy as np
from math import prod
import time
import pandas as pd
import matplotlib.pyplot as plt

import torch
import triton
import triton.language as tl

from engine import TritonTensor
from kernels import hadamard

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
BATCH, N_HEADS, SEQ_LEN, DIM = 32, 8, 1024, 64 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM
# TODO make sizes smart enough to not overload VRAM

def run_single_bench(fn, flops_calc_fn, shapes, rep=250, warmup=25):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Use fresh events for measurement
    times = []
    for _ in range(rep):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        fn()  # Should ONLY include the operation being measured
        end_event.record()
        torch.cuda.synchronize()
        
        times.append(start_event.elapsed_time(end_event))

    # Use median instead of average to filter outliers
    avg_time = np.median(times)
    return avg_time, flops_calc_fn(shapes, avg_time)

benchmark_cases = [
    { 
      "mode": mode,
      "broadcasting": broadcasting,
      "providers": ["torch", "triton"],
      "sizes": [128, 256, 512, 1024, 2048, 4096, 8192]
    }
    for mode in ["fwd"]#, "bwd"
    for broadcasting in [True, False]
]

def benchmark_hadamard(
    torch_fn, triton_fn, 
    op_name, 
    input_shapes_fn,
    flops_calc_fn,
    benchmark_cases=benchmark_cases
    ):
    
    for case in benchmark_cases:
        results = pd.DataFrame({"input_sizes": case["sizes"]})
        for provider in case["providers"]:
            provider_results = []
            for size in case["sizes"]:
                mode = case['mode']
                broadcasting = case["broadcasting"]

                # Generate input shapes and data
                shapes = input_shapes_fn(size, broadcasting)
                inputs = [torch.randn(shape, device=DEVICE) for shape in shapes]
                
                # Select implementation
                if provider == 'torch':
                    fn = lambda: torch_fn(*inputs)
                if provider == 'triton':
                    """
                    # Create fresh inputs once per size
                    triton_inputs = [TritonTensor(x.clone().detach().contiguous(),  # Ensure contiguous memory
                                                requires_grad=True) 
                                   for x in inputs]
                    for t in triton_inputs:
                        t.zero_grad()  # Clear any existing gradients
                    """

                    # getting the total number of entries of each of our inputs
                    n_elements = inputs[0].numel()
                    loop_stride = inputs[1].numel() # the name `loop_stride` will make sense in the kernel
                    
                    # Preallocating the output
                    output = torch.empty_like(inputs[0])
                    
                    # Define grid based on tensor dimensions
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
                    # define kernel as our function to avoid python overhead in measurements
                    fn = lambda: hadamard.binary_op_forward[grid](
                        inputs[0], inputs[1], output, 
                        n_elements, loop_stride,
                        OP=op_name, # designates which operation to run (addition, subtraction, multiplication, division)
                    )
                    #fn = lambda: triton_fn(*triton_inputs)
                """
                if mode == "bwd":
                    O = fn()
                    dO = torch.randn_like(O)
                    fn = lambda: O.backward(dO)
                """
                ms, tflops = run_single_bench(fn, flops_calc_fn, shapes)
                provider_results.append(tflops)
            results = pd.merge(results, pd.DataFrame({"input_sizes": case["sizes"], f"{provider}": provider_results}))

        print(results)
        
        # Save to CSV
        plot_name = f"{op_name}-{mode}-Broadcasting={broadcasting}"
        csv_path = f'./benchmarks/{plot_name}.csv'
        results.to_csv(csv_path, index=False)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        for provider in case["providers"]:
            plt.plot(results['input_sizes'], results[provider], 'o-', label=f'{provider}')
        plt.xlabel('Input Size')
        plt.ylabel('TFLOPS')
        plt.title(f'{op_name} {mode} - Broadcasting={broadcasting}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./benchmarks/{plot_name}.png')
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmarks for Triton operations')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--add', action='store_true', help='Run addition benchmarks')
    parser.add_argument('--sub', action='store_true', help='Run subtraction benchmarks')
    parser.add_argument('--mul', action='store_true', help='Run multiplication benchmarks')
    parser.add_argument('--div', action='store_true', help='Run division benchmarks')
    parser.add_argument('--matmul', action='store_true', help='Run matmul benchmarks')
    
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
        def flops_calc_add(shapes: list, ms: float):
            # TODO separate bw fwd & bwd
            flops = sum(2 * prod(shape) for shape in shapes)
            return flops * 1e-12 / (ms * 1e-3)
        benchmark_hadamard(
            torch_add, triton_add,
            op_name="add", 
            input_shapes_fn=add_shapes,
            flops_calc_fn=flops_calc_add
        )

    if args.all or args.matmul:
        print("\nRunning matmul benchmarks...")
        def triton_matmul(x, y): return x @ y
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