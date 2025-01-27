from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

from engine import TritonTensor

def test_operation(op_name: str,
                  triton_fn,
                  torch_fn,
                  input_shapes: list,
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
    torch_inputs = [torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True) 
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for Triton operations')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--add', action='store_true', help='Run addition tests')
    parser.add_argument('--sub', action='store_true', help='Run subtraction tests')
    parser.add_argument('--mul', action='store_true', help='Run multiplication tests')
    parser.add_argument('--div', action='store_true', help='Run division tests')
    parser.add_argument('--matmul', action='store_true', help='Run matrix multiplication tests')
    
    args = parser.parse_args()
    
    # If no args are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    B, N, H, D = 32, 2048, 8, 768

    ### ADDITION
    if args.all or args.add:
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
    if args.all or args.mul:
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
    if args.all or args.sub:
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
    if args.all or args.div:
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

    ### MATMUL
    if args.all or args.matmul:
        def triton_matmul(x, y): return x @ y
        def torch_matmul(x, y): return x @ y
        test_operation(
            f"matmul: ({N}, {D}) @ ({D}, {N})",
            triton_matmul,
            torch_matmul,
            [(N, D), (D, N)]
        )
        """
        test_operation(
            f"matmul with leading dimensions: ({B}, {H}, {N}, {D}) @ ({B}, {H}, {D}, {N})",
            triton_matmul,
            torch_matmul,
            [(B, H, N, D), (B, H, D, N)]
        )
        test_operation(
            f"matmul with broadcasting: ({B}, {N}, {D}) @ ({D}, {N})",
            triton_matmul,
            torch_matmul,
            [(B, N, D), (D, N)]
        )
        """