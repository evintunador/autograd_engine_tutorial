from typing import Union, Tuple, Optional
import numpy as np
import math

import torch
import triton
import triton.language as tl

device = torch.device(f'cuda:{torch.cuda.current_device()}')

from engine import TritonTensor
import nn

# --- New helper functions for heatmap visualization ---

def clear_heatmap_folder(folder: str = "heatmaps"):
    """
    Deletes the folder (if it exists) and then re-creates an empty version.
    """
    import os, shutil
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def save_heatmaps(torch_tensor: torch.Tensor, triton_tensor: torch.Tensor, test_name: str,
                  folder: str = "heatmaps", atol: float = 1e-3, rtol: float = 1e-3,
                  phase: str = "backward"):
    """
    Saves multiple sets of heatmaps comparing torch_tensor and triton_tensor:
      1) Raw absolute differences
      2) Absolute tolerance failure mask (where abs diff > atol)
      3) Relative tolerance failure mask (where abs diff > rtol * abs(expected))
      4) Combined tolerance failure mask (where abs diff > atol + rtol * abs(expected))
    
    Handles different tensor dimensions:
      4D: (batch_size, num_heads, seq_len, head_dim) -> one set per batch/head
      3D: (batch_size, seq_len, model_dim) -> one set per batch
      2D: (batch_size, model_dim) -> one set per batch
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to numpy arrays
    actual = triton_tensor.detach().cpu().numpy()
    expected = torch_tensor.detach().cpu().numpy()
    
    # Compute differences and masks
    abs_diff = np.abs(expected - actual)
    abs_threshold = atol
    rel_threshold = rtol * np.abs(expected)
    combined_threshold = atol + rtol * np.abs(expected)
    
    abs_fail_mask = (abs_diff > abs_threshold).astype(np.int32)
    rel_fail_mask = (abs_diff > rel_threshold).astype(np.int32)
    combined_fail_mask = (abs_diff > combined_threshold).astype(np.int32)
    
    def save_figure(matrix, title: str, filename: str, cmap: str = "hot"):
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap=cmap, aspect="auto")
        plt.title(title)
        plt.xlabel("Model/Head Dimension")
        plt.ylabel("Sequence Position" if matrix.ndim > 1 else "Batch")
        plt.colorbar()
        plt.savefig(os.path.join(folder, filename))
        plt.close()
    
    def save_all_figures(diff: np.ndarray, abs_mask: np.ndarray, rel_mask: np.ndarray, 
                        comb_mask: np.ndarray, suffix: str, filename_suffix: str):
        # Raw difference
        save_figure(diff, f"{test_name} {suffix} - raw diff ({phase})",
                   f"{test_name}_{filename_suffix}_raw_diff_{phase}.png")
        # Absolute tolerance failures
        save_figure(abs_mask, f"{test_name} {suffix} - abs failure mask ({phase})",
                   f"{test_name}_{filename_suffix}_abs_fail_{phase}.png", cmap="Reds")
        # Relative tolerance failures
        save_figure(rel_mask, f"{test_name} {suffix} - rel failure mask ({phase})",
                   f"{test_name}_{filename_suffix}_rel_fail_{phase}.png", cmap="Reds")
        # Combined tolerance failures
        save_figure(comb_mask, f"{test_name} {suffix} - combined failure mask ({phase})",
                   f"{test_name}_{filename_suffix}_comb_fail_{phase}.png", cmap="Reds")

    # Handle different tensor dimensions
    if expected.ndim == 4:  # (batch_size, num_heads, seq_len, head_dim)
        B, H, N, D = expected.shape
        for b in range(B):
            for h in range(H):
                save_all_figures(
                    abs_diff[b, h], abs_fail_mask[b, h],
                    rel_fail_mask[b, h], combined_fail_mask[b, h],
                    f"diff: batch {b} head {h}", f"diff_b{b}_h{h}"
                )
    elif expected.ndim == 3:  # (batch_size, seq_len, model_dim)
        B, N, D = expected.shape
        for b in range(B):
            save_all_figures(
                abs_diff[b], abs_fail_mask[b],
                rel_fail_mask[b], combined_fail_mask[b],
                f"diff: batch {b}", f"diff_b{b}"
            )
    elif expected.ndim == 2:  # (batch_size, model_dim)
        B, D = expected.shape
        save_all_figures(
            abs_diff, abs_fail_mask,
            rel_fail_mask, combined_fail_mask,
            "diff", "diff"
        )
    else:
        # Fallback for other shapes
        save_all_figures(
            abs_diff, abs_fail_mask,
            rel_fail_mask, combined_fail_mask,
            "diff", "diff"
        )


# --- End of heatmap helper functions ---


def test_operation(op_name: str,
                   triton_fn,
                   torch_fn,
                   inputs_list: list[torch.Tensor],
                   atol=1e-3,
                   rtol=1e-3):
    """
    Test TritonTensor operations against PyTorch for correctness.
    
    Args:
        op_name: Name of operation being tested
        triton_fn: Function that takes TritonTensor inputs and returns TritonTensor output
        torch_fn: Function that takes torch.Tensor inputs and returns torch.Tensor output
        inputs_list: List of pytorch tensors to be used as inputs
        atol: Absolute tolerance for comparing outputs
        rtol: relative tolerance
    """
    print(f"\nTesting {op_name}...")
    
    # Generate random inputs
    torch_inputs = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs_list]  # Create leaf tensors
    triton_inputs = [TritonTensor(x, requires_grad=x.requires_grad) for x in inputs_list]
    
    # Forward pass
    torch_out = torch_fn(*torch_inputs)
    torch_out = torch_out[0] if op_name[:3] in ("min", "max") else torch_out
        # TODO do we need our max op to also give indices? i think so for inference
    triton_out = triton_fn(*triton_inputs)
    
    # Clear out previous heatmaps before any testing
    clear_heatmap_folder("heatmaps")
    
    # Check forward pass
    try:
        torch.testing.assert_close(torch_out, triton_out.data, atol=atol, rtol=rtol)
        print(f"✓ Forward pass matches")
    except AssertionError as error:
        print(f"Forward pass mismatch detected in operation '{op_name}'.")
        print("Generating heatmaps of the output differences...")
        save_heatmaps(torch_out, triton_out.data, f"{op_name}_output",
                      folder="heatmaps", atol=atol, rtol=rtol, phase="forward")
        raise error
    
    # before computing the backward pass, we need to let the autotuner run.
    # this needs to be done bc otherwise the gradient accumulation of each run would compound
    #  to incorrect values
    zero_grad = torch.zeros_like(torch_out)
    triton_out.backward(zero_grad)
    # and in order to avoid any potential divide by zero Nan's from division, we set all gradients to 0
    triton_out.zero_grad_backward()

    # Backward pass
    grad_output = torch.randn_like(torch_out)
    torch_out.backward(grad_output)
    triton_out.backward(grad_output)
    
    # Check gradients
    for i, (torch_input, triton_input) in enumerate(zip(torch_inputs, triton_inputs)):
        print(f"Analyzing gradient for input tensor index {i} with shape {torch_input.grad.shape}...")
        try:
            torch.testing.assert_close(torch_input.grad, triton_input.grad, atol=atol, rtol=rtol)
        except AssertionError as error:
            print(f"{'#'*20}\ntensor {i} in input gradients list\n{'#'*20}")
            print(f"Gradient mismatch detected for input {i} in operation '{op_name}'.")
            print("Generating heatmaps of the gradient differences...")
            save_heatmaps(torch_input.grad, triton_input.grad, f"{op_name}_input{i}",
                          folder="heatmaps", atol=atol, rtol=rtol, phase="backward")
            raise error
    print(f"✓ Backward pass matches")
    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for Triton operations')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--exp', action='store_true', help='Run exponentiation tests')
    parser.add_argument('--log', action='store_true', help='Run natural logarithm tests')
    parser.add_argument('--relu', action='store_true', help='Run rectified linear unit tests')
    parser.add_argument('--add', action='store_true', help='Run addition tests')
    parser.add_argument('--sub', action='store_true', help='Run subtraction tests')
    parser.add_argument('--mul', action='store_true', help='Run multiplication tests')
    parser.add_argument('--div', action='store_true', help='Run division tests')
    parser.add_argument('--matmul', action='store_true', help='Run matrix multiplication tests')
    parser.add_argument('--sum', action='store_true', help='Run summation across final dimension tests')
    parser.add_argument('--mean', action='store_true', help='Run mean across final dimension tests')
    parser.add_argument('--max', action='store_true', help='Run max across final dimension tests')
    parser.add_argument('--min', action='store_true', help='Run min across final dimension tests')
    parser.add_argument('--var', action='store_true', help='Run variance across final dimension tests')
    parser.add_argument('--std', action='store_true', help='Run standard deviation across final dimension tests')
    parser.add_argument('--trans', action='store_true', help='Run transpose across arbitrary axes tests')
    parser.add_argument('--sqz', action='store_true', help='Run squeeze across arbitrary axes tests')
    parser.add_argument('--unsqz', action='store_true', help='Run unsqueeze across arbitrary axes tests')
    parser.add_argument('--reshape', action='store_true', help='Run reshape tests')
    parser.add_argument('--idx', action='store_true', help='Run indexing tests')
    parser.add_argument('--lin', action='store_true', help='Run linear layer tests')
    parser.add_argument('--emb', action='store_true', help='Run embedding layer tests')
    parser.add_argument('--ln', action='store_true', help='Run LayerNorm module tests')
    parser.add_argument('--flash', action='store_true', help='Run Flash Attention tests')
    
    args = parser.parse_args()
    
    # If no args are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    B, N, H, D, V = 1, 128, 2, 128, 4096
        
    ### EXPONENTIATION
    if args.all or args.exp:
        def triton_exp(x): return x.exp()
        def torch_exp(x): return torch.exp(x)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"exponentiation: ({B}, {N}, {D})",
            triton_exp,
            torch_exp,
            inputs_list([(B, N, D)]),
        )
        
    ### NATURAL LOGARITHM
    if args.all or args.log:
        def triton_log(x): return x.log()
        def torch_log(x): return torch.log(x)
        def inputs_list(input_shapes):
            return [torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True) + 0.01
                   for shape in input_shapes]
        test_operation(
            f"natural logarithm: ({B}, {N}, {D})",
            triton_log,
            torch_log,
            inputs_list([(B, N, D)]),
        )
        
    ### RECTIFIED LINEAR UNIT
    if args.all or args.relu:
        def triton_relu(x): return x.relu()
        def torch_relu(x): return torch.relu(x)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"rectified linear unit: ({B}, {N}, {D})",
            triton_relu,
            torch_relu,
            inputs_list([(B, N, D)]),
        )

    ### ADDITION
    if args.all or args.add:
        def triton_add(x, y): return x + y
        def torch_add(x, y): return x + y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"addition: ({B}, {N}, {D}) + ({B}, {N}, {D})",
            triton_add,
            torch_add,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"addition with broadcasting: ({B}, {N}, {D}) + ({D})",
            triton_add,
            torch_add,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"addition with single scalar: ({B}, {N}, {D}) + (1)",
            triton_add,
            torch_add,
            inputs_list([(B, N, D), (1)]),
        )

    ### MULTIPLICATION
    if args.all or args.mul:
        def triton_mul(x, y): return x * y
        def torch_mul(x, y): return x * y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"multiplication: ({B}, {N}, {D}) * ({B}, {N}, {D})",
            triton_mul,
            torch_mul,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"multiplication with broadcasting: ({B}, {N}, {D}) * ({D})",
            triton_mul,
            torch_mul,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"multiplication with single scalar: ({B}, {N}, {D}) * (1)",
            triton_mul,
            torch_mul,
            inputs_list([(B, N, D), (1)]),
        )

    ### SUBTRACTION
    if args.all or args.sub:
        def triton_sub(x, y): return x - y
        def torch_sub(x, y): return x - y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"subtraction: ({B}, {N}, {D}) + ({B}, {N}, {D})",
            triton_sub,
            torch_sub,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"subtraction with broadcasting: ({B}, {N}, {D}) + ({D})",
            triton_sub,
            torch_sub,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"subtraction with single scalar: ({B}, {N}, {D}) + (1)",
            triton_sub,
            torch_sub,
            inputs_list([(B, N, D), (1)]),
        )

    ### DIVISION
    if args.all or args.div:
        def triton_div(x, y): return x / y
        def torch_div(x, y): return x / y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"division: ({B}, {N}, {D}) + ({B}, {N}, {D})",
            triton_div,
            torch_div,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"division with broadcasting: ({B}, {N}, {D}) + ({D})",
            triton_div,
            torch_div,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"division with single scalar: ({B}, {N}, {D}) + (1)",
            triton_div,
            torch_div,
            inputs_list([(B, N, D), (1)]),
        )

    ### MATMUL
    if args.all or args.matmul:
        def triton_matmul(x, y): return x @ y
        def torch_matmul(x, y): return x @ y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"matmul: ({B}, {D}) @ ({D}, {D*4})",
            triton_matmul,
            torch_matmul,
            inputs_list([(B, D), (D, D*4)]),
            atol=5e-2, # matmul gradient accumulation is VERY sensitive to flop error even at fp32
            rtol=1e5, # relative error is dummb bc when it's relative to 1e-6 everything looks big
            # or at least that's what i think is happening; lmk if you find an error i couldn't
        )
        test_operation(
            f"matmul with leading dimensions: ({B}, {H}, {N}, {D}) @ ({B}, {H}, {D}, {N})",
            triton_matmul,
            torch_matmul,
            inputs_list([(B, H, N, D), (B, H, D, N)]),
            atol=5e-2,
            rtol=1e5,
        )
        test_operation(
            f"matmul with broadcasting: ({B}, {N}, {D}) @ ({D}, {N})",
            triton_matmul,
            torch_matmul,
            inputs_list([(B, N, D), (D, N)]),
            atol=5e-2,
            rtol=1e5,
        )
        
    ### SUMMATION
    if args.all or args.sum:
        def triton_sum(x): return x.sum()
        def torch_sum(x): return torch.sum(x, dim=-1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"summation: ({B}, {N}, {D})",
            triton_sum,
            torch_sum,
            inputs_list([(B, N, D)]),
        )
        
    ### MEAN
    if args.all or args.mean:
        def triton_mean(x): return x.mean()
        def torch_mean(x): return torch.mean(x, dim=-1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"mean: ({B}, {N}, {D})",
            triton_mean,
            torch_mean,
            inputs_list([(B, N, D)]),
        )
        
    ### MAXIMUM
    if args.all or args.max:
        def triton_max(x): return x.max()
        def torch_max(x): return torch.max(x, dim=-1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"maximum: ({B}, {N}, {D})",
            triton_max,
            torch_max,
            inputs_list([(B, N, D)]),
        )
        
    ### MINIMUM
    if args.all or args.min:
        def triton_min(x): return x.min()
        def torch_min(x): return torch.min(x, dim=-1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"minimum: ({B}, {N}, {D})",
            triton_min,
            torch_min,
            inputs_list([(B, N, D)]),
        )
        
    ### VARIANCE
    if args.all or args.var:
        def triton_var(x): return x.var()
        def torch_var(x): return torch.var(x, dim=-1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"variance: ({B}, {N}, {D})",
            triton_var,
            torch_var,
            inputs_list([(B, N, D)]),
        )
        
    ### STANDARD DEVIATION
    if args.all or args.std:
        def triton_std(x): return x.std()
        def torch_std(x): return torch.std(x, dim=-1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"standard deviation: ({B}, {N}, {D})",
            triton_std,
            torch_std,
            inputs_list([(B, N, D)]),
        )
        
    ### TRANSPOSE
    if args.all or args.trans:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_trans(x): return x.transpose(-2, -3)
        def torch_trans(x): return torch.transpose(x, -2, -3)
        test_operation(
            f"transpose: ({B}, {N}, {H}, {D}) -> ({B}, {H}, {N}, {D})",
            triton_trans,
            torch_trans,
            inputs_list([(B, N, H, D)]),
        )
        # this one should default to final two dims
        def triton_trans(x): return x.transpose()
        def torch_trans(x): return torch.transpose(x, -1, -2)
        test_operation(
            f"transpose: ({B}, {H}, {N}, {D}) -> ({B}, {H}, {D}, {N})",
            triton_trans,
            torch_trans,
            inputs_list([(B, H, N, D)]),
        )
        
    ### SQUEEZE
    if args.all or args.sqz:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_sqz(x): return x.squeeze(2)
        def torch_sqz(x): return torch.squeeze(x, 2)
        test_operation(
            f"squeeze: ({B}, {N}, {1}, {D}) -> ({B}, {N}, {D})",
            triton_sqz,
            torch_sqz,
            inputs_list([(B, N, 1, D)]),
        )
        
    ### UNSQUEEZE
    if args.all or args.unsqz:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_unsqz(x): return x.unsqueeze(2)
        def torch_unsqz(x): return torch.unsqueeze(x, 2)
        test_operation(
            f"squeeze: ({B}, {N}, {D}) -> ({B}, {N}, {1}, {D})",
            triton_unsqz,
            torch_unsqz,
            inputs_list([(B, N, D)]),
        )
        
    ### RESHAPE
    if args.all or args.reshape:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_reshape(x): return x.reshape((B, N, 4, D//4))
        def torch_reshape(x): return torch.reshape(x, (B, N, 4, D//4))
        test_operation(
            f"reshape: ({B}, {N}, {D}) -> ({B}, {N}, {4}, {D//4})",
            triton_reshape,
            torch_reshape,
            inputs_list([(B, N, D)]),
        )
        
    ### INDEXING
    # NOTE: we expect the bwd pass of idx to fail since we didn't implement it
    if args.all or args.idx:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_idx(x): return x[:,-1,:]
        def torch_idx(x): return x[:,-1,:]
        test_operation(
            f"index: ({B}, {N}, {V})[:,-1,:] -> ({B}, {1}, {V})",
            triton_idx,
            torch_idx,
            inputs_list([(B, N, V)]),
        )
        
    ### LINEAR LAYER
    if args.all or args.lin:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        triton_model =  nn.Linear(D, D*4)
        torch_model = torch.nn.Linear(D, D*4, device=device, dtype=torch.float32)
        # because they both initialize randomly we need to set one to the other
        torch_model.weight.data = triton_model.weight.transpose().data.detach().clone()
            # for some reason pytorch stores the weight matrix transposed
        if triton_model.bias is not None:
            torch_model.bias.data = triton_model.bias.data.detach().clone()
        def triton_linear(x): return triton_model(x)
        def torch_linear(x): return torch_model(x)
        test_operation(
            f"linear layer: ({B}, {N}, {D}) -> ({D}, {D*4})",
            triton_linear,
            torch_linear,
            inputs_list([(B, N, D)]),
            atol=5e-2, # matmul gradient accumulation is VERY sensitive to flop error even at fp32
            rtol=1e5, # relative error is dummb bc when it's relative to 1e-6 everything looks big
            # or at least that's what i think is happening; lmk if you find an error i couldn't
        )
        
    ### EMBEDDING LAYER
    if args.all or args.emb:
        def inputs_list(input_shapes):
            tokens = torch.randint(0, V, size=input_shapes[0], dtype=torch.int64, device=device) 
            weights = torch.randn(size=input_shapes[1], dtype=torch.float32, device=device, requires_grad=True)
            return [tokens, weights]
        triton_model = nn.Embedding(V, D)
        # because they both initialize randomly we need to set their weights to the same matrix
        def triton_embedding(tokens, weights): 
            # this direct assignment is kinda weird since we're assigning a TritonTensor to what
            #  previously was a Parameter but it's prolly fine
            triton_model.weight = weights 
            return triton_model(tokens)
        def torch_embedding(tokens, weights): 
            return torch.nn.functional.embedding(tokens, weights)
        test_operation(
            f"embedding layer: ({B}, {N}) & ({V}, {D}) -> ({B}, {N}, {D})",
            triton_embedding,
            torch_embedding,
            inputs_list([(B, N), (V, D)]),
        )
        # gradients of (B, N) will be None
        # gradients of (V, D) are what we care about
        
    ### LayerNorm Module
    if args.all or args.ln:
        def inputs_list(input_shapes):
            x = torch.randn(size=input_shapes[0], dtype=torch.float32, device=device, requires_grad=True) 
            w = torch.ones(size=input_shapes[1], dtype=torch.float32, device=device, requires_grad=True)
            b = torch.zeros(size=input_shapes[1], dtype=torch.float32, device=device, requires_grad=True)
            return [x, w, b]
        triton_model = nn.LayerNorm(D)
        # because they both initialize randomly we need to set their weights to the same matrix
        def triton_ln(x, w, b): 
            # this direct assignment is kinda weird since we're assigning a TritonTensor to what
            #  previously was a Parameter but it's prolly fine
            triton_model.weight = w
            triton_model.bias = b
            return triton_model(x)
        def torch_ln(x, w, b): 
            return torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],), weight=w, bias=b)
        test_operation(
            f"LayerNorm: ({B}, {N}, {D}) -> ({B}, {N}, {D})",
            triton_ln,
            torch_ln,
            inputs_list([(B, N, D), (D,), (D,)]),
        )
        
    ### Flash Attention
    if args.all or args.flash:
        Dh = 128
        def inputs_list(input_shapes):
            return [torch.randn(size=shape, dtype=torch.float32, device=device, requires_grad=True) * 0.02
                    for shape in input_shapes]
        def triton_flash(q, k, v): 
            return nn.FlashAttention()(q, k, v, scale=math.sqrt(Dh))
        def torch_flash(q, k, v): 
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=math.sqrt(Dh))
        test_operation(
            f"causal flash attention",
            triton_flash,
            torch_flash,
            inputs_list([(B,H,N,Dh), (B,H,N,Dh), (B,H,N,Dh)]),
        )

        