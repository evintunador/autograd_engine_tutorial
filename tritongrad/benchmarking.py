from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import torch
import triton
import triton.language as tl

from engine import TritonTensor
from kernels import elementwise, matmul, vectorwise, modules

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # each SM has a fixed amount of SRAM that it can access
    # if one SM isn't using all its available SRAM then another can be spun up to use the remainder

BATCH = 32


########################################################################################
########################### Unary Ops ############################################
########################################################################################


class _unary_op(torch.autograd.Function):
    """a simple unary operation """

    @staticmethod
    def forward(ctx, a, op_name): 
        assert a.is_contiguous() 
        n_elements = a.numel()

        # Preallocating the output
        b = torch.empty_like(a)

        # Define grid based on tensor dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        # Launch kernel
        elementwise.unary_op_forward[grid](
            a, b,
            n_elements,
            op_name, # designates which operation to run (exp, log, relu, etc)
        )

        ctx.save_for_backward(a, b)
        ctx.grid = grid
        ctx.n_elements = n_elements
        ctx.op_name = op_name
        return b

    @staticmethod
    def backward(ctx, db):
        a, b = ctx.saved_tensors
        da = torch.empty_like(a)
        # reusing the same grid from earlier
        elementwise.unary_op_backward[ctx.grid](
            a, da, 
            b, db, 
            ctx.n_elements,
            ctx.op_name, 
        )
        return da, None

unary_op_fn = _unary_op.apply

# Define the operations list based on input args
def get_unary_ops_args(args):
    ops = []
    if args.all or args.exp:
        ops.append("exp")
    if args.all or args.log:
        ops.append("log")
    if args.all or args.relu:
        ops.append("relu")
    return ops

# First define an empty list that will be populated before the decorator is used
unary_op_configs = []
def generate_unary_op_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['tot_elements'],
                    x_vals=[2**i for i in range(12, 24, 1)],
                    line_arg='provider',
                    line_vals=['torch', 'triton'],
                    line_names=['PyTorch', 'Triton'],
                    styles=[('blue', '-'), ('red', '-')],
                    ylabel='GB/s',
                    xlabel="Total elements per output tensor",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode,},
                ))
    return configs

@triton.testing.perf_report(unary_op_configs)
def benchmark_unary(tot_elements, provider, op, mode, device=DEVICE):
    """
    Benchmark Triton unary operations against PyTorch.
    
    Args:
        tot_elements: Total number of elements in the input tensor
        provider: 'torch' or 'triton'
        op: "exp", "log", "relu", etc; designates the operation to be performed
        mode: "fwd" or "bwd"
        device: Device to run on
    """
    # Generate input data
    dim = int(tot_elements ** 0.5)
    A = torch.randn((BATCH, dim, dim), device=device, requires_grad=True)
    
    # Select implementation
    if op == "exp":
        fn = lambda: unary_op_fn(A, op) if provider == 'triton' else torch.exp(A)
    if op == "log":
        fn = lambda: unary_op_fn(A, op) if provider == 'triton' else torch.log(A)
    if op == "relu":
        fn = lambda: unary_op_fn(A, op) if provider == 'triton' else torch.relu(A)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    # for entry-wise operations we'll measure memory throughput since that's the limiting factor
    if mode == "fwd": # all fwd passes have same mem read/write behavior
        gb = BATCH * 2 * tot_elements * 4 * 1e-9
        # 2 = number of memory operations (1 reads + 1 write)
        # 4) = bytes per element (for float32)
        # 1e-9 converts bytes to GB
    else: # bwd pass 
        gb = BATCH * 3 * tot_elements * 4 * 1e-9
    # 1e-3 converts milliseconds to seconds
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)

########################################################################################
########################### Binary Operations ##############################
########################################################################################

class _binary_op(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, op_name): 
        """a simple element-wise binary operation that supports broadcasting of `b` up to tot_elements `a`"""
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
        elementwise.binary_op_forward[grid](
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
        elementwise.binary_op_backward_dx[ctx.grid](
            b, da, dc, 
            ctx.n_elements, ctx.loop_stride,
            OP=ctx.op_name, 
        )
        elementwise.binary_op_backward_dy[ctx.grid](
            a, b, db, dc, 
            ctx.n_elements, ctx.loop_stride,
            OP=ctx.op_name, 
        )
        return da, db, None

binary_op_fn = _binary_op.apply

# Define the operations list based on input args
def get_binary_ops_args(args):
    ops = []
    if args.all or args.add:
        ops.append("add")
    if args.all or args.sub:
        ops.append("sub")
    if args.all or args.mul:
        ops.append("mul")
    if args.all or args.div:
        ops.append("div")
    return ops

# First define an empty list that will be populated before the decorator is used
binary_op_configs = []
def generate_binary_op_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            for broadcasting in [True, False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=['tot_elements'],
                        x_vals=[2**i for i in range(12, 24, 1)],
                        line_arg='provider',
                        line_vals=['torch', 'triton'],
                        line_names=['PyTorch', 'Triton'],
                        styles=[('blue', '-'), ('red', '-')],
                        ylabel='GB/s',
                        xlabel="Total elements per output tensor",
                        plot_name=f'{op}_{mode}_broadcasting={broadcasting}',
                        args={"op": op, "mode": mode, "broadcasting": broadcasting,},
                    ))
    return configs

@triton.testing.perf_report(binary_op_configs)
def benchmark_binary(tot_elements, provider, op, mode, broadcasting, device=DEVICE):
    """
    Benchmark Triton binary operations against PyTorch.
    
    Args:
        tot_elements: Total number of elements in the tensors
        provider: 'torch' or 'triton'
        op: "add", "sub", "mul", or "div"; designates the operation to be performed
        mode: "fwd" or "bwd"
        broadcasting: True for same-size inputs and False for smaller B to be broadcasted
        device: Device to run on
    """
    # Generate input data
    dim = int(tot_elements ** 0.5)
    A = torch.randn((BATCH, dim, dim), device=device, requires_grad=True)
    B = torch.randn((dim, ) if broadcasting else (BATCH, dim, dim), device=device, requires_grad=True)
    
    # Select implementation
    if op == "add":
        fn = lambda: binary_op_fn(A, B, op) if provider == 'triton' else A + B
    if op == "sub":
        fn = lambda: binary_op_fn(A, B, op) if provider == 'triton' else A - B
    if op == "mul":
        fn = lambda: binary_op_fn(A, B, op) if provider == 'triton' else A * B
    if op == "div":
        fn = lambda: binary_op_fn(A, B, op) if provider == 'triton' else A / B
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    # for entry-wise operations we'll measure memory throughput since that's the limiting factor
    if mode == "fwd": # all fwd passes have same mem read/write behavior
        gb = BATCH * 3 * tot_elements * 4 * 1e-9
        # 3 = number of memory operations (2 reads + 1 write)
        # 4) = bytes per element (for float32)
        # 1e-9 converts bytes to GB
    elif op in ("add", "sub"): # bwd pass of add or sub
        gb = BATCH * 5 * tot_elements * 4 * 1e-9
    elif op == "mul": # bwd pass of mul
        gb = BATCH * 7 * tot_elements * 4 * 1e-9
    else: # bwd div
        gb = BATCH * 8 * tot_elements * 4 * 1e-9
    # 1e-3 converts milliseconds to seconds
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


########################################################################################
########################### Matrix Multiplication ############################################
########################################################################################

class _matmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b): 
        """
        matmul implementation built to support only the shapes we need, so
        A: tensor @ B: tensor = C: tensor   for the self-attention mechanism and
        A: tensor @ B: matrix = C: tensor   for linear layers
        also we don't need this regular matmul shape layout, but hey why not
        A: matrix @ B: matrix = C: matrix
        """
        # check constraints
        assert a.ndim >=2 and b.ndim >= 2
        assert a.ndim >= b.ndim
        assert a.shape[-2] == b.shape[-1]
        if b.ndim > 2:
            assert a.shape[:-2] == b.shape[:-2]
        assert a.data.is_contiguous()

        # getting the total number of entries of each of our inputs
        n_elements = a.numel()
        loop_stride = b.numel() 

        # get matrix dimension lengths
        (m, k), n = a.shape[-2:], b.shape[-1]
        # how many batches and heads to parallelize along
        parallel_matrix_ct = prod(a.shape[:-2]) if a.ndim > 2 else 1

        # allocates output
        c = torch.empty(a.shape[:-2] + (m, n), device=a.device, dtype=torch.float32)

        # 2D launch kernel where each preceeding_dim and each block gets its own program
        grid = lambda meta: (
            triton.cdiv(m, meta['BLOCK_SIZE_M']) * triton.cdiv(n, meta['BLOCK_SIZE_N']), 
            parallel_matrix_ct
        )
        # Launch kernel
        matmul.matmul_fwd[grid](
            a, b, c,
            m, n, k,
            a.stride(-3) if a.ndim > 2 else 0, a.stride(-2), a.stride(-1),
            b.stride(-3) if b.ndim > 2 else 0, b.stride(-2), b.stride(-1),
            c.stride(-3) if c.ndim > 2 else 0, c.stride(-2), c.stride(-1),
        )

        ctx.save_for_backward(a, b)
        ctx.m = m
        ctx.n = n
        ctx.k = k
        ctx.parallel_matrix_ct = parallel_matrix_ct
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        
        bwd_grid_dA = lambda meta: (
            triton.cdiv(ctx.m, meta['BLOCK_SIZE_M']) * triton.cdiv(ctx.k, meta['BLOCK_SIZE_K']),
            ctx.parallel_matrix_ct
        )
        matmul.matmul_bwd_dA[bwd_grid_dA](
            b, da, db,
            ctx.m, ctx.n, ctx.k,
            b.stride(-3) if b.ndim > 2 else 0, b.stride(-2), b.stride(-1),
            da.stride(-3) if da.ndim > 2 else 0, da.stride(-2), da.stride(-1),
            db.stride(-3) if db.ndim > 2 else 0, db.stride(-2), db.stride(-1),
        )

        bwd_grid_dB = lambda meta: (
            triton.cdiv(ctx.k, meta['BLOCK_SIZE_K']) * triton.cdiv(ctx.n, meta['BLOCK_SIZE_N']),
            ctx.parallel_matrix_ct
        )
        matmul.matmul_bwd_dB[bwd_grid_dB](
            a, db, dc, 
            ctx.m, ctx.n, ctx.k,
            a.stride(-3) if a.ndim > 2 else 0, a.stride(-2), a.stride(-1),
            db.stride(-3) if db.ndim > 2 else 0, db.stride(-2), db.stride(-1),
            dc.stride(-3) if dc.ndim > 2 else 0, dc.stride(-2), dc.stride(-1),
        )

        return da, db

matmul_fn = _matmul.apply

# Create "matmul" configs
matmul_configs = []
for mode in ["fwd", "bwd"]:
    for broadcasting in [True, False]:
        matmul_configs.append(
            triton.testing.Benchmark(
                x_names=['M', 'N', 'K'],  # Argument names to vary
                x_vals=[128 * i for i in range(2, 28, 1)],  # Different input tot_elementss
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                line_vals=['torch', 'triton'],  # Possible values for line_arg
                line_names=['PyTorch', 'Triton'],  # Label name for different lines
                styles=[('blue', '-'), ('red', '-')],  # Line styles
                ylabel='TFLOPS',  # Label name for y-axis
                xlabel="M, N and K", # Label name for x-axis
                plot_name=f'matmul_{mode}_broadcasting={broadcasting}',  # Name for plot
                args={"mode": mode, "broadcasting": broadcasting,},
            ))
@triton.testing.perf_report(matmul_configs)
def benchmark_matmul(M, N, K, provider, mode, broadcasting, device=DEVICE):
    A = torch.randn((BATCH, M, K), device=device, requires_grad=True)
    B = torch.randn((K, N) if broadcasting else (BATCH, K, N), device=device, requires_grad=True)
    
    if provider == 'torch':
        fn = lambda: A @ B
    else: # triton
        fn = lambda: matmul_fn(A, B)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # for matmul we'll measure TFLOPs instead of GB/s since flops are the limiting factor
    ms = triton.testing.do_bench(fn)
    perf = (2 if mode == "fwd" else 4) * BATCH * M * N * K * 1e-12 / (ms * 1e-3)
    # 2 or 4 = number of operations per entry (mul and add for fwd & another set for two gradients during bwd)
    # BATCH * M * N * K = number of elements
    # 1e-12 converts flops to Teraflops
    # ms * 1e-3 converts milliseconds to seconds
    return perf 


########################################################################################
########################### Reduction Operations ############################################
########################################################################################

class _reduction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op_name): 
        """
        implementation of reduction ops
        only supports reduction along the final dimension
        """
        # check constraints
        assert x.data.is_contiguous()

        # allocates output
        y = torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype)

        # get tensor dimensions to ensure our parallelization scheme will work
        n_cols = x.shape[-1]
        BLOCK_SIZE_N = triton.next_power_of_2(n_cols)
        # 4 for the 4 bytes in fp32
        assert BLOCK_SIZE_N * 4 < TOTAL_SRAM_PER_SM, \
            f"vectors (each size {BLOCK_SIZE_N * 4}) too large to fit into SRAM size {TOTAL_SRAM_PER_SM}"

        # we'll parallelize with multiple rows in a PID
        grid = lambda meta: (triton.cdiv(x.numel() // n_cols, meta['BLOCK_SIZE_M']), )
        # Launch kernel
        vectorwise.reduction_op_forward[grid](
            x, y, 
            x.numel(), y.numel(), 
            x.stride()[-2], n_cols, 
            op = op_name, 
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        ctx.save_for_backward(x)
        ctx.op_name = op_name
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.n_cols = n_cols
        ctx.grid = grid
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = torch.empty(x.shape, device=dy.device, dtype=dy.dtype)
        grid = ctx.grid
        
        vectorwise.reduction_op_backward[grid](
            x,
            dx, dy,
            dx.numel(), dy.numel(),
            dx.stride()[-2], ctx.n_cols, 
            op = ctx.op_name, 
            BLOCK_SIZE_N = ctx.BLOCK_SIZE_N,
        )

        return dx, None

reduction_fn = _reduction.apply

# Define the operations list based on input args
def get_reduction_args(args):
    ops = []
    if args.all or args.sum:
        ops.append("sum")
    if args.all or args.mean:
        ops.append("mean")
    if args.all or args.var:
        ops.append("var")
    if args.all or args.std:
        ops.append("std")
    return ops

# First define an empty list that will be populated before the decorator is used
reduction_configs = []
def generate_reduction_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['tot_elements'],
                    x_vals=[2**i for i in range(12, 24, 1)],
                    line_arg='provider',
                    line_vals=['torch', 'triton'],
                    line_names=['PyTorch', 'Triton'],
                    styles=[('blue', '-'), ('red', '-')],
                    ylabel='GB/s',
                    xlabel="Total elements per output tensor",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode,},
                ))
    return configs

@triton.testing.perf_report(reduction_configs)
def benchmark_reduction(tot_elements, provider, op, mode, device=DEVICE):
    """
    Benchmark Triton reduction operations against PyTorch.
    
    Args:
        tot_elements: Total number of elements in the tensors
        provider: 'torch' or 'triton'
        op: "sum", "mean", "max", etc; designates the operation to be performed
        mode: "fwd" or "bwd"
        device: Device to run on
    """
    # Generate input data
    dim = int(tot_elements ** 0.5)
    X = torch.randn((dim, dim), device=device, requires_grad=True)
    
    # Select implementation
    if op == "sum":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.sum(X, dim=1)
    elif op == "mean":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.mean(X, dim=1)
    elif op == "var":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.var(X, dim=1)
    elif op == "std":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.std(X, dim=1)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    # for entry-wise operations we'll measure memory throughput since that's the limiting factor
    if mode == "fwd": # all fwd passes have same mem read/write behavior
        gb = 2 * tot_elements * 4 * 1e-9
        # 2 = number of memory operations (1 read + 1 write)
        # 4) = bytes per element (for float32)
        # 1e-9 converts bytes to GB
    elif mode == "bwd" and op in ("sum", "mean"):
        gb = 2 * tot_elements * 4 * 1e-9
    elif mode == "bwd" and op in ("var", "std"):
        gb = 3 * tot_elements * 4 * 1e-9
        # TODO should these two use TFLOPs instead of GB/s? not worth putting in the effort to change tbh
    # 1e-3 converts milliseconds to seconds
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


########################################################################################
########################### Embedding Module ############################################
########################################################################################

class _embedding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, E): 
        """
        implementation of embedding module
        """
        (B, N), (V, D) = tokens.shape, E.shape

        # allocates output
        x = torch.empty((B, N, D), device=E.device, dtype=E.dtype)

        grid = lambda meta: (
            triton.cdiv(B*N, meta['BLOCK_SIZE_ROWS']), 
            triton.cdiv(D, meta['BLOCK_SIZE_COLS'])
            )
        modules.embedding_forward[grid](
            tokens,
            E,
            x,
            tokens.stride(0), tokens.stride(1),
            E.stride(0), E.stride(1),
            x.stride(0), x.stride(1), x.stride(2),
            N, D, V,
            tokens.numel(), E.numel(), x.numel(),
        )

        ctx.save_for_backward(tokens)
        ctx.grid = grid
        ctx.B, ctx.N, ctx.D, ctx.V = B, N, D, V
        return x

    @staticmethod
    def backward(ctx, dLdx):
        tokens, = ctx.saved_tensors
        grid = ctx.grid
        B, N, D, V = ctx.B, ctx.N, ctx.D, ctx.V
        dLdE = torch.empty((V, D), device=dLdx.device, dtype=dLdx.dtype)
        
        modules.embedding_backward[grid](
            tokens,
            dLdE,
            dLdx,
            tokens.stride(0), tokens.stride(1),
            dLdE.stride(0), dLdE.stride(1),
            dLdx.stride(0), dLdx.stride(1), dLdx.stride(2),
            N, D, V,
            tokens.numel(), dLdE.numel(), dLdx.numel(),
        )

        return None, dLdE

embedding_fn = _embedding.apply

# Define the operations list based on input args
def get_embedding_args(args):
    ops = []
    if args.all or args.emb:
        ops.append("emb")
    return ops
# TODO i wrote it this way assuming i'd change embedding fn to all module fn's later but idk

# First define an empty list that will be populated before the decorator is used
embedding_configs = []
def generate_embedding_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['vocab_size'], # TODO can i scale vocab size & embed dim separately?
                    x_vals=[2**i for i in range(10, 20, 1)],
                    line_arg='provider',
                    line_vals=['torch', 'triton'],
                    line_names=['PyTorch', 'Triton'],
                    styles=[('blue', '-'), ('red', '-')],
                    ylabel='GB/s',
                    xlabel="Vocabulary Size (embed dim is 768)",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode,}, # TODO maybe use this for diff embed dims?
                ))
    return configs

@triton.testing.perf_report(embedding_configs)
def benchmark_embedding(vocab_size, provider, op, mode, device=DEVICE):
    """
    Benchmark Triton embedding kernels against PyTorch.
    """
    # Generate input data
    B, N, D = 32, 2048, 768
    tokens = torch.randint(0, vocab_size, size=(B, N), device=device)
    E = torch.randn((vocab_size, D), device=device, requires_grad=True)
    
    # Select implementation
    if provider == 'triton':
        fn = lambda: embedding_fn(tokens, E) 
    else: 
        fn = lambda: torch.nn.functional.embedding(tokens, E)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    
    # Benchmark
    tokens_gb = 1 * B*N * 4 * 1e-9
        # 1 -> we read tokens from mem once
        # 4 -> bytes per element 
            # even though we use tokens as tl.int32 in reality triton loads it as tl.float32
        # 1e-9 converts bytes to GB
    E_gb = 1 * B*N*D * 4 * 1e-9
        # 1 -> read (fwd) or write (bwd) to/from E once
    x_gb = 1 * B*N*D * 4 * 1e-9
        # 1 -> ead (fwd) or write (bwd) to/from
    gb = tokens_gb + E_gb + x_gb
    # 1e-3 converts milliseconds to seconds
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmarks for Triton operations')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--exp', action='store_true', help='Run exponentiation benchmarks')
    parser.add_argument('--log', action='store_true', help='Run natural logarithm benchmarks')
    parser.add_argument('--relu', action='store_true', help='Run rectified linear unit benchmarks')
    parser.add_argument('--add', action='store_true', help='Run addition benchmarks')
    parser.add_argument('--sub', action='store_true', help='Run subtraction benchmarks')
    parser.add_argument('--mul', action='store_true', help='Run multiplication benchmarks')
    parser.add_argument('--div', action='store_true', help='Run division benchmarks')
    parser.add_argument('--matmul', action='store_true', help='Run matrix multiplication benchmarks')
    parser.add_argument('--sum', action='store_true', help='Run summation benchmarks')
    parser.add_argument('--mean', action='store_true', help='Run mean benchmarks')
    parser.add_argument('--var', action='store_true', help='Run variance benchmarks')
    parser.add_argument('--std', action='store_true', help='Run standard deviation benchmarks')
    parser.add_argument('--emb', action='store_true', help='Run embedding module benchmarks')
    
    args = parser.parse_args()
    
    # If no args are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    print(f"ATTENTION:\nBENCHMARK tot_elementsS ARE DESIGNED TO FUNCTION WITHIN A LIMIT OF 16GB of VRAM.\n"
            f"IF YOU HAVE LESS YOU WILL GET ERRORS.\nTO FIX, EDIT x_vals IN EACH BENCHMARK'S CONFIG.")

    # Generate configs based on selected operations
    unary_ops_args = get_unary_ops_args(args)
    if unary_ops_args:
        print("\nRunning unary operation benchmarks...")
        unary_op_configs.extend(generate_unary_op_configs(unary_ops_args))
        benchmark_unary.run(print_data=True, save_path='./benchmarks/')
    
    binary_ops_args = get_binary_ops_args(args)
    if binary_ops_args:
        print("\nRunning binary operation benchmarks...")
        binary_op_configs.extend(generate_binary_op_configs(binary_ops_args))
        benchmark_binary.run(print_data=True, save_path='./benchmarks/')

    if args.all or args.matmul:
        print("\nRunning matmul benchmarks...")
        benchmark_matmul.run(print_data=True, save_path='./benchmarks/')
    
    reduction_args = get_reduction_args(args)
    if reduction_args:
        print("\nRunning reduction operation benchmarks...")
        reduction_configs.extend(generate_reduction_configs(reduction_args))
        benchmark_reduction.run(print_data=True, save_path='./benchmarks/')

    embedding_args = get_embedding_args(args)
    if embedding_args:
        print("\nRunning embedding module benchmarks...")
        embedding_configs.extend(generate_embedding_configs(embedding_args))
        benchmark_embedding.run(print_data=True, save_path='./benchmarks/')