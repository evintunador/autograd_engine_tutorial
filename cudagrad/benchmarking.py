"""
Benchmarks cudagrad's custom CUDA kernels against PyTorch's own ops.

This MIRRORS ``tritongrad/benchmarking.py`` one abstraction level lower: instead
of comparing Triton kernels vs PyTorch, we compare our raw-CUDA ``CudaTensor`` /
``nn`` ops (the 'cuda' provider) vs PyTorch reference ops (the 'torch' provider).

We reuse ``triton.testing`` purely as a *backend-agnostic* benchmarking harness —
``triton.testing.do_bench`` is just a timer for any Python callable, and
``triton.testing.perf_report`` + ``triton.testing.Benchmark`` produce the CSV/PNG
reports. Importing triton here does NOT make cudagrad a Triton backend; it only
borrows triton's timing/plotting utilities (triton is already installed).

Run from inside the ``cudagrad/`` directory (primary path, matches tritongrad):

    cd cudagrad && python benchmarking.py --all

or as a module from the repo root:

    python -m cudagrad.benchmarking --all

Per-category flags (e.g. ``--exp --matmul --flash``) select subsets. Outputs land
in ``cudagrad/benchmarks/`` (one CSV + one PNG per plot).

NOTE: like the rest of cudagrad, this only runs on a CUDA box — ``engine``/``nn``
touch ``torch.cuda.current_device()`` at import time.

fp32 throughout. Memory-bound ops report GB/s; compute-bound ops (matmul,
attention) report TFLOPS.
"""
from math import sqrt

import torch
import triton  # used ONLY as a generic benchmarking/timing harness (see module docstring)

# Imports mirror tritongrad/benchmarking.py's `from kernels import ...`: when run
# from inside cudagrad/ these resolve to the local modules; `python -m cudagrad.benchmarking`
# also works because the package dir is on sys.path.
from engine import CudaTensor, DEVICE
import nn


BATCH = 32


########################################################################################
########################### Unary Ops ##################################################
########################################################################################

def get_unary_ops_args(args):
    ops = []
    if args.all or args.exp:
        ops.append("exp")
    if args.all or args.log:
        ops.append("log")
    if args.all or args.relu:
        ops.append("relu")
    return ops

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
                    line_vals=['torch', 'cuda'],
                    line_names=['PyTorch', 'CUDA'],
                    styles=[('blue', '-'), ('green', '-')],
                    ylabel='GB/s',
                    xlabel="Total elements per output tensor",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode},
                ))
    return configs

@triton.testing.perf_report(unary_op_configs)
def benchmark_unary(tot_elements, provider, op, mode, device=DEVICE):
    """Benchmark cudagrad unary ops (exp/log/relu) vs PyTorch."""
    dim = int(tot_elements ** 0.5)
    # log needs positive inputs; use exp(randn) so all three ops stay well-defined
    base = torch.randn((BATCH, dim, dim), device=device)
    A_torch = base.clone().requires_grad_(True)

    def torch_apply(x):
        if op == "exp":  return torch.exp(x)
        if op == "log":  return torch.log(x.abs() + 1e-4)
        return torch.relu(x)

    def cuda_apply(x):
        if op == "exp":  return x.exp()
        if op == "log":  return (x * x + 1e-4).log()  # |x|^2+eps keeps log well-defined
        return x.relu()

    if provider == 'torch':
        if mode == "fwd":
            fn = lambda: torch_apply(A_torch)
        else:
            O = torch_apply(A_torch)
            dO = torch.randn_like(O)
            fn = lambda: O.backward(dO, retain_graph=True)
    else:  # cuda
        if mode == "fwd":
            fn = lambda: cuda_apply(CudaTensor(base, requires_grad=True))
        else:
            # rebuild the graph each call so .grad accumulation stays clean across reruns
            def fn():
                A = CudaTensor(base, requires_grad=True)
                O = cuda_apply(A)
                O.backward(torch.ones_like(O.data))

    if mode == "fwd":
        gb = BATCH * 2 * tot_elements * 4 * 1e-9   # 1 read + 1 write
    else:
        gb = BATCH * 3 * tot_elements * 4 * 1e-9   # read in, grad in, grad out
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


########################################################################################
########################### Binary Ops #################################################
########################################################################################

def get_binary_ops_args(args):
    ops = []
    if args.all or args.add:
        ops.append("add")
    if args.all or args.mul:
        ops.append("mul")
    return ops

binary_op_configs = []
def generate_binary_op_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['tot_elements'],
                    x_vals=[2**i for i in range(12, 24, 1)],
                    line_arg='provider',
                    line_vals=['torch', 'cuda'],
                    line_names=['PyTorch', 'CUDA'],
                    styles=[('blue', '-'), ('green', '-')],
                    ylabel='GB/s',
                    xlabel="Total elements per output tensor",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode},
                ))
    return configs

@triton.testing.perf_report(binary_op_configs)
def benchmark_binary(tot_elements, provider, op, mode, device=DEVICE):
    """Benchmark cudagrad binary ops (add/mul) vs PyTorch (same-shape inputs)."""
    dim = int(tot_elements ** 0.5)
    a = torch.randn((BATCH, dim, dim), device=device)
    b = torch.randn((BATCH, dim, dim), device=device)

    def torch_apply(x, y):
        return x + y if op == "add" else x * y

    def cuda_apply(x, y):
        return x + y if op == "add" else x * y

    if provider == 'torch':
        A = a.clone().requires_grad_(True)
        B = b.clone().requires_grad_(True)
        if mode == "fwd":
            fn = lambda: torch_apply(A, B)
        else:
            O = torch_apply(A, B)
            dO = torch.randn_like(O)
            fn = lambda: O.backward(dO, retain_graph=True)
    else:  # cuda
        if mode == "fwd":
            fn = lambda: cuda_apply(CudaTensor(a, requires_grad=True),
                                    CudaTensor(b, requires_grad=True))
        else:
            def fn():
                A = CudaTensor(a, requires_grad=True)
                B = CudaTensor(b, requires_grad=True)
                O = cuda_apply(A, B)
                O.backward(torch.ones_like(O.data))

    if mode == "fwd":
        gb = BATCH * 3 * tot_elements * 4 * 1e-9   # 2 reads + 1 write
    elif op == "add":
        gb = BATCH * 5 * tot_elements * 4 * 1e-9
    else:  # mul
        gb = BATCH * 7 * tot_elements * 4 * 1e-9
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


########################################################################################
########################### Matrix Multiplication ######################################
########################################################################################

matmul_configs = []
for _mode in ["fwd", "bwd"]:
    matmul_configs.append(
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=[128 * i for i in range(2, 28, 1)],
            line_arg='provider',
            line_vals=['torch', 'cuda'],
            line_names=['PyTorch', 'CUDA'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='TFLOPS',
            xlabel="M, N and K",
            plot_name=f'matmul_{_mode}',
            args={"mode": _mode},
        ))

@triton.testing.perf_report(matmul_configs)
def benchmark_matmul(M, N, K, provider, mode, device=DEVICE):
    """Benchmark cudagrad batched matmul vs PyTorch."""
    a = torch.randn((BATCH, M, K), device=device)
    b = torch.randn((BATCH, K, N), device=device)

    if provider == 'torch':
        A = a.clone().requires_grad_(True)
        B = b.clone().requires_grad_(True)
        if mode == "fwd":
            fn = lambda: A @ B
        else:
            O = A @ B
            dO = torch.randn_like(O)
            fn = lambda: O.backward(dO, retain_graph=True)
    else:  # cuda
        if mode == "fwd":
            fn = lambda: CudaTensor(a, requires_grad=True) @ CudaTensor(b, requires_grad=True)
        else:
            def fn():
                A = CudaTensor(a, requires_grad=True)
                B = CudaTensor(b, requires_grad=True)
                O = A @ B
                O.backward(torch.ones_like(O.data))

    ms = triton.testing.do_bench(fn)
    # 2 flops/entry (mul+add) for fwd; ~2x that for the two backward gradients
    perf = (2 if mode == "fwd" else 4) * BATCH * M * N * K * 1e-12 / (ms * 1e-3)
    return perf


########################################################################################
########################### Reduction Ops ##############################################
########################################################################################

def get_reduction_args(args):
    ops = []
    if args.all or args.sum:
        ops.append("sum")
    if args.all or args.mean:
        ops.append("mean")
    return ops

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
                    line_vals=['torch', 'cuda'],
                    line_names=['PyTorch', 'CUDA'],
                    styles=[('blue', '-'), ('green', '-')],
                    ylabel='GB/s',
                    xlabel="Total elements per output tensor",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode},
                ))
    return configs

@triton.testing.perf_report(reduction_configs)
def benchmark_reduction(tot_elements, provider, op, mode, device=DEVICE):
    """Benchmark cudagrad last-dim reductions (sum/mean) vs PyTorch."""
    dim = int(tot_elements ** 0.5)
    x = torch.randn((dim, dim), device=device)

    def torch_apply(t):
        return torch.sum(t, dim=-1) if op == "sum" else torch.mean(t, dim=-1)

    def cuda_apply(t):
        return t.sum() if op == "sum" else t.mean()  # cudagrad reduces over the final dim

    if provider == 'torch':
        X = x.clone().requires_grad_(True)
        if mode == "fwd":
            fn = lambda: torch_apply(X)
        else:
            O = torch_apply(X)
            dO = torch.randn_like(O)
            fn = lambda: O.backward(dO, retain_graph=True)
    else:  # cuda
        if mode == "fwd":
            fn = lambda: cuda_apply(CudaTensor(x, requires_grad=True))
        else:
            def fn():
                X = CudaTensor(x, requires_grad=True)
                O = cuda_apply(X)
                O.backward(torch.ones_like(O.data))

    gb = 2 * tot_elements * 4 * 1e-9   # 1 read + 1 write dominate for sum/mean
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


########################################################################################
########################### Softmax ####################################################
########################################################################################

softmax_configs = []
for _mode in ["fwd", "bwd"]:
    softmax_configs.append(
        triton.testing.Benchmark(
            x_names=['tot_elements'],
            x_vals=[2**i for i in range(12, 24, 1)],
            line_arg='provider',
            line_vals=['torch', 'cuda'],
            line_names=['PyTorch', 'CUDA'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            xlabel="Total elements per output tensor",
            plot_name=f'softmax_{_mode}',
            args={"mode": _mode},
        ))

@triton.testing.perf_report(softmax_configs)
def benchmark_softmax(tot_elements, provider, mode, device=DEVICE):
    """Benchmark cudagrad last-dim softmax vs PyTorch."""
    dim = int(tot_elements ** 0.5)
    x = torch.randn((dim, dim), device=device)

    if provider == 'torch':
        X = x.clone().requires_grad_(True)
        if mode == "fwd":
            fn = lambda: torch.softmax(X, dim=-1)
        else:
            O = torch.softmax(X, dim=-1)
            dO = torch.randn_like(O)
            fn = lambda: O.backward(dO, retain_graph=True)
    else:  # cuda
        if mode == "fwd":
            fn = lambda: CudaTensor(x, requires_grad=True).softmax()
        else:
            def fn():
                X = CudaTensor(x, requires_grad=True)
                O = X.softmax()
                O.backward(torch.ones_like(O.data))

    if mode == "fwd":
        gb = 2 * tot_elements * 4 * 1e-9   # 1 read + 1 write
    else:
        gb = 3 * tot_elements * 4 * 1e-9   # out, grad-in, grad-out
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


########################################################################################
########################### LayerNorm Module ###########################################
########################################################################################

def get_layernorm_args(args):
    ops = []
    if args.all or args.ln:
        ops.append("ln")
    return ops

layernorm_configs = []
def generate_layernorm_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['D'],
                    x_vals=[256 * i for i in range(1, 12, 1)],
                    line_arg='provider',
                    line_vals=['torch', 'cuda'],
                    line_names=['PyTorch', 'CUDA'],
                    styles=[('blue', '-'), ('green', '-')],
                    ylabel='GB/s',
                    xlabel="embedding dimension getting normalized",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode},
                ))
    return configs

@triton.testing.perf_report(layernorm_configs)
def benchmark_layernorm(D, provider, op, mode, device=DEVICE):
    """Benchmark cudagrad LayerNorm module vs PyTorch."""
    B, N = 32, 2048
    x = torch.randn((B, N, D), dtype=torch.float32, device=device) * 0.02
    weight = torch.randn((D,), dtype=torch.float32, device=device) * 0.02
    bias = torch.randn((D,), dtype=torch.float32, device=device) * 0.02

    if provider == 'torch':
        X = x.clone().requires_grad_(True)
        W = weight.clone().requires_grad_(True)
        Bi = bias.clone().requires_grad_(True)
        apply = lambda: torch.nn.functional.layer_norm(X, normalized_shape=(D,), weight=W, bias=Bi)
        if mode == "fwd":
            fn = apply
        else:
            O = apply()
            dO = torch.randn_like(O)
            fn = lambda: O.backward(dO, retain_graph=True)
    else:  # cuda
        # rebuild the module + graph each call so module-parameter grads stay clean
        def build():
            ln = nn.LayerNorm(D)
            ln.weight.data = weight.clone()
            ln.bias.data = bias.clone()
            return ln
        if mode == "fwd":
            ln = build()
            fn = lambda: ln(CudaTensor(x, requires_grad=True))
        else:
            def fn():
                ln = build()
                X = CudaTensor(x, requires_grad=True)
                O = ln(X)
                O.backward(torch.ones_like(O.data))

    if mode == "fwd":
        gb = ((2 * B*N*D) + (2 * B*N) + (2 * D)) * 4 * 1e-9
    else:
        gb = ((3 * B*N*D) + (2 * B*N) + (3 * D)) * 4 * 1e-9
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)


########################################################################################
########################### Flash Attention Module #####################################
########################################################################################

def get_flashattention_args(args):
    ops = []
    if args.all or args.flash:
        ops.append("flash")
    return ops

flashattention_configs = []
def generate_flashattention_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['N'],
                    x_vals=[512 * i for i in range(1, 17, 1)],
                    line_arg='provider',
                    line_vals=['torch', 'cuda'],
                    line_names=['PyTorch', 'CUDA'],
                    styles=[('blue', '-'), ('green', '-')],
                    ylabel='TFLOPs/s',
                    xlabel="sequence length (N)",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode},
                ))
    return configs

@triton.testing.perf_report(flashattention_configs)
def benchmark_flashattention(N, provider, op, mode, device=DEVICE):
    """Benchmark cudagrad FlashAttention module vs PyTorch SDPA."""
    B, H, Dh = 32, 4, 128
    scale = 1.0 / sqrt(Dh)
    q = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device) * 0.02
    k = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device) * 0.02
    v = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device) * 0.02

    if provider == 'torch':
        Q = q.clone().requires_grad_(True)
        K = k.clone().requires_grad_(True)
        V = v.clone().requires_grad_(True)
        apply = lambda: torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=True, scale=scale)
        if mode == "fwd":
            fn = apply
        else:
            O = apply()
            dO = torch.randn_like(O)
            fn = lambda: O.backward(dO, retain_graph=True)
    else:  # cuda
        flash = nn.FlashAttention()
        if mode == "fwd":
            fn = lambda: flash(CudaTensor(q, requires_grad=True),
                               CudaTensor(k, requires_grad=True),
                               CudaTensor(v, requires_grad=True), scale)
        else:
            def fn():
                Q = CudaTensor(q, requires_grad=True)
                K = CudaTensor(k, requires_grad=True)
                V = CudaTensor(v, requires_grad=True)
                O = flash(Q, K, V, scale)
                O.backward(torch.ones_like(O.data))

    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * B * H * N * N * Dh
    total_flops = 2 * flops_per_matmul * 0.5   # 0.5 for causal
    if mode == "bwd":
        total_flops *= 2.5   # 2.0 (bwd) + 0.5 (recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmarks for cudagrad CUDA kernels vs PyTorch')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--exp', action='store_true', help='Run exponentiation benchmarks')
    parser.add_argument('--log', action='store_true', help='Run natural logarithm benchmarks')
    parser.add_argument('--relu', action='store_true', help='Run rectified linear unit benchmarks')
    parser.add_argument('--add', action='store_true', help='Run addition benchmarks')
    parser.add_argument('--mul', action='store_true', help='Run multiplication benchmarks')
    parser.add_argument('--matmul', action='store_true', help='Run matrix multiplication benchmarks')
    parser.add_argument('--sum', action='store_true', help='Run summation benchmarks')
    parser.add_argument('--mean', action='store_true', help='Run mean benchmarks')
    parser.add_argument('--softmax', action='store_true', help='Run softmax benchmarks')
    parser.add_argument('--ln', action='store_true', help='Run LayerNorm module benchmarks')
    parser.add_argument('--flash', action='store_true', help='Run Flash Attention module benchmarks')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    print("ATTENTION:\nBENCHMARK SIZES ARE DESIGNED TO FUNCTION WITHIN A LIMIT OF 16GB of VRAM.\n"
          "IF YOU HAVE LESS YOU WILL GET ERRORS.\nTO FIX, EDIT x_vals IN EACH BENCHMARK'S CONFIG.")

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

    if args.all or args.softmax:
        print("\nRunning softmax benchmarks...")
        benchmark_softmax.run(print_data=True, save_path='./benchmarks/')

    layernorm_args = get_layernorm_args(args)
    if layernorm_args:
        print("\nRunning LayerNorm module benchmarks...")
        layernorm_configs.extend(generate_layernorm_configs(layernorm_args))
        benchmark_layernorm.run(print_data=True, save_path='./benchmarks/')

    flashattention_args = get_flashattention_args(args)
    if flashattention_args:
        print("\nRunning Flash Attention module benchmarks...")
        flashattention_configs.extend(generate_flashattention_configs(flashattention_args))
        benchmark_flashattention.run(print_data=True, save_path='./benchmarks/')
