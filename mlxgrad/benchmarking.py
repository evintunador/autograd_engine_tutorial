"""
Benchmarks mlxgrad's custom Metal kernels against PyTorch's own ops.

This mirrors ``cudagrad/benchmarking.py`` in spirit (the 'mlx' provider = our
Metal ``MLXTensor`` / ``nn`` ops; the 'torch' provider = PyTorch reference ops on
CPU) but with ONE deliberate difference: cudagrad borrows ``triton.testing`` as a
generic timing/plotting harness, and **triton doesn't run on a Mac**. So here we
ship a tiny self-contained timer (``do_bench``) plus a pandas/matplotlib report
writer instead — no triton import.

Because PyTorch has no Metal backend here, the 'torch' provider runs on CPU, so
these numbers compare "our Metal GPU kernels" against "PyTorch on CPU" — useful as
a sanity/relative signal, not a head-to-head same-device shootout. fp32
throughout; memory-bound ops report GB/s, compute-bound ops (matmul, attention)
report TFLOPS.

Run it directly (its own dir goes on sys.path, so the bare imports resolve):

    python mlxgrad/benchmarking.py --all

Per-category flags (e.g. ``--exp --matmul --flash``) select subsets. Outputs land
in ``mlxgrad/benchmarks/`` (one CSV + one PNG per plot).
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import mlx.core as mx

from engine import MLXTensor, DEVICE
import nn

BATCH = 8
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "benchmarks")
PROVIDERS = ["torch", "mlx"]


# ---------------------------------------------------------------------------
# tiny benchmarking harness (the triton.testing replacement)
# ---------------------------------------------------------------------------
def do_bench(fn, warmup=5, rep=20):
    """Median wall-clock ms of ``fn`` (which must force its own evaluation)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def run_plot(plot_name, xlabel, xs, ylabel, measure):
    """Sweep ``xs``, time every provider, write ``<plot_name>.csv`` + ``.png``."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for x in xs:
        row = {xlabel: x}
        for prov in PROVIDERS:
            row[prov] = measure(x, prov)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"{plot_name}.csv"), index=False)
    plt.figure()
    for prov in PROVIDERS:
        plt.plot(df[xlabel], df[prov], marker="o", label=prov)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, f"{plot_name}.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  wrote benchmarks/{plot_name}.csv + .png")


def _np(*shape):
    return np.random.randn(*shape).astype(np.float32)


# ---------------------------------------------------------------------------
# elementwise: binary (add/mul) + unary (exp/log/relu)   -> GB/s
# ---------------------------------------------------------------------------
ELEM_XS = [2 ** i for i in range(10, 17)]


def measure_binary(op, mode):
    def measure(tot, provider):
        dim = int(tot ** 0.5)
        a_np, b_np = _np(BATCH, dim, dim), _np(BATCH, dim, dim)
        f = (lambda x, y: x + y) if op == "add" else (lambda x, y: x * y)
        if provider == "torch":
            A = torch.tensor(a_np, requires_grad=True)
            B = torch.tensor(b_np, requires_grad=True)
            if mode == "fwd":
                fn = lambda: f(A, B)
            else:
                O = f(A, B); dO = torch.randn_like(O)
                fn = lambda: O.backward(dO, retain_graph=True)
        else:
            a, b = mx.array(a_np), mx.array(b_np)
            ones = mx.ones((BATCH, dim, dim))
            if mode == "fwd":
                def fn():
                    O = f(MLXTensor(a, requires_grad=True), MLXTensor(b, requires_grad=True))
                    mx.eval(O.data)
            else:
                def fn():
                    A = MLXTensor(a, requires_grad=True); B = MLXTensor(b, requires_grad=True)
                    f(A, B).backward(ones); mx.eval(A.grad, B.grad)
        ms = do_bench(fn)
        traffic = (3 if mode == "fwd" else (5 if op == "add" else 7))
        gb = BATCH * traffic * tot * 4 * 1e-9
        return gb / (ms * 1e-3)
    return measure


def measure_unary(op, mode):
    def measure(tot, provider):
        dim = int(tot ** 0.5)
        base = _np(BATCH, dim, dim)
        if provider == "torch":
            A = torch.tensor(base, requires_grad=True)
            tf = {"exp": torch.exp, "log": lambda x: torch.log(x.abs() + 1e-4), "relu": torch.relu}[op]
            if mode == "fwd":
                fn = lambda: tf(A)
            else:
                O = tf(A); dO = torch.randn_like(O)
                fn = lambda: O.backward(dO, retain_graph=True)
        else:
            a = mx.array(base)
            ones = mx.ones((BATCH, dim, dim))
            def mf(x):
                if op == "exp": return x.exp()
                if op == "log": return (x * x + 1e-4).log()
                return x.relu()
            if mode == "fwd":
                def fn():
                    mx.eval(mf(MLXTensor(a, requires_grad=True)).data)
            else:
                def fn():
                    A = MLXTensor(a, requires_grad=True); mf(A).backward(ones); mx.eval(A.grad)
        ms = do_bench(fn)
        gb = BATCH * (2 if mode == "fwd" else 3) * tot * 4 * 1e-9
        return gb / (ms * 1e-3)
    return measure


# ---------------------------------------------------------------------------
# reductions (sum/mean) + softmax   -> GB/s
# ---------------------------------------------------------------------------
def measure_reduction(op, mode):
    def measure(tot, provider):
        dim = int(tot ** 0.5)
        base = _np(dim, dim)
        if provider == "torch":
            X = torch.tensor(base, requires_grad=True)
            tf = (lambda t: t.sum(-1)) if op == "sum" else (lambda t: t.mean(-1))
            if mode == "fwd":
                fn = lambda: tf(X)
            else:
                O = tf(X); dO = torch.randn_like(O)
                fn = lambda: O.backward(dO, retain_graph=True)
        else:
            x = mx.array(base); ones = mx.ones((dim,))
            mf = (lambda t: t.sum()) if op == "sum" else (lambda t: t.mean())
            if mode == "fwd":
                def fn():
                    mx.eval(mf(MLXTensor(x, requires_grad=True)).data)
            else:
                def fn():
                    X = MLXTensor(x, requires_grad=True); mf(X).backward(ones); mx.eval(X.grad)
        ms = do_bench(fn)
        gb = 2 * tot * 4 * 1e-9
        return gb / (ms * 1e-3)
    return measure


def measure_softmax(mode):
    def measure(tot, provider):
        dim = int(tot ** 0.5)
        base = _np(dim, dim)
        if provider == "torch":
            X = torch.tensor(base, requires_grad=True)
            if mode == "fwd":
                fn = lambda: torch.softmax(X, dim=-1)
            else:
                O = torch.softmax(X, dim=-1); dO = torch.randn_like(O)
                fn = lambda: O.backward(dO, retain_graph=True)
        else:
            x = mx.array(base); ones = mx.ones((dim, dim))
            if mode == "fwd":
                def fn():
                    mx.eval(MLXTensor(x, requires_grad=True).softmax().data)
            else:
                def fn():
                    X = MLXTensor(x, requires_grad=True); X.softmax().backward(ones); mx.eval(X.grad)
        ms = do_bench(fn)
        gb = (2 if mode == "fwd" else 3) * tot * 4 * 1e-9
        return gb / (ms * 1e-3)
    return measure


# ---------------------------------------------------------------------------
# matmul   -> TFLOPS
# ---------------------------------------------------------------------------
MATMUL_XS = [128 * i for i in range(1, 6)]


def measure_matmul(mode):
    def measure(MNK, provider):
        M = N = K = MNK
        a_np, b_np = _np(BATCH, M, K), _np(BATCH, K, N)
        if provider == "torch":
            A = torch.tensor(a_np, requires_grad=True); B = torch.tensor(b_np, requires_grad=True)
            if mode == "fwd":
                fn = lambda: A @ B
            else:
                O = A @ B; dO = torch.randn_like(O)
                fn = lambda: O.backward(dO, retain_graph=True)
        else:
            a, b = mx.array(a_np), mx.array(b_np)
            ones = mx.ones((BATCH, M, N))
            if mode == "fwd":
                def fn():
                    mx.eval((MLXTensor(a, requires_grad=True) @ MLXTensor(b, requires_grad=True)).data)
            else:
                def fn():
                    A = MLXTensor(a, requires_grad=True); B = MLXTensor(b, requires_grad=True)
                    (A @ B).backward(ones); mx.eval(A.grad, B.grad)
        ms = do_bench(fn)
        return (2 if mode == "fwd" else 4) * BATCH * M * N * K * 1e-12 / (ms * 1e-3)
    return measure


# ---------------------------------------------------------------------------
# layernorm module   -> GB/s
# ---------------------------------------------------------------------------
LN_XS = [128 * i for i in range(1, 6)]


def measure_layernorm(mode):
    def measure(D, provider):
        B, N = 8, 256
        base = (_np(B, N, D) * 0.02)
        if provider == "torch":
            X = torch.tensor(base, requires_grad=True)
            ln = torch.nn.LayerNorm(D)
            if mode == "fwd":
                fn = lambda: ln(X)
            else:
                O = ln(X); dO = torch.randn_like(O)
                fn = lambda: O.backward(dO, retain_graph=True)
        else:
            x = mx.array(base); ones = mx.ones((B, N, D))
            ln = nn.LayerNorm(D)
            if mode == "fwd":
                def fn():
                    mx.eval(ln(MLXTensor(x, requires_grad=True)).data)
            else:
                def fn():
                    X = MLXTensor(x, requires_grad=True); ln(X).backward(ones); mx.eval(X.grad)
        ms = do_bench(fn)
        gb = (2 if mode == "fwd" else 3) * B * N * D * 4 * 1e-9
        return gb / (ms * 1e-3)
    return measure


# ---------------------------------------------------------------------------
# flash attention module   -> TFLOPS (causal => ~half the dense flops)
# ---------------------------------------------------------------------------
FLASH_XS = [64 * i for i in range(1, 5)]


def measure_flash(mode):
    def measure(N, provider):
        B, H, D = 4, 4, 32
        scale = D ** -0.5
        q_np, k_np, v_np = _np(B, H, N, D), _np(B, H, N, D), _np(B, H, N, D)
        if provider == "torch":
            Q = torch.tensor(q_np, requires_grad=True)
            K = torch.tensor(k_np, requires_grad=True)
            V = torch.tensor(v_np, requires_grad=True)
            def sdpa():
                return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)
            if mode == "fwd":
                fn = sdpa
            else:
                O = sdpa(); dO = torch.randn_like(O)
                fn = lambda: O.backward(dO, retain_graph=True)
        else:
            q, k, v = mx.array(q_np), mx.array(k_np), mx.array(v_np)
            ones = mx.ones((B, H, N, D))
            attn = nn.FlashAttention()
            if mode == "fwd":
                def fn():
                    Q = MLXTensor(q, requires_grad=True); K = MLXTensor(k, requires_grad=True)
                    Vv = MLXTensor(v, requires_grad=True)
                    mx.eval(attn(Q, K, Vv, scale=scale).data)
            else:
                def fn():
                    Q = MLXTensor(q, requires_grad=True); K = MLXTensor(k, requires_grad=True)
                    Vv = MLXTensor(v, requires_grad=True)
                    attn(Q, K, Vv, scale=scale).backward(ones); mx.eval(Q.grad, K.grad, Vv.grad)
        ms = do_bench(fn)
        # causal: ~ 2 matmuls * 0.5 (triangular) * 2 flops, fwd; ~2.5x for bwd
        flops = (2 if mode == "fwd" else 5) * B * H * N * N * D * 1e-12
        return flops / (ms * 1e-3)
    return measure


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="benchmark mlxgrad Metal kernels vs PyTorch (CPU)")
    p.add_argument("--all", action="store_true")
    for flag in ["add", "mul", "exp", "log", "relu", "sum", "mean",
                 "softmax", "matmul", "ln", "flash"]:
        p.add_argument(f"--{flag}", action="store_true")
    args = p.parse_args()
    A = args.all

    print(f"benchmarking on {DEVICE} (torch provider runs on CPU); writing to benchmarks/")
    for mode in ["fwd", "bwd"]:
        if A or args.add:
            run_plot(f"add_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_binary("add", mode))
        if A or args.mul:
            run_plot(f"mul_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_binary("mul", mode))
        if A or args.exp:
            run_plot(f"exp_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_unary("exp", mode))
        if A or args.log:
            run_plot(f"log_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_unary("log", mode))
        if A or args.relu:
            run_plot(f"relu_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_unary("relu", mode))
        if A or args.sum:
            run_plot(f"sum_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_reduction("sum", mode))
        if A or args.mean:
            run_plot(f"mean_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_reduction("mean", mode))
        if A or args.softmax:
            run_plot(f"softmax_{mode}", "tot_elements", ELEM_XS, "GB/s", measure_softmax(mode))
        if A or args.matmul:
            run_plot(f"matmul_{mode}", "MNK", MATMUL_XS, "TFLOPS", measure_matmul(mode))
        if A or args.ln:
            run_plot(f"layernorm_{mode}", "D", LN_XS, "GB/s", measure_layernorm(mode))
        if A or args.flash:
            run_plot(f"flash_{mode}", "N", FLASH_XS, "TFLOPS", measure_flash(mode))

    if not (A or any(getattr(args, f) for f in
                     ["add", "mul", "exp", "log", "relu", "sum", "mean",
                      "softmax", "matmul", "ln", "flash"])):
        p.print_help()


if __name__ == "__main__":
    main()
