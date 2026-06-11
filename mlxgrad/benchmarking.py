"""
Benchmarks mlxgrad's custom Metal kernels against the optimized alternatives on
Apple Silicon, on the SAME GPU. Three providers per plot:

  * mlxgrad : our hand-written Metal kernels (MLXTensor / nn) — the tutorial code
  * mlx     : MLX's own optimized ops (mx.matmul, mx.softmax, mx.fast.*) — the
              platform-native reference; the truest same-framework comparison
  * mps     : PyTorch on its Metal Performance Shaders GPU backend (falls back to
              CPU if MPS is unavailable or the op is unsupported)

On CUDA, PyTorch *is* the optimized reference (cuBLAS/cuDNN), which is why the
tritongrad/cudagrad tiers benchmark against it. On Apple Silicon the platform's
optimized library is MLX itself, so `mlx` is the primary baseline and `mps` is the
familiar "vs PyTorch" secondary. The point is to show honestly how much our
deliberately-naive educational kernels trade away vs optimized libraries.

This replaces cudagrad's triton.testing harness (triton doesn't run on a Mac) with
a tiny self-contained timer + pandas/matplotlib writer. Each provider uses its OWN
native autograd for the backward timing (our engine's `.backward`; MLX's `mx.vjp`;
PyTorch's `.backward`). fp32 throughout; memory-bound ops report GB/s,
compute-bound ops (matmul, attention) report TFLOPS. Missing/unsupported
provider+op combos show up as gaps (NaN) rather than crashing.

Run it directly (its own dir goes on sys.path, so bare imports resolve):
    python mlxgrad/benchmarking.py --all
Per-category flags (e.g. --matmul --flash) select subsets. Outputs land in
mlxgrad/benchmarks/ (one CSV + one PNG per plot).
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import torch
import torch.nn.functional as F

from engine import MLXTensor, DEVICE
import nn

BATCH = 8
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "benchmarks")

MPS = torch.backends.mps.is_available()
TORCH_DEV = "mps" if MPS else "cpu"
PROVIDERS = ["mlxgrad", "mlx", "mps"]


# ---------------------------------------------------------------------------
# tiny benchmarking harness (the triton.testing replacement)
# ---------------------------------------------------------------------------
def _nosync():
    pass


def _torch_sync():
    if MPS:
        torch.mps.synchronize()


def do_bench(fn, sync=_nosync, warmup=5, rep=20):
    """Median wall-clock ms of ``fn``. ``sync`` forces the work to finish before
    the clock stops (MLX evals inside ``fn`` so it uses ``_nosync``; the Torch-MPS
    backend dispatches asynchronously so it needs ``torch.mps.synchronize``)."""
    for _ in range(warmup):
        fn(); sync()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn(); sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def time_provider(provider, mode, nps, ours, rawmlx, torch_fn):
    """Median ms for one provider+mode, or NaN if that combo isn't supported.

    ``ours``/``rawmlx``/``torch_fn`` each take a LIST of tensors (in that
    provider's native type) and return the forward output. Backward uses each
    framework's own autograd."""
    try:
        if provider == "mlxgrad":
            arrs = [mx.array(a) for a in nps]
            if mode == "fwd":
                return do_bench(lambda: mx.eval(ours([MLXTensor(a) for a in arrs]).data))
            out0 = ours([MLXTensor(a) for a in arrs]); mx.eval(out0.data)
            cot = mx.ones(out0.shape)
            def fn():
                ts = [MLXTensor(a, requires_grad=True) for a in arrs]
                ours(ts).backward(cot)
                mx.eval(*[t.grad for t in ts])
            return do_bench(fn)

        if provider == "mlx":
            arrs = [mx.array(a) for a in nps]
            if mode == "fwd":
                return do_bench(lambda: mx.eval(rawmlx(arrs)))
            out0 = rawmlx(arrs); mx.eval(out0)
            cot = mx.ones(out0.shape)
            fun = lambda *a: rawmlx(list(a))
            return do_bench(lambda: mx.eval(*mx.vjp(fun, arrs, [cot])[1]))

        # provider == "mps"
        arrs = [torch.tensor(a, device=TORCH_DEV) for a in nps]
        if mode == "fwd":
            return do_bench(lambda: torch_fn(arrs), _torch_sync)
        ins = [a.clone().requires_grad_(True) for a in arrs]
        out = torch_fn(ins); cot = torch.ones_like(out)
        return do_bench(lambda: out.backward(cot, retain_graph=True), _torch_sync)
    except Exception:
        return float("nan")


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


def plot_category(name, xlabel, xs, ylabel, builders, metric, modes=("fwd", "bwd")):
    """``builders(x) -> (nps, ours, rawmlx, torch_fn)``; ``metric(x, mode, ms) -> value``."""
    for mode in modes:
        def measure(x, prov, _mode=mode):
            nps, ours, rawmlx, torch_fn = builders(x)
            ms = time_provider(prov, _mode, nps, ours, rawmlx, torch_fn)
            return metric(x, _mode, ms)
        run_plot(f"{name}_{mode}", xlabel, xs, ylabel, measure)


def _np(*shape):
    return np.random.randn(*shape).astype(np.float32)


# ---------------------------------------------------------------------------
# category builders: each returns (input arrays, ours, raw-mlx, torch) impls
# ---------------------------------------------------------------------------
ELEM_XS = [2 ** i for i in range(10, 17)]
MATMUL_XS = [128 * i for i in range(1, 6)]
LN_XS = [128 * i for i in range(1, 6)]
FLASH_XS = [64 * i for i in range(1, 5)]


def b_binary(op):
    def builders(tot):
        dim = int(tot ** 0.5)
        nps = [_np(BATCH, dim, dim), _np(BATCH, dim, dim)]
        if op == "add":
            return nps, (lambda t: t[0] + t[1]), (lambda a: a[0] + a[1]), (lambda a: a[0] + a[1])
        return nps, (lambda t: t[0] * t[1]), (lambda a: a[0] * a[1]), (lambda a: a[0] * a[1])
    return builders


def m_binary(op):
    def metric(tot, mode, ms):
        traffic = 3 if mode == "fwd" else (5 if op == "add" else 7)
        return BATCH * traffic * tot * 4 * 1e-9 / (ms * 1e-3)
    return metric


def b_unary(op):
    def builders(tot):
        dim = int(tot ** 0.5)
        nps = [_np(BATCH, dim, dim)]
        if op == "exp":
            return nps, (lambda t: t[0].exp()), (lambda a: mx.exp(a[0])), (lambda a: torch.exp(a[0]))
        if op == "log":  # square+eps keeps log well-defined for all three
            return (nps, (lambda t: (t[0] * t[0] + 1e-4).log()),
                    (lambda a: mx.log(a[0] * a[0] + 1e-4)),
                    (lambda a: torch.log(a[0] * a[0] + 1e-4)))
        return nps, (lambda t: t[0].relu()), (lambda a: mx.maximum(a[0], 0.0)), (lambda a: torch.relu(a[0]))
    return builders


def m_unary(tot, mode, ms):
    return BATCH * (2 if mode == "fwd" else 3) * tot * 4 * 1e-9 / (ms * 1e-3)


def b_reduction(op):
    def builders(tot):
        dim = int(tot ** 0.5)
        nps = [_np(dim, dim)]
        if op == "sum":
            return nps, (lambda t: t[0].sum()), (lambda a: mx.sum(a[0], axis=-1)), (lambda a: a[0].sum(-1))
        return nps, (lambda t: t[0].mean()), (lambda a: mx.mean(a[0], axis=-1)), (lambda a: a[0].mean(-1))
    return builders


def m_reduction(tot, mode, ms):
    return 2 * tot * 4 * 1e-9 / (ms * 1e-3)


def b_softmax():
    def builders(tot):
        dim = int(tot ** 0.5)
        nps = [_np(dim, dim)]
        return (nps, (lambda t: t[0].softmax()), (lambda a: mx.softmax(a[0], axis=-1)),
                (lambda a: torch.softmax(a[0], dim=-1)))
    return builders


def m_softmax(tot, mode, ms):
    return (2 if mode == "fwd" else 3) * tot * 4 * 1e-9 / (ms * 1e-3)


def b_matmul():
    def builders(S):
        nps = [_np(BATCH, S, S), _np(BATCH, S, S)]
        return nps, (lambda t: t[0] @ t[1]), (lambda a: a[0] @ a[1]), (lambda a: a[0] @ a[1])
    return builders


def m_matmul(S, mode, ms):
    return (2 if mode == "fwd" else 4) * BATCH * S * S * S * 1e-12 / (ms * 1e-3)


def b_layernorm():
    def builders(D):
        B, N = 8, 256
        nps = [_np(B, N, D) * 0.02]
        w_np = np.ones(D, dtype=np.float32); b_np = np.zeros(D, dtype=np.float32)
        ln = nn.LayerNorm(D)
        ln.weight.data = mx.array(w_np); ln.bias.data = mx.array(b_np)
        wmx, bmx = mx.array(w_np), mx.array(b_np)
        wt = torch.tensor(w_np, device=TORCH_DEV); bt = torch.tensor(b_np, device=TORCH_DEV)
        return (nps,
                (lambda t: ln(t[0])),
                (lambda a: mx.fast.layer_norm(a[0], wmx, bmx, 1e-5)),
                (lambda a: F.layer_norm(a[0], (D,), wt, bt, 1e-5)))
    return builders


def m_layernorm(D, mode, ms):
    B, N = 8, 256
    return (2 if mode == "fwd" else 3) * B * N * D * 4 * 1e-9 / (ms * 1e-3)


def b_flash():
    B, H, D = 4, 4, 32
    scale = D ** -0.5
    attn = nn.FlashAttention()
    def builders(N):
        nps = [_np(B, H, N, D), _np(B, H, N, D), _np(B, H, N, D)]
        return (nps,
                (lambda t: attn(t[0], t[1], t[2], scale=scale)),
                (lambda a: mx.fast.scaled_dot_product_attention(a[0], a[1], a[2], scale=scale, mask="causal")),
                (lambda a: F.scaled_dot_product_attention(a[0], a[1], a[2], is_causal=True, scale=scale)))
    return builders


def m_flash(N, mode, ms):
    B, H, D = 4, 4, 32
    return (2 if mode == "fwd" else 5) * B * H * N * N * D * 1e-12 / (ms * 1e-3)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="benchmark mlxgrad Metal kernels vs raw MLX + Torch MPS")
    p.add_argument("--all", action="store_true")
    flags = ["add", "mul", "exp", "log", "relu", "sum", "mean",
             "softmax", "matmul", "ln", "flash"]
    for flag in flags:
        p.add_argument(f"--{flag}", action="store_true")
    args = p.parse_args()
    A = args.all

    print(f"providers: mlxgrad (ours, {DEVICE}) | mlx (raw) | mps "
          f"({'MPS' if MPS else 'CPU fallback'}); writing to benchmarks/")

    if A or args.add:
        plot_category("add", "tot_elements", ELEM_XS, "GB/s", b_binary("add"), m_binary("add"))
    if A or args.mul:
        plot_category("mul", "tot_elements", ELEM_XS, "GB/s", b_binary("mul"), m_binary("mul"))
    if A or args.exp:
        plot_category("exp", "tot_elements", ELEM_XS, "GB/s", b_unary("exp"), m_unary)
    if A or args.log:
        plot_category("log", "tot_elements", ELEM_XS, "GB/s", b_unary("log"), m_unary)
    if A or args.relu:
        plot_category("relu", "tot_elements", ELEM_XS, "GB/s", b_unary("relu"), m_unary)
    if A or args.sum:
        plot_category("sum", "tot_elements", ELEM_XS, "GB/s", b_reduction("sum"), m_reduction)
    if A or args.mean:
        plot_category("mean", "tot_elements", ELEM_XS, "GB/s", b_reduction("mean"), m_reduction)
    if A or args.softmax:
        plot_category("softmax", "tot_elements", ELEM_XS, "GB/s", b_softmax(), m_softmax)
    if A or args.matmul:
        plot_category("matmul", "MNK", MATMUL_XS, "TFLOPS", b_matmul(), m_matmul)
    if A or args.ln:
        plot_category("layernorm", "D", LN_XS, "GB/s", b_layernorm(), m_layernorm)
    if A or args.flash:
        plot_category("flash", "N", FLASH_XS, "TFLOPS", b_flash(), m_flash)

    if not (A or any(getattr(args, f) for f in flags)):
        p.print_help()


if __name__ == "__main__":
    main()
