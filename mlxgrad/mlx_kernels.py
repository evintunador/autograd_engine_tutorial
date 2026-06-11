"""Python wrappers around mlxgrad's custom Metal kernels.

The real math lives as Metal Shading Language kernel *bodies* in the ``.metal``
files under ``kernels/``. Each body is fed to ``mx.fast.metal_kernel``, which
JIT-compiles it into a fast MLX op on first call (cached by MLX afterwards — the
first call to a given kernel is slightly slow, like Triton's first-call JIT).
This is the Apple-Metal analog of cudagrad's CUDA C++ kernels.

Why this module is named ``mlx_kernels`` and never ``kernels``: tritongrad's
``engine.py`` does ``from kernels import ...``, leaking a top-level ``kernels``
module into ``sys.modules`` that ``load_backend`` never cleans. Naming this
``mlx_kernels`` (and keeping ``kernels/`` a pure source directory with no
``__init__.py``, never imported as Python) sidesteps the collision — exactly the
trick cudagrad uses with ``cuda_kernels``. See the project plan.

KEY DIFFERENCE FROM cudagrad: MLX arrays are immutable, so backward wrappers
cannot accumulate in place. Instead they take the running gradient ``grad_in``
and RETURN ``grad_in + contribution`` (the add happens inside the Metal kernel);
the engine rebinds ``tensor.grad`` to the returned array. Groups not yet
implemented raise ``NotImplementedError`` so the engine methods import cleanly
while their ops stay out of the adapter's OPS/MODULES (and thus skip).
"""
import os
import re

import mlx.core as mx

_HERE = os.path.dirname(os.path.abspath(__file__))
_KDIR = os.path.join(_HERE, "kernels")

# --- .metal source loading + kernel building -------------------------------

_SRC_CACHE = {}   # group -> {kernel_name: body_source}
_KERNEL_CACHE = {}  # (group, name) -> compiled mx.fast.metal_kernel


def _sources(group):
    """Parse ``kernels/<group>.metal`` into {kernel_name: body}.

    Bodies are delimited by ``// @kernel <name>`` markers; text before the first
    marker (the file header) is ignored.
    """
    if group not in _SRC_CACHE:
        path = os.path.join(_KDIR, f"{group}.metal")
        with open(path) as f:
            text = f.read()
        sections, name, buf = {}, None, []
        for line in text.splitlines(keepends=True):
            m = re.match(r"\s*//\s*@kernel\s+(\S+)", line)
            if m:
                if name is not None:
                    sections[name] = "".join(buf)
                name, buf = m.group(1), []
            elif name is not None:
                buf.append(line)
        if name is not None:
            sections[name] = "".join(buf)
        _SRC_CACHE[group] = sections
    return _SRC_CACHE[group]


def _kernel(group, name, input_names, output_names):
    key = (group, name)
    if key not in _KERNEL_CACHE:
        src = _sources(group)[name]
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"{group}_{name}",
            input_names=input_names,
            output_names=output_names,
            source=src,
        )
    return _KERNEL_CACHE[key]


def _u32(v):
    """Pack a python int as a 1-element uint32 buffer (how scalars reach kernels)."""
    return mx.array([int(v)], dtype=mx.uint32)


def _f32(v):
    """Pack a python float as a 1-element float32 buffer."""
    return mx.array([float(v)], dtype=mx.float32)


def _launch(n):
    """(grid, threadgroup) for a 1-D launch of exactly ``n`` threads."""
    n = int(n)
    tg = min(256, n) if n > 0 else 1
    return (n, 1, 1), (tg, 1, 1)


# --- elementwise binary (add / sub / mul / div) ----------------------------
_BINARY_OP = {"add": 0, "sub": 1, "mul": 2, "div": 3}


def binary_forward(x, y, loop_stride, op):
    grid, tg = _launch(x.size)
    k = _kernel("elementwise", "binary_forward", ["x", "y", "ls", "op"], ["out"])
    (out,) = k(inputs=[x, y, _u32(loop_stride), _u32(_BINARY_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[x.shape], output_dtypes=[x.dtype])
    return out


def binary_backward_dx(grad_in, y, dout, loop_stride, op):
    grid, tg = _launch(grad_in.size)
    k = _kernel("elementwise", "binary_backward_dx",
                ["grad_in", "y", "dout", "ls", "op"], ["out"])
    (out,) = k(inputs=[grad_in, y, dout, _u32(loop_stride), _u32(_BINARY_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


def binary_backward_dy(grad_in, x, y, dout, loop_stride, op):
    grid, tg = _launch(loop_stride)  # one thread per second-operand element
    k = _kernel("elementwise", "binary_backward_dy",
                ["grad_in", "x", "y", "dout", "ls", "n", "op"], ["out"])
    (out,) = k(inputs=[grad_in, x, y, dout, _u32(loop_stride), _u32(x.size),
                       _u32(_BINARY_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


# --- elementwise unary (exp / log / relu / neg) ----------------------------
_UNARY_OP = {"exp": 0, "log": 1, "relu": 2, "neg": 3}


def unary_forward(x, op):
    grid, tg = _launch(x.size)
    k = _kernel("elementwise", "unary_forward", ["x", "op"], ["out"])
    (out,) = k(inputs=[x, _u32(_UNARY_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[x.shape], output_dtypes=[x.dtype])
    return out


def unary_backward(grad_in, x, out_fwd, dout, op):
    grid, tg = _launch(grad_in.size)
    k = _kernel("elementwise", "unary_backward",
                ["grad_in", "x", "out_fwd", "dout", "op"], ["out"])
    (out,) = k(inputs=[grad_in, x, out_fwd, dout, _u32(_UNARY_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


# --- not yet implemented (filled in by later kernel phases) ----------------
# Each raises so engine methods exist and import cleanly; the op simply stays out
# of the adapter's OPS/MODULES until its phase lands.

# --- matmul (forward / backward dA / backward dB) --------------------------
# Dims (Bsz, M, K, N, shared) are derived in Python from the array shapes (the
# CUDA tier does this in the launcher); `shared` <=> B is 2-D (broadcast across
# A's batch). One thread per output element; the grid carries Bsz.

def _mm_dims(a_like, b_like):
    """(Bsz, M, K, N, shared) for A-shaped `a_like` @ B-shaped `b_like`."""
    M, K, N = a_like.shape[-2], a_like.shape[-1], b_like.shape[-1]
    assert b_like.shape[-2] == K, \
        f"matmul inner dims must match: A K={K} vs B {b_like.shape[-2]}"
    Bsz = a_like.size // (M * K)
    shared = 1 if b_like.ndim < a_like.ndim else 0
    return Bsz, M, K, N, shared


# Tiled GEMM: TILE x TILE threadgroups, each computes a TILE x TILE output tile
# for one batch (grid.z = Bsz). Must equal TILE in kernels/matmul.metal.
_TILE = 16


def _mm_grid(out_cols, out_rows, batch):
    """(grid, threadgroup) for a tiled GEMM whose output is (batch, rows, cols).

    The grid is rounded UP to a multiple of _TILE in each output dim so every
    threadgroup is full (MLX dispatches exactly `grid` threads with non-uniform
    threadgroups); the kernel bounds-checks output coords and zero-pads the
    staged tiles for edge tiles.
    """
    def ceil_tile(x):
        return ((int(x) + _TILE - 1) // _TILE) * _TILE
    grid = (ceil_tile(out_cols), ceil_tile(out_rows), int(batch))
    tg = (_TILE, _TILE, 1)
    return grid, tg


def matmul_forward(a, b):
    Bsz, M, K, N, shared = _mm_dims(a, b)
    grid, tg = _mm_grid(N, M, Bsz)   # output (Bsz, M, N)
    k = _kernel("matmul", "matmul_forward",
                ["A", "B", "Mb", "Kb", "Nb", "Sh"], ["out"])
    out_shape = tuple(a.shape[:-2]) + (M, N)
    (out,) = k(inputs=[a, b, _u32(M), _u32(K), _u32(N), _u32(shared)],
               grid=grid, threadgroup=tg,
               output_shapes=[out_shape], output_dtypes=[a.dtype])
    return out


def matmul_backward_dA(grad_in, b, dout):
    # grad_in carries A's shape; B's layout (batched/shared) comes from b
    Bsz, M, K, N, shared = _mm_dims(grad_in, b)
    grid, tg = _mm_grid(K, M, Bsz)   # output (Bsz, M, K)
    k = _kernel("matmul", "matmul_backward_dA",
                ["grad_in", "B", "dC", "Mb", "Kb", "Nb", "Sh"], ["out"])
    (out,) = k(inputs=[grad_in, b, dout, _u32(M), _u32(K), _u32(N), _u32(shared)],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


def matmul_backward_dB(grad_in, a, dout):
    # grad_in carries B's shape; shared <=> B is lower-rank than A (2-D weight)
    M, K, N = a.shape[-2], a.shape[-1], grad_in.shape[-1]
    Bsz = a.size // (M * K)
    if grad_in.ndim < a.ndim:  # shared: sum over the batch (grid.z = 1)
        grid, tg = _mm_grid(N, K, 1)   # output (K, N), batch folded inside kernel
        k = _kernel("matmul", "matmul_backward_dB_shared",
                    ["grad_in", "A", "dC", "Mb", "Kb", "Nb", "Bsz"], ["out"])
        (out,) = k(inputs=[grad_in, a, dout, _u32(M), _u32(K), _u32(N), _u32(Bsz)],
                   grid=grid, threadgroup=tg,
                   output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    else:                      # batched: one TILE x TILE tile per (b, k, n)
        grid, tg = _mm_grid(N, K, Bsz)   # output (Bsz, K, N)
        k = _kernel("matmul", "matmul_backward_dB_batched",
                    ["grad_in", "A", "dC", "Mb", "Kb", "Nb"], ["out"])
        (out,) = k(inputs=[grad_in, a, dout, _u32(M), _u32(K), _u32(N)],
                   grid=grid, threadgroup=tg,
                   output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


# --- vectorwise: last-dim reductions (sum/mean/max/min/var/std) + softmax ----
# one thread per row (grid = n_rows); var/std use population (/n) normalization.
_REDUCTION_OP = {"sum": 0, "mean": 1, "max": 2, "min": 3, "var": 4, "std": 5}


def reduction_forward(x, n_rows, n_cols, op):
    grid, tg = _launch(n_rows)
    k = _kernel("vectorwise", "reduction_forward", ["x", "n_cols", "op"], ["out"])
    (out,) = k(inputs=[x, _u32(n_cols), _u32(_REDUCTION_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[x.shape[:-1]], output_dtypes=[x.dtype])
    return out


def reduction_backward(grad_in, x, dout, out_fwd, n_rows, n_cols, op):
    grid, tg = _launch(n_rows)
    k = _kernel("vectorwise", "reduction_backward",
                ["grad_in", "x", "dout", "out_fwd", "n_cols", "op"], ["out"])
    (out,) = k(inputs=[grad_in, x, dout, out_fwd, _u32(n_cols), _u32(_REDUCTION_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


def softmax_forward(x, n_rows, n_cols):
    grid, tg = _launch(n_rows)
    k = _kernel("vectorwise", "softmax_forward", ["x", "n_cols"], ["out"])
    (out,) = k(inputs=[x, _u32(n_cols)],
               grid=grid, threadgroup=tg,
               output_shapes=[x.shape], output_dtypes=[x.dtype])
    return out


def softmax_backward(grad_in, out_fwd, dout, n_rows, n_cols):
    grid, tg = _launch(n_rows)
    k = _kernel("vectorwise", "softmax_backward",
                ["grad_in", "y", "dout", "n_cols"], ["out"])
    (out,) = k(inputs=[grad_in, out_fwd, dout, _u32(n_cols)],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


# --- modules: embedding + layernorm ----------------------------------------
# Atomic-free: backward kernels parallelize over OUTPUT elements and gather over
# inputs (embedding: one thread per weight elem scans tokens; layernorm dw/db:
# one thread per feature scans rows). layernorm var uses population (/D) norm.

def embedding_forward(tokens, weight, N, D, V):
    rows = tokens.size  # B*N
    B = rows // N
    grid, tg = _launch(rows * D)
    k = _kernel("modules", "embedding_forward", ["tokens", "weight", "D"], ["out"])
    (out,) = k(inputs=[tokens, weight, _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[(B, N, D)], output_dtypes=[weight.dtype])
    return out


def embedding_backward(grad_in, tokens, dout, N, D, V):
    rows = tokens.size  # B*N
    grid, tg = _launch(V * D)  # one thread per weight element (v, d)
    k = _kernel("modules", "embedding_backward",
                ["grad_in", "tokens", "dout", "D", "rows"], ["out"])
    (out,) = k(inputs=[grad_in, tokens, dout, _u32(D), _u32(rows)],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


def layernorm_forward(x, w, b, rows, D, eps):
    grid, tg = _launch(rows)
    k = _kernel("modules", "layernorm_forward",
                ["x", "w", "b", "D", "epsb"], ["out", "mean", "rstd"])
    out, mean, rstd = k(inputs=[x, w, b, _u32(D), _f32(eps)],
                        grid=grid, threadgroup=tg,
                        output_shapes=[x.shape, (rows,), (rows,)],
                        output_dtypes=[x.dtype, x.dtype, x.dtype])
    return out, mean, rstd


def layernorm_backward(x, w, b, dx_in, dw_in, db_in, dout, mean, rstd, rows, D):
    # dx: one thread per row (functional accumulate into dx_in)
    grid, tg = _launch(rows)
    kdx = _kernel("modules", "layernorm_backward_dx",
                  ["dx_in", "x", "w", "dout", "mean", "rstd", "D"], ["out"])
    (dx,) = kdx(inputs=[dx_in, x, w, dout, mean, rstd, _u32(D)],
                grid=grid, threadgroup=tg,
                output_shapes=[x.shape], output_dtypes=[x.dtype])
    # dw/db: one thread per feature d, gather over rows (no atomics)
    grid, tg = _launch(D)
    kwb = _kernel("modules", "layernorm_backward_dwdb",
                  ["dw_in", "db_in", "x", "dout", "mean", "rstd", "D", "rows"],
                  ["dw", "db"])
    dw, db = kwb(inputs=[dw_in, db_in, x, dout, mean, rstd, _u32(D), _u32(rows)],
                 grid=grid, threadgroup=tg,
                 output_shapes=[dw_in.shape, db_in.shape],
                 output_dtypes=[dw_in.dtype, db_in.dtype])
    return dx, dw, db


# --- flash attention: causal attention (forward / backward) ----------------
# CAUSAL (query i attends keys j<=i). Simple one-thread-per-row (grid = B*H*N);
# `scale` is the multiplier (= sqrt(D) in the suite), used verbatim. Backward
# uses Delta[i] = Σ_d O[i,d]·dO[i,d] and accumulates functionally into dQ/dK/dV.

def flash_attention_forward(Q, K, V, scale, B, H, N, D):
    grid, tg = _launch(B * H * N)
    k = _kernel("flash_attention", "flash_forward",
                ["Q", "K", "V", "scale", "N", "D"], ["O", "LSE"])
    O, LSE = k(inputs=[Q, K, V, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape, (B, H, N)], output_dtypes=[Q.dtype, Q.dtype])
    return O, LSE


def flash_attention_backward(Q, K, V, O, dO, dQ_in, dK_in, dV_in, LSE, scale, B, H, N, D):
    grid, tg = _launch(B * H * N)

    kd = _kernel("flash_attention", "flash_delta", ["O", "dO", "N", "D"], ["Delta"])
    (Delta,) = kd(inputs=[O, dO, _u32(N), _u32(D)],
                  grid=grid, threadgroup=tg,
                  output_shapes=[(B, H, N)], output_dtypes=[O.dtype])

    kv = _kernel("flash_attention", "flash_dV",
                 ["dV_in", "Q", "K", "dO", "LSE", "scale", "N", "D"], ["out"])
    (dV,) = kv(inputs=[dV_in, Q, K, dO, LSE, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape], output_dtypes=[Q.dtype])

    kq = _kernel("flash_attention", "flash_dQ",
                 ["dQ_in", "Q", "K", "V", "dO", "LSE", "Delta", "scale", "N", "D"], ["out"])
    (dQ,) = kq(inputs=[dQ_in, Q, K, V, dO, LSE, Delta, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape], output_dtypes=[Q.dtype])

    kk = _kernel("flash_attention", "flash_dK",
                 ["dK_in", "Q", "K", "V", "dO", "LSE", "Delta", "scale", "N", "D"], ["out"])
    (dK,) = kk(inputs=[dK_in, Q, K, V, dO, LSE, Delta, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape], output_dtypes=[Q.dtype])

    return dQ, dK, dV
