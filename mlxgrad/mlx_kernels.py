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


def _kernel(group, name, input_names, output_names, header=""):
    key = (group, name)
    if key not in _KERNEL_CACHE:
        src = _sources(group)[name]
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"{group}_{name}",
            input_names=input_names,
            output_names=output_names,
            source=src,
            header=header,
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


# simdgroup_matrix (MMA) GEMM: each threadgroup computes a _BM x _BN output tile
# for one batch using a _SGM x _SGN grid of simdgroups (8 simdgroups = _NTHREADS
# threads), each accumulating _WM x _WN simdgroup_float8x8 fragments. These MUST
# match the tile #defines at the top of each kernel in kernels/matmul.metal.
_BM, _BN, _BK = 64, 64, 8
_NTHREADS = 256          # 8 simdgroups (SGM=2 x SGN=4), 32 lanes each

# simdgroup_matrix intrinsics live in these headers; prepend them to each body.
_MM_HEADER = "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"


def _mm_grid(out_cols, out_rows, batch):
    """(grid, threadgroup) for the MMA GEMM whose output is (batch, rows, cols).

    grid is TOTAL threads (MLX non-uniform dispatch): x spans
    ceil(cols/_BN) threadgroups of _NTHREADS threads each; y spans
    ceil(rows/_BM) threadgroups (1 thread-row apiece); z = batch. The kernel
    bounds-checks output coords and zero-pads staged tiles for edge tiles, so
    arbitrary M/K/N (incl. < 8) are handled.
    """
    def cdiv(x, t):
        return (int(x) + t - 1) // t
    grid = (cdiv(out_cols, _BN) * _NTHREADS, cdiv(out_rows, _BM), int(batch))
    tg = (_NTHREADS, 1, 1)
    return grid, tg


def matmul_forward(a, b):
    Bsz, M, K, N, shared = _mm_dims(a, b)
    grid, tg = _mm_grid(N, M, Bsz)   # output (Bsz, M, N): rows=M, cols=N
    k = _kernel("matmul", "matmul_forward",
                ["A", "B", "Mb", "Kb", "Nb", "Sh"], ["out"], header=_MM_HEADER)
    out_shape = tuple(a.shape[:-2]) + (M, N)
    (out,) = k(inputs=[a, b, _u32(M), _u32(K), _u32(N), _u32(shared)],
               grid=grid, threadgroup=tg,
               output_shapes=[out_shape], output_dtypes=[a.dtype])
    return out


def matmul_backward_dA(grad_in, b, dout):
    # grad_in carries A's shape; B's layout (batched/shared) comes from b
    Bsz, M, K, N, shared = _mm_dims(grad_in, b)
    grid, tg = _mm_grid(K, M, Bsz)   # output (Bsz, M, K): rows=M, cols=K
    k = _kernel("matmul", "matmul_backward_dA",
                ["grad_in", "B", "dC", "Mb", "Kb", "Nb", "Sh"], ["out"],
                header=_MM_HEADER)
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
                    ["grad_in", "A", "dC", "Mb", "Kb", "Nb", "Bsz"], ["out"],
                    header=_MM_HEADER)
        (out,) = k(inputs=[grad_in, a, dout, _u32(M), _u32(K), _u32(N), _u32(Bsz)],
                   grid=grid, threadgroup=tg,
                   output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    else:                      # batched: output (Bsz, K, N): rows=K, cols=N
        grid, tg = _mm_grid(N, K, Bsz)
        k = _kernel("matmul", "matmul_backward_dB_batched",
                    ["grad_in", "A", "dC", "Mb", "Kb", "Nb"], ["out"],
                    header=_MM_HEADER)
        (out,) = k(inputs=[grad_in, a, dout, _u32(M), _u32(K), _u32(N)],
                   grid=grid, threadgroup=tg,
                   output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


# --- vectorwise: last-dim reductions (sum/mean/max/min/var/std) + softmax ----
# one thread per row (grid = n_rows); var/std use population (/n) normalization.
_REDUCTION_OP = {"sum": 0, "mean": 1, "max": 2, "min": 3, "var": 4, "std": 5}

# Vectorwise kernels run ONE THREADGROUP PER ROW: the threadgroup's threads
# cooperatively reduce the row's columns via SIMD reductions (simd_sum/max/min)
# + threadgroup memory + a grid-strided column loop. The threadgroup size is
# chosen per-call from n_cols: a multiple of the 32-lane SIMD width, capped at
# 256 (the scratch arrays in vectorwise.metal hold 256/32 = 8 simdgroup slots),
# and at least 32. Narrow rows therefore use a SMALL threadgroup so many rows
# stay resident concurrently (recovering the row-parallelism the simple
# one-thread-per-row design had); wide rows use up to 256 threads to parallelize
# the long column reduction. The kernel reads threads_per_threadgroup at runtime
# (TPT) for the stride and derives the simdgroup count (NSG), so it stays correct
# for any size we pick here.
_VEC_SIMD = 32      # Apple GPU SIMD width
_VEC_TG_MAX = 256   # must match the scratch-array bound (256/32 = 8) in the .metal


def _vec_tg(n_cols):
    """Threadgroup size (multiple of 32, in [32, 256]) for a row of n_cols."""
    nc = max(int(n_cols), 1)
    tg = ((nc + _VEC_SIMD - 1) // _VEC_SIMD) * _VEC_SIMD   # round up to 32
    return max(_VEC_SIMD, min(_VEC_TG_MAX, tg))


# Narrow rows (n_cols <= 32) get a SECOND launch mode: ONE SIMDGROUP PER ROW.
# Several (_VEC_NARROW_RPT) simdgroups are packed per threadgroup, so each
# 32-lane simdgroup reduces one whole row with a single simd_* op — no
# threadgroup memory, no barriers, every lane doing useful work. This recovers
# the row-parallelism the simple one-thread-per-row design had (round 1's
# narrow-row regression) while keeping the SIMD reduction. The kernels branch on
# rpt[0]: rpt > 1 selects the narrow path, rpt == 1 the wide threadgroup-per-row
# path. n_rows[0] bounds the last (possibly partly-empty) threadgroup.
_VEC_NARROW_MAX = 32   # narrow path requires the whole row to fit one simdgroup
_VEC_NARROW_RPT = 8    # simdgroups (= rows) per threadgroup -> 256 threads


def _vec_launch(n_rows, n_cols):
    """((grid, threadgroup), rpt): pick narrow (one simdgroup per row) vs wide
    (one threadgroup per row) from n_cols. rpt is the rows-per-threadgroup the
    kernel reads to select its path (1 == wide)."""
    nr = max(int(n_rows), 1)
    nc = max(int(n_cols), 1)
    if nc <= _VEC_NARROW_MAX:
        rpt = _VEC_NARROW_RPT
        ntg = (nr + rpt - 1) // rpt          # threadgroups to cover all rows
        tg = rpt * _VEC_SIMD                  # rpt simdgroups of 32 lanes
        return ((tg * ntg, 1, 1), (tg, 1, 1)), rpt
    tg = _vec_tg(nc)
    return ((tg * nr, 1, 1), (tg, 1, 1)), 1


def reduction_forward(x, n_rows, n_cols, op):
    (grid, tg), rpt = _vec_launch(n_rows, n_cols)
    k = _kernel("vectorwise", "reduction_forward",
                ["x", "n_cols", "n_rows", "rpt", "op"], ["out"])
    (out,) = k(inputs=[x, _u32(n_cols), _u32(n_rows), _u32(rpt), _u32(_REDUCTION_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[x.shape[:-1]], output_dtypes=[x.dtype])
    return out


def reduction_backward(grad_in, x, dout, out_fwd, n_rows, n_cols, op):
    (grid, tg), rpt = _vec_launch(n_rows, n_cols)
    k = _kernel("vectorwise", "reduction_backward",
                ["grad_in", "x", "dout", "out_fwd", "n_cols", "n_rows", "rpt", "op"], ["out"])
    (out,) = k(inputs=[grad_in, x, dout, out_fwd, _u32(n_cols), _u32(n_rows),
                       _u32(rpt), _u32(_REDUCTION_OP[op])],
               grid=grid, threadgroup=tg,
               output_shapes=[grad_in.shape], output_dtypes=[grad_in.dtype])
    return out


def softmax_forward(x, n_rows, n_cols):
    (grid, tg), rpt = _vec_launch(n_rows, n_cols)
    k = _kernel("vectorwise", "softmax_forward",
                ["x", "n_cols", "n_rows", "rpt"], ["out"])
    (out,) = k(inputs=[x, _u32(n_cols), _u32(n_rows), _u32(rpt)],
               grid=grid, threadgroup=tg,
               output_shapes=[x.shape], output_dtypes=[x.dtype])
    return out


def softmax_backward(grad_in, out_fwd, dout, n_rows, n_cols):
    (grid, tg), rpt = _vec_launch(n_rows, n_cols)
    k = _kernel("vectorwise", "softmax_backward",
                ["grad_in", "y", "dout", "n_cols", "n_rows", "rpt"], ["out"])
    (out,) = k(inputs=[grad_in, out_fwd, dout, _u32(n_cols), _u32(n_rows), _u32(rpt)],
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


def _launch_rows(rows, D):
    """(grid, threadgroup) for ONE THREADGROUP PER ROW: TG threads cooperatively
    reduce each row over D. TG is a multiple of the 32-wide simd, capped so the
    32-slot per-simdgroup partial buffers in the kernel suffice, and no larger
    than D (rounded up to a simd) to avoid idle threads."""
    rows = int(rows)
    D = int(D)
    tg = min(256, ((D + 31) // 32) * 32)
    tg = max(32, tg)
    return (rows * tg, 1, 1), (tg, 1, 1)


def layernorm_forward(x, w, b, rows, D, eps):
    grid, tg = _launch_rows(rows, D)
    k = _kernel("modules", "layernorm_forward",
                ["x", "w", "b", "D", "epsb"], ["out", "mean", "rstd"])
    out, mean, rstd = k(inputs=[x, w, b, _u32(D), _f32(eps)],
                        grid=grid, threadgroup=tg,
                        output_shapes=[x.shape, (rows,), (rows,)],
                        output_dtypes=[x.dtype, x.dtype, x.dtype])
    return out, mean, rstd


def layernorm_backward(x, w, b, dx_in, dw_in, db_in, dout, mean, rstd, rows, D):
    # dx: one threadgroup per row (functional accumulate into dx_in)
    grid, tg = _launch_rows(rows, D)
    kdx = _kernel("modules", "layernorm_backward_dx",
                  ["dx_in", "x", "w", "dout", "mean", "rstd", "D"], ["out"])
    (dx,) = kdx(inputs=[dx_in, x, w, dout, mean, rstd, _u32(D)],
                grid=grid, threadgroup=tg,
                output_shapes=[x.shape], output_dtypes=[x.dtype])
    # dw/db: ONE THREADGROUP PER FEATURE d; the TG threads split the row gather
    # (no atomics — each feature owns its output). TG sized to ROWS (mult. of 32,
    # in [32,256]) so small-D launches still fill the GPU.
    twb = max(32, min(256, ((int(rows) + 31) // 32) * 32))
    grid, tg = (int(D) * twb, 1, 1), (twb, 1, 1)
    kwb = _kernel("modules", "layernorm_backward_dwdb",
                  ["dw_in", "db_in", "x", "dout", "mean", "rstd", "D", "rows"],
                  ["dw", "db"])
    dw, db = kwb(inputs=[dw_in, db_in, x, dout, mean, rstd, _u32(D), _u32(rows)],
                 grid=grid, threadgroup=tg,
                 output_shapes=[dw_in.shape, db_in.shape],
                 output_dtypes=[dw_in.dtype, db_in.dtype])
    return dx, dw, db


# --- flash attention: causal attention (forward / backward) ----------------
# CAUSAL (query i attends keys j<=i). TILED / simdgroup_matrix (MMA) design: each
# simdgroup (32 lanes) cooperatively processes an 8x8 BLOCK of rows. Q/K/V/dO blocks
# are staged in threadgroup memory and the score tile S=Q.K^T, the output O+=P.V and
# the gradient tiles are formed by simdgroup_multiply_accumulate on 8x8 fp32 frags;
# online-softmax row reductions run on lanes 0..7 over the materialized score tile.
# `scale` is the multiplier (= sqrt(D) in the suite), used verbatim. Backward uses
# Delta[i] = Σ_d O[i,d]·dO[i,d] and accumulates functionally into dQ/dK/dV.
#
# The MMA grid launches one simdgroup (32 lanes) per (bh, 8-row block): grid =
# (32, B*H*ceil(N/8), 1), threadgroup = (32,1,1). The kernels read the (bh, block)
# pair from threadgroup_position_in_grid.y. The MMA intrinsics require including
# <metal_simdgroup_matrix>, passed via the metal_kernel `header` argument.
# flash_delta keeps the simple one-simdgroup-per-row reduction (it is cheap).

_MMA_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"


def _flash_mma_launch(bh, N):
    """(grid, threadgroup) for one 32-lane simdgroup per (bh, 8-row block).

    grid.y enumerates bh*ceil(N/8) + block; threadgroup = a single 32-lane simdgroup.
    """
    nblk = (int(N) + 7) // 8
    return (32, int(bh) * nblk, 1), (32, 1, 1)


def _flash_delta_launch(rows):
    """(grid, threadgroup) for one 32-lane simdgroup per row (flash_delta only)."""
    total = 32 * int(rows)
    tg = min(128, total) if total > 0 else 32
    tg -= tg % 32
    if tg == 0:
        tg = 32
    return (total, 1, 1), (tg, 1, 1)


def flash_attention_forward(Q, K, V, scale, B, H, N, D):
    grid, tg = _flash_mma_launch(B * H, N)
    k = _kernel("flash_attention", "flash_forward",
                ["Q", "K", "V", "scale", "N", "D"], ["O", "LSE"], header=_MMA_HEADER)
    O, LSE = k(inputs=[Q, K, V, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape, (B, H, N)], output_dtypes=[Q.dtype, Q.dtype])
    return O, LSE


def flash_attention_backward(Q, K, V, O, dO, dQ_in, dK_in, dV_in, LSE, scale, B, H, N, D):
    grid, tg = _flash_mma_launch(B * H, N)

    dgrid, dtg = _flash_delta_launch(B * H * N)
    kd = _kernel("flash_attention", "flash_delta", ["O", "dO", "N", "D"], ["Delta"])
    (Delta,) = kd(inputs=[O, dO, _u32(N), _u32(D)],
                  grid=dgrid, threadgroup=dtg,
                  output_shapes=[(B, H, N)], output_dtypes=[O.dtype])

    kv = _kernel("flash_attention", "flash_dV",
                 ["dV_in", "Q", "K", "dO", "LSE", "scale", "N", "D"], ["out"],
                 header=_MMA_HEADER)
    (dV,) = kv(inputs=[dV_in, Q, K, dO, LSE, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape], output_dtypes=[Q.dtype])

    kq = _kernel("flash_attention", "flash_dQ",
                 ["dQ_in", "Q", "K", "V", "dO", "LSE", "Delta", "scale", "N", "D"], ["out"],
                 header=_MMA_HEADER)
    (dQ,) = kq(inputs=[dQ_in, Q, K, V, dO, LSE, Delta, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape], output_dtypes=[Q.dtype])

    kk = _kernel("flash_attention", "flash_dK",
                 ["dK_in", "Q", "K", "V", "dO", "LSE", "Delta", "scale", "N", "D"], ["out"],
                 header=_MMA_HEADER)
    (dK,) = kk(inputs=[dK_in, Q, K, V, dO, LSE, Delta, _f32(scale), _u32(N), _u32(D)],
               grid=grid, threadgroup=tg,
               output_shapes=[Q.shape], output_dtypes=[Q.dtype])

    return dQ, dK, dV
