# mlxgrad

The Apple-Metal tier of the tutorial: the *same* autograd engine as the others,
but every forward/backward operation runs in a custom **Metal** kernel written in
Metal Shading Language and JIT-compiled by MLX's
[`mx.fast.metal_kernel`](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).

It is the sibling of [`cudagrad/`](../cudagrad): where cudagrad writes raw CUDA
C++ for NVIDIA GPUs, mlxgrad writes Metal for Apple Silicon GPUs — the closest
Apple analog to thread-level CUDA. (Apple has no first-party *tile*-level DSL, so
there is no mlxgrad analog of [`tritongrad/`](../tritongrad)'s Triton tier.)

## Why this tier is the easiest of the GPU tiers to develop

`tritongrad` and `cudagrad` need an NVIDIA GPU, so on a Mac they can only be
authored, not run — every check happens over SSH on a cloud box. **mlxgrad runs
on the dev machine.** Each kernel is `pytest`-verifiable the moment it's written:

```bash
pip install mlx                              # Apple Silicon only
python -m pytest tests/ -k mlxgrad -v        # run mlxgrad against the PyTorch reference
```

On non-Apple hosts the suite skips mlxgrad cleanly (the adapter gates on
`mx.metal.is_available()`).

## Structure (mirrors cudagrad)

- `engine.py` — `MLXTensor` + `Parameter`: wraps an `mx.array`, builds the
  autograd graph, runs the topo-sort backward. Every op routes to a Metal kernel.
- `nn.py` — `Module` / `Linear` / `Embedding` / `LayerNorm` / `FlashAttention`.
- `mlx_kernels.py` — Python wrappers that load the `.metal` sources and launch
  them via `mx.fast.metal_kernel` (uniquely named to avoid the `kernels`
  `sys.modules` collision the test loader would otherwise hit).
- `kernels/*.metal` — the Metal Shading Language kernel bodies (the tutorial
  content). A pure source directory, never imported as Python.
- `model.py` / `train.py` / `benchmarking.py` — a small GPT, char-level training
  on `../input.txt`, and a benchmark harness comparing our kernels against raw MLX
  and PyTorch's MPS backend.

## Performance

The kernels stay readable but are progressively optimized with Metal threadgroup
memory, SIMD-group reductions, and `simdgroup_matrix` MMA (the Apple-silicon
"tensor core" path — usable from inside an `mx.fast.metal_kernel` body via an
`#include <metal_simdgroup_matrix>` header). `benchmarking.py` times them on the
same GPU against **raw MLX** (`mx.*` / `mx.fast.*`) and **PyTorch MPS**
(`python mlxgrad/benchmarking.py --all` → CSV/PNG in `benchmarks/`). Roughly:
softmax and LayerNorm reach MLX parity; matmul and flash attention land within
~2× of MLX — the remaining gap is hand-tuned scheduling, beyond this tier's
educational scope.

> Benchmarking note: MLX is lazily evaluated, so any timing must force `mx.eval`
> on the result (the harness does) — timing an un-eval'd expression measures only
> graph construction, not GPU compute.

## Two things that differ from cudagrad (both because MLX arrays are immutable)

1. **Functional gradient accumulation.** cudagrad's CUDA kernels accumulate in
   place (`dx += ...`). MLX arrays can't be mutated, so each backward kernel
   takes the running gradient `grad_in` and *returns* `grad_in + contribution`;
   the engine rebinds `tensor.grad` to the result. The math (including the `+=`)
   still happens inside the Metal kernel.
2. **No warmup dance.** Like cudagrad — `mx.fast.metal_kernel` does not autotune,
   so a single `backward()` accumulates each gradient exactly once.
