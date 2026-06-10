# autograd (cuda)

The fourth and lowest tier of this tutorial. Where [`../tritongrad/`](../tritongrad/)
writes the forward/backward math as tile-level [Triton](https://triton-lang.org)
kernels, `cudagrad` writes it one step lower as **raw CUDA C++ kernels**. Same
goal as every tier: an educational re-implementation of just enough of an autograd
engine to build a GPT — not a robust, general framework.

Like tritongrad, we don't start fully from scratch: CUDA kernels operate on raw
device memory, and the most ergonomic container for that here is a PyTorch
`torch.cuda` tensor. So `CudaTensor` wraps a `torch.float32` CUDA tensor in `.data`,
but every operation's math runs in our own `.cu` kernels — never PyTorch's ops.

> **Requires an NVIDIA GPU + CUDA toolkit (`nvcc`).** The kernels are JIT-compiled
> by `torch.utils.cpp_extension.load` on first use (the first run is slow while
> nvcc compiles, just like Triton's first-call JIT). On a Mac/CPU host the whole
> backend skips cleanly in the test suite.

## How it's built

- `engine.py` — `CudaTensor` / `Parameter`: the autograd graph + topo-sort
  `backward()`, mirroring `tritongrad/engine.py`. Each op allocates its output and
  calls a `cuda_kernels` wrapper.
- `cuda_kernels.py` — thin Python wrappers + the `cpp_extension.load` build glue
  (extension name `cudagrad_ext`). Deliberately **not** named `kernels` to avoid a
  `sys.modules` collision with tritongrad's leaked top-level `kernels` package.
- `kernels/` — a *source directory* (not an importable package): `kernels.h`
  (launcher declarations), `bindings.cpp` (the single pybind entry point), and one
  `.cu` per kernel group (`elementwise.cu`, then `matmul.cu`, `vectorwise.cu`,
  `modules.cu`, ...).
- `nn.py` — `Module` / `Linear` / `Embedding` / `LayerNorm` / `FlashAttention`,
  mirroring `tritongrad/nn.py`.

Unlike tritongrad, the backward pass needs **no warmup dance**: CUDA kernels don't
autotune, so a single `backward()` accumulates each gradient exactly once into the
zero-initialized `.grad` buffers.

## Testing

Correctness lives in the repo-wide [`../tests/`](../tests/) suite, which checks
every backend against PyTorch (forward + backward). Run cudagrad's slice on a CUDA
host from the repo root:

```bash
pytest tests/ -k cudagrad -v
```

## Status

Scaffold: the elementwise **binary** op (`add`/`sub`/`mul`/`div`) is implemented
fwd + bwd; only `add` is enabled in the adapter so far. Remaining kernel groups
(unary, matmul, reductions + softmax, embedding/layernorm, flash attention) plus
`model.py`/`train.py` and `benchmarking.py` are forthcoming — see the project plan.
