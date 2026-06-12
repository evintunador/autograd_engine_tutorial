# autograd_engine_tutorial

![CI](https://github.com/evintunador/autograd_engine_tutorial/actions/workflows/ci.yml/badge.svg)

An educational autograd-engine tutorial built in **five parallel tiers** of increasing hardware-closeness — scalar Python → numpy tensors → Triton tile kernels → raw CUDA / Metal thread kernels. Each tier is just enough of an autograd engine to build a small GPT, and each is checked against [PyTorch](https://pytorch.org) by a single unified `tests/` suite.

## Getting started

```bash
pip install -r requirements.txt   # GPU tiers need extra hardware-specific setup — see their READMEs
pytest tests/                      # run the whole suite; tiers whose hardware you lack are skipped, not failed
```

## The tier ladder

Each tier teaches the same autograd + GPT math, one level closer to the metal:

```
  micrograd   scalar Python `Value`      one float of data + one float of grad        (no numpy)
      |
  minigrad    numpy tensors              array-valued ops, real linear algebra        (numpy)
      |
  tritongrad  Triton tile kernels        tile-level GPU programming                   (NVIDIA GPU)
      |
   +--+--+
   |     |
 cudagrad  mlxgrad   raw thread kernels   CUDA C++ (NVIDIA)  /  Metal MSL (Apple)
```

## The tiers at a glance

|                   |                                                                                                                                                                    | micrograd            | minigrad             | autograd<br>(Triton) | autograd<br>(CUDA)     | autograd<br>(Metal / MLX) |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | -------------------- | -------------------- | ---------------------- | ------------------------- |
|                   | difficulty level                                                                                                                                                   | beginner             | intermediate         | advanced             | advanced               | advanced                  |
|                   | time commitment                                                                                                                                                    | medium               | small                | large                | large                  | large                     |
|                   | status                                                                                                                                                             | functional           | functional           | functional (GPU-verified) | complete, GPU-verified 20/20 | complete, verified on Apple Silicon 20/20 |
|                   | attempts to (mostly) resemble [PyTorch](https://pytorch.org) syntax                                                                                                | ❌                    | ✅                    | ✅                    | ✅                     | ✅                         |
|                   |                                                                                                                                                                    |                      |                      |                      |                        |                           |
| prerequisites     | basic python                                                                                                                                                       | ✅                    | ✅                    | ✅                    | ❌                     | ❌                         |
|                   | basic C                                                                                                                                                            | ❌                    | ❌                    | ❌                    | ✅                     | ✅                         |
|                   | a tiny bit of introductory calculus (what a derivative is conceptually)                                                                                            | ✅                    | ✅                    | ✅                    | ✅                     | ✅                         |
|                   | a tiny bit of introductory linear algebra (matrices and matrix multiplication)                                                                                     | ❌                    | ✅                    | ✅                    | ✅                     | ✅                         |
|                   | familiarity with [numpy](https://numpy.org)                                                                                                                        | ❌                    | ✅                    | ❌                    | ❌                     | ❌                         |
|                   | linux (or use [colab](https://colab.research.google.com), [lambda](https://lambdalabs.com) or similar)                                                             | ❌                    | ❌                    | ✅                    | ✅                     | ❌                         |
|                   | Nvidia GPU (or use [colab](https://colab.research.google.com), [lambda](https://lambdalabs.com) or similar)                                                        | ❌                    | ❌                    | ✅                    | ✅                     | ❌                         |
|                   | an Apple-Silicon Mac                                                                                                                                               | ❌                    | ❌                    | ❌                    | ❌                     | ✅                         |
|                   |                                                                                                                                                                    |                      |                      |                      |                        |                           |
| What you'll learn | basic math of autograd systems                                                                                                                                     | ✅                    | ✅                    | ✅                    | ✅                     | ✅                         |
|                   | basic math of [GPT-2](https://en.wikipedia.org/wiki/GPT-2#:~:text=Generative%20Pre%2Dtrained%20Transformer%202,of%208%20million%20web%20pages.) style transformers | ✅                    | ✅                    | ✅                    | ✅                     | ✅                         |
|                   | basics of efficient parallel programming linear algebra on GPUs                                                                                                    | ❌                    | ❌                    | ✅                    | ✅                     | ✅                         |
|                   | OpenAI's [Triton](https://triton-lang.org/main/index.html)                                                                                                         | ❌                    | ❌                    | ✅                    | ❌                     | ❌                         |
|                   | Nvidia's [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)     | ❌                    | ❌                    | ❌                    | ✅                     | ❌                         |
|                   | Apple's [Metal](https://developer.apple.com/metal/) (MSL)                                                                                                          | ❌                    | ❌                    | ❌                    | ❌                     | ✅                         |

## testing
all five implementations are checked against [PyTorch](https://pytorch.org) (forward & backward, ops & nn modules) by a single unified suite in [`tests/`](tests/). run `pytest tests/` from the repo root; backends needing hardware you don't have (e.g. `tritongrad` and `cudagrad` want a CUDA GPU, `mlxgrad` wants an Apple-Silicon Mac) are skipped rather than failed. adding a new implementation just means writing one adapter file — see [`tests/README.md`](tests/README.md).

## Videos

This maintainer makes AI-research videos. Recommended path:

- **Start here:** Andrej Karpathy's [micrograd video](https://youtu.be/VMj-3S1tku0) — the clearest introduction to autograd basics; the micrograd tier builds directly on it.
- **Triton-kernel tutorial series** *(TODO: link)* — the maintainer's walkthrough of the GPU-kernel tiers.
- **This repo's companion video** *(TODO: link)*.

## micrograd
the purpose of this lesson is for absolute beginners with a programming (as opposed to math) background to learn about the math and implementation of GPTs all the way from the barebones autograd engine and up to the GPT operations itself. the basic building block of micrograd is the `Value` object, each of which is just a single floating point number for the data and another single floating point number to keep track of the data's gradient. the first half or so of this lesson is roughly equivalent to [karpathy's `micrograd`](https://youtu.be/VMj-3S1tku0?si=FM0qtfV-cvXr2kDJ) while the second half is an extension to implement a full GPT. see [`micrograd/README.md`](micrograd/README.md).

## minigrad
the purpose of this lesson is for people already confident with linear algebra and calculus to learn exactly what's happening in GPTs all the way from the barebones autograd engine and up to the GPT operations itself. to make this happen we'll be doing everything with numpy arrays, meaning that the basic unit of our engine will be tensors as opposed to individual values. see [`minigrad/README.md`](minigrad/README.md).

## autograd (Triton)
the purpose of this lesson is for people already confident in the math behind autograd engines and GPTs to learn exactly what's happening at the level of the GPU hardware. you can think of autograd as a replication of pytorch/tensorflow/jax/mlx/tinygrad/etc except that instead of being meant to actually be used, which would require it be flexible and robust, it is meant for educational purposes. As such, we'll only be implementing the operations that are absolutely necessary in order to create a GPT and not worrying much about unexpected edge cases, just like how we did in micrograd and minigrad.

### why triton?
You might be asking: why are we using Triton instead of CUDA? Here are the pros and cons that it came down to:

|      | triton                                                                                            | cuda                                                             |
| ---- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| pros | - written in Python (quicker to learn)<br>- works on more than just Nvidia GPUs<br>- open-sourced | - broadly used<br>- linux or windows                             |
| cons | - less commonly used<br>- requires linux                                                          | - written in C<br>- only works on Nvidia GPUs<br>- closed-source |

Personally I'm on a Mac so i plan on doing all my work on a cloud provider like [lambdalabs](https://lambdalabs.com) anyways so the windows availability didn't matter much to me. That and I highly value the pythonic syntax and potential future widespread compatibility. see [`tritongrad/README.md`](tritongrad/README.md).

## autograd (CUDA)
the lowest tier. where the Triton lesson writes the forward/backward math as tile-level Triton kernels, [`cudagrad/`](cudagrad/) writes it one step lower as **raw CUDA C++ kernels** (JIT-compiled via `torch.utils.cpp_extension.load`), so you can see exactly what the GPU hardware is doing. same educational goal as every tier: just enough of an autograd engine to build a GPT. it mirrors `tritongrad/` op-for-op — `CudaTensor` wraps a `torch.float32` CUDA tensor for memory but runs every operation's math in our own `.cu` kernels. complete and GPU-verified: all 16 ops and 4 modules pass the repo-wide suite against PyTorch (`pytest tests/ -k cudagrad` → 20/20), with `model.py`/`train.py`/`benchmarking.py` mirroring the Triton tier. requires an NVIDIA GPU + CUDA toolkit (`nvcc`); see [`cudagrad/README.md`](cudagrad/README.md) for details.

## autograd (Metal / MLX)
the Apple-Silicon sibling of the CUDA tier. [`mlxgrad/`](mlxgrad/) writes the forward/backward math as **raw [Metal](https://developer.apple.com/metal/) kernels** (Metal Shading Language, JIT-compiled by [MLX](https://github.com/ml-explore/mlx)'s `mx.fast.metal_kernel`) — the closest Apple analog to thread-level CUDA. (Apple has no first-party *tile*-level DSL, so there's no Metal equivalent of the Triton tier.) `MLXTensor` wraps an `mlx.core` fp32 array for memory but runs every op's math in our own `.metal` kernels. complete and verified: all 16 ops + 4 modules pass the suite against PyTorch (`pytest tests/ -k mlxgrad` → 20/20), with `model.py`/`train.py`/`benchmarking.py` mirroring the CUDA tier. its headline advantage over the Triton/CUDA tiers: MLX runs on Apple Silicon, so unlike those (NVIDIA-GPU-only) it's authored **and** verified locally on a Mac — no cloud box. `pip install mlx` (Apple Silicon only; other hosts skip the tier cleanly). see [`mlxgrad/README.md`](mlxgrad/README.md).
