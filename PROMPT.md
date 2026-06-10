# Task: finish the tritongrad implementation

You are finishing the `tritongrad` implementation in this repo (an educational
autograd-engine tutorial with three parallel implementations: micrograd, minigrad,
tritongrad). tritongrad is the advanced tier: a `TritonTensor` wrapping a torch.cuda
tensor, with all forward/backward math written as custom OpenAI Triton GPU kernels.

**REQUIRES AN NVIDIA GPU — you must run on a GPU box.** If
`python -c "import torch;print((torch.ones(8,device='cuda')+1).sum())"` errors, the
installed torch lacks kernels for this GPU; install a matching build, e.g.
`pip install --index-url https://download.pytorch.org/whl/cu128 torch triton`.

You are working on the `finish-tritongrad` branch (a dedicated git worktree).

## Your files (all under `tritongrad/`)
- `engine.py` — `TritonTensor`/`Parameter` (binary ops, matmul, unary `exp`/`log`/
  `relu`, reductions `sum`/`mean`/`max`/`min`/`var`/`std`, shape ops, `.backward(grad)`,
  `.zero_grad_backward()`)
- `nn.py` — `Module`/`Linear`/`Embedding`/`LayerNorm`/`FlashAttention`
- `kernels/` — `elementwise.py`, `matmul.py`, `vectorwise.py`, `modules.py`,
  `flash_attention.py` (the actual Triton kernels)
- `testing.py` — the ORIGINAL standalone op-test harness (superseded by `tests/` for
  verification; keep for reference)
- `benchmarking.py` — performance benchmarks
- **NOTE: there is NO `model.py` and NO `train.py` yet — you will create them.**

## Your verification harness — use it constantly
There is a unified pytest suite in `tests/` that checks every backend against PyTorch
(forward AND backward), ops + nn modules. Run YOUR slice with (from the repo root, on
the GPU box):

    python -m pytest tests/ -k tritongrad -v

First run is slow (Triton JIT/autotune). Read `tests/README.md`. tritongrad is wired in
via `tests/adapters/tritongrad_adapter.py`, which currently declares:

- OPS: `add, sub, mul, div, matmul, exp, log, relu, sum_lastdim, mean, var, std`
  (NOT `softmax` — it's an unimplemented stub)
- MODULES: `linear, embedding, layernorm, attention`

**IMPORTANT** — the adapter encodes a critical detail: tritongrad's backward kernels
accumulate into `.grad` with `+=`, and Triton's autotuner runs each config many times
on a kernel's first call, so a naive single `.backward()` compounds gradients ~10^4x.
The adapter works around it with a warmup dance (backward with zeros ->
`zero_grad_backward` -> real backward). Keep this in mind if you touch backward kernels
or write new ones. When you add a capability (e.g. softmax), add it to the adapter
`OPS`/`MODULES` and a registry entry in `tests/core/registry.py`, then keep the slice
green.

## What's left (verify each against the suite where applicable)
1. **softmax kernel**: `TritonTensor.softmax()` (`engine.py:346`) is a `pass` stub.
   Implement it as a real Triton kernel (fwd+bwd), then enable `"softmax"` in the
   adapter `OPS` and confirm it matches `torch.softmax(dim=-1)`. Needed for
   attention/inference.
2. **Negation kernel**: `TritonTensor.__neg__` (`engine.py:156`) raises
   `NotImplementedError`. Implement it (fwd+bwd).
3. **CREATE `model.py`**: a GPT built from `nn.py`'s `Linear`/`Embedding`/`LayerNorm`/
   `FlashAttention` — use `minigrad/model.py` as the architectural reference
   (MultiHeadSelfAttention via FlashAttention, MLP, ResidualLayer, GPT). Add the larger
   module/model pieces to the test suite where feasible.
4. **CREATE `train.py`**: character-level training on `../input.txt`, mirroring
   `minigrad/train.py` but on GPU with TritonTensors. Confirm the loss decreases.
5. **Robustness TODOs** from `tritongrad/README.md` and code comments: `Embedding`
   should validate token indices `< num_embeddings` (`nn.py`); `LayerNorm` should handle
   `elementwise_affine=False` / `bias=False` (`nn.py:160` comment); `max`/`min`
   reductions should optionally return indices (needed for inference argmax); clean up
   the if-branch logic in kernels.
6. **(Optional)** Address the underlying autotuning+accumulation design so backward
   kernels don't depend on the test's warmup dance — e.g. zero-init grads per call or
   guard accumulation. If you change this, update the adapter's `_seeded_backward`
   accordingly.

## Context — a bug this suite already caught and fixed (don't reintroduce it)
In `kernels/vectorwise.py` the `var`/`std` FORWARD must subtract `mean(x)`
(`sum(x)/row_len`), not `sum(x)`, and the implementation is population variance (`/n`)
so forward, backward, and `torch.var(unbiased=False)` agree.

## Rules
- All real math stays in Triton kernels (the whole point of this tier) — don't fall back
  to torch ops for the forward/backward compute.
- fp32 only (the engine enforces this).
- Don't break other backends: after touching shared files (`tests/core/registry.py`,
  `tests/adapters/torch_ref.py`), run the FULL suite and keep it green (micrograd/
  minigrad run on CPU; they should stay passing).
- Commit on the `finish-tritongrad` branch.

## Done =
softmax + negation kernels implemented and covered by the suite, `model.py` and
`train.py` exist and `train.py` demonstrably learns on `input.txt`, and
`python -m pytest tests/ -k tritongrad` is green (no skips for things you implemented).
