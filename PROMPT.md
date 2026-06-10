# Task: finish the minigrad implementation

You are finishing the `minigrad` implementation in this repo (an educational
autograd-engine tutorial with three parallel implementations: micrograd, minigrad,
tritongrad). minigrad is the intermediate tier: a numpy-backed `Tensor` class with a
PyTorch-like API (`requires_grad`, broadcasting, `.backward()`), used to build a full
GPT. CPU-only (numpy).

You are working on the `finish-minigrad` branch (a dedicated git worktree).

## Your files (all under `minigrad/`)
- `engine.py` — `Tensor`/`Parameter`: `__add__`/`__sub__`/`__mul__`/`__truediv__`/
  `__matmul__`/`__pow__`/`__getitem__`, `sum`/`mean`/`var`/`sd`, `exp`/`log`/`relu`/
  `max`/`min`/`softmax`, `transpose`/`squeeze`/`unsqueeze`/`broadcast_to`/`reshape`/
  `masked_fill`, `.backward()`
- `nn.py` — `Module`/`Linear`/`Embedding`/`Dropout`/`LayerNorm`/`CrossEntropyLoss`
- `model.py` — `MultiHeadSelfAttention`/`MLP`/`ResidualLayer`/`GPT`
- `train.py` — character-level training on `../input.txt` (Shakespeare)

## Your verification harness — use it constantly
There is a unified pytest suite in `tests/` that checks every backend against PyTorch
(forward AND backward) for both tensor ops and nn modules. Run YOUR slice with (from
the repo root):

    python -m pytest tests/ -k minigrad -v

Read `tests/README.md`. minigrad is wired in via `tests/adapters/minigrad_adapter.py`,
which currently declares:

- OPS: `add, sub, mul, div, matmul, exp, log, relu, softmax, sum_lastdim, mean, var, std`
- MODULES: `linear, embedding, layernorm`

The engine ops and these modules already pass against PyTorch. When you add a new
capability (e.g. an attention module), add its name to the adapter's `OPS`/`MODULES`
set, add a registry entry in `tests/core/registry.py` if needed, and keep the slice
green. The adapter's `run_module` shows the weight-sync pattern (note: torch `Linear`
weight is `(out,in)` while minigrad stores `(in,out)` — transpose on sync and on grad
readback).

## What's left (verify each against the suite where applicable)
1. **FIX `model.py` `GPT.__call__`** (around line 142): it reads `input_tokens` which is
   undefined (should be `input_token_ids`), and it unpacks `B, S = input_token_ids.shape`
   BEFORE the `if ndim == 1` guard, so a single (1-D) sequence crashes. Make the 1-D /
   batched handling correct, then confirm a forward+backward pass runs.
2. **CrossEntropyLoss + inference**: make the model output logits and have
   `CrossEntropyLoss` apply log-softmax internally so inference can softmax the logits
   (README TODO).
3. **train.py**: confirm it actually trains after the `model.py` fix — loss should
   decrease on `../input.txt`; greedy-sample some characters to sanity check.
4. **(Optional but valuable) Attention-module parity test**: factor the attention in
   `model.py` into something the suite can exercise, add an `"attention"` entry to the
   adapter `MODULES` + a registry `ModuleSpec` (the tritongrad adapter + registry show
   how attention is compared against `torch.nn.functional.scaled_dot_product_attention`
   with causal masking).
5. **Audit LayerNorm init**: minigrad inits the affine weight to noise (`nn.py`) — fine
   for the suite (it weight-syncs from torch) but make sure training uses sensible init.

## Context — a bug this suite already caught and fixed (don't reintroduce it)
`Tensor.sum`'s backward must re-insert the reduced axis (`np.expand_dims`) when
`keepdim=False` before broadcasting; `mean`/`var`/`sd` depend on it.

## Rules
- Keep minigrad numpy-only and PyTorch-like; it's the intermediate lesson.
- Don't break other backends: after touching shared files (`tests/core/registry.py`,
  `tests/adapters/torch_ref.py`), run the FULL suite `python -m pytest tests/` and keep
  it green.
- Commit on the `finish-minigrad` branch.

## Done =
`model.py` fixed, `train.py` demonstrably learns on `input.txt`, logits/softmax
inference path working, and `python -m pytest tests/ -k minigrad` green with any new
ops/modules covered.
