# Task: finish the micrograd implementation

You are finishing the `micrograd` implementation in this repo (an educational
autograd-engine tutorial with three parallel implementations: micrograd, minigrad,
tritongrad). micrograd is the beginner tier: the basic unit is a scalar `Value`
object (one float of data + one float of grad); "tensors" are nested Python lists
of `Value`s. It mirrors Karpathy's micrograd and extends it to a full GPT. CPU-only,
pure Python, no numpy in the engine.

You are working on the `finish-micrograd` branch (a dedicated git worktree).

## Your files (all under `micrograd/`)
- `engine.py` — the `Value` class (scalar autograd: `__add__`/`__mul__`/`__pow__`/
  `__sub__`/`__truediv__`/`__neg__`, `exp`/`log`/`tanh`/`relu`, `.backward()`)
- `ops.py` — FREE FUNCTIONS over nested lists of Values (`tensor_matmul`,
  `entry_wise_add`, `entry_wise_mult`, `vector_wise_apply`, `softmax`, `relu`, `exp`,
  `log`, `sum`, `transpose`, …)
- `modules.py` — `Module`/`Neuron`/`Linear`/`Embedding`/`CrossEntropyLoss`
- `gpt.py` — `layer_norm` (a function), `Mask`, `MultiHeadSelfAttention`, `MLP`,
  `ResidualLayer`, `GPT`
- `train.py` — character-level training on `../input.txt` (Shakespeare)

## Your verification harness — use it constantly
There is a unified pytest suite in `tests/` that checks every backend against PyTorch
(forward AND backward) for both tensor ops and nn modules. Run YOUR slice with (from
the repo root):

    python -m pytest tests/ -k micrograd -v

Read `tests/README.md` for how it works. micrograd is wired in via
`tests/adapters/micrograd_adapter.py`, which declares what micrograd currently
supports:

- OPS currently: `add, mul, matmul, exp, log, relu, softmax, sum_lastdim`
- MODULES currently: `linear, embedding`

Anything not listed is skipped. When you ADD a capability, you MUST also cover it: add
the op/module name to the adapter's `OPS`/`MODULES` set, and if it's a brand-new op not
already in `tests/core/registry.py`, add an `OpSpec` there (a numpy input builder + a
torch reference). Then `python -m pytest tests/ -k micrograd` must stay green. The
adapter already handles the hard parts (nested-list <-> numpy marshalling and a
"union-root seeded backward" that works around `Value.backward()` being scalar-only) —
study it before adding modules.

## What's left (verify each against the suite)
1. **Tensor-level sub and div**: `ops.py` only has `entry_wise_add` / `entry_wise_mult`.
   Add `entry_wise_sub` / `entry_wise_div` (`Value` already supports `__sub__`/
   `__truediv__`), then enable `"sub"`/`"div"` in the adapter `OPS`. Confirm vs torch.
2. **max/min that maintain gradient**: there's no differentiable max/min op (README
   TODO). Add one whose backward routes the gradient to the arg-max/arg-min element.
3. **Reductions**: add `mean` (and optionally `var`/`std`) over the last dim as `ops`
   functions + adapter `OPS`, matching torch's `dim=-1` semantics.
4. **LayerNorm**: `gpt.py` has `layer_norm` as a function. Consider promoting it to a
   `modules.py` `LayerNorm` Module and wiring a `"layernorm"` module test (the adapter's
   `run_module` shows the pattern; weight-sync from the torch reference).
5. **CrossEntropyLoss + inference**: make the model output logits and have
   `CrossEntropyLoss` apply log-softmax internally, so inference can softmax the logits
   (README TODO). Keep `train.py` working.
6. **train.py**: the model is currently too small to learn. Scale it modestly and
   confirm the loss actually decreases on `input.txt`; greedy-sample a few characters to
   sanity check.

## Rules
- Keep micrograd scalar/nested-list and dependency-light — it's the "from scratch"
  beginner lesson. No numpy/torch in `engine.py`/`ops.py`/`modules.py`/`gpt.py`.
- Don't break other backends: after any change to shared files
  (`tests/core/registry.py`, `tests/adapters/torch_ref.py`), run the FULL suite
  `python -m pytest tests/` and keep it green (other backends may be skipped if their
  hardware is absent — that's fine).
- Commit on the `finish-micrograd` branch.

## Done =
micrograd implementation complete and correct, `train.py` demonstrably learns on
`input.txt`, and `python -m pytest tests/ -k micrograd` is green with the new
ops/modules covered (no silent skips for things you implemented).
