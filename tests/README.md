# Unified test suite

One pytest suite that checks **every** autograd implementation in this repo
(`micrograd`, `minigrad`, `tritongrad`, …) against **PyTorch** as the reference,
for both tensor ops and nn modules — forward *and* backward.

## Running

```bash
pip install -r requirements.txt           # (adds pytest); run from the repo root
pytest tests/                              # everything available on this machine
pytest tests/ -k minigrad                  # one backend
pytest tests/test_ops.py -k "minigrad and matmul"   # one (backend, op)
pytest tests/ --heatmaps                   # on failure, write diff PNGs to tests/heatmaps/
pytest tests/ --seed 7                      # change the input RNG seed
```

Backends that need hardware you don't have are **skipped**, not failed:
`tritongrad` requires a CUDA GPU + Triton, so on a Mac/CPU box its cases skip with
`CUDA device not available`. Run on a GPU host to exercise them
(`pytest tests/ -k tritongrad`).

## How it's structured

- `core/registry.py` — every op and module declared **once**, backend-agnostic:
  a numpy input-builder + a torch reference. No backend appears here.
- `core/base_adapter.py` — the `AdapterABC` interface each backend implements.
- `adapters/*.py` — one shim per backend, normalizing its API to the interface.
- `test_ops.py` / `test_modules.py` — parametrized over `(adapter × op/module)`;
  unsupported combinations skip, known-broken ones `xfail`.
- `test_meta.py` — sanity checks on the harness itself.

The reference is PyTorch in **fp32** (all backends are fp32). Inputs are seeded so
every backend sees identical arrays. Comparison is `np.testing.assert_allclose`
with per-op tolerances (loosened for matmul/linear/flash-attention, matching
`tritongrad/testing.py`).

## Adding a new backend (e.g. `cutilegrad`, `cudagrad`)

1. Write `adapters/cutilegrad_adapter.py` with a `class CutilegradAdapter(AdapterABC)`:
   - `available()` → `(bool, reason)` (guard any optional/hardware imports here so
     the suite skips cleanly when it can't run).
   - `from_numpy` / `to_numpy` / `grad_of` — marshal arrays in/out.
   - `forward_op(op_name, inputs)` and `backward(handle, grad_output)` — note
     `backward` must **seed** the output with the given gradient, not reseed ones.
   - `OPS` / `MODULES` — the names from the registry this backend supports.
   - optional: `run_module(...)` for nn modules, `tol_overrides`,
     `xfail_ops` / `xfail_modules` for documented-but-unfixed bugs.
2. Append it to `ADAPTERS` in `adapters/__init__.py`.

That's the only change — the registry and test files are untouched.

## Known issues this suite currently documents (xfail)

- `minigrad` `sum` / `mean` / `var` over the last dim (`keepdim=False`): the
  backward in `minigrad/engine.py` broadcasts the upstream grad to the input shape
  without re-inserting the reduced axis, so it raises. Forward is correct.

## Note on the old inline tests

The `if __name__ == "__main__"` blocks in the backend source files remain as
runnable pedagogical demos. `tritongrad/testing.py` (the original GPU harness with
heatmaps) is superseded by this suite for *verification*; its heatmap logic was
ported to `tests/core/heatmaps.py`.
