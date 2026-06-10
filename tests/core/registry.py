"""Backend-agnostic declaration of every op and module under test.

Each entry is declared ONCE here, in terms of numpy input builders and a torch
reference. Adapters never appear in this file; they opt into ops/modules via
their ``OPS`` / ``MODULES`` name sets. Adding a new backend therefore requires no
edits here.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import math

import numpy as np
import torch


# --- seeded input builders -------------------------------------------------
def _std(g, shape):
    return g.standard_normal(shape).astype(np.float32)


def _pos(g, shape):
    """Strictly-positive inputs (for log / safe division divisors)."""
    return (np.abs(g.standard_normal(shape)) + 0.5).astype(np.float32)


# standard 3D test shape (batch, seq, dim)
S = (2, 8, 16)


# --- ops -------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    name: str
    make_inputs: Callable          # (rng) -> list[np.ndarray]
    torch_fn: Callable             # (*torch.Tensor) -> torch.Tensor
    atol: float = 1e-3
    rtol: float = 1e-3


OP_REGISTRY = [
    OpSpec("add", lambda g: [_std(g, S), _std(g, S)], lambda a, b: a + b),
    OpSpec("sub", lambda g: [_std(g, S), _std(g, S)], lambda a, b: a - b),
    OpSpec("mul", lambda g: [_std(g, S), _std(g, S)], lambda a, b: a * b),
    OpSpec("div", lambda g: [_std(g, S), _pos(g, S)], lambda a, b: a / b),
    OpSpec("matmul",
           lambda g: [_std(g, (2, 8, 16)), _std(g, (2, 16, 8))],
           lambda a, b: a @ b),
    OpSpec("exp", lambda g: [_std(g, S)], torch.exp),
    OpSpec("log", lambda g: [_pos(g, S)], torch.log),
    OpSpec("relu", lambda g: [_std(g, S)], torch.relu),
    OpSpec("neg", lambda g: [_std(g, S)], lambda x: -x),
    OpSpec("softmax", lambda g: [_std(g, S)], lambda x: torch.softmax(x, dim=-1)),
    OpSpec("sum_lastdim", lambda g: [_std(g, S)], lambda x: x.sum(dim=-1)),
    OpSpec("mean", lambda g: [_std(g, S)], lambda x: x.mean(dim=-1)),
    OpSpec("var", lambda g: [_std(g, S)], lambda x: x.var(dim=-1, unbiased=False)),
    OpSpec("std", lambda g: [_std(g, S)], lambda x: x.std(dim=-1, unbiased=False)),
    OpSpec("max_lastdim", lambda g: [_std(g, S)], lambda x: x.max(dim=-1).values),
    OpSpec("min_lastdim", lambda g: [_std(g, S)], lambda x: x.min(dim=-1).values),
]


# --- modules ---------------------------------------------------------------
@dataclass(frozen=True)
class ModuleSpec:
    name: str
    config: dict
    make_inputs: Callable          # (rng) -> list[np.ndarray] (floats and/or int tokens)
    param_names: tuple             # canonical torch param names to compare
    build_torch: Optional[Callable] = None  # (config) -> torch.nn.Module
    kind: str = "standard"         # "standard" | "attention"
    atol: float = 1e-3
    rtol: float = 1e-3


_LIN = {"in": 16, "out": 8}
_EMB = {"num": 32, "dim": 8, "B": 2, "N": 5}
_LN = {"dim": 16}
_ATTN = {"B": 1, "H": 2, "N": 128, "D": 32, "scale": math.sqrt(32)}

MODULE_REGISTRY = [
    ModuleSpec(
        "linear", _LIN,
        lambda g: [_std(g, (2, 8, _LIN["in"]))],
        param_names=("weight", "bias"),
        build_torch=lambda c: torch.nn.Linear(c["in"], c["out"], bias=True),
    ),
    ModuleSpec(
        "embedding", _EMB,
        lambda g: [g.integers(0, _EMB["num"], size=(_EMB["B"], _EMB["N"])).astype(np.int64)],
        param_names=("weight",),
        build_torch=lambda c: torch.nn.Embedding(c["num"], c["dim"]),
    ),
    ModuleSpec(
        "layernorm", _LN,
        lambda g: [_std(g, (2, 8, _LN["dim"]))],
        param_names=("weight", "bias"),
        build_torch=lambda c: torch.nn.LayerNorm(c["dim"], elementwise_affine=True),
    ),
    ModuleSpec(
        "attention", _ATTN,
        lambda g: [(_std(g, (_ATTN["B"], _ATTN["H"], _ATTN["N"], _ATTN["D"])) * 0.02)
                   for _ in range(3)],
        param_names=(),
        build_torch=None,
        kind="attention",
        atol=2e-3, rtol=1e-1,
    ),
]
