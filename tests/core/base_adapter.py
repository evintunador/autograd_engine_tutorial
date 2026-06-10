"""Common interface every backend adapter implements.

The test suite is written once against this interface and parametrized across all
adapters in ``tests/adapters``. A backend declares which ops/modules it supports
via the ``OPS`` / ``MODULES`` name sets; anything not listed is cleanly skipped.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GraphHandle:
    """Opaque carrier returned by ``forward_op``: the backend's output node plus
    the leaf input nodes, so the adapter can seed grads and read them back."""
    output: Any
    inputs: list


@dataclass
class ModuleResult:
    """Everything a module test needs from a backend, all marshalled to numpy in
    canonical (torch) layout so the test can compare without backend knowledge."""
    out: np.ndarray
    param_grads: dict          # param name -> grad (torch layout)
    input_grads: dict = field(default_factory=dict)  # input index -> grad


class AdapterABC(ABC):
    name: str = "abstract"
    OPS: set = set()           # tensor-op names this backend supports
    MODULES: set = set()       # nn-module names this backend supports
    tol_overrides: dict = {}   # op/module name -> {"atol":.., "rtol":..}
    # op/module name -> reason. Known-broken-but-implemented: the case still runs
    # but is recorded as an expected failure (a green run that documents a bug).
    xfail_ops: dict = {}
    xfail_modules: dict = {}

    # ---- availability -----------------------------------------------------
    @classmethod
    @abstractmethod
    def available(cls):
        """Return ``(is_available, reason_if_not)``. Guards optional imports
        (e.g. CUDA) so unavailable backends turn into a clean pytest skip."""

    # ---- data marshalling -------------------------------------------------
    @abstractmethod
    def from_numpy(self, arr: np.ndarray, requires_grad: bool):
        """np.ndarray -> backend leaf tensor."""

    @abstractmethod
    def to_numpy(self, x) -> np.ndarray:
        """backend tensor -> np.ndarray of its ``.data``."""

    @abstractmethod
    def grad_of(self, leaf) -> np.ndarray:
        """np.ndarray of the gradient accumulated on a leaf input."""

    # ---- ops --------------------------------------------------------------
    @abstractmethod
    def forward_op(self, op_name: str, inputs: list) -> GraphHandle:
        """Run the named op forward; return the output + leaf inputs."""

    @abstractmethod
    def backward(self, handle: GraphHandle, grad_output: np.ndarray) -> None:
        """Seed ``handle.output``'s grad with ``grad_output`` and propagate, so
        ``grad_of(each input)`` becomes readable. Must NOT reseed to ones."""

    # ---- modules (optional) ----------------------------------------------
    def run_module(self, spec, ref_params: dict, input_arrays: list,
                   grad_output: np.ndarray) -> ModuleResult:
        """Instantiate the module, load ``ref_params`` (canonical torch layout),
        run fwd + seeded bwd, and return outputs/grads in torch layout."""
        raise NotImplementedError

    # ---- capability helpers (concrete) -----------------------------------
    def supports_op(self, name: str) -> bool:
        return name in self.OPS

    def supports_module(self, name: str) -> bool:
        return name in self.MODULES

    def tol_for(self, name: str, default_atol: float, default_rtol: float):
        o = self.tol_overrides.get(name, {})
        return o.get("atol", default_atol), o.get("rtol", default_rtol)
