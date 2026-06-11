"""Registry of all backend adapters.

To add a new implementation (e.g. cutilegrad, cudagrad), write a new
``<name>_adapter.py`` in this folder exposing an ``AdapterABC`` subclass and
append it to ``ADAPTERS`` below. Nothing else in the test suite needs to change.
"""
from tests.adapters.torch_ref import TorchAdapter
from tests.adapters.micrograd_adapter import MicrogradAdapter
from tests.adapters.minigrad_adapter import MinigradAdapter
from tests.adapters.tritongrad_adapter import TritongradAdapter
from tests.adapters.cudagrad_adapter import CudagradAdapter
from tests.adapters.mlxgrad_adapter import MlxgradAdapter

ADAPTERS = [
    TorchAdapter,
    MicrogradAdapter,
    MinigradAdapter,
    TritongradAdapter,
    CudagradAdapter,
    MlxgradAdapter,
]
