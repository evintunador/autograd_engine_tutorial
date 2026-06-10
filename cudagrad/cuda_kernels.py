"""Python wrappers around cudagrad's compiled CUDA extension (``cudagrad_ext``).

The real math lives as ``.cu``/``.cpp`` sources under ``kernels/`` and is
JIT-compiled by ``torch.utils.cpp_extension.load`` on first use (the first call
is slow — nvcc compiles, just like Triton's first-call JIT). Compiled artifacts
are cached by torch under ``~/.cache/torch_extensions`` (outside the repo).

Why this module is named ``cuda_kernels`` and never ``kernels``: tritongrad's
``engine.py`` does ``from kernels import ...``, which leaks a top-level
``kernels`` module into ``sys.modules`` that is never cleaned up. On a GPU box
where both backends are imported, a cudagrad ``import kernels`` would silently
resolve to *tritongrad's* package. So ``kernels/`` here is a pure source
directory (no ``__init__.py``, never imported as Python) and the build glue +
wrappers live under the unique name ``cuda_kernels`` (extension name
``cudagrad_ext``). This sidesteps the collision entirely — see the project plan.

Each op group gets a tiny wrapper here. Groups not yet implemented raise
``NotImplementedError`` so the engine methods can exist and import cleanly while
their ops simply stay out of the adapter's ``OPS``/``MODULES`` (and thus skip).
"""
import os

from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_KDIR = os.path.join(_HERE, "kernels")

# Source list. Each kernel-group phase appends its .cu file here (bindings.cpp
# always stays first — it's the single pybind entry point).
_SOURCES = [
    os.path.join(_KDIR, "bindings.cpp"),
    os.path.join(_KDIR, "elementwise.cu"),
    # os.path.join(_KDIR, "matmul.cu"),       # matmul phase
    # os.path.join(_KDIR, "vectorwise.cu"),   # reductions + softmax phase
    # os.path.join(_KDIR, "modules.cu"),      # embedding + layernorm phase
]

_ext = None


def _get_ext():
    """Lazily JIT-compile and return the cudagrad CUDA extension module."""
    global _ext
    if _ext is None:
        _ext = load(
            name="cudagrad_ext",
            sources=_SOURCES,
            extra_include_paths=[_KDIR],
            verbose=True,
        )
    return _ext


# --- elementwise binary (add / sub / mul / div) ----------------------------
_BINARY_OP = {"add": 0, "sub": 1, "mul": 2, "div": 3}


def binary_forward(x, y, out, loop_stride, op):
    _get_ext().binary_forward(x, y, out, loop_stride, _BINARY_OP[op])


def binary_backward_dx(y, dx, dout, loop_stride, op):
    _get_ext().binary_backward_dx(y, dx, dout, loop_stride, _BINARY_OP[op])


def binary_backward_dy(x, y, dy, dout, loop_stride, op):
    _get_ext().binary_backward_dy(x, y, dy, dout, loop_stride, _BINARY_OP[op])


# --- not yet implemented (filled in by later kernel phases) ----------------
# Each stub's signature is the suggested contract for the engine call site in
# engine.py; the implementing sub-agent may adjust both sides together.
def unary_forward(x, out, op):
    raise NotImplementedError("unary ops (exp/log/relu/neg) not implemented yet")


def unary_backward(x, dx, out, dout, op):
    raise NotImplementedError("unary ops (exp/log/relu/neg) not implemented yet")


def matmul_forward(a, b, out):
    raise NotImplementedError("matmul not implemented yet")


def matmul_backward_dA(b, dA, dout):
    raise NotImplementedError("matmul not implemented yet")


def matmul_backward_dB(a, dB, dout):
    raise NotImplementedError("matmul not implemented yet")


def reduction_forward(x, out, n_rows, n_cols, op):
    raise NotImplementedError("reductions (sum/mean/var/std/max/min) not implemented yet")


def reduction_backward(x, dx, dout, out, n_rows, n_cols, op):
    raise NotImplementedError("reductions (sum/mean/var/std/max/min) not implemented yet")


def softmax_forward(x, out, n_rows, n_cols):
    raise NotImplementedError("softmax not implemented yet")


def softmax_backward(out, dx, dout, n_rows, n_cols):
    raise NotImplementedError("softmax not implemented yet")


def embedding_forward(*args, **kwargs):
    raise NotImplementedError("embedding not implemented yet")


def embedding_backward(*args, **kwargs):
    raise NotImplementedError("embedding not implemented yet")


def layernorm_forward(*args, **kwargs):
    raise NotImplementedError("layernorm not implemented yet")


def layernorm_backward(*args, **kwargs):
    raise NotImplementedError("layernorm not implemented yet")


def flash_attention_forward(*args, **kwargs):
    raise NotImplementedError("flash attention not implemented yet")


def flash_attention_backward(*args, **kwargs):
    raise NotImplementedError("flash attention not implemented yet")
