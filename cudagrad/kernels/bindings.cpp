// The single pybind entry point for the cudagrad CUDA extension.
//
// torch.utils.cpp_extension.load links all of cudagrad's .cu/.cpp sources into
// ONE Python module, so there must be exactly one PYBIND11_MODULE across the
// whole build — this file. Each kernel group keeps its launchers in its own .cu
// file (declared in kernels.h); here we only bind them. A sub-agent adding a new
// kernel group declares its launcher in kernels.h, defines it in its .cu, adds
// the source to cuda_kernels.py's `sources` list, and adds one m.def line below.
#include <torch/extension.h>
#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // elementwise binary
    m.def("binary_forward", &binary_forward,
          "elementwise binary forward (0=add,1=sub,2=mul,3=div)");
    m.def("binary_backward_dx", &binary_backward_dx,
          "elementwise binary backward w.r.t. the first operand");
    m.def("binary_backward_dy", &binary_backward_dy,
          "elementwise binary backward w.r.t. the (broadcast) second operand");
}
