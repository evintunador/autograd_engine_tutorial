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

    // elementwise unary
    m.def("unary_forward", &unary_forward,
          "elementwise unary forward (0=exp,1=log,2=relu,3=neg)");
    m.def("unary_backward", &unary_backward,
          "elementwise unary backward (accumulates into dx)");

    // vectorwise: last-dim reductions + softmax
    m.def("reduction_forward", &reduction_forward,
          "last-dim reduction forward (0=sum,1=mean,2=max,3=min,4=var,5=std)");
    m.def("reduction_backward", &reduction_backward,
          "last-dim reduction backward (accumulates into dx)");
    m.def("softmax_forward", &softmax_forward,
          "last-dim numerically-stable softmax forward");
    m.def("softmax_backward", &softmax_backward,
          "last-dim softmax backward (accumulates into dx)");
}
