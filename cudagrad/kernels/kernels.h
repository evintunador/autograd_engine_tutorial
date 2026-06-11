// Declarations of every CUDA launcher exposed to Python through the
// `cudagrad_ext` extension. Each kernel group (elementwise, matmul, vectorwise,
// modules, ...) defines its launchers in its own .cu file and declares them
// here; bindings.cpp binds them all into one pybind module.
//
// Convention: launchers take already-allocated, contiguous, fp32 CUDA tensors
// and write in place. Backward launchers ACCUMULATE into grad tensors (`+=` /
// atomicAdd), so callers must pass grad buffers that start at zero — mirroring
// tritongrad's accumulate-into-.grad backward kernels.
#pragma once
#include <torch/extension.h>

// ---- elementwise binary (op: 0=add, 1=sub, 2=mul, 3=div) ------------------
// y is broadcast up to x via modulo indexing (i % loop_stride), matching the
// `loop_stride` broadcasting contract in tritongrad's binary kernels.
void binary_forward(torch::Tensor x, torch::Tensor y, torch::Tensor out,
                    int64_t loop_stride, int64_t op);
void binary_backward_dx(torch::Tensor y, torch::Tensor dx, torch::Tensor dout,
                        int64_t loop_stride, int64_t op);
void binary_backward_dy(torch::Tensor x, torch::Tensor y, torch::Tensor dy,
                        torch::Tensor dout, int64_t loop_stride, int64_t op);

// ---- elementwise unary (op: 0=exp, 1=log, 2=relu, 3=neg) ------------------
// backward ACCUMULATES into dx (`+=`), so callers pass a zero-initialized dx.
void unary_forward(torch::Tensor x, torch::Tensor out, int64_t op);
void unary_backward(torch::Tensor x, torch::Tensor dx, torch::Tensor out,
                    torch::Tensor dout, int64_t op);

// ---- (future kernel groups declare their launchers below) -----------------
// matmul (fwd / bwd_dA / bwd_dB) -> matmul.cu
// reduction + softmax            -> vectorwise.cu
// embedding + layernorm          -> modules.cu
