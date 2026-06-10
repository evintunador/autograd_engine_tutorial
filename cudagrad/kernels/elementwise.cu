// Elementwise CUDA kernels for cudagrad.
//
// This is the scaffold's worked example: the binary op (add/sub/mul/div) wired
// fwd + bwd so the test harness has a first real op to exercise. The unary ops
// (exp/log/relu/neg) are left for the "elementwise rest" phase — add their
// kernels + launchers here, declare them in kernels.h, and bind them in
// bindings.cpp.
//
// Layout/contract notes:
//   * all tensors are contiguous fp32 on the same CUDA device.
//   * `y` (the second operand) has `loop_stride` elements and is broadcast up to
//     `x`'s n elements via `i % loop_stride`, matching tritongrad's binary
//     kernels. For the suite's same-shape `add` case loop_stride == n.
//   * backward launchers ACCUMULATE: dx uses `+=`, dy uses atomicAdd (so the
//     broadcast case sums contributions correctly). Grad buffers start at zero.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 256;

inline int64_t n_blocks(int64_t n) { return (n + THREADS - 1) / THREADS; }

// out[i] = x[i] OP y[i % loop_stride]
__global__ void binary_forward_kernel(const float* __restrict__ x,
                                      const float* __restrict__ y,
                                      float* __restrict__ out,
                                      int64_t n, int64_t loop_stride, int op) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = x[i];
    float b = y[i % loop_stride];
    float r;
    switch (op) {
        case 0: r = a + b; break;  // add
        case 1: r = a - b; break;  // sub
        case 2: r = a * b; break;  // mul
        default: r = a / b; break; // div
    }
    out[i] = r;
}

// dx[i] += d(out)/d(x[i]) * dout[i]
//   add/sub: dx += dout ;  mul: dx += dout * y ;  div: dx += dout / y
__global__ void binary_backward_dx_kernel(const float* __restrict__ y,
                                          float* __restrict__ dx,
                                          const float* __restrict__ dout,
                                          int64_t n, int64_t loop_stride, int op) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = dout[i];
    float contrib;
    switch (op) {
        case 0: contrib = g; break;                       // add: d/dx = 1
        case 1: contrib = g; break;                       // sub: d/dx = 1
        case 2: contrib = g * y[i % loop_stride]; break;  // mul: d/dx = y
        default: contrib = g / y[i % loop_stride]; break; // div: d/dx = 1/y
    }
    dx[i] += contrib;
}

// dy[i % loop_stride] += d(out)/d(y) * dout[i]  (atomic: broadcast accumulation)
//   add: +dout ; sub: -dout ; mul: +dout*x ; div: -x*dout/y^2
__global__ void binary_backward_dy_kernel(const float* __restrict__ x,
                                          const float* __restrict__ y,
                                          float* __restrict__ dy,
                                          const float* __restrict__ dout,
                                          int64_t n, int64_t loop_stride, int op) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t j = i % loop_stride;
    float g = dout[i];
    float contrib;
    switch (op) {
        case 0: contrib = g; break;            // add: d/dy = 1
        case 1: contrib = -g; break;           // sub: d/dy = -1
        case 2: contrib = g * x[i]; break;     // mul: d/dy = x
        default: {                             // div: d/dy = -x / y^2
            float yv = y[j];
            contrib = -x[i] * g / (yv * yv);
            break;
        }
    }
    atomicAdd(&dy[j], contrib);
}

} // namespace

void binary_forward(torch::Tensor x, torch::Tensor y, torch::Tensor out,
                    int64_t loop_stride, int64_t op) {
    int64_t n = x.numel();
    binary_forward_kernel<<<n_blocks(n), THREADS>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(),
        n, loop_stride, (int)op);
}

void binary_backward_dx(torch::Tensor y, torch::Tensor dx, torch::Tensor dout,
                        int64_t loop_stride, int64_t op) {
    int64_t n = dx.numel();
    binary_backward_dx_kernel<<<n_blocks(n), THREADS>>>(
        y.data_ptr<float>(), dx.data_ptr<float>(), dout.data_ptr<float>(),
        n, loop_stride, (int)op);
}

void binary_backward_dy(torch::Tensor x, torch::Tensor y, torch::Tensor dy,
                        torch::Tensor dout, int64_t loop_stride, int64_t op) {
    int64_t n = x.numel();
    binary_backward_dy_kernel<<<n_blocks(n), THREADS>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), dy.data_ptr<float>(),
        dout.data_ptr<float>(), n, loop_stride, (int)op);
}
