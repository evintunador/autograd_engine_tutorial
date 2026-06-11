// Elementwise CUDA kernels for cudagrad.
//
// Binary ops (add/sub/mul/div) and unary ops (exp/log/relu/neg), each wired
// fwd + bwd so the test harness exercises them against PyTorch.
//
// Layout/contract notes:
//   * all tensors are contiguous fp32 on the same CUDA device.
//   * `y` (the second operand) has `loop_stride` elements and is broadcast up to
//     `x`'s n elements via `i % loop_stride`, matching tritongrad's binary
//     kernels. For the suite's same-shape case loop_stride == n.
//   * backward launchers ACCUMULATE: dx uses `+=`, dy uses atomicAdd (so the
//     broadcast case sums contributions correctly). Grad buffers start at zero.
//
// Performance strategy (memory-bound -> we want max effective bandwidth):
//   1. Grid-stride loops. Each kernel processes elements in a `gridStride`
//      pattern so a fixed-size launch covers any n with good occupancy and a
//      bit of instruction-level parallelism, instead of one-element-per-thread.
//   2. float4 vectorized loads/stores. For the common contiguous case we load
//      4 floats per thread as a single 128-bit transaction, which roughly
//      doubles achieved bandwidth vs scalar 32-bit accesses.
//
// float4 is only correct under guards (checked at runtime in the launchers):
//   * every pointer touched by the vector path must be 16-byte aligned, and
//   * n must be a multiple of 4 (so there is no ragged tail to handle).
//   * for binary ops the vector path additionally requires loop_stride == n
//     (no broadcast): under broadcast `y[i % loop_stride]` is not a contiguous
//     vector load. The dy kernel always broadcasts (atomicAdd into a modulo
//     index), so it never vectorizes.
// When any guard fails we fall back to the scalar grid-stride kernels, which
// are always valid for any n / alignment / broadcast. The launchers pick.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 256;
// Cap the grid so each launch is a fixed size and threads loop via grid stride.
// 65535 blocks * 256 threads is plenty to saturate any current GPU.
constexpr int64_t MAX_BLOCKS = 65535;

inline int64_t n_blocks(int64_t n) {
    int64_t b = (n + THREADS - 1) / THREADS;
    return b < 1 ? 1 : (b > MAX_BLOCKS ? MAX_BLOCKS : b);
}

// True iff `p` is 16-byte aligned (required for float4 / 128-bit accesses).
inline bool aligned16(const void* p) { return ((uintptr_t)p & 0xF) == 0; }

// Apply one binary op to a scalar pair. __forceinline__ so the switch is
// hoisted/specialized at the call site and shared by scalar + vector paths.
__device__ __forceinline__ float bin_op(float a, float b, int op) {
    switch (op) {
        case 0: return a + b;  // add
        case 1: return a - b;  // sub
        case 2: return a * b;  // mul
        default: return a / b; // div
    }
}

// ---- binary forward -------------------------------------------------------
// out[i] = x[i] OP y[i % loop_stride]   (scalar grid-stride; always valid)
__global__ void binary_forward_scalar(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       float* __restrict__ out,
                                       int64_t n, int64_t loop_stride, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = bin_op(x[i], y[i % loop_stride], op);
    }
}

// Vectorized: 4 contiguous floats per thread. Only launched when aligned,
// n % 4 == 0, and loop_stride == n (no broadcast, so y is read contiguously).
// We index over nvec = n/4 vector slots in a grid-stride loop.
__global__ void binary_forward_vec4(const float4* __restrict__ x,
                                    const float4* __restrict__ y,
                                    float4* __restrict__ out,
                                    int64_t nvec, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nvec; i += stride) {
        float4 a = x[i];
        float4 b = y[i];
        float4 r;
        r.x = bin_op(a.x, b.x, op);
        r.y = bin_op(a.y, b.y, op);
        r.z = bin_op(a.z, b.z, op);
        r.w = bin_op(a.w, b.w, op);
        out[i] = r;
    }
}

// ---- binary backward dx ---------------------------------------------------
// dx[i] += d(out)/d(x[i]) * dout[i]
//   add/sub: dx += dout ;  mul: dx += dout * y ;  div: dx += dout / y
// Each thread owns a distinct dx[i] (no race), so `+=` is safe.
__device__ __forceinline__ float bin_dx(float g, float yv, int op) {
    switch (op) {
        case 0: return g;        // add: d/dx = 1
        case 1: return g;        // sub: d/dx = 1
        case 2: return g * yv;   // mul: d/dx = y
        default: return g / yv;  // div: d/dx = 1/y
    }
}

__global__ void binary_backward_dx_scalar(const float* __restrict__ y,
                                          float* __restrict__ dx,
                                          const float* __restrict__ dout,
                                          int64_t n, int64_t loop_stride, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dx[i] += bin_dx(dout[i], y[i % loop_stride], op);
    }
}

// Vectorized dx: valid when aligned, n % 4 == 0, loop_stride == n (so y[i] is
// contiguous, matching dx[i]/dout[i]). Loads dx4, accumulates, stores.
__global__ void binary_backward_dx_vec4(const float4* __restrict__ y,
                                        float4* __restrict__ dx,
                                        const float4* __restrict__ dout,
                                        int64_t nvec, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nvec; i += stride) {
        float4 g = dout[i];
        float4 yv = y[i];
        float4 acc = dx[i];
        acc.x += bin_dx(g.x, yv.x, op);
        acc.y += bin_dx(g.y, yv.y, op);
        acc.z += bin_dx(g.z, yv.z, op);
        acc.w += bin_dx(g.w, yv.w, op);
        dx[i] = acc;
    }
}

// ---- binary backward dy ---------------------------------------------------
// dy[i % loop_stride] += d(out)/d(y) * dout[i]  (atomic: broadcast accumulation)
//   add: +dout ; sub: -dout ; mul: +dout*x ; div: -x*dout/y^2
// Always scalar: the destination is a modulo index requiring atomicAdd, which
// is not vectorizable.
__global__ void binary_backward_dy_scalar(const float* __restrict__ x,
                                          const float* __restrict__ y,
                                          float* __restrict__ dy,
                                          const float* __restrict__ dout,
                                          int64_t n, int64_t loop_stride, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        int64_t j = i % loop_stride;
        float g = dout[i];
        float contrib;
        switch (op) {
            case 0: contrib = g; break;        // add: d/dy = 1
            case 1: contrib = -g; break;       // sub: d/dy = -1
            case 2: contrib = g * x[i]; break; // mul: d/dy = x
            default: {                         // div: d/dy = -x / y^2
                float yv = y[j];
                contrib = -x[i] * g / (yv * yv);
                break;
            }
        }
        atomicAdd(&dy[j], contrib);
    }
}

// ---- unary (op: 0=exp, 1=log, 2=relu, 3=neg) ------------------------------
__device__ __forceinline__ float un_op(float a, int op) {
    switch (op) {
        case 0: return expf(a);        // exp
        case 1: return logf(a);        // log
        case 2: return fmaxf(a, 0.0f); // relu
        default: return -a;            // neg
    }
}

// out[i] = UNARY_OP(x[i])   (scalar grid-stride; always valid)
__global__ void unary_forward_scalar(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     int64_t n, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = un_op(x[i], op);
    }
}

// Vectorized unary forward: valid when aligned and n % 4 == 0.
__global__ void unary_forward_vec4(const float4* __restrict__ x,
                                   float4* __restrict__ out,
                                   int64_t nvec, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nvec; i += stride) {
        float4 a = x[i];
        float4 r;
        r.x = un_op(a.x, op);
        r.y = un_op(a.y, op);
        r.z = un_op(a.z, op);
        r.w = un_op(a.w, op);
        out[i] = r;
    }
}

// dx[i] += d(out)/d(x[i]) * dout[i]   (ACCUMULATES)
//   exp: out * dout ;  log: dout / x ;  relu: (out>0)?dout:0 ;  neg: -dout
__device__ __forceinline__ float un_dx(float xv, float outv, float g, int op) {
    switch (op) {
        case 0: return outv * g;                 // exp
        case 1: return g / xv;                   // log
        case 2: return (outv > 0.0f) ? g : 0.0f; // relu
        default: return -g;                      // neg
    }
}

__global__ void unary_backward_scalar(const float* __restrict__ x,
                                      float* __restrict__ dx,
                                      const float* __restrict__ out,
                                      const float* __restrict__ dout,
                                      int64_t n, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dx[i] += un_dx(x[i], out[i], dout[i], op);
    }
}

// Vectorized unary backward: valid when aligned and n % 4 == 0. Each thread
// owns distinct dx indices, so read-modify-write the dx4 slot is race-free.
__global__ void unary_backward_vec4(const float4* __restrict__ x,
                                    float4* __restrict__ dx,
                                    const float4* __restrict__ out,
                                    const float4* __restrict__ dout,
                                    int64_t nvec, int op) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nvec; i += stride) {
        float4 xv = x[i];
        float4 ov = out[i];
        float4 g = dout[i];
        float4 acc = dx[i];
        acc.x += un_dx(xv.x, ov.x, g.x, op);
        acc.y += un_dx(xv.y, ov.y, g.y, op);
        acc.z += un_dx(xv.z, ov.z, g.z, op);
        acc.w += un_dx(xv.w, ov.w, g.w, op);
        dx[i] = acc;
    }
}

} // namespace

void binary_forward(torch::Tensor x, torch::Tensor y, torch::Tensor out,
                    int64_t loop_stride, int64_t op) {
    int64_t n = x.numel();
    const float* xp = x.data_ptr<float>();
    const float* yp = y.data_ptr<float>();
    float* op_ = out.data_ptr<float>();
    // Fast path: no broadcast, divisible by 4, all pointers 16-byte aligned.
    if (loop_stride == n && (n & 3) == 0 &&
        aligned16(xp) && aligned16(yp) && aligned16(op_)) {
        int64_t nvec = n / 4;
        binary_forward_vec4<<<n_blocks(nvec), THREADS>>>(
            reinterpret_cast<const float4*>(xp),
            reinterpret_cast<const float4*>(yp),
            reinterpret_cast<float4*>(op_), nvec, (int)op);
    } else {
        binary_forward_scalar<<<n_blocks(n), THREADS>>>(
            xp, yp, op_, n, loop_stride, (int)op);
    }
}

void binary_backward_dx(torch::Tensor y, torch::Tensor dx, torch::Tensor dout,
                        int64_t loop_stride, int64_t op) {
    int64_t n = dx.numel();
    const float* yp = y.data_ptr<float>();
    float* dxp = dx.data_ptr<float>();
    const float* doutp = dout.data_ptr<float>();
    // Fast path: no broadcast (y[i] contiguous), divisible by 4, all aligned.
    if (loop_stride == n && (n & 3) == 0 &&
        aligned16(yp) && aligned16(dxp) && aligned16(doutp)) {
        int64_t nvec = n / 4;
        binary_backward_dx_vec4<<<n_blocks(nvec), THREADS>>>(
            reinterpret_cast<const float4*>(yp),
            reinterpret_cast<float4*>(dxp),
            reinterpret_cast<const float4*>(doutp), nvec, (int)op);
    } else {
        binary_backward_dx_scalar<<<n_blocks(n), THREADS>>>(
            yp, dxp, doutp, n, loop_stride, (int)op);
    }
}

void binary_backward_dy(torch::Tensor x, torch::Tensor y, torch::Tensor dy,
                        torch::Tensor dout, int64_t loop_stride, int64_t op) {
    int64_t n = x.numel();
    // Always scalar: atomicAdd into a broadcast (modulo) index is not
    // vectorizable. Grid-stride still gives good occupancy.
    binary_backward_dy_scalar<<<n_blocks(n), THREADS>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), dy.data_ptr<float>(),
        dout.data_ptr<float>(), n, loop_stride, (int)op);
}

void unary_forward(torch::Tensor x, torch::Tensor out, int64_t op) {
    int64_t n = x.numel();
    const float* xp = x.data_ptr<float>();
    float* op_ = out.data_ptr<float>();
    if ((n & 3) == 0 && aligned16(xp) && aligned16(op_)) {
        int64_t nvec = n / 4;
        unary_forward_vec4<<<n_blocks(nvec), THREADS>>>(
            reinterpret_cast<const float4*>(xp),
            reinterpret_cast<float4*>(op_), nvec, (int)op);
    } else {
        unary_forward_scalar<<<n_blocks(n), THREADS>>>(xp, op_, n, (int)op);
    }
}

void unary_backward(torch::Tensor x, torch::Tensor dx, torch::Tensor out,
                    torch::Tensor dout, int64_t op) {
    int64_t n = x.numel();
    const float* xp = x.data_ptr<float>();
    float* dxp = dx.data_ptr<float>();
    const float* outp = out.data_ptr<float>();
    const float* doutp = dout.data_ptr<float>();
    if ((n & 3) == 0 && aligned16(xp) && aligned16(dxp) &&
        aligned16(outp) && aligned16(doutp)) {
        int64_t nvec = n / 4;
        unary_backward_vec4<<<n_blocks(nvec), THREADS>>>(
            reinterpret_cast<const float4*>(xp),
            reinterpret_cast<float4*>(dxp),
            reinterpret_cast<const float4*>(outp),
            reinterpret_cast<const float4*>(doutp), nvec, (int)op);
    } else {
        unary_backward_scalar<<<n_blocks(n), THREADS>>>(
            xp, dxp, outp, doutp, n, (int)op);
    }
}
