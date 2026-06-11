// Module CUDA kernels for cudagrad: Embedding + LayerNorm (fwd + bwd).
//
// Mirrors tritongrad/kernels/modules.py (the GPU-verified reference) for the
// exact math. Simplicity over peak perf (tutorial): ONE THREAD PER ROW (or per
// output element). Test sizes are tiny, so plain looping is fine.
//
// Layout/contract notes:
//   * float tensors are contiguous fp32; `tokens` is contiguous int64.
//   * backward launchers ACCUMULATE into grad buffers (`+=` / atomicAdd), so
//     callers pass grad buffers that start at zero. Where multiple rows write
//     the SAME grad element (embedding scatter-add into a shared embedding row;
//     layernorm dweight/dbias summed across rows) atomicAdd is REQUIRED.
//   * layernorm var uses POPULATION (/D) normalization (subtracting the row
//     mean), matching torch.nn.LayerNorm; mean[r]/rstd[r] are computed in
//     forward and reused in backward.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 256;

inline int64_t n_blocks(int64_t n) { return (n + THREADS - 1) / THREADS; }

// ---- embedding ------------------------------------------------------------
// out[row, d] = weight[tokens[row], d]   (one thread per (row, d) element)
// tokens is viewed flat as `rows` ids; out/weight are (rows, D)/(V, D).
// NOTE: the engine is fp32-only (CudaTensor.__init__ forces float32), so token
// ids arrive as FLOATS — we cast back to int for indexing, exactly like
// tritongrad. Ids are exact small integers (< 2^24), so the cast is lossless;
// they're range-checked in nn.Embedding before we get here.
__global__ void embedding_forward_kernel(const float* __restrict__ tokens,
                                        const float* __restrict__ weight,
                                        float* __restrict__ out,
                                        int64_t rows, int64_t D, int64_t V) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * D) return;
    int64_t row = i / D;
    int64_t d = i % D;
    int64_t t = (int64_t)llroundf(tokens[row]);
    out[row * D + d] = weight[t * D + d];
}

// dweight[tokens[row], d] += dout[row, d]   (scatter-add; one thread per (row,d))
// Multiple rows can share a token id, so the accumulation into the shared
// embedding row MUST be atomic. (tokens float -> int, as in the forward.)
__global__ void embedding_backward_kernel(const float* __restrict__ tokens,
                                         float* __restrict__ dweight,
                                         const float* __restrict__ dout,
                                         int64_t rows, int64_t D, int64_t V) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * D) return;
    int64_t row = i / D;
    int64_t d = i % D;
    int64_t t = (int64_t)llroundf(tokens[row]);
    atomicAdd(&dweight[t * D + d], dout[row * D + d]);
}

// ---- layernorm ------------------------------------------------------------
// out[r, :] = ((x[r,:] - mean) * rstd) * w + b ; stores mean[r], rstd[r].
// ONE THREAD PER ROW (each thread owns its row; no races).
__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        const float* __restrict__ b,
                                        float* __restrict__ out,
                                        float* __restrict__ mean,
                                        float* __restrict__ rstd,
                                        int64_t rows, int64_t D, float eps) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xrow = x + r * D;
    float* orow = out + r * D;

    float s = 0.0f;
    for (int64_t d = 0; d < D; ++d) s += xrow[d];
    float mu = s / (float)D;

    float acc = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        float diff = xrow[d] - mu;
        acc += diff * diff;
    }
    float var = acc / (float)D;  // population (/D) normalization
    float rs = 1.0f / sqrtf(var + eps);

    mean[r] = mu;
    rstd[r] = rs;

    for (int64_t d = 0; d < D; ++d) {
        float xhat = (xrow[d] - mu) * rs;
        orow[d] = xhat * w[d] + b[d];
    }
}

// dx (accumulate, per-row, no race), dweight/dbias (atomicAdd across rows).
//   dxhat[d] = dout[r,d]*w[d];  c1 = (1/D) Σ dxhat;  c2 = (1/D) Σ dxhat*xhat;
//   dx[r,d] += rstd[r] * (dxhat[d] - c1 - xhat[r,d]*c2)
//   dweight[d] += dout[r,d]*xhat[r,d] ;  dbias[d] += dout[r,d]
// ONE THREAD PER ROW. Uses mean[r]/rstd[r] saved in forward.
__global__ void layernorm_backward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         float* __restrict__ dx,
                                         const float* __restrict__ dout,
                                         float* __restrict__ dw,
                                         float* __restrict__ db,
                                         const float* __restrict__ mean,
                                         const float* __restrict__ rstd,
                                         int64_t rows, int64_t D) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xrow = x + r * D;
    const float* grow = dout + r * D;
    float* dxrow = dx + r * D;
    float mu = mean[r];
    float rs = rstd[r];

    // first pass: c1 = mean(dxhat), c2 = mean(dxhat * xhat)
    float c1 = 0.0f, c2 = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        float xhat = (xrow[d] - mu) * rs;
        float dxhat = grow[d] * w[d];
        c1 += dxhat;
        c2 += dxhat * xhat;
    }
    c1 /= (float)D;
    c2 /= (float)D;

    // second pass: dx (per-row, no race) + dweight/dbias (atomic across rows)
    for (int64_t d = 0; d < D; ++d) {
        float xhat = (xrow[d] - mu) * rs;
        float dxhat = grow[d] * w[d];
        dxrow[d] += rs * (dxhat - c1 - xhat * c2);
        atomicAdd(&dw[d], grow[d] * xhat);
        atomicAdd(&db[d], grow[d]);
    }
}

} // namespace

void embedding_forward(torch::Tensor tokens, torch::Tensor weight,
                       torch::Tensor out, int64_t N, int64_t D, int64_t V) {
    int64_t rows = tokens.numel();  // B*N
    embedding_forward_kernel<<<n_blocks(rows * D), THREADS>>>(
        tokens.data_ptr<float>(), weight.data_ptr<float>(),
        out.data_ptr<float>(), rows, D, V);
}

void embedding_backward(torch::Tensor tokens, torch::Tensor dweight,
                        torch::Tensor dout, int64_t N, int64_t D, int64_t V) {
    int64_t rows = tokens.numel();  // B*N
    embedding_backward_kernel<<<n_blocks(rows * D), THREADS>>>(
        tokens.data_ptr<float>(), dweight.data_ptr<float>(),
        dout.data_ptr<float>(), rows, D, V);
}

void layernorm_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                       torch::Tensor out, torch::Tensor mean, torch::Tensor rstd,
                       int64_t rows, int64_t D, double eps) {
    layernorm_forward_kernel<<<n_blocks(rows), THREADS>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        rows, D, (float)eps);
}

void layernorm_backward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                        torch::Tensor dx, torch::Tensor dout, torch::Tensor dw,
                        torch::Tensor db, torch::Tensor mean, torch::Tensor rstd,
                        int64_t rows, int64_t D) {
    layernorm_backward_kernel<<<n_blocks(rows), THREADS>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), dx.data_ptr<float>(),
        dout.data_ptr<float>(), dw.data_ptr<float>(), db.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), rows, D);
}
