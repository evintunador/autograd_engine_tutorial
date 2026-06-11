// Module CUDA kernels for cudagrad: Embedding + LayerNorm (fwd + bwd).
//
// Mirrors tritongrad/kernels/modules.py (the GPU-verified reference) for the
// exact math.
//
// Embedding: ONE THREAD PER (row, d) output element; coalesced, no reductions.
//
// LayerNorm: ONE BLOCK PER ROW. The threads of a block cooperatively stride
// over the D feature dimension and combine partial sums via a SHARED-MEMORY
// TREE REDUCTION. This replaces the old one-thread-per-row design, where a
// single thread looped serially over all D and every (row,d) element issued an
// atomicAdd into dw/db (massive atomic contention + no intra-row parallelism).
//
// Layout/contract notes:
//   * float tensors are contiguous fp32; `tokens` arrives as fp32 (engine is
//     fp32-only) carrying exact small-integer ids.
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

// LayerNorm block size: one block per row, threads stride over D. 256 gives
// good occupancy and a clean power-of-two tree reduction.
constexpr int LN_THREADS = 256;

inline int64_t n_blocks(int64_t n) { return (n + THREADS - 1) / THREADS; }

// Block-wide sum reduction over `val`, one value per thread, using shared mem
// `sdata` (size >= blockDim.x). All threads must call it (uniform control
// flow). Returns the total to EVERY thread (broadcast via sdata[0]).
__device__ inline float block_reduce_sum(float val, float* sdata) {
    int t = threadIdx.x;
    sdata[t] = val;
    __syncthreads();
    // tree reduction: halve the active range each step
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) sdata[t] += sdata[t + stride];
        __syncthreads();
    }
    float total = sdata[0];
    __syncthreads();  // make sure all threads read before sdata is reused
    return total;
}

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
//
// ONE BLOCK PER ROW. blockIdx.x == row r. The block's threads grid-stride over
// d in [0, D) (handles D > blockDim and D < blockDim alike). Two block-wide
// tree reductions give the row sum (-> mean) and the sum of squared deviations
// (-> population var). Then each thread writes the output elements it owns.
__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        const float* __restrict__ b,
                                        float* __restrict__ out,
                                        float* __restrict__ mean,
                                        float* __restrict__ rstd,
                                        int64_t rows, int64_t D, float eps) {
    extern __shared__ float sdata[];  // size = blockDim.x floats
    int64_t r = blockIdx.x;
    if (r >= rows) return;
    int t = threadIdx.x;
    const float* xrow = x + r * D;
    float* orow = out + r * D;

    // pass 1: partial sum of x over this thread's columns, then block-reduce.
    float s = 0.0f;
    for (int64_t d = t; d < D; d += blockDim.x) s += xrow[d];
    float mu = block_reduce_sum(s, sdata) / (float)D;

    // pass 2: partial sum of squared deviations -> population (/D) variance.
    float acc = 0.0f;
    for (int64_t d = t; d < D; d += blockDim.x) {
        float diff = xrow[d] - mu;
        acc += diff * diff;
    }
    float var = block_reduce_sum(acc, sdata) / (float)D;
    float rs = 1.0f / sqrtf(var + eps);

    // one thread stores the per-row stats for backward.
    if (t == 0) {
        mean[r] = mu;
        rstd[r] = rs;
    }

    // pass 3: normalize + affine. Each (r,d) written by exactly one thread.
    for (int64_t d = t; d < D; d += blockDim.x) {
        float xhat = (xrow[d] - mu) * rs;
        orow[d] = xhat * w[d] + b[d];
    }
}

// dx (accumulate, per-row, no race), dweight/dbias (atomicAdd across rows).
//   dxhat[d] = dout[r,d]*w[d];  c1 = (1/D) Σ dxhat;  c2 = (1/D) Σ dxhat*xhat;
//   dx[r,d] += rstd[r] * (dxhat[d] - c1 - xhat[r,d]*c2)
//   dweight[d] += dout[r,d]*xhat[r,d] ;  dbias[d] += dout[r,d]
//
// ONE BLOCK PER ROW. Threads grid-stride over d. We block-reduce the two sums
// for c1/c2 in one shot by packing them into a length-2*blockDim shared buffer
// (sum of dxhat in the low half, sum of dxhat*xhat in the high half). Then each
// thread, for each d it owns, writes dx (per-row, no race) and issues ONE
// atomicAdd per (row,d) into dw[d]/db[d] (summed across rows, so atomics are
// required). That is one atomic per (row,d) total — the same count as before,
// but the per-row work is now parallel across the block and the c1/c2 sums are
// a parallel reduction instead of a serial loop, which is the dominant win.
// Uses mean[r]/rstd[r] saved in forward.
__global__ void layernorm_backward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         float* __restrict__ dx,
                                         const float* __restrict__ dout,
                                         float* __restrict__ dw,
                                         float* __restrict__ db,
                                         const float* __restrict__ mean,
                                         const float* __restrict__ rstd,
                                         int64_t rows, int64_t D) {
    extern __shared__ float sdata[];  // size = blockDim.x floats
    int64_t r = blockIdx.x;
    if (r >= rows) return;
    int t = threadIdx.x;
    const float* xrow = x + r * D;
    const float* grow = dout + r * D;
    float* dxrow = dx + r * D;
    float mu = mean[r];
    float rs = rstd[r];

    // pass 1: partial c1 = Σ dxhat, c2 = Σ dxhat*xhat over this thread's cols.
    float p1 = 0.0f, p2 = 0.0f;
    for (int64_t d = t; d < D; d += blockDim.x) {
        float xhat = (xrow[d] - mu) * rs;
        float dxhat = grow[d] * w[d];
        p1 += dxhat;
        p2 += dxhat * xhat;
    }
    // two block-wide reductions (reuse the same shared buffer sequentially).
    float c1 = block_reduce_sum(p1, sdata) / (float)D;
    float c2 = block_reduce_sum(p2, sdata) / (float)D;

    // pass 2: dx (per-row, no race) + dweight/dbias (atomic across rows).
    for (int64_t d = t; d < D; d += blockDim.x) {
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
    if (rows == 0) return;
    // one block per row; threads stride over D. shared mem = blockDim floats.
    size_t shmem = (size_t)LN_THREADS * sizeof(float);
    layernorm_forward_kernel<<<(unsigned int)rows, LN_THREADS, shmem>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        rows, D, (float)eps);
}

void layernorm_backward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                        torch::Tensor dx, torch::Tensor dout, torch::Tensor dw,
                        torch::Tensor db, torch::Tensor mean, torch::Tensor rstd,
                        int64_t rows, int64_t D) {
    if (rows == 0) return;
    // one block per row; threads stride over D. shared mem = blockDim floats.
    size_t shmem = (size_t)LN_THREADS * sizeof(float);
    layernorm_backward_kernel<<<(unsigned int)rows, LN_THREADS, shmem>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), dx.data_ptr<float>(),
        dout.data_ptr<float>(), dw.data_ptr<float>(), db.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), rows, D);
}
