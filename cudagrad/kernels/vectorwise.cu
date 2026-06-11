// Vectorwise CUDA kernels for cudagrad: last-dim reductions + softmax.
//
// These ops all act along the FINAL dim of a contiguous fp32 tensor, viewed as
// an (n_rows, n_cols) matrix: row r occupies x[r*n_cols .. r*n_cols + n_cols-1].
// Mirrors tritongrad/kernels/vectorwise.py (the GPU-verified reference) for the
// exact math.
//
// DESIGN: ONE BLOCK PER ROW (grid = n_rows blocks, blockDim = THREADS=256).
// The old design used one *thread* per row, so a (1024, 1024) tensor only ran
// ~1024 threads, each doing a long serial loop with uncoalesced strided reads.
// Here every block owns a single row and its THREADS threads cooperate:
//
//   1. Threads grid-stride over the row's columns (c = tid, tid+blockDim, ...).
//      Consecutive threads touch consecutive addresses => COALESCED loads. This
//      also handles any n_cols: smaller than blockDim (some threads idle),
//      larger (each thread visits several columns), odd, or 1.
//   2. Partial results are combined with a SHARED-MEMORY TREE REDUCTION: each
//      step halves the number of active threads, summing/maxing pairs, until
//      thread 0 holds the row's result. __syncthreads() between steps.
//   3. For elementwise outputs (sum/mean backward, var/std backward column
//      writes, softmax forward/backward writes) threads just stride over their
//      columns in parallel — no reduction needed, each (r,c) is written once.
//
// Reduction op codes (kept in sync with cuda_kernels.py's _REDUCTION_OP):
//   0=sum  1=mean  2=max  3=min  4=var  5=std
//
// var/std use POPULATION normalization (divide by C, subtracting the row MEAN
// = sum/C). We use the TWO-PASS approach: first block-reduce the sum to get the
// mean, then block-reduce sum-of-squared-deviations. (The one-pass
// E[x^2]-E[x]^2 identity is cheaper but less numerically stable; tolerances are
// loose but we prefer to match the exact serial math.)
//
// MAX/MIN TIE-BREAK (subtle): torch routes a max/min gradient to the FIRST
// (lowest column index) occurrence of the extremum. The old serial loop used a
// STRICT comparison (`row[c] > m`), so the earliest extremum wins. A parallel
// reduction must reproduce this: we reduce over (value, index) PAIRS and, on a
// tie in value, KEEP THE SMALLER INDEX. Then only the winning thread (the one
// still holding that argmax/argmin) does `drow[argm] += g`.
//
// Backward launchers ACCUMULATE into dx with `+=`; each (r,c) element is touched
// by exactly one thread (distinct row per block, distinct column per thread), so
// no atomics are needed.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 256;

// ---------------------------------------------------------------------------
// Shared-memory tree reductions. Each writes the final result into smem[0] for
// thread 0 (and leaves other entries clobbered). Caller must __syncthreads()
// after seeding smem[tid] and before reading smem[0].
// ---------------------------------------------------------------------------

// Sum reduction over smem[0..blockDim-1].
__device__ inline void block_reduce_sum(float* smem, int tid, int nthreads) {
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
}

// Max reduction over smem[0..blockDim-1].
__device__ inline void block_reduce_max(float* smem, int tid, int nthreads) {
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
}

// Min reduction over smem[0..blockDim-1].
__device__ inline void block_reduce_min(float* smem, int tid, int nthreads) {
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fminf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
}

// Argmax/argmin reduction over (value, index) pairs, tie-broken toward the
// SMALLER index (to match torch's first-occurrence gradient routing).
// WANT_MAX selects the larger value; otherwise the smaller value.
// On exit, sval[0]/sidx[0] hold the winning (value, index).
template <bool WANT_MAX>
__device__ inline void block_reduce_argext(float* sval, int64_t* sidx,
                                           int tid, int nthreads) {
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = sval[tid],        b = sval[tid + stride];
            int64_t ia = sidx[tid],     ib = sidx[tid + stride];
            // Partner (the +stride entry) wins when it has a strictly better
            // value, OR ties the value but carries the smaller column index.
            bool take_b;
            if (WANT_MAX) take_b = (b > a) || (b == a && ib < ia);
            else          take_b = (b < a) || (b == a && ib < ia);
            if (take_b) { sval[tid] = b; sidx[tid] = ib; }
        }
        __syncthreads();
    }
}

} // namespace

// ===========================================================================
// reduction_forward: out[r] = REDUCE_c x[r, c]
// ===========================================================================
namespace {
__global__ void reduction_forward_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int64_t n_rows, int64_t n_cols, int op) {
    int64_t r = blockIdx.x;
    if (r >= n_rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* row = x + r * n_cols;

    __shared__ float smem[THREADS];

    // Empty-row edge case: nothing to reduce. Match the serial fallbacks
    // (sum/mean/var/std -> 0; max/min would read row[0] out of bounds, which
    // the old code also did, so we just emit 0 to stay safe).
    if (n_cols <= 0) {
        if (tid == 0) out[r] = 0.0f;
        return;
    }

    switch (op) {
        case 0:    // sum
        case 1: {  // mean
            float partial = 0.0f;
            for (int64_t c = tid; c < n_cols; c += nthreads) partial += row[c];
            smem[tid] = partial;
            __syncthreads();
            block_reduce_sum(smem, tid, nthreads);
            if (tid == 0)
                out[r] = (op == 1) ? smem[0] / (float)n_cols : smem[0];
            break;
        }
        case 2: {  // max
            float partial = -INFINITY;
            for (int64_t c = tid; c < n_cols; c += nthreads)
                partial = fmaxf(partial, row[c]);
            smem[tid] = partial;
            __syncthreads();
            block_reduce_max(smem, tid, nthreads);
            if (tid == 0) out[r] = smem[0];
            break;
        }
        case 3: {  // min
            float partial = INFINITY;
            for (int64_t c = tid; c < n_cols; c += nthreads)
                partial = fminf(partial, row[c]);
            smem[tid] = partial;
            __syncthreads();
            block_reduce_min(smem, tid, nthreads);
            if (tid == 0) out[r] = smem[0];
            break;
        }
        default: {  // 4: var, 5: std  (two-pass: mean, then sum sq deviations)
            // Pass 1: sum -> mean.
            float partial = 0.0f;
            for (int64_t c = tid; c < n_cols; c += nthreads) partial += row[c];
            smem[tid] = partial;
            __syncthreads();
            block_reduce_sum(smem, tid, nthreads);
            // Broadcast the mean to all threads via shared memory.
            __shared__ float s_mean;
            if (tid == 0) s_mean = smem[0] / (float)n_cols;
            __syncthreads();
            float mean = s_mean;
            // Pass 2: sum of squared deviations.
            float acc = 0.0f;
            for (int64_t c = tid; c < n_cols; c += nthreads) {
                float d = row[c] - mean;
                acc += d * d;
            }
            __syncthreads();  // all threads done reading s_mean before reuse smem
            smem[tid] = acc;
            __syncthreads();
            block_reduce_sum(smem, tid, nthreads);
            if (tid == 0) {
                float v = smem[0] / (float)n_cols;
                out[r] = (op == 4) ? v : sqrtf(v);
            }
            break;
        }
    }
}
} // namespace

// ===========================================================================
// reduction_backward: dx[r,c] += d(out[r])/d(x[r,c]) * dout[r]  (ACCUMULATES)
// out[r] holds the forward result (used by std).
// ===========================================================================
namespace {
__global__ void reduction_backward_kernel(const float* __restrict__ x,
                                          float* __restrict__ dx,
                                          const float* __restrict__ dout,
                                          const float* __restrict__ out,
                                          int64_t n_rows, int64_t n_cols, int op) {
    int64_t r = blockIdx.x;
    if (r >= n_rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* row = x + r * n_cols;
    float* drow = dx + r * n_cols;
    float g = dout[r];

    if (n_cols <= 0) return;

    switch (op) {
        case 0: {  // sum: dx += dout  (embarrassingly parallel over columns)
            for (int64_t c = tid; c < n_cols; c += nthreads) drow[c] += g;
            break;
        }
        case 1: {  // mean: dx += dout / C
            float gc = g / (float)n_cols;
            for (int64_t c = tid; c < n_cols; c += nthreads) drow[c] += gc;
            break;
        }
        case 2:    // max: route grad to the FIRST max element only
        case 3: {  // min: route grad to the FIRST min element only
            __shared__ float sval[THREADS];
            __shared__ int64_t sidx[THREADS];
            const bool want_max = (op == 2);
            // Each thread finds its local extremum over its strided columns,
            // tie-breaking toward the smaller column index.
            float best = want_max ? -INFINITY : INFINITY;
            int64_t bidx = n_cols;  // sentinel: "no column seen"
            for (int64_t c = tid; c < n_cols; c += nthreads) {
                float v = row[c];
                bool better = want_max ? (v > best) : (v < best);
                if (better) { best = v; bidx = c; }
                // strict comparison above => on a tie the earlier (smaller c,
                // visited first within this thread's stride) is kept.
            }
            sval[tid] = best;
            sidx[tid] = bidx;
            __syncthreads();
            if (want_max) block_reduce_argext<true>(sval, sidx, tid, nthreads);
            else          block_reduce_argext<false>(sval, sidx, tid, nthreads);
            // Only the winning thread writes (single distinct element).
            if (tid == 0) drow[sidx[0]] += g;
            break;
        }
        case 4: {  // var: dx += dout * 2*(x - mean)/C
            __shared__ float smem[THREADS];
            float partial = 0.0f;
            for (int64_t c = tid; c < n_cols; c += nthreads) partial += row[c];
            smem[tid] = partial;
            __syncthreads();
            block_reduce_sum(smem, tid, nthreads);
            __shared__ float s_mean;
            if (tid == 0) s_mean = smem[0] / (float)n_cols;
            __syncthreads();
            float mean = s_mean;
            float coef = g * 2.0f / (float)n_cols;
            for (int64_t c = tid; c < n_cols; c += nthreads)
                drow[c] += coef * (row[c] - mean);
            break;
        }
        default: {  // 5: std: dx += dout * (x - mean)/(C * out); guard out==0
            float sv = out[r];
            if (sv == 0.0f) break;  // forward result is std; 0 -> contribute nothing
            __shared__ float smem[THREADS];
            float partial = 0.0f;
            for (int64_t c = tid; c < n_cols; c += nthreads) partial += row[c];
            smem[tid] = partial;
            __syncthreads();
            block_reduce_sum(smem, tid, nthreads);
            __shared__ float s_mean;
            if (tid == 0) s_mean = smem[0] / (float)n_cols;
            __syncthreads();
            float mean = s_mean;
            float coef = g / ((float)n_cols * sv);
            for (int64_t c = tid; c < n_cols; c += nthreads)
                drow[c] += coef * (row[c] - mean);
            break;
        }
    }
}
} // namespace

// ===========================================================================
// softmax_forward: out[r,c] = softmax(x[r,:])_c  (numerically stable)
//   1. block-reduce max over the row
//   2. block-reduce sum of exp(x - max)
//   3. parallel write out = exp(x - max) / sum
// ===========================================================================
namespace {
__global__ void softmax_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int64_t n_rows, int64_t n_cols) {
    int64_t r = blockIdx.x;
    if (r >= n_rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* row = x + r * n_cols;
    float* orow = out + r * n_cols;

    if (n_cols <= 0) return;

    __shared__ float smem[THREADS];
    __shared__ float s_max;
    __shared__ float s_sum;

    // 1. max over the row.
    float pmax = -INFINITY;
    for (int64_t c = tid; c < n_cols; c += nthreads) pmax = fmaxf(pmax, row[c]);
    smem[tid] = pmax;
    __syncthreads();
    block_reduce_max(smem, tid, nthreads);
    if (tid == 0) s_max = smem[0];
    __syncthreads();
    float mx = s_max;

    // 2. write unnormalized exp(x - max) and accumulate their sum.
    float psum = 0.0f;
    for (int64_t c = tid; c < n_cols; c += nthreads) {
        float e = expf(row[c] - mx);
        orow[c] = e;
        psum += e;
    }
    smem[tid] = psum;
    __syncthreads();
    block_reduce_sum(smem, tid, nthreads);
    if (tid == 0) s_sum = smem[0];
    __syncthreads();
    float inv = 1.0f / s_sum;

    // 3. normalize.
    for (int64_t c = tid; c < n_cols; c += nthreads) orow[c] *= inv;
}
} // namespace

// ===========================================================================
// softmax_backward: dx[r,c] += y[r,c] * (dout[r,c] - dot[r])  (ACCUMULATES)
//   dot[r] = sum_c dout[r,c]*y[r,c]  (block-reduction). y is forward softmax.
// ===========================================================================
namespace {
__global__ void softmax_backward_kernel(const float* __restrict__ y,
                                        float* __restrict__ dx,
                                        const float* __restrict__ dout,
                                        int64_t n_rows, int64_t n_cols) {
    int64_t r = blockIdx.x;
    if (r >= n_rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* yrow = y + r * n_cols;
    const float* grow = dout + r * n_cols;
    float* drow = dx + r * n_cols;

    if (n_cols <= 0) return;

    __shared__ float smem[THREADS];
    __shared__ float s_dot;

    // dot[r] = sum_c dout*y.
    float pdot = 0.0f;
    for (int64_t c = tid; c < n_cols; c += nthreads) pdot += grow[c] * yrow[c];
    smem[tid] = pdot;
    __syncthreads();
    block_reduce_sum(smem, tid, nthreads);
    if (tid == 0) s_dot = smem[0];
    __syncthreads();
    float dot = s_dot;

    // parallel column write.
    for (int64_t c = tid; c < n_cols; c += nthreads)
        drow[c] += yrow[c] * (grow[c] - dot);
}
} // namespace

// ===========================================================================
// Launchers (signatures must stay byte-for-byte identical to kernels.h).
// Grid = n_rows blocks (one block per row); blockDim = THREADS.
// ===========================================================================
void reduction_forward(torch::Tensor x, torch::Tensor out,
                       int64_t n_rows, int64_t n_cols, int64_t op) {
    if (n_rows <= 0) return;
    reduction_forward_kernel<<<(unsigned int)n_rows, THREADS>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols, (int)op);
}

void reduction_backward(torch::Tensor x, torch::Tensor dx, torch::Tensor dout,
                        torch::Tensor out, int64_t n_rows, int64_t n_cols,
                        int64_t op) {
    if (n_rows <= 0) return;
    reduction_backward_kernel<<<(unsigned int)n_rows, THREADS>>>(
        x.data_ptr<float>(), dx.data_ptr<float>(), dout.data_ptr<float>(),
        out.data_ptr<float>(), n_rows, n_cols, (int)op);
}

void softmax_forward(torch::Tensor x, torch::Tensor out,
                     int64_t n_rows, int64_t n_cols) {
    if (n_rows <= 0) return;
    softmax_forward_kernel<<<(unsigned int)n_rows, THREADS>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols);
}

void softmax_backward(torch::Tensor y, torch::Tensor dx, torch::Tensor dout,
                      int64_t n_rows, int64_t n_cols) {
    if (n_rows <= 0) return;
    softmax_backward_kernel<<<(unsigned int)n_rows, THREADS>>>(
        y.data_ptr<float>(), dx.data_ptr<float>(), dout.data_ptr<float>(),
        n_rows, n_cols);
}
