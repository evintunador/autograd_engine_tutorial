// Flash-attention CUDA kernels for cudagrad: CAUSAL attention fwd + bwd.
//
// This is NOT tritongrad's tiled online-softmax flash kernel. It is a
// straightforward, obviously-correct CAUSAL attention computed per query row.
// It is the project's CAUSAL-attention correctness reference, so it favours an
// easy-to-verify decomposition over peak performance.
//
// COOPERATIVE BLOCK-PER-ROW DESIGN
// --------------------------------
// The original code used ONE THREAD per (b,h,row): each thread serially walked
// every key j and every channel d with uncoalesced global loads — ~150x slower
// than PyTorch SDPA. This version launches ONE THREADBLOCK per output row and
// lets `blockDim.x` threads (THREADS = 128) cooperate:
//
//   * grid.x = B*H*N, so block `idx` owns exactly one row. Decode i = idx % N,
//     bh = idx / N (flat b*H+h). EVERY thread in a block shares the same `i`/`j`
//     "owned" row, hence the causal loop bound (`j <= i` fwd/dQ, `i >= j`
//     dV/dK) is IDENTICAL for all threads in the block — no divergent
//     __syncthreads(), every thread reaches every barrier.
//   * The "owned" query/key row is staged into __shared__ ONCE (dynamic shared
//     mem, D floats) so the hot dot-product loop reads it from smem, not global.
//   * Each dot product (Q_i.K_j, dO_i.V_j, ...) is split across threads over d,
//     then combined with a warp-shuffle + cross-warp block reduction
//     (block_reduce_sum). The scalar result is broadcast via shared mem.
//   * Output accumulation over d (O[i,:], dQ[i,:], dV[j,:], dK[j,:]) is also
//     split across threads over d. Since each BLOCK owns a DISTINCT output row,
//     and within a block each d is written by exactly one thread, the `+=`
//     accumulation is race-free WITHOUT atomics.
//   * General D: every per-d loop is a grid-strided `for (d = tid; d < D; d +=
//     blockDim.x)`, so D need not divide or be bounded by blockDim. Shared mem
//     for the staged row is sized D*sizeof(float) at launch.
//
// Layout/contract notes (UNCHANGED — match precisely, this is a reference):
//   * Q/K/V/O/dO/dQ/dK/dV are (B,H,N,D) contiguous fp32; LSE/Delta are (B,H,N).
//     Q[b,h,i,d] = Qptr[((b*H + h)*N + i)*D + d];  LSE[b,h,i] = LSEptr[(b*H+h)*N + i].
//   * CAUSAL masking: query i attends only to keys j <= i.
//   * score s_ij = scale * (Q_i . K_j)  — `scale` is the MULTIPLIER passed in
//     (= sqrt(D) in the suite), used verbatim (NOT 1/sqrt(D)).
//   * forward stores LSE[i] = m_i + log(l_i) for backward (P_ij = exp(s_ij-LSE[i])).
//   * backward ACCUMULATES into dQ/dK/dV (`+=`), so callers pass zeroed grads.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 128;            // threads per block (one block per row)
constexpr int WARP = 32;
constexpr int MAX_WARPS = THREADS / WARP;

// ---- reductions -----------------------------------------------------------
// Warp-level sum via shuffles: lane 0 of each warp ends up with the warp sum.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Block-level sum. `scratch` is a shared array of size MAX_WARPS supplied by the
// caller. Returns the full block sum, broadcast to EVERY thread. All threads in
// the block MUST call this (it contains __syncthreads()).
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
    int lane = threadIdx.x % WARP;
    int wid  = threadIdx.x / WARP;
    v = warp_reduce_sum(v);                 // reduce within each warp
    if (lane == 0) scratch[wid] = v;        // warp leaders publish partials
    __syncthreads();
    // first warp reduces the per-warp partials
    int n_warps = (blockDim.x + WARP - 1) / WARP;
    float total = (threadIdx.x < n_warps) ? scratch[threadIdx.x] : 0.0f;
    if (wid == 0) total = warp_reduce_sum(total);
    // broadcast the result to all threads through shared mem
    if (threadIdx.x == 0) scratch[0] = total;
    __syncthreads();
    float out = scratch[0];
    __syncthreads();                        // ensure all read before reuse
    return out;
}

// (The forward's per-row max m_i is a serial fmaxf over the per-j scalar scores
//  s_ij, which are themselves block_reduce_sum results — so no separate
//  block-max reduction is needed.)

// ---- forward --------------------------------------------------------------
// One BLOCK per (b,h,i). Two-pass (numerically-stable) softmax:
//   pass 1: m_i = max_{j<=i} scale*(Q_i.K_j)
//   pass 2: l_i = Sum_{j<=i} exp(s_ij - m_i),  O[i,:] = Sum p_ij V[j,:] / l_i
// Each dot product Q_i.K_j is a block reduction over d; O accumulation splits
// over d. Q_i is staged in shared memory once.
//
// Dynamic shared layout: [ Qsh : D floats ][ scratch : MAX_WARPS floats ].
__global__ void flash_forward_kernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     float* __restrict__ O,
                                     float* __restrict__ LSE,
                                     float scale,
                                     int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = blockIdx.x;            // one block per row
    int64_t i   = idx % N;
    int64_t bh  = idx / N;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* Qsh     = smem;               // D floats: staged query row
    float* scratch = smem + D;           // MAX_WARPS floats: reduction scratch

    const float* Qrow  = Q + (bh * N + i) * D;
    const float* Kbase = K + bh * N * D;
    const float* Vbase = V + bh * N * D;
    float* Orow = O + (bh * N + i) * D;

    // stage Q_i into shared memory (coalesced) and zero the output accumulator
    for (int64_t d = tid; d < D; d += blockDim.x) {
        Qsh[d]  = Qrow[d];
        Orow[d] = 0.0f;
    }
    __syncthreads();

    // pass 1: running max of scores over j <= i (uniform bound for all threads)
    float m = -INFINITY;
    for (int64_t j = 0; j <= i; ++j) {
        const float* Krow = Kbase + j * D;
        float partial = 0.0f;
        for (int64_t d = tid; d < D; d += blockDim.x) partial += Qsh[d] * Krow[d];
        float s = block_reduce_sum(partial, scratch) * scale;
        m = fmaxf(m, s);                 // identical on every thread (s is broadcast)
    }

    // pass 2: l_i and O[i,:] += exp(s_ij - m) * V[j,:]
    float l = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        const float* Krow = Kbase + j * D;
        const float* Vrow = Vbase + j * D;
        float partial = 0.0f;
        for (int64_t d = tid; d < D; d += blockDim.x) partial += Qsh[d] * Krow[d];
        float s = block_reduce_sum(partial, scratch) * scale;
        float p = expf(s - m);
        l += p;                          // identical on every thread
        for (int64_t d = tid; d < D; d += blockDim.x) Orow[d] += p * Vrow[d];
    }

    // normalize: each d written by exactly one thread -> race-free
    float inv_l = 1.0f / l;
    for (int64_t d = tid; d < D; d += blockDim.x) Orow[d] *= inv_l;

    if (tid == 0) LSE[bh * N + i] = m + logf(l);
}

// ---- backward: Delta ------------------------------------------------------
// Delta[b,h,i] = Sum_d O[i,d]*dO[i,d]. One BLOCK per (b,h,i); the dot product is
// a block reduction over d. Shared mem: scratch[MAX_WARPS].
__global__ void flash_delta_kernel(const float* __restrict__ O,
                                   const float* __restrict__ dO,
                                   float* __restrict__ Delta,
                                   int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = blockIdx.x;
    int64_t i   = idx % N;
    int64_t bh  = idx / N;
    int tid = threadIdx.x;

    extern __shared__ float scratch[];   // MAX_WARPS floats

    const float* Orow  = O  + (bh * N + i) * D;
    const float* dOrow = dO + (bh * N + i) * D;

    float partial = 0.0f;
    for (int64_t d = tid; d < D; d += blockDim.x) partial += Orow[d] * dOrow[d];
    float acc = block_reduce_sum(partial, scratch);
    if (tid == 0) Delta[bh * N + i] = acc;
}

// ---- backward: dV ---------------------------------------------------------
// dV[j,d] += Sum_{i >= j} P_ij * dO[i,d],  P_ij = exp(scale*Q_i.K_j - LSE[i]).
// One BLOCK per (b,h,j); owns dV row j -> race-free `+=`. K_j staged in smem.
// s_ij = Q_i.K_j is a block reduction over d; the dV accumulation splits over d.
//
// Dynamic shared layout: [ Ksh : D ][ scratch : MAX_WARPS ].
__global__ void flash_dV_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ dO,
                                float* __restrict__ dV,
                                const float* __restrict__ LSE,
                                float scale,
                                int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = blockIdx.x;
    int64_t j   = idx % N;
    int64_t bh  = idx / N;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* Ksh     = smem;               // D floats: staged key row j
    float* scratch = smem + D;           // MAX_WARPS floats

    const float* Qbase  = Q + bh * N * D;
    const float* Krow   = K + (bh * N + j) * D;
    const float* dObase = dO + bh * N * D;
    const float* LSErow = LSE + bh * N;
    float* dVrow = dV + (bh * N + j) * D;

    for (int64_t d = tid; d < D; d += blockDim.x) Ksh[d] = Krow[d];
    __syncthreads();

    // accumulate into a per-thread register slice of dV, flush once at the end.
    // (i loop bound `i >= j` is uniform across the block.)
    for (int64_t i = j; i < N; ++i) {
        const float* Qrow = Qbase + i * D;
        float partial = 0.0f;
        for (int64_t d = tid; d < D; d += blockDim.x) partial += Qrow[d] * Ksh[d];
        float s = block_reduce_sum(partial, scratch);
        float p = expf(scale * s - LSErow[i]);   // identical on every thread
        const float* dOrow = dObase + i * D;
        for (int64_t d = tid; d < D; d += blockDim.x) dVrow[d] += p * dOrow[d];
    }
}

// ---- backward: dQ ---------------------------------------------------------
// dp_ij = Sum_d dO[i,d]*V[j,d];  ds_ij = P_ij*(dp_ij - Delta[i]);
// dQ[i,d] += scale * Sum_{j <= i} ds_ij * K[j,d].
// One BLOCK per (b,h,i); owns dQ row i -> race-free `+=`. Q_i and dO_i staged.
// Each j needs two block reductions (s_ij and dp_ij), done as two sequential
// block_reduce_sum calls that safely reuse the same scratch.
//
// Dynamic shared layout: [ Qsh : D ][ dOsh : D ][ scratch : MAX_WARPS floats ].
__global__ void flash_dQ_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                const float* __restrict__ dO,
                                float* __restrict__ dQ,
                                const float* __restrict__ LSE,
                                const float* __restrict__ Delta,
                                float scale,
                                int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = blockIdx.x;
    int64_t i   = idx % N;
    int64_t bh  = idx / N;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* Qsh     = smem;               // D floats: staged query row i
    float* dOsh    = smem + D;           // D floats: staged dO row i
    float* scratch = smem + 2 * D;       // MAX_WARPS floats

    const float* Qrow  = Q + (bh * N + i) * D;
    const float* Kbase = K + bh * N * D;
    const float* Vbase = V + bh * N * D;
    const float* dOrow = dO + (bh * N + i) * D;
    float* dQrow = dQ + (bh * N + i) * D;
    float lse_i   = LSE[bh * N + i];
    float delta_i = Delta[bh * N + i];

    for (int64_t d = tid; d < D; d += blockDim.x) {
        Qsh[d]  = Qrow[d];
        dOsh[d] = dOrow[d];
    }
    __syncthreads();

    for (int64_t j = 0; j <= i; ++j) {   // uniform bound across the block
        const float* Krow = Kbase + j * D;
        const float* Vrow = Vbase + j * D;
        float ps = 0.0f, pdp = 0.0f;
        for (int64_t d = tid; d < D; d += blockDim.x) {
            ps  += Qsh[d]  * Krow[d];
            pdp += dOsh[d] * Vrow[d];
        }
        float s  = block_reduce_sum(ps,  scratch);   // each has its own barriers
        float dp = block_reduce_sum(pdp, scratch);
        float p  = expf(scale * s - lse_i);
        float ds = p * (dp - delta_i);
        float coef = scale * ds;                     // identical on every thread
        for (int64_t d = tid; d < D; d += blockDim.x) dQrow[d] += coef * Krow[d];
    }
}

// ---- backward: dK ---------------------------------------------------------
// dK[j,d] += scale * Sum_{i >= j} ds_ij * Q[i,d].
// One BLOCK per (b,h,j); owns dK row j -> race-free `+=`. K_j and V_j staged.
//
// Dynamic shared layout: [ Ksh : D ][ Vsh : D ][ scratch : MAX_WARPS ].
__global__ void flash_dK_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                const float* __restrict__ dO,
                                float* __restrict__ dK,
                                const float* __restrict__ LSE,
                                const float* __restrict__ Delta,
                                float scale,
                                int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = blockIdx.x;
    int64_t j   = idx % N;
    int64_t bh  = idx / N;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* Ksh     = smem;               // D floats: staged key row j
    float* Vsh     = smem + D;           // D floats: staged value row j
    float* scratch = smem + 2 * D;       // MAX_WARPS floats

    const float* Qbase    = Q + bh * N * D;
    const float* Krow     = K + (bh * N + j) * D;
    const float* Vrow     = V + (bh * N + j) * D;
    const float* dObase   = dO + bh * N * D;
    const float* LSErow   = LSE + bh * N;
    const float* Deltarow = Delta + bh * N;
    float* dKrow = dK + (bh * N + j) * D;

    for (int64_t d = tid; d < D; d += blockDim.x) {
        Ksh[d] = Krow[d];
        Vsh[d] = Vrow[d];
    }
    __syncthreads();

    for (int64_t i = j; i < N; ++i) {    // uniform bound across the block
        const float* Qrow  = Qbase + i * D;
        const float* dOrow = dObase + i * D;
        float ps = 0.0f, pdp = 0.0f;
        for (int64_t d = tid; d < D; d += blockDim.x) {
            ps  += Qrow[d]  * Ksh[d];
            pdp += dOrow[d] * Vsh[d];
        }
        float s  = block_reduce_sum(ps,  scratch);
        float dp = block_reduce_sum(pdp, scratch);
        float p  = expf(scale * s - LSErow[i]);
        float ds = p * (dp - Deltarow[i]);
        float coef = scale * ds;                     // identical on every thread
        for (int64_t d = tid; d < D; d += blockDim.x) dKrow[d] += coef * Qrow[d];
    }
}

} // namespace

void flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                             torch::Tensor O, torch::Tensor LSE, double scale,
                             int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t total = B * H * N;           // one block per (b,h,row)
    size_t smem = (D + MAX_WARPS) * sizeof(float);   // Qsh[D] + scratch[MAX_WARPS]
    flash_forward_kernel<<<total, THREADS, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), LSE.data_ptr<float>(), (float)scale, B, H, N, D);
}

void flash_attention_backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              torch::Tensor O, torch::Tensor dO,
                              torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV,
                              torch::Tensor LSE, double scale,
                              int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t total = B * H * N;           // one block per (b,h,row)
    auto Delta = torch::empty({B, H, N}, LSE.options());

    size_t smem_delta = MAX_WARPS * sizeof(float);
    flash_delta_kernel<<<total, THREADS, smem_delta>>>(
        O.data_ptr<float>(), dO.data_ptr<float>(), Delta.data_ptr<float>(),
        B, H, N, D);

    size_t smem_dV = (D + MAX_WARPS) * sizeof(float);        // Ksh[D] + scratch
    flash_dV_kernel<<<total, THREADS, smem_dV>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), dO.data_ptr<float>(),
        dV.data_ptr<float>(), LSE.data_ptr<float>(), (float)scale, B, H, N, D);

    size_t smem_dQ = (2 * D + MAX_WARPS) * sizeof(float);    // Qsh+dOsh + scratch
    flash_dQ_kernel<<<total, THREADS, smem_dQ>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), dQ.data_ptr<float>(), LSE.data_ptr<float>(),
        Delta.data_ptr<float>(), (float)scale, B, H, N, D);

    size_t smem_dK = (2 * D + MAX_WARPS) * sizeof(float);    // Ksh+Vsh + scratch
    flash_dK_kernel<<<total, THREADS, smem_dK>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), dK.data_ptr<float>(), LSE.data_ptr<float>(),
        Delta.data_ptr<float>(), (float)scale, B, H, N, D);
}
