// Flash-attention CUDA kernels for cudagrad: CAUSAL attention fwd + bwd.
//
// This is the project's CAUSAL-attention correctness reference. The FORWARD is
// a tiled online-softmax flash kernel (the real performance win); the BACKWARD
// is the obviously-correct cooperative block-per-output-row decomposition (a
// correct backward + a fast forward beats a fragile fast backward).
//
// =====================================================================
// FORWARD: QUERY-TILED ONLINE-SOFTMAX FLASH ATTENTION
// =====================================================================
// The previous forward processed ONE query row per block and therefore
// re-read ALL of K and V from global memory for every single query row — that
// HBM traffic was the bottleneck (~2% of PyTorch SDPA). The flash idea fixes
// this by REUSING K/V tiles across many queries:
//
//   * One block owns Br CONSECUTIVE query rows for a fixed (b,h).
//     grid.x = B*H*ceil(N/Br). Decode the block's (b,h, query-tile) from
//     blockIdx.x. blockDim.x = Br: thread r in the block owns query row
//     q_row = qtile*Br + r (one thread = one query row).
//   * The block loops over KEY TILES of Bc keys. Each K-tile and V-tile is
//     loaded into shared memory ONCE (cooperatively by all Br threads) and
//     then reused by ALL Br queries — so K/V are read from HBM O(N/Bc) times
//     per (b,h) instead of O(N) times.
//   * Each thread keeps its query's running online-softmax state in registers:
//     m (running max), l (running denom), and an O accumulator acc[D] (D<=DMAX,
//     held in a per-thread register/local array). After all key tiles:
//     O = acc / l, LSE = m + log(l).
//
// CAUSAL TILE BOOKKEEPING (the easy place to get a bug):
//   For query row i and key index j the contribution exists iff j <= i.
//   Process key tiles kt = 0 .. ceil(N/Bc)-1, key index j = kt*Bc + c:
//     * The whole block shares query rows [qtile*Br, qtile*Br + Br).
//       The MAX query row owned by the block is qmax = qtile*Br + Br - 1
//       (clamped to N-1). A key tile whose MINIMUM key index kt*Bc already
//       exceeds qmax is strictly above the diagonal for EVERY query in the
//       block -> skip the whole tile. Loop bound: kt*Bc <= qmax, i.e.
//       kt <= qmax/Bc. This bound depends only on qtile (block-uniform), so
//       every thread runs the SAME number of key-tile iterations -> all
//       __syncthreads() are reached uniformly.
//     * Within a kept tile, a (query i, key j) pair contributes iff
//       j <= i AND j < N. We apply that as a per-element mask (score set to
//       -inf when masked) rather than an early return, so barriers stay
//       uniform. Tiles strictly BELOW the diagonal still get the cheap mask
//       check; it is simply always true there.
//
// ONLINE-SOFTMAX RESCALING (must match the reference numerics so the backward,
// which recomputes P = exp(scale*QK - LSE), stays consistent):
//   For each new score s, m_new = max(m, s); correction = exp(m_old - m_new);
//   l = l*correction + exp(s - m_new); acc[:] = acc[:]*correction + exp(s-m_new)*V[j,:].
//   Equivalently we fold the whole tile's contributions in before rescaling
//   once per key (mathematically identical to the canonical flash update). The
//   FINAL LSE = m + log(l) uses the FINAL m,l — identical to the old two-pass
//   kernel's result up to floating-point reassociation, well within tolerance.
//
// RACE-FREE WRITES: each query row's O and LSE is owned by exactly one thread
// in exactly one block -> no atomics needed in the forward.
//
// =====================================================================
// BACKWARD: COOPERATIVE BLOCK-PER-OUTPUT-ROW (unchanged, obviously correct)
// =====================================================================
// One THREADBLOCK per output row; blockDim.x threads cooperate over the channel
// dim D. Each dot product is a warp-shuffle + cross-warp block reduction; output
// accumulation splits over d. Because each block owns a DISTINCT output row and
// each d is written by exactly one thread, the `+=` accumulation is race-free
// WITHOUT atomics. The causal loop bound (`i >= j` for dV/dK, `j <= i` for dQ)
// is identical for every thread in a block, so every __syncthreads() is reached
// uniformly.
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

constexpr int THREADS = 128;            // threads per block (backward: one block per row)
constexpr int WARP = 32;
constexpr int MAX_WARPS = THREADS / WARP;

// ---- forward tiling parameters --------------------------------------------
// Br query rows per block, Bc keys per key-tile. blockDim.x = Br in the forward.
// DMAX caps the per-thread register O accumulator; the bench uses D=64, the test
// D=32. 128 leaves headroom while keeping register pressure reasonable.
constexpr int FWD_BR = 64;              // query rows per block (== forward blockDim.x)
constexpr int FWD_BC = 32;              // keys per key-tile (keeps K/V smem modest)
constexpr int FWD_DMAX = 128;           // max D supported by the register accumulator

// ---- reductions (backward only) -------------------------------------------
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

// ---- forward (tiled) -------------------------------------------------------
// One BLOCK per (b,h, query-tile). blockDim.x = Br; thread r owns query row
// q_row = qtile*Br + r. The block loops over key tiles of Bc keys, loading each
// K/V tile to shared memory once and reusing it across all Br queries with the
// standard online-softmax update.
//
// Dynamic shared layout (Br = blockDim.x, Bc = FWD_BC):
//   [ Ksh : Bc*D floats ][ Vsh : Bc*D floats ]
// Q rows are loaded straight to per-thread registers (q[d]); the O accumulator
// acc[d] also lives in per-thread registers. D <= FWD_DMAX.
__global__ void flash_forward_kernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     float* __restrict__ O,
                                     float* __restrict__ LSE,
                                     float scale,
                                     int64_t B, int64_t H, int64_t N, int64_t D) {
    const int Br = blockDim.x;               // query rows per block
    const int Bc = FWD_BC;                   // keys per tile
    const int tid = threadIdx.x;             // local query index r in [0,Br)

    // Decode this block's (b,h) and query-tile.
    const int64_t n_qtiles = (N + Br - 1) / Br;
    const int64_t bh    = blockIdx.x / n_qtiles;
    const int64_t qtile = blockIdx.x % n_qtiles;
    const int64_t q_row = qtile * Br + tid;  // global query row this thread owns
    const bool active = (q_row < N);         // last tile may be partial

    const float* Qbase = Q + bh * N * D;
    const float* Kbase = K + bh * N * D;
    const float* Vbase = V + bh * N * D;
    float* Obase = O + bh * N * D;

    extern __shared__ float smem[];
    float* Ksh = smem;                       // Bc*D floats
    float* Vsh = smem + (int64_t)Bc * D;     // Bc*D floats

    // Stage this thread's query row into registers. Inactive threads (padding in
    // the last query-tile) load nothing but still participate in every barrier.
    float q[FWD_DMAX];
    float acc[FWD_DMAX];
    if (active) {
        const float* Qrow = Qbase + q_row * D;
        for (int d = 0; d < D; ++d) { q[d] = Qrow[d]; acc[d] = 0.0f; }
    } else {
        for (int d = 0; d < D; ++d) { q[d] = 0.0f; acc[d] = 0.0f; }
    }
    float m = -INFINITY;                     // running max for this query
    float l = 0.0f;                          // running softmax denominator

    // Causal: the largest key index any query in this block can attend to is
    // qmax = min(qtile*Br + Br - 1, N-1). Tiles whose first key index exceeds
    // qmax are entirely above the diagonal -> skip. This bound is block-uniform.
    int64_t qmax = qtile * (int64_t)Br + (int64_t)Br - 1;
    if (qmax > N - 1) qmax = N - 1;
    const int64_t last_kt = qmax / Bc;       // inclusive last key-tile index

    for (int64_t kt = 0; kt <= last_kt; ++kt) {
        const int64_t k0 = kt * Bc;          // first key index of this tile
        // Cooperatively load the K and V tiles into shared memory. Bc*D elements
        // each, strided by blockDim.x. Out-of-range keys (k0+c >= N) are left
        // as-is and masked later via the score, so we still zero them to be safe.
        for (int e = tid; e < Bc * D; e += blockDim.x) {
            int c = e / D;                   // key-within-tile
            int d = e % D;
            int64_t kj = k0 + c;
            if (kj < N) {
                Ksh[e] = Kbase[kj * D + d];
                Vsh[e] = Vbase[kj * D + d];
            } else {
                Ksh[e] = 0.0f;
                Vsh[e] = 0.0f;
            }
        }
        __syncthreads();                     // tile ready (uniform barrier)

        // Each active thread folds this tile's Bc keys into its online softmax.
        if (active) {
            for (int c = 0; c < Bc; ++c) {
                int64_t kj = k0 + c;
                // CAUSAL + bounds mask: contributes iff kj <= q_row and kj < N.
                if (kj > q_row || kj >= N) continue;
                const float* Krow = Ksh + (int64_t)c * D;
                float s = 0.0f;
                for (int d = 0; d < D; ++d) s += q[d] * Krow[d];
                s *= scale;
                // online-softmax update with rescaling on a new running max
                float m_new = fmaxf(m, s);
                float corr  = expf(m - m_new);   // exp(m_old - m_new); 1 if m unchanged
                float p     = expf(s - m_new);
                l = l * corr + p;
                const float* Vrow = Vsh + (int64_t)c * D;
                for (int d = 0; d < D; ++d) acc[d] = acc[d] * corr + p * Vrow[d];
                m = m_new;
            }
        }
        __syncthreads();                     // all done reading tile before reload
    }

    // Finalize: O = acc / l, LSE = m + log(l). Each thread owns its own row.
    if (active) {
        float inv_l = 1.0f / l;
        float* Orow = Obase + q_row * D;
        for (int d = 0; d < D; ++d) Orow[d] = acc[d] * inv_l;
        LSE[bh * N + q_row] = m + logf(l);
    }
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
    // Tiled forward: one block per (b,h, query-tile of FWD_BR rows); blockDim.x
    // = FWD_BR (one thread per query row). Each block loops over key tiles of
    // FWD_BC keys, staging K/V tiles in shared memory for reuse across queries.
    const int Br = FWD_BR;
    const int Bc = FWD_BC;
    int64_t n_qtiles = (N + Br - 1) / Br;
    int64_t blocks = B * H * n_qtiles;
    size_t smem = (size_t)2 * Bc * D * sizeof(float);  // Ksh[Bc*D] + Vsh[Bc*D]
    flash_forward_kernel<<<blocks, Br, smem>>>(
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
