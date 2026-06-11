// Flash-attention CUDA kernels for cudagrad: CAUSAL attention fwd + bwd.
//
// This is the project's CAUSAL-attention correctness reference. Both passes are
// now TILED flash kernels that stage K/V (and for backward Q/dO) tiles in shared
// memory and reuse them across many rows, instead of re-reading global memory
// per row.
//
// =====================================================================
// FORWARD: QUERY-TILED ONLINE-SOFTMAX FLASH ATTENTION (shared-mem accumulator)
// =====================================================================
// The original forward processed ONE query row per block and re-read ALL of K/V
// from global for every query row. The first tiled rewrite reused K/V tiles but
// kept each query's O accumulator and Q row in per-thread LOCAL arrays
// (q[DMAX], acc[DMAX]); those fixed-size arrays SPILLED to local memory and
// throttled the kernel to ~9% of PyTorch. This version removes the spill by
// putting the per-query state in SHARED memory (Option 1):
//
//   * One block owns Br CONSECUTIVE query rows for a fixed (b,h).
//     grid.x = B*H*ceil(N/Br). blockDim.x = Br: thread r owns query row
//     q_row = qtile*Br + r (one thread = one query row).
//   * Dynamic shared layout (Br = blockDim.x, Bc = FWD_BC):
//       [ Qsh : Br*D ][ Osh : Br*D ][ Ksh : Bc*D ][ Vsh : Bc*D ]
//     Qsh[r*D+d] holds query r's row; Osh[r*D+d] is query r's O accumulator.
//     NO per-thread D-length arrays exist any more -> no register spill. Each
//     thread still owns only TWO scalars in registers: its query's running
//     online-softmax max `m` and denom `l`.
//   * The block loops over KEY TILES of Bc keys. Each K/V tile is loaded into
//     shared memory ONCE (cooperatively) and reused by ALL Br queries, so K/V
//     are read from HBM O(N/Bc) times per (b,h) instead of O(N) times.
//   * After all key tiles: O = Osh/l, LSE = m + log(l), written by the owning
//     thread.
//
// CAUSAL TILE BOOKKEEPING (the easy place to get a bug):
//   For query row i and key index j the contribution exists iff j <= i.
//   Process key tiles kt, key index j = kt*Bc + c:
//     * qmax = min(qtile*Br + Br - 1, N-1) is the MAX query row in the block.
//       A key tile whose MINIMUM key index kt*Bc exceeds qmax is strictly above
//       the diagonal for EVERY query in the block -> skip. Loop bound
//       kt <= qmax/Bc depends only on qtile (block-uniform), so every thread
//       runs the SAME number of key-tile iterations -> __syncthreads() uniform.
//     * Within a kept tile a (query i, key j) pair contributes iff j <= i AND
//       j < N. Applied as a per-element `continue` on the score loop (the
//       score is simply never folded in), NOT an early return out of the tile
//       loop, so barriers stay uniform.
//
// ONLINE-SOFTMAX RESCALING (must match the reference numerics so the backward,
// which recomputes P = exp(scale*QK - LSE), stays consistent):
//   For each new score s, m_new = max(m, s); corr = exp(m_old - m_new);
//   l = l*corr + exp(s - m_new); Osh[r,:] = Osh[r,:]*corr + exp(s-m_new)*V[j,:].
//   The FINAL LSE = m + log(l) uses the FINAL m,l. The first score seen has
//   m_old=-inf so corr=exp(-inf)=0 (no NaN: 0*anything stored is the prior 0
//   accumulator), p=exp(0)=1. Every query attends at least key j=i, so l>=1 and
//   log(l) is finite. expf (not __expf) keeps LSE tight.
//
// RACE-FREE WRITES: each query row's O and LSE is owned by exactly one thread
// in exactly one block -> no atomics needed in the forward.
//
// =====================================================================
// BACKWARD: KEY-TILED FLASH BACKWARD (dK/dV block-owned, dQ via atomicAdd)
// =====================================================================
// Delta[b,h,i] = Sum_d O[i,d]*dO[i,d] is computed first by a cheap per-row
// reduction kernel (flash_delta_kernel). The main backward is then tiled:
//
//   * One block owns a KEY TILE of Bc keys for a fixed (b,h):
//     grid.x = B*H*ceil(N/Bc). blockDim.x = Bc: thread c owns key
//     j = ktile*Bc + c (one thread = one key). Each thread keeps its key's
//     K_j[D] and V_j[D] staged, and OWNS dK_j and dV_j: because exactly ONE
//     block touches a given key tile, those dK/dV accumulations are race-free
//     plain `+=` (accumulated in shared mem across query tiles, flushed once).
//   * The block loops over QUERY TILES of Bq rows with i >= j (causal). Each
//     Q/dO tile is staged in shared memory once (cooperatively) and reused by
//     all Bc keys. For each query i in the tile and each key j in the block:
//         s_ij  = Q_i . K_j ;  P_ij = exp(scale*s_ij - LSE[i])   (matches fwd)
//         dp_ij = dO_i . V_j ;  ds_ij = P_ij * (dp_ij - Delta[i])
//         dV_j += P_ij  * dO_i           (block-owned, plain +=)
//         dK_j += scale*ds_ij * Q_i      (block-owned, plain +=)
//         dQ_i += scale*ds_ij * K_j      (SHARED across key-tile blocks)
//   * ATOMICS — dQ ONLY. dQ[i,:] is touched by every key-tile block whose keys
//     are <= i (and, within one block, by every key-thread c that attends i),
//     so dQ accumulation MUST use atomicAdd. dK/dV are each owned by a single
//     block (the one owning that key tile) and written by a single thread (the
//     one owning that key), so they use plain `+=` with no race. atomicAdd on
//     fp32 global memory is supported and gives a correct (order-independent)
//     sum here because addition is associative up to fp rounding, well within
//     the test tolerance.
//
//   CAUSAL TILE BOOKKEEPING (mirrors the forward):
//     * A query tile whose MAX query row is below this block's MIN key index is
//       entirely above the diagonal -> skip. The first query tile that can
//       contribute is qt0 = (ktile*Bc)/Bq; query tiles qt < qt0 are skipped.
//       The loop runs qt = qt0 .. ceil(N/Bq)-1, a bound that depends only on
//       ktile (block-uniform) -> every thread runs the same iteration count ->
//       __syncthreads() uniform.
//     * Within a kept tile, a (query i, key j) pair contributes iff j <= i AND
//       i < N AND j < N, applied as a per-element guard (skip the update), not
//       an early return, so barriers stay uniform.
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

constexpr int THREADS = 128;            // threads per block (Delta kernel: one block per row)
constexpr int WARP = 32;
constexpr int MAX_WARPS = THREADS / WARP;

// ---- forward tiling parameters --------------------------------------------
// Br query rows per block, Bc keys per key-tile. blockDim.x = Br in the forward.
// Per-query state (Q row + O accumulator) lives in SHARED memory, not in
// per-thread arrays, so there is no DMAX register-array cap any more.
// Shared use = (2*Br + 2*Bc)*D floats. For D=64: (128+64)*64 = 12288 f = 48KB
// (at the default 48KB cap); for D=32 it is 24KB. The bench uses Dh=64.
constexpr int FWD_BR = 64;              // query rows per block (== forward blockDim.x)
constexpr int FWD_BC = 32;              // keys per key-tile (keeps K/V smem modest)

// ---- backward tiling parameters -------------------------------------------
// One block owns BWD_BC keys (blockDim.x = BWD_BC, one thread per key) and loops
// over query tiles of BWD_BQ rows. Shared use =
//   (BWD_BC + BWD_BQ) * D            (Q/dO tile)  ... see kernel for full layout
//   + 2*BWD_BC*D (dK/dV accumulators) + 2*BWD_BC*D (K/V staged) + 2*BWD_BQ*D.
// Sized in the launcher from the actual D. Kept small (32x32) to stay well under
// the 48KB cap for D up to 64.
// Shared use = (4*BWD_BC + 2*BWD_BQ)*D floats. For D=64: (128+32)*64 = 10240 f
// = 40KB, comfortably under the 48KB default dynamic-smem cap. For D=32: 20KB.
constexpr int BWD_BC = 32;              // keys per block (== backward blockDim.x)
constexpr int BWD_BQ = 16;              // query rows per query-tile

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

// ---- forward (tiled, shared-mem accumulator) -------------------------------
// One BLOCK per (b,h, query-tile). blockDim.x = Br; thread r owns query row
// q_row = qtile*Br + r. The block loops over key tiles of Bc keys, loading each
// K/V tile to shared memory once and reusing it across all Br queries.
//
// Dynamic shared layout (Br = blockDim.x, Bc = FWD_BC):
//   [ Qsh : Br*D ][ Osh : Br*D ][ Ksh : Bc*D ][ Vsh : Bc*D ]
// Qsh[r*D+d] is query r's row; Osh[r*D+d] is query r's O accumulator. These
// replace the old per-thread q[DMAX]/acc[DMAX] LOCAL arrays (which spilled).
// Each thread keeps only two scalar registers m,l for its own query.
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
    float* Qsh = smem;                       // Br*D floats
    float* Osh = Qsh + (int64_t)Br * D;      // Br*D floats
    float* Ksh = Osh + (int64_t)Br * D;      // Bc*D floats
    float* Vsh = Ksh + (int64_t)Bc * D;      // Bc*D floats

    // Stage this thread's query row into shared mem and zero its O accumulator.
    // Inactive threads (padding in the last query-tile) zero theirs but still
    // participate in every barrier. q[] is read only by the owning thread, so
    // a single __syncthreads() before the key loop is enough for visibility.
    float* qrow_sh = Qsh + (int64_t)tid * D; // this thread's staged query row
    float* orow_sh = Osh + (int64_t)tid * D; // this thread's O accumulator
    if (active) {
        const float* Qrow = Qbase + q_row * D;
        for (int d = 0; d < D; ++d) { qrow_sh[d] = Qrow[d]; orow_sh[d] = 0.0f; }
    } else {
        for (int d = 0; d < D; ++d) { qrow_sh[d] = 0.0f; orow_sh[d] = 0.0f; }
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
        // each, strided by blockDim.x. Out-of-range keys (k0+c >= N) are zeroed
        // (and also masked later via the per-element causal guard).
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
        __syncthreads();                     // tile (and on kt==0, Qsh/Osh) ready

        // Each active thread folds this tile's Bc keys into its online softmax,
        // operating on its query's shared-mem O accumulator orow_sh[].
        if (active) {
            for (int c = 0; c < Bc; ++c) {
                int64_t kj = k0 + c;
                // CAUSAL + bounds mask: contributes iff kj <= q_row and kj < N.
                if (kj > q_row || kj >= N) continue;
                const float* Krow = Ksh + (int64_t)c * D;
                float s = 0.0f;
                for (int d = 0; d < D; ++d) s += qrow_sh[d] * Krow[d];
                s *= scale;
                // online-softmax update with rescaling on a new running max
                float m_new = fmaxf(m, s);
                float corr  = expf(m - m_new);   // exp(m_old - m_new); 1 if m unchanged
                float p     = expf(s - m_new);
                l = l * corr + p;
                const float* Vrow = Vsh + (int64_t)c * D;
                for (int d = 0; d < D; ++d) orow_sh[d] = orow_sh[d] * corr + p * Vrow[d];
                m = m_new;
            }
        }
        __syncthreads();                     // all done reading tile before reload
    }

    // Finalize: O = Osh / l, LSE = m + log(l). Each thread owns its own row.
    if (active) {
        float inv_l = 1.0f / l;
        float* Orow = Obase + q_row * D;
        for (int d = 0; d < D; ++d) Orow[d] = orow_sh[d] * inv_l;
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

// ---- backward: tiled dK/dV/dQ ---------------------------------------------
// One BLOCK owns a KEY TILE of Bc = blockDim.x keys for a fixed (b,h); thread c
// owns key j = ktile*Bc + c. The block loops over QUERY TILES of Bq rows with
// i >= j (causal), staging each Q/dO tile in shared memory and reusing it across
// all Bc keys. For each (query i, key j):
//     s_ij  = Q_i . K_j ;  P_ij = exp(scale*s_ij - LSE[i])      (matches fwd)
//     dp_ij = dO_i . V_j ;  ds_ij = P_ij * (dp_ij - Delta[i])
//     dV_j += P_ij  * dO_i           (block-owned, plain +=, in dVsh)
//     dK_j += scale*ds_ij * Q_i      (block-owned, plain +=, in dKsh)
//     dQ_i += scale*ds_ij * K_j      (atomicAdd: dQ_i is shared across blocks)
//
// dK/dV: each key tile is owned by exactly ONE block and each key by exactly ONE
// thread, so dKsh/dVsh accumulate race-free with plain += and are flushed to
// global once at the end. dQ: a given dQ[i,:] is touched by every key-tile block
// with keys <= i (and by every key-thread within a block), so it MUST use
// atomicAdd. atomicAdd on fp32 global is supported and order-independent up to
// fp rounding (within tolerance).
//
// Dynamic shared layout (Bc = blockDim.x = BWD_BC, Bq = BWD_BQ):
//   [ Ksh : Bc*D ][ Vsh : Bc*D ][ dKsh : Bc*D ][ dVsh : Bc*D ]
//   [ Qsh : Bq*D ][ dOsh : Bq*D ]
__global__ void flash_bwd_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 const float* __restrict__ dO,
                                 float* __restrict__ dQ,
                                 float* __restrict__ dK,
                                 float* __restrict__ dV,
                                 const float* __restrict__ LSE,
                                 const float* __restrict__ Delta,
                                 float scale,
                                 int64_t B, int64_t H, int64_t N, int64_t D) {
    const int Bc = blockDim.x;               // keys per block
    const int Bq = BWD_BQ;                   // queries per query-tile
    const int tid = threadIdx.x;             // local key index c in [0,Bc)

    const int64_t n_ktiles = (N + Bc - 1) / Bc;
    const int64_t bh    = blockIdx.x / n_ktiles;
    const int64_t ktile = blockIdx.x % n_ktiles;
    const int64_t j     = ktile * Bc + tid;  // global key this thread owns
    const bool kactive  = (j < N);

    const float* Qbase  = Q  + bh * N * D;
    const float* Kbase  = K  + bh * N * D;
    const float* Vbase  = V  + bh * N * D;
    const float* dObase = dO + bh * N * D;
    float* dQbase = dQ + bh * N * D;
    float* dKbase = dK + bh * N * D;
    float* dVbase = dV + bh * N * D;
    const float* LSErow   = LSE   + bh * N;
    const float* Deltarow = Delta + bh * N;

    extern __shared__ float smem[];
    float* Ksh  = smem;                       // Bc*D
    float* Vsh  = Ksh  + (int64_t)Bc * D;     // Bc*D
    float* dKsh = Vsh  + (int64_t)Bc * D;     // Bc*D  (accumulator)
    float* dVsh = dKsh + (int64_t)Bc * D;     // Bc*D  (accumulator)
    float* Qsh  = dVsh + (int64_t)Bc * D;     // Bq*D
    float* dOsh = Qsh  + (int64_t)Bq * D;     // Bq*D

    // Stage this thread's key/value row; zero its dK/dV accumulators. Inactive
    // key threads (j >= N) zero theirs but still hit every barrier.
    float* ksh_row  = Ksh  + (int64_t)tid * D;
    float* vsh_row  = Vsh  + (int64_t)tid * D;
    float* dksh_row = dKsh + (int64_t)tid * D;
    float* dvsh_row = dVsh + (int64_t)tid * D;
    if (kactive) {
        const float* Krow = Kbase + j * D;
        const float* Vrow = Vbase + j * D;
        for (int d = 0; d < D; ++d) { ksh_row[d] = Krow[d]; vsh_row[d] = Vrow[d]; }
    } else {
        for (int d = 0; d < D; ++d) { ksh_row[d] = 0.0f; vsh_row[d] = 0.0f; }
    }
    for (int d = 0; d < D; ++d) { dksh_row[d] = 0.0f; dvsh_row[d] = 0.0f; }

    // Causal: this block's MIN key index is k0 = ktile*Bc, so only queries with
    // i >= k0 can attend any key in the tile. The first query tile that can
    // contribute is qt0 = k0/Bq. The loop bound depends only on ktile (and N),
    // hence is block-uniform -> all __syncthreads() are reached uniformly.
    const int64_t k0  = ktile * (int64_t)Bc;
    const int64_t qt0 = k0 / Bq;
    const int64_t n_qtiles = (N + Bq - 1) / Bq;

    for (int64_t qt = qt0; qt < n_qtiles; ++qt) {
        const int64_t i0 = qt * Bq;          // first query of this tile
        // Cooperatively stage the Q and dO tiles (Bq*D elements each).
        for (int e = tid; e < Bq * D; e += Bc) {
            int r = e / D;                   // query-within-tile
            int d = e % D;
            int64_t ii = i0 + r;
            if (ii < N) {
                Qsh[e]  = Qbase[ii * D + d];
                dOsh[e] = dObase[ii * D + d];
            } else {
                Qsh[e]  = 0.0f;
                dOsh[e] = 0.0f;
            }
        }
        __syncthreads();                     // Q/dO tile ready (uniform)

        // Each active key thread walks the Bq queries of this tile.
        if (kactive) {
            for (int r = 0; r < Bq; ++r) {
                int64_t i = i0 + r;
                // CAUSAL + bounds: contributes iff i < N and j <= i.
                if (i >= N || j > i) continue;
                const float* qrow  = Qsh  + (int64_t)r * D;
                const float* dorow = dOsh + (int64_t)r * D;
                // s_ij = Q_i . K_j ; dp_ij = dO_i . V_j
                float s = 0.0f, dp = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s  += qrow[d]  * ksh_row[d];
                    dp += dorow[d] * vsh_row[d];
                }
                float p  = expf(scale * s - LSErow[i]);
                float ds = p * (dp - Deltarow[i]);
                float coef = scale * ds;
                // dV_j += p*dO_i ; dK_j += coef*Q_i  (block-owned, plain +=).
                // dQ_i += coef*K_j  (shared across blocks -> atomicAdd).
                float* dQrow = dQbase + i * D;
                for (int d = 0; d < D; ++d) {
                    dvsh_row[d] += p    * dorow[d];
                    dksh_row[d] += coef * qrow[d];
                    atomicAdd(&dQrow[d], coef * ksh_row[d]);
                }
            }
        }
        __syncthreads();                     // done reading Q/dO tile before reload
    }

    // Flush this key tile's dK/dV accumulators to global (block-owned -> +=).
    // Callers pass zeroed grads, so a plain store of the accumulated sum is the
    // correct contribution from this single owning block.
    if (kactive) {
        float* dKrow = dKbase + j * D;
        float* dVrow = dVbase + j * D;
        for (int d = 0; d < D; ++d) {
            dKrow[d] += dksh_row[d];
            dVrow[d] += dvsh_row[d];
        }
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
    // Qsh[Br*D] + Osh[Br*D] + Ksh[Bc*D] + Vsh[Bc*D], sized from the actual D.
    size_t smem = (size_t)(2 * Br + 2 * Bc) * D * sizeof(float);
    // At D=64 this is exactly 48KB, which sits right at the default dynamic-smem
    // cap — opt into the larger per-block limit so the launch is guaranteed
    // (Volta+; harmless when the request is at/under the default).
    if (smem >= 48 * 1024)
        cudaFuncSetAttribute(flash_forward_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    flash_forward_kernel<<<blocks, Br, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), LSE.data_ptr<float>(), (float)scale, B, H, N, D);
}

void flash_attention_backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              torch::Tensor O, torch::Tensor dO,
                              torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV,
                              torch::Tensor LSE, double scale,
                              int64_t B, int64_t H, int64_t N, int64_t D) {
    auto Delta = torch::empty({B, H, N}, LSE.options());

    // Delta[b,h,i] = sum_d O[i,d]*dO[i,d]: one block per (b,h,row), unchanged.
    int64_t total = B * H * N;
    size_t smem_delta = MAX_WARPS * sizeof(float);
    flash_delta_kernel<<<total, THREADS, smem_delta>>>(
        O.data_ptr<float>(), dO.data_ptr<float>(), Delta.data_ptr<float>(),
        B, H, N, D);

    // Tiled backward: one block per (b,h, key-tile of BWD_BC keys); blockDim.x =
    // BWD_BC (one thread per key). Each block loops over query tiles of BWD_BQ
    // rows, staging Q/dO tiles in shared memory. dK/dV are block-owned (plain
    // +=); dQ is accumulated via atomicAdd (shared across key-tile blocks).
    const int Bc = BWD_BC;
    const int Bq = BWD_BQ;
    int64_t n_ktiles = (N + Bc - 1) / Bc;
    int64_t bwd_blocks = B * H * n_ktiles;
    // Ksh+Vsh+dKsh+dVsh (4*Bc*D) + Qsh+dOsh (2*Bq*D), sized from the actual D.
    size_t smem_bwd = (size_t)(4 * Bc + 2 * Bq) * D * sizeof(float);
    // 40KB at D=64 (under the default cap), but opt in for larger D for safety.
    if (smem_bwd >= 48 * 1024)
        cudaFuncSetAttribute(flash_bwd_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bwd);
    flash_bwd_kernel<<<bwd_blocks, Bc, smem_bwd>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), dQ.data_ptr<float>(), dK.data_ptr<float>(),
        dV.data_ptr<float>(), LSE.data_ptr<float>(), Delta.data_ptr<float>(),
        (float)scale, B, H, N, D);
}
