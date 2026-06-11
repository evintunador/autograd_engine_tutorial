// Matmul CUDA kernels for cudagrad: forward + backward (dA, dB).
//
// REGISTER-BLOCKED (a.k.a. thread-tiled) shared-memory GEMM (tutorial). This is
// the standard next step up from single-element tiling and is the classic 2-4x
// win, because it raises arithmetic intensity: each value loaded from shared
// memory is reused across several outputs instead of just one.
//
// Two levels of tiling:
//   * BLOCK TILE  (BM x BN): the chunk of the output matrix one threadblock owns.
//     We contract over the shared dimension in BK-wide steps; for each step we
//     cooperatively stage an A-subtile (BM x BK) and a B-subtile (BK x BN) into
//     __shared__ memory.
//   * MICRO TILE  (TM x TN): the small block of outputs ONE THREAD owns, held in
//     a TM x TN register accumulator array. So the block has
//     (BN/TN) x (BM/TM) threads. With BM=BN=64, TM=TN=4 that is 16x16 = 256
//     threads, and each thread owns a 4x4 patch of the 64x64 output tile.
//
// Inner loop (the "outer product" accumulation): for each of the BK contraction
// positions we load this thread's TM-long column slice of the A-subtile and its
// TN-long row slice of the B-subtile into registers, then do the TM x TN
// outer-product MACs into the accumulators. Each of those TM (resp. TN) register
// values feeds TN (resp. TM) multiplies => the reuse that makes this fast.
//
// COOPERATIVE LOADS when #threads != #tile-elements: a block has BLOCK_THREADS
// threads but each shared subtile has BM*BK (or BK*BN) elements, which need not
// match. We flatten both the thread set and the subtile and walk the subtile in
// strides of BLOCK_THREADS, so every element is loaded by exactly one thread and
// each thread may load several. This is fully general for any of these counts.
//
// Each thread owns a DISTINCT set of output elements, so forward uses plain
// writes and backward uses plain `+=` (read-modify-write) into pre-zeroed grads
// — NO atomics, NO races.
//
// Layout/contract notes (all tensors contiguous, row-major, fp32):
//   * A has shape (..., M, K); flatten the leading batch dims to Bsz.
//   * Two B layouts are supported, detected from dim() in the launcher:
//       - BATCHED  : B is (..., K, N) with the SAME leading dims as A
//                    (b.dim() == a.dim()).  C[b,m,n] = Σ_k A[b,m,k]·B[b,k,n].
//       - SHARED   : B is 2-D (K, N), broadcast across the batch
//                    (b.dim() < a.dim()).   C[b,m,n] = Σ_k A[b,m,k]·B[k,n].
//     Plain 2-D@2-D is just Bsz==1 of the batched case.
//   * Flat row-major offsets:
//       A[b,m,k]         = A_ptr[(b*M + m)*K + k]
//       B_batched[b,k,n] = B_ptr[(b*K + k)*N + n]
//       B_shared[k,n]    = B_ptr[k*N + n]
//       C[b,m,n]         = C_ptr[(b*M + m)*N + n]
//   * `shared` (bool as int): 1 -> B is shared (2-D), 0 -> B is batched.
//
// The batch dimension is mapped onto blockIdx.z, so one grid covers all Bsz
// batches; for the SHARED-B forward/dA kernels every batch reads the same B.
//
// CRITICAL: M, N, K are NOT assumed to be multiples of the block/micro tiles
// (benchmark uses 384, 1152, ...; tests use M=8,K=16,N=8). Every global load is
// bounds-checked (loads 0.0f into shared memory for out-of-range rows/cols, so
// the padded lanes contribute nothing to the dot product) and every output write
// is guarded. All threads of a block run the same loop trip counts (they depend
// only on the uniform M/N/K), so every __syncthreads() is reached uniformly —
// we never early-return a thread before a sync.
//
// Backward launchers ACCUMULATE into zero-initialized grads (`+=`). For the
// SHARED-B case the dB kernel sums over BOTH the batch and M dims (that batch
// sum is exactly what makes the linear-layer weight grad correct) — preserved
// here by looping the M-contraction over every batch into the one (K,N) grad.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

// Block tile / micro tile / contraction step. blockDim is (BN/TN, BM/TM).
constexpr int BM = 64;            // output rows per block tile
constexpr int BN = 64;            // output cols per block tile
constexpr int BK = 8;             // contraction step
constexpr int TM = 4;             // output rows per thread (micro-tile height)
constexpr int TN = 4;             // output cols per thread (micro-tile width)
constexpr int BLOCK_DIM_X = BN / TN;             // 16
constexpr int BLOCK_DIM_Y = BM / TM;             // 16
constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;  // 256

// ===========================================================================
// Generic register-blocked tile engine.
//
// Every one of the three kernels is the same shape: an output matrix
//   OUT[row, col]  with  row in [0, ROWS), col in [0, COLS)
// formed by contracting over CONTRACT, reading from two operands:
//   LHS[row, c]    (the "A side", indexed [output-row][contract])
//   RHS[c, col]    (the "B side", indexed [contract][output-col])
// so that OUT[row,col] += Σ_c LHS[row,c] * RHS[c,col].
//
// This helper accumulates ONE block tile's worth of that product (BM x BN
// outputs owned by the block, this thread owning a TM x TN micro-tile) into the
// `acc[TM][TN]` register array. It bounds-checks every global load against the
// caller-supplied extents. The caller supplies row-major strides + base offsets
// for LHS and RHS so the same engine serves all three operand layouts.
//
//   lhs[r, c] = lhs_ptr[lhs_base + (block_row_base + r) * lhs_rs + c]   (r is an
//               output-row offset within the tile, c a contraction offset)
//   rhs[c, n] = rhs_ptr[rhs_base + c * rhs_rs + (block_col_base + n)]   (n is an
//               output-col offset within the tile)
//
// block_row_base = blockIdx.y * BM, block_col_base = blockIdx.x * BN. The caller
// passes the contraction extent CONTRACT and the row/col extents (ROWS, COLS)
// for bounds-checking.
// ===========================================================================
__device__ __forceinline__ void tile_engine(
        const float* __restrict__ lhs_ptr, int64_t lhs_base, int64_t lhs_rs,
        const float* __restrict__ rhs_ptr, int64_t rhs_base, int64_t rhs_rs,
        int64_t ROWS, int64_t COLS, int64_t CONTRACT,
        int64_t block_row_base, int64_t block_col_base,
        float acc[TM][TN]) {
    __shared__ float As[BM][BK];   // LHS subtile: [output-row][contract]
    __shared__ float Bs[BK][BN];   // RHS subtile: [contract][output-col]

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..BLOCK_THREADS-1

    // This thread's micro-tile origin inside the block tile.
    const int thread_row0 = threadIdx.y * TM;   // 0..BM-TM
    const int thread_col0 = threadIdx.x * TN;   // 0..BN-TN

    int64_t n_steps = (CONTRACT + BK - 1) / BK;
    for (int64_t t = 0; t < n_steps; ++t) {
        int64_t c_base = t * BK;   // contraction offset for this step

        // --- Cooperative load of the LHS subtile (BM x BK) -----------------
        // Flatten the BM*BK elements; each thread strides by BLOCK_THREADS.
        for (int i = tid; i < BM * BK; i += BLOCK_THREADS) {
            int r = i / BK;                 // output-row within tile
            int c = i % BK;                 // contraction within step
            int64_t gr = block_row_base + r;
            int64_t gc = c_base + c;
            As[r][c] = (gr < ROWS && gc < CONTRACT)
                       ? lhs_ptr[lhs_base + gr * lhs_rs + gc]
                       : 0.0f;
        }
        // --- Cooperative load of the RHS subtile (BK x BN) -----------------
        for (int i = tid; i < BK * BN; i += BLOCK_THREADS) {
            int c = i / BN;                 // contraction within step
            int n = i % BN;                 // output-col within tile
            int64_t gc = c_base + c;
            int64_t gn = block_col_base + n;
            Bs[c][n] = (gc < CONTRACT && gn < COLS)
                       ? rhs_ptr[rhs_base + gc * rhs_rs + gn]
                       : 0.0f;
        }

        __syncthreads();   // both subtiles fully staged

        // --- Outer-product accumulation over the BK contraction positions --
        #pragma unroll
        for (int cc = 0; cc < BK; ++cc) {
            float a_reg[TM];   // this thread's TM-long slice of A's column cc
            float b_reg[TN];   // this thread's TN-long slice of B's row cc
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[thread_row0 + i][cc];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[cc][thread_col0 + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();   // done reading shared before next step overwrites it
    }
}

// ---------------------------------------------------------------------------
// FORWARD: C[b,m,n] = Σ_k A[b,m,k] · B[(b),k,n]
// OUT=(M,N), contract K. LHS = A[b] (row=m, c=k, rs=K); RHS = B[(b)] (c=k,
// col=n, rs=N). blockIdx.z -> batch.
// ---------------------------------------------------------------------------
__global__ void matmul_forward_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int64_t Bsz, int64_t M, int64_t K, int64_t N,
                                     int shared) {
    int64_t b = blockIdx.z;
    int64_t block_row_base = (int64_t)blockIdx.y * BM;   // m base
    int64_t block_col_base = (int64_t)blockIdx.x * BN;   // n base

    int64_t lhs_base = b * M * K;                        // A[b,:,:]
    int64_t rhs_base = shared ? 0 : b * K * N;           // B[(b),:,:]

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    tile_engine(A, lhs_base, /*lhs_rs=*/K,
                B, rhs_base, /*rhs_rs=*/N,
                /*ROWS=*/M, /*COLS=*/N, /*CONTRACT=*/K,
                block_row_base, block_col_base, acc);

    // Write the TM x TN micro-tile; every element bounds-checked (distinct elem).
    int thread_row0 = threadIdx.y * TM;
    int thread_col0 = threadIdx.x * TN;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int64_t m = block_row_base + thread_row0 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int64_t n = block_col_base + thread_col0 + j;
            if (n < N) C[(b * M + m) * N + n] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// dA: dA[b,m,k] += Σ_n dC[b,m,n] · B[(b),k,n]   == dC @ B^T
// OUT=(M,K), contract N. LHS = dC[b] (row=m, contract=n, rs=N) — plain row-major.
// RHS wants rhs[c=n][col=k] = B[(b),k,n], but B is stored row-major as
// B[k*N + n], i.e. TRANSPOSED relative to the generic engine's
// rhs[c][col] = base + c*rhs_rs + col (which assumes unit col stride). The
// transpose can't be expressed through that addressing, so dA uses its own
// engine variant whose RHS cooperative load reads B[gk*N + gn] into Bs[n][k].
// blockIdx.z -> batch. threadIdx maps to (m micro-rows, k micro-cols).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tile_engine_dA(
        const float* __restrict__ dC, int64_t dc_base,   // dC[b], row=m, c=n, rs=N
        const float* __restrict__ B, int64_t b_base,     // B[(b)], B[k,n]=base+k*N+n
        int64_t M, int64_t K, int64_t N,
        int64_t block_row_base, int64_t block_col_base,  // m base, k base
        float acc[TM][TN]) {
    __shared__ float dCs[BM][BK];  // [m within tile][n within step]
    __shared__ float Bs[BK][BN];   // [n within step][k within tile]

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_row0 = threadIdx.y * TM;   // m offset in tile
    const int thread_col0 = threadIdx.x * TN;   // k offset in tile

    int64_t n_steps = (N + BK - 1) / BK;         // contract over N
    for (int64_t t = 0; t < n_steps; ++t) {
        int64_t c_base = t * BK;                 // n offset for this step

        // dCs[m][n] = dC[b, m, n] = dC[dc_base + m*N + n]
        for (int i = tid; i < BM * BK; i += BLOCK_THREADS) {
            int r = i / BK;            // m within tile
            int c = i % BK;            // n within step
            int64_t gm = block_row_base + r;
            int64_t gn = c_base + c;
            dCs[r][c] = (gm < M && gn < N) ? dC[dc_base + gm * N + gn] : 0.0f;
        }
        // Bs[n][k] = B[(b), k, n] = B[b_base + k*N + n]   (transposed load)
        for (int i = tid; i < BK * BN; i += BLOCK_THREADS) {
            int c = i / BN;            // n within step
            int n_k = i % BN;          // k within tile
            int64_t gn = c_base + c;
            int64_t gk = block_col_base + n_k;
            Bs[c][n_k] = (gn < N && gk < K) ? B[b_base + gk * N + gn] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int cc = 0; cc < BK; ++cc) {     // cc indexes n within the step
            float a_reg[TM];   // dC slice over m
            float b_reg[TN];   // B slice over k
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = dCs[thread_row0 + i][cc];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[cc][thread_col0 + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }
}

__global__ void matmul_backward_dA_kernel(const float* __restrict__ B,
                                         float* __restrict__ dA,
                                         const float* __restrict__ dC,
                                         int64_t Bsz, int64_t M, int64_t K,
                                         int64_t N, int shared) {
    int64_t b = blockIdx.z;
    int64_t block_row_base = (int64_t)blockIdx.y * BM;   // m base
    int64_t block_col_base = (int64_t)blockIdx.x * BN;   // k base

    int64_t dc_base = b * M * N;
    int64_t b_base = shared ? 0 : b * K * N;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    tile_engine_dA(dC, dc_base, B, b_base, M, K, N,
                   block_row_base, block_col_base, acc);

    int thread_row0 = threadIdx.y * TM;   // m
    int thread_col0 = threadIdx.x * TN;   // k
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int64_t m = block_row_base + thread_row0 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int64_t k = block_col_base + thread_col0 + j;
            if (k < K) dA[(b * M + m) * K + k] += acc[i][j];   // ACCUMULATE
        }
    }
}

// ---------------------------------------------------------------------------
// dB: dB[(b),k,n] += Σ_m A[b,m,k] · dC[b,m,n]   == A^T @ dC
// OUT=(K,N), contract M (and, for SHARED B, sum over the batch too).
// LHS must be indexed [row=k][c=m] = A[b, m, k] (A stored A[m*K + k], so a
// transposed load); RHS = dC[b] indexed [c=m][col=n] = dC[m*N + n] (plain row-
// major). For BATCHED B each block owns one batch (blockIdx.z); for SHARED B a
// single (K,N) grad sums over ALL batches, so we loop b in [0,Bsz) with grid
// z==1 — preserving the batch sum that makes the linear weight grad correct.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tile_engine_dB_onebatch(
        const float* __restrict__ A, int64_t a_base,     // A[b], A[m,k]=base+m*K+k
        const float* __restrict__ dC, int64_t dc_base,   // dC[b], dC[m,n]=base+m*N+n
        int64_t M, int64_t K, int64_t N,
        int64_t block_row_base, int64_t block_col_base,  // k base, n base
        float acc[TM][TN]) {
    __shared__ float As[BK][BM];   // [m within step][k within tile]
    __shared__ float dCs[BK][BN];  // [m within step][n within tile]

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_row0 = threadIdx.y * TM;   // k offset in tile
    const int thread_col0 = threadIdx.x * TN;   // n offset in tile

    int64_t m_steps = (M + BK - 1) / BK;         // contract over M
    for (int64_t t = 0; t < m_steps; ++t) {
        int64_t c_base = t * BK;                 // m offset for this step

        // As[m][k] = A[b, m, k] = A[a_base + m*K + k]   (transposed load)
        for (int i = tid; i < BK * BM; i += BLOCK_THREADS) {
            int c = i / BM;            // m within step
            int r = i % BM;            // k within tile
            int64_t gm = c_base + c;
            int64_t gk = block_row_base + r;
            As[c][r] = (gm < M && gk < K) ? A[a_base + gm * K + gk] : 0.0f;
        }
        // dCs[m][n] = dC[b, m, n] = dC[dc_base + m*N + n]
        for (int i = tid; i < BK * BN; i += BLOCK_THREADS) {
            int c = i / BN;            // m within step
            int n = i % BN;            // n within tile
            int64_t gm = c_base + c;
            int64_t gn = block_col_base + n;
            dCs[c][n] = (gm < M && gn < N) ? dC[dc_base + gm * N + gn] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int cc = 0; cc < BK; ++cc) {     // cc indexes m within the step
            float a_reg[TM];   // A slice over k
            float b_reg[TN];   // dC slice over n
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[cc][thread_row0 + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = dCs[cc][thread_col0 + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }
}

__global__ void matmul_backward_dB_kernel(const float* __restrict__ A,
                                         float* __restrict__ dB,
                                         const float* __restrict__ dC,
                                         int64_t Bsz, int64_t M, int64_t K,
                                         int64_t N, int shared) {
    int64_t block_row_base = (int64_t)blockIdx.y * BM;   // k base
    int64_t block_col_base = (int64_t)blockIdx.x * BN;   // n base

    int64_t batch_lo, batch_hi;
    if (shared) { batch_lo = 0; batch_hi = Bsz; }        // sum over ALL batches
    else        { batch_lo = blockIdx.z; batch_hi = blockIdx.z + 1; }

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    // Loop over the contributing batches, accumulating into the SAME registers
    // (this is the batch sum for SHARED B; a single iteration for BATCHED B).
    for (int64_t b = batch_lo; b < batch_hi; ++b) {
        int64_t a_base = b * M * K;
        int64_t dc_base = b * M * N;
        tile_engine_dB_onebatch(A, a_base, dC, dc_base, M, K, N,
                                block_row_base, block_col_base, acc);
    }

    int thread_row0 = threadIdx.y * TM;   // k
    int thread_col0 = threadIdx.x * TN;   // n
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int64_t k = block_row_base + thread_row0 + i;
        if (k >= K) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int64_t n = block_col_base + thread_col0 + j;
            if (n >= N) continue;
            // BATCHED writes the per-batch slice; SHARED writes the single (K,N).
            int64_t out_off = shared ? (k * N + n)
                                     : ((blockIdx.z * K + k) * N + n);
            dB[out_off] += acc[i][j];                    // ACCUMULATE
        }
    }
}

// Derive (Bsz, M, K, N, shared) from a (A-like) and b (B-like) tensors.
// shared <=> b.dim() < a.dim() (B is 2-D, broadcast across A's batch dims).
struct Dims { int64_t Bsz, M, K, N; int shared; };

inline Dims derive_dims(const torch::Tensor& a, const torch::Tensor& b) {
    int64_t M = a.size(-2);
    int64_t K = a.size(-1);
    int64_t N = b.size(-1);
    TORCH_CHECK(b.size(-2) == K, "matmul: inner dims must match (A's K vs B's K)");
    int64_t Bsz = a.numel() / (M * K);
    int shared = (b.dim() < a.dim()) ? 1 : 0;
    return {Bsz, M, K, N, shared};
}

// Grid helper: ceil-div an output extent into block-tile-wide blocks. `tile` is
// the block-tile extent along that axis (BM for rows, BN for cols).
inline unsigned int grid_dim(int64_t extent, int tile) {
    return (unsigned int)((extent + tile - 1) / tile);
}

} // namespace

void matmul_forward(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    Dims d = derive_dims(a, b);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    // x -> N block-tiles, y -> M block-tiles, z -> batch.
    dim3 grid(grid_dim(d.N, BN), grid_dim(d.M, BM), (unsigned int)d.Bsz);
    matmul_forward_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}

// dA: A's shape comes from dA; B's layout (batched/shared) comes from b.
void matmul_backward_dA(torch::Tensor b, torch::Tensor dA, torch::Tensor dC) {
    Dims d = derive_dims(dA, b);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    // Output is (M,K): x -> K block-tiles, y -> M block-tiles, z -> batch.
    dim3 grid(grid_dim(d.K, BN), grid_dim(d.M, BM), (unsigned int)d.Bsz);
    matmul_backward_dA_kernel<<<grid, block>>>(
        b.data_ptr<float>(), dA.data_ptr<float>(), dC.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}

// dB: A's shape comes from a; B's layout (batched/shared) comes from dB
// (dB.dim() == a.dim() -> batched, else shared).
void matmul_backward_dB(torch::Tensor a, torch::Tensor dB, torch::Tensor dC) {
    Dims d = derive_dims(a, dB);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    // Output is (K,N): x -> N block-tiles, y -> K block-tiles. z spans the batch
    // for the BATCHED case; for SHARED the kernel sums batches internally so z==1.
    unsigned int gz = d.shared ? 1u : (unsigned int)d.Bsz;
    dim3 grid(grid_dim(d.N, BN), grid_dim(d.K, BM), gz);
    matmul_backward_dB_kernel<<<grid, block>>>(
        a.data_ptr<float>(), dB.data_ptr<float>(), dC.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}
