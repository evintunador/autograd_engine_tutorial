// Matmul CUDA kernels for cudagrad: forward + backward (dA, dB).
//
// Shared-memory TILED GEMM (tutorial; single-level tiling). Each threadblock
// computes one TILE x TILE output tile. Threads cooperatively stage TILE-wide
// sub-tiles of the two input operands into __shared__ memory, __syncthreads(),
// then each thread accumulates the TILE-long partial dot product from shared
// memory. Looping that over the contracted dimension in TILE-wide chunks builds
// the full result. This reuses each loaded value TILE times (vs the naive
// one-thread-per-output version that re-reads every operand from global memory),
// which is the whole point of the optimization.
//
// Each thread still owns a DISTINCT output element, so forward uses plain writes
// and backward uses plain `+=` (read-modify-write) — NO atomics, NO races.
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
// CRITICAL: M, N, K are NOT assumed to be multiples of TILE (benchmark uses 384,
// 1152, ...; tests use tiny sizes). Every global load is bounds-checked (loads
// 0.0f into shared memory for out-of-range rows/cols, so the padded lanes
// contribute nothing to the dot product) and every output write is guarded.
//
// Backward launchers ACCUMULATE into zero-initialized grads (`+=`). For the
// SHARED-B case the dB kernel sums over BOTH the batch and M dims (that batch
// sum is exactly what makes the linear-layer weight grad correct) — preserved
// here by looping the K-contraction tiles over (Bsz * M) flattened rows.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int TILE = 16;  // TILE x TILE output tile == TILE x TILE threadblock.

// ---------------------------------------------------------------------------
// FORWARD: C[b,m,n] = Σ_k A[b,m,k] · B[(b),k,n]
// Output tile is (M,N); we contract over K. One block -> one TILE x TILE tile
// of C for one batch (blockIdx.z). threadIdx.(y,x) -> (row-in-tile, col-in-tile).
// ---------------------------------------------------------------------------
__global__ void matmul_forward_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int64_t Bsz, int64_t M, int64_t K, int64_t N,
                                     int shared) {
    int64_t b = blockIdx.z;                              // batch this block owns
    int64_t row = (int64_t)blockIdx.y * TILE + threadIdx.y;  // m
    int64_t col = (int64_t)blockIdx.x * TILE + threadIdx.x;  // n

    const float* Abatch = A + b * M * K;                 // A[b,:,:]
    const float* Bbase = shared ? B : (B + b * K * N);   // B[(b),:,:]

    __shared__ float As[TILE][TILE];   // staged A[b, row, k-chunk]
    __shared__ float Bs[TILE][TILE];   // staged B[(b), k-chunk, col]

    float acc = 0.0f;
    int64_t n_chunks = (K + TILE - 1) / TILE;            // K-tiles to walk
    for (int64_t t = 0; t < n_chunks; ++t) {
        int64_t k_a = t * TILE + threadIdx.x;            // k index for A's load
        int64_t k_b = t * TILE + threadIdx.y;            // k index for B's load

        // Bounds-check each load; pad with 0.0f so out-of-range lanes are inert.
        As[threadIdx.y][threadIdx.x] =
            (row < M && k_a < K) ? Abatch[row * K + k_a] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (k_b < K && col < N) ? Bbase[k_b * N + col] : 0.0f;

        __syncthreads();                                 // tile fully staged

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }

        __syncthreads();                                 // done before reload
    }

    if (row < M && col < N) {
        C[(b * M + row) * N + col] = acc;                // distinct elem -> write
    }
}

// ---------------------------------------------------------------------------
// dA: dA[b,m,k] += Σ_n dC[b,m,n] · B[(b),k,n]   == dC @ B^T
// Output tile is (M,K); we contract over N. blockIdx.z -> batch.
// threadIdx.(y,x) -> (row-in-tile = m, col-in-tile = k).
// We stage a TILE-wide chunk of dC[b, m, n] and of B[(b), k, n], indexed so that
// the shared-memory dot product runs over n.
// ---------------------------------------------------------------------------
__global__ void matmul_backward_dA_kernel(const float* __restrict__ B,
                                         float* __restrict__ dA,
                                         const float* __restrict__ dC,
                                         int64_t Bsz, int64_t M, int64_t K,
                                         int64_t N, int shared) {
    int64_t b = blockIdx.z;
    int64_t row = (int64_t)blockIdx.y * TILE + threadIdx.y;  // m
    int64_t col = (int64_t)blockIdx.x * TILE + threadIdx.x;  // k

    const float* dCbatch = dC + b * M * N;               // dC[b,:,:]
    const float* Bbase = shared ? B : (B + b * K * N);   // B[(b),:,:]

    __shared__ float dCs[TILE][TILE];  // staged dC[b, row, n-chunk]
    __shared__ float Bs[TILE][TILE];   // staged B[(b), col=k, n-chunk]

    float acc = 0.0f;
    int64_t n_chunks = (N + TILE - 1) / TILE;            // N-tiles to walk
    for (int64_t t = 0; t < n_chunks; ++t) {
        int64_t n_dc = t * TILE + threadIdx.x;           // n for dC load
        int64_t n_b = t * TILE + threadIdx.x;            // n for B load

        // dCs[row-in-tile][n-in-tile] = dC[b, row, n]
        dCs[threadIdx.y][threadIdx.x] =
            (row < M && n_dc < N) ? dCbatch[row * N + n_dc] : 0.0f;
        // Bs[k-in-tile][n-in-tile] = B[(b), k=col, n]   (k from blockIdx.y? no:
        // here the y index of this tile addresses the k owned by this block's
        // columns; threadIdx.y picks the k row of the B tile).
        int64_t k_b = (int64_t)blockIdx.x * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (k_b < K && n_b < N) ? Bbase[k_b * N + n_b] : 0.0f;

        __syncthreads();

        // acc += Σ_n dC[b,row,n] · B[(b),col,n].  dCs is indexed [m][n];
        // Bs is indexed [k][n]; this thread's k is threadIdx.x.
        #pragma unroll
        for (int nn = 0; nn < TILE; ++nn) {
            acc += dCs[threadIdx.y][nn] * Bs[threadIdx.x][nn];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        dA[(b * M + row) * K + col] += acc;              // ACCUMULATE
    }
}

// ---------------------------------------------------------------------------
// dB: dB[(b),k,n] += Σ_m A[b,m,k] · dC[b,m,n]   == A^T @ dC
// Output tile is (K,N); we contract over M (and, for SHARED B, over the batch).
// threadIdx.(y,x) -> (row-in-tile = k, col-in-tile = n).
//
// `batch_lo`/`batch_hi` select which batches contribute:
//   * BATCHED : each block owns one batch (blockIdx.z), so [b, b+1).
//   * SHARED  : a single (K,N) grad sums over ALL batches, so [0, Bsz) and
//               blockIdx.z is unused (grid z == 1) — this preserves the
//               batch-sum that makes the linear weight grad correct.
// We stage A[b, m, k] and dC[b, m, n] tiles and contract over m.
// ---------------------------------------------------------------------------
__global__ void matmul_backward_dB_kernel(const float* __restrict__ A,
                                         float* __restrict__ dB,
                                         const float* __restrict__ dC,
                                         int64_t Bsz, int64_t M, int64_t K,
                                         int64_t N, int shared) {
    int64_t row = (int64_t)blockIdx.y * TILE + threadIdx.y;  // k
    int64_t col = (int64_t)blockIdx.x * TILE + threadIdx.x;  // n

    int64_t batch_lo, batch_hi;
    if (shared) { batch_lo = 0; batch_hi = Bsz; }        // sum over ALL batches
    else        { batch_lo = blockIdx.z; batch_hi = blockIdx.z + 1; }

    __shared__ float As[TILE][TILE];   // staged A[b, m-chunk, k=row]
    __shared__ float dCs[TILE][TILE];  // staged dC[b, m-chunk, n=col]

    float acc = 0.0f;
    int64_t m_chunks = (M + TILE - 1) / TILE;            // M-tiles per batch
    for (int64_t b = batch_lo; b < batch_hi; ++b) {
        const float* Abatch = A + b * M * K;             // A[b,:,:]
        const float* dCbatch = dC + b * M * N;           // dC[b,:,:]
        for (int64_t t = 0; t < m_chunks; ++t) {
            int64_t m_a = t * TILE + threadIdx.y;        // m for A's load
            int64_t m_dc = t * TILE + threadIdx.y;       // m for dC's load

            // As[m-in-tile][k-in-tile] = A[b, m, k=col-of-block].  This thread's
            // k is blockIdx.y*TILE+threadIdx.x for the staged A tile.
            int64_t k_a = (int64_t)blockIdx.y * TILE + threadIdx.x;
            As[threadIdx.y][threadIdx.x] =
                (m_a < M && k_a < K) ? Abatch[m_a * K + k_a] : 0.0f;
            // dCs[m-in-tile][n-in-tile] = dC[b, m, n=col]
            dCs[threadIdx.y][threadIdx.x] =
                (m_dc < M && col < N) ? dCbatch[m_dc * N + col] : 0.0f;

            __syncthreads();

            // acc += Σ_m A[b,m,row=k] · dC[b,m,col=n].  As is indexed [m][k],
            // dCs is indexed [m][n]; this thread's k is threadIdx.y, n threadIdx.x.
            #pragma unroll
            for (int mm = 0; mm < TILE; ++mm) {
                acc += As[mm][threadIdx.y] * dCs[mm][threadIdx.x];
            }

            __syncthreads();
        }
    }

    if (row < K && col < N) {
        // BATCHED writes the per-batch slice; SHARED writes the single (K,N).
        int64_t out_off = shared ? (row * N + col)
                                 : ((blockIdx.z * K + row) * N + col);
        dB[out_off] += acc;                              // ACCUMULATE
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

// Grid helper: ceil-div an output extent into TILE-wide blocks.
inline unsigned int grid_dim(int64_t extent) {
    return (unsigned int)((extent + TILE - 1) / TILE);
}

} // namespace

void matmul_forward(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    Dims d = derive_dims(a, b);
    dim3 block(TILE, TILE);
    // x -> N tiles, y -> M tiles, z -> batch.
    dim3 grid(grid_dim(d.N), grid_dim(d.M), (unsigned int)d.Bsz);
    matmul_forward_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}

// dA: A's shape comes from dA; B's layout (batched/shared) comes from b.
void matmul_backward_dA(torch::Tensor b, torch::Tensor dA, torch::Tensor dC) {
    Dims d = derive_dims(dA, b);
    dim3 block(TILE, TILE);
    // Output is (M,K): x -> K tiles, y -> M tiles, z -> batch.
    dim3 grid(grid_dim(d.K), grid_dim(d.M), (unsigned int)d.Bsz);
    matmul_backward_dA_kernel<<<grid, block>>>(
        b.data_ptr<float>(), dA.data_ptr<float>(), dC.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}

// dB: A's shape comes from a; B's layout (batched/shared) comes from dB
// (dB.dim() == a.dim() -> batched, else shared).
void matmul_backward_dB(torch::Tensor a, torch::Tensor dB, torch::Tensor dC) {
    Dims d = derive_dims(a, dB);
    dim3 block(TILE, TILE);
    // Output is (K,N): x -> N tiles, y -> K tiles. z spans the batch for the
    // BATCHED case; for SHARED the kernel sums batches internally so z == 1.
    unsigned int gz = d.shared ? 1u : (unsigned int)d.Bsz;
    dim3 grid(grid_dim(d.N), grid_dim(d.K), gz);
    matmul_backward_dB_kernel<<<grid, block>>>(
        a.data_ptr<float>(), dB.data_ptr<float>(), dC.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}
