// Matmul CUDA kernels for cudagrad: forward + backward (dA, dB).
//
// Naive, obviously-correct version (tutorial; test sizes are tiny, tolerances
// loose): ONE THREAD PER OUTPUT ELEMENT, each thread loops over the contracted
// dim. No tiling / shared memory. Each thread owns a distinct output element, so
// plain writes (forward) and plain `+=` (backward) suffice — NO atomics.
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
// Backward launchers ACCUMULATE into zero-initialized grads (`+=`). For the
// SHARED-B case the dB kernel sums over BOTH the batch and M dims (that batch
// sum is exactly what makes the linear-layer weight grad correct).
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 256;

inline int64_t n_blocks(int64_t n) { return (n + THREADS - 1) / THREADS; }

// C[b,m,n] = Σ_k A[b,m,k]·B[(b),k,n]   (one thread per (b,m,n); WRITES out)
__global__ void matmul_forward_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int64_t Bsz, int64_t M, int64_t K, int64_t N,
                                     int shared) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = Bsz * M * N;
    if (idx >= total) return;
    int64_t n = idx % N;
    int64_t m = (idx / N) % M;
    int64_t b = idx / (M * N);

    const float* Arow = A + (b * M + m) * K;          // A[b,m,:]
    const float* Bbase = shared ? B : (B + b * K * N); // B[(b),:,:]

    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        acc += Arow[k] * Bbase[k * N + n];             // B[(b),k,n]
    }
    C[(b * M + m) * N + n] = acc;
}

// dA[b,m,k] += Σ_n dC[b,m,n]·B[(b),k,n]   (one thread per (b,m,k); ACCUMULATES)
__global__ void matmul_backward_dA_kernel(const float* __restrict__ B,
                                         float* __restrict__ dA,
                                         const float* __restrict__ dC,
                                         int64_t Bsz, int64_t M, int64_t K,
                                         int64_t N, int shared) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = Bsz * M * K;
    if (idx >= total) return;
    int64_t k = idx % K;
    int64_t m = (idx / K) % M;
    int64_t b = idx / (M * K);

    const float* dCrow = dC + (b * M + m) * N;         // dC[b,m,:]
    const float* Bbase = shared ? B : (B + b * K * N); // B[(b),:,:]

    float acc = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        acc += dCrow[n] * Bbase[k * N + n];            // B[(b),k,n]
    }
    dA[(b * M + m) * K + k] += acc;
}

// Batched dB: dB[b,k,n] += Σ_m A[b,m,k]·dC[b,m,n]  (one thread per (b,k,n))
__global__ void matmul_backward_dB_batched_kernel(const float* __restrict__ A,
                                                 float* __restrict__ dB,
                                                 const float* __restrict__ dC,
                                                 int64_t Bsz, int64_t M,
                                                 int64_t K, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = Bsz * K * N;
    if (idx >= total) return;
    int64_t n = idx % N;
    int64_t k = (idx / N) % K;
    int64_t b = idx / (K * N);

    const float* Abase = A + b * M * K;                // A[b,:,:]
    const float* dCbase = dC + b * M * N;              // dC[b,:,:]

    float acc = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        acc += Abase[m * K + k] * dCbase[m * N + n];   // A[b,m,k]·dC[b,m,n]
    }
    dB[(b * K + k) * N + n] += acc;
}

// Shared (2-D) dB: dB[k,n] += Σ_b Σ_m A[b,m,k]·dC[b,m,n]  (one thread per (k,n);
// SUMS OVER THE BATCH — this is what makes the linear weight grad correct).
__global__ void matmul_backward_dB_shared_kernel(const float* __restrict__ A,
                                                float* __restrict__ dB,
                                                const float* __restrict__ dC,
                                                int64_t Bsz, int64_t M,
                                                int64_t K, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = K * N;
    if (idx >= total) return;
    int64_t n = idx % N;
    int64_t k = idx / N;

    float acc = 0.0f;
    for (int64_t b = 0; b < Bsz; ++b) {
        const float* Abase = A + b * M * K;            // A[b,:,:]
        const float* dCbase = dC + b * M * N;          // dC[b,:,:]
        for (int64_t m = 0; m < M; ++m) {
            acc += Abase[m * K + k] * dCbase[m * N + n];
        }
    }
    dB[k * N + n] += acc;
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

} // namespace

void matmul_forward(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    Dims d = derive_dims(a, b);
    int64_t total = d.Bsz * d.M * d.N;
    matmul_forward_kernel<<<n_blocks(total), THREADS>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}

// dA: A's shape comes from dA; B's layout (batched/shared) comes from b.
void matmul_backward_dA(torch::Tensor b, torch::Tensor dA, torch::Tensor dC) {
    Dims d = derive_dims(dA, b);
    int64_t total = d.Bsz * d.M * d.K;
    matmul_backward_dA_kernel<<<n_blocks(total), THREADS>>>(
        b.data_ptr<float>(), dA.data_ptr<float>(), dC.data_ptr<float>(),
        d.Bsz, d.M, d.K, d.N, d.shared);
}

// dB: A's shape comes from a; B's layout (batched/shared) comes from dB
// (dB.dim() == a.dim() -> batched, else shared).
void matmul_backward_dB(torch::Tensor a, torch::Tensor dB, torch::Tensor dC) {
    Dims d = derive_dims(a, dB);
    if (d.shared) {
        int64_t total = d.K * d.N;
        matmul_backward_dB_shared_kernel<<<n_blocks(total), THREADS>>>(
            a.data_ptr<float>(), dB.data_ptr<float>(), dC.data_ptr<float>(),
            d.Bsz, d.M, d.K, d.N);
    } else {
        int64_t total = d.Bsz * d.K * d.N;
        matmul_backward_dB_batched_kernel<<<n_blocks(total), THREADS>>>(
            a.data_ptr<float>(), dB.data_ptr<float>(), dC.data_ptr<float>(),
            d.Bsz, d.M, d.K, d.N);
    }
}
