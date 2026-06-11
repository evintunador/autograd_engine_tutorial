// Flash-attention CUDA kernels for cudagrad: CAUSAL attention fwd + bwd.
//
// This is NOT tritongrad's tiled online-softmax flash kernel. It is a
// straightforward, obviously-correct CAUSAL attention computed per query row —
// the test sizes are tiny (B=1,H=2,N=128,D=32) so a plain non-tiled reference
// is plenty. Simplicity over peak perf, as elsewhere in cudagrad.
//
// Layout/contract notes:
//   * Q/K/V/O/dO/dQ/dK/dV are (B,H,N,D) contiguous fp32; LSE/Delta are (B,H,N).
//     Q[b,h,i,d] = Qptr[((b*H + h)*N + i)*D + d];  LSE[b,h,i] = LSEptr[(b*H+h)*N + i].
//   * CAUSAL masking: query i attends only to keys j <= i.
//   * score s_ij = scale * (Q_i . K_j)  — `scale` is the MULTIPLIER passed in
//     (= sqrt(D) in the suite), used verbatim (NOT 1/sqrt(D)).
//   * forward stores the per-row logsumexp LSE[i] = m_i + log(l_i) for reuse in
//     backward (so P_ij = exp(s_ij - LSE[i]) can be recomputed without m/l).
//   * backward ACCUMULATES into dQ/dK/dV (`+=`), so callers pass zeroed grads.
//     Each thread owns a DISTINCT output row (dQ_i / dK_j / dV_j), so plain `+=`
//     suffices — NO atomics needed.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 256;

inline int64_t n_blocks(int64_t n) { return (n + THREADS - 1) / THREADS; }

// ---- forward --------------------------------------------------------------
// One thread per (b,h,i) query row. Decode b,h,i from the flat index.
//   s_ij = scale * Sum_d Q[i,d]*K[j,d]   for j = 0..i
//   m_i  = max_{j<=i} s_ij ;  l_i = Sum_{j<=i} exp(s_ij - m_i)
//   O[i,d] = Sum_{j<=i} (exp(s_ij - m_i)/l_i) * V[j,d]
//   LSE[i] = m_i + log(l_i)
__global__ void flash_forward_kernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     float* __restrict__ O,
                                     float* __restrict__ LSE,
                                     float scale,
                                     int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N) return;
    int64_t i  = idx % N;
    int64_t bh = idx / N;             // flat (b*H + h)

    const float* Qrow = Q + (bh * N + i) * D;
    const float* Kbase = K + bh * N * D;
    const float* Vbase = V + bh * N * D;
    float* Orow = O + (bh * N + i) * D;

    // pass 1: running max of scores over j <= i
    float m = -INFINITY;
    for (int64_t j = 0; j <= i; ++j) {
        const float* Krow = Kbase + j * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += Qrow[d] * Krow[d];
        s *= scale;
        m = fmaxf(m, s);
    }

    // pass 2: l_i and accumulate O[i,:] = Sum_j exp(s_ij - m) * V[j,:]
    for (int64_t d = 0; d < D; ++d) Orow[d] = 0.0f;
    float l = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        const float* Krow = Kbase + j * D;
        const float* Vrow = Vbase + j * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += Qrow[d] * Krow[d];
        s *= scale;
        float p = expf(s - m);
        l += p;
        for (int64_t d = 0; d < D; ++d) Orow[d] += p * Vrow[d];
    }

    float inv_l = 1.0f / l;
    for (int64_t d = 0; d < D; ++d) Orow[d] *= inv_l;

    LSE[bh * N + i] = m + logf(l);
}

// ---- backward: Delta ------------------------------------------------------
// Delta[b,h,i] = Sum_d O[i,d] * dO[i,d]   (one thread per (b,h,i))
__global__ void flash_delta_kernel(const float* __restrict__ O,
                                   const float* __restrict__ dO,
                                   float* __restrict__ Delta,
                                   int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N) return;
    int64_t i  = idx % N;
    int64_t bh = idx / N;

    const float* Orow = O + (bh * N + i) * D;
    const float* dOrow = dO + (bh * N + i) * D;
    float acc = 0.0f;
    for (int64_t d = 0; d < D; ++d) acc += Orow[d] * dOrow[d];
    Delta[bh * N + i] = acc;
}

// ---- backward: dV ---------------------------------------------------------
// dV[j,d] += Sum_{i >= j} P_ij * dO[i,d],  P_ij = exp(scale*Q_i.K_j - LSE[i])
// One thread per (b,h,j); owns dV row j, so plain `+=` (no atomics).
__global__ void flash_dV_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ dO,
                                float* __restrict__ dV,
                                const float* __restrict__ LSE,
                                float scale,
                                int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N) return;
    int64_t j  = idx % N;
    int64_t bh = idx / N;

    const float* Qbase = Q + bh * N * D;
    const float* Krow = K + (bh * N + j) * D;
    const float* dObase = dO + bh * N * D;
    const float* LSErow = LSE + bh * N;
    float* dVrow = dV + (bh * N + j) * D;

    for (int64_t i = j; i < N; ++i) {       // causal: only i >= j contribute
        const float* Qrow = Qbase + i * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += Qrow[d] * Krow[d];
        float p = expf(scale * s - LSErow[i]);
        const float* dOrow = dObase + i * D;
        for (int64_t d = 0; d < D; ++d) dVrow[d] += p * dOrow[d];
    }
}

// ---- backward: dQ ---------------------------------------------------------
// dp_ij = Sum_d dO[i,d]*V[j,d];  ds_ij = P_ij * (dp_ij - Delta[i])
// dQ[i,d] += scale * Sum_{j <= i} ds_ij * K[j,d]
// One thread per (b,h,i); owns dQ row i, so plain `+=`.
__global__ void flash_dQ_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                const float* __restrict__ dO,
                                float* __restrict__ dQ,
                                const float* __restrict__ LSE,
                                const float* __restrict__ Delta,
                                float scale,
                                int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N) return;
    int64_t i  = idx % N;
    int64_t bh = idx / N;

    const float* Qrow = Q + (bh * N + i) * D;
    const float* Kbase = K + bh * N * D;
    const float* Vbase = V + bh * N * D;
    const float* dOrow = dO + (bh * N + i) * D;
    float* dQrow = dQ + (bh * N + i) * D;
    float lse_i = LSE[bh * N + i];
    float delta_i = Delta[bh * N + i];

    for (int64_t j = 0; j <= i; ++j) {       // causal: only j <= i
        const float* Krow = Kbase + j * D;
        const float* Vrow = Vbase + j * D;
        float s = 0.0f, dp = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            s += Qrow[d] * Krow[d];
            dp += dOrow[d] * Vrow[d];
        }
        float p = expf(scale * s - lse_i);
        float ds = p * (dp - delta_i);
        float coef = scale * ds;
        for (int64_t d = 0; d < D; ++d) dQrow[d] += coef * Krow[d];
    }
}

// ---- backward: dK ---------------------------------------------------------
// dK[j,d] += scale * Sum_{i >= j} ds_ij * Q[i,d]
// One thread per (b,h,j); owns dK row j, so plain `+=`.
__global__ void flash_dK_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                const float* __restrict__ dO,
                                float* __restrict__ dK,
                                const float* __restrict__ LSE,
                                const float* __restrict__ Delta,
                                float scale,
                                int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N) return;
    int64_t j  = idx % N;
    int64_t bh = idx / N;

    const float* Qbase = Q + bh * N * D;
    const float* Krow = K + (bh * N + j) * D;
    const float* Vrow = V + (bh * N + j) * D;
    const float* dObase = dO + bh * N * D;
    const float* LSErow = LSE + bh * N;
    const float* Deltarow = Delta + bh * N;
    float* dKrow = dK + (bh * N + j) * D;

    for (int64_t i = j; i < N; ++i) {        // causal: only i >= j
        const float* Qrow = Qbase + i * D;
        const float* dOrow = dObase + i * D;
        float s = 0.0f, dp = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            s += Qrow[d] * Krow[d];
            dp += dOrow[d] * Vrow[d];
        }
        float p = expf(scale * s - LSErow[i]);
        float ds = p * (dp - Deltarow[i]);
        float coef = scale * ds;
        for (int64_t d = 0; d < D; ++d) dKrow[d] += coef * Qrow[d];
    }
}

} // namespace

void flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                             torch::Tensor O, torch::Tensor LSE, double scale,
                             int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t total = B * H * N;
    flash_forward_kernel<<<n_blocks(total), THREADS>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), LSE.data_ptr<float>(), (float)scale, B, H, N, D);
}

void flash_attention_backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              torch::Tensor O, torch::Tensor dO,
                              torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV,
                              torch::Tensor LSE, double scale,
                              int64_t B, int64_t H, int64_t N, int64_t D) {
    int64_t total = B * H * N;
    auto Delta = torch::empty({B, H, N}, LSE.options());

    flash_delta_kernel<<<n_blocks(total), THREADS>>>(
        O.data_ptr<float>(), dO.data_ptr<float>(), Delta.data_ptr<float>(),
        B, H, N, D);

    flash_dV_kernel<<<n_blocks(total), THREADS>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), dO.data_ptr<float>(),
        dV.data_ptr<float>(), LSE.data_ptr<float>(), (float)scale, B, H, N, D);

    flash_dQ_kernel<<<n_blocks(total), THREADS>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), dQ.data_ptr<float>(), LSE.data_ptr<float>(),
        Delta.data_ptr<float>(), (float)scale, B, H, N, D);

    flash_dK_kernel<<<n_blocks(total), THREADS>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), dK.data_ptr<float>(), LSE.data_ptr<float>(),
        Delta.data_ptr<float>(), (float)scale, B, H, N, D);
}
