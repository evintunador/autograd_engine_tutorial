// Matmul Metal kernels for mlxgrad: forward + backward (dA, dB).
//
// The mlxgrad analog of cudagrad/kernels/matmul.cu. Naive, obviously-correct
// (tutorial; test sizes are tiny, tolerances loose): ONE THREAD PER OUTPUT
// ELEMENT, each looping over the contracted dim. No tiling / threadgroup memory.
// Each thread owns a distinct output element, so forward writes and the backward
// functional accumulation (out = grad_in + contribution) need NO atomics.
//
// Layout/contract notes (all tensors contiguous, row-major, fp32):
//   * A has shape (..., M, K); leading batch dims flatten to Bsz.
//   * Two B layouts, picked by the wrapper from ndim:
//       - BATCHED : B is (..., K, N), same leading dims as A.  C[b,m,n]=Σ_k A[b,m,k]·B[b,k,n].
//       - SHARED  : B is 2-D (K, N), broadcast across the batch. C[b,m,n]=Σ_k A[b,m,k]·B[k,n].
//     Plain 2-D@2-D is Bsz==1 of the batched case.
//   * Flat row-major offsets:
//       A[b,m,k]=A[(b*M+m)*K+k]   B_batched[b,k,n]=B[(b*K+k)*N+n]
//       B_shared[k,n]=B[k*N+n]    C[b,m,n]=C[(b*M+m)*N+n]
//   * Sh (uint, as bool): 1 -> B shared (2-D), 0 -> B batched.
//   * the grid is launched with exactly `total` threads, so idx is always in range.
//   * for the SHARED-B case the dB kernel sums over BOTH batch and M (that batch
//     sum is exactly what makes the linear-layer weight grad correct).

// @kernel matmul_forward
// C[b,m,n] = Σ_k A[b,m,k]·B[(b),k,n]   (grid = Bsz*M*N; WRITES out)
uint idx = thread_position_in_grid.x;
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint SH = Sh[0];
uint n = idx % N_;
uint m = (idx / N_) % M_;
uint b = idx / (M_ * N_);
uint Arow = (b * M_ + m) * K_;
uint Bbase = (SH != 0u) ? 0u : b * K_ * N_;
float acc = 0.0f;
for (uint k = 0; k < K_; ++k) acc += A[Arow + k] * B[Bbase + k * N_ + n];
out[(b * M_ + m) * N_ + n] = acc;

// @kernel matmul_backward_dA
// dA[b,m,k] = grad_in[...] + Σ_n dC[b,m,n]·B[(b),k,n]   (grid = Bsz*M*K)
uint idx = thread_position_in_grid.x;
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint SH = Sh[0];
uint k = idx % K_;
uint m = (idx / K_) % M_;
uint b = idx / (M_ * K_);
uint dCrow = (b * M_ + m) * N_;
uint Bbase = (SH != 0u) ? 0u : b * K_ * N_;
float acc = 0.0f;
for (uint n = 0; n < N_; ++n) acc += dC[dCrow + n] * B[Bbase + k * N_ + n];
uint o = (b * M_ + m) * K_ + k;
out[o] = grad_in[o] + acc;

// @kernel matmul_backward_dB_batched
// dB[b,k,n] = grad_in[...] + Σ_m A[b,m,k]·dC[b,m,n]   (grid = Bsz*K*N)
uint idx = thread_position_in_grid.x;
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0];
uint n = idx % N_;
uint k = (idx / N_) % K_;
uint b = idx / (K_ * N_);
uint Abase = b * M_ * K_;
uint dCbase = b * M_ * N_;
float acc = 0.0f;
for (uint m = 0; m < M_; ++m) acc += A[Abase + m * K_ + k] * dC[dCbase + m * N_ + n];
uint o = (b * K_ + k) * N_ + n;
out[o] = grad_in[o] + acc;

// @kernel matmul_backward_dB_shared
// dB[k,n] = grad_in[...] + Σ_b Σ_m A[b,m,k]·dC[b,m,n]   (grid = K*N; sums batch)
uint idx = thread_position_in_grid.x;
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint BS = Bsz[0];
uint n = idx % N_;
uint k = idx / N_;
float acc = 0.0f;
for (uint b = 0; b < BS; ++b) {
    uint Abase = b * M_ * K_;
    uint dCbase = b * M_ * N_;
    for (uint m = 0; m < M_; ++m) acc += A[Abase + m * K_ + k] * dC[dCbase + m * N_ + n];
}
uint o = k * N_ + n;
out[o] = grad_in[o] + acc;
