// Matmul Metal kernels for mlxgrad: forward + backward (dA, dB).
//
// The mlxgrad analog of cudagrad/kernels/matmul.cu. TILED shared-memory GEMM:
// each threadgroup is TILE x TILE threads computing one TILE x TILE output tile
// for one batch (grid.z = Bsz). We stage TILE x TILE sub-tiles of the two
// operands into threadgroup memory, barrier, accumulate over the staged tile,
// and march along the contracted dim in steps of TILE. Each thread owns a
// distinct output element, so forward writes and the backward functional
// accumulation (out = grad_in + contribution) need NO atomics.
//
// EDGE HANDLING: the grid is rounded UP to a multiple of TILE in each output
// dim (see the Python wrappers), so threadgroups are always FULL (MLX uses
// non-uniform dispatch, which would otherwise hand boundary threadgroups a
// partial thread block and break the cooperative loads). Threads whose output
// coords fall outside [M)/[N)/[K) load 0 into the staging tiles and skip the
// final store. This makes the kernels correct for ARBITRARY M/K/N, not just
// multiples of TILE.
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
//   * for the SHARED-B case the dB kernel sums over BOTH batch and M (that batch
//     sum is exactly what makes the linear-layer weight grad correct).
//
// TILE must match TILE in mlx_kernels.py (the wrappers size the grid/threadgroup).

// @kernel matmul_forward
// C[b,m,n] = Σ_k A[b,m,k]·B[(b),k,n]
//   grid=(ceilN*TILE, ceilM*TILE, Bsz)  threadgroup=(TILE,TILE,1)  WRITES out
#define TILE 16
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint SH = Sh[0];
uint b   = thread_position_in_grid.z;
uint lr  = thread_position_in_threadgroup.y;   // local row within tile
uint lc  = thread_position_in_threadgroup.x;   // local col within tile
uint m   = threadgroup_position_in_grid.y * TILE + lr;   // output row in [M)
uint n   = threadgroup_position_in_grid.x * TILE + lc;   // output col in [N)
uint Abase = b * M_ * K_;
uint Bbase = (SH != 0u) ? 0u : b * K_ * N_;
threadgroup float As[TILE][TILE];
threadgroup float Bs[TILE][TILE];
float acc = 0.0f;
uint nTiles = (K_ + TILE - 1) / TILE;
for (uint t = 0; t < nTiles; ++t) {
    uint kA = t * TILE + lc;     // column of A this thread stages
    uint kB = t * TILE + lr;     // row of B this thread stages
    As[lr][lc] = (m < M_ && kA < K_) ? A[Abase + m * K_ + kA] : 0.0f;
    Bs[lr][lc] = (kB < K_ && n < N_) ? B[Bbase + kB * N_ + n] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) acc += As[lr][p] * Bs[p][lc];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (m < M_ && n < N_) out[(b * M_ + m) * N_ + n] = acc;
#undef TILE

// @kernel matmul_backward_dA
// dA[b,m,k] = grad_in[...] + Σ_n dC[b,m,n]·B[(b),k,n]
//   This is dC @ B^T : contract over n. Output is (M x K) per batch.
//   grid=(ceilK*TILE, ceilM*TILE, Bsz)  threadgroup=(TILE,TILE,1)
#define TILE 16
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint SH = Sh[0];
uint b   = thread_position_in_grid.z;
uint lr  = thread_position_in_threadgroup.y;
uint lc  = thread_position_in_threadgroup.x;
uint m   = threadgroup_position_in_grid.y * TILE + lr;   // output row in [M)
uint k   = threadgroup_position_in_grid.x * TILE + lc;   // output col in [K)
uint dCbase = b * M_ * N_;
uint Bbase  = (SH != 0u) ? 0u : b * K_ * N_;
threadgroup float dCs[TILE][TILE];   // dC tile: rows = m, cols = n
threadgroup float Bs[TILE][TILE];    // B^T tile: rows = n, cols = k  (B[k,n])
float acc = 0.0f;
uint nTiles = (N_ + TILE - 1) / TILE;
for (uint t = 0; t < nTiles; ++t) {
    uint nC = t * TILE + lc;     // dC column this thread stages
    uint nB = t * TILE + lr;     // B column (the contracted n) this thread stages
    dCs[lr][lc] = (m < M_ && nC < N_) ? dC[dCbase + m * N_ + nC] : 0.0f;
    // Bs[lr][lc] holds B[k=col, n=row] so the inner product contracts over n.
    Bs[lr][lc] = (k < K_ && nB < N_) ? B[Bbase + k * N_ + nB] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) acc += dCs[lr][p] * Bs[p][lc];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (m < M_ && k < K_) {
    uint o = (b * M_ + m) * K_ + k;
    out[o] = grad_in[o] + acc;
}
#undef TILE

// @kernel matmul_backward_dB_batched
// dB[b,k,n] = grad_in[...] + Σ_m A[b,m,k]·dC[b,m,n]
//   This is A^T @ dC : contract over m. Output is (K x N) per batch.
//   grid=(ceilN*TILE, ceilK*TILE, Bsz)  threadgroup=(TILE,TILE,1)
#define TILE 16
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0];
uint b   = thread_position_in_grid.z;
uint lr  = thread_position_in_threadgroup.y;
uint lc  = thread_position_in_threadgroup.x;
uint k   = threadgroup_position_in_grid.y * TILE + lr;   // output row in [K)
uint n   = threadgroup_position_in_grid.x * TILE + lc;   // output col in [N)
uint Abase  = b * M_ * K_;
uint dCbase = b * M_ * N_;
threadgroup float As[TILE][TILE];    // A^T tile: rows = k, cols = m  (A[m,k])
threadgroup float dCs[TILE][TILE];   // dC tile: rows = m, cols = n
float acc = 0.0f;
uint nTiles = (M_ + TILE - 1) / TILE;
for (uint t = 0; t < nTiles; ++t) {
    uint mA = t * TILE + lc;     // A row (contracted m) this thread stages
    uint mC = t * TILE + lr;     // dC row (contracted m) this thread stages
    // As[lr][lc] holds A[m=col, k=row] so the inner product contracts over m.
    As[lr][lc]  = (k < K_ && mA < M_) ? A[Abase + mA * K_ + k] : 0.0f;
    dCs[lr][lc] = (mC < M_ && n < N_) ? dC[dCbase + mC * N_ + n] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) acc += As[lr][p] * dCs[p][lc];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (k < K_ && n < N_) {
    uint o = (b * K_ + k) * N_ + n;
    out[o] = grad_in[o] + acc;
}
#undef TILE

// @kernel matmul_backward_dB_shared
// dB[k,n] = grad_in[...] + Σ_b Σ_m A[b,m,k]·dC[b,m,n]   (sums over batch AND m)
//   A^T @ dC contracted over m, then summed across the batch. Output (K x N).
//   grid=(ceilN*TILE, ceilK*TILE, 1)  threadgroup=(TILE,TILE,1)
#define TILE 16
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint BS = Bsz[0];
uint lr  = thread_position_in_threadgroup.y;
uint lc  = thread_position_in_threadgroup.x;
uint k   = threadgroup_position_in_grid.y * TILE + lr;   // output row in [K)
uint n   = threadgroup_position_in_grid.x * TILE + lc;   // output col in [N)
threadgroup float As[TILE][TILE];    // A^T tile: rows = k, cols = m
threadgroup float dCs[TILE][TILE];   // dC tile: rows = m, cols = n
float acc = 0.0f;
uint mTiles = (M_ + TILE - 1) / TILE;
for (uint b = 0; b < BS; ++b) {
    uint Abase  = b * M_ * K_;
    uint dCbase = b * M_ * N_;
    for (uint t = 0; t < mTiles; ++t) {
        uint mA = t * TILE + lc;
        uint mC = t * TILE + lr;
        As[lr][lc]  = (k < K_ && mA < M_) ? A[Abase + mA * K_ + k] : 0.0f;
        dCs[lr][lc] = (mC < M_ && n < N_) ? dC[dCbase + mC * N_ + n] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < TILE; ++p) acc += As[lr][p] * dCs[p][lc];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
if (k < K_ && n < N_) {
    uint o = k * N_ + n;
    out[o] = grad_in[o] + acc;
}
#undef TILE
