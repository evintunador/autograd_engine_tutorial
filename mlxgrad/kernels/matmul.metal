// Matmul Metal kernels for mlxgrad: forward + backward (dA, dB).
//
// Round-2 GEMM built on Metal's simdgroup_matrix MMA intrinsics (the 8x8
// matrix-multiply-accumulate ops MLX uses internally; Apple-silicon's tensor
// cores). Each threadgroup computes a BM x BN output tile for one batch using a
// 2-D grid of simdgroups (SGM x SGN). Every simdgroup owns a register block of
// WM x WN simdgroup_float8x8 accumulator fragments, so it produces a
// (WM*8) x (WN*8) sub-tile. With BM=BN=64, BK=8, SGM=2, SGN=4, WM=4, WN=2 the
// threadgroup is 8 simdgroups = 256 threads.
//
// Per K-step we stage BM x BK of the left operand and BK x BN of the right
// operand into threadgroup memory (cooperatively, all threads), barrier, then
// each simdgroup simdgroup_load's its fragments out of the staged tiles and
// accumulates with simdgroup_multiply_accumulate. At the end each simdgroup
// simdgroup_store's its fragments to a shared Cs tile which is written out (with
// edge guards) to global memory. Accumulating in fragment registers keeps the
// staged tiles small and the FLOP:byte ratio high.
//
// EDGE HANDLING: the grid is rounded UP so every threadgroup is full (MLX uses
// non-uniform dispatch). Cooperative loads zero-fill out-of-range elements and
// the final store guards output coords, so the kernels are correct for
// ARBITRARY M/K/N (incl. dims smaller than 8, e.g. the (2,8,16)@(2,16,8) tests).
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
// The tile constants below MUST match _BM/_BN/_BK and the launch math in
// mlx_kernels.py (the wrappers size the grid/threadgroup from them).

// @kernel matmul_forward
// C[b,m,n] = Σ_k A[b,m,k]·B[(b),k,n]
//   threadgroup = 256 threads (8 simdgroups); grid sized by the wrapper.  WRITES out
#define BM 64u
#define BN 64u
#define BK 8u
#define SGN 4u
#define WM 4u
#define WN 2u
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint SH = Sh[0];
uint b    = threadgroup_position_in_grid.z;
uint tg_m = threadgroup_position_in_grid.y * BM;   // base output row (m)
uint tg_n = threadgroup_position_in_grid.x * BN;   // base output col (n)
uint sg   = simdgroup_index_in_threadgroup;
uint sg_m = sg / SGN;                               // simdgroup row index
uint sg_n = sg % SGN;                               // simdgroup col index
uint tid  = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
uint Abase = b * M_ * K_;
uint Bbase = (SH != 0u) ? 0u : b * K_ * N_;
threadgroup float As[BM][BK];   // A sub-tile  (rows m, cols k)
threadgroup float Bs[BK][BN];   // B sub-tile  (rows k, cols n)
simdgroup_float8x8 Cfrag[WM][WN];
for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
uint nK = (K_ + BK - 1) / BK;
for (uint kt = 0; kt < nK; ++kt) {
    uint k0 = kt * BK;
    for (uint idx = tid; idx < BM * BK; idx += nthreads) {
        uint r = idx / BK, c = idx % BK;            // r in [0,BM) m, c in [0,BK) k
        uint gm = tg_m + r, gk = k0 + c;
        As[r][c] = (gm < M_ && gk < K_) ? A[Abase + gm * K_ + gk] : 0.0f;
    }
    for (uint idx = tid; idx < BK * BN; idx += nthreads) {
        uint r = idx / BN, c = idx % BN;            // r in [0,BK) k, c in [0,BN) n
        uint gk = k0 + r, gn = tg_n + c;
        Bs[r][c] = (gk < K_ && gn < N_) ? B[Bbase + gk * N_ + gn] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Af[WM], Bf[WN];
    for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[(sg_m * WM + i) * 8u][0],            BK);
    for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[0][(sg_n * WN + j) * 8u],            BN);
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup float Cs[BM][BN];
for (uint i = 0; i < WM; ++i)
    for (uint j = 0; j < WN; ++j)
        simdgroup_store(Cfrag[i][j], &Cs[(sg_m * WM + i) * 8u][(sg_n * WN + j) * 8u], BN);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint idx = tid; idx < BM * BN; idx += nthreads) {
    uint r = idx / BN, c = idx % BN;
    uint gm = tg_m + r, gn = tg_n + c;
    if (gm < M_ && gn < N_) out[(b * M_ + gm) * N_ + gn] = Cs[r][c];
}
#undef BM
#undef BN
#undef BK
#undef SGN
#undef WM
#undef WN

// @kernel matmul_backward_dA
// dA[b,m,k] = grad_in[...] + Σ_n dC[b,m,n]·B[(b),k,n]    (= dC @ B^T, contract n)
//   Output is (M x K) per batch.  i=m (BM rows), j=k (BN cols), contract p=n.
//   Left  operand L[m,n] = dC[m,n]                 -> staged into As[m][n]
//   Right operand R[n,k] = B[k,n]  (B transposed)  -> staged into Bs[n][k]
//   FUNCTIONAL accumulate: out = grad_in + contribution.
#define BM 64u
#define BN 64u
#define BK 8u
#define SGN 4u
#define WM 4u
#define WN 2u
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint SH = Sh[0];
uint b    = threadgroup_position_in_grid.z;
uint tg_m = threadgroup_position_in_grid.y * BM;   // base output row (m)
uint tg_k = threadgroup_position_in_grid.x * BN;   // base output col (k)
uint sg   = simdgroup_index_in_threadgroup;
uint sg_m = sg / SGN;
uint sg_n = sg % SGN;
uint tid  = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
uint dCbase = b * M_ * N_;
uint Bbase  = (SH != 0u) ? 0u : b * K_ * N_;
threadgroup float As[BM][BK];   // dC sub-tile : rows m, cols n  (contract = n)
threadgroup float Bs[BK][BN];   // B^T sub-tile: rows n, cols k
simdgroup_float8x8 Cfrag[WM][WN];
for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
uint nP = (N_ + BK - 1) / BK;                       // contract over n in steps of BK
for (uint pt = 0; pt < nP; ++pt) {
    uint p0 = pt * BK;
    for (uint idx = tid; idx < BM * BK; idx += nthreads) {
        uint r = idx / BK, c = idx % BK;            // r m, c n-within-step
        uint gm = tg_m + r, gn = p0 + c;
        As[r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
    }
    for (uint idx = tid; idx < BK * BN; idx += nthreads) {
        uint r = idx / BN, c = idx % BN;            // r n-within-step, c k
        uint gn = p0 + r, gk = tg_k + c;
        Bs[r][c] = (gn < N_ && gk < K_) ? B[Bbase + gk * N_ + gn] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Af[WM], Bf[WN];
    for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[(sg_m * WM + i) * 8u][0], BK);
    for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[0][(sg_n * WN + j) * 8u], BN);
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup float Cs[BM][BN];
for (uint i = 0; i < WM; ++i)
    for (uint j = 0; j < WN; ++j)
        simdgroup_store(Cfrag[i][j], &Cs[(sg_m * WM + i) * 8u][(sg_n * WN + j) * 8u], BN);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint idx = tid; idx < BM * BN; idx += nthreads) {
    uint r = idx / BN, c = idx % BN;
    uint gm = tg_m + r, gk = tg_k + c;
    if (gm < M_ && gk < K_) {
        uint o = (b * M_ + gm) * K_ + gk;
        out[o] = grad_in[o] + Cs[r][c];
    }
}
#undef BM
#undef BN
#undef BK
#undef SGN
#undef WM
#undef WN

// @kernel matmul_backward_dB_batched
// dB[b,k,n] = grad_in[...] + Σ_m A[b,m,k]·dC[b,m,n]    (= A^T @ dC, contract m)
//   Output is (K x N) per batch.  i=k (BM rows), j=n (BN cols), contract p=m.
//   Left  operand L[k,m] = A[m,k]  (A transposed)  -> staged into As[k][m]
//   Right operand R[m,n] = dC[m,n]                 -> staged into Bs[m][n]
//   FUNCTIONAL accumulate: out = grad_in + contribution.
#define BM 64u
#define BN 64u
#define BK 8u
#define SGN 4u
#define WM 4u
#define WN 2u
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0];
uint b    = threadgroup_position_in_grid.z;
uint tg_k = threadgroup_position_in_grid.y * BM;   // base output row (k)
uint tg_n = threadgroup_position_in_grid.x * BN;   // base output col (n)
uint sg   = simdgroup_index_in_threadgroup;
uint sg_m = sg / SGN;
uint sg_n = sg % SGN;
uint tid  = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
uint Abase  = b * M_ * K_;
uint dCbase = b * M_ * N_;
threadgroup float As[BM][BK];   // A^T sub-tile: rows k, cols m  (contract = m)
threadgroup float Bs[BK][BN];   // dC  sub-tile: rows m, cols n
simdgroup_float8x8 Cfrag[WM][WN];
for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
uint nP = (M_ + BK - 1) / BK;                       // contract over m in steps of BK
for (uint pt = 0; pt < nP; ++pt) {
    uint p0 = pt * BK;
    for (uint idx = tid; idx < BM * BK; idx += nthreads) {
        uint r = idx / BK, c = idx % BK;            // r k, c m-within-step
        uint gk = tg_k + r, gm = p0 + c;
        As[r][c] = (gk < K_ && gm < M_) ? A[Abase + gm * K_ + gk] : 0.0f;
    }
    for (uint idx = tid; idx < BK * BN; idx += nthreads) {
        uint r = idx / BN, c = idx % BN;            // r m-within-step, c n
        uint gm = p0 + r, gn = tg_n + c;
        Bs[r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Af[WM], Bf[WN];
    for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[(sg_m * WM + i) * 8u][0], BK);
    for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[0][(sg_n * WN + j) * 8u], BN);
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup float Cs[BM][BN];
for (uint i = 0; i < WM; ++i)
    for (uint j = 0; j < WN; ++j)
        simdgroup_store(Cfrag[i][j], &Cs[(sg_m * WM + i) * 8u][(sg_n * WN + j) * 8u], BN);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint idx = tid; idx < BM * BN; idx += nthreads) {
    uint r = idx / BN, c = idx % BN;
    uint gk = tg_k + r, gn = tg_n + c;
    if (gk < K_ && gn < N_) {
        uint o = (b * K_ + gk) * N_ + gn;
        out[o] = grad_in[o] + Cs[r][c];
    }
}
#undef BM
#undef BN
#undef BK
#undef SGN
#undef WM
#undef WN

// @kernel matmul_backward_dB_shared
// dB[k,n] = grad_in[...] + Σ_b Σ_m A[b,m,k]·dC[b,m,n]   (sums over batch AND m)
//   A^T @ dC contracted over m, then summed across the batch. Output (K x N).
//   i=k (BM rows), j=n (BN cols), contract p=m, outer loop over batch.
//   FUNCTIONAL accumulate: out = grad_in + contribution.
#define BM 64u
#define BN 64u
#define BK 8u
#define SGN 4u
#define WM 4u
#define WN 2u
uint M_ = Mb[0]; uint K_ = Kb[0]; uint N_ = Nb[0]; uint BS = Bsz[0];
uint tg_k = threadgroup_position_in_grid.y * BM;   // base output row (k)
uint tg_n = threadgroup_position_in_grid.x * BN;   // base output col (n)
uint sg   = simdgroup_index_in_threadgroup;
uint sg_m = sg / SGN;
uint sg_n = sg % SGN;
uint tid  = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
threadgroup float As[BM][BK];   // A^T sub-tile: rows k, cols m
threadgroup float Bs[BK][BN];   // dC  sub-tile: rows m, cols n
simdgroup_float8x8 Cfrag[WM][WN];
for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
uint nP = (M_ + BK - 1) / BK;
for (uint b = 0; b < BS; ++b) {
    uint Abase  = b * M_ * K_;
    uint dCbase = b * M_ * N_;
    for (uint pt = 0; pt < nP; ++pt) {
        uint p0 = pt * BK;
        for (uint idx = tid; idx < BM * BK; idx += nthreads) {
            uint r = idx / BK, c = idx % BK;        // r k, c m-within-step
            uint gk = tg_k + r, gm = p0 + c;
            As[r][c] = (gk < K_ && gm < M_) ? A[Abase + gm * K_ + gk] : 0.0f;
        }
        for (uint idx = tid; idx < BK * BN; idx += nthreads) {
            uint r = idx / BN, c = idx % BN;        // r m-within-step, c n
            uint gm = p0 + r, gn = tg_n + c;
            Bs[r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 Af[WM], Bf[WN];
        for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[(sg_m * WM + i) * 8u][0], BK);
        for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[0][(sg_n * WN + j) * 8u], BN);
        for (uint i = 0; i < WM; ++i)
            for (uint j = 0; j < WN; ++j)
                simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
threadgroup float Cs[BM][BN];
for (uint i = 0; i < WM; ++i)
    for (uint j = 0; j < WN; ++j)
        simdgroup_store(Cfrag[i][j], &Cs[(sg_m * WM + i) * 8u][(sg_n * WN + j) * 8u], BN);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint idx = tid; idx < BM * BN; idx += nthreads) {
    uint r = idx / BN, c = idx % BN;
    uint gk = tg_k + r, gn = tg_n + c;
    if (gk < K_ && gn < N_) {
        uint o = gk * N_ + gn;
        out[o] = grad_in[o] + Cs[r][c];
    }
}
#undef BM
#undef BN
#undef BK
#undef SGN
#undef WM
#undef WN
