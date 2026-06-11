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
// operand into threadgroup memory (cooperatively, all threads); each simdgroup
// simdgroup_load's its fragments out of the staged tiles and accumulates with
// simdgroup_multiply_accumulate. Accumulating in fragment registers keeps the
// staged tiles small and the FLOP:byte ratio high.
//
// ROUND-3 OPTIMISATIONS (close the gap to raw MLX on M2):
//   1. DOUBLE-BUFFERING / software pipelining of the K-loop. Two threadgroup
//      buffers per operand (As[2], Bs[2]): while the simdgroups MMA on tile t
//      (buffer cur), all threads cooperatively prefetch tile t+1 into buffer
//      nxt. This needs only ONE barrier per K-step (down from two) and hides
//      global-load latency behind the matrix math.
//   2. DIRECT device store of the accumulator fragments. Interior ("full")
//      output tiles simdgroup_store straight to global memory; backward kernels
//      seed their fragments by simdgroup_load-ing grad_in first so the MMA adds
//      onto the incoming grad and the final store is a plain write (the
//      functional accumulate out = grad_in + contribution falls out for free).
//      This removes the old 64x64 (16 KB) Cs threadgroup roundtrip + its barrier
//      + cooperative copy. Crucially it FREES that 16 KB so the doubled As/Bs
//      buffers still fit and occupancy is preserved (M2 ~32 KB/threadgroup) —
//      without it, double-buffering REGRESSED. Edge tiles use a tiny 2 KB
//      per-simdgroup 8x8 scratch (Sc) with guarded writes instead of the big Cs.
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
threadgroup float As[2][BM][BK];   // A sub-tile  (rows m, cols k), double-buffered
threadgroup float Bs[2][BK][BN];   // B sub-tile  (rows k, cols n), double-buffered
simdgroup_float8x8 Cfrag[WM][WN];
for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
uint nK = (K_ + BK - 1) / BK;
// prologue: stage tile 0 into buffer 0
for (uint idx = tid; idx < BM * BK; idx += nthreads) {
    uint r = idx / BK, c = idx % BK;
    uint gm = tg_m + r, gk = c;
    As[0][r][c] = (gm < M_ && gk < K_) ? A[Abase + gm * K_ + gk] : 0.0f;
}
for (uint idx = tid; idx < BK * BN; idx += nthreads) {
    uint r = idx / BN, c = idx % BN;
    uint gk = r, gn = tg_n + c;
    Bs[0][r][c] = (gk < K_ && gn < N_) ? B[Bbase + gk * N_ + gn] : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// software-pipelined K-loop: prefetch tile kt+1 while MMA-ing tile kt (1 barrier/iter)
for (uint kt = 0; kt < nK; ++kt) {
    uint cur = kt & 1u, nxt = cur ^ 1u;
    if (kt + 1 < nK) {
        uint k0 = (kt + 1) * BK;
        for (uint idx = tid; idx < BM * BK; idx += nthreads) {
            uint r = idx / BK, c = idx % BK;
            uint gm = tg_m + r, gk = k0 + c;
            As[nxt][r][c] = (gm < M_ && gk < K_) ? A[Abase + gm * K_ + gk] : 0.0f;
        }
        for (uint idx = tid; idx < BK * BN; idx += nthreads) {
            uint r = idx / BN, c = idx % BN;
            uint gk = k0 + r, gn = tg_n + c;
            Bs[nxt][r][c] = (gk < K_ && gn < N_) ? B[Bbase + gk * N_ + gn] : 0.0f;
        }
    }
    simdgroup_float8x8 Af[WM], Bf[WN];
    for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[cur][(sg_m * WM + i) * 8u][0], BK);
    for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[cur][0][(sg_n * WN + j) * 8u], BN);
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Epilogue: full interior tiles store fragments straight to device memory
// (skips the 16 KB Cs roundtrip + barrier, freeing threadgroup memory for
// occupancy). Edge tiles fall back to the staged-and-guarded copy.
if (tg_m + BM <= M_ && tg_n + BN <= N_) {
    device float* C = out + (b * M_ + tg_m) * N_ + tg_n;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_store(Cfrag[i][j],
                C + ((sg_m * WM + i) * 8u) * N_ + (sg_n * WN + j) * 8u, N_);
} else {
    // Edge tile: each simdgroup stores its fragments to a private 8x8 scratch
    // (2 KB total vs a 16 KB Cs band) and guard-copies to global. Small scratch
    // keeps threadgroup-memory pressure low so double-buffering stays occupant.
    threadgroup float Sc[8][8 * 8];          // [simdgroup][8*8]
    uint lane = thread_index_in_simdgroup;   // 0..31
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j) {
            simdgroup_store(Cfrag[i][j], &Sc[sg][0], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint base_m = (sg_m * WM + i) * 8u, base_n = (sg_n * WN + j) * 8u;
            for (uint e = lane; e < 64u; e += 32u) {
                uint rr = e / 8u, cc = e % 8u;
                uint gm = tg_m + base_m + rr, gn = tg_n + base_n + cc;
                if (gm < M_ && gn < N_) out[(b * M_ + gm) * N_ + gn] = Sc[sg][rr * 8u + cc];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
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
bool full = (tg_m + BM <= M_ && tg_k + BN <= K_);   // interior output tile
threadgroup float As[2][BM][BK];   // dC sub-tile : rows m, cols n  (contract = n)
threadgroup float Bs[2][BK][BN];   // B^T sub-tile: rows n, cols k
simdgroup_float8x8 Cfrag[WM][WN];
// FUNCTIONAL accumulate: seed fragments with grad_in for full tiles so the MMA
// adds straight onto the incoming grad (then a plain device store closes it).
if (full) {
    device const float* G = grad_in + (b * M_ + tg_m) * K_ + tg_k;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_load(Cfrag[i][j],
                G + ((sg_m * WM + i) * 8u) * K_ + (sg_n * WN + j) * 8u, K_);
} else {
    for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
}
uint nP = (N_ + BK - 1) / BK;                       // contract over n in steps of BK
// prologue: stage tile 0
for (uint idx = tid; idx < BM * BK; idx += nthreads) {
    uint r = idx / BK, c = idx % BK;
    uint gm = tg_m + r, gn = c;
    As[0][r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
}
for (uint idx = tid; idx < BK * BN; idx += nthreads) {
    uint r = idx / BN, c = idx % BN;
    uint gn = r, gk = tg_k + c;
    Bs[0][r][c] = (gn < N_ && gk < K_) ? B[Bbase + gk * N_ + gn] : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint pt = 0; pt < nP; ++pt) {
    uint cur = pt & 1u, nxt = cur ^ 1u;
    if (pt + 1 < nP) {
        uint p0 = (pt + 1) * BK;
        for (uint idx = tid; idx < BM * BK; idx += nthreads) {
            uint r = idx / BK, c = idx % BK;
            uint gm = tg_m + r, gn = p0 + c;
            As[nxt][r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
        }
        for (uint idx = tid; idx < BK * BN; idx += nthreads) {
            uint r = idx / BN, c = idx % BN;
            uint gn = p0 + r, gk = tg_k + c;
            Bs[nxt][r][c] = (gn < N_ && gk < K_) ? B[Bbase + gk * N_ + gn] : 0.0f;
        }
    }
    simdgroup_float8x8 Af[WM], Bf[WN];
    for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[cur][(sg_m * WM + i) * 8u][0], BK);
    for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[cur][0][(sg_n * WN + j) * 8u], BN);
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (full) {
    device float* C = out + (b * M_ + tg_m) * K_ + tg_k;   // Cfrag already holds grad_in + contrib
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_store(Cfrag[i][j],
                C + ((sg_m * WM + i) * 8u) * K_ + (sg_n * WN + j) * 8u, K_);
} else {
    threadgroup float Sc[8][8 * 8];
    uint lane = thread_index_in_simdgroup;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j) {
            simdgroup_store(Cfrag[i][j], &Sc[sg][0], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint base_m = (sg_m * WM + i) * 8u, base_k = (sg_n * WN + j) * 8u;
            for (uint e = lane; e < 64u; e += 32u) {
                uint rr = e / 8u, cc = e % 8u;
                uint gm = tg_m + base_m + rr, gk = tg_k + base_k + cc;
                if (gm < M_ && gk < K_) {
                    uint o = (b * M_ + gm) * K_ + gk;
                    out[o] = grad_in[o] + Sc[sg][rr * 8u + cc];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
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
bool full = (tg_k + BM <= K_ && tg_n + BN <= N_);   // interior output tile
threadgroup float As[2][BM][BK];   // A^T sub-tile: rows k, cols m  (contract = m)
threadgroup float Bs[2][BK][BN];   // dC  sub-tile: rows m, cols n
simdgroup_float8x8 Cfrag[WM][WN];
if (full) {                                         // seed with grad_in (functional accumulate)
    device const float* G = grad_in + (b * K_ + tg_k) * N_ + tg_n;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_load(Cfrag[i][j],
                G + ((sg_m * WM + i) * 8u) * N_ + (sg_n * WN + j) * 8u, N_);
} else {
    for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
}
uint nP = (M_ + BK - 1) / BK;                       // contract over m in steps of BK
// prologue: stage tile 0
for (uint idx = tid; idx < BM * BK; idx += nthreads) {
    uint r = idx / BK, c = idx % BK;
    uint gk = tg_k + r, gm = c;
    As[0][r][c] = (gk < K_ && gm < M_) ? A[Abase + gm * K_ + gk] : 0.0f;
}
for (uint idx = tid; idx < BK * BN; idx += nthreads) {
    uint r = idx / BN, c = idx % BN;
    uint gm = r, gn = tg_n + c;
    Bs[0][r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint pt = 0; pt < nP; ++pt) {
    uint cur = pt & 1u, nxt = cur ^ 1u;
    if (pt + 1 < nP) {
        uint p0 = (pt + 1) * BK;
        for (uint idx = tid; idx < BM * BK; idx += nthreads) {
            uint r = idx / BK, c = idx % BK;
            uint gk = tg_k + r, gm = p0 + c;
            As[nxt][r][c] = (gk < K_ && gm < M_) ? A[Abase + gm * K_ + gk] : 0.0f;
        }
        for (uint idx = tid; idx < BK * BN; idx += nthreads) {
            uint r = idx / BN, c = idx % BN;
            uint gm = p0 + r, gn = tg_n + c;
            Bs[nxt][r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
        }
    }
    simdgroup_float8x8 Af[WM], Bf[WN];
    for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[cur][(sg_m * WM + i) * 8u][0], BK);
    for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[cur][0][(sg_n * WN + j) * 8u], BN);
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (full) {
    device float* C = out + (b * K_ + tg_k) * N_ + tg_n;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_store(Cfrag[i][j],
                C + ((sg_m * WM + i) * 8u) * N_ + (sg_n * WN + j) * 8u, N_);
} else {
    threadgroup float Sc[8][8 * 8];
    uint lane = thread_index_in_simdgroup;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j) {
            simdgroup_store(Cfrag[i][j], &Sc[sg][0], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint base_k = (sg_m * WM + i) * 8u, base_n = (sg_n * WN + j) * 8u;
            for (uint e = lane; e < 64u; e += 32u) {
                uint rr = e / 8u, cc = e % 8u;
                uint gk = tg_k + base_k + rr, gn = tg_n + base_n + cc;
                if (gk < K_ && gn < N_) {
                    uint o = (b * K_ + gk) * N_ + gn;
                    out[o] = grad_in[o] + Sc[sg][rr * 8u + cc];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
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
bool full = (tg_k + BM <= K_ && tg_n + BN <= N_);   // interior output tile
threadgroup float As[2][BM][BK];   // A^T sub-tile: rows k, cols m
threadgroup float Bs[2][BK][BN];   // dC  sub-tile: rows m, cols n
simdgroup_float8x8 Cfrag[WM][WN];
if (full) {                                         // seed with grad_in (functional accumulate)
    device const float* G = grad_in + tg_k * N_ + tg_n;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_load(Cfrag[i][j],
                G + ((sg_m * WM + i) * 8u) * N_ + (sg_n * WN + j) * 8u, N_);
} else {
    for (uint i = 0; i < WM; ++i) for (uint j = 0; j < WN; ++j) Cfrag[i][j] = simdgroup_float8x8(0.0f);
}
uint nP = (M_ + BK - 1) / BK;
// Flatten (batch, pt) into one pipelined contraction stream so each prefetch
// (incl. the batch rollover) overlaps the current MMA.
uint nTot = BS * nP;
{   // prologue: stage first (b=0, pt=0)
    for (uint idx = tid; idx < BM * BK; idx += nthreads) {
        uint r = idx / BK, c = idx % BK;
        uint gk = tg_k + r, gm = c;
        As[0][r][c] = (gk < K_ && gm < M_) ? A[gm * K_ + gk] : 0.0f;
    }
    for (uint idx = tid; idx < BK * BN; idx += nthreads) {
        uint r = idx / BN, c = idx % BN;
        uint gm = r, gn = tg_n + c;
        Bs[0][r][c] = (gm < M_ && gn < N_) ? dC[gm * N_ + gn] : 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint t = 0; t < nTot; ++t) {
    uint cur = t & 1u, nxt = cur ^ 1u;
    if (t + 1 < nTot) {
        uint b1 = (t + 1) / nP, p1 = (t + 1) % nP;
        uint Abase = b1 * M_ * K_, dCbase = b1 * M_ * N_, p0 = p1 * BK;
        for (uint idx = tid; idx < BM * BK; idx += nthreads) {
            uint r = idx / BK, c = idx % BK;
            uint gk = tg_k + r, gm = p0 + c;
            As[nxt][r][c] = (gk < K_ && gm < M_) ? A[Abase + gm * K_ + gk] : 0.0f;
        }
        for (uint idx = tid; idx < BK * BN; idx += nthreads) {
            uint r = idx / BN, c = idx % BN;
            uint gm = p0 + r, gn = tg_n + c;
            Bs[nxt][r][c] = (gm < M_ && gn < N_) ? dC[dCbase + gm * N_ + gn] : 0.0f;
        }
    }
    simdgroup_float8x8 Af[WM], Bf[WN];
    for (uint i = 0; i < WM; ++i) simdgroup_load(Af[i], &As[cur][(sg_m * WM + i) * 8u][0], BK);
    for (uint j = 0; j < WN; ++j) simdgroup_load(Bf[j], &Bs[cur][0][(sg_n * WN + j) * 8u], BN);
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_multiply_accumulate(Cfrag[i][j], Af[i], Bf[j], Cfrag[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (full) {
    device float* C = out + tg_k * N_ + tg_n;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j)
            simdgroup_store(Cfrag[i][j],
                C + ((sg_m * WM + i) * 8u) * N_ + (sg_n * WN + j) * 8u, N_);
} else {
    threadgroup float Sc[8][8 * 8];
    uint lane = thread_index_in_simdgroup;
    for (uint i = 0; i < WM; ++i)
        for (uint j = 0; j < WN; ++j) {
            simdgroup_store(Cfrag[i][j], &Sc[sg][0], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint base_k = (sg_m * WM + i) * 8u, base_n = (sg_n * WN + j) * 8u;
            for (uint e = lane; e < 64u; e += 32u) {
                uint rr = e / 8u, cc = e % 8u;
                uint gk = tg_k + base_k + rr, gn = tg_n + base_n + cc;
                if (gk < K_ && gn < N_) {
                    uint o = gk * N_ + gn;
                    out[o] = grad_in[o] + Sc[sg][rr * 8u + cc];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}
#undef BM
#undef BN
#undef BK
#undef SGN
#undef WM
#undef WN
