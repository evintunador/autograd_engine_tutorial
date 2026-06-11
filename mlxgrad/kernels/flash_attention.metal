// Flash-attention Metal kernels for mlxgrad: CAUSAL attention fwd + bwd.
//
// The mlxgrad analog of cudagrad/kernels/flash_attention.cu. TILED / MMA design
// (round-2 optimization). Each SIMD GROUP (32 lanes) cooperatively processes an
// 8x8 BLOCK of rows using Metal's simdgroup_matrix MMA units. Q/K/V/dO blocks are
// staged into threadgroup memory; the score tile S = Q.K^T, the output tile
// O += P.V, and the gradient tiles are all formed by simdgroup_multiply_accumulate
// on 8x8 fp32 fragments. The per-row online-softmax reductions (running max/denom)
// run on lanes 0..7 over the materialized 8x8 score tile in threadgroup memory.
//
// This replaces round-1's serial one-simd_sum-per-key reduction chain (which only
// exposed 32 lanes of parallelism and a long latency-bound dependency loop) with
// dense 8x8 matrix tiles, exactly the structure the M-series GPU MMA units want.
//
// D=32 in the suite splits into NCH = D/8 = 4 fragments of 8 channels each. The
// implementation assumes D is a multiple of 8 (true for the suite, D=32); the QK^T
// and P.V matmuls accumulate over those NCH fragments. N need not be a multiple of
// 8: out-of-range rows are zero-padded on load and masked/guarded on store.
//
// Launch convention (see mlx_kernels._flash_mma_launch): grid = (32, B*H*nblk, 1),
// threadgroup = (32,1,1), so threadgroup_position_in_grid.y selects the (bh, block)
// pair and thread_index_in_simdgroup is the lane 0..31. `nblk = ceil(N/8)` for both
// query-blocked and key-blocked kernels.
//
// Layout/contract notes (match cudagrad precisely):
//   * Q/K/V/O/dO/dQ/dK/dV are (B,H,N,D) contiguous fp32; LSE/Delta are (B,H,N).
//     Q[b,h,i,d] = Q[((b*H+h)*N + i)*D + d];  LSE[b,h,i] = LSE[(b*H+h)*N + i].
//     bh = b*H+h; the row block base for a given bh is (bh*N)*D.
//   * CAUSAL: query i attends only to keys j <= i.
//   * score s_ij = scale*(Q_i.K_j); `scale` is the MULTIPLIER passed in
//     (= sqrt(D) in the suite), used verbatim (NOT 1/sqrt(D)).
//   * forward stores LSE[i] = m_i + log(l_i); P_ij = exp(scale*Q_i.K_j - LSE[i]).
//   * backward kernels accumulate functionally into dQ_in/dK_in/dV_in.
//
// The MMA intrinsics need `#include <metal_simdgroup_matrix>` which the wrapper
// passes via the metal_kernel `header` argument.

// @kernel flash_forward
// O[i,:], LSE[i] for an 8-query-row block. One simdgroup per (bh, query-block).
// Online (single-pass) stable softmax over 8-key blocks, MMA for Q.K^T and P.V.
uint sg = thread_index_in_simdgroup;          // lane 0..31
uint blk = threadgroup_position_in_grid.y;    // bh*nqb + qb
uint N_ = N[0];
uint D_ = D[0];
float scl = scale[0];
uint NCH = D_ / 8u;                           // channel fragments (4 for D=32)
uint nqb = (N_ + 7u) / 8u;
uint bh = blk / nqb;
uint qb = blk % nqb;
uint qi0 = qb * 8u;
uint base = (bh * N_) * D_;

threadgroup float Qtg[8 * 32];
threadgroup float Ktg[8 * 32];
threadgroup float Vtg[8 * 32];
threadgroup float Stg[8 * 8];
threadgroup float Ptg[8 * 8];
threadgroup float Otg[8 * 32];
threadgroup float Mtg[8];
threadgroup float Ltg[8];
threadgroup float Ctg[8];   // per-row rescale factor for the current key block

for (uint t = sg; t < 8u * D_; t += 32u) {
    uint r = t / D_, c = t % D_;
    uint gi = qi0 + r;
    Qtg[t] = (gi < N_) ? Q[base + gi * D_ + c] : 0.0f;
    Otg[t] = 0.0f;
}
if (sg < 8u) { Mtg[sg] = -INFINITY; Ltg[sg] = 0.0f; }
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 Qf[4];
for (uint c = 0; c < NCH; ++c) simdgroup_load(Qf[c], Qtg + c * 8u, D_);

uint qimax = qi0 + 7u;
uint klast = (qimax < N_ - 1u) ? qimax : (N_ - 1u);   // causal: keys up to qimax
uint nkb = klast / 8u + 1u;
for (uint kb = 0; kb < nkb; ++kb) {
    uint kj0 = kb * 8u;
    for (uint t = sg; t < 8u * D_; t += 32u) {
        uint r = t / D_, c = t % D_;
        uint gj = kj0 + r;
        Ktg[t] = (gj < N_) ? K[base + gj * D_ + c] : 0.0f;
        Vtg[t] = (gj < N_) ? V[base + gj * D_ + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S(8q x 8k) = Q(8 x D) . K^T(D x 8) = sum_c Qf[c] . Kf[c]^T
    simdgroup_float8x8 Sf = simdgroup_float8x8(0);
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 KTf;
        simdgroup_load(KTf, Ktg + c * 8u, D_, ulong2(0, 0), true);   // transposed
        simdgroup_multiply_accumulate(Sf, Qf[c], KTf, Sf);
    }
    simdgroup_store(Sf, Stg, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // per-row online softmax (lanes 0..7 own rows 0..7)
    if (sg < 8u) {
        uint gi = qi0 + sg;
        float mprev = Mtg[sg];
        float mcur = mprev;
        float row[8];
        for (uint c = 0; c < 8u; ++c) {
            uint gj = kj0 + c;
            float s = Stg[sg * 8u + c] * scl;
            if (gj > gi || gj >= N_) s = -INFINITY;
            row[c] = s;
            mcur = metal::max(mcur, s);
        }
        float corr = metal::exp(mprev - mcur);
        float lsum = 0.0f;
        for (uint c = 0; c < 8u; ++c) {
            float p = metal::exp(row[c] - mcur);
            Ptg[sg * 8u + c] = p;
            lsum += p;
        }
        Ltg[sg] = Ltg[sg] * corr + lsum;
        Mtg[sg] = mcur;
        Ctg[sg] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rescale existing O by per-row corr, using all 32 lanes (coalesced)
    for (uint t = sg; t < 8u * D_; t += 32u) Otg[t] *= Ctg[t / D_];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O(8 x D) += P(8 x 8) . V(8 x D)
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 Pf, Vf, Of;
        simdgroup_load(Pf, Ptg, 8);
        simdgroup_load(Vf, Vtg + c * 8u, D_);
        simdgroup_load(Of, Otg + c * 8u, D_);
        simdgroup_multiply_accumulate(Of, Pf, Vf, Of);
        simdgroup_store(Of, Otg + c * 8u, D_);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (sg < 8u) {
    uint gi = qi0 + sg;
    if (gi < N_) {
        float inv = 1.0f / Ltg[sg];
        for (uint d = 0; d < D_; ++d) O[base + gi * D_ + d] = Otg[sg * D_ + d] * inv;
        LSE[bh * N_ + gi] = Mtg[sg] + metal::log(Ltg[sg]);
    }
}

// @kernel flash_delta
// Delta[i] = Σ_d O[i,d]*dO[i,d]. One simdgroup per row; lane d owns channel d.
uint gid = thread_position_in_grid.x;
uint lane = gid & 31u;
uint row = gid >> 5;
uint N_ = N[0];
uint D_ = D[0];
uint i = row % N_;
uint bh = row / N_;
uint base = (bh * N_ + i) * D_;
const uint NCH = (D_ + 31u) / 32u;
float partial = 0.0f;
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    if (d < D_) partial += O[base + d] * dO[base + d];
}
float acc = simd_sum(partial);
if (lane == 0) Delta[bh * N_ + i] = acc;

// @kernel flash_dV
// dV[j,:] = dV_in[j,:] + Σ_{i>=j} P_ij*dO[i,:]. One simdgroup per 8-KEY block.
// P_ij = exp(scale*Q_i.K_j - LSE[i]); dV = P^T . dO over query blocks i >= j.
uint sg = thread_index_in_simdgroup;
uint blk = threadgroup_position_in_grid.y;
uint N_ = N[0];
uint D_ = D[0];
float scl = scale[0];
uint NCH = D_ / 8u;
uint nkb = (N_ + 7u) / 8u;
uint bh = blk / nkb;
uint kbk = blk % nkb;
uint base = (bh * N_) * D_;
uint kj0 = kbk * 8u;

threadgroup float Ktg[8 * 32];
threadgroup float dVtg[8 * 32];
threadgroup float Qtg[8 * 32];
threadgroup float dOtg[8 * 32];
threadgroup float Ptg[8 * 8];

for (uint t = sg; t < 8u * D_; t += 32u) {
    uint r = t / D_, c = t % D_;
    uint gj = kj0 + r;
    Ktg[t] = (gj < N_) ? K[base + gj * D_ + c] : 0.0f;
    dVtg[t] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 Kf[4];
for (uint c = 0; c < NCH; ++c) simdgroup_load(Kf[c], Ktg + c * 8u, D_);

uint qb0 = kj0 / 8u;                       // causal: query blocks i >= kj0
uint nqb = (N_ + 7u) / 8u;
for (uint qb = qb0; qb < nqb; ++qb) {
    uint qi0 = qb * 8u;
    for (uint t = sg; t < 8u * D_; t += 32u) {
        uint r = t / D_, c = t % D_;
        uint gi = qi0 + r;
        Qtg[t] = (gi < N_) ? Q[base + gi * D_ + c] : 0.0f;
        dOtg[t] = (gi < N_) ? dO[base + gi * D_ + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Qf[4];
    for (uint c = 0; c < NCH; ++c) simdgroup_load(Qf[c], Qtg + c * 8u, D_);
    simdgroup_float8x8 Sf = simdgroup_float8x8(0);
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 KTf;
        simdgroup_load(KTf, Ktg + c * 8u, D_, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Sf, Qf[c], KTf, Sf);
    }
    simdgroup_store(Sf, Ptg, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // P_ij = exp(scale*S - LSE[i]); mask non-causal (j>i) / OOB to 0
    if (sg < 8u) {
        uint gi = qi0 + sg;
        float lse = (gi < N_) ? LSE[bh * N_ + gi] : 0.0f;
        for (uint c = 0; c < 8u; ++c) {
            uint gj = kj0 + c;
            float p = metal::exp(scl * Ptg[sg * 8u + c] - lse);
            if (gj > gi || gj >= N_ || gi >= N_) p = 0.0f;
            Ptg[sg * 8u + c] = p;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dV(8k x D) += P^T(8k x 8q) . dO(8q x D)
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 PTf, dOf, dVf;
        simdgroup_load(PTf, Ptg, 8, ulong2(0, 0), true);
        simdgroup_load(dOf, dOtg + c * 8u, D_);
        simdgroup_load(dVf, dVtg + c * 8u, D_);
        simdgroup_multiply_accumulate(dVf, PTf, dOf, dVf);
        simdgroup_store(dVf, dVtg + c * 8u, D_);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint t = sg; t < 8u * D_; t += 32u) {
    uint r = t / D_, c = t % D_;
    uint gj = kj0 + r;
    if (gj < N_) out[base + gj * D_ + c] = dV_in[base + gj * D_ + c] + dVtg[t];
}

// @kernel flash_dQ
// dQ[i,:] = dQ_in[i,:] + scale * Σ_{j<=i} ds_ij*K[j,:]. One simdgroup per 8-QUERY
// block. ds_ij = P_ij*(dp_ij - Delta[i]); dp_ij = Σ_d dO[i,d]*V[j,d]. The MMA
// accumulates the UNSCALED Σ ds.K into dQtg (init 0), then writes dQ_in + scale*acc.
uint sg = thread_index_in_simdgroup;
uint blk = threadgroup_position_in_grid.y;
uint N_ = N[0];
uint D_ = D[0];
float scl = scale[0];
uint NCH = D_ / 8u;
uint nqb = (N_ + 7u) / 8u;
uint bh = blk / nqb;
uint qbk = blk % nqb;
uint base = (bh * N_) * D_;
uint qi0 = qbk * 8u;

threadgroup float Qtg[8 * 32];
threadgroup float dOtg[8 * 32];
threadgroup float dQtg[8 * 32];
threadgroup float Ktg[8 * 32];
threadgroup float Vtg[8 * 32];
threadgroup float Stg[8 * 8];
threadgroup float DPtg[8 * 8];
threadgroup float DStg[8 * 8];

for (uint t = sg; t < 8u * D_; t += 32u) {
    uint r = t / D_, c = t % D_;
    uint gi = qi0 + r;
    Qtg[t] = (gi < N_) ? Q[base + gi * D_ + c] : 0.0f;
    dOtg[t] = (gi < N_) ? dO[base + gi * D_ + c] : 0.0f;
    dQtg[t] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 Qf[4], dOf[4];
for (uint c = 0; c < NCH; ++c) { simdgroup_load(Qf[c], Qtg + c * 8u, D_); simdgroup_load(dOf[c], dOtg + c * 8u, D_); }

uint qimax = qi0 + 7u;
uint klast = (qimax < N_ - 1u) ? qimax : (N_ - 1u);
uint kbmax = klast / 8u;
for (uint kb = 0; kb <= kbmax; ++kb) {
    uint kj0 = kb * 8u;
    for (uint t = sg; t < 8u * D_; t += 32u) {
        uint r = t / D_, c = t % D_;
        uint gj = kj0 + r;
        Ktg[t] = (gj < N_) ? K[base + gj * D_ + c] : 0.0f;
        Vtg[t] = (gj < N_) ? V[base + gj * D_ + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S(8q x 8k) = Q.K^T ; DP(8q x 8k) = dO.V^T
    simdgroup_float8x8 Sf = simdgroup_float8x8(0), DPf = simdgroup_float8x8(0);
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 KTf, VTf;
        simdgroup_load(KTf, Ktg + c * 8u, D_, ulong2(0, 0), true);
        simdgroup_load(VTf, Vtg + c * 8u, D_, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Sf, Qf[c], KTf, Sf);
        simdgroup_multiply_accumulate(DPf, dOf[c], VTf, DPf);
    }
    simdgroup_store(Sf, Stg, 8);
    simdgroup_store(DPf, DPtg, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < 8u) {
        uint gi = qi0 + sg;
        float lse = (gi < N_) ? LSE[bh * N_ + gi] : 0.0f;
        float de = (gi < N_) ? Delta[bh * N_ + gi] : 0.0f;
        for (uint c = 0; c < 8u; ++c) {
            uint gj = kj0 + c;
            float p = metal::exp(scl * Stg[sg * 8u + c] - lse);
            if (gj > gi || gj >= N_ || gi >= N_) p = 0.0f;
            DStg[sg * 8u + c] = p * (DPtg[sg * 8u + c] - de);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dQ(8q x D) += dS(8q x 8k) . K(8k x D)   [unscaled; scale applied at store]
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 DSf, Kf, dQf;
        simdgroup_load(DSf, DStg, 8);
        simdgroup_load(Kf, Ktg + c * 8u, D_);
        simdgroup_load(dQf, dQtg + c * 8u, D_);
        simdgroup_multiply_accumulate(dQf, DSf, Kf, dQf);
        simdgroup_store(dQf, dQtg + c * 8u, D_);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint t = sg; t < 8u * D_; t += 32u) {
    uint r = t / D_, c = t % D_;
    uint gi = qi0 + r;
    if (gi < N_) out[base + gi * D_ + c] = dQ_in[base + gi * D_ + c] + scl * dQtg[t];
}

// @kernel flash_dK
// dK[j,:] = dK_in[j,:] + scale * Σ_{i>=j} ds_ij*Q[i,:]. One simdgroup per 8-KEY
// block. The MMA accumulates the UNSCALED Σ ds^T.Q into dKtg (init 0); the store
// writes dK_in + scale*acc.
uint sg = thread_index_in_simdgroup;
uint blk = threadgroup_position_in_grid.y;
uint N_ = N[0];
uint D_ = D[0];
float scl = scale[0];
uint NCH = D_ / 8u;
uint nkb = (N_ + 7u) / 8u;
uint bh = blk / nkb;
uint kbk = blk % nkb;
uint base = (bh * N_) * D_;
uint kj0 = kbk * 8u;

threadgroup float Ktg[8 * 32];
threadgroup float Vtg[8 * 32];
threadgroup float dKtg[8 * 32];
threadgroup float Qtg[8 * 32];
threadgroup float dOtg[8 * 32];
threadgroup float Stg[8 * 8];
threadgroup float DPtg[8 * 8];
threadgroup float DStg[8 * 8];

for (uint t = sg; t < 8u * D_; t += 32u) {
    uint r = t / D_, c = t % D_;
    uint gj = kj0 + r;
    Ktg[t] = (gj < N_) ? K[base + gj * D_ + c] : 0.0f;
    Vtg[t] = (gj < N_) ? V[base + gj * D_ + c] : 0.0f;
    dKtg[t] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 Kf[4], Vf[4];
for (uint c = 0; c < NCH; ++c) { simdgroup_load(Kf[c], Ktg + c * 8u, D_); simdgroup_load(Vf[c], Vtg + c * 8u, D_); }

uint qb0 = kj0 / 8u;                       // causal: query blocks i >= kj0
uint nqb = (N_ + 7u) / 8u;
for (uint qb = qb0; qb < nqb; ++qb) {
    uint qi0 = qb * 8u;
    for (uint t = sg; t < 8u * D_; t += 32u) {
        uint r = t / D_, c = t % D_;
        uint gi = qi0 + r;
        Qtg[t] = (gi < N_) ? Q[base + gi * D_ + c] : 0.0f;
        dOtg[t] = (gi < N_) ? dO[base + gi * D_ + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Qf[4], dOf[4];
    for (uint c = 0; c < NCH; ++c) { simdgroup_load(Qf[c], Qtg + c * 8u, D_); simdgroup_load(dOf[c], dOtg + c * 8u, D_); }

    // S(8q x 8k) = Q.K^T ; DP(8q x 8k) = dO.V^T
    simdgroup_float8x8 Sf = simdgroup_float8x8(0), DPf = simdgroup_float8x8(0);
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 KTf, VTf;
        simdgroup_load(KTf, Ktg + c * 8u, D_, ulong2(0, 0), true);
        simdgroup_load(VTf, Vtg + c * 8u, D_, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Sf, Qf[c], KTf, Sf);
        simdgroup_multiply_accumulate(DPf, dOf[c], VTf, DPf);
    }
    simdgroup_store(Sf, Stg, 8);
    simdgroup_store(DPf, DPtg, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < 8u) {
        uint gi = qi0 + sg;
        float lse = (gi < N_) ? LSE[bh * N_ + gi] : 0.0f;
        float de = (gi < N_) ? Delta[bh * N_ + gi] : 0.0f;
        for (uint c = 0; c < 8u; ++c) {
            uint gj = kj0 + c;
            float p = metal::exp(scl * Stg[sg * 8u + c] - lse);
            if (gj > gi || gj >= N_ || gi >= N_) p = 0.0f;
            DStg[sg * 8u + c] = p * (DPtg[sg * 8u + c] - de);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dK(8k x D) += dS^T(8k x 8q) . Q(8q x D)
    for (uint c = 0; c < NCH; ++c) {
        simdgroup_float8x8 DSTf, Qf2, dKf;
        simdgroup_load(DSTf, DStg, 8, ulong2(0, 0), true);
        simdgroup_load(Qf2, Qtg + c * 8u, D_);
        simdgroup_load(dKf, dKtg + c * 8u, D_);
        simdgroup_multiply_accumulate(dKf, DSTf, Qf2, dKf);
        simdgroup_store(dKf, dKtg + c * 8u, D_);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint t = sg; t < 8u * D_; t += 32u) {
    uint r = t / D_, c = t % D_;
    uint gj = kj0 + r;
    if (gj < N_) out[base + gj * D_ + c] = dK_in[base + gj * D_ + c] + scl * dKtg[t];
}
