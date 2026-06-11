// Flash-attention Metal kernels for mlxgrad: CAUSAL attention fwd + bwd.
//
// The mlxgrad analog of cudagrad/kernels/flash_attention.cu. COOPERATIVE design:
// ONE SIMD GROUP (32 lanes) per (b,h,row). The launch grid is (32, B*H*N, 1) with
// threadgroup (32,1,1) so each simdgroup owns one output row and lane `d` owns
// channel d. Dot products Q_i.K_j (and dO_i.V_j etc.) are formed by each lane
// multiplying its own channel and reducing with `simd_sum` (which broadcasts the
// result to all 32 lanes). Per-lane accumulators (O[i,:], dQ, dK, dV) keep one
// channel each. Loads of K/V/Q rows are coalesced across the 32 lanes. For D<32
// the extra lanes contribute 0 (guarded); for D>32 each lane loops over a strided
// set of channels (d, d+32, ...). Far better occupancy than one-thread-per-row.
//
// Layout/contract notes (match cudagrad precisely):
//   * Q/K/V/O/dO/dQ/dK/dV are (B,H,N,D) contiguous fp32; LSE/Delta are (B,H,N).
//     Q[b,h,i,d] = Q[((b*H+h)*N + i)*D + d];  LSE[b,h,i] = LSE[(b*H+h)*N + i].
//     We flatten bh = b*H+h; the row index is bh*N + row (= grid.y).
//   * CAUSAL: query i attends only to keys j <= i.
//   * score s_ij = scale*(Q_i.K_j); `scale` is the MULTIPLIER passed in
//     (= sqrt(D) in the suite), used verbatim (NOT 1/sqrt(D)).
//   * forward stores LSE[i] = m_i + log(l_i); P_ij = exp(scale*Q_i.K_j - LSE[i]).
//   * backward kernels accumulate functionally into dQ_in/dK_in/dV_in.

// @kernel flash_forward
// O[i,:], LSE[i] for one query row i. One simdgroup per row; lane d owns channel d.
// Online (single-pass) stable softmax: running max m, running denom l, per-lane
// accumulator acc_d = Σ p_ij V[j,d] (rescaled when m grows).
uint gid = thread_position_in_grid.x;
uint lane = gid & 31u;            // 0..31, owns channel(s) lane, lane+32, ...
uint row = gid >> 5;             // global row = bh*N + i
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint i = row % N_;
uint bh = row / N_;
uint qbase = (bh * N_ + i) * D_;
uint obase = qbase;

// per-lane accumulators for the (up to ceil(D/32)) channels this lane owns
const uint NCH = (D_ + 31u) / 32u;   // channels per lane (1 when D<=32)
float acc[8];
for (uint c = 0; c < NCH; ++c) acc[c] = 0.0f;
// cache this lane's Q channels
float qreg[8];
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    qreg[c] = (d < D_) ? Q[qbase + d] : 0.0f;
}

float m = -INFINITY;
float l = 0.0f;
for (uint j = 0; j <= i; ++j) {
    uint kbase = (bh * N_ + j) * D_;
    float partial = 0.0f;
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) partial += qreg[c] * K[kbase + d];
    }
    float s = simd_sum(partial) * sc;     // broadcast to all lanes
    float m_new = metal::max(m, s);
    float corr = metal::exp(m - m_new);   // rescale factor for old accumulators
    float p = metal::exp(s - m_new);
    l = l * corr + p;
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) acc[c] = acc[c] * corr + p * V[kbase + d];
    }
    m = m_new;
}
float inv = 1.0f / l;
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    if (d < D_) O[obase + d] = acc[c] * inv;
}
if (lane == 0) LSE[bh * N_ + i] = m + metal::log(l);

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
// dV[j,:] = dV_in[j,:] + Σ_{i>=j} P_ij * dO[i,:]. One simdgroup per KEY row j;
// lane d owns channel d. P_ij needs the dot Q_i.K_j (reduced via simd_sum).
uint gid = thread_position_in_grid.x;
uint lane = gid & 31u;
uint row = gid >> 5;
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint j = row % N_;
uint bh = row / N_;
uint jbase = (bh * N_ + j) * D_;
const uint NCH = (D_ + 31u) / 32u;
float kreg[8];
float acc[8];
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    kreg[c] = (d < D_) ? K[jbase + d] : 0.0f;
    acc[c] = 0.0f;
}
for (uint i = j; i < N_; ++i) {
    uint ibase = (bh * N_ + i) * D_;
    float partial = 0.0f;
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) partial += Q[ibase + d] * kreg[c];
    }
    float s = simd_sum(partial);
    float p = metal::exp(sc * s - LSE[bh * N_ + i]);
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) acc[c] += p * dO[ibase + d];
    }
}
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    if (d < D_) out[jbase + d] = dV_in[jbase + d] + acc[c];
}

// @kernel flash_dQ
// dQ[i,:] = dQ_in[i,:] + scale * Σ_{j<=i} ds_ij * K[j,:]. One simdgroup per QUERY
// row i; lane d owns channel d. dp_ij = Σ_d dO[i,d]*V[j,d]; ds_ij = P_ij*(dp - Delta[i]).
uint gid = thread_position_in_grid.x;
uint lane = gid & 31u;
uint row = gid >> 5;
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint i = row % N_;
uint bh = row / N_;
uint ibase = (bh * N_ + i) * D_;
float lse_i = LSE[bh * N_ + i];
float delta_i = Delta[bh * N_ + i];
const uint NCH = (D_ + 31u) / 32u;
float qreg[8];
float doreg[8];
float acc[8];
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    qreg[c]  = (d < D_) ? Q[ibase + d]  : 0.0f;
    doreg[c] = (d < D_) ? dO[ibase + d] : 0.0f;
    acc[c] = 0.0f;
}
for (uint j = 0; j <= i; ++j) {
    uint jbase = (bh * N_ + j) * D_;
    float ps = 0.0f, pdp = 0.0f;
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) { float kd = K[jbase + d]; ps += qreg[c] * kd; pdp += doreg[c] * V[jbase + d]; }
    }
    float s  = simd_sum(ps);
    float dp = simd_sum(pdp);
    float p = metal::exp(sc * s - lse_i);
    float ds = p * (dp - delta_i);
    float coef = sc * ds;
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) acc[c] += coef * K[jbase + d];
    }
}
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    if (d < D_) out[ibase + d] = dQ_in[ibase + d] + acc[c];
}

// @kernel flash_dK
// dK[j,:] = dK_in[j,:] + scale * Σ_{i>=j} ds_ij * Q[i,:]. One simdgroup per KEY
// row j; lane d owns channel d.
uint gid = thread_position_in_grid.x;
uint lane = gid & 31u;
uint row = gid >> 5;
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint j = row % N_;
uint bh = row / N_;
uint jbase = (bh * N_ + j) * D_;
const uint NCH = (D_ + 31u) / 32u;
float kreg[8];
float vreg[8];
float acc[8];
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    kreg[c] = (d < D_) ? K[jbase + d] : 0.0f;
    vreg[c] = (d < D_) ? V[jbase + d] : 0.0f;
    acc[c] = 0.0f;
}
for (uint i = j; i < N_; ++i) {
    uint ibase = (bh * N_ + i) * D_;
    float ps = 0.0f, pdp = 0.0f;
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) { ps += Q[ibase + d] * kreg[c]; pdp += dO[ibase + d] * vreg[c]; }
    }
    float s  = simd_sum(ps);
    float dp = simd_sum(pdp);
    float p = metal::exp(sc * s - LSE[bh * N_ + i]);
    float ds = p * (dp - Delta[bh * N_ + i]);
    float coef = sc * ds;
    for (uint c = 0; c < NCH; ++c) {
        uint d = lane + c * 32u;
        if (d < D_) acc[c] += coef * Q[ibase + d];
    }
}
for (uint c = 0; c < NCH; ++c) {
    uint d = lane + c * 32u;
    if (d < D_) out[jbase + d] = dK_in[jbase + d] + acc[c];
}
