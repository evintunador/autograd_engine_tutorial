// Flash-attention Metal kernels for mlxgrad: CAUSAL attention fwd + bwd.
//
// The mlxgrad analog of cudagrad/kernels/flash_attention.cu — but deliberately
// SIMPLER. cudagrad uses a cooperative block-per-row design (warp shuffles +
// threadgroup memory) to claw back performance. Here, as the project's
// easy-to-verify CAUSAL reference and given tiny test sizes (B*H*N = 256 rows,
// D = 32), we use the obviously-correct ONE THREAD PER ROW design: each thread
// serially walks every key j and channel d. No threadgroup memory, no
// reductions. Each thread owns a DISTINCT output row, so writing the row
// directly (and the functional accumulation out = grad_in + ...) is race-free.
//
// Layout/contract notes (match cudagrad precisely):
//   * Q/K/V/O/dO/dQ/dK/dV are (B,H,N,D) contiguous fp32; LSE/Delta are (B,H,N).
//     Q[b,h,i,d] = Q[((b*H+h)*N + i)*D + d];  LSE[b,h,i] = LSE[(b*H+h)*N + i].
//     We flatten bh = b*H+h; the grid index is bh*N + row.
//   * CAUSAL: query i attends only to keys j <= i.
//   * score s_ij = scale*(Q_i.K_j); `scale` is the MULTIPLIER passed in
//     (= sqrt(D) in the suite), used verbatim (NOT 1/sqrt(D)).
//   * forward stores LSE[i] = m_i + log(l_i); P_ij = exp(scale*Q_i.K_j - LSE[i]).
//   * backward kernels accumulate functionally into dQ_in/dK_in/dV_in.

// @kernel flash_forward
// O[i,:], LSE[i] for one query row i (grid = B*H*N). Two-pass stable softmax.
uint idx = thread_position_in_grid.x;
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint i = idx % N_;
uint bh = idx / N_;
uint qbase = (bh * N_ + i) * D_;
uint obase = qbase;  // O has same (B,H,N,D) layout as Q
// pass 1: running max of scores over j <= i
float m = -INFINITY;
for (uint j = 0; j <= i; ++j) {
    uint kbase = (bh * N_ + j) * D_;
    float s = 0.0f;
    for (uint d = 0; d < D_; ++d) s += Q[qbase + d] * K[kbase + d];
    s *= sc;
    m = metal::max(m, s);
}
// pass 2: l_i and O[i,:] = (Σ_j p_ij V[j,:]) / l_i
for (uint d = 0; d < D_; ++d) O[obase + d] = 0.0f;
float l = 0.0f;
for (uint j = 0; j <= i; ++j) {
    uint kbase = (bh * N_ + j) * D_;
    float s = 0.0f;
    for (uint d = 0; d < D_; ++d) s += Q[qbase + d] * K[kbase + d];
    s *= sc;
    float p = metal::exp(s - m);
    l += p;
    for (uint d = 0; d < D_; ++d) O[obase + d] += p * V[kbase + d];
}
float inv = 1.0f / l;
for (uint d = 0; d < D_; ++d) O[obase + d] *= inv;
LSE[bh * N_ + i] = m + metal::log(l);

// @kernel flash_delta
// Delta[i] = Σ_d O[i,d]*dO[i,d]   (grid = B*H*N)
uint idx = thread_position_in_grid.x;
uint N_ = N[0];
uint D_ = D[0];
uint i = idx % N_;
uint bh = idx / N_;
uint base = (bh * N_ + i) * D_;
float acc = 0.0f;
for (uint d = 0; d < D_; ++d) acc += O[base + d] * dO[base + d];
Delta[bh * N_ + i] = acc;

// @kernel flash_dV
// dV[j,:] = dV_in[j,:] + Σ_{i>=j} P_ij * dO[i,:]   (grid = B*H*N; owns key row j)
uint idx = thread_position_in_grid.x;
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint j = idx % N_;
uint bh = idx / N_;
uint jbase = (bh * N_ + j) * D_;
for (uint d = 0; d < D_; ++d) out[jbase + d] = dV_in[jbase + d];
for (uint i = j; i < N_; ++i) {
    uint ibase = (bh * N_ + i) * D_;
    float s = 0.0f;
    for (uint d = 0; d < D_; ++d) s += Q[ibase + d] * K[jbase + d];
    float p = metal::exp(sc * s - LSE[bh * N_ + i]);
    for (uint d = 0; d < D_; ++d) out[jbase + d] += p * dO[ibase + d];
}

// @kernel flash_dQ
// dQ[i,:] = dQ_in[i,:] + scale * Σ_{j<=i} ds_ij * K[j,:]   (grid = B*H*N; owns query row i)
//   dp_ij = Σ_d dO[i,d]*V[j,d];  ds_ij = P_ij*(dp_ij - Delta[i])
uint idx = thread_position_in_grid.x;
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint i = idx % N_;
uint bh = idx / N_;
uint ibase = (bh * N_ + i) * D_;
float lse_i = LSE[bh * N_ + i];
float delta_i = Delta[bh * N_ + i];
for (uint d = 0; d < D_; ++d) out[ibase + d] = dQ_in[ibase + d];
for (uint j = 0; j <= i; ++j) {
    uint jbase = (bh * N_ + j) * D_;
    float s = 0.0f, dp = 0.0f;
    for (uint d = 0; d < D_; ++d) { s += Q[ibase + d] * K[jbase + d]; dp += dO[ibase + d] * V[jbase + d]; }
    float p = metal::exp(sc * s - lse_i);
    float ds = p * (dp - delta_i);
    float coef = sc * ds;
    for (uint d = 0; d < D_; ++d) out[ibase + d] += coef * K[jbase + d];
}

// @kernel flash_dK
// dK[j,:] = dK_in[j,:] + scale * Σ_{i>=j} ds_ij * Q[i,:]   (grid = B*H*N; owns key row j)
uint idx = thread_position_in_grid.x;
uint N_ = N[0];
uint D_ = D[0];
float sc = scale[0];
uint j = idx % N_;
uint bh = idx / N_;
uint jbase = (bh * N_ + j) * D_;
for (uint d = 0; d < D_; ++d) out[jbase + d] = dK_in[jbase + d];
for (uint i = j; i < N_; ++i) {
    uint ibase = (bh * N_ + i) * D_;
    float s = 0.0f, dp = 0.0f;
    for (uint d = 0; d < D_; ++d) { s += Q[ibase + d] * K[jbase + d]; dp += dO[ibase + d] * V[jbase + d]; }
    float p = metal::exp(sc * s - LSE[bh * N_ + i]);
    float ds = p * (dp - Delta[bh * N_ + i]);
    float coef = sc * ds;
    for (uint d = 0; d < D_; ++d) out[jbase + d] += coef * Q[ibase + d];
}
