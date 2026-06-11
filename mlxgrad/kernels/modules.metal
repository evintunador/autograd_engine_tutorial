// Module Metal kernels for mlxgrad: Embedding + LayerNorm (fwd + bwd).
//
// The mlxgrad analog of cudagrad/kernels/modules.cu. Simplicity over peak perf
// (tutorial): one thread per row / per output element; test sizes are tiny.
//
// cudagrad uses atomicAdd where multiple input rows scatter into the SAME grad
// element (embedding rows sharing a token id; layernorm dw/db summed across
// rows). We avoid atomics entirely by flipping the parallelization to ONE THREAD
// PER OUTPUT ELEMENT, which then *gathers* (loops over the inputs that feed it).
// Each thread owns a distinct output slot, so the functional accumulation
// (out = grad_in + gathered contribution) is race-free.
//
// Conventions: float tensors contiguous fp32; token ids arrive as FLOATS (the
// engine is fp32-only) and are cast back to int for indexing — lossless for the
// small ids used here, range-checked in nn.Embedding. layernorm var uses
// POPULATION (/D) normalization; mean[r]/rstd[r] are computed in forward and
// reused in backward.

// ---- embedding ------------------------------------------------------------

// @kernel embedding_forward
// out[row, d] = weight[tokens[row], d]   (one thread per (row, d); grid = rows*D)
uint i = thread_position_in_grid.x;
uint D_ = D[0];
uint row = i / D_;
uint d = i % D_;
uint t = (uint)metal::round(tokens[row]);
out[i] = weight[t * D_ + d];

// @kernel embedding_backward
// dweight[v, d] = grad_in[v,d] + Σ_{row: tokens[row]==v} dout[row, d]
// one thread per WEIGHT element (v, d) (grid = V*D); gathers over rows, so the
// scatter-add that cudagrad does with atomics becomes a race-free gather.
uint i = thread_position_in_grid.x;
uint D_ = D[0];
uint ROWS = rows[0];
uint v = i / D_;
uint d = i % D_;
float acc = grad_in[i];
for (uint row = 0; row < ROWS; ++row) {
    uint t = (uint)metal::round(tokens[row]);
    if (t == v) acc += dout[row * D_ + d];
}
out[i] = acc;

// ---- layernorm ------------------------------------------------------------

// @kernel layernorm_forward
// out[r,:] = ((x[r,:] - mean) * rstd) * w + b ; also writes mean[r], rstd[r].
// ONE THREADGROUP PER ROW (grid = rows*TG, threadgroup = TG). The TG threads
// cooperatively reduce the row over the feature dim D via a grid-strided loop
// (so D > TG still works), a SIMD reduction (simd_sum), then a final reduce of
// the per-simdgroup partials through threadgroup memory. Two passes: mean, then
// the centered sum-of-squares for population (/D) variance.
uint tid = thread_position_in_threadgroup.x;
uint r = threadgroup_position_in_grid.x;
uint tgsize = threads_per_threadgroup.x;
uint D_ = D[0];
float eps = epsb[0];
uint base = r * D_;

threadgroup float psum[8];     // per-simdgroup partial Σ(x-K)  (TG<=256 -> <=8 sg)
threadgroup float psq[8];      // per-simdgroup partial Σ(x-K)²
threadgroup float bc[2];       // [0]=mean, [1]=rstd broadcast
uint lane = thread_index_in_simdgroup;
uint sgid = simdgroup_index_in_threadgroup;
uint nsimd = (tgsize + 31u) / 32u;

// --- SINGLE fused pass (one read of x), numerically stable via SHIFTED data ---
// Reduce Σ(x-K) and Σ(x-K)² with K = x[base] (the row's first element) as the
// shift, then mean = K + Σ(x-K)/D and var = (Σ(x-K)² - (Σ(x-K))²/D)/D. Shifting
// by an in-row value keeps the deviations small, so the sum-of-squares identity
// no longer cancels catastrophically (the raw E[x²]-mean² form did: e.g. D=1,
// where var is exactly 0). One read of x instead of the two-pass mean+variance,
// cutting the reduction-phase memory traffic.
float K = x[base];
float s = 0.0f, sq = 0.0f;
for (uint d = tid; d < D_; d += tgsize) { float v = x[base + d] - K; s += v; sq += v * v; }
s = simd_sum(s);
sq = simd_sum(sq);
if (lane == 0) { psum[sgid] = s; psq[sgid] = sq; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float ts = 0.0f, tsq = 0.0f;
    for (uint i = 0; i < nsimd; ++i) { ts += psum[i]; tsq += psq[i]; }
    float mu = K + ts / (float)D_;
    float var = (tsq - ts * ts / (float)D_) / (float)D_;  // population (/D)
    var = metal::max(var, 0.0f);             // guard tiny negative from rounding
    float rs = 1.0f / metal::sqrt(var + eps);
    bc[0] = mu; bc[1] = rs;
    mean[r] = mu; rstd[r] = rs;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mu = bc[0];
float rs = bc[1];

// --- normalize + affine (each thread writes its strided slice) ---
for (uint d = tid; d < D_; d += tgsize) {
    float xhat = (x[base + d] - mu) * rs;
    out[base + d] = xhat * w[d] + b[d];
}

// @kernel layernorm_backward_dx
// dx[r,d] = dx_in[r,d] + rstd[r]*(dxhat - c1 - xhat*c2)
//   dxhat = dout[r,d]*w[d];  c1 = mean_d(dxhat);  c2 = mean_d(dxhat*xhat)
// ONE THREADGROUP PER ROW (grid = rows*TG, threadgroup = TG); uses mean[r]/rstd[r]
// saved in forward. TG threads cooperatively reduce c1,c2 over D (SIMD + tg-mem),
// then each thread writes its grid-strided slice of dx.
uint tid = thread_position_in_threadgroup.x;
uint r = threadgroup_position_in_grid.x;
uint tgsize = threads_per_threadgroup.x;
uint D_ = D[0];
uint base = r * D_;
float mu = mean[r];
float rs = rstd[r];

threadgroup float p1[32];   // simd partials for c1
threadgroup float p2[32];   // simd partials for c2
threadgroup float cbroad[2];  // [0]=c1, [1]=c2 broadcast
uint lane = thread_index_in_simdgroup;
uint sgid = simdgroup_index_in_threadgroup;
uint nsimd = (tgsize + 31u) / 32u;

float c1 = 0.0f, c2 = 0.0f;
for (uint d = tid; d < D_; d += tgsize) {
    float xhat = (x[base + d] - mu) * rs;
    float dxhat = dout[base + d] * w[d];
    c1 += dxhat;
    c2 += dxhat * xhat;
}
c1 = simd_sum(c1);
c2 = simd_sum(c2);
if (lane == 0) { p1[sgid] = c1; p2[sgid] = c2; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float t1 = 0.0f, t2 = 0.0f;
    for (uint i = 0; i < nsimd; ++i) { t1 += p1[i]; t2 += p2[i]; }
    cbroad[0] = t1 / (float)D_;
    cbroad[1] = t2 / (float)D_;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float cc1 = cbroad[0];
float cc2 = cbroad[1];

for (uint d = tid; d < D_; d += tgsize) {
    float xhat = (x[base + d] - mu) * rs;
    float dxhat = dout[base + d] * w[d];
    out[base + d] = dx_in[base + d] + rs * (dxhat - cc1 - xhat * cc2);
}

// @kernel layernorm_backward_dwdb
// dw[d] = dw_in[d] + Σ_r dout[r,d]*xhat[r,d] ;  db[d] = db_in[d] + Σ_r dout[r,d]
// ONE THREADGROUP PER FEATURE d (grid = D*TG, threadgroup = TG): the TG threads
// split the ROWS gather via a grid-strided row loop, then simd_sum + a tg-memory
// merge of the per-simdgroup partials reduce to the final dw[d]/db[d]. This keeps
// the scatter-add atomic-free (each feature owns its output) while giving small-D
// launches enough threads to fill the GPU — the old one-thread-per-feature design
// left only D threads resident (e.g. D=128), starving occupancy.
uint d      = threadgroup_position_in_grid.x;
uint tid    = thread_position_in_threadgroup.x;
uint tgsize = threads_per_threadgroup.x;
uint D_     = D[0];
uint ROWS   = rows[0];
uint lane   = thread_index_in_simdgroup;
uint sgid   = simdgroup_index_in_threadgroup;
uint nsimd  = (tgsize + 31u) / 32u;

threadgroup float pw[8];   // per-simdgroup partial dw (TG<=256 -> <=8 simdgroups)
threadgroup float pb[8];   // per-simdgroup partial db

float accw = 0.0f, accb = 0.0f;
for (uint r = tid; r < ROWS; r += tgsize) {
    float xhat = (x[r * D_ + d] - mean[r]) * rstd[r];
    float g = dout[r * D_ + d];
    accw += g * xhat;
    accb += g;
}
accw = simd_sum(accw);
accb = simd_sum(accb);
if (lane == 0) { pw[sgid] = accw; pb[sgid] = accb; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float tw = 0.0f, tb = 0.0f;
    for (uint i = 0; i < nsimd; ++i) { tw += pw[i]; tb += pb[i]; }
    dw[d] = dw_in[d] + tw;
    db[d] = db_in[d] + tb;
}
