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

threadgroup float partial[32];   // <=32 simdgroups (TG<=1024); holds simd partials
uint lane = thread_index_in_simdgroup;
uint sgid = simdgroup_index_in_threadgroup;
uint nsimd = (tgsize + 31u) / 32u;

// --- pass 1: mean ---
float s = 0.0f;
for (uint d = tid; d < D_; d += tgsize) s += x[base + d];
s = simd_sum(s);
if (lane == 0) partial[sgid] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float tot = 0.0f;
    for (uint i = 0; i < nsimd; ++i) tot += partial[i];
    partial[0] = tot / (float)D_;        // store mean for broadcast
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mu = partial[0];
threadgroup_barrier(mem_flags::mem_threadgroup);

// --- pass 2: population variance ---
float acc = 0.0f;
for (uint d = tid; d < D_; d += tgsize) { float diff = x[base + d] - mu; acc += diff * diff; }
acc = simd_sum(acc);
if (lane == 0) partial[sgid] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float tot = 0.0f;
    for (uint i = 0; i < nsimd; ++i) tot += partial[i];
    float var = tot / (float)D_;         // population (/D) normalization
    partial[0] = 1.0f / metal::sqrt(var + eps);   // store rstd for broadcast
    mean[r] = mu;
    rstd[r] = partial[0];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float rs = partial[0];

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
// ONE THREAD PER FEATURE d (grid = D); gathers over rows (race-free, no atomics).
uint d = thread_position_in_grid.x;
uint D_ = D[0];
uint ROWS = rows[0];
float accw = dw_in[d];
float accb = db_in[d];
for (uint r = 0; r < ROWS; ++r) {
    float xhat = (x[r * D_ + d] - mean[r]) * rstd[r];
    float g = dout[r * D_ + d];
    accw += g * xhat;
    accb += g;
}
dw[d] = accw;
db[d] = accb;
