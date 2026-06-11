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
// ONE THREAD PER ROW (grid = rows); each thread owns its row, no races.
uint r = thread_position_in_grid.x;
uint D_ = D[0];
float eps = epsb[0];
uint base = r * D_;
float s = 0.0f;
for (uint d = 0; d < D_; ++d) s += x[base + d];
float mu = s / (float)D_;
float acc = 0.0f;
for (uint d = 0; d < D_; ++d) { float diff = x[base + d] - mu; acc += diff * diff; }
float var = acc / (float)D_;             // population (/D) normalization
float rs = 1.0f / metal::sqrt(var + eps);
mean[r] = mu;
rstd[r] = rs;
for (uint d = 0; d < D_; ++d) {
    float xhat = (x[base + d] - mu) * rs;
    out[base + d] = xhat * w[d] + b[d];
}

// @kernel layernorm_backward_dx
// dx[r,d] = dx_in[r,d] + rstd[r]*(dxhat - c1 - xhat*c2)
//   dxhat = dout[r,d]*w[d];  c1 = mean_d(dxhat);  c2 = mean_d(dxhat*xhat)
// ONE THREAD PER ROW (grid = rows); uses mean[r]/rstd[r] saved in forward.
uint r = thread_position_in_grid.x;
uint D_ = D[0];
uint base = r * D_;
float mu = mean[r];
float rs = rstd[r];
float c1 = 0.0f, c2 = 0.0f;
for (uint d = 0; d < D_; ++d) {
    float xhat = (x[base + d] - mu) * rs;
    float dxhat = dout[base + d] * w[d];
    c1 += dxhat;
    c2 += dxhat * xhat;
}
c1 /= (float)D_;
c2 /= (float)D_;
for (uint d = 0; d < D_; ++d) {
    float xhat = (x[base + d] - mu) * rs;
    float dxhat = dout[base + d] * w[d];
    out[base + d] = dx_in[base + d] + rs * (dxhat - c1 - xhat * c2);
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
