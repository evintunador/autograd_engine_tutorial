// Vectorwise Metal kernels for mlxgrad: last-dim reductions + softmax.
//
// The mlxgrad analog of cudagrad/kernels/vectorwise.cu. These ops all act along
// the FINAL dim of a contiguous fp32 tensor, viewed as an (n_rows, n_cols)
// matrix: row r occupies x[r*n_cols .. r*n_cols + n_cols-1].
//
// Design: ONE THREADGROUP PER ROW. The threadgroup's threads cooperatively
// reduce over the row's n_cols columns via a grid-strided loop (so a row wider
// than the threadgroup still works), combining per-thread partials with SIMD
// reductions (simd_sum/simd_max/simd_min) plus threadgroup memory to merge the
// per-simdgroup partials. grid = (TG * n_rows, 1, 1), threadgroup = (TG, 1, 1).
// Row index = threadgroup_position_in_grid.x; lane within row = local tid.
// Distinct rows write distinct output elements -> no cross-row races, and the
// functional accumulation (out[idx] = grad_in[idx] + contribution) needs no
// atomics.
//
// Reduction op codes (kept in sync with mlx_kernels.py's _REDUCTION_OP):
//   0=sum  1=mean  2=max  3=min  4=var  5=std
//
// var/std use POPULATION normalization (divide by C, subtracting the row MEAN
// = sum/C), so forward, backward, and torch.var/std(unbiased=False) all agree —
// the bug the suite caught in tritongrad. Do NOT divide by C-1.
//
// Threadgroup size is hardcoded to 256 (= 8 simdgroups of 32 lanes). It MUST
// match _VEC_TG in mlx_kernels.py. Threadgroup scratch arrays hold one float per
// simdgroup -> size 8 = 256/32. NOTE: the constants are inlined as literals
// (256u / 8u) rather than #define, because the .metal loader strips everything
// before the first "// @kernel" marker, so a header #define never reaches the
// kernel body.

// @kernel reduction_forward
// out[r] = REDUCE_c x[r, c].
//
// TWO LAUNCH MODES (chosen in the wrapper from n_cols), selected by rpt[0] =
// rows-per-threadgroup:
//   * NARROW (rpt > 1, used when NC <= 32): ONE SIMDGROUP PER ROW. Each
//     simdgroup reduces its row with a single simd_* op — no threadgroup memory,
//     no barriers, and every lane does useful work. We pack rpt simdgroups per
//     threadgroup so many rows stay resident, recovering the row-parallelism a
//     one-thread-per-row design has while keeping the SIMD reduction.
//   * WIDE (rpt == 1): ONE THREADGROUP PER ROW (grid-strided column loop + SIMD
//     + threadgroup-memory merge of the per-simdgroup partials).
uint opc  = op[0];
uint NC   = n_cols[0];
uint RPT  = rpt[0];
uint sl   = thread_index_in_simdgroup;
bool wantMax = (opc == 2u);
bool wantMin = (opc == 3u);

if (RPT > 1u) {
    // ---- NARROW: one simdgroup per row, NC <= 32 ----
    uint sg  = simdgroup_index_in_threadgroup;
    uint r   = threadgroup_position_in_grid.x * RPT + sg;
    if (r >= n_rows[0]) return;
    uint base = r * NC;
    float v = (sl < NC) ? x[base + sl] : (wantMax ? -INFINITY : (wantMin ? INFINITY : 0.0f));
    if (wantMax) {
        out[r] = simd_max(v);
    } else if (wantMin) {
        out[r] = simd_min(v);
    } else {
        float s = simd_sum(v);
        if (opc == 0u)       out[r] = s;                       // sum
        else if (opc == 1u)  out[r] = s / (float)NC;           // mean
        else {                                                 // var / std
            float mean = s / (float)NC;
            float d = (sl < NC) ? (x[base + sl] - mean) : 0.0f;
            float varv = simd_sum(d * d) / (float)NC;
            out[r] = (opc == 4u) ? varv : metal::sqrt(varv);
        }
    }
    return;
}

uint r    = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg   = simdgroup_index_in_threadgroup;
uint TPT  = threads_per_threadgroup.x;
uint NSG  = (TPT + 31u) / 32u;
uint base = r * NC;

threadgroup float tg_a[8u];
threadgroup float tg_b[8u];

// ---- pass 1: row sum (for sum/mean/var/std) or row max/min ----
float p;
if (wantMax) {
    p = -INFINITY;
    for (uint c = lane; c < NC; c += TPT) p = metal::max(p, x[base + c]);
    p = simd_max(p);
} else if (wantMin) {
    p = INFINITY;
    for (uint c = lane; c < NC; c += TPT) p = metal::min(p, x[base + c]);
    p = simd_min(p);
} else {
    p = 0.0f;
    for (uint c = lane; c < NC; c += TPT) p += x[base + c];
    p = simd_sum(p);
}
if (sl == 0) tg_a[sg] = p;
threadgroup_barrier(mem_flags::mem_threadgroup);

// combine per-simdgroup partials in simdgroup 0
if (sg == 0) {
    float v;
    if (wantMax)      v = (sl < NSG) ? tg_a[sl] : -INFINITY;
    else if (wantMin) v = (sl < NSG) ? tg_a[sl] :  INFINITY;
    else              v = (sl < NSG) ? tg_a[sl] :  0.0f;
    if (wantMax)      v = simd_max(v);
    else if (wantMin) v = simd_min(v);
    else              v = simd_sum(v);
    if (sl == 0) tg_b[0] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float red = tg_b[0];

if (opc == 0u) {            // sum
    if (lane == 0) out[r] = red;
} else if (opc == 1u) {     // mean
    if (lane == 0) out[r] = red / (float)NC;
} else if (opc == 2u) {     // max
    if (lane == 0) out[r] = red;
} else if (opc == 3u) {     // min
    if (lane == 0) out[r] = red;
} else {                    // var / std: second pass for sum of squared devs
    float mean = red / (float)NC;
    float q = 0.0f;
    for (uint c = lane; c < NC; c += TPT) { float d = x[base + c] - mean; q += d * d; }
    q = simd_sum(q);
    if (sl == 0) tg_a[sg] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (sl < NSG) ? tg_a[sl] : 0.0f;
        v = simd_sum(v);
        if (sl == 0) tg_b[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float varv = tg_b[0] / (float)NC;
    if (lane == 0) out[r] = (opc == 4u) ? varv : metal::sqrt(varv);
}

// @kernel reduction_backward
// out[r,c] = grad_in[r,c] + d(out_fwd[r])/d(x[r,c]) * dout[r].
// out_fwd[r] holds the forward reduction result (used by std). dout has n_rows
// elems. Two launch modes mirroring reduction_forward (see rpt below).
uint opc  = op[0];
uint NC   = n_cols[0];
uint RPT  = rpt[0];
uint sl   = thread_index_in_simdgroup;

if (RPT > 1u) {
    // ---- NARROW: one simdgroup per row, NC <= 32 ----
    uint sg  = simdgroup_index_in_threadgroup;
    uint r   = threadgroup_position_in_grid.x * RPT + sg;
    if (r >= n_rows[0]) return;
    uint base = r * NC;
    float g   = dout[r];
    bool in   = (sl < NC);
    uint idx  = base + sl;
    // All lanes run the simd reductions (out-of-range lanes contribute the
    // identity); only in-range lanes write back.
    if (opc == 0u) {                 // sum
        if (in) out[idx] = grad_in[idx] + g;
    } else if (opc == 1u) {          // mean
        if (in) out[idx] = grad_in[idx] + g / (float)NC;
    } else if (opc == 2u || opc == 3u) {  // max/min: grad to first arg-extreme
        float xv  = in ? x[idx] : (opc == 2u ? -INFINITY : INFINITY);
        float ext = (opc == 2u) ? simd_max(xv) : simd_min(xv);
        // first column equal to the extreme (smallest lane index)
        uint cand = (in && xv == ext) ? sl : NC;
        uint argm = simd_min(cand);
        if (in) out[idx] = grad_in[idx] + ((sl == argm) ? g : 0.0f);
    } else {                         // var / std
        float sv = (opc == 5u) ? out_fwd[r] : 1.0f;
        if (opc == 5u && sv == 0.0f) {
            if (in) out[idx] = grad_in[idx];
        } else {
            float xv   = in ? x[idx] : 0.0f;
            float mean = simd_sum(xv) / (float)NC;
            float coef = (opc == 4u) ? (g * 2.0f / (float)NC)
                                     : (g / ((float)NC * sv));
            if (in) out[idx] = grad_in[idx] + coef * (xv - mean);
        }
    }
    return;
}

uint r    = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg   = simdgroup_index_in_threadgroup;
uint TPT  = threads_per_threadgroup.x;
uint NSG  = (TPT + 31u) / 32u;
uint base = r * NC;
float g   = dout[r];

threadgroup float tg_a[8u];
threadgroup float tg_b[8u];

if (opc == 0u) {            // sum: +dout
    for (uint c = lane; c < NC; c += TPT) out[base + c] = grad_in[base + c] + g;
} else if (opc == 1u) {     // mean: +dout / C
    float gc = g / (float)NC;
    for (uint c = lane; c < NC; c += TPT) out[base + c] = grad_in[base + c] + gc;
} else if (opc == 2u || opc == 3u) {  // max/min: route grad to (first) arg-extreme
    // step 1: cooperative row extreme value
    float p = (opc == 2u) ? -INFINITY : INFINITY;
    for (uint c = lane; c < NC; c += TPT)
        p = (opc == 2u) ? metal::max(p, x[base + c]) : metal::min(p, x[base + c]);
    p = (opc == 2u) ? simd_max(p) : simd_min(p);
    if (sl == 0) tg_a[sg] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (sl < NSG) ? tg_a[sl] : ((opc == 2u) ? -INFINITY : INFINITY);
        v = (opc == 2u) ? simd_max(v) : simd_min(v);
        if (sl == 0) tg_b[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ext = tg_b[0];
    // step 2: cooperative argmin of column index c where x[base+c] == ext
    // (smallest such c, matching the serial "first" tie-break). Each lane walks
    // its grid-strided columns in increasing order and stops at its first hit.
    uint amin = NC;
    for (uint c = lane; c < NC; c += TPT)
        if (x[base + c] == ext) { amin = c; break; }
    uint a = simd_min(amin);
    if (sl == 0) tg_a[sg] = (float)a;   // NC small -> exact in float
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (sl < NSG) ? tg_a[sl] : (float)NC;
        v = simd_min(v);
        if (sl == 0) tg_b[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint argm = (uint)tg_b[0];
    for (uint c = lane; c < NC; c += TPT)
        out[base + c] = grad_in[base + c] + ((c == argm) ? g : 0.0f);
} else {                    // var / std: need row mean (and out_fwd for std)
    float sv = (opc == 5u) ? out_fwd[r] : 1.0f;
    if (opc == 5u && sv == 0.0f) {
        for (uint c = lane; c < NC; c += TPT) out[base + c] = grad_in[base + c];
    } else {
        float s = 0.0f;
        for (uint c = lane; c < NC; c += TPT) s += x[base + c];
        s = simd_sum(s);
        if (sl == 0) tg_a[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (sl < NSG) ? tg_a[sl] : 0.0f;
            v = simd_sum(v);
            if (sl == 0) tg_b[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = tg_b[0] / (float)NC;
        float coef = (opc == 4u) ? (g * 2.0f / (float)NC)
                                 : (g / ((float)NC * sv));
        for (uint c = lane; c < NC; c += TPT)
            out[base + c] = grad_in[base + c] + coef * (x[base + c] - mean);
    }
}

// @kernel softmax_forward
// out[r, c] = softmax(x[r, :])_c   (numerically stable). Two launch modes
// mirroring reduction_forward (see rpt below).
uint NC   = n_cols[0];
uint RPT  = rpt[0];
uint sl   = thread_index_in_simdgroup;

if (RPT > 1u) {
    // ---- NARROW: one simdgroup per row, NC <= 32 ----
    uint sg  = simdgroup_index_in_threadgroup;
    uint r   = threadgroup_position_in_grid.x * RPT + sg;
    if (r >= n_rows[0]) return;
    uint base = r * NC;
    float xv = (sl < NC) ? x[base + sl] : -INFINITY;
    float mx = simd_max(xv);
    float e  = (sl < NC) ? metal::exp(xv - mx) : 0.0f;
    float denom = simd_sum(e);
    if (sl < NC) out[base + sl] = e / denom;
    return;
}

uint r    = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg   = simdgroup_index_in_threadgroup;
uint TPT  = threads_per_threadgroup.x;
uint NSG  = (TPT + 31u) / 32u;
uint base = r * NC;

threadgroup float tg_a[8u];
threadgroup float tg_b[1];

// pass 1: row max
float p = -INFINITY;
for (uint c = lane; c < NC; c += TPT) p = metal::max(p, x[base + c]);
p = simd_max(p);
if (sl == 0) tg_a[sg] = p;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float v = (sl < NSG) ? tg_a[sl] : -INFINITY;
    v = simd_max(v);
    if (sl == 0) tg_b[0] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mx = tg_b[0];

// pass 2: exp + partial sum (write exp into out)
float s = 0.0f;
for (uint c = lane; c < NC; c += TPT) { float e = metal::exp(x[base + c] - mx); out[base + c] = e; s += e; }
s = simd_sum(s);
if (sl == 0) tg_a[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float v = (sl < NSG) ? tg_a[sl] : 0.0f;
    v = simd_sum(v);
    if (sl == 0) tg_b[0] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float denom = tg_b[0];

// pass 3: normalize
for (uint c = lane; c < NC; c += TPT) out[base + c] /= denom;

// @kernel softmax_backward
// out[r,c] = grad_in[r,c] + y[r,c] * (dout[r,c] - dot[r]),  dot[r] = sum_c dout*y
// y is the forward softmax output. Two launch modes (see rpt below).
uint NC   = n_cols[0];
uint RPT  = rpt[0];
uint sl   = thread_index_in_simdgroup;

if (RPT > 1u) {
    // ---- NARROW: one simdgroup per row, NC <= 32 ----
    uint sg  = simdgroup_index_in_threadgroup;
    uint r   = threadgroup_position_in_grid.x * RPT + sg;
    if (r >= n_rows[0]) return;
    uint base = r * NC;
    // All lanes participate in the simd reduction (out-of-range lanes add 0) so
    // simd_sum sees the full simdgroup; only in-range lanes write back.
    float yv = (sl < NC) ? y[base + sl] : 0.0f;
    float dv = (sl < NC) ? dout[base + sl] : 0.0f;
    float d  = simd_sum(dv * yv);
    if (sl < NC) out[base + sl] = grad_in[base + sl] + yv * (dv - d);
    return;
}

uint r    = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg   = simdgroup_index_in_threadgroup;
uint TPT  = threads_per_threadgroup.x;
uint NSG  = (TPT + 31u) / 32u;
uint base = r * NC;

threadgroup float tg_a[8u];
threadgroup float tg_b[1];

float dot = 0.0f;
for (uint c = lane; c < NC; c += TPT) dot += dout[base + c] * y[base + c];
dot = simd_sum(dot);
if (sl == 0) tg_a[sg] = dot;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float v = (sl < NSG) ? tg_a[sl] : 0.0f;
    v = simd_sum(v);
    if (sl == 0) tg_b[0] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float d = tg_b[0];

for (uint c = lane; c < NC; c += TPT)
    out[base + c] = grad_in[base + c] + y[base + c] * (dout[base + c] - d);
