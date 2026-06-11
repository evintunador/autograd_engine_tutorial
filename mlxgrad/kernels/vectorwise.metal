// Vectorwise Metal kernels for mlxgrad: last-dim reductions + softmax.
//
// The mlxgrad analog of cudagrad/kernels/vectorwise.cu. These ops all act along
// the FINAL dim of a contiguous fp32 tensor, viewed as an (n_rows, n_cols)
// matrix: row r occupies x[r*n_cols .. r*n_cols + n_cols-1].
//
// Simplicity over peak perf (tutorial): ONE THREAD PER ROW (grid = n_rows). Each
// thread loops over the n_cols columns of its row. Distinct rows write distinct
// output elements, so there are no races and the functional accumulation
// (out[idx] = grad_in[idx] + contribution) needs no atomics.
//
// Reduction op codes (kept in sync with mlx_kernels.py's _REDUCTION_OP):
//   0=sum  1=mean  2=max  3=min  4=var  5=std
//
// var/std use POPULATION normalization (divide by C, subtracting the row MEAN
// = sum/C), so forward, backward, and torch.var/std(unbiased=False) all agree —
// the bug the suite caught in tritongrad. Do NOT divide by C-1.

// @kernel reduction_forward
// out[r] = REDUCE_c x[r, c]   (one thread per row r)
uint r = thread_position_in_grid.x;
uint NC = n_cols[0];
uint base = r * NC;
float result;
switch (op[0]) {
    case 0u: {  // sum
        float s = 0.0f;
        for (uint c = 0; c < NC; ++c) s += x[base + c];
        result = s;
        break;
    }
    case 1u: {  // mean
        float s = 0.0f;
        for (uint c = 0; c < NC; ++c) s += x[base + c];
        result = s / (float)NC;
        break;
    }
    case 2u: {  // max
        float m = x[base];
        for (uint c = 1; c < NC; ++c) m = metal::max(m, x[base + c]);
        result = m;
        break;
    }
    case 3u: {  // min
        float m = x[base];
        for (uint c = 1; c < NC; ++c) m = metal::min(m, x[base + c]);
        result = m;
        break;
    }
    case 4u: {  // var (population): mean of squared deviations from row mean
        float s = 0.0f;
        for (uint c = 0; c < NC; ++c) s += x[base + c];
        float mean = s / (float)NC;
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) { float d = x[base + c] - mean; acc += d * d; }
        result = acc / (float)NC;
        break;
    }
    default: {  // 5: std = sqrt(population var)
        float s = 0.0f;
        for (uint c = 0; c < NC; ++c) s += x[base + c];
        float mean = s / (float)NC;
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) { float d = x[base + c] - mean; acc += d * d; }
        result = metal::sqrt(acc / (float)NC);
        break;
    }
}
out[r] = result;

// @kernel reduction_backward
// out[r,c] = grad_in[r,c] + d(out_fwd[r])/d(x[r,c]) * dout[r]   (one thread per row)
// out_fwd[r] holds the forward reduction result (used by std). dout has n_rows elems.
uint r = thread_position_in_grid.x;
uint NC = n_cols[0];
uint base = r * NC;
float g = dout[r];
switch (op[0]) {
    case 0u: {  // sum: +dout
        for (uint c = 0; c < NC; ++c) out[base + c] = grad_in[base + c] + g;
        break;
    }
    case 1u: {  // mean: +dout / C
        float gc = g / (float)NC;
        for (uint c = 0; c < NC; ++c) out[base + c] = grad_in[base + c] + gc;
        break;
    }
    case 2u: {  // max: route grad to the (first) max element only
        float m = x[base];
        uint argm = 0;
        for (uint c = 1; c < NC; ++c) if (x[base + c] > m) { m = x[base + c]; argm = c; }
        for (uint c = 0; c < NC; ++c)
            out[base + c] = grad_in[base + c] + ((c == argm) ? g : 0.0f);
        break;
    }
    case 3u: {  // min: route grad to the (first) min element only
        float m = x[base];
        uint argm = 0;
        for (uint c = 1; c < NC; ++c) if (x[base + c] < m) { m = x[base + c]; argm = c; }
        for (uint c = 0; c < NC; ++c)
            out[base + c] = grad_in[base + c] + ((c == argm) ? g : 0.0f);
        break;
    }
    case 4u: {  // var: +dout * 2*(x - mean)/C
        float s = 0.0f;
        for (uint c = 0; c < NC; ++c) s += x[base + c];
        float mean = s / (float)NC;
        float coef = g * 2.0f / (float)NC;
        for (uint c = 0; c < NC; ++c)
            out[base + c] = grad_in[base + c] + coef * (x[base + c] - mean);
        break;
    }
    default: {  // 5: std: +dout * (x - mean)/(C * out_fwd); guard out_fwd==0 -> 0
        float sv = out_fwd[r];
        if (sv == 0.0f) {
            for (uint c = 0; c < NC; ++c) out[base + c] = grad_in[base + c];
            break;
        }
        float s = 0.0f;
        for (uint c = 0; c < NC; ++c) s += x[base + c];
        float mean = s / (float)NC;
        float coef = g / ((float)NC * sv);
        for (uint c = 0; c < NC; ++c)
            out[base + c] = grad_in[base + c] + coef * (x[base + c] - mean);
        break;
    }
}

// @kernel softmax_forward
// out[r, c] = softmax(x[r, :])_c   (numerically stable; one thread per row)
uint r = thread_position_in_grid.x;
uint NC = n_cols[0];
uint base = r * NC;
float mx = x[base];
for (uint c = 1; c < NC; ++c) mx = metal::max(mx, x[base + c]);
float s = 0.0f;
for (uint c = 0; c < NC; ++c) { float e = metal::exp(x[base + c] - mx); out[base + c] = e; s += e; }
for (uint c = 0; c < NC; ++c) out[base + c] /= s;

// @kernel softmax_backward
// out[r,c] = grad_in[r,c] + y[r,c] * (dout[r,c] - dot[r]),  dot[r] = sum_c dout*y
// (one thread per row). y is the forward softmax output.
uint r = thread_position_in_grid.x;
uint NC = n_cols[0];
uint base = r * NC;
float dot = 0.0f;
for (uint c = 0; c < NC; ++c) dot += dout[base + c] * y[base + c];
for (uint c = 0; c < NC; ++c)
    out[base + c] = grad_in[base + c] + y[base + c] * (dout[base + c] - dot);
