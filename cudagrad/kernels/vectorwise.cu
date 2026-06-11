// Vectorwise CUDA kernels for cudagrad: last-dim reductions + softmax.
//
// These ops all act along the FINAL dim of a contiguous fp32 tensor, viewed as
// an (n_rows, n_cols) matrix: row r occupies x[r*n_cols .. r*n_cols + n_cols-1].
// Mirrors tritongrad/kernels/vectorwise.py (the GPU-verified reference) for the
// exact math.
//
// Simplicity over peak perf (tutorial): ONE THREAD PER ROW. Each thread loops
// over the n_cols columns of its row. With distinct rows writing distinct output
// elements (and, in backward, distinct dx columns) there are no races, so plain
// `+=` accumulation suffices — no atomics needed.
//
// Reduction op codes (kept in sync with cuda_kernels.py's _REDUCTION_OP):
//   0=sum  1=mean  2=max  3=min  4=var  5=std
//
// var/std use POPULATION normalization (divide by C, subtracting the row MEAN
// = sum/C), so forward, backward, and torch.var/std(unbiased=False) all agree.
// This is the bug the suite caught in tritongrad; do not divide by C-1 here.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace {

constexpr int THREADS = 256;

inline int64_t n_blocks(int64_t n) { return (n + THREADS - 1) / THREADS; }

// out[r] = REDUCE_c x[r, c]   (one thread per row r)
__global__ void reduction_forward_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int64_t n_rows, int64_t n_cols, int op) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const float* row = x + r * n_cols;

    float result;
    switch (op) {
        case 0: {  // sum
            float s = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) s += row[c];
            result = s;
            break;
        }
        case 1: {  // mean
            float s = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) s += row[c];
            result = s / (float)n_cols;
            break;
        }
        case 2: {  // max
            float m = row[0];
            for (int64_t c = 1; c < n_cols; ++c) m = fmaxf(m, row[c]);
            result = m;
            break;
        }
        case 3: {  // min
            float m = row[0];
            for (int64_t c = 1; c < n_cols; ++c) m = fminf(m, row[c]);
            result = m;
            break;
        }
        case 4: {  // var (population): mean of squared deviations from row mean
            float s = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) s += row[c];
            float mean = s / (float)n_cols;
            float acc = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) {
                float d = row[c] - mean;
                acc += d * d;
            }
            result = acc / (float)n_cols;
            break;
        }
        default: {  // 5: std = sqrt(population var)
            float s = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) s += row[c];
            float mean = s / (float)n_cols;
            float acc = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) {
                float d = row[c] - mean;
                acc += d * d;
            }
            result = sqrtf(acc / (float)n_cols);
            break;
        }
    }
    out[r] = result;
}

// dx[r, c] += d(out[r])/d(x[r,c]) * dout[r]   (ACCUMULATES; one thread per row)
// out[r] holds the forward result (used by std).
__global__ void reduction_backward_kernel(const float* __restrict__ x,
                                          float* __restrict__ dx,
                                          const float* __restrict__ dout,
                                          const float* __restrict__ out,
                                          int64_t n_rows, int64_t n_cols, int op) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const float* row = x + r * n_cols;
    float* drow = dx + r * n_cols;
    float g = dout[r];

    switch (op) {
        case 0: {  // sum: dx += dout
            for (int64_t c = 0; c < n_cols; ++c) drow[c] += g;
            break;
        }
        case 1: {  // mean: dx += dout / C
            float gc = g / (float)n_cols;
            for (int64_t c = 0; c < n_cols; ++c) drow[c] += gc;
            break;
        }
        case 2: {  // max: route grad to the (first) max element only
            float m = row[0];
            int64_t argm = 0;
            for (int64_t c = 1; c < n_cols; ++c) {
                if (row[c] > m) { m = row[c]; argm = c; }
            }
            drow[argm] += g;
            break;
        }
        case 3: {  // min: route grad to the (first) min element only
            float m = row[0];
            int64_t argm = 0;
            for (int64_t c = 1; c < n_cols; ++c) {
                if (row[c] < m) { m = row[c]; argm = c; }
            }
            drow[argm] += g;
            break;
        }
        case 4: {  // var: dx += dout * 2*(x - mean)/C
            float s = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) s += row[c];
            float mean = s / (float)n_cols;
            float coef = g * 2.0f / (float)n_cols;
            for (int64_t c = 0; c < n_cols; ++c) drow[c] += coef * (row[c] - mean);
            break;
        }
        default: {  // 5: std: dx += dout * (x - mean)/(C * out);  guard out==0 -> 0
            float sv = out[r];
            if (sv == 0.0f) break;  // forward result is std; 0 -> contribute nothing
            float s = 0.0f;
            for (int64_t c = 0; c < n_cols; ++c) s += row[c];
            float mean = s / (float)n_cols;
            float coef = g / ((float)n_cols * sv);
            for (int64_t c = 0; c < n_cols; ++c) drow[c] += coef * (row[c] - mean);
            break;
        }
    }
}

// out[r, c] = softmax(x[r, :])_c   (numerically stable; one thread per row)
__global__ void softmax_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int64_t n_rows, int64_t n_cols) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const float* row = x + r * n_cols;
    float* orow = out + r * n_cols;

    float mx = row[0];
    for (int64_t c = 1; c < n_cols; ++c) mx = fmaxf(mx, row[c]);

    float s = 0.0f;
    for (int64_t c = 0; c < n_cols; ++c) {
        float e = expf(row[c] - mx);
        orow[c] = e;
        s += e;
    }
    for (int64_t c = 0; c < n_cols; ++c) orow[c] /= s;
}

// dx[r, c] += y[r,c] * (dout[r,c] - dot[r])  where dot[r] = sum_c dout*y
// (ACCUMULATES; one thread per row). y is the forward softmax output.
__global__ void softmax_backward_kernel(const float* __restrict__ y,
                                        float* __restrict__ dx,
                                        const float* __restrict__ dout,
                                        int64_t n_rows, int64_t n_cols) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const float* yrow = y + r * n_cols;
    const float* grow = dout + r * n_cols;
    float* drow = dx + r * n_cols;

    float dot = 0.0f;
    for (int64_t c = 0; c < n_cols; ++c) dot += grow[c] * yrow[c];
    for (int64_t c = 0; c < n_cols; ++c) drow[c] += yrow[c] * (grow[c] - dot);
}

} // namespace

void reduction_forward(torch::Tensor x, torch::Tensor out,
                       int64_t n_rows, int64_t n_cols, int64_t op) {
    reduction_forward_kernel<<<n_blocks(n_rows), THREADS>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols, (int)op);
}

void reduction_backward(torch::Tensor x, torch::Tensor dx, torch::Tensor dout,
                        torch::Tensor out, int64_t n_rows, int64_t n_cols,
                        int64_t op) {
    reduction_backward_kernel<<<n_blocks(n_rows), THREADS>>>(
        x.data_ptr<float>(), dx.data_ptr<float>(), dout.data_ptr<float>(),
        out.data_ptr<float>(), n_rows, n_cols, (int)op);
}

void softmax_forward(torch::Tensor x, torch::Tensor out,
                     int64_t n_rows, int64_t n_cols) {
    softmax_forward_kernel<<<n_blocks(n_rows), THREADS>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols);
}

void softmax_backward(torch::Tensor y, torch::Tensor dx, torch::Tensor dout,
                      int64_t n_rows, int64_t n_cols) {
    softmax_backward_kernel<<<n_blocks(n_rows), THREADS>>>(
        y.data_ptr<float>(), dx.data_ptr<float>(), dout.data_ptr<float>(),
        n_rows, n_cols);
}
