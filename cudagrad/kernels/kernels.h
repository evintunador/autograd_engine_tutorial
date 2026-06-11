// Declarations of every CUDA launcher exposed to Python through the
// `cudagrad_ext` extension. Each kernel group (elementwise, matmul, vectorwise,
// modules, ...) defines its launchers in its own .cu file and declares them
// here; bindings.cpp binds them all into one pybind module.
//
// Convention: launchers take already-allocated, contiguous, fp32 CUDA tensors
// and write in place. Backward launchers ACCUMULATE into grad tensors (`+=` /
// atomicAdd), so callers must pass grad buffers that start at zero — mirroring
// tritongrad's accumulate-into-.grad backward kernels.
#pragma once
#include <torch/extension.h>

// ---- elementwise binary (op: 0=add, 1=sub, 2=mul, 3=div) ------------------
// y is broadcast up to x via modulo indexing (i % loop_stride), matching the
// `loop_stride` broadcasting contract in tritongrad's binary kernels.
void binary_forward(torch::Tensor x, torch::Tensor y, torch::Tensor out,
                    int64_t loop_stride, int64_t op);
void binary_backward_dx(torch::Tensor y, torch::Tensor dx, torch::Tensor dout,
                        int64_t loop_stride, int64_t op);
void binary_backward_dy(torch::Tensor x, torch::Tensor y, torch::Tensor dy,
                        torch::Tensor dout, int64_t loop_stride, int64_t op);

// ---- elementwise unary (op: 0=exp, 1=log, 2=relu, 3=neg) ------------------
// backward ACCUMULATES into dx (`+=`), so callers pass a zero-initialized dx.
void unary_forward(torch::Tensor x, torch::Tensor out, int64_t op);
void unary_backward(torch::Tensor x, torch::Tensor dx, torch::Tensor out,
                    torch::Tensor dout, int64_t op);

// ---- matmul (fwd / bwd_dA / bwd_dB) ---------------------------------------
// All tensors contiguous, row-major, fp32. A is (..., M, K); B is either
// BATCHED (..., K, N) with A's leading dims, or SHARED 2-D (K, N) broadcast
// across A's batch. The launchers read all shapes from the tensors (no shape
// ints) and detect the B layout from b.dim() vs a.dim(). Backward launchers
// ACCUMULATE into zero-initialized grads (`+=`); for shared (2-D) B the dB
// kernel sums over the batch dim. One thread per output element, no atomics.
void matmul_forward(torch::Tensor a, torch::Tensor b, torch::Tensor out);
void matmul_backward_dA(torch::Tensor b, torch::Tensor dA, torch::Tensor dC);
void matmul_backward_dB(torch::Tensor a, torch::Tensor dB, torch::Tensor dC);

// ---- vectorwise: last-dim reductions + softmax ----------------------------
// Tensors are viewed as (n_rows, n_cols) and reduced/softmaxed along the last
// dim. reduction op codes: 0=sum, 1=mean, 2=max, 3=min, 4=var, 5=std (var/std
// use population /n normalization). Backward launchers ACCUMULATE into dx (`+=`),
// so callers pass a zero-initialized dx. reduction_backward also takes `out` (the
// forward result) for ops that need it (std). softmax_backward takes `y` (the
// forward softmax output).
void reduction_forward(torch::Tensor x, torch::Tensor out,
                       int64_t n_rows, int64_t n_cols, int64_t op);
void reduction_backward(torch::Tensor x, torch::Tensor dx, torch::Tensor dout,
                        torch::Tensor out, int64_t n_rows, int64_t n_cols,
                        int64_t op);
void softmax_forward(torch::Tensor x, torch::Tensor out,
                     int64_t n_rows, int64_t n_cols);
void softmax_backward(torch::Tensor y, torch::Tensor dx, torch::Tensor dout,
                      int64_t n_rows, int64_t n_cols);

// ---- modules: embedding + layernorm ---------------------------------------
// Embedding: tokens is contiguous int64 viewed flat as (B*N) ids; weight/out
// are (V, D)/(B*N, D) fp32. forward writes out[row,:] = weight[tokens[row],:];
// backward SCATTER-ADDs dout into dweight[tokens[row],:] via atomicAdd (rows can
// share a token id), so dweight must start zeroed.
void embedding_forward(torch::Tensor tokens, torch::Tensor weight,
                       torch::Tensor out, int64_t N, int64_t D, int64_t V);
void embedding_backward(torch::Tensor tokens, torch::Tensor dweight,
                        torch::Tensor dout, int64_t N, int64_t D, int64_t V);

// LayerNorm: x/out are (rows, D) fp32; w/b are (D,). forward computes mean[r]/
// rstd[r] (population /D var) and stores them for backward. backward ACCUMULATES
// into dx (`+=`, per-row, no race) and into dw/db via atomicAdd (summed across
// rows); all grad buffers must start zeroed. eps comes in as double (cast to
// float). `b` is taken for call-site symmetry but unused by the backward math.
void layernorm_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                       torch::Tensor out, torch::Tensor mean, torch::Tensor rstd,
                       int64_t rows, int64_t D, double eps);
void layernorm_backward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                        torch::Tensor dx, torch::Tensor dout, torch::Tensor dw,
                        torch::Tensor db, torch::Tensor mean, torch::Tensor rstd,
                        int64_t rows, int64_t D);

// ---- flash attention: causal attention fwd + bwd --------------------------
// Q/K/V/O/dO/dQ/dK/dV are (B,H,N,D) contiguous fp32; LSE is (B,H,N) scratch.
// CAUSAL: query i attends only to keys j<=i. score s_ij = scale*(Q_i.K_j) with
// `scale` the multiplier passed in (used verbatim). forward fills O and stores
// the per-row logsumexp LSE for reuse in backward. backward ACCUMULATES into
// dQ/dK/dV (`+=`), so callers pass zeroed grads; each thread owns a distinct
// output row, so NO atomics. The Delta scratch is allocated inside the backward
// launcher. scale arrives as double (cast to float inside the kernels).
void flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                             torch::Tensor O, torch::Tensor LSE, double scale,
                             int64_t B, int64_t H, int64_t N, int64_t D);
void flash_attention_backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              torch::Tensor O, torch::Tensor dO,
                              torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV,
                              torch::Tensor LSE, double scale,
                              int64_t B, int64_t H, int64_t N, int64_t D);
