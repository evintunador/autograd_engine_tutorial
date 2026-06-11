// Elementwise Metal kernels for mlxgrad.
//
// This is the mlxgrad analog of cudagrad/kernels/elementwise.cu, one tier over
// in Apple's stack: instead of CUDA C++ launched by nvcc, these are Metal
// Shading Language kernel *bodies* fed to `mx.fast.metal_kernel` (MLX generates
// the function signature from the input/output names — so each section below is
// just the body, referencing the named buffers directly).
//
// Each kernel body lives between a `// @kernel <name>` marker and the next
// marker; `mlx_kernels.py` slices them out by name. Lines above the first marker
// (this header) are ignored.
//
// Conventions shared with cudagrad:
//   * all data buffers are contiguous fp32 (MLX copies non-contiguous inputs to
//     row-contiguous before launch, so no contiguity asserts are needed).
//   * the grid is launched with exactly `n` threads (MLX uses non-uniform
//     dispatch), so `thread_position_in_grid.x` is always in range — no bounds
//     check needed.
//   * scalars (`ls` loop-stride, `n` element count, `op` selector) arrive as
//     1-element uint32 buffers; read element [0].
//   * `y` (second operand) has `ls` elements and is broadcast up to `x`'s `n`
//     elements via `i % ls`, matching tritongrad/cudagrad. Same-shape ops pass
//     ls == n.
//   * BACKWARD KERNELS ACCUMULATE FUNCTIONALLY: they read the running gradient
//     `grad_in[...]`, add this op's contribution, and write the sum to `out`.
//     The engine then rebinds `tensor.grad = out`. (MLX arrays are immutable, so
//     we cannot do the in-place `dx += ...` that cudagrad's CUDA kernels do —
//     instead the "+=" happens inside the kernel and a fresh array comes back.)
//   * the dy kernel parallelizes over `ls` (one thread per second-operand
//     element), each thread looping over its broadcast group `i = j, j+ls, ...`.
//     This sums broadcast contributions WITHOUT atomics.

// op selectors (must match _BINARY_OP / _UNARY_OP in mlx_kernels.py):
//   binary: 0=add 1=sub 2=mul 3=div     unary: 0=exp 1=log 2=relu 3=neg

// @kernel binary_forward
// out[i] = x[i] OP y[i % ls]
uint i = thread_position_in_grid.x;
float a = x[i];
float b = y[i % ls[0]];
float r;
switch (op[0]) {
    case 0u: r = a + b; break;  // add
    case 1u: r = a - b; break;  // sub
    case 2u: r = a * b; break;  // mul
    default: r = a / b; break;  // div
}
out[i] = r;

// @kernel binary_backward_dx
// out[i] = grad_in[i] + d(out)/d(x[i]) * dout[i]
//   add/sub: +dout ;  mul: +dout*y ;  div: +dout/y
uint i = thread_position_in_grid.x;
float g = dout[i];
float b = y[i % ls[0]];
float contrib;
switch (op[0]) {
    case 0u: contrib = g; break;       // add: d/dx = 1
    case 1u: contrib = g; break;       // sub: d/dx = 1
    case 2u: contrib = g * b; break;   // mul: d/dx = y
    default: contrib = g / b; break;   // div: d/dx = 1/y
}
out[i] = grad_in[i] + contrib;

// @kernel binary_backward_dy
// one thread per second-operand element j; sum over its broadcast group.
// out[j] = grad_in[j] + sum_{i: i%ls==j} d(out)/d(y[j]) * dout[i]
//   add: +dout ; sub: -dout ; mul: +dout*x ; div: -x*dout/y^2
uint j = thread_position_in_grid.x;
uint LS = ls[0];
uint N = n[0];
float yv = y[j];
float acc = grad_in[j];
for (uint i = j; i < N; i += LS) {
    float g = dout[i];
    float contrib;
    switch (op[0]) {
        case 0u: contrib = g; break;                  // add: d/dy = 1
        case 1u: contrib = -g; break;                 // sub: d/dy = -1
        case 2u: contrib = g * x[i]; break;           // mul: d/dy = x
        default: contrib = -x[i] * g / (yv * yv); break;  // div: d/dy = -x/y^2
    }
    acc += contrib;
}
out[j] = acc;

// @kernel unary_forward
// out[i] = UNARY_OP(x[i])
uint i = thread_position_in_grid.x;
float a = x[i];
float r;
switch (op[0]) {
    case 0u: r = metal::exp(a); break;          // exp
    case 1u: r = metal::log(a); break;          // log
    case 2u: r = metal::max(a, 0.0f); break;    // relu
    default: r = -a; break;                     // neg
}
out[i] = r;

// @kernel unary_backward
// out[i] = grad_in[i] + d(out)/d(x[i]) * dout[i]
//   exp: out_fwd*dout ; log: dout/x ; relu: (out_fwd>0)?dout:0 ; neg: -dout
uint i = thread_position_in_grid.x;
float g = dout[i];
float contrib;
switch (op[0]) {
    case 0u: contrib = out_fwd[i] * g; break;                  // exp
    case 1u: contrib = g / x[i]; break;                        // log
    case 2u: contrib = (out_fwd[i] > 0.0f) ? g : 0.0f; break;  // relu
    default: contrib = -g; break;                              // neg
}
out[i] = grad_in[i] + contrib;
