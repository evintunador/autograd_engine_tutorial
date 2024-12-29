'''
we'll learn about
- block-level matmuls
- multi-dimensional pointer arithmetic
- program re-ordering for improved L2 cache hit rate
- automatic performance tuning

this is the algorithm our code will roughly be implementing, but in parallel
for m in range(0, M, BLOCK_SIE_M): # do in parallel
    for n in range(0, N, BLOCK_SIZE_N): # do in parallel
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
            acc += dot(a,b)
        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
'''
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

# this is just setting up a bunch of different potential config files that we'll choose from later based
# on how well they perform on our specific GPU
def get_cuda_autotune_conig(): # why is it a function instead of just storing a list tho?
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2)
    ]

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator which consumes
#   - a list of `triton.Config` objects that define different configs of meta-parameters and compilation options
#   - an auto-tuning *key* whose change in values will trigger evaluation of all the provided configs
@triton.autotune(configs = get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, # pointers to matrices
                M, N, K, # matrix dimensions
                stride_am, stride_ak, # how much to increase the ptr by when moving by 1 element
                stride_bk, stride_bn, # ex. increase a_ptr by stride_am to get the element one row down (A has M rows)
                stride_cm, stride_cn,
                # meta-parameters
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr,
                ACTIVATION: tl.constexpr):
    '''
    kernel for computing the matmul A @ B = C
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    '''
    # first we map program ids (pids) to the block of C it should compute
    # this is done in grouped ordering to promote L2 data reuse
    # see https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # create pointers for the first blocks of A and B
    # we'll advance these as we move in the K direction and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # iterate to compute a block of the C matrix
    # we accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of fp32 values for higher accuracy
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # load the next block of A and B; generate a mask by checking the K dimension
        # if it is out of bounds, set it to 0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # we accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator)
        # advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # you can fuse arbitrary activation functions here whil ehte accumulator is still in FP32
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# we can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

# now our wrapper function is relatively simple compared to the previous two lessons
def matmul(a, b, activation=""):
    # check constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous" # <- what does contiguous mean in this context?
    M, K = a.shape
    K, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c

# unit test
torch.manual_seed(0)
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# benchmark
configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"],
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["cublas", "triton"],
        line_names = ["cuBLAS", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul-performance",
        #args = {"fp8_inputs": False}
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        # 2 = number of memory operations (1 read + 1 write)
        # M * N * K = number of elements
        # 1e-12 converts flops to Teraflops
        # ms * 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)
benchmark.run(print_data=True, save_path='./benchmark_results/')