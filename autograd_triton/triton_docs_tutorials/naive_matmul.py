import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _naive_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # meta-parameters
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    program_id_m, program_id_n = tl.program_id(0), tl.program_id(1)
    # chunks along m/n/k dimensions
    rm = program_id_m * block_size_m + tl.arange(0, block_size_m)
    rn = program_id_n * block_size_n + tl.arange(0, block_size_n)
    rk = tl.arange(0, block_size_k)
    # relevant offsets of a, b
    offsets_a = a_ptr + rm.expand_dims(1) * stride_am + rk.expand_dims(0) * stride_ak
    offsets_b = b_ptr + rk.expand_dims(1) * stride_bk + rn.expand_dims(0) * stride_bn

    # initialize and iteratively update accumulator
    accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    mask = (rm.expand_dims(1) < m) & (rn.expand_dims(0) < n)
    for _ in range(0, k, block_size_k):
        # todo umer: don't we need mask when loading a & b?
        a = tl.load(offsets_a, mask=mask)
        b = tl.load(offsets_b, mask=mask)
        accumulator += tl.dot(a, b)#, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offsets_a += block_size_k * stride_ak
        offsets_b += block_size_k * stride_bk

    # you can fuse arbitrary activation functions here while the accumulator is still in FP32
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    accumulator = accumulator.to(tl.float16)

    # find the desired location for this chunk of c and store it there
    c_chunk_ptr = c_ptr + rm.expand_dims(1) * stride_cm + rn.expand_dims(0) * stride_cn
    tl.store(c_chunk_ptr, accumulator, mask=mask)

# we can fuse a nonlinearity (here `leaky_relu`) by providing it as an `ACTIVATION` 
#  meta-parameter in `_naive_matmul_kernel`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def naive_matmul(a, b, block_size=64, activation=""): 
    # you may need to lower block_size if you get an SRAM error. in `matmul.py` we learn how to autotune it
    
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    assert a.is_contiguous(), "matrix A must be contiguous" # Returns True if tensor is contiguous in memory
        # i think this means that all elements are lined up back-to-back without interruption
        # needs to be true so that our indexing makes sense

    # get dimesion lengths
    (m, k), (_, n) = a.shape, b.shape

    # allocates output
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    
    # 2D launch kernel, meaning we parallelize across both rows and columns (called "row-major ordering")
    grid = lambda meta: (triton.cdiv(m, meta['block_size_m']),  triton.cdiv(n, meta['block_size_n']))
    _naive_matmul_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1), # the jump necessary to go from one element to the next one in that dimension
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        block_size_m=block_size, block_size_n=block_size, block_size_k=block_size,
        ACTIVATION=activation # not used by default since "" is being passed in
    )
    return c

# unit test
torch.manual_seed(0)
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = naive_matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
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
        plot_name = "naive_matmul-performance",
        args={}, # values for funciton arguments not in x_names and y_names; need it even if not using
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        # 2 = number of memory operations (1 read + 1 write)
        # M * N * K = number of elements
        # 1e-12 converts flops to Teraflops
        # ms * 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)
benchmark.run(print_data=True, save_path='./benchmark_results/')