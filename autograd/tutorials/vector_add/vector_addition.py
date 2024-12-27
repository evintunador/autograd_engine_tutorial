import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
print(DEVICE)


@triton.jit
def add_kernel(x_ptr, # pointer to first input vector
               y_ptr, # ptr to second
               output_ptr, # ptr to output vector
               n_elements, # size of vector
               BLOCK_SIZE: tl.constexpr, # number of elements each program should process
               ):
    # there are multiple "programs" processing data
    # here we identify which program we are:
    pid = tl.program_id(axis=0) # 1d launch grid for the vector so axis is 0
    # this program will process inputs that are offset from the initial data
    # for instance, for a vector of length 256 and block_size 64, the programs
    #   would each access the elements [0:64, 64:128, 128:192, 192:256]
    # note that offsets is a list of pointers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    # load x and y from DRAM, masking out any extra elements in case the input
    #   is not a multiple of block size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    '''
    helper/wrapper function to 
        1) allocate the output tensor and 
        2) enque the above kernel with appropriate grid/block sizes
    '''
    # preallocating the output
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE,\
        f'DEVICE: {DEVICE}, x.device: {x.device}, y.device: {y.device}, output.device: {output.device}'
    n_elements = output.numel()
    # the SPMD (what does that acronym stand for??) denotes the number of kernel
    #   instances that run in parallel
    # it can be either Tuple[int] or Callable(metaparameters) -> Tuple[int]
    # in this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        # so 'BLOCK_SIZE' is a parameter to be passed into meta()
        # and i guess triton.cdiv just figures out how many chunks and provides them like a range
        # then meta() returns a Tuple
    # NOTE:
    #   - each torch.tensor object is implicitly converted into a pointer to its first element
    #   - `triton.jit`'ed functionis can be indexed with a launch grid to obtain a callable GPU kernel
    #   - don't forget to pass meta-paramters as keyword arguments (<-???)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # we return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, 
    #   the kernel is still running asynchronously at this point
    return output

# we can now use the above function to comput ethe element-wise sum of two torch.tensor objects
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The max diff bw torch & triton is: '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# BENCHMARK
# triton has a set of built-in utilities that make it easy for us to plot performance of custom ops
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different possible values for x_name
        x_log = True, # makes x-axis logarithmic
        line_arg='provider', # title of the legend 
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s', # label name for y-axis
        plot_name='vector-add-performance', # also used as file name for saving plot
        args={}, # values for funciton arguments not in x_names and y_names
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
benchmark.run(print_data=True, show_plots=True, save_path='.')