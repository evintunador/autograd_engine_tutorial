# this "fused softmax" operation will be significantly faster than pytorch's native op
# for a particular class of matrices: those whose rows can fit in the GPU's SRAM

import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
print(DEVICE)

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"
def is_cdna():
    return (is_hip() and 
            triton.runtime.driver.active.get_current_target.arch in ('gfx940', 'gfx941',
                                                                    'gfx942', 'gfx90a',
                                                                    'gfx908'))

# first we'll look at how pytorch does it
def naive_softmax(x):
    '''
    Built for input of sizee (M,N)
    we subtract the maximum element in order to avoid numerical overflows when doing .exp()
    softmax is invariant to this shift
    '''
    # read MN elements; write M elements
    x_max = x.max(dim=1)[0] #[0] grabs the values as opposed to the indicees
    # read MN + M lements; write MN elements
    z = x - x_max[:, None]
    # read MN elements; write MN elemnts
    numerator = torch.exp(z)
    # read MN elements; write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements; wrote 3MN + 2M elements
    return ret

# we'd prefer to have a custom "fused" kernel that only reads x once and does all the necessary
# computations on-chip as opposed to repeatedly reading & writing to DRAM
# that would give a ~4x speedup since 
# (5MN + 2M + 3MN + 2M)/(MN + MN) = (8MN + 4M)/2MN = 4 (ignoring the M term)
# torch.jit.script flag actually aims to do this fusion automaticall but can't pull it off as well

# our fused softmax kernel works as follows:
# each program (individual call of the kernel) loads a set of rows of the input matrix X
#   strideed by number of programs, normalizes it and writes back the result to the output Y

# note an important limitation of Triton is that each block must have a power-of-two number of
#   elements, so we need to internally "pad" each row and guart the memory operations properly

@triton.jit
def softmax_kernel(input_ptr, output_ptr, 
                    input_row_stride, output_row_stride,
                    n_rows, n_cols,
                    BLOCK_SIZE: tl.constexpr,
                    num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # the stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # the block size is the next power of two greater than n_cols, 
        #   so we can fit each row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # note that exponentiation in Triton is fast but approximate
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

# and now we'll create a helper function that enqueues the kernel and its meta-arguments
#   for any given input tensor
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape

    # the block size of each loop iteration is the smallest power of 2 greater than the
    #   number of columns in x
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # another trick we can use is to ask the compiler to use more threads perrow by
    #   increasing the number of warps (`num_warps`) over which each row is distributed.
    # you will see in the next tutorial how to auto-tune this value in a more natural way
    #   so you don't have to come up with manual heuristics yourself
    num_warps = 8

    # number of software pipelining stages
    num_stages = 4 if SIZ_SMEM > 200_000 else 2

    # allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel = softmax_kernel.warmup(x, y,
                                    x.stride(0), y.stride(0),
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps, # @triton.jit has arguments we didnt' define
                                    grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.max_shared_mem # .shared # i think the triton documentation is outdated
    