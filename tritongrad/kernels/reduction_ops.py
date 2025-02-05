import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

"""
all of our reduction ops will assume that the reduction is happening along the final vector in the tensor
and that said vector fits into SRAM
this should make our kernels far less flexible but also far more efficient
"""

# this decorator figures out what meta-parameters will be most efficient with brute-force
@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_M": BLOCK_SIZE_M}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_M in [1, 2, 4, 8, 16, 32]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["x_num_elements"], # auto-tune will re-run every time this value is different in a new input
)
# this decorator uses a heuristic we choose to avoid having to brute-force autotune
#@triton.heuristics(values={'BLOCK_SIZE_N': lambda args: triton.next_power_of_2(args['row_len'])})
@triton.jit
def reduction_op_forward(
    x_ptr,
    z_ptr,
    x_num_elements,
    z_num_elements,
    stride_row,                     # number of places to move forward in memory to get to same entry of next row
    row_len: tl.constexpr,          # row length; used in determining BLOCK_SIZE_N
    op: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,     # the number of rows to hold in a block
    BLOCK_SIZE_N: tl.constexpr,     # must be smaller than SRAM and greater than final dim length
):
    pid = tl.program_id(axis=0)
    
    # Reshape the offsets to handle the reduction properly
    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.arange(0, BLOCK_SIZE_N)
    x_offsets = row_idx[:, None] * stride_row + col_idx[None, :]
    
    # Load data
    mask = (row_idx[:, None] < (x_num_elements // row_len)) & (col_idx[None, :] < row_len)
    x_block = tl.load(x_ptr + x_offsets, mask=mask)
    
    # Perform reduction
    z = tl.sum(x_block, axis=1)
    
    # Store result
    store_mask = row_idx < z_num_elements
    tl.store(z_ptr + row_idx, z, mask=store_mask)

