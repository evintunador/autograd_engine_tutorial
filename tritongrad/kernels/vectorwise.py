import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

"""
all of our reduction ops will assume that the reduction is happening along the final vector in the tensor
and that said vector fits into SRAM
this should make our kernels far less flexible but also far more efficient
"""

@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_M": BLOCK_SIZE_M}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_M in [1, 2, 4, 8, 16, 32]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["x_num_elements"], # auto-tune will re-run every time this value is different in a new input
)
@triton.jit
def reduction_op_forward(
    x_ptr,
    y_ptr,
    x_num_elements,
    y_num_elements,
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
    x = tl.load(x_ptr + x_offsets, mask=mask)
    
    # Perform reduction
    if op == "sum":
        y = tl.sum(x, axis=1)
    if op == "mean":
        y = tl.sum(x, axis=1) / row_len
    if op == "max":
        y = tl.max(x, axis=1)
    if op == "min":
        y = tl.min(x, axis=1)
    if op == "var":
        err = x - tl.sum(x, axis=1, keep_dims=True)
        y = tl.sum(err * err, axis=1) / (row_len - 1)
    if op == "std":
        err = x - tl.sum(x, axis=1, keep_dims=True)
        y = tl.sum(err * err, axis=1) / (row_len - 1)
        y = tl.sqrt(y)

    # Store result
    store_mask = row_idx < y_num_elements
    tl.store(y_ptr + row_idx, y, mask=store_mask)


@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_M": BLOCK_SIZE_M}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_M in [1, 2, 4, 8, 16, 32]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["x_num_elements"],
)
@triton.jit
def reduction_op_backward(
    x_ptr,
    dLdx_ptr,
    dLdOut_ptr,
    x_num_elements,
    dLdOut_num_elements,
    stride_row,                     # number of places to move forward in memory to get to same entry of next row
    row_len: tl.constexpr,          # row length; used in determining BLOCK_SIZE_N
    op: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,     # the number of rows to hold in a block
    BLOCK_SIZE_N: tl.constexpr,     # must be smaller than SRAM and greater than final dim length
):
    pid = tl.program_id(axis=0)
    
    # Load data
    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.arange(0, BLOCK_SIZE_N)
    x_offsets = row_idx[:, None] * stride_row + col_idx[None, :]
    x_mask = (row_idx[:, None] < (x_num_elements // row_len)) & (col_idx[None, :] < row_len)
    dLdx = tl.load(dLdx_ptr + x_offsets, mask=x_mask)
    dLdOut_mask = row_idx < dLdOut_num_elements 
    dLdOut = tl.load(dLdOut_ptr + row_idx, mask=dLdOut_mask)
    
    # Perform broadcasting up to input shape & any other gradient calcs
    if op == "sum":
        dLdx += tl.broadcast_to(dLdOut[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    if op == "mean":
        dLdx += tl.broadcast_to(dLdOut[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N)) / row_len
    if op == "var":
        # Out = Var(x) = sum((x - mean(x)) ** 2) / (n-1) 
        # Breaking down into nested functions:
        # mean = sum(x) / n
        # y = x - mean
        # z = y ** 2
        # Out = sum(z) / (n-1)
        # Chain rule: dLdx = dLdOut * dOutdz * dzdy * dydx
        # where:
        # dydx = 1 - 1/n
        # dzdy = 2y 
        # dOutdz = 1/(n-1)

        x = tl.load(x_ptr + x_offsets, mask=x_mask)
        mean = tl.sum(x, axis=1, keep_dims=True) / row_len
            # i think it makes more sense to re-calculate mean here than it would to
            # invest the memory read/writes into storing it during the fwd pass for use now
        dydx = tl.full(mean.shape, 1.0, tl.float32) - (1.0 / row_len)
        y = x - mean
        dzdy = 2.0 * y
        dOutdz = 1.0 / (row_len - 1)
        dLdOut = tl.broadcast_to(dLdOut[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        dLdx += dLdOut * dOutdz * dzdy * dydx
    if op == "std":
        # Out = Std(x) = sqrt(Var(x)) = sqrt(sum((x - mean(x)) ** 2) / (n-1))
        # Breaking down into nested functions:
        # mean = sum(x) / n
        # y = x - mean
        # z = y ** 2
        # w = sum(z) / (n-1)  [this is variance]
        # Out = sqrt(w)
        # Chain rule: dLdx = dLdOut * dOutdw * dwdz * dzdy * dydx
        # where:
        # dydx = 1 - 1/n (from d/dx(x - mean(x)))
        # dzdy = 2y = 2(x - mean)
        # dwdz = 1/(n-1)
        # dOutdw = 0.5 * (w)**(-0.5)

        x = tl.load(x_ptr + x_offsets, mask=x_mask)
        mean = tl.sum(x, axis=1, keep_dims=True) / row_len
        dydx = tl.full(mean.shape, 1.0, tl.float32) - (1.0 / row_len)
        y = x - mean
        dzdy = 2.0 * y
        dwdz = 1.0 / (row_len - 1)
        # Calculate variance (w) for dOutdw
        w = tl.sum(y * y, axis=1, keep_dims=True) / (row_len - 1)
        dOutdw = 0.5 * tl.rsqrt(w)
        dLdOut = tl.broadcast_to(dLdOut[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        dLdx += dLdOut * dOutdw * dwdz * dzdy * dydx

    # Store result
    tl.store(dLdx_ptr + x_offsets, dLdx, mask=x_mask)