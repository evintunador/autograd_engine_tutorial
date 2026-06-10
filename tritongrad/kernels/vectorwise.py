import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # each SM has a fixed amount of SRAM that it can access
    # if one SM isn't using all its available SRAM then another can be spun up to use the remainder

"""
all of our vector-wise ops will assume that the calculation is happening along the final 
vector in the tensor and that said vector fits into SRAM
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
        err = x - tl.sum(x, axis=1, keep_dims=True) / row_len  # subtract the mean, not the sum
        y = tl.sum(err * err, axis=1) / row_len                # population variance
    if op == "std":
        err = x - tl.sum(x, axis=1, keep_dims=True) / row_len  # subtract the mean, not the sum
        y = tl.sum(err * err, axis=1) / row_len                # population variance
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
        w = tl.sum(y * y, axis=1, keep_dims=True) / row_len  # population variance
        dOutdw = 0.5 * tl.rsqrt(w)
        dLdOut = tl.broadcast_to(dLdOut[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        dLdx += dLdOut * dOutdw * dwdz * dzdy * dydx

    # Store result
    tl.store(dLdx_ptr + x_offsets, dLdx, mask=x_mask)


########################################################################################
########################### Softmax ####################################################
########################################################################################
# softmax is vector-wise like the reductions above (it operates along the final dim and
# assumes the whole vector fits in SRAM) but unlike a reduction its output is the SAME
# shape as the input rather than collapsing the final dim.

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
def softmax_forward(
    x_ptr,
    y_ptr,
    x_num_elements,
    stride_row,                     # places to move in memory to get to the same entry of the next row
    row_len: tl.constexpr,          # length of the dim we softmax over; used to set BLOCK_SIZE_N
    BLOCK_SIZE_M: tl.constexpr,     # the number of rows to hold in a block
    BLOCK_SIZE_N: tl.constexpr,     # next power of 2 >= row_len, must fit in SRAM
):
    pid = tl.program_id(axis=0)

    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.arange(0, BLOCK_SIZE_N)
    offsets = row_idx[:, None] * stride_row + col_idx[None, :]
    mask = (row_idx[:, None] < (x_num_elements // row_len)) & (col_idx[None, :] < row_len)

    # masked-out columns load a large negative so they (a) never win the max and
    # (b) exponentiate to ~0 and thus contribute nothing to the denominator
    x = tl.load(x_ptr + offsets, mask=mask, other=-1e9)

    # numerically-stable softmax: subtract the row max before exponentiating
    x_max = tl.max(x, axis=1)[:, None]                 # (BLOCK_SIZE_M, 1)
    numerator = tl.exp(x - x_max)                       # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    denominator = tl.sum(numerator, axis=1)[:, None]    # (BLOCK_SIZE_M, 1)
    y = numerator / denominator

    tl.store(y_ptr + offsets, y, mask=mask)


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
def softmax_backward(
    y_ptr,                          # the forward-pass softmax output (the probabilities)
    dLdx_ptr,                       # input gradient we accumulate into
    dLdy_ptr,                       # incoming upstream gradient
    x_num_elements,
    stride_row,
    row_len: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.arange(0, BLOCK_SIZE_N)
    offsets = row_idx[:, None] * stride_row + col_idx[None, :]
    mask = (row_idx[:, None] < (x_num_elements // row_len)) & (col_idx[None, :] < row_len)

    y = tl.load(y_ptr + offsets, mask=mask, other=0.)          # softmax probs
    dLdy = tl.load(dLdy_ptr + offsets, mask=mask, other=0.)    # upstream grad

    # the jacobian-vector product for softmax simplifies to
    #   dLdx_i = y_i * (dLdy_i - sum_j(y_j * dLdy_j))
    # the masked columns contribute 0 to the dot product since y=0 there
    dot = tl.sum(y * dLdy, axis=1)[:, None]                    # (BLOCK_SIZE_M, 1)
    dLdx_new = y * (dLdy - dot)

    # accumulate (like the reduction backward) so the autotuner warmup dance applies
    dLdx = tl.load(dLdx_ptr + offsets, mask=mask, other=0.)
    tl.store(dLdx_ptr + offsets, dLdx + dLdx_new, mask=mask)


########################################################################################
########################### Argmax #####################################################
########################################################################################
# forward-only (no gradient flows through an argmax); used for greedy inference.

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
def argmax_forward(
    x_ptr,
    y_ptr,                          # int32 output of shape (n_rows,)
    x_num_elements,
    y_num_elements,
    stride_row,
    row_len: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.arange(0, BLOCK_SIZE_N)
    offsets = row_idx[:, None] * stride_row + col_idx[None, :]
    mask = (row_idx[:, None] < (x_num_elements // row_len)) & (col_idx[None, :] < row_len)

    # masked-out columns load a large negative so they can never be the argmax
    x = tl.load(x_ptr + offsets, mask=mask, other=-1e9)
    idx = tl.argmax(x, axis=1).to(tl.int32)               # (BLOCK_SIZE_M,)

    store_mask = row_idx < y_num_elements
    tl.store(y_ptr + row_idx, idx, mask=store_mask)

