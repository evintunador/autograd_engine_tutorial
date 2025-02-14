import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


########################################################################################
########################### Embedding ############################################
########################################################################################

@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS, "BLOCK_SIZE_COLS": BLOCK_SIZE_COLS}, 
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_ROWS in [32, 64, 128]
        for BLOCK_SIZE_COLS in [32, 64, 128]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["N", "D"],
)
@triton.jit
def embedding_forward(
    ids_ptr,    # ponter to tensor shape (B, N) that we can treat like (B * N)
    e_ptr,      # ponter to tensor shape (V, D)
    x_ptr,      # ponter to tensor shape (B, N, D) that we can treat like (B * N, D)
    ids_B_stride, ids_N_stride,
    e_V_stride, e_D_stride,
    x_B_stride, x_N_stride, x_D_stride,
    N, D, V,
    ids_num_elements, e_num_elements, x_num_elements,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_offsets = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = pid_col * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)

    ids_offsets = row_offsets * ids_N_stride
    ids_mask = row_offsets < ids_num_elements
    ids = tl.load(ids_ptr + ids_offsets, mask=ids_mask).to(tl.int32)
        # i think pytorch uses int64 for embeddings but since we'll only ever have
        #  a relatively small vocab size this can save us some memory. 
        # also FYI Triton doesn't support int8 nor int16

    e_offsets = ids[:, None] * e_V_stride + col_offsets[None, :] * e_D_stride
    e_mask = (ids[:, None] < V) & (col_offsets[None, :] < D)
    e_block = tl.load(e_ptr + e_offsets, mask=e_mask)

    x_offsets = row_offsets[:, None] * x_N_stride + col_offsets[None, :] * x_D_stride
    x_mask = (row_offsets[:, None] < (x_num_elements // D)) & (col_offsets[None, :] < D)
    tl.store(x_ptr + x_offsets, e_block, mask=x_mask)

@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS, "BLOCK_SIZE_COLS": BLOCK_SIZE_COLS}, 
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_ROWS in [32, 64, 128]
        for BLOCK_SIZE_COLS in [32, 64, 128]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["N", "D"],
)
@triton.jit
def embedding_backward(
    ids_ptr,        # ponter to tensor shape (B, N) that we can treat like (B * N)
    dLde_ptr,          # ponter to tensor shape (V, D)
    dLdx_ptr,       # ponter to tensor shape (B, N, D) that we can treat like (B * N, D)
    ids_B_stride, ids_N_stride,
    dLde_V_stride, dLde_D_stride,
    dLdx_B_stride, dLdx_N_stride, dLdx_D_stride,
    N, D, V,
    ids_num_elements, dLde_num_elements, dLdx_num_elements,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_offsets = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = pid_col * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)

    ids_offsets = row_offsets * ids_N_stride
    ids_mask = row_offsets < ids_num_elements
    ids = tl.load(ids_ptr + ids_offsets, mask=ids_mask).to(tl.int32)

    dLdx_offsets = row_offsets[:, None] * dLdx_N_stride + col_offsets[None, :] * dLdx_D_stride
    dLdx_mask = (row_offsets[:, None] < (dLdx_num_elements // D)) & (col_offsets[None, :] < D)
    dLdx_block = tl.load(dLdx_ptr + dLdx_offsets, mask=dLdx_mask)

    dLde_offsets = ids[:, None] * dLde_V_stride + col_offsets[None, :] * dLde_D_stride
    dLde_mask = (ids[:, None] < V) & (col_offsets[None, :] < D)
    tl.atomic_add(dLde_ptr + dLde_offsets, dLdx_block, mask=dLde_mask)



########################################################################################
########################### LayerNorm ############################################
########################################################################################

@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS}, 
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_ROWS in [1, 2, 4, 8, 16, 32]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["BLOCK_SIZE_COLS"],
)
@triton.jit
def layernorm_forward(
    x_ptr, w_ptr, b_ptr, y_ptr,
    x_N_stride, x_D_stride,
    w_D_stride, b_D_stride,
    y_N_stride, y_D_stride,
    rows, D, 
    eps,
    mean_ptr, rstd_ptr,
    BLOCK_SIZE_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = tl.arange(0, BLOCK_SIZE_COLS)

    x_offsets = row_offsets[:, None] * x_N_stride + col_offsets[None, :] * x_D_stride
    x_mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < D)
    x = tl.load(x_ptr + x_offsets, mask=x_mask)
    
    mean = tl.sum(x, axis=1) / D
    err = tl.where(col_offsets[None, :] < D, x - mean[:, None], 0.0)
    var = tl.sum(err * err, axis=1) / D 
        # LayerNorm uses biased variance (as opposed to D - 1) since we're not sampling from a population
    rstd = 1 / tl.sqrt(var + eps)
    x_normalized = err * rstd[:, None]

    # saving mean and rstd for use later in the backward pass
    tl.store(mean_ptr + row_offsets, mean) # write BLOCK_SIZE_ROWS entries to memory
    tl.store(rstd_ptr + row_offsets, rstd)

    tl.static_assert(w_D_stride == b_D_stride)
    wb_offsets = col_offsets * w_D_stride
    wb_mask = col_offsets < D
    w = tl.load(w_ptr + wb_offsets, mask=wb_mask)
    b = tl.load(b_ptr + wb_offsets, mask=wb_mask)
    x_shifted = x_normalized * w + b 

    # i'd like to assert that y_N_stride==x_N_stride and same with D here but you can
    #  only use tl.static_assert() on tl.constexpr arguments, which these are not
    y_offsets = row_offsets[:, None] * y_N_stride + col_offsets[None, :] * y_D_stride
    tl.store(y_ptr + y_offsets, x_shifted, mask=x_mask)


@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS}, 
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_ROWS in [1, 2, 4, 8, 16, 32]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["BLOCK_SIZE_COLS"],
)
@triton.jit
def layernorm_backward(
    x_ptr, w_ptr, b_ptr, 
    dLdx_ptr, dLdy_ptr,
    dLdw_ptr, dLdb_ptr, 
    mean_ptr, rstd_ptr,
    x_N_stride, x_D_stride,
    w_D_stride,
    b_D_stride,
    dLdx_N_stride, dLdx_D_stride,
    dLdy_N_stride, dLdy_D_stride,
    dLdw_D_stride,
    dLdb_D_stride,
    mean_N_stride,
    rstd_N_stride,
    N, D,
    BLOCK_SIZE_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    rows_mask = row_offsets < N
    col_offsets = tl.arange(0, BLOCK_SIZE_COLS)
    cols_mask = col_offsets < D

    # Load data to SRAM
    x_offsets = row_offsets[:, None] * x_N_stride + col_offsets[None, :] * x_D_stride
    x_mask = rows_mask[:, None] & cols_mask[None, :]
    x = tl.load(x_ptr + x_offsets, mask = x_mask) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COLS)
    dLdy_offsets = row_offsets[:, None] * dLdy_N_stride + col_offsets[None, :] * dLdy_D_stride
    dLdy_mask = rows_mask[:, None] & cols_mask[None, :]
    dLdy = tl.load(dLdy_ptr + dLdy_offsets, mask = dLdy_mask) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COLS)
    w = tl.load(w_ptr + col_offsets * w_D_stride, mask = cols_mask) # shape (BLOCK_SIZE_COLS)
    mean = tl.load(mean_ptr + row_offsets[:, None] * mean_N_stride, mask = rows_mask[:, None])
    rstd = tl.load(rstd_ptr + row_offsets[:, None] * rstd_N_stride, mask = rows_mask[:, None])
        # shape (BLOCK_SIZE_ROWS, 1)

    """
    LayerNorm is
        y = xhat * w + b
    where
        xhat = (x - mean) * rstd        <- aka normalized x
        mean = sum(x) / D
        rstd = 1 / sqrt(var + eps)
        var = sum((x - mean) ** 2) / (D - 1)

    So to get the derivative dLdx given the upstream gradient dLdy, 
    we first do
        dLdxhat = dLdy * dydxhat = dLdy * w
    then use a rearrangement of the raw chain-rule form to get
        dLdx = rstd * (dLdxhat - rowMean(dLdxhat) - xhat * rowMean(dLdxhat * xhat))
    """
    xhat = tl.where(cols_mask, (x - mean) * rstd, 0.) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COLS)
    dLdxhat = tl.where(cols_mask, dLdy * w, 0.) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COLS)
    # c1 and c2 are just intermediary labels; no real meaning
    c1 = tl.sum(xhat * dLdxhat, axis=1) / D # shape (BLOCK_SIZE_ROW)
    c2 = tl.sum(dLdxhat, axis=1) / D # shape (BLOCK_SIZE_ROW)
    dLdx = (dLdxhat - (xhat * c1[:, None] + c2[:, None])) * rstd # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COLS)

    # assuming x offsets & mask will work for dLdx
    tl.store(dLdx_ptr + x_offsets, dLdx, mask = x_mask)

    # accumulate partial sums for dLdw and dLdb
    dLdw_portion = tl.sum(dLdy * xhat, axis=0) # shape (BLOCK_SIZE_COLS)
    dLdb_portion = tl.sum(dLdy, axis=0) # shape (BLOCK_SIZE_COLS)

    tl.atomic_add(dLdw_ptr + col_offsets * dLdw_D_stride, dLdw_portion, mask = cols_mask)
    tl.atomic_add(dLdb_ptr + col_offsets * dLdb_D_stride, dLdb_portion, mask = cols_mask)




########################################################################################
########################### Flash-attention ############################################
########################################################################################

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_T_block_ptrs,
    V_block_ptrs,
    block_index_q,
    softmax_scale,
    #stride_K_seq, # if you go for a manual pointer implementation then you'll need to pass in these strides.
    #stride_V_seq, # in the automatic implementation relevant stride lengths are saved in the pointer object
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_q: tl.constexpr,
    offsets_kv: tl.constexpr,
    N: tl.constexpr,
):
    # range of values handled by this stage
    if CAUSAL and DIAGONAL:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo = block_index_q * BLOCK_SIZE_Q
        hi = (block_index_q + 1) * BLOCK_SIZE_Q
        # let the compiler know lo is a muliple of BLOCK_SIZE_Q to speed things up
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q) # TODO not sure why this doesn't also help with hi
    elif CAUSAL: # any blocks in the causal mask below the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    else: # runs on every single block for the case that we're not using a causal mask
        lo, hi = 0, N

    K_T_block_ptrs = tl.advance(K_T_block_ptrs, (0, lo)) # tuple because you choose which dimension to advance
    V_block_ptrs = tl.advance(V_block_ptrs, (lo, 0))
    """
    Here are the above ^ two lines implemented with manual pointers.
    Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
    K_block_ptrs += lo * stride_K_seq
    V_block_ptrs += lo * stride_V_seq
    """

    # loop over blocks along the sequence length dimension of k & v and update accumulator while doing so
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
            # when in doubt, use tl.multiple_of() for any dynamic variable (as opposed to static variables)

        # compute (Q @ K^T) / sqrt{head_dim}
        K_T_block = tl.load(K_T_block_ptrs)
        QK_block = tl.dot(Q_block, K_T_block) * softmax_scale # becomes shape (BLOCK_SIZE_Q, BLOCK_SIZE_KV)

        if CAUSAL and DIAGONAL: # if causal mask and we're currently on a block containing the diagonal
            mask = offsets_q[:, None] >= (start_kv + offsets_kv[None, :])
            QK_block += tl.where(mask, 0, -1.0e6)
        
        # find the max values of the new block and compare them to those of all previous blocks to get an update
        m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1)) # shape is (BLOCK_SIZE_Q)
        # adjust QK block for safe softmax
        QK_block -= m_ij[:, None]

        # Compute the exponential of each safe dot product, which is the numerator of our softmax
        P_block = tl.exp2(QK_block) 
            # we're using bases 2 instead of base e because it's faster and softmax is invariant to the change
        
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, axis=1) # shape (BLOCK_SIZE_Q)
        # This is the correction factor for the previous l_i
        alpha = tl.exp2(m_i - m_ij) # shape (BLOCK_SIZE_Q)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij
        
        # This computes O_new = P @ V + O_old * alpha
        V_block = tl.load(V_block_ptrs) # shape (BLOCK_SIZE_KV, HEAD_DIM)
        P_block = P_block#.to(tl.float16) # since V is fp16. we only wanted to accumulate attention logits in fp32
        O_block = O_block * alpha[:, None] # adjusts previous values based on potential new max
        # accumulated P and V block dot product into O block
        O_block = tl.dot(P_block, V_block, acc=O_block) # shape (BLOCK_SIZE_Q, HEAD_DIM)
            # notice we're doing this V projection before we've actually divided by our softmax denominator l_i
            # which is possible because in this context the two operations are associative

        m_i = m_ij # sets old max equal to new max, ready to be used by next iteration of for loop

        # Move to the next block of K and V along the N dimension
        V_block_ptrs = tl.advance(V_block_ptrs, (BLOCK_SIZE_KV, 0))
        K_T_block_ptrs = tl.advance(K_T_block_ptrs, (0, BLOCK_SIZE_KV))
        """
        Here are the above ^ two lines implemented with manual pointers.
        Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
        V_block_ptrs += BLOCK_SIZE_KV * stride_V_seq
        K_block_ptrs += BLOCK_SIZE_KV * stride_K_seq
        """

    return O_block, l_i, m_i # we save these three specifically for use later in the backward pass


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [32]
        for BLOCK_SIZE_KV in [32]
        for num_stages in ([1])
        for num_warps in [2, 4]
    ],
    key=["N", "HEAD_DIM"], # auto-tune will re-run every time either of these values changes in a new input
)
@triton.jit
def _attn_fwd(
    Q_ptr, K_ptr,  V_ptr,  # each shape (BATCH_SIZE, NUM_HEADS, N, HEAD_DIM)
    softmax_scale,
    M_ptr,  # shape (BATCH_SIZE, NUM_HEADS, N). here we first store the max values of each row & later the logsumexp trick 
    O_ptr,  # shape (BATCH_SIZE, NUM_HEADS, N, HEAD_DIM). where we store the final output
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim, # dist to move thru mem to next entry in that dim of that tensor
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    BATCH_SIZE, # unlike other tensor dimensions, batch size needs to be flexible for runtime differences
    # meta-parameters (decided at compile-time)
    NUM_HEADS: tl.constexpr, N: tl.constexpr, 
    HEAD_DIM: tl.constexpr, # should always be a power of 2, and really 128 and 256 are the only reasonable options
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    this implementation of the flash-attention forward pass was created entirely from the psuedocode in the two papers
    https://arxiv.org/abs/2205.14135
    https://arxiv.org/abs/2307.08691
    """
    # as opposed to regular assert, static_assert occurs at compile-time
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
        # head_dim is usually relatively small (128 or 256) so it wouldn't make sense to parallelize within it

    # This indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicates which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicates the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # This allows to get the shape (N, HEAD_DIM) block in the Q, K, V by indexing it by batch and head
    qkv_offset = index_batch * stride_Q_batch + index_head * stride_Q_head

    # so here's a new function that does the math of finding the right pointer for us
    Q_block_ptrs = tl.make_block_ptr(
        base=Q_ptr + qkv_offset, # base pointer to the parent tensor
            # notice our parent tensor is actually a single splice of the original Q rather than the full Q
        shape=(N, HEAD_DIM), # shape of the parent tensor
        strides=(stride_Q_seq, stride_Q_dim), # strides of the parent tensor
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # offsets to get to the block of interest for this PID
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), # shape of the block
        order=(1, 0), # we'll explain what `order1 is for at K_block_ptrs below
    )
    """
    # Here is the above ^ function implemented manually.

    # Our base pointer is actually going to be a specific batch and head, meaing we're working with a (N,HEAD_DIM) matrix.
    Q_ptr += qkv_offset 

    # Offsets for N are split by pids but for head_dim we keep the whole thing in SRAM.
    offsets_Q_N = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offsets_Q_head_dim = tl.arange(0, HEAD_DIM)
    
    # putting it all together we 
        # 1. start at the first entry of the (N,HEAD_DIM matrix),
        # 2. turn the N and head_dim components into 2D tensors to match
        # 3. adjust for stride length in memory
    Q_block_ptrs = Q_ptr + (offsets_Q_N[:, None] * stride_Q_seq + offsets_Q_head_dim[None, :] * stride_Q_dim)
    
    HERE'S THE THING:
    when writing a kernel, you have to choose between whether you're going to use make_block_ptr or do it manually.
    once you choose you can't mix the two because make_block_ptr actually returns a weird triton object that is
    incompatible with your manually created pointers. throughout the rest of this kernel (and the sub-kernel it calls)
    i am going to be adding in the manual version of each relevant line of code as a comment and reminding you to not mix them
    """
    
    # we transpose K while loading it (as opposed to writing a whole separate kernel for transpose)
    K_T_block_ptrs = tl.make_block_ptr(
        base=K_ptr + qkv_offset,
        shape=(HEAD_DIM, N), # notice the transposed dims
        strides=(stride_K_dim, stride_K_seq),  # by inverting the strides, we are transposing the matrix
        offsets=(0, 0), # no N offsets because for K & V we parallelize across N in for loop in _attn_inner_fwd()
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV), # don't forget transpose means this shape is flipped
        order=(0, 1), # order is how we tell tl.make_block_ptr that we're transposing, it denotes the "order of the original data format"
    )
    """
    Here is the above ^ function implemented manually. 
    Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
    K_ptr += qkv_offset
    offsets_K_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_V_head_dim = tl.arange(0, HEAD_DIM)
    K_T_block_ptrs = K_ptr + (offsets_K_N[None, :] * stride_V_seq + offsets_V_head_dim[:, None] * stride_V_dim)
    #"""

    V_block_ptrs = tl.make_block_ptr(
        base=V_ptr + qkv_offset,
        shape=(N, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0), # no N offsets because for K & V we parallelize across N in for loop in _attn_inner_fwd()
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    """
    Here is the above ^ function implemented manually. 
    Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
    V_ptr += qkv_offset
    offsets_V_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_V_head_dim = tl.arange(0, HEAD_DIM)
    V_block_ptrs = V_ptr + (offsets_V_N[:, None] * stride_V_seq + offsets_V_head_dim[None, :] * stride_V_dim)
    #"""

    O_block_ptrs = tl.make_block_ptr( # this should all look the same as Q
        base=O_ptr + qkv_offset,
        shape=(N, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0), # TODO still don't know what order does
    )
    """
    Here is the above ^ function implemented manually. 
    Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
    O_ptr += qkv_offset
    offsets_O_N = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offsets_V_head_dim = tl.arange(0, HEAD_DIM)
    O_block_ptrs = O_ptr + (offsets_O_N[:, None] * stride_O_seq + offsets_V_head_dim[None, :] * stride_O_dim)
    #"""

    # these next two were calculated internally by calls of make_block_ptr() but not given to us and still needed by us.
    # the offsets for the tokens in the Q to process
    offsets_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # the offsets for the tokens in the K and V sequence to process
    offsets_kv = tl.arange(0, BLOCK_SIZE_KV)

    # the running maximum. We have one for each query in the block we're currently working on
    #m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf") 
    m_i = tl.full(shape=[BLOCK_SIZE_Q], value=-1e6, dtype=tl.float32)
    # the running sum. We have one for each query (since we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0 # the +1 is because we'll be using exponentials and e^0=1
    
    # the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptrs) # shape (BLOCK_SIZE_Q, HEAD_DIM)

    # calculate attention for dense blocks (those where the mask if full of 1's). This step runs for 
    # the entirety of non-causal attention and for the blocks below the diagonal in causal attention
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_T_block_ptrs,
        V_block_ptrs,
        block_index_q,
        softmax_scale,
        #stride_K_seq, # if you go for a manual pointer implementation then you'll need to pass in these strides.
        #stride_V_seq, # in the automatic implementation relevant stride lengths are saved in the pointer object
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        CAUSAL,
        False, # blocks on the DIAGONAL get special treatment if this is set to true; we use it below
        offsets_q,
        offsets_kv,
        N,
    )

    if CAUSAL: # This step runs for the blocks on the diagonal in the causal attention mask
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptrs,
            V_block_ptrs,
            block_index_q,
            softmax_scale,
            #stride_K_seq, # if you go for a manual pointer implementation then you'll need to pass in these strides.
            #stride_V_seq, # in the automatic implementation relevant stride lengths are saved in the pointer object
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            CAUSAL,
            True, # blocks on the diagonal get special masking treatment
            offsets_q,
            offsets_kv,
            N,
        )
    
    # finally dividing by the denominator of our softmax.
    # notice we've already multiplied by V to get O, so this was done out-of-order from naive softmax implementations
    O_block = O_block / l_i[:, None] # shapes (BLOCK_SIZE_Q, HEAD_DIM) / (BLOCK_SIZE_Q, 1) = (BLOCK_SIZE_Q, HEAD_DIM)
        # we can do this out-of-order since the matmul (the tl.dot in _attn_fwd_inner) and this entry-wise division 
        #  are associative. matmul and entry-wise-ops are not normally, but at this level of granularity it's no longer
        #  actually a matmul but instead individual dot-products

    # This is needed to compute the logsumexp (LSE) for the backwards pass. basically instead of saving the maxes 
    #  and the sums separately, we save them together which still works thanks to exponential arithmetic
    m_i += tl.math.log2(l_i)  # l_i was composed using the sum & exp operations in _attn_fwd_inner()
        # this will work because softmax(x_i) = exp(x_i - m_i) / l_i 
        #                                     = exp(x_i - m_i) / exp(log(l_i)) 
        #                                     = exp(x_i - m_i - log(l_i))
        # so a better re-naming / way to think about thsi would be LSE_i = m_i + tl.math.log(l_i)
        #  but the name was decided earlier and still makes sense in the context of using m_i for maxes in most ops
    
    # storing it all back to DRAM
    m_block_ptrs = M_ptr + index_batch_head * N + offsets_q
    tl.store(m_block_ptrs, m_i)
    tl.store(O_block_ptrs, O_block)


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_ROW": BLOCK_SIZE_ROW},
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_ROW in [128]
        for num_stages in ([5])
        for num_warps in [4]
    ],
    key=["N", "HEAD_DIM"], # auto-tune will re-run every time either of these values changes in a new input
)
@triton.jit
def attn_backward_preprocess(
    O_ptr,
    dLdO_ptr,
    Delta_ptr,
    N,
    BLOCK_SIZE_ROW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """the job of this kernel is to pre-compute Delta since Delta is used by both of the following two kernels"""
    index_batch_head = tl.program_id(0) # BATCH_SIZE * NUM_HEADS number of pids
    row = tl.program_id(1) # N / BLOCK_SIZE_ROW number of pids

    row_offsets = row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, HEAD_DIM)

    # Load BLOCK_SIZE_ROW rows of O
    O_ptr += idx_batch_head * HEAD_DIM * N # moves O_ptr to the correct batch & head for this pid.
        # HEAD_DIM * N is equal to stride_num_heads. we can use it instead of .stride() assuming we know dLdO is contiguous
    O_offsets = row_offsets[:, None] * HEAD_DIM + col_offsets[None, :]
    O_block = tl.load(O_ptr + O_offsets) # shape (BLOCK_SIZE_ROW, HEAD_DIM)

    # Load BLOCK_SIZE_ROW rows of dLdO
    dLdO_ptr += index_batch_head * HEAD_DIM * N
    dLdO_offsets = row_offsets[:, None] * HEAD_DIM + col_offsets[None, :]
    dLdO_block = tl.load(dLdO_ptr + dLdO_offsets) # shape (BLOCK_SIZE_ROW, HEAD_DIM) 

    # Delta is the dot product of O and dLdO along HEAD_DIM, giving us a single scalar Delta_i per token in N
    # it will be useful in later parts of the backward pass
    Delta_block = tl.sum(dO_block * O_block, axis=1) # shape (BLOCK_SIZE_ROW)
    Delta_offsets = index_batch_head * N + row_offsets
    tl.store(Delta_ptr + Delta_offsets, Delta_block)


@triton.jit
def attn_backward(
    Q_ptr, K_ptr, V_ptr, 
    scale,
    dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    M, Delta,
    stride_B, stride_H, stride_N, stride_D,
    H, N,
    D: tl.constexpr, 
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
):
    """
    this implementation of the flash-attention backward pass is derived from the Triton documentation tutorials,
    which is actually a bit faster than the flash-attention implementation described in the original papers
    https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py
    
    my edits focus on changing variable names, rearranging, and adding hella comments to help it all make sense
    """
    # TODO so i guess we'll use this on K later?
    ln2: tl.constexpr = 0.6931471824645996  # = ln(2)
        # generally defining a known constant as an approximation of itself to some number of digits
        #  is more efficient than calculating the actual value every time
    
    pid = tl.program_id(0)
    idx_batch_head = tl.program_id(2)

    # move pointers of (B, H, N, D) matrices to get to the correct batch and head
    idx_batch = idx_batch_head // H
    idx_head = idx_batch_head % H 
    batch_head_jump = (idx_batch * stride_B + idx_head * stride_H)#.to(tl.int64)
    Q_ptr += batch_head_jump
    K_ptr += batch_head_jump
    V_ptr += batch_head_jump
    dLDO_ptr += batch_head_jump
    dLDQ_ptr += batch_head_jump
    dLDK_ptr += batch_head_jump
    dLDV_ptr += batch_head_jump

    # move pointers of (B, H, N) matrices to get to the correct batch and head
    batch_head_jump = (idx_batch_head * N)#.to(tl.int64)
    M_ptr += batch_head_jump
    Delta_ptr += batch_head_jump

    # load K and V
    # in the fwd loop we held a block of Q in SRAM and iterated through K & V; here we'll do the opposite
    start = pid * BLOCK_N1
    offsets_N1 = start + tl.arange(0, BLOCK_N1)
    offsets_D = tl.arange(0, D)
    KV_offsets = offsets_N1[:, None] * stride_N + offsets_D[None, :] * stride_D
    K_block = tl.load(K_ptr + KV_offsets) # remember K was pre-scaled earlier in the wrapper around the kernel
    V_block = tl.load(V_ptr + KV_offsets)
    
    # we'll accumulate the gradients into here
    dLdv_block = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dLdk_block = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # TODO why are we cutting our block in half and how is this a mask?
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR 
    num_steps = BLOCK_N1 // MASK_BLOCK_M1 

    # compute dLdK and dLdV for masked blocks
    dLdK, dLdV = _attn_bwd_KV(
        K_block, V_block, 
        dLdK_block, dLdV_block,
        Q_ptr, dLdO_ptr
        M_ptr, Delta_ptr,
        scale,
        stride_N, stride_D,
        H, N,
        D,
        MASK_BLOCK_M1, BLOCK_N1,
        # TODO
    )

    start_n = start
    start_m = start + num_steps * MASK_BLOCK_M1
        # TODO isn't this^ just BLOCK_M1 assuming BLOCK_M1 is divisible by MASK_BLOCK_M1?
    num_steps = (N - start_m) // BLOCK_M1

    # compute dLdK and dLdV for non-masked blocks
    dLdK, dLdV = _attn_bwd_KV(
        # TODO
    )

    # write back dLdK and dLdV
    dLdK *= scale # TODO explain the weird scaling
    tl.store(dLdK_ptr + KV_offsets, dLdK)
    tl.store(dLdV_ptr + KV_offsets, dLdV)

    ### Now we do dLdQ