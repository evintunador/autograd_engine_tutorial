import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

"""
this implementation of flash-attention only supports a causal mask, no other masks or lack of a mask
the forward pass is based primarily on the pseudocode from the two original papers
https://arxiv.org/abs/2205.14135
https://arxiv.org/abs/2307.08691
and the backward passs is based primarily on the triton documentation implementation since it's 
significantly faster than the pseudocode from the original papers
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py
"""

@triton.jit
def _attn_fwd_inner(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    softmax_scale,
    stride_K_N, stride_V_N,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    N: tl.constexpr, Dh: tl.constexpr,
):
    if DIAGONAL:
        # Used only for the blocks along the diagonal in which there is transition between non-masked and masked keys
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        # let the compiler know lo is a muliple of BLOCK_SIZE_QO to speed things up
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO) # TODO not sure why this doesn't also help with hi
    else: 
        # this part is for any blocks in the causal mask below the diagonal
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    # loop over blocks along the N dimension of K & V and update the O accumulator while doing so
    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_KV is a multiple of BLOCK_SIZE_KV, so the compiler can do optimizations
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
            # when in doubt, use tl.multiple_of() for any dynamic variable (as opposed to static variables)

        # compute (Q @ K^T) / sqrt{Dh}
        N_mask_KV = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=N_mask_KV[None, :], other=0.) # shape (Dh, BLOCK_SIZE_KV)
            # sequence mask sets non-existent tokens in the block past N to zero vector
        S = tl.dot(Q, K_T) * softmax_scale # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
            # the masked tokens create columns & rows of zeros hugging the bottom and right edges of S

        if DIAGONAL: # if we're currently on a block containing the diagonal
            # the causal mask is True on the lower-triangular including the diagonal
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            # causal mask addition sets upper-triangular values (excluding diagonal) to -inf
            S += tl.where(causal_mask, 0, -1.0e6) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        # notice that the masked out tokens previously hugging the right edge of S have all been replaced with -inf
        #  and the masked out tokens hugging the bottom edge are still mostly 0's but with some -infs towards the
        #  right edge of each of them, except for the last one which is only 0's
        
        # find the max values of the new block and compare them to those of all previous blocks to get an update
        M_new = tl.maximum(M, tl.max(S, axis=1)) # shape is (BLOCK_SIZE_QO)
            # masked token rows at the bottom will return a maximum value of 0 since their only values are 0 and -inf
        # adjust S block for safe softmax
        S -= M_new[:, None]
            # in the case of masked non-existent tokens that means subtracting by 0 so no difference

        # Compute the exponential of each safe dot product, which will be the numerator of our softmax
        P = tl.exp2(S) 
            # we're using base 2 instead of base e because it's faster and softmax is invariant to the change,
            #  however it does make the derivative in the backward pass a bit more complicated.
            # for the masked non-existent tokens as the bottom that will be 2^0=1 for all those entries
        
        # Compute the sum by rows of the attention scores
        L_new = tl.sum(P, axis=1) # shape (BLOCK_SIZE_QO)
            # for the masked non-existent tokens we're summing a bunch of 1's with some -infs, except for the
            #  very bottom one which is just 1's and therefore its sum is the largest being equal to BLOCK_SIZE_QO
        # This alpha is the correction factor we'll use on the previous L
        alpha = tl.exp2(M - M_new) # shape (BLOCK_SIZE_Q)
            # for the masked non-existent tokens that's just 2^(1-1)=2^0=1=alpha_i so no correction
        # Apply the correction factor to the previous L and add the new L
        L = L * alpha + L_new
            # for each of the masked non-existent tokens they approach N for their entry L_i

        # This computes O = P @ V + O * alpha
        V = tl.load(V_ptr + V_offsets, mask=N_mask_KV[:, None], other=0.) # shape (BLOCK_SIZE_KV, Dh)
        O = O * alpha[:, None] # adjusts previous values based on potential new max
        # accumulated P and V block dot product into O
        O = tl.dot(P, V, acc=O) # shape (BLOCK_SIZE_QO, Dh)
            # notice we're doing this V projection before we've actually divided by our softmax denominator l_i
            #  which is possible because in this context the two operations are associative
            # acc tells triton to accumulate the values into O_block
            # the masked non-existent tokens are a bunch of 1's in the bottom rows of P and 0's in the bottom
            #  rows of V. This matmul leaves O with a bunch of incorrect values in its bottom rows, but they
            #  will get ignored later when we store O with a proper mask

        M = M_new # sets old max equal to new max, ready to be used by next iteration of for loop

        # iterate pointers
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M # we save these three specifically for use later in the backward pass


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [32]
        for BLOCK_SIZE_KV in [32]
        for num_stages in ([1, 3, 5])
        for num_warps in [2, 4, 8]
    ],
    key=["N", "D"], # auto-tune will re-run every time either of these values changes in a new input
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr,  V_ptr,  # each shape (B, H, N, Dh)
    O_ptr,  # shape (B, H, N, Dh). where we store the final output
    LSE_ptr,  # shape (B, H, N). here we first store the max values of each row & later the logsumexp trick 
    softmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh, # dist to move thru mem to next entry in that dim of that tensor
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, # unlike other tensor dimensions, batch size needs to be flexible for runtime differences
    # meta-parameters (decided at compile-time)
    H: tl.constexpr, N: tl.constexpr, 
    Dh: tl.constexpr, # should always be a power of 2, and really 128 and 256 are the only reasonable options
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
):
    """
    this implementation of the flash-attention forward pass was created primarily from the psuedocode in the two papers
    https://arxiv.org/abs/2205.14135
    https://arxiv.org/abs/2307.08691
    plus some optimizations from the Triton docs implementation
    https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py
    """
    # in order to use tl.exp2 later isntead of tl.exp (the former is faster) we need to scale our softmax scale by ln2
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    
    # as opposed to regular assert, static_assert occurs at compile-time
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
        # Dh is usually relatively small (128 or 256) so it wouldn't make sense to parallelize within it

    # This indicates which block in the sequence length to process
    block_index_QO = tl.program_id(0)
    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_BH = tl.program_id(1)
    # This indicates which batch this program is associated with (each batch has H heads)
    index_B = index_BH // H
    # This indicates the position of the head in the batch
    index_H = index_BH % H

    # This allows to get the shape (N, Dh) block in the Q, K, V, and O by indexing it by batch and head
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    # Offsets for N are split by pids but for Dh we keep the whole thing in SRAM.
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) # shape (BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)
    
    # create offsets specific to each tensor
    Q_offsets = (offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh)
        # shape (BLOCK_SIZE_QO, Dh)
    # we transpose K while loading it (as opposed to writing a whole separate kernel for transpose)
    K_T_offsets = (offsets_Dh[:, None] * stride_V_Dh + offsets_KV_N[None, :] * stride_V_N)
        # shape (Dh, BLOCK_SIZE_KV)
    V_offsets = (offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh)
        # shape (BLOCK_SIZE_KV, Dh)

    # load the block of Q that this PID will use: it will stay in SRAM throughout the inner loop
    N_mask_QO = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=N_mask_QO[:, None], other=0.) # shape (BLOCK_SIZE_QO, Dh)
        # sequence mask sets non-existent tokens in the block past N to zero vector

    ## pre-allocate tensors for storing intermediate & output values
    # the running maximum. We have one entry for each query in the block we're currently working on
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32) # large negative number will get ignored by tl.max()
    # the running sum. We have one entry for each query (since we sum the attention scores by rows)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32) # 1 is because we'll be using exponentials and e^0=1
    # the accumulator for the output, which is a group of rows of the O matrix
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)

    # calculate attention for dense blocks (those where the mask if full of 1's). 
    # This step runs for the blocks below the diagonal in causal attention
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False, # blocks on the DIAGONAL get special treatment if this is set to true; we use it below
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )

    # This step runs for the blocks on the diagonal in the causal attention mask
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True, # blocks on the diagonal get special masking treatment
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )
    
    # finally dividing by the denominator of our softmax.
    # notice we've already multiplied by V to get O, so this was done out-of-order from naive softmax implementations
    O = O / L[:, None] # shapes (BLOCK_SIZE_QO, Dh) / (BLOCK_SIZE_QO, 1) = (BLOCK_SIZE_QO, Dh)
        # we can do this out-of-order since the matmul (the tl.dot in _attn_fwd_inner) and this entry-wise division 
        #  are associative. matmul and entry-wise-ops are not normally, but at this level of granularity it's no longer
        #  actually a matmul but instead individual dot-products
        # the masked non-existent tokens are a bunch of meaningless values in the bottom rows of O and generally
        #  roughly equal to N in the bottom entries of L. Dividing the former by the latter isn't going to break
        #  anything and we'll mask them out later when storing

    # This is needed to compute the logsumexp (LSE) for the backwards pass. basically instead of saving the maxes 
    #  and the sums separately, we save them together which still works thanks to exponential arithmetic
    LSE = M + tl.math.log2(L)  # L was composed using the sum & exp operations in _attn_fwd_inner()
        # this will work because softmax(x_i) = exp(x_i - m_i) / l_i 
        #                                     = exp(x_i - m_i) / exp(log(l_i)) 
        #                                     = exp(x_i - m_i - log(l_i))
        # the masked non-existent tokens are a bunch of 0's in the bottom entries of M and a bunch of values roughly
        #  equal to N in the bottom entries of L. So in LSE they'll be a bunch of log_2(N) entries at the bottom
        #  that we of course don't plan to use

    ## storing it all back to DRAM
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)
        # the mask prevents us from saving the useless log_2(n) values at the bottom of LSE
    O_offsets = (offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh)
        # shape (BLOCK_SIZE_Q, Dh)
    tl.store(O_ptr + O_offsets, O, mask=N_mask_QO[:, None])
        # the mask prevents us from saving the useless values at the bottom of O corresponding to non-existent tokens


@triton.autotune(
    [
        triton.Config({"PRE_BLOCK_SIZE_ROW": PRE_BLOCK_SIZE_ROW},
                        num_stages=num_stages, num_warps=num_warps,)
        for PRE_BLOCK_SIZE_ROW in [32, 64, 128]
        for num_stages in [1, 3, 5]
        for num_warps in [2, 4, 8]
    ],
    key=["N", "D"], # auto-tune will re-run every time either of these values changes in a new input
)
@triton.jit
def attn_backward_preprocess(
    O_ptr, dLdO_ptr, Delta_ptr,
    N, D: tl.constexpr,
    PRE_BLOCK_SIZE_ROW: tl.constexpr,
):
    """the job of this kernel is to pre-compute Delta since Delta is used by both of the following two kernels"""
    index_batch_head = tl.program_id(1) # B * H number of pids
    row = tl.program_id(0) # N / BLOCK_SIZE_ROW number of pids

    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, D)

    mask = row_offsets < N

    # Load PRE_BLOCK_SIZE_ROW rows of O
    O_ptr += index_batch_head * D * N # moves O_ptr to the correct batch & head for this pid.
        # D * N is equal to stride_H. we can use it instead of .stride() assuming we know dLdO is contiguous
    O_offsets = row_offsets[:, None] * D + col_offsets[None, :]
    O_block = tl.load(O_ptr + O_offsets, mask = mask[:, None], other=0.) # shape (PRE_BLOCK_SIZE_ROW, D)

    # Load PRE_BLOCK_SIZE_ROW rows of dLdO
    dLdO_ptr += index_batch_head * D * N
    dLdO_offsets = row_offsets[:, None] * D + col_offsets[None, :]
    dLdO_block = tl.load(dLdO_ptr + dLdO_offsets, mask = mask[:, None], other=0.) # shape (PRE_BLOCK_SIZE_ROW, D) 

    # Delta is the dot product of O and dLdO along D, giving us a single scalar Delta_i per token in N
    # it will be useful in later parts of the backward pass
    Delta_block = tl.sum(dLdO_block * O_block, axis=1) # shape (PRE_BLOCK_SIZE_ROW)
    Delta_ptr += index_batch_head * N
    tl.store(Delta_ptr + row_offsets, Delta_block, mask = mask)


@triton.jit
def _attn_backward_KV(
    K, V, dLdK, dLdV,               # shape (BLOCK_SIZE_COL, D)
    Q_ptr, dLdO_ptr,
    M_ptr, Delta_ptr, 
    stride_N, stride_D,
    H, N, D: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,   # no more _1 because this sub-kernel is the _1
    BLOCK_SIZE_COL: tl.constexpr, 
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    """
    this kernel will be looking at a specific chunk of K & V , where we call
    the sequence length of K & V the columns of our NxN attention matrix,
    and iterating through rows of Q's sequence length to calculate that 
    chunk of dLdK and dLdV
                    N of K & V
               |    |   |   |   |
               |    |   |   |   |
    N of Q     |    |   |   |   |
               |    |   |   |   |
              \|/  \|/ \|/ \|/ \|/
    arrows indicate direction of our for loop; each arrow is a different PID
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_D = tl.arange(0, D)

    # we transpose Q while loading it rather than in a separate kernel
    Q_T_ptrs = Q_ptr        + offsets_D[:, None] * stride_D     + offsets_ROW[None, :] * stride_N
    dLdO_ptrs = dLdO_ptr    + offsets_ROW[:, None] * stride_N   + offsets_D[None, :] * stride_D

    for block_idx in range(num_steps):
        # we load M before computing S to reduce pipeline stall (and dLdO before computing dLdV)
        # meaning the Triton compiler can have an easier time doing the loading of M
        # and the dot product of K and QT simultaneously. in general you should load a bunch
        # of stuff then calc a bunch of stuff rather than flipping b/w loads and calcs
        Q_T = tl.load(Q_T_ptrs) # shape (D, BLOCK_SIZE_ROW)
        M = tl.load(M_ptr + offsets_ROW) # shape (BLOCK_SIZE_ROW)
        dLdO = tl.load(dLdO_ptrs) # shape (BLOCK_SIZE_ROW, D)
        Delta = tl.load(Delta_ptr + offsets_ROW) # shape (BLOCK_SIZE_ROW)
        # ^ notice the order i load these in is based on the order we use them below

        # calculate transpose of S and P matrices
        S_T = tl.dot(K, Q_T) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
            # no scale here because the operation is associative so we did it earlier on K
        P_T = tl.exp2(S_T - M[None, :]) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
            # this derivative actually requires an extra *ln(2) which we do below at dLdS_T

        if MASK: # if we're on the block-diagonal
            # weird shape here with columns on the rows and rows on the columns bc of transpose
            mask = (offsets_COL[:, None] <= offsets_ROW[None, :]) # (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
            P_T = tl.where(mask, P_T, 0.)

        # compute dLdV
        dLdV = tl.dot(P_T, dLdO, acc=dLdV) # shape (BLOCK_SIZE_COL, D)

        # compute dLdP_T and dLdS_T to get dLdK
        dLdP_T = tl.dot(V, tl.trans(dLdO)) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdS_T = P_T * (dLdP_T - Delta[None, :]) * ln2 # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdK = tl.dot(dLdS_T, tl.trans(Q_T), acc=dLdK) # shape (BLOCK_SIZE_COL, D)
            # acc tells the tl.dot to accumulate into dLdK

        # increment pointers
        offsets_ROW += BLOCK_SIZE_ROW
        Q_T_ptrs += BLOCK_SIZE_ROW * stride_N
        dLdO_ptrs += BLOCK_SIZE_ROW * stride_N
    
    return dLdK, dLdV


@triton.jit
def _attn_backward_Q(
    dLdQ, Q, dLdO, M, 
    K_ptr, V_ptr, Delta_ptr,
    stride_N, stride_D,
    H, N, D: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr, 
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    """
    this kernel will be looking at a specific chunk of Q and iterating through
    rows of K & V to calculate that chunk of dLdQ
    I say "rows" of K and V but really we refer to them as colums since we're thinking
    not in terms of the (B, H, N, D) shaped matrices but rather the (B, H, N, N) shaped
    attention logits, where the first N are split up by "BLOCK_SIZE_ROW" and the second N
    is split up by "BLOCK_SIZE_COL"
                    N of K & V
               ------------------->
               ------------------->
    N of Q     ------------------->
               ------------------->
               ------------------->
    arrows indicate direction of our for loop; each arrow is a different PID
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_D = tl.arange(0, D)

    # we transpose V while loading it
    K_and_V_T_offsets = offsets_COL[None, :] * stride_N + offsets_D[:, None] * stride_D
    K_T_ptrs = K_ptr + K_and_V_T_offsets
    V_T_ptrs = V_ptr + K_and_V_T_offsets

    Delta = tl.load(Delta_ptr + offsets_ROW) # shape (BLOCK_SIE_ROW)

    for block_idx in range(num_steps):
        K_T = tl.load(K_T_ptrs) # shape (BLOCK_SIZE_COL, D)
        V_T = tl.load(V_T_ptrs) # shape (BLOCK_SIZE_COL, D)

        S = tl.dot(Q, K_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
            # no scale here because the operation is associative so we did it earlier on Q
        P = tl.exp2(S - M) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        if MASK: # if we're on the block-diagonal
            mask = (offsets_ROW[:, None] >= offsets_COL[None, :]) # (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
            # setting lower-triangular values to zero since the gradient is upper-triangular
            P = tl.where(mask, P, 0.) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        # calc dLdP and dLdS to get dLdQ
        dLdP = tl.dot(dLdO, V_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        dLdS = P * (dLdP - Delta[:, None]) * ln2
        # ^this line is equivalent to:
        #weighted_dLdP = tl.sum(dLdP * P, axis=1)  # row-sum over keys
        #dLdS = P * (dLdP - weighted_dLdP[:, None])
        # but trades-off a memory access for a binary & then a reduction op
        dLdQ += tl.dot(dLdS, tl.trans(K_T))
            # we'll need to de-sdcale dLdQ in the end because K_T was pre-scaled
            # we do it later instead of now bc now would mean num_steps * flops versus just flops
        
        # increment pointers
        offsets_COL += BLOCK_SIZE_COL
        K_T_ptrs += BLOCK_SIZE_COL * stride_N
        V_T_ptrs += BLOCK_SIZE_COL * stride_N
    
    return dLdQ


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_MICRO in [16, 32]
        for BLOCK_SIZE_MACRO in [32, 64]
        for num_stages in [1, 3, 5]
        for num_warps in [2, 4, 8]
        if BLOCK_SIZE_MACRO > BLOCK_SIZE_MICRO # could do >= but i wanna get mileage out of the loop code we wrote
    ],
    key=["N", "D"], # auto-tune will re-run every time either of these values changes in a new input
)
@triton.jit
def attn_backward(
    Q_ptr, K_ptr, V_ptr, 
    dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    M_ptr, Delta_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_D,
    H, N, D: tl.constexpr, 
    BLOCK_SIZE_MICRO: tl.constexpr,  #
    BLOCK_SIZE_MACRO: tl.constexpr,  #
):
    """
    this implementation of the flash-attention backward pass is derived from the Triton documentation tutorials,
    which is actually a bit faster than the flash-attention implementation described in the original papers
    https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py
    
    my edits focus on changing variable names, rearranging, and adding hella comments to help it all make sense
    """
    # we'll use these constants later on Q
    ln2: tl.constexpr = 0.6931471824645996  # = ln(2), natural logarithm of 2
    rln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2), the reciprocal of the natural logarithm of 2
        # generally defining a known constant as an approximation of itself to some number of digits
        #  is more efficient than calculating the actual value every time
    # TODO explain how/why we use these

    # move pointers of (B, H, N, D) matrices to get to the correct batch and head
    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head = idx_batch_head % H 
    batch_head_jump = idx_batch * stride_B + idx_head * stride_H
    Q_ptr += batch_head_jump
    K_ptr += batch_head_jump
    V_ptr += batch_head_jump
    dLdO_ptr += batch_head_jump
    dLdQ_ptr += batch_head_jump
    dLdK_ptr += batch_head_jump
    dLdV_ptr += batch_head_jump

    # move pointers of (B, H, N) matrices to get to the correct batch and head
    batch_head_jump = idx_batch_head * N
    M_ptr += batch_head_jump
    Delta_ptr += batch_head_jump

    # BLOCK_SIZE_MACRO must be a multiple of BLOCK_SIZE_MICRO
    # because we use them to determine num_steps and don't want a remainder
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    ### STAGE 1: First we'll do dLdK and dLdV
    # in the fwd loop we held a block of Q in SRAM and iterated through K & V; here we'll do the opposite

    # ROW and COL refer to the N dimension of Q and K/V respectively
    # for stage 1 each PID will look at BLOCK_SIZE_MACRO tokens of K & V and there will 
    #  be an inner for-loop iterating over BLOCK_SIZE_MICRO tokens of Q at a time
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO

    # first we'l do the gradients along the block diagonal since they get treated differently
    #  in that they have an triangular causal mask
    pid = tl.program_id(0)
    start_COL = pid * BLOCK_SIZE_COL_1
    start_ROW = start_COL
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1
    
    # load K & V
    offsets_COL_1 = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_D = tl.arange(0, D)
    KV_offsets = offsets_COL_1[:, None] * stride_N + offsets_D[None, :] * stride_D
    K_block = tl.load(K_ptr + KV_offsets) # shape (BLOCK_SIZE_COL_1, D)
    # pre-scaling K allows us to do the multiplication once here as opposed to
    #  num_steps times inside _attn_backward_KV
    # we also scale by rln2 to account for the derivative of tl.exp2() and do it
    #  here instead of inside _attn_backward_KV for the same reason
    K_block *= scale * rln2
    V_block = tl.load(V_ptr + KV_offsets) # shape (BLOCK_SIZE_COL_1, D)

    # we'll accumulate the gradients into these
    dLdK = tl.zeros([BLOCK_SIZE_COL_1, D], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, D], dtype=tl.float32)

    # compute dLdK and dLdV portions along the blocked diagonal
    dLdK, dLdV = _attn_backward_KV(
        K_block, V_block, 
        dLdK, dLdV,
        Q_ptr, dLdO_ptr,
        M_ptr, Delta_ptr,
        stride_N, stride_D,
        H, N, D,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    # next we'll do all the blocks that don't need the triangular mask on the block-diagonal.
    # this moves us forward to get off of the block-diagonal
    start_ROW += BLOCK_SIZE_COL_1
        # start_COL doesn't change since that's determined by our PID
    # then we calculate how many blocks need to be done
    num_steps = (N - start_ROW) // BLOCK_SIZE_ROW_1
        # TODO use a ceiling above N since N isn't forced to be a multiple of 128

    # compute dLdK and dLdV for non-masked blocks
    dLdK, dLdV = _attn_backward_KV(
        K_block, V_block, 
        dLdK, dLdV,
        Q_ptr, dLdO_ptr,
        M_ptr, Delta_ptr,
        stride_N, stride_D,
        H, N, D,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False ###
    )

    # scale since we didn't do it inside _attn_backward_KV to save flops
    dLdK *= scale * rln2
    # write back dLdK and dLdV
    tl.store(dLdK_ptr + KV_offsets, dLdK)
    tl.store(dLdV_ptr + KV_offsets, dLdV)

    ### STAGE 2: Now we do dLdQ
    # in this part, like the forward pass we look at a specific block of Q & iterate through K & V

    # ROW and COL refer to the N dimension of Q and K/V respectively
    # for stage 1 each PID will look at BLOCK_SIZE_MACRO tokens of K & V
    # and there will be an inner for loop iterating over BLOCK_SIZE_MICRO tokens of Q
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO

    # we again start off doing the block-diagonal
    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    # BLOCK_SIZE_ROW_2 must be a multiple of BLOCK_SIZE_COL_2
    tl.static_assert(BLOCK_SIZE_ROW_2 % BLOCK_SIZE_COL_2 == 0)
    # because we use them to determine num_steps and don't want a remainder
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2
    # ^this is number of steps for a single block, aka the blocks on the diagonal

    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    Q_offsets = offsets_ROW[:, None] * stride_N + offsets_D[None, :] * stride_D
    Q = tl.load(Q_ptr + Q_offsets) # shape (BLOCK_SIZE_ROW_2, D)
    Q *= scale * rln2
    dLdO = tl.load(dLdO_ptr + Q_offsets) # dLdO shares the same shape as Q so it can use same offsets
    M = tl.load(M_ptr + offsets_ROW)[:, None] # shape (BLOCK_SIZE_ROW_2, 1)

    # accumulate the gradients into here
    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, D], dtype=tl.float32)

    # compute dQ for blocks on the diagonal
    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, M, 
        K_ptr, V_ptr, Delta_ptr, 
        stride_N, stride_D,
        H, N, D,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    # now we'll do the parts that are not on the block-diagonal
    end_COL = start_COL
    start_COL = 0 #end_COL - num_steps * BLOCK_SIZE_COL_2 # could just call it 0 lmao
    num_steps = end_COL // BLOCK_SIZE_COL_2
    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, M, 
        K_ptr, V_ptr, Delta_ptr, 
        stride_N, stride_D,
        H, N, D,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )
    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + Q_offsets, dLdQ)