import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]

@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_fwd(
    a_ptr, b_ptr, c_ptr, # pointers to first entries of matrices
    M, N, K, # matrix dimensions
    stride_a_preceeding_dims, stride_am, stride_ak, # tuple of how much to increase the ptr by when moving by 1 element along that dimension
    stride_b_preceeding_dims, stride_bk, stride_bn, 
    stride_c_preceeding_dims, stride_cm, stride_cn,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # parallelizing across preceeding dimensions
    pid_preceeding_dims = tl.program_id(axis=1)
    # we split the preceeding dimensions across 
    a_ptr += pid_preceeding_dims * stride_a_preceeding_dims
    b_ptr += pid_preceeding_dims * stride_b_preceeding_dims
    c_ptr += pid_preceeding_dims * stride_c_preceeding_dims

    # first we map program ids (pids) to the block of C it should compute
    # this is done in grouped ordering to promote SRAM data reuse (pids within a group share data along k dimension)
    # for a visual see https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    # Each GPU has multiple Streaming Multiprocessors (SMs), and each SM has its own SRAM.
    # Thread blocks (represented by PIDs) that run on the same SM can share SRAM, but blocks on different SMs cannot.
    # Triton is smart enough to reuse data already loaded onto the same SRAM than load on a duplicate copy.
    # The grouping strategy below tries to arrange the computation so that blocks that need the same data
    # are more likely to run on the same SM, allowing them to share data through SRAM.
    # we start with a 1D launch grid that we will turn into a 2D grid WITH A COMPLICATED ORDERING
    pid = tl.program_id(axis=0) 
    num_pid_along_m = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_pid_along_n = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_pid_in_group = GROUP_SIZE * num_pid_along_n 
        # so for the GROUP_SIZE number of rows we'll be using, we need to grab a matching set of columns, specifically
        # we'll grab GROUP_SIZE number of columns each with num_pid_along_n number of blocks in them.
        # meaning these are groups OF BLOCKS
    group_id = pid // num_pid_in_group # figurinig out which group we are
    first_pid_in_group_along_m = group_id * GROUP_SIZE # tells us which row to start at for this group
    group_size_adj = min(num_pid_along_m - first_pid_in_group_along_m, GROUP_SIZE) # usually equal to GROUP_SIZE.
        # alternative case happens when we're at edge of tensor and tensor dimensions don't cleanly divde into GROUP_SIZE
        # ^ is this true? idk

    # this is the bulk of the actual mapping from 1D to 2D - the weird pattern that is the reason we didn't use a 2D grid
    # (pid % num_pid_in_group) puts the current program id into the context of a group
    pid_m = first_pid_in_group_along_m + ((pid % num_pid_in_group) % group_size_adj)
        # (first_pid_in_group_along_m +) shifts the pid into the correct group
        # (% group_size_adj) removes the column component to get us onto the correct row
    pid_n = (pid % num_pid_in_group) // group_size_adj
        # (// group_size_adj) removes the row component to get us onto the correct column
    """
    as an example, pick any PID from the first matrix and follow it through the above calculations to the second matrix
    M=N=K=8
    BLOCK_SIZE_M=BLOCK_SIZE_N=BLOCK_SIZE_K=2
    GROUP_SIZE=2
    # naive 1D placement of pid's
    [' 0', ' 1', ' 2', ' 3']
    [' 4', ' 5', ' 6', ' 7']
    [' 8', ' 9', '10', '11']
    [' 12', '13', '14', '15']
    # final 2D pid access grid
    [' 0', ' 2', ' 4', ' 6']
    [' 1', ' 3', ' 5', ' 7']
    [' 8', '10', '12', '14']
    [' 9', '11', '13', '15'] 
    note that the pid number is the order in which these get loaded into an SM and onto SRAM, meaning that
    if a given SM can handle 4 pid's, then 0, 1, 2, and 3 will all get loaded onto the same one. Since 0 and 1
    make tl.load() calls to the same columns, first 0 will load it and then when 1 goes to load it Triton will
    say "oh, it's already here, just use that" and same for the column that 2 & 3 share as well as the row that
    0 & 2 share and the row that 1 & 3 share. It's not guaranteed that the SM handles an ideal number of pid's 
    for your matrix shape and the way you loaded it in, but that's why we did autotuning earlier, to find 
    ideal block and group sizes given the properties of our hardware, such as the number of pid's an SM can handle.
    """

    # now we'll create pointer vectors for the first group of blocks of the input matrices
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    offsets_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offsets_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offsets_k = tl.arange(0, BLOCK_SIZE_K) # this is used to setup k dimension of initial a & b offsets
    # now we convert our group/block/pid based indices to their actual tensor mappings using first-entry pointers & stride length
    a_ptrs = a_ptr + (offsets_m.expand_dims(1) * stride_am + offsets_k.expand_dims(0) * stride_ak)
    b_ptrs = b_ptr + (offsets_k.expand_dims(1) * stride_bk + offsets_n.expand_dims(0) * stride_bn)
        
    # iterate to compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)): # iterate from 0 to the number of blocks along K dimension
        # out-of-bounds entries need to be masked out
        a_mask = (offsets_m.expand_dims(1) < M) & (offsets_k.expand_dims(0) < K)
        b_mask =(offsets_k.expand_dims(1) < K) & (offsets_n.expand_dims(0) < N)
        
        # Now we load blocks of A and B matrices. If multiple blocks in a group are on the same SM, 
        # they can share these loaded values, which reduces the number of expensive loads from DRAM
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            # fill in any masked-out parts with 0.0 which don't have any effect on matmuls

        # we accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator)
            # triton is weird with operation notation; this is actually a tiny matmul not just a dot product

        # advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offsets_k += BLOCK_SIZE_K

    # write back the block of the output matrix C with masks
    c_ptrs = c_ptr + stride_cm * offsets_m.expand_dims(1) + stride_cn * offsets_n.expand_dims(0)
    c_mask = (offsets_m.expand_dims(1) < M) & (offsets_n.expand_dims(0) < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_bwd(
    a_ptr, b_ptr, c_ptr, # pointers to first entries of data matrices
    da_ptr, db_ptr, dc_ptr, # pointers to first entries of gradient matrices
    M, N, K, # matrix dimensions
    stride_a_preceeding_dims, stride_am, stride_ak,
    stride_b_preceeding_dims, stride_bk, stride_bn, 
    stride_c_preceeding_dims, stride_cm, stride_cn,
    stride_da_preceeding_dims, stride_dam, stride_dak,
    stride_db_preceeding_dims, stride_dbk, stride_dbn, 
    stride_dc_preceeding_dims, stride_dcm, stride_dcn,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    The big differences between matmul_fwd and matmul_bwd are that 
        1) the latter has to include the transpose operation
        2) the latter actually does two matmuls
        
    Forward:
    A @ B = C                               (M, K) @ (K, N) -> (M, N)
    Backward:
    dC @ B^T = dA                           (M, N) @ (N, K) -> (M, K)
    A^T @ dC = dB                           (K, M) @ (M, N) -> (K, N)

    but instead of effectively doing two separate kernels, 
    in order to ensure we're always utilizing the same block of dC, we do
    (dC^T @ A)^T = (dB^T)^T = dB            ((N, M) @ (M, K))^T -> (N, K)^T -> (K, N)
    but really we'll the transpose of dB^T back dB to at tl.store() time
    """
    # parallelizing across preceeding dimensions
    pid_preceeding_dims = tl.program_id(axis=1)
    # we split the preceeding dimensions across 
    a_ptr += pid_preceeding_dims * stride_a_preceeding_dims
    b_ptr += pid_preceeding_dims * stride_b_preceeding_dims
    c_ptr += pid_preceeding_dims * stride_c_preceeding_dims
    da_ptr += pid_preceeding_dims * stride_da_preceeding_dims
    db_ptr += pid_preceeding_dims * stride_db_preceeding_dims
    dc_ptr += pid_preceeding_dims * stride_dc_preceeding_dims

    # first we map program ids (pids) to the block of C it should compute
    pid = tl.program_id(axis=0) 
    num_pid_along_m = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_pid_along_n = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_pid_in_group = GROUP_SIZE * num_pid_along_n 
    group_id = pid // num_pid_in_group # figurinig out which group we are
    first_pid_in_group_along_m = group_id * GROUP_SIZE # tells us which row to start at for this group
    group_size_adj = min(num_pid_along_m - first_pid_in_group_along_m, GROUP_SIZE) # usually equal to GROUP_SIZE.
    # (pid % num_pid_in_group) puts the current program id into the context of a group
    pid_m = first_pid_in_group_along_m + ((pid % num_pid_in_group) % group_size_adj)
        # (first_pid_in_group_along_m +) shifts the pid into the correct group
        # (% group_size_adj) removes the column component to get us onto the correct row
    pid_n = (pid % num_pid_in_group) // group_size_adj
        # (// group_size_adj) removes the row component to get us onto the correct column

    # now we'll create pointer vectors for the first group of blocks of the input matrices
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    offsets_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offsets_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offsets_k = tl.arange(0, BLOCK_SIZE_K) # this is used to setup k dimension of initial a & b offsets
    
    # we convert our group/block/pid based indices to their actual tensor mappings using first-entry pointers & stride length
    a_ptrs = a_ptr + (offsets_m.expand_dims(1) * stride_am + offsets_k.expand_dims(0) * stride_ak)
    dc_ptrs = dc_ptr + stride_dcm * offsets_m.expand_dims(1) + stride_dcn * offsets_n.expand_dims(0)
    # notice how instead we have a block of B being BLOCK_SIZE_N rows and BLOCK_SIZE_K columns for the transpose
    b_T_ptrs = b_ptr + (offsets_n.expand_dims(1) * stride_bn + offsets_k.expand_dims(0) * stride_bk)
        
    # iterate to compute a block of each of the dA and dB^T matrices
    accumulator_dA = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    accumulator_dB_T = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, max(tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N))): 

        # out-of-bounds entries need to be masked out
        a_mask = (offsets_m.expand_dims(1) < M) & (offsets_k.expand_dims(0) < K)
        b_T_mask = (offsets_n.expand_dims(1) < N) & (offsets_k.expand_dims(0) < K)
        dc_mask = (offsets_m.expand_dims(1) < M) & (offsets_n.expand_dims(0) < N)
        
        # Now we load blocks. If multiple blocks in a group are on the same SM, 
        # they can share these loaded values, which reduces the number of expensive loads from DRAM
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_T = tl.load(b_T_ptrs, mask=b_T_mask, other=0.0)
        dc = tl.load(dc_ptrs, mask=dc_mask, other=0.0)
            # the even more beautiful thing is that we're sharing dC within the kernel itself

        # we accumulate
        accumulator_dA = tl.dot(dc, b_T, accumulator_dA)
        accumulator_dB_T = tl.dot(tl.trans(dc), A, accumulator_db_T)

        # advance the ptrs to the next block
        a_ptrs += BLOCK_SIZE_M * stride_am
        b_T_ptrs += BLOCK_SIZE_N * stride_bn
        offsets_k += BLOCK_SIZE_K
        dC_ptrs += 
        # TODO hold up, does dC's advancement split it into two separate tensors?


    da_ptrs = da_ptr + (offsets_m.expand_dims(1) * stride_am + offsets_k.expand_dims(0) * stride_ak)
    db_ptrs = db_ptr + (offsets_k.expand_dims(1) * stride_bk + offsets_n.expand_dims(0) * stride_bn)

    # write back the block of the output matrix C with masks
    c_ptrs = c_ptr + stride_cm * offsets_m.expand_dims(1) + stride_cn * offsets_n.expand_dims(0)
    c_mask = (offsets_m.expand_dims(1) < M) & (offsets_n.expand_dims(0) < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)