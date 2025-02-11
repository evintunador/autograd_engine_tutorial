import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.autotune( 
    [
        triton.Config({"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N}, 
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_M in [32, 64, 128]
        for BLOCK_SIZE_N in [32, 64, 128]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["embedding_dimension?"], # TODO
)
@triton.jit
def embedder_forward(
    ids_ptr,                        # shape (B, N) that we can treat like (B * N)
    E_ptr,                          # shape (V, D)
    x_ptr,                          # shape (B, N, D)
    embed_dim_stride,
    ids_num_elements, E_num_elements, x_num_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.assume(BLOCK_SIZE_N >= embed_dim_stride)

    pid = tl.program_id(0)
    
    col_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_offsets = tl.arange(0, BLOCK_SIZE_N)

    ids_mask = col_offsets < ids_num_elements
    ids = tl.load(ids_ptr + col_offsets, mask=ids_mask, padding_option="")

    E_offsets = ids[:, None] * embed_dim_stride + row_offsets[None, :]
    E_mask = col_offsets[:, None] < 
    E_block = tl.load(E_ptr + E_offsets)

    x_offsets = pid * BLOCK_SIZE_M * BLOCK_SIZE_N \
                + col_offsets[:, None] * BLOCK_SIZE_N \
                + row_offsets[None, :]
    tl.store(x_ptr + x_offsets, E_block)