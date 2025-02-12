import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

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