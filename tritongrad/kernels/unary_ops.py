import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE in [32, 64]#, 128, 256, 512, 1024, 2048, 4096] # values chosen by totally guessing
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["n_elements"], # auto-tune will re-run every time this value is different in a new input
)
@triton.jit
def unary_op_forward(
    x_ptr,                      # pointer to input tensor
    z_ptr,                      # pointer to desired output tensor
    n_elements,                 # number of elements in input & output tensors
    op: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if op == "exp":
        tl.store(z_ptr + offsets, tl.exp(x), mask=mask)
    if op == "log":
        tl.store(z_ptr + offsets, tl.log(x), mask=mask)


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE in [32, 64]#, 128, 256, 512, 1024, 2048, 4096] # values chosen by totally guessing
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["n_elements"], # auto-tune will re-run every time this value is different in a new input
)
@triton.jit
def unary_op_backward(
    x_ptr,                      # pointer to input tensor
    dx_ptr,                     # pointer to gradient of input
    z_ptr,                      # pointer to the forward pass' output tensor
    dz_ptr,                     # pointer to the incoming gradient
    n_elements,                 # number of elements in input & output tensors
    op: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dz = tl.load(dz_ptr + offsets, mask=mask)

    if op == "exp":
        z = tl.load(z_ptr + offsets, mask=mask)
        tl.store(dx_ptr + offsets, z * dz, mask=mask)
    if op == "log":
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(dx_ptr + offsets, dz / x, mask=mask)