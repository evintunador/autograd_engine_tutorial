import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# TODO make add_kernel support different input shapes
# TODO 

"""
@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"BLOCK_SIZE": BLOCK_SIZE},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE in [32, 64, 128, 256] # values chosen heuristically
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["n_elements", "loop_stride"], # auto-tune will re-run every time either of these values are different in a new input
)
"""
@triton.jit # this decorator tells Triton to compile this function into GPU code
def binary_op_kernel(
    x_ptr, y_ptr,               # pointers to input vectors (triton converts torch.tensor objects into pointers to their first element)
    output_ptr,                 # ptr to output vector
    n_elements,                 # size of x tensor
    loop_stride,                # size of y tensor
    BLOCK_SIZE: tl.constexpr,   # number of elements each program should process
    OP: tl.constexpr,           # the operation to be performed
):   
    # tl.constexpr is a type that tells the compiler that the value must be known at compile-time (not runtime)
    # there are multiple "programs" processing data (a program is a unique instantiation of this kernel)
    # programs can be defined along multiple dimensions when the inputs have multiple dimensions
    # this op is 1D so axis=0 is the only option, but bigger operations later may define program_id as a tuple
    # here we identify which program we are:
    program_id = tl.program_id(axis=0) 
        # Each program instance gets a unique ID along the specified axis
        # For a vector of length 256 and BLOCK_SIZE=64:
        # program_id=0 processes elements [0:64]
        # program_id=1 processes elements [64:128]
        # program_id=2 processes elements [128:192]
        # program_id=3 processes elements [192:256]

    # this program will process inputs that are offset from the initial data (^ described above)
    # note that offsets is a list of pointers a la [0, 1, 2, ...., 62, 63]
    block_start = program_id * BLOCK_SIZE
    offsets_x = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_y = offsets_x % loop_stride
    
    # Create masks to guard memory operations
    mask_x = offsets_x < n_elements
    mask_y = offsets_y < loop_stride

    # load x and y from DRAM (global GPU memory) into SRAM (on-chip memory)
    # SRAM is much faster but limited in size
    # These masks ensure we don't access memory beyond the tensors' ends
    x = tl.load(x_ptr + offsets_x, mask = mask_x)
    y = tl.load(y_ptr + offsets_y, mask = mask_y)

    # perform the operation for this block on SRAM
    # triton has its own internal definitions of all the basic ops that deal with the actual entry-wise details
    # The conditional here is on a compile-time constant,
    # which Triton can “fold” or “inline” so there’s no runtime overhead.
    # You’ll get a separate compiled kernel per value of OP.
    if OP == "add":
        out = x + y
    elif OP == "sub":
        out = x - y
    elif OP == "div":
        out = x / y
    elif OP == "mul":
        out = x * y
    else:
        raise ValueError(f"input operation must be either 'add', 'sub', 'div', or 'mul' but isntead got {OP}")

    # write back to DRAM, being sure to mask to avoid out-of-bounds accesses
    tl.store(output_ptr + offsets_x, out, mask = mask_x)

@triton.jit
def binary_op_backward_kernel(
    x_ptr, y_ptr,               # pointers to input vectors
    dx_ptr,                     # pointer to first input's gradient, or None if x doesn't require a gradient
    dy_ptr,                     # pointer to second input's gradient, or None if y doesn't require a gradient
    do_ptr,                     # pointer to incoming gradient
    n_elements,                 # total number of elements in x and output tensors
    loop_stride,                # total number of elements in y tensor
    BLOCK_SIZE: tl.constexpr,   # number of elements each program should process
    OP: tl.constexpr,           # the operation to be performed
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this program instance
    block_start_x = pid * BLOCK_SIZE
    block_start_y = block_start_x % loop_stride # the looping is how we handle broadcasting
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
    offsets_y = (block_start_y + tl.arange(0, BLOCK_SIZE)) % loop_stride
    
    # Create masks to guard memory operations
    mask_x = offsets_x < n_elements
    mask_y = offsets_y < loop_stride
    
    # Load incoming gradient do
    do = tl.load(do_ptr + offsets_x, mask=mask_x)
    
    if dx_ptr is not None:
        dx = tl.load(dx_ptr + offsets_x, mask=mask_x)

        if OP == "add":
            dx += do
        elif OP == "sub":
            dx += do
        elif OP == "mul":
            # y_val must be loaded from y_ptr + offsets_y
            y_val = tl.load(y_ptr + offsets_y, mask=mask_y)
            dx += do * y_val
        elif OP == "div":
            # We do x / y => dx = (1 / y) * do
            y_val = tl.load(y_ptr + offsets_y, mask=mask_y)
            dx += do / y_val

        tl.store(dx_ptr + offsets_x, dx, mask=mask_x)
    
    if dy_ptr is not None:
        # if we were to use the same code as above, then in the case of broadcasting the threads would be
        #  overwriting each other. instead, we use atomic_add which uses a locking mechanism to ensure
        #  that only one thread works on a given entry in dy at a time

        if OP == "add":
            tl.atomic_add(dy_ptr + offsets_y, do, mask=mask_y)
        elif OP == "sub":
            tl.atomic_add(dy_ptr + offsets_y, -do, mask=mask_y)
        elif OP == "mul":
            # y gradient = do * x
            # load x_ptr + offsets_x
            x_val = tl.load(x_ptr + offsets_x, mask=mask_x)
            tl.atomic_add(dy_ptr + offsets_y, x_val * do, mask=mask_y)
        elif OP == "div":
            # out = x / y => dy = -(x*do)/y^2
            x_val = tl.load(x_ptr + offsets_x, mask=mask_x)
            y_val = tl.load(y_ptr + offsets_y, mask=mask_y)
            partial_dy = -x_val * do / (y_val * y_val)
            tl.atomic_add(dy_ptr + offsets_y, partial_dy, mask=mask_y)