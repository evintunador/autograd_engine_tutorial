import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit # this decorator tells Triton to compile this function into GPU code
def add_kernel(x_ptr, y_ptr,# pointers to input vectors
               output_ptr, # ptr to output vector
                    # each torch.tensor object is implicitly converted into a pointer to its first element
               n_elements, # size of vector
               BLOCK_SIZE: tl.constexpr): # number of elements each program should process
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
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements

    # load x and y from DRAM (global GPU memory) into SRAM (on-chip memory)
    # SRAM is much faster but limited in size
    # The mask ensures we don't access memory beyond the vector's end
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # perform the operation on SRAM
    # triton has its own internal definitions of all the basic ops
    output = x + y

    # write back to DRAM, being sure to mask to avoid out-of-bounds accesses
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def add_backward_kernel(
    grad_output_ptr,    # pointer to incoming gradient
    grad_self_ptr,      # pointer to first input's gradient
    grad_other_ptr,     # pointer to second input's gradient
    n_elements,         # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this program instance
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to guard memory operations
    mask = offsets < n_elements
    
    # Load incoming gradient
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    
    # For addition, both gradients are simply the output gradient
    # Load and accumulate gradient for first input
    grad_self = tl.load(grad_self_ptr + offsets, mask=mask)
    grad_self = grad_self + grad_output
    
    # Load and accumulate gradient for second input
    grad_other = tl.load(grad_other_ptr + offsets, mask=mask)
    grad_other = grad_other + grad_output
    
    # Store accumulated gradients
    tl.store(grad_self_ptr + offsets, grad_self, mask=mask)
    tl.store(grad_other_ptr + offsets, grad_other, mask=mask)