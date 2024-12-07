import random as r
from engine import Value
#from nn import Module, Neuron, Linear, MLP

def pretty_print_tensor(tensor, indent=0):
    """
    Pretty print a nested list (tensor-like structure).
    
    Parameters:
        tensor (list): The nested list to print.
        indent (int): Current indentation level (used for recursive calls).
    """
    if not isinstance(tensor, list):
        print(" " * indent + str(tensor))
        return

    if all(not isinstance(item, list) for item in tensor):
        print(" " * indent + "[" + ", ".join(map(str, tensor)) + "]")
    else:
        print(" " * indent + "[")
        for item in tensor:
            pretty_print_tensor(item, indent + 2)
        print(" " * indent + "]")

def vector_wise_apply(function, x):
    '''
    applies the input function to the tensor vector-wise
    
    inputs: 
        function - a function meant to be applied to a list of Value objects
        x - list of lists of .... of Value objects
    output: 
        out - list of lists of .... of Value objects
    '''
    assert isinstance(x, list), "input must be at least a vector (aka a list of Value objects)"
    if isinstance(x[0], list):
        return [vector_wise_apply(function, sub_x) for sub_x in x]
    else: # base case: the final vector dimension
        return function(x)

def matrix_wise_apply(function, x):
    '''
    applies the input function to the tensor matrix-wise
    
    inputs: 
        function - a function meant to be applied to a list of lists of Value objects
        x - list of lists of .... of Value objects
    output: 
        out - list of lists of .... of Value objects
    '''
    assert isinstance(x[0], list), "input must be at least a matrix (aka a list of lists of Value objects)"
    if isinstance(x[0][0], list):
        return [matrix_wise_apply(function, sub_x) for sub_x in x]
    else: # base case: the final two dimensions which compose a matrix
        return function(x)

def transpose_matrix(x):
    '''
    input: x - list of lists of Value objects where first list is length m and second is length n
    output: new_matrix - list of lists of Value objects where first list is length n and second is length m
    '''
    m, n = len(x), len(x[0])
    new_matrix = [[None] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            new_matrix[j][i] = x[i][j]
    return new_matrix

def transpose(x, dims: tuple):
    """
    Transpose any arbitrary two dimensions of a nested list of Value objects
    
    Inputs:
        x : nested list of Value objects
        dims : tuple - The two dimension indices to transpose
        
    out : nested list of Value objects with the specified dimensions swapped.
    """
    if dims[0] == dims[1]:
        # No change if the dimensions are the same
        return x
    
    # Get the shape of the tensor
    shape = []
    pointer = x[:]
    while isinstance(pointer, list):
        shape.append(len(pointer))
        pointer = pointer[0] if len(pointer) > 0 else None
        if pointer is None:
            break
    ndims = len(shape)

    assert ndims >= 0, "x must have at least 2 dimensions to transpose."
    assert 0 <= dims[0] < ndims and 0 <= dims[1] < ndims, f"Dimension indices {dims} out of range."
    
    # Create the shape of the new tensor by swapping the two specified dimensions
    new_shape = shape[:]
    new_shape[dims[0]], new_shape[dims[1]] = new_shape[dims[1]], new_shape[dims[0]]

    # Create the output tensor with the new shape, initialized with None
    def create_nested_list(shape):
        if len(shape) == 1:
            return [None] * shape[0]
        else:
            return [create_nested_list(shape[1:]) for _ in range(shape[0])]
    out = create_nested_list(new_shape)

    # We will iterate over all possible indices of the original tensor and 
    # place them into the correct position in the new tensor.
    # For every old coordinate, the new coordinate is just the old coordinate
    # with dims[0] and dims[1] swapped.

    # A helper function to recursively iterate over all indices
    def recurse_indices(current_shape, current_idx=[]):
        #print(current_shape, current_idx)
        if len(current_shape) == 0:
            yield current_idx
        else:
            for i in range(current_shape[0]):
                yield from recurse_indices(current_shape[1:], current_idx + [i])

    # Get an element from nested list by a list of indices
    def nested_get(pointer, idxs):
        for i in idxs:
            pointer = pointer[i]
        return pointer

    # Set an element in nested list by a list of indices
    def nested_set(pointer, idxs, value):
        for i in idxs[:-1]:
            pointer = pointer[i]
        pointer[idxs[-1]] = value

    # Fill the new tensor with transposed elements
    for old_idx in recurse_indices(shape):
        #print(old_idx)
        # Compute the new index by swapping dims[0] and dims[1]
        new_idx = list(old_idx)
        new_idx[dims[0]], new_idx[dims[1]] = new_idx[dims[1]], new_idx[dims[0]]
        #print(new_idx)
        value = nested_get(x, old_idx)
        nested_set(out, new_idx, value)

    return out

def entry_wise_add(x, y):
    '''
    entry-wise addition function that does not support broadcasting, aka inputs must be same shapes
    
    inputs: 
        x - list of lists of .... of Value objects
        y - list of lists of .... of Value objects of the same shape as x
    output: 
        out - list of lists of .... of Value objects of the same shape as x and y
    '''
    # Check if x and y have same dimensions
    itemx, itemy = x, y
    dimsx, dimsy = [], []
    while isinstance(itemx, list):
        dimsx.append(len(itemx))
        itemx = itemx[0]
    while isinstance(itemy, list):
        dimsy.append(len(itemy))
        itemy = itemy[0]
    assert dimsx == dimsy, f"tensors must have matching dimensions but instead have {dimsx} and {dimsy}"
        
    # helper function to recursively apply entry-wise add. this lets us move through a dynamic number of dimensions
    def recursive_entry_wise_add(sub_tensor_x, sub_tensor_y):
        if isinstance(sub_tensor_x[0], list):
            return [recursive_entry_wise_add(sub_part_x, sub_part_y) for sub_part_x, sub_part_y in zip(sub_tensor_x, sub_tensor_y)]
        else: # base case: the final vector dimension
            return [xi + yi for xi, yi in zip(sub_tensor_x, sub_tensor_y)]

    # Recursively apply the entry-wise addition operation to the final dimension
    return recursive_entry_wise_add(x, y)

def entry_wise_mult(x, y):
    '''
    entry-wise multiplication function that does not support broadcasting, aka inputs must be same shapes
    
    inputs: 
        x - list of lists of .... of Value objects
        y - list of lists of .... of Value objects of the same shape as x
    output: 
        out - list of lists of .... of Value objects of the same shape as x and y
    '''
    # Check if x and y have same dimensions
    itemx, itemy = x, y
    dimsx, dimsy = [], []
    while isinstance(itemx, list):
        dimsx.append(len(itemx))
        itemx = itemx[0]
    while isinstance(itemy, list):
        dimsy.append(len(itemy))
        itemy = itemy[0]
    assert dimsx == dimsy, f"tensors must have matching dimensions but instead have {dimsx} and {dimsy}"
        
    # helper function to recursively apply entry-wise add. this lets us move through a dynamic number of dimensions
    def recursive_entry_wise_mult(sub_tensor_x, sub_tensor_y):
        if isinstance(sub_tensor_x[0], list):
            return [recursive_entry_wise_mult(sub_part_x, sub_part_y) for sub_part_x, sub_part_y in zip(sub_tensor_x, sub_tensor_y)]
        else: # base case: the final vector dimension
            return [xi * yi for xi, yi in zip(sub_tensor_x, sub_tensor_y)]

    # Recursively apply the entry-wise addition operation to the final dimension
    return recursive_entry_wise_mult(x, y)

def tensor_matmul(x, y):
    """
    Perform a matrix multiplication over the last two dimensions of two tensors (lists of lists of ... of Value objects).

    Shape constraints:
    - Let x have shape: [..., M, N]
    - Let y have shape: [..., N, P]
    
    The leading dimensions (everything except the last two) must match.
    The result will have shape: [..., M, P].
    """

    # Get the shape of a nested list
    def get_shape(tensor):
        shape = []
        pointer = tensor
        while isinstance(pointer, list) and len(pointer) > 0:
            shape.append(len(pointer))
            pointer = pointer[0]
        return shape

    x_shape = get_shape(x)
    y_shape = get_shape(y)

    assert len(x_shape) >= 2 and len(y_shape) >= 2, "tensors must have at least 2D for matrix multiplication."
    assert x_shape[-1] == y_shape[-2], f"Inner dimensions must match: got {x_shape[-1]} and {y_shape[-2]}."
    if len(x_shape) > 2 or len(y_shape) > 2:
        # For higher dimensional tensors, the leading dimensions (all but the last two) must be identical
        assert x_shape[:-2] == y_shape[:-2], f"Leading dimensions must match, got {x_shape[:-2]} and {y_shape[:-2]}"
    
    M, N = x_shape[-2], x_shape[-1]
    P = y_shape[-1]

    # Function to multiply two 2D matrices of Value objects
    def matmul_2d(a, b):
        '''
        MxN @ NxP -> MxP
        '''
        out = []
        for i in range(M):
            row = []
            for j in range(P):
                # Compute sum over k of a[i,k]*b[k,j]
                val = a[i][0] * b[0][j]
                for k in range(1, N):
                    val = val + (a[i][k] * b[k][j])
                row.append(val)
            out.append(row)
        return out

    # If we only have 2D matrices, just do matmul directly
    if len(x_shape) == 2 and len(y_shape) == 2:
        return matmul_2d(x, y)

    # For higher dimensions, we recursively apply matmul across the leading dimensions
    def recurse_mm(subx, suby):
        # If not at the final 2D level, keep recursing
        if isinstance(subx[0], list) and isinstance(subx[0][0], list):
            return [recurse_mm(x_elem, y_elem) for x_elem, y_elem in zip(subx, suby)]
        else:
            # At the 2D base case
            return matmul_2d(subx, suby)

    return recurse_mm(x, y)

if __name__ == "__main__":
    batch_size = 2
    vocab_len = 5
    model_dim = 8
    seq_len = 3
    num_heads = 2
    head_dim = 4

    ### test pretty tensor printer
    print('-------------- test pretty tensor printer -------------')
    nested_list = [[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]]
    pretty_print_tensor(nested_list)

    ### test transpose
    # 2-dim
    print('\n\n-------------- test transpose 2-dim -------------')
    x = [[Value(r.uniform(-1,1)) for _ in range(model_dim)]
        for _ in range(seq_len)]
    pretty_print_tensor(x)
    print('\n')
    y = transpose_matrix(x)
    pretty_print_tensor(y)
    # more than 2 dims, but only last 2 dims are to be transposed
    print('\n\n-------------- test transpose on tensor of more than 2 dims, but only last 2 dims are to be transposed -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = matrix_wise_apply(transpose_matrix, x)
    pretty_print_tensor(y)
    # transpose any arbitrary combination of dimensions
    print('\n\n-------------- test transpose of any arbitrary combination of dimensions -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
      for _ in range(seq_len)]
     for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = transpose(x, dims=(0, 2))
    pretty_print_tensor(y)

    ### test entry-wise addition
    print('\n\n-------------- test entry-wise addition -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    y = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
              for _ in range(seq_len)]
             for _ in range(batch_size)]
    pretty_print_tensor(y)
    print('\n')
    z = entry_wise_add(x, y)
    pretty_print_tensor(z)

    ### test entry-wise multiplication
    print('\n\n-------------- test entry-wise multiplication -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    y = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
              for _ in range(seq_len)]
             for _ in range(batch_size)]
    pretty_print_tensor(y)
    print('\n')
    z = entry_wise_mult(x, y)
    pretty_print_tensor(z)

    ### test tensor matmul
    print('\n\n-------------- test tensor matmul -------------')
    q = [[[[Value(r.uniform(-1,1)) for _ in range(head_dim)]
       for _ in range(seq_len)]
      for _ in range(num_heads)]
     for _ in range(batch_size)]
    pretty_print_tensor(q)
    print('\n')
    k = [[[[Value(r.uniform(-1,1)) for _ in range(head_dim)]
           for _ in range(seq_len)]
          for _ in range(num_heads)]
         for _ in range(batch_size)]
    v = [[[[Value(r.uniform(-1,1)) for _ in range(head_dim)]
           for _ in range(seq_len)]
          for _ in range(num_heads)]
         for _ in range(batch_size)]
    k_transpose = transpose(k, dims=(2,3))
    pretty_print_tensor(k_transpose)
    print('\n')
    logits = tensor_matmul(q, k_transpose)
    pretty_print_tensor(logits)
    print('\n')
    out = tensor_matmul(logits, v)
    pretty_print_tensor(out)