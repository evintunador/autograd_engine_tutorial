import random as r
from engine import Value

def get_shape(tensor):
    '''
    finds the shape of a list of lists of... of lists of Value objects
    '''
    assert isinstance(tensor, list)
    if isinstance(tensor[0], list):
        return [len(tensor)] + get_shape(tensor[0])
    else:
        return [len(tensor)]
        
def pretty_tensor_print(tensor, indent=0):
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
            pretty_tensor_print(item, indent + 2)
        print(" " * indent + "]")

def vector_wise_apply(function, tensor, *args, **kwargs):
    '''
    applies the input function to the tensor vector-wise
    
    inputs: 
        function - a function meant to be applied to a list of Value objects
        tensor - list of lists of .... of Value objects
        *args & **kwargs - any extra arguments to be passed into function
    output: list of lists of .... of Value objects
    '''
    assert isinstance(tensor, list), "input must be at least a vector (aka a list of Value objects)"
    if isinstance(tensor[0], list):
        return [vector_wise_apply(function, sub_tensor, *args, **kwargs) for sub_tensor in tensor]
    else: # base case: the final vector dimension
        return function(tensor, *args, **kwargs)

def matrix_wise_apply(function, tensor, *args, **kwargs):
    '''
    applies the input function to the tensor matrix-wise
    
    inputs: 
        function - a function meant to be applied to a list of lists of Value objects
        tensor - list of lists of .... of Value objects
        *args & **kwargs - any extra arguments to be passed into function
    output: list of lists of .... of Value objects
    '''
    assert isinstance(tensor[0], list), "input must be at least a matrix (aka a list of lists of Value objects)"
    if isinstance(tensor[0][0], list):
        return [matrix_wise_apply(function, sub_tensor, *args, **kwargs) for sub_tensor in tensor]
    else: # base case: the final two dimensions which compose a matrix
        return function(tensor, *args, **kwargs)

def split_dim(vec, dims):
    '''
    splits input vector of shape (dims[0] + dims[1]) into matrix of shape (dims[0], dims[1])
    '''
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    assert len(dims) == 2
    assert len(vec) == dims[0] * dims[1], f'vector length {len(vec)} must match desired reshape ({dims[0]},{dims[1]})'

    mat = [[None]*dims[1] for _ in range(dims[0])]
    for i in range(dims[0]):
        for j in range(dims[1]):
            mat[i][j] = vec[(i * dims[1]) + j]
    return mat

def flatten(mat):
    assert isinstance(mat[0], list) and isinstance(mat[0][0], Value),\
        'mat should be a matrix (AKA list of list of Value objects)'
    return [mat[i][j] for i in range(len(mat)) for j in range(len(mat[0]))]

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
    shape = get_shape(x)
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

def add_scalar_to_vec(vec, scalar):
    '''
    adds all elements in the vector x by the constant scalar
    for subtraction just input a negative number
    '''
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    assert isinstance(scalar, (int, float, Value)), f"scalar should be an int, float, or Value, but instead is {type(scalar)}"
    return [x + scalar for x in vec]

def mult_vec_by_scalar(vec, scalar):
    '''
    multiplies all elements in the vector x by the constant scalar
    for division just input a fraction
    '''
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    assert isinstance(scalar, (int, float, Value)), f"scalar should be an int, float, or Value, but instead is {type(scalar)}"
    return [x * scalar for x in vec]

def entry_wise_add(x, y):
    '''
    entry-wise addition function that does not support broadcasting, aka inputs must be same shapes
    
    inputs: 
        x - list of lists of .... of Value objects
        y - list of lists of .... of Value objects of the same shape as x
    output: 
        out - list of lists of .... of Value objects of the same shape as x and y
    '''
    assert get_shape(x) == get_shape(y), f"tensors must have matching dimensions but instead have {get_shape(x)} and {get_shape(y)}"
        
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
    assert get_shape(x) == get_shape(y), f"tensors must have matching dimensions but instead have {get_shape(x)} and {get_shape(y)}"
        
    # helper function to recursively apply entry-wise mult. this lets us move through a dynamic number of dimensions
    def recursive_entry_wise_mult(sub_tensor_x, sub_tensor_y):
        if isinstance(sub_tensor_x[0], list):
            return [recursive_entry_wise_mult(sub_part_x, sub_part_y) for sub_part_x, sub_part_y in zip(sub_tensor_x, sub_tensor_y)]
        else: # base case: the final vector dimension
            return [xi * yi for xi, yi in zip(sub_tensor_x, sub_tensor_y)]

    # Recursively apply the entry-wise multiplication operation to the final dimension
    return recursive_entry_wise_mult(x, y)

def sum(vec):
    '''
    sums up all values in a vector
    returns a single Value object, so if it's being called by vector_wise_apply that means it removes the last dimension in the process
    '''
    assert isinstance(vec, list) and isinstance(vec[0], (float, int, Value))
    tot = vec[0]
    for x in vec[1:]:
        tot = tot + x
    return tot

def tensor_matmul(x, y):
    """
    Perform a matrix multiplication over the last two dimensions of two tensors (lists of lists of ... of Value objects).

    Shape constraints:
    - Let x have shape: [..., M, N]
    - Let y have shape: [..., N, P]
    
    The leading dimensions (everything except the last two) must match.
    The result will have shape: [..., M, P].
    """
    x_shape, y_shape = get_shape(x), get_shape(y)

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

def relu(vec):
    '''
    applies Rectified Linear Unit to all elements in the vector
    '''
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    return [x.relu() for x in vec]

def exp(vec):
    '''
    exponentiates  all elements in the vector
    '''
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    return [x.exp() for x in vec]

def softmax(vec):
    '''
    performs the softmax operation across the input vector, giving us a list of probaiblities
    '''
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    # Subtract the max value from each element for numerical stability
    max_val = max(x.data for x in vec)
    vec_shifted = [x - max_val for x in vec]
    # perform entry-wise exponentiation
    vec_exp = exp(vec_shifted)
    # calculate the sum of the newly exponentiated vector
    sum_vec_exp = sum(vec_exp)
    # return a vector of each exponentiated entry divided by the sum
    return [x / sum_vec_exp for x in vec_exp]

def log(vec):
    '''
    takes the natural log of all elements in the vector
    '''
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    return [x.log() for x in vec]

def dropout(vec, rate = 0.1):
    assert isinstance(vec, list), "vec should be a list of Value objects"
    assert all(isinstance(x, Value) for x in vec), "All elements in vec must be Value objects"
    assert 0 <= rate < 1, f"dropout rate must be scalar value in [0,1) but instead is print({rate})"
    return [xi if r.uniform(0,1) > rate else Value(0.) for xi in vec]

if __name__ == "__main__":
    batch_size = 2
    vocab_len = 10
    model_dim = 8
    max_seq_len = 5
    seq_len = 3
    num_heads = 2
    head_dim = model_dim // num_heads

    print('\n\n-------------- test get_shape -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    print('\n')
    print(get_shape(x))
    
    print('-------------- test pretty tensor printer -------------')
    nested_list = [[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]]
    pretty_tensor_print(nested_list)

    print('\n\n-------------- test split_dim on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    y = split_dim(x, dims=(num_heads, head_dim))
    pretty_tensor_print(y)
    print('\n\n-------------- test split_dim on a tensor -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    print('\n')
    y = vector_wise_apply(split_dim, tensor = x, dims=(num_heads, head_dim))
    pretty_tensor_print(y)

    print('\n\n-------------- test flatten on a tensor -------------')
    x = [[[[Value(r.gauss(0,0.02)) for _ in range(head_dim)] for _ in range(num_heads)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    y = matrix_wise_apply(flatten, x)
    print(get_shape(y))
    pretty_tensor_print(y)

    print('\n\n-------------- test transpose on a single matrix -------------')
    x = [[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)]
    pretty_tensor_print(x)
    print('\n')
    y = transpose_matrix(x)
    pretty_tensor_print(y)
    print('\n\n-------------- test transpose on tensor of more than 2 dims, but only last 2 dims are to be transposed -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    print('\n')
    y = matrix_wise_apply(transpose_matrix, x)
    pretty_tensor_print(y)
    print('\n\n-------------- test transpose of any arbitrary combination of dimensions -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    print('\n')
    dims=(0, 2)
    print(f'dims to be transposed: {dims}')
    y = transpose(x, dims)
    pretty_tensor_print(y)

    print('\n\n-------------- test entry-wise add by a single scalar on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    y = add_scalar_to_vec(x, 100.)
    print(y)
    
    print('\n\n-------------- test entry-wise mult by a single scalar on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    y = mult_vec_by_scalar(x, 2.)
    print(y)
    
    print('\n\n-------------- test entry-wise addition for tensors -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    y = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(y)
    print('\n')
    z = entry_wise_add(x, y)
    pretty_tensor_print(z)
    
    print('\n\n-------------- test entry-wise multiplication for tensors -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    y = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(y)
    print('\n')
    z = entry_wise_mult(x, y)
    pretty_tensor_print(z)

    print('\n\n-------------- test sum on a tensor -------------')
    x = [[[Value(r.gauss(0,0.02)).exp() for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    y = vector_wise_apply(sum, x)
    pretty_tensor_print(y)

    print('\n\n-------------- test tensor matmul -------------')
    q = [[[[Value(r.gauss(0,0.02)) for _ in range(head_dim)] for _ in range(seq_len)] for _ in range(num_heads)] for _ in range(batch_size)]
    pretty_tensor_print(q)
    print('\n')
    k = [[[[Value(r.gauss(0,0.02)) for _ in range(head_dim)] for _ in range(seq_len)] for _ in range(num_heads)] for _ in range(batch_size)]
    v = [[[[Value(r.gauss(0,0.02)) for _ in range(head_dim)] for _ in range(seq_len)] for _ in range(num_heads)] for _ in range(batch_size)]
    k_transpose = transpose(k, dims=(2,3))
    pretty_tensor_print(k_transpose)
    print('\n')
    logits = tensor_matmul(q, k_transpose)
    pretty_tensor_print(logits)
    print('\n')
    out = tensor_matmul(logits, v)
    pretty_tensor_print(out)

    print('\n\n-------------- test relu on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    y = relu(x)
    print(y)

    print('\n\n-------------- test exp on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    y = exp(x)
    print(y)

    print('\n\n-------------- test softmax on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    y = softmax(x)
    print(y)

    print('\n\n-------------- test log on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    y = log(softmax(x))
    print(y)
    print('\n\n-------------- test log on a tensor -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    y = vector_wise_apply(log, vector_wise_apply(softmax, x))
    pretty_tensor_print(y)

    print('\n\n-------------- test dropout on a tensor -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    rate = 0.5
    y = vector_wise_apply(dropout, x, rate)
    pretty_tensor_print(y)