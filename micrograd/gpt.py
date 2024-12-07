import random as r
from engine import Value
from nn import Module, Neuron, Linear, MLP

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

class Embedding(Module):
    def __init__(self, num_classes: int, dim: int):
        self.weight = [[Value(r.uniform(-1,1)) for _ in range(dim)] 
                       for _ in range(num_classes)]

    def __call__(self, x):
        assert isinstance(x, list), "x should be a list of integers"
        assert all(isinstance(idx, int) for idx in x), "All elements in x must be integers"
        # grab embedding assigned to each token
        out = [self.weight[idx] for idx in x]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        weights_repr = "\n".join(
            f"[{', '.join(str(p) for p in row)}]" for row in self.weight
        )
        return f"Embedding with weights:\n{weights_repr}"

    def parameters(self):
        return [p for c in self.weight for p in c]

def layer_norm(x):
    '''
    Layer normalization module that only takes as input a single vector, 
    meaning you've gotta handle the tensor logic outside the call
    '''
    assert isinstance(x, list), "x should be a list of Value objects"
    assert all(isinstance(idx, Value) for idx in x), "All elements in x must be Value objects"

    n = len(x)
    # mean
    mean = Value(x[0].data / n, (x[0],)) # for some reason sum() gives me an error so i do the addition manually
    for xi in x[1:]: 
        mean = mean + (xi / n)
    # sd
    tot = (x[0] - mean)**2
    for xi in x[1:]:
        tot = tot + (xi - mean)**2
    sd = (tot / n) ** (-0.5)
    # normalization
    out = [None] * n
    for i in range(n):
        out[i] = (x[i] - mean) / sd

    return out

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

def transpose_tensor_final_dims(x):
    '''
    input: x - list of lists of .... of Value objects
    output: out - list of lists of .... of Value objects where the final two dimensions are transposed
    '''
    # Check if x has more than 1 dimension
    item = x
    dims = []
    while isinstance(item, list):
        dims.append(len(item))
        item = item[0]
    
    if len(dims) < 2:
        raise ValueError('x must have more than 1 dimension to be transposed')
        
    # Helper function to recursively apply transpose. this lets us move through a dynamic number of dimensions
    def recursive_transpose(sub_tensor):
        if isinstance(sub_tensor[0][0], list):
            return [recursive_transpose(sub_part) for sub_part in sub_tensor]
        else: # Base case: when the tensor has only two dimensions
            return transpose_2dim(sub_tensor)

    # Recursively apply the transpose operation to the final two dimensions
    return recursive_transpose(x)

def tensor_transpose_arbitrary(x, dims: tuple):
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

def tensor_entry_wise_add(x, y):
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

if __name__ == "__main__":
    batch_size = 2
    seq_len = 3
    model_dim = 4

    ### test pretty tensor printer
    nested_list = [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [7, 8],
            [9, 10, 11],
            [12]
        ]
    ]
    pretty_print_tensor(nested_list)
    print('\n')
    print('\n')

    ### test embedding
    E = Embedding(vocab_len, model_dim)
    print(E)
    print('\n')
    x = E([1,2,3])
    pretty_print_tensor(x)
    print('\n')
    print('\n')

    ### test layernorm
    # single vector
    x = [Value(r.uniform(-1,1)) for _ in range(model_dim)]
    print(x)
    y = layer_norm(x)
    print(y)
    print('\n')
    print('\n')
    # tensor
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = [[layer_norm(xi) for xi in seq] for seq in x]
    pretty_print_tensor(y)
    print('\n')
    print('\n')

    ### test transpose
    # 2-dim
    x = [[Value(r.uniform(-1,1)) for _ in range(model_dim)]
        for _ in range(seq_len)]
    pretty_print_tensor(x)
    print('\n')
    y = transpose_matrix(x)
    pretty_print_tensor(y)
    # more than 2 dims, but only last 2 dims are to be transposed
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = transpose_tensor_final_dims(x)
    pretty_print_tensor(y)
    # transpose any arbitrary combination of dimensions
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
      for _ in range(seq_len)]
     for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = tensor_transpose_arbitrary(x, dims=(0, 2))
    pretty_print_tensor(y)

    ### test entry-wise addition
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    y = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
              for _ in range(seq_len)]
             for _ in range(batch_size)]
    pretty_print_tensor(y)
    z = tensor_entry_wise_add(x, y)
    pretty_print_tensor(z)
