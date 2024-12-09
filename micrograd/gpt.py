import random as r
from modules import *
from ops import *

def layer_norm(x):
    '''
    Layer normalization module that only takes as input a single vector, 
    meaning you've gotta handle the tensor logic outside the call
    we'll do that using vector_wise_apply
    '''
    assert isinstance(x, list), "x should be a list of Value objects"
    assert all(isinstance(idx, Value) for idx in x), "All elements in x must be Value objects"

    n = len(x)
    # mean
    mean = x[0] / n # for some reason sum() gives me an error so i do the addition manually
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

class MultiLayerPerceptron(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.up = Linear(input_dim, hidden_dim)
        self.down = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        up = self.up(x)
        act = relu(up)
        down = self.down(act)
        return down

    def parameters(self):
        return [p for p in self.up.parameters()] + [p for p in self.down.parameters()]

    def __repr__(self):
        return f"MLP of [{self.up}, {self.down}]"

class Mask(Module):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.mask = [ [1] * (i + 1) + [0] * (max_seq_len - i - 1) for i in range(max_seq_len)]

    def __call__(self, seq_len):
        assert 0 < seq_len <= self.max_seq_len, f'seq_len {seq_len} must be less than max_seq_len {max_seq_len}'
        return [[i for i in row[:seq_len]] for row in self.mask[:seq_len]]

    def masked_fill(self, matrix, val = float('-inf')):
        mat_shape  = get_shape(matrix)
        assert mat_shape[0] == mat_shape[1], f"masked_fill requires input to be square matrix but instead got shape {mat_shape}"
        mask = self(len(matrix))
        return [[matrix[i][j] if mask[i][j] else Value(val) for j in range(mat_shape[1])] for i in range(mat_shape[0])]

    def __repr__(self):
        weights_repr = "\n".join(
            f"[{', '.join(str(p) for p in row)}]" for row in self.mask
        )
        return f"Causal self-attention mask:\n{weights_repr}"

class MultiHeadSelfAttention(Module):
    def __init__(self, model_dim, num_heads, head_dim, max_seq_len):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        self.Wq = Linear(model_dim, num_heads * head_dim)
        self.Wk = Linear(model_dim, num_heads * head_dim)
        self.Wv = Linear(model_dim, num_heads * head_dim)
        
        self.scale = head_dim ** -0.5

        self.mask = Mask(max_seq_len)
        
        self.Wo = Linear(num_heads * head_dim, model_dim)
    
    def __call__(self, x):
        assert isinstance(x, list) and isinstance(x[0], list) and isinstance(x[0][0], list) and isinstance(x[0][0][0], Value),\
            "input to MHSA mechanism must be tensor of ndim==3 for (batch_size, seq_len, model_dim)"
        batch_size, seq_len, model_dim = tuple(get_shape(x))
        assert self.model_dim == model_dim,\
            f"input final dimension {model_dim} must equal MHSA mechanism's given model_dim value at initialization of {self.model_dim}"
        
        # apply query, key, and value projections to our input
        q = vector_wise_apply(self.Wq, x) # shape (batch_size, seq_len, num_heads * head_dim)
        k = vector_wise_apply(self.Wk, x) # Linear object is meant to take in a single vector, so we use vector_wise_apply
        v = vector_wise_apply(self.Wv, x)

        # split apart heads
        q = vector_wise_apply(split_dim, q, dims=(self.num_heads, self.head_dim)) # shape (batch_size, seq_len, num_heads, head_dim)
        k = vector_wise_apply(split_dim, k, dims=(self.num_heads, self.head_dim))
        v = vector_wise_apply(split_dim, v, dims=(self.num_heads, self.head_dim))

        # transpose to put seq_len in the path of the matmul for our attention computation
        q = transpose(q, (1,2)) # shape (batch_size, num_heads, seq_len, head_dim)
        k = transpose(k, (1,2))
        v = transpose(v, (1,2))

        # get keys ready for attention computation
        k_t = transpose(k, (2,3)) # shape (batch_size, num_heads, head_dim, seq_len)
        # compute attention logits
        logits = tensor_matmul(q, k_t) # shape (batch_size, num_heads, seq_len, seq_len)
        # scale logits
        scaled_logits = vector_wise_apply(mult_vec_by_scalar, logits, self.scale)
        # apply mask
        masked_logits = matrix_wise_apply(self.mask.masked_fill, scaled_logits)
        # turn the logits into probability scores
        scores = vector_wise_apply(softmax, masked_logits)

        # use scores to select from values
        output_values = tensor_matmul(scores, v) # shape (batch_size, num_heads, seq_len, head_dim)
        # rearrange back to be of size model_dim
        output_values = transpose(output_values, (1,2)) # shape (batch_size, seq_len, num_heads, head_dim)
        output_values = matrix_wise_apply(flatten, output_values) # shape (batch_size, seq_len, num_heads * head_dim)

        # mix output values of each head together
        return vector_wise_apply(self.Wo, output_values) # shape (batch_size, seq_len, model_dim)

class ResidualLayer(Module):
    def __init__(self, model_dim, num_heads, head_dim, max_seq_len, mlp_mult):
        self.mhsa = MultiHeadSelfAttention(model_dim, num_heads, head_dim, max_seq_len)
        self.mlp = MultiLayerPerceptron(model_dim, mlp_mult * model_dim, model_dim)

    def __call__(self, x):
        x_normed = vector_wise_apply(layer_norm, x)
        x_mhsa = self.mhsa(x_normed)
        x_2 = entry_wise_add(x, x_mhsa)
        x_2_normed = vector_wise_apply(layer_norm, x_2)
        x_mlp = vector_wise_apply(self.mlp, x_2_normed)
        return entry_wise_add(x_2, x_mlp)

if __name__ == "__main__":
    batch_size = 2
    vocab_len = 10
    model_dim = 8
    max_seq_len = 5
    seq_len = 3
    num_heads = 2
    head_dim = model_dim // num_heads
    mlp_mult = 4

    print('\n\n-------------- test layernorm on a single vector -------------')
    x = [Value(r.uniform(-1,1)) for _ in range(model_dim)]
    print(x)
    y = layer_norm(x)
    print(y)

    print('\n\n-------------- test MLP on a vector -------------')
    x = [Value(r.uniform(-1,1)) for _ in range(model_dim)]
    print(x)
    mlp = MultiLayerPerceptron(model_dim, 4 * model_dim, model_dim)
    y = mlp(x)
    print(y)

    print('\n\n-------------- test causal self-attention mask -------------')
    mask = Mask(max_seq_len)
    print(mask)
    pretty_tensor_print(mask(seq_len))
    pretty_tensor_print(mask(seq_len - 1))
    x = [[[[Value(r.uniform(-1,1)) for _ in range(seq_len)] for _ in range(seq_len)] for _ in range(num_heads)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    mask = Mask(max_seq_len)
    y = matrix_wise_apply(mask.masked_fill, x)
    pretty_tensor_print(y)

    print('\n\n-------------- test causal multi-head self-attention mechanism -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    print(get_shape(x))
    mhsa = MultiHeadSelfAttention(model_dim, num_heads, head_dim, max_seq_len)
    y = mhsa(x)
    print(get_shape(y))

    print('\n\n-------------- test residual layer -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    print(get_shape(x))
    layer = ResidualLayer(model_dim, num_heads, head_dim, max_seq_len, mlp_mult)
    y = layer(x)
    print(get_shape(y))