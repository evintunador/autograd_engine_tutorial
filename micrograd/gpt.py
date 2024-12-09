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

class MLP(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.up = Linear(input_dim, hidden_dim)
        self.down = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        up = self.up(x)
        act = [i.relu() for i in up]
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

    def __repr__(self):
        weights_repr = "\n".join(
            f"[{', '.join(str(p) for p in row)}]" for row in self.mask
        )
        return f"Causal self-attention mask:\n{weights_repr}"

if __name__ == "__main__":
    batch_size = 2
    vocab_len = 10
    model_dim = 4
    seq_len = 5
    num_heads = 2
    head_dim = 2

    print('\n\n-------------- test layernorm on a single vector -------------')
    x = [Value(r.uniform(-1,1)) for _ in range(model_dim)]
    print(x)
    y = layer_norm(x)
    print(y)
    # tensor
    print('\n\n-------------- test layernorm on a tensor -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = vector_wise_apply(layer_norm, x)
    pretty_print_tensor(y)

    print('\n\n-------------- test MLP on a vector -------------')
    x = [Value(r.uniform(-1,1)) for _ in range(model_dim)]
    print(x)
    mlp = MLP(model_dim, 4 * model_dim, model_dim)
    y = mlp(x)
    print(y)
    # tensor
    print('\n\n-------------- test MLP on a tensor -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = vector_wise_apply(mlp, x)
    pretty_print_tensor(y)

    print('\n\n-------------- test causal self-attention mask -------------')
    mask = Mask(max_seq_len)
    print(mask)
    pretty_print_tensor(mask(seq_len))
    pretty_print_tensor(mask(seq_len - 1))