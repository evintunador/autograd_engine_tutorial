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