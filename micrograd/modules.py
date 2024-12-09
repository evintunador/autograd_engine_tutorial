import random as r
from engine import Value
from ops import *

class Module: # just to make our syntax the same as pytorch's
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

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

class Neuron(Module):
    def __init__(self, input_dim):
        self.w = [Value(r.uniform(-1,1)) for _ in range(input_dim)]
        self.b = Value(0.0)

    def __call__(self, x):
        assert len(x) == len(self.w), f'mismatch between input dim {len(x)} and weight dim {len(self.w)}'
        # w * x + b
        wixi = [wi*xi for wi, xi in zip(self.w, x)]
        
        sum = wixi[0] # for some reason sum() gives me an error so i do the addition manually
        for i in wixi[1:]: 
            sum = sum + i
        
        act = sum + self.b
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self.neurons = [Neuron(input_dim) for _ in range(output_dim)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

if __name__ == "__main__":
    batch_size = 2
    vocab_len = 10
    model_dim = 8
    max_seq_len = 5
    seq_len = 3
    num_heads = 2
    head_dim = model_dim // num_heads

    print('\n\n-------------- test embedding on a single input sequence -------------')
    E = Embedding(vocab_len, model_dim)
    print(E)
    print('\n')
    x = E([1,2,3])
    pretty_print_tensor(x)
    print('\n\n-------------- test embedding on a batch of input sequences -------------')
    E = Embedding(vocab_len, model_dim)
    print(E)
    print('\n')
    input_tokens = [[r.randint(0,vocab_len-1) for _ in range(seq_len)]
                    for _ in range(batch_size)]
    x = vector_wise_apply(E, input_tokens)
    pretty_print_tensor(x)
    
    print('\n\n-------------- test linear layer on a vector -------------')
    x = [Value(r.uniform(-1,1)) for _ in range(model_dim)]
    print(x)
    w = Linear(model_dim, head_dim)
    y = w(x)
    print(y)
    print('\n\n-------------- test linear layer on a tensor -------------')
    x = [[[Value(r.uniform(-1,1)) for _ in range(model_dim)]
          for _ in range(seq_len)]
         for _ in range(batch_size)]
    pretty_print_tensor(x)
    print('\n')
    y = vector_wise_apply(w, x)
    pretty_print_tensor(y)