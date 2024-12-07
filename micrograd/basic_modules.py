import random as r
from engine import Value

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

if __name__ == "__main__":
    batch_size = 2
    vocab_len = 10
    model_dim = 4
    seq_len = 5
    num_heads = 2
    head_dim = 2

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
    
    