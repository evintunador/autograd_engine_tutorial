import numpy as np
from engine import Tensor

class Module: # just to make our syntax the same as pytorch's
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int, bias = True):
        self.weight = Tensor(np.random.normal(loc=0.0, scale=0.2, size=(input_dim, output_dim)))
        if bias: self.bias = Tensor(np.random.normal(loc=0.0, scale=0.2, size=(1,output_dim)))

    def __repr__(self):
        return f"Weight:\n({self.weight})\nBias:\n({self.bias})" if self.bias else f"Weight:\n({self.weight})"

    def parameters(self):
        return [self.weight, self.bias]

    def __call__(self, x: Tensor):
        while x.ndim > self.weight.ndim:
            self.weight.unsqueeze(0)
            self.bias.unsqueeze(0)
        return x @ self.weight + self.bias if self.bias else x @ self.weight

class Embedding(Module):
    def __init__(self, num_classes: int, embed_dim: int):
        self.weight = Tensor(np.random.normal(loc=0.0, scale=0.2, size=(num_classes, embed_dim)))

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
        return [p for row in self.weight for p in row]