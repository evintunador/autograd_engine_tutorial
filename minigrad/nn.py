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
        self.weight = Tensor(np.random.normal(loc=0.0, scale=0.02, size=(input_dim, output_dim)).astype(np.float32))
        if bias: self.bias = Tensor(np.random.normal(loc=0.0, scale=0.02, size=(1,output_dim)).astype(np.float32))

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
        self.weight = Tensor(np.random.normal(loc=0.0, scale=0.02, size=(num_classes, embed_dim)).astype(np.float32))

    def __call__(self, tokens):
        assert np.issubdtype(tokens.dtype, np.dtype('uint')),\
                f"input dtype should be np.uint but instead got {tokens.dtype}"
        # grab embedding assigned to each token
        return self.weight[tokens]

    def __repr__(self):
        return f"Emedding:\n({self.weight})"

    def parameters(self):
        return [self.weight]