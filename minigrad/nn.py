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

