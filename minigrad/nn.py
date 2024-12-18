import numpy as np
from engine import Tensor, Parameter

class Module: # just to make our syntax the same as pytorch's
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias = True):
        self.w = Parameter(np.random.normal(scale=0.02, size=(in_dim, out_dim)).astype(np.float32))
        if bias: self.b = Parameter(np.zeros((1,out_dim)).astype(np.float32))

    def __repr__(self):
        return f"Weight:\n({self.w})\nBias:\n({self.b})" if self.b else f"Weight:\n({self.w})"

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x: Tensor):
        while x.ndim > self.w.ndim:
            self.w.unsqueeze(0)
            self.b.unsqueeze(0)
        return x @ self.w + self.b if self.b else x @ self.w

class Embedding(Module):
    def __init__(self, num_classes: int, embed_dim: int):
        self.w = Parameter(np.random.normal(scale=0.02, size=(num_classes, embed_dim)).astype(np.float32))

    def __call__(self, tokens):
        assert np.issubdtype(tokens.dtype, np.dtype('int')),\
                f"input dtype should be np.uint but instead got {tokens.dtype}"
        # grab embedding assigned to each token
        return self.w[tokens]

    def __repr__(self):
        return f"Emedding:\n({self.w})"

    def parameters(self):
        return [self.w]

if __name__ == "__main__":
    b = 2
    d = 4
    v = 5
    s = 3
    
    print("---------------- test linear ----------------")
    x = Tensor([[1,2],[3,4],[5,6]])
    print(x)
    w = Linear(2,4)
    print(w)
    y = w(x)
    print(y)
    y.backward()
    print(y)
    print(w)
    print(x)
    
    print("---------------- test embedding ----------------")
    x = np.random.randint(v, size=(b,s))
    print(x)
    E = Embedding(v, d)
    print(E)
    toks = E(x)
    print(toks)
    toks.backward()
    print(toks)
    print(E)
    print(x)
    