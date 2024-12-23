import numpy as np
from engine import Tensor, Parameter

class Module: # just to make our syntax the same as pytorch's
    def __init__(self):
        self.training = True

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

    def train(self, mode=True):
        self.training = mode
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def children(self):
        # returns a list or iterator of immediate children modules
        # this will be useful for recursively setting training mode
        return []

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias = True):
        self.w = Parameter(np.random.normal(scale=0.02, size=(in_dim, out_dim)).astype(np.float32))
        if bias: self.b = Parameter(np.zeros((1,out_dim)).astype(np.float32))

    def __call__(self, x: Tensor):
        while x.ndim > self.w.ndim:
            self.w.unsqueeze(0)
            self.b.unsqueeze(0)
        return x @ self.w + self.b if self.b else x @ self.w

    def __repr__(self):
        return f"Weight:\n({self.w})\nBias:\n({self.b})" if self.b else f"Weight:\n({self.w})"

    def parameters(self):
        out = [self.w]
        if self.b: out.append(self.b)
        return out

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

class Dropout(Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def __call__(self, x: Tensor):
        if not self.training:
            return x
        
        # create a mask of the same shape as x
        # with probability (1 - p) for each element to be 1, and p to be 0
        mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
        mask = Tensor(mask, requires_grad=False) # mask doesn't need grad

        return x * mask

    def __repr__(self):
        return f"Dropout(p={self.p})"
    
class LayerNorm(Module):
    def __init__(self, dim: int, elementwise_affine: bool = True, bias: bool = True):
        self.dim = dim
        self.eps = 1e-5

        if elementwise_affine: 
            # TODO: should this be np.ones???
            self.affine = Parameter(np.random.normal(scale=0.02, size=dim).astype(np.float32))
            if bias: # bias will only be created if elementwise_affine is also created
                self.bias = Parameter(np.zeros(dim).astype(np.float32))

    def __call__(self, x):
        assert self.dim == x.shape[-1]
        # normalize
        mean = x.mean(keepdim=True)
        var = x.var(keepdim=True)
        out = (x - mean) / (var + self.eps) ** 0.5
        # affine transformation
        if self.affine:
            out = out * self.affine
            if self.bias:
                out = out + self.bias
        return out

    def __repr__(self):
        out = "LayerNorm"
        if self.affine: out += f"\nElement-wise affine:\n({self.affine})"
        if self.bias: out += f"\nBias:\n({self.bias})"
        return out

    def parameters(self):
        out = []
        if self.affine: out += [self.affine]
        if self.bias: out += [self.bias]
        return out

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
    
    print("---------------- test dropout ----------------")
    input_tensor = Tensor(np.random.randn(2, 3, 4))
    dropout = Dropout(p=0.5)
    print("----- training mode -----")
    dropout.train()
    output_train = dropout(input_tensor)
    print("Input:", input_tensor.data)
    print("Output (Train):", output_train.data)

    print("----- evaluation mode -----")
    dropout.eval()
    output_eval = dropout(input_tensor)
    print("Input:", input_tensor.data)
    print("Output (Eval):", output_eval.data)

    print("---------------- test layernorm ----------------")
    x = Tensor(np.random.randn(2, 3, 4))
    print(x)
    ln = LayerNorm(x.shape[-1])
    print(ln)
    y = ln(x)
    print(y)
    print(x)
    