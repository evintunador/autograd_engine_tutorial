import random as r
from engine import Value

class Module: # just to make our syntax the same as pytorch's
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, input_dim):
        self.w = [Value(r.uniform(-1,1)) for _ in range(input_dim)]
        self.b = Value(0.0)

    def __call__(self, x):
        assert len(x) == len(self.w), f'mismatch between input dim {len(x)} and weight dim {len(self.w)}'
        # w * x + b
        wixi = [wi*xi for wi, xi in zip(self.w, x)]
        
        sum = Value(0.0) # for some reason sum() gives me an error so i do the addition manually
        for i in wixi: 
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
    
