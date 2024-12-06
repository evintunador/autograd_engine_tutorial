import random as r
from engine import Value
from nn import Module, Neuron, Linear, MLP

class Embedding(Module):
    def __init__(self, num_classes: int, dim: int):
        self.weight = [[Value(r.uniform(-1,1)) for _ in range(dim)] 
                       for _ in range(num_classes)]

    def __call__(self, x):
        # x should be list of integers
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