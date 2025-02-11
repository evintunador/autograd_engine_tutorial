import torch
import triton
import triton.language as tl
import math

from engine import TritonTensor, Parameter
#from kernels import ?

class Module: # just to make our syntax the same as pytorch's
    def __init__(self):
        self.training = True

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
    
    def parameters(self):
        '''
        default parameter-yielding method
        modules which actually have parameters should overwrite this method
        '''
        out = []
        for child in self.children():
            if child.parameters() is not None:
                out += child.parameters()
        return out if out else None

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias = True, device = None):
        super().__init__()
        self.weight = Parameter(
            data = torch.randn(size=(in_features, out_features)) * math.sqrt(1/in_features),
            device = device
        )
        if bias: 
            self.bias = Parameter(
                torch.randn((out_features,)) * math.sqrt(1/in_features),
                device = device
            )
        else:
            self.bias = None

    def __call__(self, x: TritonTensor):
        # First compute matmul
        out = x @ self.weight
        # Then add bias if it exists
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"nn.Linear\nWeight:\n{self.weight}" + f"\nBias:\n{self.bias}" if self.bias is not None else ""
