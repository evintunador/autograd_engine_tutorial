import math

class Value:
    """stores a single scalar value and its gradient"""
    
    def __init__(self, data, _children=()): 
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0 
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.3f}, grad={self.grad:.3f})"

    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            self.grad += out.grad # equivalent to multiplying by 1
            other.grad += out.grad
        out._backward = _backward 
        
        return out

    def __radd__(self, other): # so that int + Value redirects to Value + int aka __add__. r stands for reverse
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other): 
        return self + (-other) # so instead of writing its own we can just take advantage of __add__ and __neg__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data * other.data, (self, other))
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other): # so that int * Value redirects to Value * int aka __mul__. r stands for reverse
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1 # simpler expression now that we have __pow__

    def __rtruediv__(self, other):
        return self / other

    def exp(self):
        out = Value(math.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad # local derivative of e^x is just e^x, aka out.data
        out._backward = _backward

        return out

    def log(self):
        out = Value(math.log(self.data), (self,))

        def _backward():
            self.grad += (1 / self.data) * out.grad # local derivative of ln(x) is 1/x
        out._backward = _backward

        return out

    def tanh(self):
        out = ((2*self).exp() - 1) / ((2*self).exp() + 1)

        def _backward():
            self.grad += (1-out**2) * out.grad # i looked up the local gradient for tanh on wikipedia
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """backpropogates all gradients"""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
            