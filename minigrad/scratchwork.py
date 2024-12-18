import numpy as np
from typing import Tuple, Union

Arrayable = Union[float, int, np.ndarray, 'Tensor', 'Parameter']

class Tensor:
    """
    A generic Tensor that supports automatic differentiation. By default, it
    can either be a leaf node (e.g., input data that doesn't require gradients)
    or part of a computational graph where gradients are tracked.
    """
    def __init__(self, data: Union[float, int, np.ndarray], 
                 requires_grad: bool = False, 
                 _children: Tuple['Tensor', ...] = ()):
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            # ensure float type for consistency
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
        else:
            raise ValueError("Invalid data type for Tensor.")

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None # np.zeros_like(data) if requires_grad else None
        self._prev = set(_children)
        self._backward = lambda: None  # function to compute local gradient updates

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other: Arrayable):
        # Ensure other is a Tensor
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

        out = Tensor(self.data + other.data, 
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad = self._safe_add_grad(self.grad, out.grad)
            if other.requires_grad:
                other.grad = self._safe_add_grad(other.grad, out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other: Arrayable):
        # Similar pattern to __add__, omitted for brevity
        pass

    def __neg__(self):
        out = Tensor(-self.data, 
                     requires_grad=self.requires_grad,
                     _children=(self,))
        def _backward():
            if self.requires_grad:
                self.grad = self._safe_add_grad(self.grad, -out.grad)
        out._backward = _backward
        return out

    def backward(self):
        """
        Run backpropagation starting from this tensor. 
        Typically called on a scalar loss Tensor.
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go backward in reverse topological order
        for node in reversed(topo):
            node._backward()

    @staticmethod
    def _safe_add_grad(grad, new_grad):
        return new_grad if grad is None else grad + new_grad

    def zero_grad(self):
        self.grad = None

    # Additional methods like __truediv__, exp, log, etc. would follow the same pattern.
    # Methods for broadcasting, slicing, reshaping would also be implemented here if needed.


class Parameter(Tensor):
    """
    A Parameter is a special kind of Tensor that is meant to be trainable.
    Typically used for model weights and biases in neural network layers.
    By default, Parameters require gradients.
    """
    def __init__(self, data: Union[float, int, np.ndarray]):
        super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Parameter(data={self.data}, requires_grad={self.requires_grad})"


# Example usage within a simple module (not fully implemented):
class Linear:
    """
    A simple linear (fully-connected) layer for demonstration:
    y = xW + b

    W and b would be Parameters.
    """

    def __init__(self, in_features, out_features):
        # Initialize parameters
        self.W = Parameter(np.random.randn(in_features, out_features) * 0.01)
        self.b = Parameter(np.zeros(out_features))

    def __call__(self, x: Tensor):
        # forward pass
        # This is simplified; in a real scenario you would implement proper matmul:
        return x @ self.W + self.b

    def parameters(self):
        # Return parameters in a list for optimizer updates
        return [self.W, self.b]


# Example usage within an optimizer:
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.requires_grad:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
