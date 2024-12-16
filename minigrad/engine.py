import numpy as np

class Tensor:
    '''Stores a tensor and its gradient information'''
    def __init__(self, data, _children: tuple =()):
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, (np.ndarray, np.float64)):
            self.data = data
        elif isinstance(data, Tensor):
            self.data = data.data
        else:
            raise ValueError('input must either be list, np.ndarray, np.float64, or Tensor')
        self.grad = np.zeros_like(self.data)
        
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        #if isinstance(self.data, Tensor): # numpy prints "Tensor:\n..." by default and this supresses that
            #return f"Tensor:\n({self.data.data})\nGrad:\n({self.grad.data})"
        return f"Tensor:\n({self.data})\nGrad:\n({self.grad})"

    def __add__(self, other): # entry-wise addition
        assert isinstance(other, (float, int, np.ndarray, np.float64, Tensor)),\
                f'input must either be int, float, np.ndarray, np.float64, or Tensor but is {type(other)}'
        
        if isinstance(other, (float, int)):
            out = Tensor(self.data + other, (self,))
            def _backward():
                self.grad += out.grad
            out._backward = _backward
            
        if isinstance(other, (np.ndarray, np.float64, Tensor)):
            assert self.ndim == other.ndim, f'tensor ndim mismatch x1: {self.shape} x2: {other.shape}'

            # ensure other is of type Tensor in order to simplify later code
            if not isinstance(other, Tensor): other = Tensor(other)
                # if other does not start off as a tensor then the gradient that gets recorded here 
                # will get ignored when the fwd pass inputs a fresh non-tensor every time

            # calc forward pass
            out = Tensor(self.data + other.data, (self, other)) if self.shape == other.shape else \
                    Tensor(self.data + np.broadcast_to(other.data, self.shape), (self, other))
            
            def _backward():
                self.grad += out.grad
                if other.shape == self.shape:
                    other.grad += out.grad
                else:
                    # identify broadcasted dimensions
                    axes = tuple(i for i, (other_dim, out_dim) in enumerate(zip(other.shape, out.shape)) if other_dim != out_dim)
                    # sum over broadcasted dimensions since gradient must be reduced back to self's shape
                    other.grad += out.grad.sum(axis=axes).reshape(other.shape)
            out._backward = _backward
        
        return out

    def __radd__(self, other): 
        # so that int + Value redirects to Value + int aka __add__. r stands for reverse
        return self + other

    def __neg__(self):
        out = Tensor(self.data * -1, (self,))
        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other): 
        # so instead of writing its own we can just take advantage of __add__ and __neg__
        return other + (-self)#self + (-other) 

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other): # entry-wise multiplication
        assert isinstance(other, (float, int, np.ndarray, np.float64, Tensor)),\
                f'input must either be int, float, np.ndarray, np.float64, or Tensor but is {type(other)}'
        
        if isinstance(other, (float, int)):
            out = Tensor(self.data * other, (self,))
            def _backward():
                self.grad += other * out.grad
            out._backward = _backward
            
        if isinstance(other, (np.ndarray, np.float64, Tensor)):
            assert self.ndim == other.ndim, f'tensor ndim mismatch x1: {self.shape} x2: {other.shape}'
            
            # ensure other is of type Tensor in order to simplify later code
            if not isinstance(other, Tensor): other = Tensor(other)
                # if other does not start off as a tensor then the gradient that gets recorded here 
                # will get ignored when the fwd pass inputs a fresh non-tensor every time
            
            # calc forward pass
            out = Tensor(self.data * other.data, (self, other)) if self.shape == other.shape else \
                    Tensor(self.data * np.broadcast_to(other.data, self.shape), (self, other))
            
            def _backward():
                self.grad += other.data * out.grad
                if other.shape == self.shape:
                    other.grad += self.data * out.grad
                else:
                    # identify broadcasted dimensions
                    axes = tuple(i for i, (other_dim, out_dim) in enumerate(zip(other.shape, out.shape)) if other_dim != out_dim)
                    # sum over broadcasted dimensions since gradient must be reduced back to self's shape
                    other.grad += (out.grad * self.data).sum(axis=axes).reshape(other.shape)
            out._backward = _backward
            
        return out

    def __truediv__(self, other):
        """
        Perform element-wise division: self / other, with support for broadcasting.
        
        Forward pass:
          out = x / y
        
        Backward pass:
          d/dx (x/y) = 1 / y
          d/dy (x/y) = -x / (y^2)
        """
        assert isinstance(other, (float, int, np.ndarray, np.float64, Tensor)),\
                f'input must either be int, float, np.ndarray, np.float64, or Tensor but is {type(other)}'
        
        if isinstance(other, (float, int)):
            out = Tensor(self.data / other, (self,))
            def _backward():
                self.grad += out.grad / other
            out._backward = _backward
        
        if isinstance(other, (np.ndarray, np.float64, Tensor)):
            # ensure other is of type Tensor in order to simplify later code
            if not isinstance(other, Tensor): other = Tensor(other)
                # if other does not start off as a tensor then the gradient that gets recorded here 
                # will get ignored when the fwd pass inputs a fresh non-tensor every time

            # calc forward pass
            out = Tensor(self.data / other.data, (self, other)) if self.shape == other.shape else \
                    Tensor(self.data / np.broadcast_to(other.data, self.shape), (self, other))
            
            def _backward():
                ### Compute gradients considering broadcasting
                # local grad for self (the numerator): d/dx (x/y) = 1/y
                self.grad += out.grad / other.data
                # local grad for other (the denominator): d/dy (x/y) = -x / (y^2)
                if other.shape == self.shape:
                    other.grad += (-self.data / (other.data ** 2)) * out.grad
                else:
                    # identify broadcasted dimensions
                    axes = tuple(i for i, (other_dim, out_dim) in enumerate(zip(other.shape, out.shape)) if other_dim != out_dim)
                    # sum over broadcasted dimensions since gradient must be reduced back to self's shape
                    other.grad += (-self.data / (other.data ** 2) * out.grad).sum(axis=axes).reshape(other.shape)
            out._backward = _backward
        
        return out

    def __rtruediv__(self, other):
        return self / other
        
    def sum(self, dim: int = -1):
        out = Tensor(np.sum(self.data, axis = dim), (self,))
        def _backward():
            self.grad += np.broadcast_to(out.grad, self.shape)
        out._backward = _backward
        return out
        
    def __matmul__(self, other):
        assert isinstance(other, (list, np.ndarray, np.float64, Tensor)), \
            f"x2 must be list, np.ndarray, np.float64, Tensor"
        if not isinstance(other, Tensor): other = Tensor(other)
        assert self.ndim >= 2 and other.ndim >= 2, \
            f'Both tensors must have at least 2 dimensions for matrix multiplication: Got x1:{self.ndim}, x2:{other.ndim}'
        assert self.shape[-1] == other.shape[-2], \
            f"Mismatch in matrix multiplication dimensions: {self.shape[-1]} != {other.shape[-2]}"
        assert other.shape[:-1] == () or other.shape[:-2] == self.shape[:-2], \
            f"Preceding dimensions must either match exactly or be absent. Got x1:{self.shape[:-2]}, x2:{other.shape[:-2]}"

        # calc forward pass
        out = Tensor(self.data @ other.data, (self, other)) # (..., m, n) @ (..., n, p) -> (..., m, p)
        
        def _backward():
            self.grad += out.grad @ other.data.transpose() # (..., m, p)   @ (..., n, p).T -> (..., m, p) @ (..., p, n) -> (..., m, n)
            other.grad += self.data.transpose() @ out.grad # (..., m, n).T @ (..., m, p)   -> (..., n, m) @ (..., m, p) -> (..., n, p)
        out._backward = _backward
        
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad # derivative of e^x is just e^x, therefore out.data
        out.backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out.backward = _backward
        return out

    def max(self, axis = None):
        assert not axis or isinstance(axis, int)
        return np.max(self.data) if axis is None else np.max(self.data, axis=axis)

    def min(self, axis = None):
        assert not axis or isinstance(axis, int)
        return np.min(self.data) if axis is None else np.min(self.data, axis=axis)

    def softmax(self, dim: int = -1):
        # to make make softmax stable (avoid numerical overflow during .exp()) we subtract by the maximum values)
        maximums = np.max(self.data, axis=dim)
        print(maximums)
        self.data -= np.broadcast_to(np.expand_dims(maximums, axis=dim), self.shape)
        # subtraction of max doesn't have to worry about gradient since local grad of addition is 1
        # the following ops have their gradient calculated by all the Tensor methods that we call
        print(self)
        exps = self.exp()
        sum_exps = exps.sum(dim=dim).unsqueeze(dim).broadcast_to(self.shape)
        print(exps)
        print(sum_exps)
        return exps / sum_exps

    def __pow__(self, pow: int):
        '''
        entry-wise exponentiation that supports integer powers
        '''
        assert isinstance(pow, int), f'power must be int but got {type(pow)}'
        out = Tensor(self.data ** pow, (self,))
        def _backward(): # local grad: d/dx (x^p) = p * x^(p - 1)
            self.grad += out.grad * pow * self.data ** (pow - 1)
        out.backward = _backward
        return out

    def transpose(self):
        out = Tensor(self.data.transpose(), (self,))
        def _backward():
            self.grad += out.grad.transpose()
        out.backward = _backward
        return out

    def squeeze(self, dim):
        out = Tensor(np.squeeze(self.data, axis=dim), (self,))
        def _backward():
            self.grad += np.expand_dims(out.grad, axis=dim)
        out.backward = _backward
        return out
        
    def unsqueeze(self, dim):
        out = Tensor(np.expand_dims(self.data, axis=dim), (self,))
        def _backward():
            self.grad += np.squeeze(out.grad, axis=dim)
        out.backward = _backward
        return out

    def broadcast_to(self, shape: tuple):
        assert self.shape != shape, f"broadcast shape {shape} must be different from original {self.shape}"
        for i in range(self.ndim):
            if self.shape[i] != shape[i]:
                dim = i
        assert self.shape[dim] == 1, f"original shape must be 1 on dimension to be broadcast but is {self.shape[dim]}"
        out = Tensor(np.broadcast_to(self.data, shape), (self,))
        def _backward():
            self.grad += np.sum(out.grad, axis=dim)
        out.backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.grad)
        for node in reversed(topo):
            node._backward()