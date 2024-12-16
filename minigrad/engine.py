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
            ### assert working shape - only implementing shapes we expect in order to simplify backprop
            # either shapes are exact same (eg. residual connection) or
            # final dim of self is same as *only* dim of other (eg. linear layer biases)
            assert ((self.shape == other.shape) or 
                    (other.ndim == 1 and self.shape[-1] == other.shape[0])), \
                    f'input shapes {self.shape} and {other.shape} invalid for entry-wise addition'

            # ensure other is of type Tensor in order to simplify later code
            if not isinstance(other, Tensor): other = Tensor(other)
                # if other does not start off as a tensor then the gradient that gets recorded here 
                # will get ignored when the fwd pass inputs a fresh non-tensor every time

            # calc forward pass
            out = Tensor(self.data + other.data, (self, other)) if self.shape == other.shape else \
                    Tensor(self.data + np.broadcast_to(other.data, self.shape), (self, other))
            
            def _backward():
                self.grad += out.grad
                if self.shape == other.shape:
                    other.grad += out.grad
                else:
                    # we'll iterate through the dims until we find the one where they become equal
                    for i, (self_dim, other_dim) in enumerate(zip(self.shape, other.shape)):
                        if self_dim != other_dim: break
                    # then we'll use that dim to inform our calculation
                    other.grad += np.sum(out.grad, axis = i).reshape(other.shape)
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
            ### assert working shape - only implementing shapes we expect in order to simplify backprop
            # either shapes are exact same (eg. GLU hidden dimension mult) or
            # final dim of self is same as *only* dim of other (eg. norm affine scaling)
            assert ((self.shape == other.shape) or
                    (other.ndim == 1 and self.shape[-1] == other.shape[0])), \
                    f'input shapes {self.shape} and {other.shape} invalid for entry-wise multiplication'

            # ensure other is of type Tensor in order to simplify later code
            if not isinstance(other, Tensor): other = Tensor(other)
                # if other does not start off as a tensor then the gradient that gets recorded here 
                # will get ignored when the fwd pass inputs a fresh non-tensor every time

            # calc forward pass
            out = Tensor(self.data * other.data, (self, other)) if self.shape == other.shape else \
                    Tensor(self.data * np.broadcast_to(other.data, self.shape), (self, other))
            
            def _backward():
                self.grad += other.data * out.grad
                if self.shape == other.shape:
                    other.grad += self.data * out.grad
                else:
                    # we'll iterate through the dims until we find the one where they become equal
                    for i, (self_dim, other_dim) in enumerate(zip(self.shape, other.shape)):
                        if self_dim != other_dim: break
                    # then we'll use that dim to inform our calculation
                    other.grad += np.sum(self.data * out.grad, axis = i).reshape(other.shape)
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
            ### Assert working shape - handling broadcasting as in __mul__
            # Either shapes are the same or broadcasting is required
            assert ((self.shape == other.shape) or
                    (other.ndim == 1 and self.shape[-1] == other.shape[0])), \
                    f'input shapes {self.shape} and {other.shape} invalid for entry-wise division'

            # ensure other is of type Tensor in order to simplify later code
            if not isinstance(other, Tensor): other = Tensor(other)
                # if other does not start off as a tensor then the gradient that gets recorded here 
                # will get ignored when the fwd pass inputs a fresh non-tensor every time

            # calc forward pass; Determine if broadcasting is needed
            if self.shape != other.shape:
                # Broadcast both arrays to a common shape before division
                broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
                self_data_broadcasted = np.broadcast_to(self.data, broadcast_shape)
                other_data_broadcasted = np.broadcast_to(other.data, broadcast_shape)
                out_data = self_data_broadcasted / other_data_broadcasted
            else:
                # Shapes are the same; no broadcasting required
                out_data = self.data / other.data
                
            out = Tensor(out_data, (self, other))
            
            def _backward():
                ### Compute gradients considering broadcasting
                # local grad for slef (the numerator): d/dx (x/y) = 1/y
                if self.shape != out.shape:
                    # identify broadcasted dimensions
                    axes = tuple(i for i, (self_dim, other_dim) in enumerate(zip(self.shape, out.shape)) if self_dim != other_dim)
                    # sum over broadcasted dimensions since gradient must be reduced back to self's shape
                    self.grad += (out.grad / other.data).sum(axis=axes).reshape(self.shape)
                else:
                    self.grad += out.grad / other.data
                # local grad for other (the denominator): d/dy (x/y) = -x / (y^2)
                if other.shape != out.shape:
                    # identify broadcasted dimensions
                    axes = tuple(i for i, (other_dim, out_dim) in enumerate(zip(other.shape, out.shape)) if other_dim != out_dim)
                    # sum over broadcasted dimensions since gradient must be reduced back to self's shape
                    other.grad += (-self.data / (other.data ** 2) * out.grad).sum(axis=axes).reshape(other.shape)
                else:
                    other.grad += (-self.data / (other.data ** 2)) * out.grad
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
        return ou

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out.backward = _backward
        return out

    def softmax(self, dim: int = -1):
        exps = self.exp()
        sum_exps = exps.sum(dim=dim)
        out = exps / sum_exps
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
            if self.shape[i] != shape:
                dim = i
        assert dim and self.shape[dim] == 1, f"original shape must be 1 on dimension to be broadcast but is {self.shape[i]}"
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