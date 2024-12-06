import numpy as np

class Tensor:
    '''Stores a tensor and its gradient information'''
    def __init__(self, data, _children: tuple =()):
        if isinstance(data, list):
            self.data = np.array(data)
            self.grad = np.zeros_like(self.data)
        elif isinstance(data, (np.ndarray, np.float64)):
            self.data = data
            self.grad = np.zeros_like(data)
        elif isinstance(data, Tensor):
            self.data = data.data
            self.grad = np.zeros_like(data.data)
        else:
            raise ValueError('input must either be list, np.ndarray, np.float64, or Tensor')
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        if isinstance(self.data, Tensor): # numpy prints "Tensor:\n..." by default and this supresses that
            return f"Tensor:\n({self.data.data})\nGrad:\n({self.grad.data})"
        return f"Tensor:\n({self.data})\nGrad:\n({self.grad})"

    def __add__(self, other):
        if isinstance(other, (float, int)):
            out = Tensor(self.data + other, (self,))
            def _backward():
                self.grad += out.grad
            out._backward = _backward
            
        elif isinstance(other, (np.ndarray, np.float64, Tensor)):
            ### assert working shape - only implementing shapes we expect in order to simplify backprop
            # either shapes are exact same (eg. residual connection) or
            # final dim of self is same as *only* dim of other (eg. linear layer biases)
            assert ((self.shape == other.shape) or 
                    (other.ndim == 1 and self.shape[-1] == other.shape[0])), \
                    f'input shapes {self.shape} and {other.shape} invalid for entry-wise addition'

            # calc forward pass
            other = other if isinstance(other, Tensor) else Tensor(other)
            if self.shape != other.shape:
                out = Tensor(self.data + np.broadcast_to(other.data, self.shape), (self, other))
            else:
                out = Tensor(self.data + other.data, (self, other))
            
            def _backward():
                self.grad += out.grad
                if self.shape == other.shape:
                    other.grad += out.grad
                else:
                    for i, (self_dim, other_dim) in enumerate(zip(self.shape, other.shape)):
                        if self_dim != other_dim: break
                    other.grad += np.sum(out.grad, axis = i).reshape(other.shape)
            out._backward = _backward
            
        else:
            raise ValueError('input must either be int, float, np.ndarray, np.float64, or Tensor')
        
        return out

    def __radd__(self, other): 
        # so that int + Value redirects to Value + int aka __add__. r stands for reverse
        return self + other

    def __neg__(self):
        return self.data * -1

    def __sub__(self, other): 
        # so instead of writing its own we can just take advantage of __add__ and __neg__
        return other + (-self)#self + (-other) 

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other): # entry-wise multiplication
        assert isinstance(other, (float, int, np.ndarray, np.float64, Tensor)),\
            f'input must either be int, float, np.ndarray, np.float64, or Tensor, not {type(other)}'
        
        if isinstance(other, (float, int)):
            out = Tensor(self.data * other, (self,))
            def _backward():
                self.grad += other * out.grad
            out._backward = _backward
            
        elif isinstance(other, (np.ndarray, np.float64, Tensor)):
            ### assert working shape - only implementing shapes we expect in order to simplify backprop
            # either shapes are exact same (eg. GLU hidden dimension mult) or
            # final dim of self is same as *only* dim of other (eg. norm affine scaling)
            assert ((self.shape == other.shape) or
                    (other.ndim == 1 and self.shape[-1] == other.shape[0])), \
                    f'input shapes {self.shape} and {other.shape} invalid for entry-wise multiplication'

            # calc forward pass
            other = other if isinstance(other, Tensor) else Tensor(other)
            out = Tensor(self.data * other.data, (self, other))
            
            def _backward():
                self.grad += other.data * out.grad
                if self.shape == other.shape:
                    other.grad += self.data * out.grad
                else:
                    for i, (self_dim, other_dim) in enumerate(zip(self.shape, other.shape)):
                        if self_dim != other_dim: break
                    other.grad += np.sum(self.data * out.grad, axis = i).reshape(other.shape)
            out._backward = _backward
            
        return out

    def __truediv__(self, other):
        assert isinstance(other, (float, int, np.ndarray, np.float64, Tensor)), \
            f'Input must be int, float, np.ndarray, np.float64, or Tensor, not {type(other)}'
        
        if isinstance(other, (float, int)):
            out = Tensor(self.data / other, (self,))
            def _backward():
                self.grad += out.grad / other
            out._backward = _backward
        
        elif isinstance(other, (np.ndarray, np.float64, Tensor)):
            ### Assert working shape - handling broadcasting as in __mul__
            # Either shapes are the same or broadcasting is required
            other = other if isinstance(other, Tensor) else Tensor(other)
            
            # Determine if broadcasting is needed
            if self.shape != other.shape:
                # Broadcasting occurs; need to handle gradients carefully
                broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
                self_data_broadcasted = np.broadcast_to(self.data, broadcast_shape)
                other_data_broadcasted = np.broadcast_to(other.data, broadcast_shape)
                out_data = self_data_broadcasted / other_data_broadcasted
            else:
                # Shapes are the same; element-wise division
                out_data = self.data / other.data
            
            out = Tensor(out_data, (self, other))
            
            def _backward():
                # Compute gradients considering broadcasting
                grad = out.grad
                if self.shape != out.shape:
                    # Sum over broadcasted dimensions for self.grad
                    axis = tuple([i for i, (s_dim, o_dim) in enumerate(zip(self.shape, out.shape)) if s_dim != o_dim])
                    self_grad = (grad / other.data).sum(axis=axis).reshape(self.shape)
                    self.grad += self_grad
                else:
                    self.grad += grad / other.data
                
                if other.shape != out.shape:
                    # Sum over broadcasted dimensions for other.grad
                    axis = tuple([i for i, (o_dim, out_dim) in enumerate(zip(other.shape, out.shape)) if o_dim != out_dim])
                    other_grad = (-self.data * grad / (other.data ** 2)).sum(axis=axis).reshape(other.shape)
                    other.grad += other_grad
                else:
                    other.grad += -self.data * grad / (other.data ** 2)
            out._backward = _backward
        
        else:
            raise ValueError('Unsupported type for division')
        
        return out
        
        '''
    def __truediv__(self, other):
        assert isinstance(other, (float, int, np.ndarray, np.float64, Tensor)), \
            f'Input must be int, float, np.ndarray, np.float64, or Tensor, not {type(other)}'
        
        if isinstance(other, (float, int)):
            out = Tensor(self.data / other, (self,))
            def _backward():
                self.grad += out.grad / other
            out._backward = _backward
            
        elif isinstance(other, (np.ndarray, np.float64, Tensor)):
            ### assert working shape - only implementing shapes we expect in order to simplify backprop
            # either shapes are exact same (eg. GLU hidden dimension mult) or
            # final dim of self is same as *only* dim of other (eg. division during softmax calc)
            assert ((self.shape == other.shape) or
                    (other.ndim == 1 and self.shape[-1] == other.shape[0])), \
                    f'input shapes {self.shape} and {other.shape} invalid for entry-wise division'
            
            other = other if isinstance(other, Tensor) else Tensor(other)
            out = Tensor(self.data / other.data, (self, other))
            
            def _backward():
                self.grad += out.grad / other.data
                other.grad -= (self.data * out.grad) / (other.data ** 2)
                    # bc chain rule w/ starting exponent on other.data as -1
                if self.shape == other.shape:
                    other.grad -= (self.data * out.grad) / (other.data ** 2)
                else:
                    for i, (self_dim, other_dim) in enumerate(zip(self.shape, other.shape)):
                        if self_dim != other_dim: break
                    other.grad -= (self.data * out.grad) / (other.data ** 2).reshape(self.shape)
            out._backward = _backward
        
        return out'''
        
    def sum(self, dim: int = -1):
        out = Tensor(np.sum(self.data, axis = dim), (self,))
        def _backward():
            self.grad += np.broadcast_to(out.grad, self.shape) / self.shape[dim] # is the divide necessary?
        out._backward = _backward
        return out
        
    def __matmul__(self, other):
        assert isinstance(other, (list, np.ndarray, np.float64, Tensor)), \
            f"x2 must be list, np.ndarray, np.float64, Tensor"
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.shape[-1] == other.data.shape[0], \
            f"x1 shape ({self.data.shape[-1]}) must match x2 shape ({self.data.shape[-1]})"
        
        out = Tensor(self.data @ other.data, (self, other)) # (m,n) @ (n,p) -> (m,p)
        
        def _backward():
            self.grad += out.grad @ other.data.transpose() # (m,p) @ (n,p).T -> (m,p) @ (p,n) -> (m,n)
            other.grad += self.data.transpose() @ out.grad # (m,n).T @ (m,p) -> (n,m) @ (m,p) -> (n,p)
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

    def broadcast_to(self, shape: tuple):
        assert self.shape != shape, f"broadcast shape {shape} must be different from original {self.shape}"
        out = Tensor(np.broadcast_to(self.data, shape), (self,))
        for i in range(self.ndim):
            if self.shape[i] != shape:
                dim = i
        assert self.shape[dim] == 1, f"original shape must be 1 on dimension to be broadcast but is {self.shape[i]}"
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