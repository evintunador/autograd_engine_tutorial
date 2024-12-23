import numpy as np

from typing import Tuple, Union

class Tensor:
    '''Stores a tensor and its gradient information'''
    def __init__(self, 
                 data: Union[float, int, list, np.ndarray], 
                 requires_grad: bool = False, 
                 _children: Tuple[Union['Tensor', 'Parameter'], ...] = ()):
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=np.float32)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            # ensure float type for consistency
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
        else:
            raise ValueError('Invalid data type for Tensor.')
        
        self.data = data
        
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self._prev = set(_children)
        self._backward = lambda: None  # function to compute local gradient updates

    def __repr__(self):
        #if isinstance(self.data, Tensor): # numpy prints "Tensor:\n..." by default and this supresses that
            #return f"Tensor:\n({self.data.data})\nGrad:\n({self.grad.data})"
        return f"Tensor:\n({self.data})\nGrad:\n({self.grad})"

    def __add__(self, other): # entry-wise addition
        # Ensure other is a Tensor; also takes advantage of __init__'s type assertions
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

        # do i need this? if yes then it'll require we use .unsqueeze() before inputting here
        if (other.ndim == 1 and other.shape[0] == 1) or other.shape[-other.ndim:] == self.shape[-other.ndim:]:
            while other.ndim < self.ndim:
                other = other.unsqueeze(0)
        assert self.ndim == other.ndim, f'tensor ndim mismatch x1: {self.shape} x2: {other.shape}'

        # calc forward pass
        other_data = other.data if self.shape == other.shape else np.broadcast_to(other.data, self.shape)
        out = Tensor(self.data + other_data,
                     requires_grad = (self.requires_grad or other.requires_grad),
                     _children = (self, other))
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            
            if other.requires_grad:
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
        out = Tensor(self.data * -1, self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += -1 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other): 
        # so instead of writing its own we can just take advantage of __add__ and __neg__
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other): # entry-wise multiplication
        # Ensure other is a Tensor; also takes advantage of __init__'s type assertions
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

        # do i need this? if yes then it'll require we use .unsqueeze() before inputting here
        if (other.ndim == 1 and other.shape[0] == 1) or other.shape[-other.ndim:] == self.shape[-other.ndim:]:
            while other.ndim < self.ndim:
                other = other.unsqueeze(0)
        assert self.ndim == other.ndim, f'tensor ndim mismatch x1: {self.shape} x2: {other.shape}'

        # calc forward pass
        other_data = other.data if self.shape == other.shape else np.broadcast_to(other.data, self.shape)
        out = Tensor(self.data * other_data,
                     requires_grad = (self.requires_grad or other.requires_grad),
                     _children = (self, other))
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other.data
            
            if other.requires_grad:
                if other.shape == self.shape:
                    other.grad += out.grad * self.data
                else:
                    # identify broadcasted dimensions
                    axes = tuple(i for i, (other_dim, out_dim) in enumerate(zip(other.shape, out.shape)) if other_dim != out_dim)
                    # sum over broadcasted dimensions since gradient must be reduced back to self's shape
                    other.grad += (out.grad * self.data).sum(axis=axes).reshape(other.shape)
        out._backward = _backward
        
        return out

    def __truediv__(self, other): # entry-wise division
        # Ensure other is a Tensor; also takes advantage of __init__'s type assertions
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

        # do i need this? if yes then it'll require we use .unsqueeze() before inputting here
        if (other.ndim == 1 and other.shape[0] == 1) or other.shape[-other.ndim:] == self.shape[-other.ndim:]:
            while other.ndim < self.ndim:
                other = other.unsqueeze(0)
        assert self.ndim == other.ndim, f'tensor ndim mismatch x1: {self.shape} x2: {other.shape}'

        # calc forward pass
        eps = 1e-7 if self.data.dtype == np.float32 else 1e-16  # Adjust epsilon based on precision
        other_data = other.data + eps # for stability; wouldn't want to divide by 0
        other_data = other_data if self.shape == other.shape else np.broadcast_to(other_data, self.shape)
        out = Tensor(self.data / other_data,
                     requires_grad = (self.requires_grad or other.requires_grad),
                     _children = (self, other))
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad / other.data
            
            if other.requires_grad:
                if other.shape == self.shape:
                    other.grad += out.grad * (-self.data / (other_data ** 2))
                else:
                    # identify broadcasted dimensions
                    axes = tuple(i for i, (other_dim, out_dim) in enumerate(zip(other.shape, out.shape)) if other_dim != out_dim)
                    # sum over broadcasted dimensions since gradient must be reduced back to self's shape
                    other.grad += (out.grad * (-self.data / (other_data ** 2))).sum(axis=axes).reshape(other.shape)
        out._backward = _backward
        
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self
        
    def __matmul__(self, other):
        # Ensure other is a Tensor; also takes advantage of __init__'s type assertions
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        assert self.ndim >= 2 and other.ndim >= 2, \
            f'Both tensors must have at least 2 dimensions for matrix multiplication: Got x1:{self.ndim}, x2:{other.ndim}'
        assert self.shape[-1] == other.shape[-2], \
            f"Mismatch in matrix multiplication dimensions: {self.shape[-1]} != {other.shape[-2]}"
        assert other.shape[:-1] == () or other.shape[:-2] == self.shape[:-2], \
            f"Preceding dimensions must either match exactly or be absent. Got x1:{self.shape[:-2]}, x2:{other.shape[:-2]}"

        # calc forward pass
        out = Tensor(self.data @ other.data, # (..., m, n) @ (..., n, p) -> (..., m, p)
                     requires_grad = (self.requires_grad or other.requires_grad),
                     _children = (self, other)) 
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.transpose() # (..., m, p)   @ (..., n, p).T -> (..., m, p) @ (..., p, n) -> (..., m, n)
            if other.requires_grad:
                other.grad += self.data.transpose() @ out.grad # (..., m, n).T @ (..., m, p)   -> (..., n, m) @ (..., m, p) -> (..., n, p)
        out._backward = _backward
        
        return out
        
    def sum(self, dim: int = -1):
        out = Tensor(np.sum(self.data, axis = dim), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.broadcast_to(out.grad, self.shape)
        out._backward = _backward
        return out
    
    def mean(self, dim: int = -1):
        # i don't think i need a _backward because sum and truediv already implement them into out
        return self.sum(dim) / self.shape[dim]

    def exp(self):
        out = Tensor(np.exp(self.data), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad # derivative of e^x is just e^x, therefore out.data
        out._backward = _backward
        return out

    def log(self):
        assert np.all(self.data > 0), f'matrix contains values below 0; cannot take natural logaritm'
        eps = 1e-7 if self.data.dtype == np.float32 else 1e-16  # Adjust epsilon based on precision
        stabilized_data = np.clip(self.data, eps, None)  # Clip values to [eps, ∞)
        out = Tensor(np.log(stabilized_data), self.requires_grad, (self,))
        def _backward(): # local gradient: d/dx (ln(x)) = 1/x
            if self.requires_grad:
                self.grad += out.grad  / self.data
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def _normalize_dim(self, dim: int):
        # normalize dim to ensure that calculations which don't like negative numbers work
        if dim is not None and dim < 0:
            # Convert negative axes to positive
            dim += self.ndim  # e.g. -1 => +2 for a 3D tensor
        return dim

    def max(self, dim = None):
        assert not dim or isinstance(dim, int), "dim must be None or an integer"
        dim = self._normalize_dim(dim)

        # grab indices of the maximum values
        idx = np.argmax(self.data, dim)
        if dim is None: # returs the un-flattened index
            idx = np.unravel_index(idx, self.shape)

        # calc fwd pass
        maximums = Tensor(np.max(self.data, dim), self.requires_grad, (self,))
        
        def _backward():
            if self.requires_grad:
                if dim is None:
                    self.grad[idx] += maximums.grad[0]
                else:
                    # self.shape = (d0, d1, ..., dN)
                    # If dim = k then 'reduced_shape' = maximums.shape = (d0, ... d_{k-1}, d_{k+1}, ... dN)
                    # Create a range array for all axes except the reduced one
                    coords = [np.arange(d) for i, d in enumerate(self.shape) if i != dim]
                
                    # Now use meshgrid to create full coordinate arrays
                    grids = list(np.meshgrid(*coords, indexing='ij')) 
                        # https://www.geeksforgeeks.org/numpy-meshgrid-function/

                    # We need to interleave 'idx' into these at the correct position.
                    final_indices = grids[:dim] + [idx] + grids[dim:]

                    # Now apply np.add.at to get our gradients into the correct place
                    np.add.at(self.grad, tuple(final_indices), maximums.grad)
        maximums._backward = _backward
        
        return maximums, idx

    def min(self, dim = None):
        assert not dim or isinstance(dim, int), "dim must be None or an integer"
        dim = self._normalize_dim(dim)

        # grab indices of the maximum values
        idx = np.argmin(self.data, dim)
        if dim is None:
            idx = np.unravel_index(idx, self.shape)

        # calc fwd pass
        minimums = Tensor(np.min(self.data, dim), self.requires_grad, (self,))
        
        def _backward():
            if self.requires_grad:
                if dim is None:
                    self.grad[idx] += minimums.grad[0]
                else:
                    # self.shape = (d0, d1, ..., dN)
                    # If dim = k, then 'reduced_shape' = maximums.shape = (d0, ... d_{k-1}, d_{k+1}, ... dN)
                    # Create a range array for all axes except the reduced one
                    coords = [np.arange(d) for i, d in enumerate(self.shape) if i != dim]
                
                    # Now use meshgrid to create full coordinate arrays
                    grids = list(np.meshgrid(*coords, indexing='ij')) 
                        # https://www.geeksforgeeks.org/numpy-meshgrid-function/
                
                    # We need to interleave 'idx' into these at the correct position.
                    final_indices = grids[:dim] + [idx] + grids[dim:]
                            
                    # Now apply np.add.at to get our gradients into the correct place
                    np.add.at(self.grad, tuple(final_indices), minimums.grad)
        minimums._backward = _backward
        
        return minimums, idx

    def softmax(self, dim: int = -1):
        # Stabilize (avoid numerical overflow) by subtracting max
        maximums = self.max(dim)[0]
        stable_self = self - maximums.unsqueeze(dim)
        # calculate softmax
        exps = stable_self.exp()
        sum_exps = exps.sum(dim=dim).unsqueeze(dim)#.broadcast_to(self.shape)
        return exps / sum_exps

    def __pow__(self, pow: int):
        '''entry-wise exponentiation that supports integer powers'''
        assert isinstance(pow, (int, float)), f'power must be int or float but got {type(pow)}'
        out = Tensor(self.data ** pow, self.requires_grad, (self,))
        def _backward(): # local grad: d/dx (x^p) = p * x^(p - 1)
            if self.requires_grad:
                self.grad += out.grad * pow * self.data ** (pow - 1)
        out._backward = _backward
        return out
    
    def var(self, dim: int = -1):
        return ((self - self.mean(dim).unsqueeze(dim)) ** 2).sum(dim) / self.shape[dim]
    
    def sd(self, dim: int = -1):
        return self.var(dim) ** 0.5

    def transpose(self, axes: tuple = None):
        if axes is None: # defaults to transposing final two dims
            axes = tuple(dim for dim in range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2)
        out = Tensor(np.transpose(self.data, axes=axes), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.transpose(out.grad, axes=axes)
        out._backward = _backward
        return out

    def squeeze(self, dim):
        out = Tensor(np.squeeze(self.data, axis=dim), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.expand_dims(out.grad, axis=dim)
        out._backward = _backward
        return out
        
    def unsqueeze(self, dim):
        out = Tensor(np.expand_dims(self.data, axis=dim), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.squeeze(out.grad, axis=dim)
        out._backward = _backward
        return out

    def broadcast_to(self, shape: tuple):
        assert self.shape != shape, f"broadcast shape {shape} must be different from original {self.shape}"
        for i in range(self.ndim):
            if self.shape[i] != shape[i]:
                dim = i
        assert self.shape[dim] == 1, f"original shape must be 1 on dimension to be broadcast but is {self.shape[dim]}"
        out = Tensor(np.broadcast_to(self.data, shape), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.sum(out.grad, axis=dim)
        out._backward = _backward
        return out

    def reshape(self, shape: tuple):
        out = Tensor(np.reshape(self.data, shape), self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.reshape(out.grad, self.shape)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        # numpy handles the actual indexing behavior for us
        sliced_data = self.data[idx]
        out = Tensor(sliced_data, self.requires_grad, (self,))
        def _backward():
            if self.requires_grad:
                # np.add.at() correctly distributes the gradient from the sliced tensor back to the original tensor
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self):
        """
        Run backpropagation starting from this tensor. 
        Typically called on a scalar loss Tensor.
        """
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


class Parameter(Tensor):
    """
    A Parameter is a special kind of Tensor that is meant to be trainable.
    Typically used for model weights and biases in neural network layers.
    By default, Parameters require gradients.
    """
    def __init__(self, data: Union[float, int, np.ndarray]):
        super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Tensor:\n({self.data})\nGrad:\n({self.grad})"


if __name__ == "__main__":
    print('-------- test add/mul/truediv same shape --------')
    x = Tensor([[1,2],[3,4]])
    print(x)
    w = Parameter([[0.1,-0.2],[-0.1,0.1]])
    print(w)
    y = x / w
    print(y)
    y.backward()
    print(y)
    print(w)
    print(x)

    print('-------- test add/mul/truediv broadcasted --------')
    x = Tensor([[1,2],[3,4]])
    print(x)
    w = Parameter([[0.1,-0.2]])
    print(w)
    y = x / w
    print(y)
    y.backward()
    print(y)
    print(w)
    print(x)

    print('------------------------ test min/max ------------------------')
    print('-------- single entry --------')
    a = Tensor(np.array([[[1.,2],[3,4]],[[5,6],[7,8]]]), requires_grad=True)
    print(a)
    m = a.max()[0]
    print(m)
    m.backward()
    print(m)
    print(a)
    print('-------- along a specific dim --------')
    b = Tensor(np.array([[[1.,2],[3,4]],[[5,6],[7,8]]]), requires_grad=True)
    print(b)
    m = b.max(1)[0]
    print(m)
    m.backward()
    print(m)
    print(b)

    print('------------------------ test softmax ------------------------')
    print('-------- single entry --------')
    a = Tensor(np.array([[[1.,2],[3,4]],[[5,6],[7,8]]]), requires_grad=True)
    print(a)
    m = a.softmax()
    print(m)
    m.backward()
    print(m)
    print(a)
    print('-------- along a specific dim --------')
    b = Tensor(np.array([[[1.,2],[3,4]],[[5,6],[7,8]]]), requires_grad=True)
    print(b)
    m = b.softmax(1)
    print(m)
    m.backward()
    print(m)
    print(b)
