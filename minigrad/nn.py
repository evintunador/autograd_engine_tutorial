import numpy as np
from engine import Tensor, Parameter

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
    def __init__(self, in_dim: int, out_dim: int, bias = True):
        super().__init__()
        self.w = Parameter(np.random.normal(scale=0.02, size=(in_dim, out_dim)).astype(np.float32))
        if bias: self.b = Parameter(np.zeros((1,out_dim)).astype(np.float32))

    def __call__(self, x: Tensor):
        while x.ndim > self.b.ndim:
            self.b = self.b.unsqueeze(0)
        return x @ self.w + self.b if self.b else x @ self.w

    def __repr__(self):
        return f"Weight:\n({self.w})\nBias:\n({self.b})" if self.b else f"Weight:\n({self.w})"

    def parameters(self):
        return [self.w, self.b] if self.b is not None else [self.w]

class Embedding(Module):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.w = Parameter(np.random.normal(scale=0.02, size=(num_classes, embed_dim)).astype(np.float32))

    def __call__(self, tokens):
        assert np.issubdtype(tokens.dtype, np.dtype('int')),\
                f"input dtype should be np.int but instead got {tokens.dtype}"
        # grab embedding assigned to each token
        return self.w[tokens]

    def __repr__(self):
        return f"Emedding:\n({self.w})"

    def parameters(self):
        return [self.w]

class Dropout(Module):
    '''
    so we really don't need to save self.mask here. pytorch does because their backward pass works
    differently, but ours gets handled entirely by Tensor. I've made Dropout a Module here just for
    sake of making our final code resemble pytorch in use, even if it does not in implementation
    '''
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        self.mask = None

    def __call__(self, x: Tensor):
        if not self.training:
            return x
        
        # create a mask of the same shape as x
        # with probability (1 - p) for each element to be 1, and p to be 0
        mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
        self.mask = Tensor(mask, requires_grad=False) # mask doesn't need grad
        
        return x * self.mask

    def __repr__(self):
        return f"Dropout(p={self.p})"
    
    def parameters(self):
        return

class LayerNorm(Module):
    def __init__(self, dim: int, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = 1e-5

        if elementwise_affine: 
            # TODO: should this be np.ones???
            self.affine = Parameter(np.random.normal(scale=0.02, size=dim).astype(np.float32))
            self.bias = Parameter(np.zeros(dim).astype(np.float32))

    def __call__(self, x):
        assert self.dim == x.shape[-1]
        # normalize
        mean = x.mean(keepdim=True)
        var = x.var(keepdim=True)
        out = (x - mean) / (var + self.eps) ** 0.5
        # affine transformation
        if self.affine: out = out * self.affine + self.bias
        return out

    def __repr__(self):
        out = "LayerNorm"
        if self.affine: out += f"\nElement-wise affine:\n({self.affine})\nBias:\n({self.bias})"
        return out

    def parameters(self):
        return [self.affine, self.bias] if self.affine else None
    
class CrossEntropyLoss(Module):
    def __init__(self, vocab_len: int, pad_token: int = None):
        super().__init__()
        self.vocab_len = vocab_len
        self.pad_token = pad_token

    def __call__(self, probabilities: Tensor, targets: np.ndarray) -> Tensor:
        '''
        inputs: 
          probabilities - Tensor of shape (batch_size, seq_len, vocab_len) 
                          representing predicted probabilities. Each slice along last dimension
                          must sum to 1 (i.e., a proper probability distribution).
          targets       - np.ndarray of shape (batch_size, seq_len) of integer token indices

        output: 
          Scalar Tensor representing average cross-entropy loss across the whole batch/sequence.
          If pad_token is set, those positions are ignored in the average.
        '''
        B, L, V = probabilities.shape
        # Basic shape checks
        assert (B, L) == targets.shape, \
            f"Shape mismatch: probabilities {probabilities.shape} vs targets {targets.shape}"
        
        # Flatten predictions and targets so we can index easily
        probabilities_2d = probabilities.reshape((B*L, V))  # (B, L, V) -> (B*L, V)
        targets_flat = targets.ravel()                  # (B, L) -> (B*L)
        
        # Gather probabilities of the correct classes
        picked_probs = probabilities_2d[range(probabilities_2d.shape[0]), targets_flat] # (B*L, V) -> (B*L)
        log_picked = picked_probs.log()
        
        # If we have a pad token, ignore those positions
        if self.pad_token is not None:
            pad_mask = (targets_flat == self.pad_token).astype(np.float32) # (B*L)
            log_picked = log_picked.masked_fill(pad_mask, 0.)

        return - log_picked.mean() # (B*L) -> (1)
    
    def parameters(self):
        return

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
    
    print("---------------- test cross entropy loss ----------------")
    # Test case 1: Basic functionality without pad token
    batch_size, seq_len, vocab_size = 2, 3, 5
    # Create some "logits" and apply softmax to get probabilities
    logits = Tensor(np.random.randn(batch_size, seq_len, vocab_size), requires_grad=True)
    probs = logits.softmax(dim=-1)
    # Create some random target indices
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    targets_pad = targets.copy()
    targets_pad[:,-1] = 0
    
    celoss = CrossEntropyLoss(vocab_size)
    loss = celoss(probs, targets)
    print("Basic loss (no padding):", loss.data)
    loss.backward()
    print(logits)
    logits.grad = np.zeros_like(logits.grad)
    
    # Test case 2: With pad token
    pad_token = 0
    celoss_pad = CrossEntropyLoss(vocab_size, pad_token=pad_token)
    loss_pad = celoss_pad(probs, targets_pad)
    print("Loss with padding:", loss_pad.data)
    loss_pad.backward()
    print(logits)
    