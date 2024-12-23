import numpy as np

from engine import Tensor, Parameter
import nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.up = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        return self.down(self.up(x).relu())
    
    def children(self):
        return [self.up, self.down]

    def __repr__(self):
        return f"MLP of\nUp Projection: ({self.up.w.shape})\nDown Projection: ({self.down.w.shape})"
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, max_seq_len, dropout_rate, mask):
        super().__init__()
        self.model_dim = model_dim # referred to as "d" in shapes
        self.num_heads = num_heads # referred to as "nh" in shapes
        self.head_dim = head_dim # referred to as "hd" in shapes
        self.max_seq_len = max_seq_len
        
        self.Wq = nn.Linear(model_dim, num_heads * head_dim)
        self.Wk = nn.Linear(model_dim, num_heads * head_dim)
        self.Wv = nn.Linear(model_dim, num_heads * head_dim)
        
        self.scale = head_dim ** -0.5

        self.mask = mask
        
        self.Wo = nn.Linear(num_heads * head_dim, model_dim)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
    
    def children(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo, self.drop1, self.drop2]

    def __repr__(self):
        return f"MHSA of\nWq: ({self.Wq.w.shape})\nWk: ({self.Wk.w.shape})"\
                f"\nWv: ({self.Wv.w.shape})\nWo: ({self.Wo.w.shape})"
    
    def __call__(self, x):
        b, s, d = x.shape
        assert self.model_dim == d,\
            f"input final dimension {d} must equal MHSA mechanism's given model_dim value at initialization of {self.model_dim}"
        
        # get our query, key and value projections
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
            # (b, s, d) @ (d, nh * hd) -> (b, s, hd)

        # split apart our separate heads
        q = q.reshape((b, s, self.num_heads, self.head_dim)) # (b, s, hd) -> (b, s, nh, hd)
        k = k.reshape((b, s, self.num_heads, self.head_dim))
        v = v.reshape((b, s, self.num_heads, self.head_dim))

        # transpose to put s in the path of the matmul for our attention computation
        q = q.transpose((0,2,1,3)) # (b, s, nh, hd) -> (b, nh, s, hd)
        k = k.transpose((0,2,1,3))
        v = v.transpose((0,2,1,3))

        # prepare keys for attention calc
        k = k.transpose() # (b, nh, s, hd) -> (b, nh, hd, s)
        # compute attention logits
        logits = q @ k # (b, nh, s, hd) @ (b, nh, hd, s) -> (b, nh, s, s)
        # scale logits
        scaled_logits = logits * self.scale
        # apply mask
        masked_logits = scaled_logits.masked_fill(mask, float('-inf'))
        # turn the logits into probability scores
        scores = masked_logits.softmax()
        # dropout; if we're training then dropout_rate>0 but if doing inference it'll be set ==0 in the model class
        scores = self.drop1(scores)

        # use scores to select from values
        output_values = scores @ v # (b, nh, s, s) @ (b, nh, s, hd) -> (b, nh, s, hd)
        # rearrange back to be of size model_dim
        output_values = output_values.transpose((0,2,1,3)) # (b, nh, s, hd) -> (b, s, nh, hd)
        output_values = output_values.reshape((b, s, self.num_heads * self.head_dim)) # (b, s, nh, hd) -> (b, s, nh * hd)

        # mix output values of each head together
        out = self.Wo(output_values) # (b, s, nh * hd) @ (nh * hd, d) -> (b, s, d)
        # before returning, dropout IF we're training
        return self.drop2(out)

if __name__ == "__main__":
    b = 2
    dim = 8
    vocab_len = 5
    seq_len = 7
    max_seq_len = 10
    num_heads = 4
    head_dim = 2
    dropout_rate = 0.1

    print("---------------- test mlp ----------------")
    x = Tensor(np.random.randn(b, seq_len, dim))
    mlp = MultiLayerPerceptron(dim, 4*dim, dim)
    print(mlp)
    y = mlp(x)
    print(y.shape == x.shape)

    print("---------------- test mhsa ----------------")
    x = Tensor(np.random.randn(b, seq_len, dim))
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    mhsa = MultiHeadSelfAttention(dim, num_heads, head_dim, max_seq_len, dropout_rate, mask)
    print(mhsa)
    y = mhsa(x)
    print(x.shape == y.shape)