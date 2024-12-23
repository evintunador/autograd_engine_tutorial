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
    
    def parameters(self):
        return [self.up, self.down]

    def __repr__(self):
        return f"MLP of\nUp Projection: ({self.up.w.shape})\nDown Projection: ({self.down.w.shape})"
    

if __name__ == "__main__":
    batch_size = 2
    dim = 8
    vocab_len = 5
    seq_len = 10
    heads = 4
    head_dim = 2

    print("---------------- test mlp ----------------")
    x = Tensor(np.random.randn(batch_size, seq_len, dim))
    mlp = MultiLayerPerceptron(dim, 4*dim, dim)
    print(mlp)
    y = mlp(x)
    print(y.shape == x.shape)