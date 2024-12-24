import numpy as np

from engine import Tensor, Parameter
import nn
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, dropout_rate, mask):
        super().__init__()
        self.model_dim = model_dim # referred to as "d" in shapes
        self.num_heads = num_heads # referred to as "nh" in shapes
        self.head_dim = head_dim # referred to as "hd" in shapes
        
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
        masked_logits = scaled_logits.masked_fill(mask[:s,:s], float('-inf'))
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
    
class ResidualLayer(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, dropout_rate, mask, mlp_mult):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.mhsa = MultiHeadSelfAttention(model_dim, num_heads, head_dim, dropout_rate, mask)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = MultiLayerPerceptron(model_dim, mlp_mult * model_dim, model_dim)
    
    def children(self):
        return [self.ln1, self.mhsa, self.ln2, self.mlp]

    def __repr__(self):
        return f"Residual Layer of:\n{self.ln1}\n{self.mhsa}\n{self.ln2}\n{self.mlp}"

    def __call__(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config['max_seq_len'] # we'll be using this one in __call__

        self.tok_embeddings = nn.Embedding(config['vocab_len'], config['model_dim'])
        self.scale = config['model_dim'] ** -0.5
        self.pos_embeddings = nn.Embedding(config['max_seq_len'], config['model_dim'])

        self.mask = np.triu(np.ones((config['max_seq_len'], config['max_seq_len'])), k=1).astype(bool)
        self.layers = [ResidualLayer(config['model_dim'], 
                                     config['num_heads'], 
                                     config['head_dim'], 
                                     config['dropout_rate'],
                                     self.mask,
                                     config['mlp_mult']) 
                       for _ in range(config['num_layers'])]

        self.final_norm = nn.LayerNorm(config['model_dim'])
        self.output_proj = nn.Linear(config['model_dim'], config['vocab_len'])
        self.criterion = nn.CrossEntropyLoss(config['vocab_len'], pad_token = None)#config['vocab_len'] - 1)

    def children(self):
        return [self.tok_embeddings, self.pos_embeddings, self.output_proj, self.criterion] + self.layers

    def __call__(self, input_token_ids, target_token_ids = None):
        B, S = input_token_ids.shape
        if input_token_ids.ndim == 1: # if only one sequence is passed in, aka batch_size==1
            input_tokens = input_tokens.unsqueeze(0)

        if target_token_ids is not None: # if training
            assert B, S == target_token_ids.shape
            assert S == self.max_seq_len
        else: # if inference
            assert S <= self.max_seq_len

        x = self.tok_embeddings(input_token_ids)
        pos = self.pos_embeddings(np.array([range(S)])).broadcast_to(x.shape)
        x = (x + pos) * self.scale

        for layer in self.layers:
            x = layer(x)

        logits = self.output_proj(self.final_norm(x))
        probabilities = logits.softmax()

        loss = None
        if target_token_ids is not None:
            loss = self.criterion(probabilities, target_token_ids)
        
        return probabilities, loss

if __name__ == "__main__":
    b = 2
    dim = 8
    vocab_len = 5
    seq_len = 7
    max_seq_len = 10
    num_heads = 4
    head_dim = 2
    dropout_rate = 0.1

    print("\n\n---------------- test mlp ----------------")
    x = Tensor(np.random.randn(b, seq_len, dim))
    mlp = MultiLayerPerceptron(dim, 4*dim, dim)
    print(mlp)
    y = mlp(x)
    print(y.shape == x.shape)

    print("\n\n---------------- test mhsa ----------------")
    x = Tensor(np.random.randn(b, seq_len, dim))
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    mhsa = MultiHeadSelfAttention(dim, num_heads, head_dim, dropout_rate, mask)
    print(mhsa)
    y = mhsa(x)
    print(x.shape == y.shape)

    print("\n\n---------------- test residual ----------------")
    x = Tensor(np.random.randn(b, seq_len, dim))
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    layer = ResidualLayer(dim, num_heads, head_dim, dropout_rate, mask, mlp_mult = 4)
    print(layer)
    y = layer(x)
    print(x.shape == y.shape)

    print('\n\n-------------- test gpt model -------------')
    batch_size = 2
    config = {
        'vocab_len':10,
        'model_dim':8,
        'max_seq_len':5,
        'num_heads':2,
        'head_dim':4,
        'mlp_mult':4,
        'dropout_rate':0.1,
        'num_layers':2
    }
    gpt = GPT(config)
    input_token_ids = np.random.randint(0, config['vocab_len'], size=(batch_size, config['max_seq_len']))
    target_token_ids = np.random.randint(0, config['vocab_len'], size=(batch_size, config['max_seq_len']))
    probabilities, loss = gpt(input_token_ids, target_token_ids)
    loss.backward()
    print(loss)
    print(probabilities)