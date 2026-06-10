import random as r
from engine import Value
from ops import *

class Module: # just to make our syntax the same as pytorch's
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, input_dim):
        self.w = [Value(r.gauss(0,0.02)) for _ in range(input_dim)]
        self.b = Value(0.0)

    def __call__(self, x):
        assert len(x) == len(self.w), f'mismatch between input dim {len(x)} and weight dim {len(self.w)}'
        # w * x + b
        wixi = [wi*xi for wi, xi in zip(self.w, x)]
        
        tot = wixi[0] # for some reason sum() gives me an error so i do the addition manually
        for i in wixi[1:]: 
            tot = tot + i
        
        act = tot + self.b
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self.neurons = [Neuron(input_dim) for _ in range(output_dim)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class Embedding(Module):
    def __init__(self, num_classes: int, dim: int):
        self.weight = [[Value(r.gauss(0,0.02)) for _ in range(dim)] 
                       for _ in range(num_classes)]

    def __call__(self, x):
        assert isinstance(x, list), "x should be a list of integers"
        assert all(isinstance(idx, int) for idx in x), "All elements in x must be integers"
        # grab embedding assigned to each token
        out = [self.weight[idx] for idx in x]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        weights_repr = "\n".join(
            f"[{', '.join(str(p) for p in row)}]" for row in self.weight
        )
        return f"Embedding with weights:\n{weights_repr}"

    def parameters(self):
        return [p for row in self.weight for p in row]

class LayerNorm(Module):
    '''
    Layer normalization over the final (feature) dimension, with the learnable
    affine transform that torch.nn.LayerNorm applies by default:
        out = (x - mean) / sqrt(var + eps) * weight + bias
    where mean/var are the population statistics over the vector (unbiased=False).
    Like the other modules here it operates on a single vector of Value objects;
    use vector_wise_apply to run it over a whole tensor.
    '''
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = [Value(1.0) for _ in range(dim)] # gamma, init to ones
        self.bias = [Value(0.0) for _ in range(dim)]   # beta, init to zeros

    def __call__(self, x):
        assert isinstance(x, list), "x should be a list of Value objects"
        assert all(isinstance(xi, Value) for xi in x), "All elements in x must be Value objects"
        assert len(x) == self.dim, f'input dim {len(x)} must match LayerNorm dim {self.dim}'
        n = len(x)
        # mean over the vector
        mu = x[0] / n
        for xi in x[1:]:
            mu = mu + (xi / n)
        # population variance over the vector
        tot = (x[0] - mu) ** 2
        for xi in x[1:]:
            tot = tot + (xi - mu) ** 2
        var = tot / n
        # reciprocal standard deviation (note: we DIVIDE by std, hence the +eps under the root)
        rstd = (var + self.eps) ** (-0.5)
        # normalize, then apply the learnable affine transform
        return [(x[i] - mu) * rstd * self.weight[i] + self.bias[i] for i in range(n)]

    def parameters(self):
        return self.weight + self.bias

    def __repr__(self):
        return f"LayerNorm({self.dim})"

class CrossEntropyLoss(Module):
    def __init__(self, vocab_len: int, pad_token: int = None):
        self.vocab_len = vocab_len
        self.pad_token = pad_token

    def __call__(self, logits, targets):
        '''
        inputs:
        logits - list of lists of lists of shape (batch_size, seq_len, vocab_len) full of Value objects.
                 these are RAW scores straight out of the model -- we apply log-softmax internally so the
                 model can stay an unnormalized logit producer (handy for inference).
        targets - list of lists of shape (batch_size, seq_len) full of integers representing token indices

        output: a single Value object representing the mean negative-log-likelihood loss
                (matches torch.nn.CrossEntropyLoss default reduction='mean')
        '''
        assert isinstance(targets, list) and isinstance(targets[0], list) and isinstance(targets[0][0], int)
        assert len(logits) == len(targets) and len(logits[0]) == len(targets[0])

        one_hots = vector_wise_apply(self._one_hot, targets)
        # log-softmax turns logits into log-probabilities in one numerically-stable op
        log_probabilities = vector_wise_apply(log_softmax, logits)
        individual_losses = entry_wise_mult(one_hots, log_probabilities)

        # sum the per-token log-likelihoods across every dimension down to a single Value
        total = vector_wise_apply(sum, vector_wise_apply(sum, vector_wise_apply(sum, individual_losses)))

        # average over the number of (non-pad) target tokens, then negate -> mean NLL
        count = 0
        for seq in targets:
            for t in seq:
                if t != self.pad_token:
                    count += 1
        return (-1.0 / count) * total

    def _one_hot(self, targets_vec):
        '''
        turns list of tokens into list of one-hot vectors with 1's at the index of the given token
        meant to be used with vector_wise_apply
        '''
        assert all(isinstance(t, int) for t in targets_vec)
        return [[0] * t + [1] + [0] * (self.vocab_len - t - 1) 
                if t != self.pad_token 
                else [0] * self.vocab_len
                for t in targets_vec]

if __name__ == "__main__":
    batch_size = 2
    vocab_len = 10
    model_dim = 8
    max_seq_len = 5
    seq_len = 3
    num_heads = 2
    head_dim = model_dim // num_heads

    print('\n\n-------------- test embedding on a single input sequence -------------')
    E = Embedding(vocab_len, model_dim)
    print(E)
    print('\n')
    x = E([1,2,3])
    pretty_tensor_print(x)
    print('\n\n-------------- test embedding on a batch of input sequences -------------')
    E = Embedding(vocab_len, model_dim)
    print(E)
    print('\n')
    input_tokens = [[r.randint(0,vocab_len-1) for _ in range(seq_len)] for _ in range(batch_size)]
    x = vector_wise_apply(E, input_tokens)
    pretty_tensor_print(x)
    
    print('\n\n-------------- test linear layer on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    w = Linear(model_dim, head_dim)
    y = w(x)
    print(y)
    print('\n\n-------------- test linear layer on a tensor -------------')
    x = [[[Value(r.gauss(0,0.02)) for _ in range(model_dim)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(x)
    print('\n')
    y = vector_wise_apply(w, x)
    pretty_tensor_print(y)

    print('\n\n-------------- test cross-entropoy loss -------------')
    # CrossEntropyLoss now takes raw logits and applies log-softmax internally
    logits = [[[Value(r.gauss(0,0.02)) for _ in range(vocab_len)] for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(logits)
    celoss = CrossEntropyLoss(vocab_len, pad_token = vocab_len - 1)
    targets = [[r.randint(0, vocab_len - 1) for _ in range(seq_len)] for _ in range(batch_size)]
    pretty_tensor_print(targets)
    loss = celoss(logits, targets)
    print(loss)

    print('\n\n-------------- test layernorm on a vector -------------')
    x = [Value(r.gauss(0,0.02)) for _ in range(model_dim)]
    print(x)
    ln = LayerNorm(model_dim)
    print(ln(x))