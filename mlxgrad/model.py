"""
A minimal GPT built entirely out of mlxgrad's custom Metal-kernel autograd engine.

This mirrors ``cudagrad/model.py`` (and ``tritongrad/model.py``): token + position
embeddings, a stack of pre-norm residual layers each with multi-head self-attention
+ an MLP, a final norm and an output projection. Every bit of forward/backward math
runs through MLXTensor ops and the custom Metal kernels in ``kernels/`` — the only
difference from cudagrad is the kernel backend (Metal vs raw CUDA C++).

Attention uses the fused causal ``FlashAttention`` module, so the causal masking
happens inside the kernel and we never materialize the full N x N score matrix.

Runs on Apple Silicon (no GPU box required) — ``python -m mlxgrad.model`` is a
self-contained smoke test.
"""
import numpy as np
import mlx.core as mx

from engine import MLXTensor, DEVICE
import nn


def _as_int_tokens(tokens):
    """coerce a numpy/MLXTensor of token ids into an MLXTensor (the embedding
    kernel re-casts the fp32 ids to int on load, exactly like cudagrad)."""
    if isinstance(tokens, MLXTensor):
        return tokens
    arr = np.asarray(tokens)
    if arr.ndim == 1:  # a single unbatched sequence -> add a batch dim
        arr = arr[None, :]
    return MLXTensor(mx.array(arr.astype(np.int32)))


class MultiHeadSelfAttention(nn.Module):
    """Projects x into Q/K/V, splits into heads, runs fused causal FlashAttention,
    then re-mixes the heads with an output projection."""
    def __init__(self, model_dim, num_heads, head_dim, device=None):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.Wq = nn.Linear(model_dim, num_heads * head_dim, device=device)
        self.Wk = nn.Linear(model_dim, num_heads * head_dim, device=device)
        self.Wv = nn.Linear(model_dim, num_heads * head_dim, device=device)
        self.Wo = nn.Linear(num_heads * head_dim, model_dim, device=device)

        # standard attention scale; FlashAttention multiplies the Q@K^T logits by this
        self.scale = head_dim ** -0.5

        self.attn = nn.FlashAttention()

    def children(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo, self.attn]

    def __call__(self, x):
        B, N, D = x.shape
        assert D == self.model_dim

        def split_heads(proj):
            # (B, N, H*Dh) -> (B, N, H, Dh) -> (B, H, N, Dh). contiguous() is a
            # no-op in mlxgrad (MLX makes kernel inputs row-contiguous on launch).
            t = proj.reshape((B, N, self.num_heads, self.head_dim))
            t = t.transpose(1, 2)
            return t.contiguous()

        q = split_heads(self.Wq(x))
        k = split_heads(self.Wk(x))
        v = split_heads(self.Wv(x))

        o = self.attn(q, k, v, scale=self.scale)        # (B, H, N, Dh)

        # merge the heads back: (B, H, N, Dh) -> (B, N, H, Dh) -> (B, N, H*Dh)
        o = o.transpose(1, 2).reshape((B, N, self.num_heads * self.head_dim))
        return self.Wo(o)

    def __repr__(self):
        return f"MultiHeadSelfAttention(model_dim={self.model_dim}, num_heads={self.num_heads}, head_dim={self.head_dim})"


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=None):
        super().__init__()
        self.up = nn.Linear(input_dim, hidden_dim, device=device)
        self.down = nn.Linear(hidden_dim, output_dim, device=device)

    def children(self):
        return [self.up, self.down]

    def __call__(self, x):
        return self.down(self.up(x).relu())

    def __repr__(self):
        return "MultiLayerPerceptron"


class ResidualLayer(nn.Module):
    """pre-norm transformer block: x = x + attn(norm(x)); x = x + mlp(norm(x))"""
    def __init__(self, model_dim, num_heads, head_dim, mlp_mult, device=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim, device=device)
        self.mhsa = MultiHeadSelfAttention(model_dim, num_heads, head_dim, device=device)
        self.ln2 = nn.LayerNorm(model_dim, device=device)
        self.mlp = MultiLayerPerceptron(model_dim, mlp_mult * model_dim, model_dim, device=device)

    def children(self):
        return [self.ln1, self.mhsa, self.ln2, self.mlp]

    def __call__(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def __repr__(self):
        return f"ResidualLayer(\n  {self.ln1}\n  {self.mhsa}\n  {self.ln2}\n  {self.mlp}\n)"


class GPT(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = DEVICE if device is None else device
        self.max_seq_len = config['max_seq_len']
        self.vocab_len = config['vocab_len']

        self.tok_embeddings = nn.Embedding(config['vocab_len'], config['model_dim'], device=device)
        self.pos_embeddings = nn.Embedding(config['max_seq_len'], config['model_dim'], device=device)

        self.layers = [
            ResidualLayer(config['model_dim'], config['num_heads'], config['head_dim'],
                          config['mlp_mult'], device=device)
            for _ in range(config['num_layers'])
        ]

        self.final_norm = nn.LayerNorm(config['model_dim'], device=device)
        self.output_proj = nn.Linear(config['model_dim'], config['vocab_len'], device=device)

        # constant (non-trainable) scalars, kept as (1,)-shaped MLXTensors so they
        # flow through our broadcasting binary-op kernels (no high-level MLX math)
        self.scale = MLXTensor(mx.array([config['model_dim'] ** -0.5]), requires_grad=False)
        # a tiny epsilon added before log() in the loss so a confidently-wrong softmax
        # output that underflows to exactly 0 can't produce log(0) = -inf (and then a
        # 0 * -inf = nan once multiplied by the one-hot targets)
        self.eps = MLXTensor(mx.array([1e-9]), requires_grad=False)

    def children(self):
        return self.layers + [self.tok_embeddings, self.pos_embeddings,
                              self.final_norm, self.output_proj]

    def __call__(self, input_token_ids, target_token_ids=None):
        tokens = _as_int_tokens(input_token_ids)
        B, S = tokens.shape
        assert S <= self.max_seq_len, f"sequence length {S} exceeds max_seq_len {self.max_seq_len}"

        # token + (learned) position embeddings. positions are [0, 1, ..., S-1]
        # tiled across the batch.
        pos_ids = np.tile(np.arange(S)[None, :], (B, 1))
        pos_tokens = MLXTensor(mx.array(pos_ids.astype(np.int32)))

        x = self.tok_embeddings(tokens)                 # (B, S, D)
        pos = self.pos_embeddings(pos_tokens)           # (B, S, D)
        x = (x + pos) * self.scale

        for layer in self.layers:
            x = layer(x)

        logits = self.output_proj(self.final_norm(x))   # (B, S, V)
        probabilities = logits.softmax()                # (B, S, V)

        loss = None
        if target_token_ids is not None:
            loss = self.cross_entropy(probabilities, target_token_ids)
        return probabilities, loss

    def cross_entropy(self, probabilities, targets):
        """mean negative log-likelihood of the target tokens.

        Built purely from differentiable MLXTensor ops (we can't gather via
        __getitem__ because its backward is a no-op). We select the target
        log-probs with a one-hot mask:
            loss = -mean_over_BS( sum_V( one_hot * log(probs) ) )
        The one-hot tensor is constant input data (requires_grad=False), so all
        the actual math still runs through our Metal kernels.
        """
        B, S, V = probabilities.shape
        targets = np.asarray(targets).astype(np.int64)
        assert tuple(targets.shape) == (B, S), \
            f"targets shape {tuple(targets.shape)} must match (B, S) = {(B, S)}"

        one_hot_data = np.eye(V, dtype=np.float32)[targets]   # (B, S, V)
        one_hot = MLXTensor(mx.array(one_hot_data), requires_grad=False)

        log_probs = (probabilities + self.eps).log()    # (B, S, V)
        picked = (log_probs * one_hot).sum()             # (B, S, V) -> (B, S) (sum over V)
        nll = picked.reshape((1, B * S))                 # (B, S) -> (1, B*S)
        loss = -(nll.mean())                             # mean over B*S -> (1,), then negate
        return loss

    def __repr__(self):
        return f"GPT(\n  layers={len(self.layers)}, " \
               f"model_dim={self.config['model_dim']}, vocab_len={self.vocab_len}\n)"


if __name__ == "__main__":
    # tiny smoke test (runs on Apple Silicon).
    config = {
        'vocab_len': 65,
        'model_dim': 64,
        'max_seq_len': 16,
        'num_heads': 2,
        'head_dim': 32,
        'mlp_mult': 4,
        'num_layers': 2,
    }
    model = GPT(config)
    B, S = 2, config['max_seq_len']
    rng = np.random.default_rng(0)
    inp = rng.integers(0, config['vocab_len'], size=(B, S))
    tgt = rng.integers(0, config['vocab_len'], size=(B, S))
    probs, loss = model(inp, tgt)
    mx.eval(probs.data, loss.data)
    print("probs shape:", probs.shape, "loss:", float(loss.data.item()))
    loss.backward()
    g = model.output_proj.weight.grad
    print("backward ok; example param grad norm:",
          float(mx.sqrt(mx.sum(g * g)).item()))
