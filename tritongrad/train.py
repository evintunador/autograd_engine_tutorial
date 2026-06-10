"""
Character-level training of our Triton-kernel GPT on tiny-shakespeare (../input.txt).

This mirrors ``minigrad/train.py`` but runs on the GPU with TritonTensors. The goal
is simply to demonstrate that the engine learns: the training loss should fall well
below its random-init starting point of ln(vocab_len).

REQUIRES a CUDA GPU + Triton. Run from the ``tritongrad/`` directory:
    python train.py
"""
import os
import time

import numpy as np
import torch

from engine import TritonTensor, DEVICE
from model import GPT

# ---------------------------------------------------------------------------
# data + tokenizer
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(HERE, "..", "input.txt")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# a simple deterministic char-level tokenizer (sorted so encode/decode are stable
# across runs, unlike minigrad's set()-based version)
chars = sorted(set(text))
vocab_len = len(chars)
encode_dict = {c: i for i, c in enumerate(chars)}
decode_dict = {i: c for i, c in enumerate(chars)}
data = np.array([encode_dict[c] for c in text], dtype=np.int64)

split = int(0.95 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(batch_size, seq_len, val=False):
    """grab a batch of random (input, target) windows; targets are inputs shifted by 1"""
    source = val_data if val else train_data
    # rng is seeded below for reproducibility
    ix = rng.integers(0, len(source) - seq_len - 1, size=batch_size)
    x = np.stack([source[i:i + seq_len] for i in ix])
    y = np.stack([source[i + 1:i + seq_len + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# model + hyperparameters
# ---------------------------------------------------------------------------
config = {
    'vocab_len': vocab_len,
    'model_dim': 64,
    'max_seq_len': 64,
    'num_heads': 2,
    'head_dim': 32,        # MUST be in {32, 64, 128, 256} for FlashAttention
    'mlp_mult': 4,
    'num_layers': 2,
}

seed = 1234
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

model = GPT(config)

eta = 0.05                 # SGD learning rate (plain SGD; lower = safer against divergence)
batch_size = 16
seq_len = config['max_seq_len']
train_iterations = min(len(train_data) // (batch_size * seq_len), 3000)
print_every = max(1, train_iterations // 30)
print(f"vocab_len={vocab_len} | params live on {DEVICE} | train_iterations={train_iterations}")
print(f"(random-init cross-entropy should be ~ln({vocab_len}) = {np.log(vocab_len):.3f})")


def greedy_inference(model, prompt, gen_len):
    """argmax-sample gen_len chars after the prompt (forward only, no autograd)."""
    toks = [encode_dict[c] for c in prompt if c in encode_dict]
    for _ in range(gen_len):
        context = toks[-model.max_seq_len:]
        inp = torch.tensor([context], dtype=torch.long, device=DEVICE)
        probs, _ = model(inp)
        next_id = int(probs.argmax()[0, -1].item())
        toks.append(next_id)
    return "".join(decode_dict[t] for t in toks)


if __name__ == "__main__":
    # ---- autotune warmup -------------------------------------------------
    # Triton autotunes each kernel on its first call by benchmarking every candidate
    # config many times; our backward kernels accumulate into .grad with +=, so a
    # naive first backward would compound the gradient ~10^4x. We trigger all the
    # autotuning once here, seeding the backward with ZEROS so nothing real
    # accumulates, then wipe every grad. After this each kernel's chosen config is
    # cached and subsequent backward passes run exactly once.
    print("warming up Triton autotuner (first iteration is slow)...")
    t0 = time.time()
    xb, yb = get_batch(batch_size, seq_len)
    _, warm_loss = model(xb, yb)
    warm_loss.backward(torch.zeros_like(warm_loss.data))
    warm_loss.zero_grad_backward()
    for p in model.parameters():
        p.zero_grad()
    # reset the rng so real training starts from a clean, reproducible stream
    rng = np.random.default_rng(seed)
    print(f"warmup done in {time.time() - t0:.1f}s\n")

    # ---- training loop ---------------------------------------------------
    # NOTE: greedy inference is deliberately NOT run inside the loop. Each new
    # (growing) sequence length forces Triton to re-autotune the forward kernels,
    # which would dwarf training time. We sample once at the end instead.
    start = time.time()
    first_loss = last_loss = None
    for i in range(train_iterations):
        xb, yb = get_batch(batch_size, seq_len)
        _, loss = model(xb, yb)

        # zero param grads, backprop, SGD step
        for p in model.parameters():
            p.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= eta * p.grad

        loss_val = float(loss.data.item())
        last_loss = loss_val
        if first_loss is None:
            first_loss = loss_val

        if i % print_every == 0 or i == train_iterations - 1:
            print(f"step {i:>5} | loss {loss_val:.4f} | {int(time.time() - start)}s")

    print(f"\nloss went from {first_loss:.4f} -> {last_loss:.4f} "
          f"over {train_iterations} steps "
          f"({'DECREASED' if last_loss < first_loss else 'did NOT decrease'})")

    # ---- one greedy sample at the end (display only) ---------------------
    try:
        sample = greedy_inference(model, "JULIET:\n", 80)
        print(f"\nsample after training:\n{sample}")
    except Exception as e:
        print(f"\n(greedy inference skipped: {e})")
