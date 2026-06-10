import time
import math
from ops import split_dim
from gpt import GPT

# load the dataset
with open('../input.txt', 'r', encoding='utf-8') as f:
    tinyShakespeare_string = f.read()

# an atrocious terrible no-good tokenizer (sorted so the mapping is deterministic run-to-run)
unique_chars = sorted(set(tinyShakespeare_string))
v = len(unique_chars)
encode_dict, decode_dict = {}, {}
for i, c in enumerate(unique_chars):
    encode_dict[c] = i
    decode_dict[i] = c
tinyShakespeare_chars = [encode_dict[c] for c in tinyShakespeare_string]

# define the model and all the hyperparameters.
# micrograd is pure-python scalar autograd, so every Value op is a slow Python object;
# the model is therefore kept small and we train on a tiny slice to keep the demo snappy
# while still clearly demonstrating that the engine learns.
config = {
    'vocab_len':v,
    'model_dim':16,
    'max_seq_len':8,
    'num_heads':4,
    'head_dim':4,
    'mlp_mult':4,
    'dropout_rate':0.0,
    'num_layers':2
}
model = GPT(config)
print(f'vocab size: {v} | model parameters: {len(model.parameters())}')

eta = 0.3            # learning rate (kept modest so vanilla SGD descends smoothly)
batch_size = 1
seq_len = config['max_seq_len']
toks_per_batch = batch_size * seq_len

# carve a small training set out of the front of the corpus and overfit it: because each
# Value op is so expensive we can't sweep the whole 1M-character corpus, but cycling over a
# tiny slice for many passes drives the loss far below the ln(vocab_len) baseline of an
# untrained model -- and overfitting this hard means greedy decoding actually REPRODUCES the
# memorized text, a clear end-to-end sanity check that gradients flow correctly. The corpus
# starts "First Citizen", so the first 8-token window is "First Ci"; after overfitting,
# greedily continuing the prompt "First" should regenerate "First Ci".
num_batches = 1
epochs = 30
sample_every = 5

def make_batches(n):
    '''slice n consecutive (input, target) batches off the front of the corpus'''
    batches = []
    for b in range(n):
        start = b * toks_per_batch
        chunk = tinyShakespeare_chars[start:start + toks_per_batch + 1]
        if len(chunk) < toks_per_batch + 1:
            break
        inp = split_dim(chunk[:toks_per_batch], (batch_size, seq_len))
        tgt = split_dim(chunk[1:toks_per_batch + 1], (batch_size, seq_len))
        batches.append((inp, tgt))
    return batches

batches = make_batches(num_batches)
print(f'training on {len(batches)} batches for {epochs} epochs '
      f'(untrained-model loss baseline ~= ln({v}) = {math.log(v):.2f})')

# a very simple and nonrandom inference function
def greedy_inference(model, prompt, gen_len):
    gen_len = min(gen_len, config['max_seq_len'] - len(prompt))
    toks = [[encode_dict[c] for c in prompt]]
    for _ in range(gen_len):
        logits, _ = model(toks)              # model now emits raw logits
        last = logits[0][-1]                 # logits for the next token
        # argmax over the logits == argmax over softmax(logits), so no need to normalize
        argmax_idx, argmax = 0, float('-inf')
        for j, val in enumerate(last):
            if val.data > argmax:
                argmax_idx, argmax = j, val.data
        toks[0].append(argmax_idx)
    return "".join(decode_dict[t] for t in toks[0])

if __name__ == "__main__":
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inp, tgt in batches:
            # forward pass
            _, loss = model(inp, tgt)
            epoch_loss += loss.data

            # backward pass
            for p in model.parameters():
                p.grad = 0.0
            loss.backward()
            # a step of vanilla SGD
            for p in model.parameters():
                p.data -= eta * p.grad

        # sampling runs extra forward passes, so only do it occasionally
        if epoch % sample_every == 0 or epoch == epochs - 1:
            sample = greedy_inference(model, "First", 3)
            print(f'epoch {epoch:2d} | mean train loss: {epoch_loss / len(batches):.3f} | '
                  f'time: {int(time.time() - start_time)}sec | greedy("First")={sample!r}')
        else:
            print(f'epoch {epoch:2d} | mean train loss: {epoch_loss / len(batches):.3f} | '
                  f'time: {int(time.time() - start_time)}sec')

    # the corpus begins "First Citizen", so a model that has overfit batch 0 should
    # greedily continue "First" -> "First Ci"
    print(f'\nfinal greedy sample from "First": {greedy_inference(model, "First", 3)!r}')
