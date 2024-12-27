import random
import time
import numpy as np

from model import GPT

# load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    tinyShakespeare_string = f.read()

# an atrocious terrible no-good tokenizer
unique_chars = set(tinyShakespeare_string)
v = len(unique_chars)
encode_dict, decode_dict = {}, {}
for i, c in enumerate(unique_chars):
    encode_dict[c] = i
    decode_dict[i] = c
tinyShakespeare_chars = [encode_dict[c] for c in tinyShakespeare_string]

# split into train vs validation datasets
split_size = int(0.95 * len(tinyShakespeare_chars))
train_dataset, val_dataset = tinyShakespeare_chars[:split_size], tinyShakespeare_chars[split_size:]

# grab batch from datasets
train_pointer, val_pointer = 0, 0
def get_batch(batch_size, seq_len, val = False):
    '''an atrocious terrible no-good way to get data batches'''
    global train_pointer, val_pointer
    dataset_size = len(tinyShakespeare_chars) - split_size if val else split_size
    dataset = val_dataset if val else train_dataset
    pointer = val_pointer if val else train_pointer
    input_toks, target_toks = [], []
    for b in range(batch_size):
        input_toks.append([t for t in dataset[pointer + (b * seq_len):pointer + (b * seq_len) + seq_len]])
        target_toks.append([t for t in dataset[pointer + (b * seq_len) + 1:pointer + (b * seq_len) + seq_len + 1]])
    tok_ct = batch_size * seq_len
    if val:
        val_pointer += tok_ct
    else:
        train_pointer += tok_ct
    return np.array(input_toks), np.array(target_toks)


# define the model and all the hyperparameters
config = {
    'vocab_len':v,
    'model_dim':32,
    'max_seq_len':20,
    'num_heads':4,
    'head_dim':8,
    'mlp_mult':4,
    'dropout_rate':0.1,
    'num_layers':2
}
model = GPT(config)

eta = 0.01 # learning rate

batch_size = 16
toks_per_batch = batch_size * config['max_seq_len']
train_iterations = min(split_size // toks_per_batch, 10_000)
val_iterations = min((len(tinyShakespeare_chars) - split_size) // toks_per_batch, 50)
val_frequency = train_iterations // val_iterations
print(f'train iterations: {train_iterations}, frequency of validation: {val_frequency}')

# a very simple and nonrandom inference function
def greedy_inference(model, input, gen_len):
    gen_len = min(gen_len, config['max_seq_len'] - len(input) - 1)
    toks = [encode_dict[c] for c in input]
    for i in range(gen_len):
        probabilities, _ = model(np.array([toks]))
        toks.append(probabilities.max()[1][-1])
    return "".join(decode_dict[t] for t in toks)

if __name__ == "__main__":
    start_time = time.time()
    for i in range(train_iterations):
        # forward pass
        train_input_toks, train_target_toks = get_batch(batch_size, config['max_seq_len'])
        probabilities, train_loss = model(train_input_toks, train_target_toks)
            
        if i % val_frequency == 0:
            val_input_toks, val_target_toks = get_batch(batch_size, config['max_seq_len'], val = True)
            probabilities, val_loss = model(val_input_toks, val_target_toks)
            
            print(f'step {i} | train loss: {train_loss.data} | val loss: {val_loss.data} | ' 
                    f'time: {int(time.time() - start_time)}sec | example: {greedy_inference(model, "King Ri", 10)}')

        ## backward pass
        #set param gradients to 0
        for p in model.parameters():
            p.zero_grad()
        # clac gradients
        train_loss.backward()
        # performing a step of SGD
        for p in model.parameters():
            p.data -= eta * p.grad

    # a final display of wht the model has learned (not much)
    greedy_inference(model, "King Ri", 10)