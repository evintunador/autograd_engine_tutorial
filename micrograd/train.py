import time
from ops import split_dim
from gpt import GPT

# load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    tinyShakespeare_string = f.read()

# Convert the string to bytes
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
    global train_pointer, val_pointer
    tok_ct = batch_size * seq_len
    dataset = val_dataset if val else train_dataset
    pointer = val_pointer if val else train_pointer
    input_toks = dataset[pointer:pointer + tok_ct] # grabbing sequential data is bad practice but whatever
    target_toks = dataset[pointer + 1:pointer + tok_ct + 1]
    if val:
        val_pointer += tok_ct
    else:
        train_pointer += tok_ct
    input_toks = split_dim(input_toks, (batch_size, seq_len))
    target_toks = split_dim(target_toks, (batch_size, seq_len))
    return input_toks, target_toks

# define the model and all the hyperparameters
config = {
    'vocab_len':v,
    'model_dim':4,
    'max_seq_len':20,
    'num_heads':2,
    'head_dim':2,
    'mlp_mult':2,
    'dropout_rate':0.1,
    'num_layers':1
}
model = GPT(config)

eta = 0.01

batch_size = 8
toks_per_batch = batch_size * config['max_seq_len']
train_iterations = min(split_size // toks_per_batch, 1000)
val_iterations = min((len(tinyShakespeare_chars) - split_size) // toks_per_batch, 50)
val_frequency = train_iterations // val_iterations
print(f'train iterations: {train_iterations}, frequency of validation: {val_frequency}')

# a very simple and nonrandom inference function
def greedy_inference(model, input, gen_len):
    gen_len = min(gen_len, config['max_seq_len'] - len(input) - 1)
    toks = [[encode_dict[c] for c in input]]
    for i in range(gen_len):
        probabilities, _ = model(toks)
        argmax = float('-inf')
        argmax_idx = None
        for i, val in enumerate(probabilities[0][-1]):
            if val.data > argmax:
                argmax_idx = i
                argmax = val.data
        toks[0].append(argmax_idx)
    return "".join(decode_dict[t] for t in toks[0])

if __name__ == "__main__":
    start_time = time.time()
    for i in range(train_iterations):
        # forward pass
        train_input_toks, train_target_toks = get_batch(batch_size, config['max_seq_len'])
        probabilities, train_loss = model(train_input_toks, train_target_toks)
            
        if i % val_frequency == 0:
            val_input_toks, val_target_toks = get_batch(batch_size, config['max_seq_len'], val = True)
            probabilities, val_loss = model(val_input_toks, val_target_toks)
            
            print(f'step {i} | train loss: {train_loss.data:.2f} | val loss: {val_loss.data:.2f} | ' 
                  f'time: {int(time.time() - start_time)}sec | example: {greedy_inference(model, "King Ri", 10)}')
    
        ## backward pass
        #set param gradients to 0
        for p in model.parameters():
            p.grad = 0.0
        # clac gradients
        train_loss.backward()
        # performing a step of SGD
        for p in model.parameters():
            p.data -= eta * p.grad

    # a final display of wht the model has learned (not much)
    greedy_inference(model, "King Ri", 10)