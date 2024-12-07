full autograd engine with an implementation of a transformer in raw python (no pytorch, not even numpy). The two files `engine.py` and `nn.py` are from Andrej Karpathy's own [micrograd](https://github.com/karpathy/micrograd); the rest are meant to extend his original up to a transformer

file guide:
- `engine.py`: the base of our autograd engine
- `nn.py`: simple matmul, linear layer, and MLP to make a minimal neural network
- `more_ops.py`: simple operations needed before we can move onto the transformer
- `gpt.py`: the self-attention mechanism, residual connection, cross-entropy loss and the GPT itself
- `micrograd_lesson.py`: equivalent to karpathy's own original lesson
- `transformer_lesson.py`: the extension of karpathy's original lesson found in [my own video]()