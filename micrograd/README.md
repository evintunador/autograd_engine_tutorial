full autograd engine with an implementation of a transformer in raw python (no pytorch, not even numpy). The two file `engine.py` and some parts of `modules.py` are from Andrej Karpathy's own [micrograd](https://github.com/karpathy/micrograd); the rest are meant to extend his original up to a transformer

file guide:
- `engine.py`: the base of our autograd engine
- `ops.py`: simple operations needed before we can move onto the transformer
- `modules.py`: simple modules that you'd take for granted when using something glike pytorch
- `gpt.py`: the multi-layer perceptron, self-attention mechanism, residual connection, aand the GPT itself
- `micrograd_lesson.py`: currently equivalent to karpathy's own original lesson
- `transformer_lesson.py`: the extension of karpathy's original lesson found in [my own video]()