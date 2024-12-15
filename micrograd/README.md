full autograd engine with an implementation of a transformer in raw python (no pytorch, not even numpy). The two file `engine.py` and some parts of `modules.py` are from Andrej Karpathy's own [micrograd](https://github.com/karpathy/micrograd); the rest are meant to extend his original lesson up to and including a full GPT

file guide:
- `engine.py`: the base of our autograd engine
- `ops.py`: simple operations needed before we can move on
- `modules.py`: simple modules that you'd take for granted when using something like pytorch
- `gpt.py`: the multi-layer perceptron, self-attention mechanism, residual layer, aand the GPT itself
- `engine_lesson.py`: roughly equivalent to karpathy's own original lesson
- `mlp_demo.py`: roughly equivalent to karpathy's own original lesson
- `gpt_lesson.py`: the extension of karpathy's original lesson; takes up the bulk of [my own video]()
- `train.py`: run it to train an absurdly tiny autoregressive GPT; when i say absurdly tiny i mean it's literally too small and slow to learn anything