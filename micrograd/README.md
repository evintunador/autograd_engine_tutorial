# micrograd

full autograd engine with an implementation of a transformer in raw python (no pytorch, not even numpy). The file `engine.py` and some parts of `modules.py` are from Andrej Karpathy's own [micrograd](https://github.com/karpathy/micrograd); the rest are meant to extend his original lesson up to and including a full GPT.

This is the beginner tier: the basic building block is the `Value` object, each of which is just a single floating point number for the data plus another for its gradient. The first half is roughly equivalent to [Karpathy's `micrograd`](https://youtu.be/VMj-3S1tku0); the second half extends it into a full GPT.

## file guide
- `engine.py`: the base of our autograd engine
- `ops.py`: simple operations needed before we can move on
- `modules.py`: simple modules that you'd take for granted when using something like pytorch
- `gpt.py`: the multi-layer perceptron, self-attention mechanism, residual layer, and the GPT itself
- `train.py`: run it to train a tiny autoregressive GPT; it's still small and slow (pure-python scalar autograd), so it overfits a small slice of the corpus to demonstrate that the engine actually learns (the loss falls well below the `ln(vocab_len)` untrained baseline)

## how to run / test
from the repo root:

```bash
pytest tests/ -k micrograd   # check ops & modules against PyTorch (runs on CPU)
python micrograd/train.py    # train a tiny GPT to show the engine learns
```

## videos
- Andrej Karpathy's [micrograd video](https://youtu.be/VMj-3S1tku0) — the recommended starting point for autograd basics.
- this tier's companion walkthrough *(TODO: link)*.
