# minigrad

the tensor-valued autograd tier. the purpose of this lesson is for people already comfortable with linear algebra and calculus to learn exactly what's happening in GPTs all the way from the barebones autograd engine up to the GPT operations themselves. Where [`../micrograd/`](../micrograd) makes the basic unit a single scalar `Value`, minigrad makes it a **tensor**: everything is done with [numpy](https://numpy.org) arrays, so one op moves a whole matrix at once and you get to see the real linear algebra. same educational goal as every tier — just enough of an autograd engine to build a small GPT, not a robust general framework.

## file guide
- `engine.py`: the base of our autograd engine — the `Tensor` object, its ops, and the topo-sorted `backward()` pass over numpy arrays
- `nn.py`: the building-block modules (Linear, Embedding, LayerNorm, attention, etc.) you'd take for granted in pytorch
- `model.py`: assembles the modules into a small GPT
- `train.py`: run it to train a tiny autoregressive GPT char-level on `../input.txt`; it overfits a small slice of the corpus to demonstrate that the engine actually learns (the loss falls well below the `ln(vocab_len)` untrained baseline)

## how to run / test
from the repo root:

```bash
pytest tests/ -k minigrad   # check ops & modules against PyTorch (runs on CPU)
python minigrad/train.py    # train a tiny GPT to show the engine learns
```

runs entirely on CPU — no GPU required.
