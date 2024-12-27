# autograd_engine_tutorial
the plan for this repo is to create tutorials on autograd engines for three levels of difficulty:

|                   |                                                                                                                                                                    | micrograd  | minigrad     | autograd (Triton) |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ------------ | ----------------- |
|                   | difficulty level                                                                                                                                                   | beginner   | intermediate | advanced          |
|                   | time commitment                                                                                                                                                    | medium     | small        | large             |
|                   | status                                                                                                                                                             | functional | WIP          | not-yet-started   |
|                   | attempts to (mostly) resemble [PyTorch](https://pytorch.org) syntax                                                                                                | ❌          | ✅            | TBD               |
|                   |                                                                                                                                                                    |            |              |                   |
| prerequisites     | basic python                                                                                                                                                       | ✅          | ✅            | ✅                 |
|                   | a tiny bit of introductory calculus (what a derivative is conceptually)                                                                                            | ✅          | ✅            | ✅                 |
|                   | a tiny bit of introductory linear algebra (matrices and matrix multiplication)                                                                                     | ❌          | ✅            | ✅                 |
|                   | the python package [numpy](https://numpy.org)                                                                                                                      | ❌          | ✅            | ❌                 |
|                   | linux (or use [colab](https://colab.research.google.com), [lambda](https://lambdalabs.com) or similar)                                                             | ❌          | ❌            | ✅                 |
|                   | Nvidia GPU (or use [colab](https://colab.research.google.com), [lambda](https://lambdalabs.com) or similar)                                                        | ❌          | ❌            | ✅                 |
|                   |                                                                                                                                                                    |            |              |                   |
| What you'll learn | basic math of autograd systems                                                                                                                                     | ✅          | ✅            | ✅                 |
|                   | basic math of [GPT-2](https://en.wikipedia.org/wiki/GPT-2#:~:text=Generative%20Pre%2Dtrained%20Transformer%202,of%208%20million%20web%20pages.) style transformers | ✅          | ✅            | ✅                 |
|                   | basics of parallel programming linear algebra on GPUs                                                                                                              | ❌          | ❌            | ✅                 |
|                   | OpenAI's [Triton](https://triton-lang.org/main/index.html)                                                                                                         | ❌          | ❌            | ✅                 |

## micrograd
the purpose of this lesson is for absolute beginners with a programming (as opposed to math) background to learn about the math and implementation of GPTs all the way from the barebones autograd engine and up to the GPT operations itself. the basic building block of micrograd is the `Value` object, each of which is just a single floating point number for the data and another single floating point number to keep track of the data's gradient. the first half or so of this lesson is roughly equivalent to [karpathy's `micrograd`](https://youtu.be/VMj-3S1tku0?si=FM0qtfV-cvXr2kDJ) while the second half is an extension to implement a full GPT
##### TODO:
- [ ] make max/min functions maintain gradient
- [ ] make model output logits & celoos do softmax within so that inference can do softmax
- [ ] build a way to test against pytorch (or maybe against minigrad, that sounds easier)
- [ ] write jupyter notebook guide
- [ ] make video

## minigrad
the purpose of this lesson is for people already confident with linear algebra and calculus to learn exactly what's happening in GPTs all the way from the barebones autograd engine and up to the GPT operations itself. to make this happen we'll be doing everything with numpy arrays, meaning that the basic unit of our engine will be tensors as opposed to individual values
##### TODO:
- [x] celoss
	- [x] one-hot
- [x] model itself
- [ ] make model output logits & celoos do softmax within so that inference can do softmax
- [x] tokenizer
	- reused from micrograd bc i'm lazy & this isn't a lesson about tokenization
- [x] get batch
	- not sure why i felt the need to swap out micrograd's terrible get_batch funciton for a different but equally terrible one
		- [ ] improve?
- [ ] inference
- [ ] train loop
- [ ] build a way to test against pytorch (or maybe against micrograd or autograd, that sounds easier)
- [ ] *optional:* code to convert trained model to pytorch and show how much faster it is?
- [ ] write jupyter notebook guide
- [ ] make video

## autograd (Triton)
the purpose of this lesson is for people already confident in the math behind autograd engines and GPTs to learn exactly what's happening at the level of the GPU hardware. you can think of autograd as a replication of pytorch/tensorflow/jax/mlx/tinygrad/etc except that instead of being meant to actually be used, which would require it be flexible and robust, it is meant for educational purposes. As such, we'll only be implementing the operations that are absolutely necessary in order to create a GPT and not worrying much about unexpected edge cases, just like how we did in micrograd and minigrad
##### TODO:
- [ ] begin learning Triton
	- [ ] realize i'm in way over my head
		- [ ] abandon project lmao

## autograd (CUDA)
I'll consider building this if 1) the other three perform well 2) people are interested and 3) i become a masochist

## autograd (MPS)
I'll consider building this if 1) the other three perform well 2) people are interested and 3) i become a masochist
