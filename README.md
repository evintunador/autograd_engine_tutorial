# autograd_engine_tutorial
the plan for this repo is to create tutorials on autograd engines for three levels of difficulty:

|                   |                                                                                                                                                                    | micrograd            | minigrad             | autograd<br>(Triton) | autograd<br>(CUDA)     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | -------------------- | ----------------- | ---------------------- |
|                   | difficulty level                                                                                                                                                   | beginner             | intermediate         | advanced          | advanced               |
|                   | time commitment                                                                                                                                                    | medium               | small                | large             | large                  |
|                   | status                                                                                                                                                             | minimally-functional | minimally-functional | WIP               | might not even make it |
|                   | attempts to (mostly) resemble [PyTorch](https://pytorch.org) syntax                                                                                                | ❌                    | ✅                    | ✅                 | TBD                    |
|                   |                                                                                                                                                                    |                      |                      |                   |                        |
| prerequisites     | basic python                                                                                                                                                       | ✅                    | ✅                    | ✅                 | ❌                      |
|                   | basic C                                                                                                                                                            | ❌                    | ❌                    | ❌                 | ✅                      |
|                   | a tiny bit of introductory calculus (what a derivative is conceptually)                                                                                            | ✅                    | ✅                    | ✅                 | ✅                      |
|                   | a tiny bit of introductory linear algebra (matrices and matrix multiplication)                                                                                     | ❌                    | ✅                    | ✅                 | ✅                      |
|                   | familiarity with [numpy](https://numpy.org)                                                                                                                        | ❌                    | ✅                    | ❌                 | ❌                      |
|                   | linux (or use [colab](https://colab.research.google.com), [lambda](https://lambdalabs.com) or similar)                                                             | ❌                    | ❌                    | ✅                 | ❌                      |
|                   | Nvidia GPU (or use [colab](https://colab.research.google.com), [lambda](https://lambdalabs.com) or similar)                                                        | ❌                    | ❌                    | ✅                 | ✅                      |
|                   |                                                                                                                                                                    |                      |                      |                   |                        |
| What you'll learn | basic math of autograd systems                                                                                                                                     | ✅                    | ✅                    | ✅                 | ✅                      |
|                   | basic math of [GPT-2](https://en.wikipedia.org/wiki/GPT-2#:~:text=Generative%20Pre%2Dtrained%20Transformer%202,of%208%20million%20web%20pages.) style transformers | ✅                    | ✅                    | ✅                 | ✅                      |
|                   | basics of efficient parallel programming linear algebra on GPUs                                                                                                    | ❌                    | ❌                    | ✅                 | ✅                      |
|                   | OpenAI's [Triton](https://triton-lang.org/main/index.html)                                                                                                         | ❌                    | ❌                    | ✅                 | ❌                      |
|                   | Nvidia's [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)     | ❌                    | ❌                    | ❌                 | ✅                      |

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
- [ ] make model output logits & celoos do softmax within so that inference can do softmax
- [ ] build a way to test against pytorch (or maybe against micrograd or autograd, that sounds easier)
- [ ] write jupyter notebook guide
- [ ] make video

## autograd (Triton)
the purpose of this lesson is for people already confident in the math behind autograd engines and GPTs to learn exactly what's happening at the level of the GPU hardware. you can think of autograd as a replication of pytorch/tensorflow/jax/mlx/tinygrad/etc except that instead of being meant to actually be used, which would require it be flexible and robust, it is meant for educational purposes. As such, we'll only be implementing the operations that are absolutely necessary in order to create a GPT and not worrying much about unexpected edge cases, just like how we did in micrograd and minigrad

### why triton?
You might be asking: why are we using Triton instead of CUDA? I'm open to the idea of learning and then doing a lesson on CUDA (and MPS for that matter) in the future, but for now here are the pros and cons that it came down to:

|      | triton                                                                                            | cuda                                                             |
| ---- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| pros | - written in Python (quicker to learn)<br>- works on more than just Nvidia GPUs<br>- open-sourced | - broadly used<br>- linux or windows                             |
| cons | - less commonly used<br>- requires linux                                                          | - written in C<br>- only works on Nvidia GPUs<br>- closed-source |

Personally I'm on a Mac so i plan on doing all my work on a cloud provider like [lambdalabs](https://lambdalabs.com) anyways so the windows availability didn't matter much to me. That and I highly value the pythonic syntax and potential future widespread compatibility. 

##### TODO:
- [x] begin learning Triton
	- [ ] ~~realize i'm in way over my head~~ *PSYC DAT SHIT WAS EASY*
		- [ ] ~~abandon project lmao~~
- [ ] engine.py
	- [ ] TritonTensor class
		- [x] __init__
		- [x] __add__
			- [x] figure out broadcasting
			- [x] fwd kernel
			- [x] bwd kernel
		- [ ] the rest of the ops
	- [x] testing
- [x] benchmarking

## autograd (CUDA)
I'll consider building this if 1) the other three perform well 2) people are interested and 3) i become a masochist

## autograd (MPS)
I'll consider building this if 1) the other three perform well 2) people are interested and 3) i become a masochist
