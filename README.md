# autograd_engine_tutorial
the plan for this repo is to create tutorials on autograd engines for three levels of difficulty:

## micrograd 
the purpose of this folder is for absolute beginners to learn exactly what's happening in GPTs all the way from the barebones autograd engine and up to the GPT operations itself. the only prerequisites will be basic calculus (specifically chain rule) and raw regular python (the only packages we're using are `random` and `math`). the basic building block of micrograd is the `Value` object, each of which is just a single floating point number for the data and another single floating point number to keep track of the data's gradient. the first half or so of this lesson is roughly equivalent to [karpathy's `micrograd`](https://youtu.be/VMj-3S1tku0?si=FM0qtfV-cvXr2kDJ) while the second half is an extension to implement a full GPT

prerequisites:
- a tiny bit of calculus
- the python packages random and math

time commitment: *medium*

what you'll learn:
- the math behind autograd systems and GPT-2 style transformers

## minigrad (intermediate)
the purpose of this folder is for people already confident with linear algebra and calculus to learn exactly what's happening in GPTs all the way from the barebones autograd engine and up to the GPT operations itself. to make this happen we'll be doing everything with numpy arrays, meaning that the basic unit of our engine will be tensors as opposed to individual values. honestly in that sense it's a bit redundant, but it makes the code much prettier than micrograd so it's nice to look through

prerequisites
- calculus
- linear algebra
- the python package numpy

time commitment: *small*

what you'll learn:
- the math behind autograd systems and GPT-2 style transformers

## autograd (beginners)
the purpose of this folder is for people already confident in the math behind GPTs to learn exactly what's happening at the level of the GPU hardware. you can think of autograd as a replication of pytorch/tensorflow/jax/mlx/tinygrad/etc except that instead of being meant to actually be used, which would require it be flexible and robust, it is meant for educational purposes. As such, we'll only be implementing the operations that are absolutely necessary in order to create a GPT and not worrying much about unexpected edge cases, just like how we did in micrograd and minigrad

prerequisites:
- calculus
- linear algebra

time commitment: *large*

what you'll learn:
- the math behind autograd systems and GPT-2 style transformers
- the way this math is implemented efficiently on GPUs
- how to implement your own custom GPU kernels in Triton