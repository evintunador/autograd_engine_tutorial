from engine import Tensor

class Module: # just to make our syntax the same as pytorch's
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []