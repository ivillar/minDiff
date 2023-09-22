class SGD:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = list(parameters)
        self.lr = lr

    def zero(self):
        for p in self.parameters:
            p.tensor.grad.data *= 0.0

    def step(self, zero=True):
        for p in self.parameters:
            p.tensor.data -= p.tensor.grad * self.lr

            if zero:
                p.tensor.grad *= 0.0
