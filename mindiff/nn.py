import numpy as np
from . import ts


class Linear:
    def __init__(self, in_features, out_features):
        self.W = ts.Tensor(np.random.normal(size=(in_features, out_features)) * 10)
        self.b = ts.Tensor(np.zeros((out_features,)))

    def __call__(self, X):
        return (X @ self.W) + self.b


class ReLU:
    def __init__(self, in_place=False):
        self.in_place = in_place

    def __call__(self, tensor):
        out = ts.Tensor(np.maximum(tensor.data, 0), (tensor,), "ReLU")

        def backward():
            only_positive = tensor.data.copy()
            only_positive[only_positive <= 0] = 0.0
            only_positive[only_positive > 0] = 1.0
            tensor.grad += only_positive * out.grad

        out._backward = backward
        return out
