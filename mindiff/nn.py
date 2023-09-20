import numpy as np
from . import ts


class Linear:
    """
    Implements a linear layer in a neural network.

    Attributes
    ----------
    W : ts.Tensor
        The weight matrix of the linear layer.
    b : ts.Tensor
        The bias vector of the linear layer.

    Methods
    -------
    __call__(X)
        Performs the forward pass of the linear layer.
    """

    def __init__(self, in_features, out_features):
        self.W = ts.Tensor(np.random.normal(size=(in_features, out_features)) * 10)
        self.b = ts.Tensor(np.zeros((out_features,)))

    def __call__(self, X):
        """
        Performs the forward pass of the linear layer.

        Parameters
        ----------
        X : ts.Tensor
            The input tensor to the layer.

        Returns
        -------
        ts.Tensor
            The output tensor from the layer.
        """
        return (X @ self.W) + self.b


class ReLU:
    """
    Implements a ReLU (Rectified Linear Unit) activation function in a neural network.

    Attributes
    ----------
    in_place : bool
        Whether to apply the ReLU operation in-place or not.

    Methods
    -------
    __call__(tensor)
        Performs the forward pass of the ReLU activation function.
    """

    def __init__(self, in_place=False):
        self.in_place = in_place

    def __call__(self, tensor):
        """
        Performs the forward pass of the ReLU activation function.

        Parameters
        ----------
        tensor : ts.Tensor
            The input tensor to the activation function.

        Returns
        -------
        ts.Tensor
            The output tensor from the activation function.
        """
        out = ts.Tensor(np.maximum(tensor.data, 0), (tensor,), "ReLU")

        def backward():
            only_positive = tensor.data.copy()
            only_positive[only_positive <= 0] = 0.0
            only_positive[only_positive > 0] = 1.0
            tensor.grad += only_positive * out.grad

        out._backward = backward
        return out
