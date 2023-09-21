import numpy as np
from . import ts
import math


class Module:
    def __init__(self):
        self._parameters = {}

    def register_parameter(self, name, param):
        self._parameters[name] = param

    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for child in self.children():
            for param in child.parameters():
                yield param

    def children(self):
        for name, attr in vars(self).items():
            if isinstance(attr, Module):
                yield attr

    def forward(self):
        pass

    def __call__(self, X):
        return self.forward(X)


class Parameter:
    def __init__(self, tensor):
        self.tensor = tensor


class Linear(Module):
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
        super().__init__()
        weight_tensor = ts.Tensor(
            np.random.normal(size=(in_features, out_features))
            * math.sqrt(2.0 / in_features)
        )
        bias_tensor = ts.Tensor(np.zeros((out_features,)))
        self.weight = Parameter(weight_tensor)
        self.bias = Parameter(bias_tensor)
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, X):
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
        return (X @ self.weight.tensor) + self.bias.tensor


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
        pass

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


class Tanh:
    def __init__(self):
        pass

    def __call__(self, tensor):
        out = ts.Tensor(np.tanh(tensor.data), (tensor,), "Tanh")

        def backward():
            tensor.grad += (1 - np.tanh(tensor.data) ** 2) * out.grad

        out._backward = backward
        return out


class Softmax:
    def __init__(self, in_place=False):
        pass

    def _softmax(self, array, axis=None):
        """
        Performs numerically stable softmax operation on the input array.

        Parameters
        ----------
        array : np.ndarray
            The input 2D numpy array.

        Returns
        -------
        np.ndarray
            The output array after applying softmax.
        """
        shiftx = array - np.max(array, axis=axis, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=axis, keepdims=True)

    def __call__(self, tensor):
        out = ts.Tensor(self._softmax(tensor.data, axis=-1), (tensor,), "sftmx")

        def backward():
            s = self._softmax(tensor.data, axis=-1)  # (m, ..., n)
            n = s.shape[-1]
            s = np.expand_dims(s, axis=-1)  # (m, n, 1)
            diag = s * np.eye(n)  # (m, n, n)
            s_swapped = np.swapaxes(s, -1, -2)  # (m, 1, n)
            sst = s @ s_swapped  # (m, n, n)
            dsdx = diag - sst
            outgrad_expanded = np.expand_dims(out.grad, axis=-1)  # (m, n, 1)
            result = dsdx @ outgrad_expanded
            squeezed_result = np.squeeze(result, axis=-1)
            tensor.grad += squeezed_result

        out._backward = backward

        return out


class CrossEntropyLoss:
    def __init__(self, in_place=False):
        pass

    def __call__(self, output_logits, targets):
        m = output_logits.data.shape[0]
        logyhat = np.log(output_logits.data)  # (m, k)
        # (m, 1, k) @ (m, k, 1) = m(m, 1, 1)
        L = -np.expand_dims(targets.data, axis=-2) @ np.expand_dims(logyhat, axis=-1)
        J = np.mean(L)
        out = ts.Tensor(J, (output_logits,), "CEL")

        def backward():
            epsilon = 1e-10
            output_logits.grad += (
                -(targets.data / (output_logits.data + epsilon)) * out.grad * (1 / m)
            )

        out._backward = backward

        return out
