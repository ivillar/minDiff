import math
import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None
        self.label = None

    def backward(self):
        visited = set()
        topo_result = []

        def toposort(tensor):
            visited.add(tensor)
            for child in tensor._prev:
                if child not in visited:
                    toposort(child)
            topo_result.append(tensor)

        toposort(self)
        topo_result.reverse()

        self.grad = 1

        for tensor in topo_result:
            tensor._backward()

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = self._other_check(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def backward():
            self_unbroadcast_outgrad = self._unbroadcast_gradient(self.data, out.grad)
            other_unbroadcast_outgrad = self._unbroadcast_gradient(other.data, out.grad)
            self.grad += 1.0 * self_unbroadcast_outgrad
            other.grad += 1.0 * other_unbroadcast_outgrad

        out._backward = backward
        return out

    def __mul__(self, other):
        """
        Multiplies the two tensors together.
        """
        other = self._other_check(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def backward():
            self_unbroadcast_outgrad = self._unbroadcast_gradient(self.data, out.grad)
            other_unbroadcast_outgrad = self._unbroadcast_gradient(other.data, out.grad)
            self.grad += other.data * self_unbroadcast_outgrad
            other.grad += self.data * other_unbroadcast_outgrad

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other._other_check(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        other = self._other_check(other)
        out = Tensor(self.data**other.data, (self, other), "**")

        def backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad
            other.grad += out.data * math.log(self.data) * out.grad

        out._backward = backward
        return out

    def __rpow__(self, other):
        other = self._other_check(other)
        out = Tensor(other.data**self.data, (self, other), "**")

        def backward():
            self.grad += out.data * math.log(other.data) * out.grad
            other.grad += (self.data * other.data ** (self.data - 1)) * out.grad

        out._backward = backward

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __neg__(self):
        out = Tensor(-self.data, (self,), "-")

        def backward():
            self.grad += -out.grad

        out._backward = backward
        return out

    def __sub__(self, other):
        """
        TODO: add __rsub__ method
        """
        return self + (-other)

    def _other_check(self, other):
        """
        Checks to see whether other is an instance of Tensor. If it isn't it
        converts it to one.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other

    def sum(self, axis=None):
        """
        TODO: Add backward support for different dimensions of summation.
        Backward is only guaranteed to work for summing last axis of 2D array.
        """
        out = Tensor(np.sum(self.data, axis=axis), (self,), "sum")

        def backward():
            m = self.data.shape[0]
            # distribute global gradient to m
            self.grad += np.tile(np.expand_dims(out.grad, axis=0), (m, 1))

        out._backward = backward
        return out

    def _unbroadcast_gradient(self, child, global_gradient):
        """
        This function unbroadcasts the global gradient so that it can then be passed
        on to the child.
        """
        correct_global_gradient = global_gradient

        # start unbroadcast if child and global_gradient shapes mismatch
        if child.shape != global_gradient.shape:
            # dimensions preppended to child to broadcast
            n_extra_dims = global_gradient.ndim - child.ndim
            extra_dims = tuple(range(n_extra_dims)) if n_extra_dims > 0 else ()

            # dimensions in global_gradient where child originally had ones
            one_dims = tuple(
                [
                    axis + n_extra_dims
                    for axis, size in enumerate(child.shape)
                    if size == 1
                ]
            )
            # dimensions where we will sum over in the global gradient
            summation_dims = extra_dims + one_dims
            correct_global_gradient = global_gradient.sum(
                axis=summation_dims, keepdims=True
            )

            # squeezing over the extra dimensions
            if n_extra_dims > 0:
                correct_global_gradient = np.squeeze(
                    correct_global_gradient, axis=extra_dims
                )

        return correct_global_gradient
