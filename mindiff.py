import math
import numpy as np

class Tensor:
    def __init__(self, data, _children = (), _op = ''):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda : None
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
        out = Tensor(self.data + other.data, (self, other), '+')

        def backward():
            self_unbroadcast_outgrad = self.unbroadcast_gradient(self.data, out.grad)
            other_unbroadcast_outgrad = self.unbroadcast_gradient(other.data, out.grad)
            self.grad += 1.0 * self_unbroadcast_outgrad
            other.grad += 1.0 * other_unbroadcast_outgrad

        out._backward = backward
        return out
    
    def __mul__(self, other):
        other = self._other_check(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        other._other_check(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        other = self._other_check(other)
        out = Tensor(self.data ** other.data, (self, other), '**')

        def backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad
            other.grad += out.data * math.log(self.data) * out.grad

        out._backward = backward
        return out
    
    def __rpow__(self, other):
        other = self._other_check(other)
        out = Tensor(other.data ** self.data, (self, other), '**')

        def backward():
            self.grad += out.data * math.log(other.data) * out.grad
            other.grad += (self.data * other.data ** (self.data - 1)) * out.grad

        out._backward = backward

        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** - 1)

    def __neg__(self):
        out = Tensor(-self.data, (self,), '-')
        
        def backward():
            self.grad += - out.grad

        out._backward = backward
        return out

    def __sub__(self, other):
        return self + (- other)

    def _other_check(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other
    
    def sum(self, axis=None):
        """
        TODO: Add backward support for different dimensions of summation.
        Backward is only guaranteed to work for summing last axis of 2D array.
        """
        out = Tensor(np.sum(self.data, axis=axis), (self,), 'sum')

        def backward():
            m = self.data.shape[0]
            # distribute global gradient to m
            self.grad += np.tile(np.expand_dims(out.grad, axis = 0), (m,1))

        out._backward = backward
        return out


    def unbroadcast_gradient(self, child, global_gradient):
        correct_global_gradient = global_gradient

        if child.shape != global_gradient.shape:
            dimensions_diff = np.abs(global_gradient.ndim - child.ndim)
            if dimensions_diff != 0:
                summation_dims = tuple(range(dimensions_diff))
                correct_global_gradient = np.sum(global_gradient, axis=summation_dims)
    
                originally_ones = tuple([axis  for axis, size in enumerate(child.shape) if size == 1])
                if len(originally_ones) != 0:
                    correct_global_gradient = np.sum(correct_global_gradient, axis=axis, keepdims=True)
    
        return correct_global_gradient
    