import math
import numpy as np


class Tensor:
    """
    A class used to represent a Tensor for automatic differentiation.

    This class provides the basic operations needed to build a computation
    graph for automatic differentiation, including addition, subtraction,
    multiplication, division, and exponentiation. It also provides methods for
    reshaping, tiling, expanding dimensions, and squeezing dimensions of the tensor.

    Attributes
    ----------
    data : numpy.ndarray
        The data contained in the tensor.
    _prev : set
        The set of tensors that this tensor depends on.
    _op : str
        The operation that produced this tensor.
    grad : numpy.ndarray
        The gradient of this tensor.
    _backward : function
        The function to compute the gradient of this tensor's children.
    label : str
        The label of this tensor.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = (
            data.astype("float64")
            if isinstance(data, np.ndarray)
            else np.array(data, dtype="float64")
        )
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None
        self.label = None

    def backward(self):
        """
        Performs the backward pass of automatic differentiation.

        This function computes the gradient of the tensor with respect to each
        of its dependencies, using the chain rule of calculus. It first
        performs a topological sort of the computation graph to determine the
        order in which to compute the gradients, then it iterates over the
        sorted list and invokes the `_backward` method of each tensor.

        This function should be called after the forward pass and loss computation.

        Note: This function modifies the `grad` attribute of the tensor and its
        dependencies in-place.

        Raises:
        Exception: If called before the forward pass and loss computation.
        """
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

        self.grad = np.array(1)

        for tensor in topo_result:
            tensor._backward()

    def __repr__(self):
        """
        Returns a string representation of the Tensor object.

        Returns:
        str: A string representation of the Tensor object.
        """
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        """
        Adds this tensor to another tensor.

        Parameters:
        other (Tensor or numeric type): The tensor to add to this tensor. If
        `other` is not a Tensor, it is converted to one.

        Returns:
        Tensor: A new tensor that is the result of adding this tensor to
        `other`.
        """
        other = self._other_check(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def backward():
            selfgrad = 1.0 * out.grad
            othergrad = 1.0 * out.grad

            self.grad += self._unbroadcast_gradient(self.data, selfgrad)
            other.grad += self._unbroadcast_gradient(other.data, othergrad)

        out._backward = backward
        return out

    def __mul__(self, other):
        """
        Performs element-wise multiplication between this tensor and another
        tensor.

        Parameters:
        other (Tensor or numeric type): The tensor to element-wise multiply to
        this tensor.

        Returns: A new tensor that is the result of element-wise multiplying
        this tensor and `other`.
        """
        other = self._other_check(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def backward():
            selfgrad = other.data * out.grad
            othergrad = self.data * out.grad

            self.grad += self._unbroadcast_gradient(self.data, selfgrad)
            other.grad += self._unbroadcast_gradient(other.data, othergrad)

        out._backward = backward
        return out

    def __rmul__(self, other):
        """
        Performs element-wise multiplication between this tensor and another
        tensor, with the operation commuted.

        Parameters:
        other (Tensor or numeric type): The tensor to element-wise multiply to
        this tensor.

        Returns:
        A new tensor that is the result of element-wise multiplying this tensor
        and `other`.
        """
        return self * other

    def __matmul__(self, other):
        """
        Performs matrix multiplication between this tensor and another
        tensor.

        Parameters:
        other (Tensor or numeric type): The tensor to matrix multiply to
        this tensor.

        Returns: A new tensor that is the result of matrix multiplying
        this tensor and `other`.
        """
        other._other_check(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        """
        Performs element-wise exponentiation of this tensor to the power of
        `other`.

        Parameters:
        other (Tensor or numeric type): The exponent tensor.

        Returns:
        A new tensor that is the result of raising this tensor to the power of
        `other`.
        """
        other = self._other_check(other)
        out = Tensor(self.data**other.data, (self, other), "**")

        def backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad
            other.grad += out.data * math.log(self.data) * out.grad

        out._backward = backward
        return out

    def __rpow__(self, other):
        """
        Performs element-wise exponentiation of `other` to the power of this tensor.

        Parameters:
        other (Tensor or numeric type): The base tensor.

        Returns:
        A new tensor that is the result of raising `other` to the power of this tensor.
        """
        other = self._other_check(other)
        out = Tensor(other.data**self.data, (self, other), "**")

        def backward():
            self.grad += out.data * math.log(other.data) * out.grad
            other.grad += (self.data * other.data ** (self.data - 1)) * out.grad

        out._backward = backward

        return out

    def __truediv__(self, other):
        """
        Performs element-wise division of this tensor by `other`.

        Parameters:
        other (Tensor or numeric type): The denominator tensor.

        Returns:
        A new tensor that is the result of dividing this tensor by `other`.
        """
        return self * (other**-1)

    def __rtruediv__(self, other):
        """
        Performs element-wise division of `other` by this tensor.

        Parameters:
        other (Tensor or numeric type): The numerator tensor.

        Returns:
        A new tensor that is the result of dividing `other` by this tensor.
        """
        return other * (self**-1)

    def __neg__(self):
        """
        Returns a new tensor that is the negation of this tensor.

        Returns:
        A new tensor that is the negation of this tensor.
        """
        out = Tensor(-self.data, (self,), "-")

        def backward():
            self.grad += -out.grad

        out._backward = backward
        return out

    def expand_dims(self, axis=None):
        """
        Expands the shape of this tensor by adding a new axis at the specified
        position.

        Parameters:
        axis (int, optional): The position in the expanded axes where the new
        axis is placed.

        Returns:
        A new tensor with an additional dimensions at specified axes.
        """
        out = Tensor(np.expand_dims(self.data, axis=axis), (self,), "exdm")

        def backward():
            self.grad += np.squeeze(out.grad, axis=axis)

        out._backward = backward
        return out

    def squeeze(self, axis=None):
        """
        Removes single-dimensional entries from the shape of this tensor.

        Parameters:
        axis (int, optional): Selects a subset of single-dimensional entries in
        the shape to squeeze.

        Returns:
        A new tensor with the same data but with the dimensions of size one
        removed at specified axes.
        """
        out = Tensor(np.squeeze(self.data, axis=axis), (self,), "sqz")

        def backward():
            self.grad += np.expand_dims(out.grad, axis=axis)

        out._backward = backward
        return out

    def reshape(self, newshape):
        """
        Gives a new shape to this tensor without changing its data.

        Parameters:
        newshape (int or tuple of ints): The new shape should be compatible
        with the original shape.

        Returns:
        A new tensor with the new shape.
        """
        oldshape = self.data.shape
        out = Tensor(np.reshape(self.data, newshape=newshape), (self,), "rshp")

        def backward():
            self.grad += np.reshape(out.grad, oldshape)

        out._backward = backward
        return out

    def tile(self, reps):
        """
        Constructs a new tensor by repeating this tensor the number of times given by reps.

        Parameters:
        reps (int or tuple of ints): The number of repetitions of this tensor along each axis.

        Returns:
        A new tensor with the repeated data.
        """
        out = Tensor(np.tile(self.data, reps), (self,), "tl")

        def backward():
            self.grad += self._backward_tile(out.grad, self.data.shape, reps)

        out._backward = backward
        return out

    def __sub__(self, other):
        """
        Performs element-wise subtraction of `other` from this tensor.

        Parameters:
        other (Tensor): The tensor to subtract from this tensor.

        Returns:
        A new tensor that is the result of subtracting `other` from this tensor.
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Performs element-wise subtraction of this tensor from `other`.

        Parameters:
        other (Tensor or numeric type): The tensor from which this tensor is subtracted.

        Returns:
        A new tensor that is the result of subtracting this tensor from `other`.
        """
        return (-self) + other

    def _other_check(self, other):
        """
        Checks to see whether `other` is an instance of Tensor. If it isn't it
        converts it to one.

        Parameters:
        other (Tensor or numeric type): The object to check and possibly convert.

        Returns:
        A Tensor instance.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other

    def sum(self, axis=None):
        """
        Sum of tensor elements over a given axis.

        Parameters:
        axis (None or int or tuple of ints, optional): Axis or axes along which a sum is performed.

        Returns:
        A new tensor with the sum of the elements over the specified axis.
        """
        out = Tensor(np.sum(self.data, axis=axis), (self,), "sum")

        def backward():
            # distribute global gradient to m
            expanded_out_grad = (
                np.expand_dims(out.grad, axis=axis)
                if isinstance(out.grad, np.ndarray)
                else np.array([out.grad], dtype="float64")
            )
            self.grad += expanded_out_grad * np.ones(self.data.shape)

        out._backward = backward
        return out

    def _unbroadcast_gradient(self, child, global_gradient):
        """
        This function unbroadcasts the global gradient so that it can then be passed
        on to the child. This is necessary when the child's shape doesn't match the
        global gradient's shape due to broadcasting during the forward pass.

        Parameters:
        child (Tensor): The tensor to which the gradient will be passed.
        global_gradient (Tensor): The gradient to be unbroadcasted.

        Returns:
        The unbroadcasted gradient.
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

    def _backward_tile(self, dY, input_shape, reps):
        """
        This function computes the gradient of the tile operation during the backward pass.

        Parameters:
        dY (Tensor): The gradient of the output of the tile operation.
        input_shape (tuple): The shape of the input to the tile operation.
        reps (int or tuple): The number of repetitions of each element in the input.

        Returns:
        The gradient of the input of the tile operation.
        """
        if isinstance(reps, int):
            reps = (reps,)
        dim_diff = max(0, len(reps) - len(input_shape))
        original_array = dY[
            tuple([0 for _ in range(dim_diff)] + [slice(dim) for dim in input_shape])
        ]
        ones = np.ones(((np.prod(reps),) + input_shape))
        dX = original_array * ones
        dX = np.sum(dX, axis=0)
        return dX
