import math

class Tensor:
    def __init__(self, data, _children = (), _op = ''):
        self.data = data
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
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

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

    def __pow__(self, other):
        other = self._other_check(other)
        out = Tensor(self.data ** other.data, (self, other), '**')

        def backward():
            self.grad = (other.data * self.data ** (other.data - 1)) * out.grad
            other.grad = out.data * math.log(self.data) * out.grad

        out._backward = backward
        return out
    
    def __rpow__(self, other):
        other = self._other_check(other)
        out = Tensor(other.data ** self.data, (self, other), '**')

        def backward():
            self.grad = out.data * math.log(other.data) * out.grad
            other.grad = (self.data * other.data ** (self.data - 1)) * out.grad

        out._backward = backward

        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** - 1)

    def __neg__(self):
        out = Tensor(-self.data, (self,), '-')
        
        def backward():
            self.grad = - out.grad

        out._backward = backward
        return out

    def __sub__(self, other):
        return self + (- other)

    def _other_check(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other
    