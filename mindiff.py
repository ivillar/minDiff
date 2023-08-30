class Tensor:
    def __init__(self, data, children = (), _op = ''):
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"Tensor(data={self.data})"
    
    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')

        def backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = backward
        return out
    
    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        
        out._backward = backward
        return out

    