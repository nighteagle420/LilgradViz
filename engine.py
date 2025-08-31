import numpy as np

# def list(data):
#     if isinstance(data,list):
#         return data
#     else:
#         return data.tolist()
def array(data):
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,TensorVal):
        return data.toarray()
    else:
        return np.array(data)
def Tval(data):
    if isinstance(data,TensorVal):
        return data
    else:
        return TensorVal(data)

class TensorVal:
    def __init__(self, data, _children=(), _op='',label=""):
        self.data = array(data)
        self.grad = np.zeros_like(self.data,dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self.shape = self.data.shape
        self._op = _op
        self.label = label
        
    def __repr__(self) -> str:
        return f"""TensorVal[{self.label}]({self.data})"""
    
    def zero_grad(self):
        visited = set()
        def _zero(v):
            if v not in visited:
                v.grad = np.zeros_like(v.data, dtype=float)
                visited.add(v)
                for child in v._prev:
                    _zero(child)
        _zero(self)

    # ---- helper for broadcasting-safe reduction ----
    def _reduce_grad(self, grad, shape):
        """Reduce gradient to match original shape (undo broadcasting)."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # ---- arithmetic ----
    def __add__(self, other):
        other = other if isinstance(other, TensorVal) else TensorVal(other)
        out = TensorVal(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += self._reduce_grad(out.grad, self.data.shape)
            other.grad += self._reduce_grad(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, TensorVal) else TensorVal(other)
        out = TensorVal(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += self._reduce_grad(out.grad, self.data.shape)
            other.grad -= self._reduce_grad(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        out = TensorVal(-self.data, (self,), 'neg')

        def _backward():
            self.grad -= out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, TensorVal) else TensorVal(other)
        out = TensorVal(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += self._reduce_grad(other.data * out.grad, self.data.shape)
            other.grad += self._reduce_grad(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, TensorVal) else TensorVal(other)
        out = TensorVal(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += self._reduce_grad((1 / other.data) * out.grad, self.data.shape)
            other.grad -= self._reduce_grad((self.data / (other.data**2)) * out.grad, other.data.shape)
        out._backward = _backward
        return out

    # ---- reductions ----
    def sum(self, axis=None, keepdims=False):
        out = TensorVal(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    # ---- matrix multiply ----
    def __matmul__(self, other):
        other = other if isinstance(other, TensorVal) else TensorVal(other)
        out = TensorVal(self.data @ other.data, (self, other), 'matmul')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    
    # def transpose(self, axes=None):
    #     out = TensorVal(self.data.transpose(axes), (self,), 'transpose')

    #     def _backward():
    #         self.grad += out.grad.transpose(axes if axes is not None else None)
    #     out._backward = _backward
    #     return out

    # # shorthand like numpy: x.T
    # @property
    # def T(self):
    #     return self.transpose()

    # ---- nonlinearities ----
    def relu(self):
        out = TensorVal(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = TensorVal(out_data, (self,), 'exp')

        def _backward():
            self.grad += out_data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = TensorVal(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
      out = TensorVal(np.tanh(self.data), (self,), 'tanh')

      def _backward():
        self.grad += (1 - np.tanh(self.data)**2) * out.grad
      out._backward = _backward
      return out

    # ---- autograd engine ----
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)  # seed gradient
        for v in reversed(topo):
            v._backward()