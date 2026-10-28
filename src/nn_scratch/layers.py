import numpy as np
import sys
from abc import ABC, abstractmethod
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.nn_scratch.initializers import he_init

class Layer(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, grad_out):
        pass

    def params(self):
        return []


class Dense(Layer):
    
    def __init__(self, n_in, n_out, rng, init = he_init):
        self.W = init(n_in, n_out, rng)
        self.b = np.zeros([1, n_out])
        self.dW = np.zeros[(n_in, n_out)]
        self.db = np.zeros[(1, n_out)]
        self._x = None

    def forward(self, x):
        if x.ndim == 2 and x.shape[1] == self.W.shape[0]:
            self._x = x
            return x @ self.W + self.b
        
        raise TypeError("Invalid x and/or W size")

    def backward(self, grad_out):
        self.dW = self._x.T @ grad_out
        self.db = grad_out.sum(axis=0, keepdims=0)
        return grad_out @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]