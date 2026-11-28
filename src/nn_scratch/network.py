import numpy as np
from src.nn_scratch.activations import softmax

class Network:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list:
        out = []
        for layer in self.layers:
            for parameter, gradient in layer.params():
                out.append((parameter, gradient))
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=1)
