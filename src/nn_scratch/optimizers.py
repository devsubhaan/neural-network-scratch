import numpy as np
from abc import ABC, abstractmethod

"""
Self note: Include ADAM and compare to SGD later on
"""

class Optimizer(ABC):
    @abstractmethod
    def step(self, layers: list) -> None:
        """Update every parameter in place using its stored gradient."""
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, layers: list) -> None:
        for layer in layers:
            for parameter, gradient in layer.params():
                parameter -= self.learning_rate * gradient

