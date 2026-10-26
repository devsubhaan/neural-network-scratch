import numpy as np
from abc import abstractmethod
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.nn_scratch.initializers import he_init

class Layer:

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


