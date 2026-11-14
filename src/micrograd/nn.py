import random
from micrograd.values import Value

class Module:

    def zero_grad(self):
        pass

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self):
        pass

    def __call__(self):
        pass

    def parameters(self):
        pass

    def __repr__(self):
        pass

class Layer(Module):
    def __init__(self):
            pass
    
    def __call__(self):
        pass
    
    def parameters(self):
        pass
    
    def __repr__(self):
        pass

class MLP(Module):
    def __init__(self):
            pass
    
    def __call__(self):
        pass
    
    def parameters(self):
        pass
    
    def __repr__(self):
        pass