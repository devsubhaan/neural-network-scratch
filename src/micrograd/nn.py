import random
from micrograd.values import Value

class Module:
    """
    Base class that provides zero_grad and parameters
    """

    def zero_grad(self):
        """
        Resets all gradients to 0.0
        Called before every backwards pass
        """
        pass

    def parameters(self):
        """
        Return a list of all parameters in the module
        """
        return []

class Neuron(Module):
    """
    Represents a single perceptron
    Holds a list of weights and a singular bias
    """
    def __init__(self):
        """
        Initialize weights randomly and the bias
        """
        pass

    def __call__(self):
        """
        Calculates the sum of all 'wx' and adds the bias
        Gives wx + b
        """
        pass

    def parameters(self):
        """
        Returns a list of all weights and the bias
        """
        pass

    def __repr__(self):
        """
        Returns string representation of the neuron
        Shows ReLU or linear along with the number of inputs
        """
        pass

class Layer(Module):
    """
    Represents a dense, fully connected neuron layer
    passes inputs in parallel to different neurons
    """
    def __init__(self):
        """
        Initialises a collection of parallel neurons
        """
        pass
    
    def __call__(self, x):
        """
        Evaluate the input vector across all neurons in parallel.
        """
        pass
    
    def parameters(self):
        """
        Returns a flattened list of all parameters from all neurons.
        """
        pass
    
    def __repr__(self):
        """
        Return a string listing the representations of every neuron in the layer.
        """
        pass

class MLP(Module):
    """
    Represents a Multi-Layer Perceptron
    Chains multiple dense layers together to process inputs.
    """
    def __init__(self, nin, nouts):
        """
        Build the network layers using the input and output sizes.
        Keeps ReLU for hidden layers but off for the final output.
        """
        pass

    def __call__(self, x):
        """
        Pass the input through all network layers.
        Updates the data and returns the final layer's output.
        """
        pass

    def parameters(self):
        """
        Returns a flattened list of all parameters across all layers.
        """
        pass

    def __repr__(self):
        """
        Returns a string of the structural representations of all layers.
        """
        pass
