import random
from values import Value

class Module:
    """
    Base class that provides zero_grad and parameters
    """

    def zero_grad(self):
        """
        Resets all gradients to 0.0
        Called before every backwards pass
        """
        for p in self.parameters:
            p.grad = 0

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
    def __init__(self, nin, nonlin=True):
        """
        Initialize weights randomly and the bias
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin


    def __call__(self, x):
        """
        Calculates the sum of all 'wx' and adds the bias
        Gives wx + b
        """
        ret = sum(wx*ix for wx, ix in zip(self.w, x)) + self.b
        return ret.relu() if self.nonlin else ret

    def parameters(self):
        """
        Returns a list of all weights and the bias 
        """
        return self.w + [self.b]

    def __repr__(self):
        """
        Returns string representation of the neuron
        Shows ReLU or linear along with the number of inputs
        """
        return f"{'Relu' if self.nonlin else 'Linear'} -> Neuron({len(self.w)}"

class Layer(Module):
    """
    Represents a dense, fully connected neuron layer
    passes inputs in parallel to different neurons
    """
    def __init__(self, nin, nout, **kwargs):
        """
        Initialises a collection of parallel neurons
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        """
        Evaluate the input vector across all neurons in parallel.
        """
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        """
        Returns a flattened list of all parameters from all neurons.
        """
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        """
        Return a string listing the representations of every neuron in the layer.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

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
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Pass the input through all network layers.
        Updates the data and returns the final layer's output.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Returns a flattened list of all parameters across all layers.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """
        Returns a string of the structural representations of all layers.
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
