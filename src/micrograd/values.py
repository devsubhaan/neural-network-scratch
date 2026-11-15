


class Value:
    """
    Stores a single value along with its gradient
    """

    def __init__(self, data, _children=()):
        """
        Initialises a new Value node
        Tracks the data, gradient, children nodes 
        """
        self.data = data
        self.grad = 0

        self._children= _children
        self._previous = set(_children)
        self.backward = lambda: None

    def __add__(self, other):
        """
        Returns self added to other (self + other)

        Makes sure that 'other' is a Value object
        Simply adds the output to the self and other gradients
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            pass

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Returns self multiplied by other (self * other)

        Makes sure that 'other' is a Value object
        Uses chain rule for both self and other gradients
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            pass
                
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        """
        Returns self raised to the other (self ** other)

        Checks if the power is an integer or float
        Applies power rule for the backwards pass
        """
        assert isinstance(other, (int, float)), "Only floats and integers allowed!"
        out = Value(self.data ** other.data, (self, other), f"**{other}")

        def _backward():
            pass
                
        out._backward = _backward
        
        return out

    def relu(self):
        """
        Values less than zero become zero
        Values bigger than zero stay
        """
        out = 0 if self.data < 0 else self.data

        def _backward():
            pass
                
        out._backward = _backward
        
        return out

    def backward(self):
        """
        Calculates the gradients via backpropagation by first sorting the children in reverse order
        Then applys chain rule to get their gradients
        Uses a topological sort
        """
        pass

    def __neg__(self):
        """
        Negates a value turning it negative
        """
        return self.data * -1

    def __radd__(self, other):
        """
        Fallback for when self is not a value object and cant add other (b + a is done)
        """
        return self + other

    def __sub__(self, other):
        """
        Subtracts self and other (a - b)
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Fallback for when a is not a value object and cant subtract b
        """
        return other + (-self)
    
    def __rmul__(self, other):
        """
        Fallback for when a is not a value object and cant multiply b
        """
        return self * other

    def __truediv__(self, other):
        """
        Divides self by other (a/b)
        """
        return self * (other ** -1)

    def __rtruediv__(self, other):
        """
        Fallback for when a is not a value object and cant divide b
        """
        return other * (self ** -1)

    def __repr__(self):
        """
        Output the values in the format Value(data='', grad = '')
        """
        return f"Value(data={self.data}, grad={self.grad})"