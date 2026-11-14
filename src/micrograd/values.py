


class Value:
    """
    Stores a single value along with its gradient
    """

    def __init__(self, data, _children=(), _op=''):
        """
        Initialises a new Value node
        Tracks the data, gradient, children nodes 
        and the operation used to create it
        """
        pass

    def __add__(self, other):
        """
        Returns self added to other (self + other)

        Makes sure that 'other' is a Value object
        Simply adds the output to the self and other gradients
        """
        pass

    def __mul__(self, other):
        """
        Returns self multiplied by other (self * other)

        Makes sure that 'other' is a Value object
        Uses chain rule for both self and other gradients
        """
        pass

    def __pow__(self, other):
        """
        Returns self raised to the other (self ** other)

        Checks if the power is an integer or float
        Applies power rule for the backwards pass
        """
        pass

    def relu(self):
        """
        Values less than zero become zero
        Values bigger than zero stay
        """
        pass

    def backward(self):
        """
        Calculates the gradients via backpropagation by first sorting the children in reverse order
        Then applys chain rule to get their gradients
        """
        pass

    def __neg__(self):
        """
        Negates a value turning it negative
        """
        pass

    def __radd__(self, other):
        """
        Fallback for when self is not a value object and cant add other (b + a is done)
        """
        pass

    def __sub__(self, other):
        """
        Subtracts self and other (a - b)
        """
        pass

    def __rsub__(self, other):
        """
        Fallback for when a is not a value object and cant subtract b
        """
        pass

    def __rmul__(self, other):
        """
        Fallback for when a is not a value object and cant multiply b
        """
        pass

    def __truediv__(self, other):
        """
        Divides self by other (a/b)
        """
        pass

    def __rtruediv__(self, other):
        """
        Fallback for when a is not a value object and cant divide b
        """
        pass

    def __repr__(self):
        """
        Output the values in the format Value(data='', grad = '')
        """
        pass