


class Value:

    def __init__(self, data, _children=(), _op=''):
        # _children() holds the values that were operated on, like (5, 4) in 5 + 4
        pass

    def __add__(self, other):
        # check if self/other is a value object
        # make a _backward() that uses chain rule with += and make the outputs backward equal to its call
        pass

    def __mul__(self, other):
        pass

    def __pow__(self, other):
        pass

    def relu(self):
        pass

    def backward(self):
        pass

    def __neg__(self):
        pass

    def __radd__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __repr__(self):
        pass