import numpy as np

from src.nn_scratch.layers import Dense, ReLU
from src.nn_scratch.activations import softmax
from src.nn_scratch.losses import categorical_cross_entropy, categorical_cross_entropy_derivative

def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """labels: (batch,) ints -> (num_classes, batch)"""
    out = np.zeros((len(labels), num_classes))
    out[np.arange(len(labels)), labels] = 1
    return out

def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Max relative error between two arrays of the same shape."""
    return np.max(np.abs(a-b) / np.maximum((np.abs(a) + np.abs(b)), 1e-12))

def forward_loss(layers: list, x: np.ndarray, y_true: np.ndarray) -> float:
    """Forward through every layer, softmax the output and return the loss."""
    out = x
    for layer in layers:
        out = layer.forward(out)
    probs = softmax(out)
    return categorical_cross_entropy(y_true, probs)

def analytic_grads(layers: list, x: np.ndarray, y_true: np.ndarray):
    """
    Run forward/backward once
    Returns the gradient
    """
    out = x
    for layer in layers:
        out = layer.forward(out)

    probs = softmax(out)
    grad = categorical_cross_entropy_derivative(y_true, probs)

    for layer in reversed(layers):
        grad = layer.backward(grad)

    return grad

def numerical_grad(f, param: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    f: lambda function that returns the loss
    param: the array (a weight matrix, a bias, or x)
    """
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + h
        loss_plus = f()

        param[idx] = original - h
        loss_minus = f()

        grad[idx] = (loss_plus - loss_minus) / (2*h)

        param[idx] = original

    return grad

def check_network(layers: list, x: np.ndarray, y_true: np.ndarray) -> None:
    dx_analytic = analytic_grads(layers, x, y_true)

    saved = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "W"): 
            saved.append((i, "W", layer.W, layer.dW.copy()))
            saved.append((i, "b", layer.b, layer.db.copy()))
    saved.append((-1, "x", x, dx_analytic.copy()))

    f = lambda: forward_loss(layers, x, y_true)
    for i, name, param, analytic in saved:
        numeric = numerical_grad(f, param)
        err = relative_error(analytic, numeric)
        print(f"Layer index: {i}, Name: {name}, Relative Error: {err}, Status: {'Pass' if err < 1e-6 else 'Fail'}")

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_in, n_hidden, n_classes, batch = 5, 4, 3, 6

    x = rng.standard_normal((batch, n_in))
    labels = np.random.randint(0, n_classes, size=batch)
    y_true = one_hot(labels, n_classes)

    layers = [Dense(n_in, n_classes, rng)]
    check_network(layers, x, y_true)

    layers = [Dense(n_in, n_hidden, rng), ReLU(), Dense(n_hidden, n_classes, rng)]
    check_network(layers, x, y_true)