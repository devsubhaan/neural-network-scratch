import numpy as np
import pytest

from src.nn_scratch.layers import Dense, ReLU
from src.nn_scratch.gradient_check import analytic_grads, forward_loss, numerical_grad, relative_error, one_hot

def assert_gradients_match(layers, x, y_true):
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
        assert err < 1e-6, f"layer {i} {name}: relative error {err:.2e}"

@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n_in, n_classes, batch = 5, 3, 6
    x = rng.standard_normal((batch, n_in))
    labels = rng.integers(0, n_classes, size=batch)
    return rng, x, one_hot(labels, n_classes)

def test_single_dense_layer(data):
    rng, x, y_true = data
    layers = [Dense(5, 3, rng)]
    assert_gradients_match(layers, x, y_true)

def test_dense_relu_dense(data):
    rng, x, y_true = data
    layers = [Dense(5, 4, rng), ReLU(), Dense(4, 3, rng)]
    assert_gradients_match(layers, x, y_true)