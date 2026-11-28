import numpy as np

from src.nn_scratch.layers import Dense, ReLU
from src.nn_scratch.network import Network
from src.nn_scratch.optimizers import SGD
from src.nn_scratch.activations import softmax
from src.nn_scratch.losses import categorical_cross_entropy_derivative
from src.nn_scratch.gradient_check import one_hot

def test_network_learns(rng):
    # two clusters far apart: one around +5, one around -5
    x = np.vstack([rng.standard_normal((20, 2)) + 5, rng.standard_normal((20, 2)) - 5])
    labels = np.array([0] * 20 + [1] * 20)
    y_true = one_hot(labels, 2)

    network = Network([Dense(2, 8, rng), ReLU(), Dense(8, 2, rng)])
    optimizer = SGD(learning_rate=0.1)

    for _ in range(100):
        probs = softmax(network.forward(x))
        network.backward(categorical_cross_entropy_derivative(y_true, probs))
        optimizer.step(network.layers)

    assert np.mean(np.argmax(network.forward(x), axis=1) == labels) > 0.95