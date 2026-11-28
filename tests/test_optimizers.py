import numpy as np
from src.nn_scratch.layers import Dense
from src.nn_scratch.optimizers import SGD

def test_sgd_step(rng):
    d = Dense(2, 2, rng)
    d.forward(np.ones((1, 2)))
    d.backward(np.ones((1, 2)))
    W_before = d.W.copy()
    SGD(learning_rate=0.1).step([d])
    np.testing.assert_allclose(d.W, W_before - 0.1 * d.dW)