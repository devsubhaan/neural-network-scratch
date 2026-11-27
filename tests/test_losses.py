import numpy as np
from src.nn_scratch.losses import categorical_cross_entropy, categorical_cross_entropy_derivative

Y_PRED = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
Y_TRUE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

def test_cross_entropy():
    expected = -(np.log(0.7) + np.log(0.3)) / 2
    assert np.isclose(categorical_cross_entropy(Y_TRUE, Y_PRED), expected)

def test_cross_entropy_derivative():
    expected = np.array([[-0.15, 0.10, 0.05], [0.05, -0.35, 0.30]])

    np.testing.assert_allclose(categorical_cross_entropy_derivative(Y_TRUE, Y_PRED), expected)