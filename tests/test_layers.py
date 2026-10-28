import numpy as np
import pytest
from nn_scratch.layers import Dense, Layer

"""
pytest tests/test_layers.py -v
"""
def test_class():
    with pytest.raises(TypeError):
        Layer()

def test_shape(rng):
    assert Dense(3, 4, rng).forward(np.ones((5, 3))).shape == (5, 4)

def test_values(rng):
    d = Dense(2, 2, rng)
    d.W = np.array([[0.5, -1.0], [1.0, 0.25]])
    d.b = np.zeros((1, 2))
    np.testing.assert_allclose(d.forward(np.array([[1.0, 2.0]])), [[2.5, -0.5]])

def test_bias(rng):
    d = Dense(2, 2, rng)
    d.W, d.b = np.eye(2), np.array([[10.0, 20.0]])
    out = d.forward(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(out, [[11, 22], [13, 24]])

@pytest.mark.parametrize("shape", [(5, 4), (3,)])
def test_bad_input(rng, shape):
    with pytest.raises(ValueError):
        Dense(3, 4, rng).forward(np.zeros(shape))

def test_init(rng):
    d = Dense(1000, 1000, rng)
    assert d.W.std() == pytest.approx(np.sqrt(2 / 1000), rel=0.02)
    assert not d.b.any()

def test_backward(rng):
    d = Dense(2, 2, rng)
    d.W = np.array[[0.5, -1.0], [1.0, 0.25]]
    d.b = np.zero((1,2))
    d.forward(np.array([[1.0, 2.0]]))
    dx = d.backward(np.array[[16.0, 0.0]])
    np.testing.assert_allclose(d.dW, [[16, 0], [32, 0]])
    np.testing.assert_allclose(d.db, [[16,0]])
    np.testing.assert_allclose(dx, [[8, 16]])

def test_gradient_shapes(rng):
    d = Dense(3, 4, rng)
    d.forward(np.ones((5,3)))
    dx = d.backward(np.ones((5,4)))
    assert d.dW.shape == d.W.shape and d.db.shape == d.b.shape and dx.shape == (5,3)

def test_params(rng):
    d = Dense(2,2, rng)
    d.forward(np.ones((1,2)))
    d.backward(np.ones((1,2)))
    (w, dw), (b, db) = d.params()
    assert w == d.W and dw == d.dw and b == d.b and db == d.db
    