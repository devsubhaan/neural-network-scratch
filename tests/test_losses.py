import numpy as np
import pytest

from nn_scratch.losses import cross_entropy, cross_entropy_derirative

def test_cross_entropy():
    y_pred = np.array(
        [
            [0.7, 0.1],
            [0.2, 0.3],
            [0.1, 0.6],
        ]
    )
    y_true = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    expected_loss = -(np.log(0.7) + np.log(0.3)) / 2

    assert np.isclose(cross_entropy(y_pred, y_true), expected_loss)


def test_cross_entropy_derivative():
    y_pred = np.array(
        [
            [0.7, 0.1],
            [0.2, 0.3],
            [0.1, 0.6],
        ]
    )
    y_true = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    expected_derivative = np.array(
        [
            [-0.15, 0.05],
            [0.10, -0.35],
            [0.05, 0.30],
        ]
    )

    np.testing.assert_allclose(
        cross_entropy_derirative(y_pred, y_true),
        expected_derivative,
    )