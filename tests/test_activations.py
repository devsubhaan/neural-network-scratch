import numpy as np
import sys
import pytest
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.nn_scratch.activations import *

# python -m pytest tests/test_activations.py -v

@pytest.mark.parametrize(
    "input_val, expected_sigmoid, expected_derivative",
    [
        (-2.0, 0.1192, 0.1050),
        (-1.0, 0.2689, 0.1966),
        ( 0.0, 0.5000, 0.2500),
        ( 1.0, 0.7311, 0.1966),
        ( 2.0, 0.8808, 0.1050),
    ],
)

def test_sigmoid_and_derivative(input_val, expected_sigmoid, expected_derivative):
    actual_sigmoid = round(sigmoid(input_val), 4)
    actual_derivative = round(sigmoid_derivative(input_val), 4)

    assert actual_sigmoid == expected_sigmoid, f"Sigmoid failed for inputs: {input_val}"
    assert actual_derivative == expected_derivative, f"Derivative failed for inputs: {input_val}"

@pytest.mark.parametrize (
    "input_val, expected_relu, expected_derivative",
    [
        (-2.0, 0.0, 0.0),
        ( 0.0, 0.0, 0.0),
        ( 2.0, 2.0, 1.0),
    ],
)

def test_relu_and_derivative(input_val, expected_relu, expected_derivative):
    actual_relu = round(relu(input_val), 4)
    actual_derivative = round(relu_derivative(input_val), 4)

    assert actual_relu == expected_relu
    assert actual_derivative == expected_derivative


@pytest.mark.parametrize(
    "input_val, expected_output",
    [
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.09003057, 0.24472847, 0.66524096]),
        ),
        (
            np.array([-1.0, 0.0, 1.0]),
            np.array([0.09003057, 0.24472847, 0.66524096]),
        ),
    ],
)
def test_softmax(input_val, expected_output):
    np.testing.assert_allclose(softmax(input_val), expected_output, rtol=1e-6)


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