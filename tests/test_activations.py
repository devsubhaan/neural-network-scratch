import numpy as np
import sys
import pytest
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.nn_scratch.activations import *

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.nn_scratch.activations import sigmoid, sigmoid_derivative


# pytest tests/test_activations.py -v

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
