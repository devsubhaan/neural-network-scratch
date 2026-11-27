import numpy as np

def sigmoid(z: np.ndarray[float]) -> np.ndarray[float]:
    return 1.0  / (1.0 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray[float]) -> np.ndarray[float]:
    e_z = np.exp(-z)
    return e_z * (1.0 + e_z) ** - 2.0

def softmax(z: np.ndarray[float]) -> np.ndarray[float]:
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

def relu(z: float) -> float:
    return np.maximum(0, z)

def relu_derivative(z: float) -> float:
    if z > 0.0:
        return 1.0
    elif z < 0.:
        return 0.0
    else:
        return 0.0

