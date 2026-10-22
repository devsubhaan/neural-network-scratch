import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    e_z = np.exp(-z)
    return e_z * (1 + e_z) ** -2

def softmax(z: np.ndarray) -> np.ndarray:
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / e_z.sum(axis=0)

def relu(z: float) -> float:
    return np.maximum(0, z)

def relu_derivative(z: float) -> float:
    return (z > 0).astype(float)

