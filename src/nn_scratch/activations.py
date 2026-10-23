import numpy as np

def sigmoid(z: np.ndarray[float]) -> np.ndarray[float]:
    return 1.0  / (1.0 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray[float]) -> np.ndarray[float]:
    e_z = np.exp(-z)
    return e_z * (1.0 + e_z) ** - 2.0

def softmax(z: np.ndarray[float]) -> np.ndarray[float]:
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / e_z.sum(axis=0)

def relu(z: float) -> float:
    return np.maximum(0, z)

def relu_derivative(z: float) -> float:
    return (z > 0).astype(float)

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    batch_size = y_true.shape[1]
    return -np.sum(y_true * np.log(np.clip(y_pred, 0, 1))) / batch_size

def cross_entropy_derirative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    batch_size = y_true.shape[1]
    return (y_pred - y_true) / batch_size