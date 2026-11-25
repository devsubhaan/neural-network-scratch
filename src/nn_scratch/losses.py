import numpy as np

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    batch_size = y_true.shape[1]
    return -np.sum(y_true * np.log(np.clip(y_pred, 0, 1))) / batch_size

def cross_entropy_derirative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    batch_size = y_true.shape[1]
    return (y_pred - y_true) / batch_size