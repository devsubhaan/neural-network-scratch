import numpy as np

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    batch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(np.clip(y_pred, 0, 1))) / batch_size

def cross_entropy_derirative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    y_pred is softmaxed probabilities
    y_true is just an one hot encoded vector array
    """
    y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
    confidences = np.sum(y_pred_clipped * y_true, axis=1)
    negative_log_likelihoods_mean = -np.mean(np.log(confidences))

    return negative_log_likelihoods_mean

def categorical_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_pred - y_true) / y_true.shape[0]