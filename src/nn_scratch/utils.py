import numpy as np

def one_hot(y, k):
    return np.eye(k)[y]