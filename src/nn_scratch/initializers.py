import numpy as np


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))

def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=(fan_in, fan_out))
