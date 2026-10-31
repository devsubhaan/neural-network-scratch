import numpy as np

def accuracy(logits):
    return float((logits.argmax(axis=1) == y).mean())
    