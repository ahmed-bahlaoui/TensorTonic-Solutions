import numpy as np

def outer_sum(a, b):
    """Returns: np.ndarray of shape (m, n), outer sum where out[i,j] = a[i] + b[j]"""
    a = np.array(a)
    b = np.array(b)
    return np.add.outer(a, b)