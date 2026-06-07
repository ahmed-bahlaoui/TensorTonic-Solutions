import numpy as np

def pairwise_diff(a):
    """Returns: np.ndarray of shape (n, n) where out[i,j] = a[i] - a[j]"""
    a = np.asarray(a, dtype=np.float64)
    l = []
    for i in range(len(a)):
        l.append(a[i] - a)
    return np.asarray(l, dtype=np.float64)
