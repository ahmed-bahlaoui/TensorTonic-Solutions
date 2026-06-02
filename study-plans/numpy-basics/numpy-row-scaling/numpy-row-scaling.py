import numpy as np

def scale_rows(data, weights):
    """ Returns: np.ndarray of shape (m, n), each row scaled by the corresponding weight"""
    weights = np.array(weights)
    data = np.array(data)
    return data * weights[:, np.newaxis]
    